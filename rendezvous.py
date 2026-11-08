r"""
rendezvous.py — WebSocket rendezvous server for Kaggle notebook model proxies.

Replaces an ngrok-style public tunnel: the notebook dials OUT to this server
over a single WebSocket, and the server relays inbound HTTP requests to it.

Topology:
    bench client --HTTPS--> funnel --HTTP--> rendezvous --WS--> notebook
                                                      \--> kbench client --> Kaggle Model Proxy

Notebook connection
-------------------
GET /connect (WebSocket upgrade)
  Headers:
    Authorization: Bearer <RENDEZVOUS_KEY>   (required; invalid -> HTTP 403)
    X-Notebook-Name: <label>                 (optional)

Wire protocol (JSON text frames; binary bodies are base64):

  server -> notebook
    {"type": "welcome", "notebook_id": "...", "public_base_url": ".../v1"}
    {"type": "http_request", "id": "<uuid>", "method": "POST",
     "path": "/v1/chat/completions", "headers": {...}, "body_b64": "..."}
    {"type": "http_cancel", "id": "<uuid>"}

  notebook -> server
    {"type": "http_response_start", "id": "...", "status": 200, "headers": {...}}
    {"type": "http_response_chunk", "id": "...", "data_b64": "..."}
    {"type": "http_response_end",   "id": "..."}
    {"type": "http_error", "id": "...", "status": 502, "message": "..."}
    {"type": "drain"}   notebook is draining (session limit) — stop routing

  server -> notebook (in addition to the above)
    {"type": "drain"}   admin-initiated: ask the notebook to drain

HTTP API (client-facing; Bearer key from CLIENT_API_KEYS if configured):
  GET  /healthz                      liveness + connected notebook count (no auth)
  GET  /v1/models                    merged model list across notebooks
  POST /v1/chat/completions          forwarded; SSE streams pass through live
  ANY  /v1/{path}                    generic forward for anything else

Admin API (Bearer key from ADMIN_API_KEYS; disabled if unset):
  GET  /admin/status                 notebooks, pending requests, model policy
  GET  /admin/models                 refresh + merged models with enabled flags
  POST /admin/models/disable         {"model": "<id-or-alias>"}
  POST /admin/models/enable          {"model": "<id-or-alias>"}
  POST /admin/policy                 {"mode": "whitelist"|"blacklist",
                                      "models": [...optional replace...]}
  POST /admin/notebooks/{id}/drain   ask a notebook to drain (stop new work)
  POST /admin/notebooks/{id}/disconnect
  GET  /admin/keys                   list keys (hints only) by kind
  POST /admin/keys                   {"kind": "rendezvous"|"client"|"admin",
                                      "label": "..."} -> returns FULL key once
  DELETE /admin/keys/{key_id}        revoke (last admin key refused)

  GET  /ui                           minimal management web UI

Keys live in memory; the *_KEYS env vars only seed the store at startup.
Kinds: "rendezvous" (notebook auth), "client" (/v1 Bearer), "admin"
(/admin + /ui Bearer). Runtime-created keys vanish on restart — export
them in the environment for persistence.

Model policy is mutually exclusive — one mode at a time:
  whitelist   only listed models run (seed: MODEL_ALLOWLIST env)
  blacklist   everything except listed models runs (seed: MODEL_BLOCKLIST env)
In whitelist mode, /admin/models/enable adds to the allowed set and
disable removes from it; in blacklist mode they mutate the blocked set.
Setting both env vars -> whitelist wins.

Config (env):
  RENDEZVOUS_KEYS     comma-separated keys notebooks may connect with.
                      If unset, one is generated at startup and printed.
  CLIENT_API_KEYS     comma-separated Bearer keys for /v1/*. Empty = open.
  ADMIN_API_KEYS      comma-separated Bearer keys for /admin/*. Empty = disabled.
  MODEL_ALLOWLIST     comma-separated ids/aliases -> whitelist mode.
  MODEL_BLOCKLIST     comma-separated ids/aliases -> blacklist mode.
                      Both unset = allow everything (blacklist mode, empty).
  REQUEST_TIMEOUT     seconds to wait for response start / idle stream (def 600).
  PUBLIC_BASE_URL     advertised base URL sent to notebooks in `welcome`.
  HOST / PORT         bind address (defaults 127.0.0.1:8377).
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import os
import secrets
import time
import uuid

import httpx
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import (
    Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)
log = logging.getLogger("rendezvous")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "600"))
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8377"))
WS_MAX_SIZE = 64 * 1024 * 1024  # forwarded bodies (e.g. base64 images) can be large

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
    # stripped because bodies travel as decoded, re-chunked bytes:
    "content-length", "content-encoding",
}


# ---------------------------------------------------------------------------
# Key store — env-seeded, mutable via /admin/keys
# ---------------------------------------------------------------------------

KEY_KINDS = ("rendezvous", "client", "admin")


class KeyStore:
    """API keys by kind. Only hints are ever listed; full values are
    returned once at creation."""

    def __init__(self):
        self.keys: dict[str, dict] = {}

    def seed(self, kind: str, env_var: str) -> None:
        for v in os.environ.get(env_var, "").split(","):
            if v.strip():
                self.add(kind, v.strip(), label=f"env:{env_var}")

    def add(self, kind: str, value: str | None = None,
            label: str = "") -> dict:
        value = value or secrets.token_urlsafe(24)
        key = {
            "id": f"k_{uuid.uuid4().hex[:8]}",
            "kind": kind,
            "label": label,
            "value": value,
            "hint": f"{value[:4]}…{value[-4:]}",
            "created_at": time.time(),
        }
        self.keys[key["id"]] = key
        return key

    def revoke(self, key_id: str) -> dict | None:
        return self.keys.pop(key_id, None)

    def count(self, kind: str) -> int:
        return sum(1 for k in self.keys.values() if k["kind"] == kind)

    def valid(self, kind: str, candidate: str) -> bool:
        return any(
            k["kind"] == kind
            and hmac.compare_digest(k["value"].encode(), candidate.encode())
            for k in self.keys.values()
        )

    def public(self, kind: str | None = None) -> list[dict]:
        return [
            {k2: v for k2, v in k.items() if k2 != "value"}
            for k in sorted(self.keys.values(),
                            key=lambda k: k["created_at"])
            if kind is None or k["kind"] == kind
        ]


KEYS = KeyStore()
KEYS.seed("rendezvous", "RENDEZVOUS_KEYS")
KEYS.seed("client", "CLIENT_API_KEYS")
KEYS.seed("admin", "ADMIN_API_KEYS")

# --- pawsapi identity hook -------------------------------------------------
# Local keys still work (env-seeded). A key that isn't local is checked
# against pawsapi, so a notebook can present the user's own API key and the
# connection is attributed to that account.
PAWSAPI_VERIFY_URL = os.environ.get(
    "PAWSAPI_INTERNAL_URL", "http://127.0.0.1:8765/v1/internal/kaggle/verify"
)
PAWSAPI_INTERNAL_TOKEN = os.environ.get("PAWSAPI_INTERNAL_TOKEN", "").strip()
_REMOTE_CACHE: dict[str, tuple[float, dict]] = {}
_REMOTE_TTL = 60.0


async def verify_with_pawsapi(key: str) -> dict | None:
    """Resolve a key to {user_id, username, key_id} via pawsapi."""
    if not PAWSAPI_INTERNAL_TOKEN:
        return None
    now = time.time()
    hit = _REMOTE_CACHE.get(key)
    if hit and hit[0] > now:
        return hit[1]
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                PAWSAPI_VERIFY_URL,
                json={"key": key},
                headers={"X-Internal-Token": PAWSAPI_INTERNAL_TOKEN},
            )
    except Exception as exc:
        log.warning("pawsapi verify unreachable: %r", exc)
        return None
    if resp.status_code != 200:
        return None
    data = resp.json()
    _REMOTE_CACHE[key] = (now + _REMOTE_TTL, data)
    return data



# ---------------------------------------------------------------------------
# Notebook connection + pool
# ---------------------------------------------------------------------------


class RequestState:
    """One in-flight relayed request; notebook pushes events into .queue."""

    def __init__(self, request_id: str):
        self.id = request_id
        self.queue: asyncio.Queue[tuple] = asyncio.Queue()
        self.started_at = time.time()


class NotebookConn:
    """A connected notebook's WebSocket + its pending relayed requests."""

    def __init__(self, ws: WebSocket, name: str):
        self.id = f"nb-{uuid.uuid4().hex[:8]}"
        self.ws = ws
        self.name = name
        self.connected_at = time.time()
        self.send_lock = asyncio.Lock()
        self.pending: dict[str, RequestState] = {}
        # model id -> upstream id, learned from the last /v1/models fetch
        self.models: dict[str, str | None] = {}
        self.relayed = 0
        self.draining = False

    async def send_json(self, msg: dict) -> None:
        async with self.send_lock:
            await self.ws.send_text(json.dumps(msg))

    async def request(
        self, method: str, path: str, headers: dict, body: bytes
    ) -> RequestState:
        state = RequestState(uuid.uuid4().hex)
        self.pending[state.id] = state
        self.relayed += 1
        await self.send_json({
            "type": "http_request",
            "id": state.id,
            "method": method,
            "path": path,
            "headers": headers,
            "body_b64": base64.b64encode(body).decode(),
        })
        return state

    async def cancel(self, request_id: str) -> None:
        state = self.pending.pop(request_id, None)
        try:
            await self.send_json({"type": "http_cancel", "id": request_id})
        except Exception:
            pass
        if state:
            await state.queue.put(("error", 499, "cancelled"))

    async def dispatch(self, msg: dict) -> None:
        mtype = msg.get("type")
        rid = msg.get("id")
        state = self.pending.get(rid) if rid else None
        log.debug("%s: dispatch %s id=%s pending=%s",
                  self.id, mtype, rid, rid in self.pending)
        if mtype == "http_response_start" and state:
            await state.queue.put(("start", msg.get("status", 200), msg.get("headers") or {}))
        elif mtype == "http_response_chunk" and state:
            try:
                data = base64.b64decode(msg.get("data_b64") or "")
            except Exception:
                data = b""
            await state.queue.put(("chunk", data))
        elif mtype == "http_response_end" and state:
            self.pending.pop(rid, None)
            await state.queue.put(("end",))
        elif mtype == "http_error":
            state = self.pending.pop(rid, None)
            if state:
                await state.queue.put((
                    "error", msg.get("status", 502), msg.get("message", "notebook error")
                ))
        elif mtype == "drain":
            self.draining = True
            log.info("%s: notebook is draining", self.id)
        elif mtype == "bye":
            raise WebSocketDisconnect()

    async def recv_loop(self) -> None:
        while True:
            raw = await self.ws.receive_text()
            try:
                await self.dispatch(json.loads(raw))
            except WebSocketDisconnect:
                raise
            except Exception:
                log.exception("%s: bad frame from notebook", self.id)

    def fail_all(self, message: str = "notebook disconnected") -> None:
        for state in self.pending.values():
            state.queue.put_nowait(("error", 502, message))
        self.pending.clear()


class Pool:
    def __init__(self):
        self.conns: dict[str, NotebookConn] = {}
        self._rr = 0

    def add(self, conn: NotebookConn) -> None:
        self.conns[conn.id] = conn

    def remove(self, conn: NotebookConn) -> None:
        self.conns.pop(conn.id, None)
        conn.fail_all()

    def pick(self, model: str | None = None) -> NotebookConn | None:
        conns = [c for c in self.conns.values() if not c.draining]
        if not conns:
            return None
        if model:
            serving = [
                c for c in conns
                if model in c.models or model in c.models.values()
            ]
            if serving:
                conns = serving
        self._rr += 1
        return conns[self._rr % len(conns)]


POOL = Pool()
# Global alias->upstream index, refreshed whenever a notebook reports models.
MODEL_INDEX: dict[str, str | None] = {}

# Model gate — mutually exclusive policy, mutable via /admin/policy:
#   whitelist: only models in POLICY_SET run
#   blacklist: everything except models in POLICY_SET runs
_ENV_ALLOW = {m.strip() for m in os.environ.get("MODEL_ALLOWLIST", "").split(",") if m.strip()}
_ENV_BLOCK = {m.strip() for m in os.environ.get("MODEL_BLOCKLIST", "").split(",") if m.strip()}
POLICY_MODE = "whitelist" if _ENV_ALLOW else "blacklist"
POLICY_SET: set[str] = set(_ENV_ALLOW if _ENV_ALLOW else _ENV_BLOCK)

# ---------------------------------------------------------------------------
# Relay helpers
# ---------------------------------------------------------------------------


def _filtered_headers(headers: dict) -> dict[str, str]:
    return {
        k: v for k, v in headers.items()
        if k.lower() not in HOP_BY_HOP
    }


def _fwd_headers(request: Request) -> dict[str, str]:
    return _filtered_headers(dict(request.headers))


async def _next_event(state: RequestState) -> tuple:
    return await asyncio.wait_for(state.queue.get(), REQUEST_TIMEOUT)


async def _collect(state: RequestState) -> tuple[int, dict, bytes]:
    """Wait for response_start + all chunks; returns (status, headers, body)."""
    first = await _next_event(state)
    if first[0] == "error":
        return first[1], {}, json.dumps(
            {"error": {"message": first[2], "type": "rendezvous_error"}}
        ).encode()
    if first[0] == "end":
        return 502, {}, b'{"error": {"message": "empty upstream response"}}'
    _, status, headers = first
    body = bytearray()
    while True:
        ev = await _next_event(state)
        if ev[0] == "chunk":
            body += ev[1]
        elif ev[0] == "end":
            break
        elif ev[0] == "error":
            return ev[1], {}, json.dumps(
                {"error": {"message": ev[2], "type": "rendezvous_error"}}
            ).encode()
    return status, _filtered_headers(headers), bytes(body)


async def relay(request: Request, conn: NotebookConn, path: str) -> Response:
    """Forward one HTTP request to the notebook and relay its response back."""
    body = await request.body()
    state = await conn.request(request.method, path, _fwd_headers(request), body)
    try:
        first = await _next_event(state)
        if first[0] == "error":
            return JSONResponse(
                status_code=first[1],
                content={"error": {"message": first[2], "type": "rendezvous_error"}},
            )
        if first[0] == "end":
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "empty upstream response"}},
            )

        _, status, headers = first
        headers = _filtered_headers(headers)

        if "text/event-stream" in headers.get("content-type", ""):
            async def chunks():
                finished = False
                try:
                    while True:
                        ev = await _next_event(state)
                        log.debug("%s: stream ev %s", conn.id, ev[0])
                        if ev[0] == "chunk":
                            yield ev[1]
                        else:  # end or mid-stream error -> just close the stream
                            finished = True
                            break
                finally:
                    conn.pending.pop(state.id, None)
                    if not finished:  # client bailed mid-stream -> abort upstream
                        try:
                            await conn.cancel(state.id)
                        except Exception:
                            pass

            return StreamingResponse(
                chunks(), status_code=status, headers=headers,
                media_type="text/event-stream",
            )

        body_bytes = bytearray()
        while True:
            ev = await _next_event(state)
            if ev[0] == "chunk":
                body_bytes += ev[1]
            elif ev[0] == "end":
                break
            elif ev[0] == "error":
                return JSONResponse(
                    status_code=ev[1],
                    content={"error": {"message": ev[2], "type": "rendezvous_error"}},
                )
        return Response(bytes(body_bytes), status_code=status, headers=headers)
    except asyncio.TimeoutError:
        await conn.cancel(state.id)
        return JSONResponse(
            status_code=504,
            content={"error": {"message": "notebook response timed out",
                               "type": "rendezvous_error"}},
        )
    except BaseException:
        try:
            await conn.cancel(state.id)
        except Exception:
            pass
        raise


async def fetch_models(conn: NotebookConn) -> dict[str, str | None]:
    """Pull /v1/models from a notebook; update its + the global model index."""
    try:
        state = await conn.request("GET", "/v1/models", {}, b"")
        status, _headers, body = await _collect(state)
    except Exception as exc:
        log.warning("%s: model list fetch failed: %r", conn.id, exc)
        return conn.models
    if status != 200:
        return conn.models
    try:
        data = json.loads(body).get("data") or []
    except Exception:
        return conn.models
    conn.models = {
        m.get("id"): m.get("upstream_model") for m in data if m.get("id")
    }
    MODEL_INDEX.update(conn.models)
    return conn.models


async def warm_models(conn: NotebookConn) -> None:
    """Fill the model list right after connect, with a few retries.

    The notebook may still be starting its proxy when the socket opens, so a
    single attempt can come back empty. Three passes with backoff is enough.
    """
    for delay in (0, 5, 15):
        if delay:
            await asyncio.sleep(delay)
        models = await fetch_models(conn)
        if models:
            log.info("%s: %d model(s) reported", conn.id, len(models))
            return
    log.info("%s: no models reported yet", conn.id)


def _model_equivalents(model: str) -> set[str]:
    """All ids that could refer to `model`: itself, its upstream, its aliases."""
    ids = {model}
    upstream = MODEL_INDEX.get(model)
    if upstream:
        ids.add(upstream)
    ids.update(alias for alias, up in MODEL_INDEX.items() if up == model)
    return ids


def _denied_reason(model: str | None) -> tuple[str, str] | None:
    """(message, error_type) if the model fails the active policy, else None."""
    if not model:
        return None
    eq = _model_equivalents(model)
    if POLICY_MODE == "whitelist":
        if not (eq & POLICY_SET):
            return (f"model {model!r} is not in the allowlist",
                    "model_not_allowed")
    elif eq & POLICY_SET:
        return f"model {model!r} is disabled", "model_disabled"
    return None


def _model_enabled(model_id: str) -> bool:
    hit = _model_equivalents(model_id) & POLICY_SET
    return bool(hit) if POLICY_MODE == "whitelist" else not hit


def _no_notebooks() -> JSONResponse:
    msg = ("all notebooks draining" if POOL.conns
           else "no notebooks connected")
    return JSONResponse(
        status_code=503,
        content={"error": {"message": msg, "type": "rendezvous_error"}},
    )


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


async def require_client(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    if KEYS.count("client") and not (
        creds and KEYS.valid("client", creds.credentials)
    ):
        raise HTTPException(401, "invalid client API key")


async def require_admin(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    if not KEYS.count("admin") or not (
        creds and KEYS.valid("admin", creds.credentials)
    ):
        raise HTTPException(403, "admin access denied")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Kaggle Rendezvous Server", version="1.0.0")
_UI_PATH = Path(__file__).parent / "ui.html"


@app.websocket("/connect")
async def notebook_connect(ws: WebSocket) -> None:
    auth = ws.headers.get("authorization", "")
    key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not key:
        await ws.close(code=4403, reason="invalid rendezvous key")
        return
    identity = None
    if not KEYS.valid("rendezvous", key):
        identity = await verify_with_pawsapi(key)
        if identity is None:
            await ws.close(code=4403, reason="invalid rendezvous key")
            return
    name = ws.headers.get("x-notebook-name", "notebook")
    await ws.accept()
    conn = NotebookConn(ws, name)
    conn.user_id = (identity or {}).get("user_id")
    conn.username = (identity or {}).get("username")
    log.info("%s auth: key=%s… source=%s user=%s", conn.id, key[:5],
             "pawsapi" if identity else "env", conn.username or "-")
    asyncio.create_task(warm_models(conn))
    POOL.add(conn)
    log.info("%s (%s) connected; %d notebook(s) online", conn.id, name, len(POOL.conns))
    await conn.send_json({
        "type": "welcome",
        "notebook_id": conn.id,
        "public_base_url": f"{PUBLIC_BASE_URL}/v1" if PUBLIC_BASE_URL else None,
        "user": conn.username,
        "ts": time.time(),
    })
    try:
        await conn.recv_loop()
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("%s: connection error", conn.id)
    finally:
        POOL.remove(conn)
        log.info("%s (%s) disconnected; %d notebook(s) online",
                 conn.id, name, len(POOL.conns))


@app.get("/healthz")
async def healthz():
    return {"ok": True, "notebooks": len(POOL.conns), "ts": time.time()}


@app.get("/")
async def root():
    return {
        "service": "kaggle-rendezvous",
        "notebooks": len(POOL.conns),
        "endpoints": ["/v1/models", "/v1/chat/completions", "/admin/status"],
    }


@app.get("/v1/models", dependencies=[Depends(require_client)])
async def list_models():
    merged: dict[str, dict] = {}
    for conn in list(POOL.conns.values()):
        models = await fetch_models(conn)
        for mid, upstream in models.items():
            entry = merged.setdefault(mid, {
                "id": mid, "object": "model", "created": 0,
                "owned_by": "kaggle", "upstream_model": upstream,
            })
            entry.setdefault("notebooks", []).append(conn.id)
    if POLICY_MODE == "whitelist":
        merged = {mid: e for mid, e in merged.items()
                  if _model_equivalents(mid) & POLICY_SET}
    for entry in merged.values():
        entry["enabled"] = _model_enabled(entry["id"])
    if not merged and not POOL.conns:
        return _no_notebooks()
    return {"object": "list", "data": sorted(merged.values(), key=lambda e: e["id"])}


@app.post("/v1/chat/completions", dependencies=[Depends(require_client)])
async def chat_completions(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "request body must be JSON"}},
        )
    model = payload.get("model")
    denied = _denied_reason(model)
    if denied:
        message, etype = denied
        return JSONResponse(
            status_code=403,
            content={"error": {"message": message, "type": etype}},
        )
    conn = POOL.pick(model)
    if conn is None:
        return _no_notebooks()
    return await relay(request, conn, "/v1/chat/completions")


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    dependencies=[Depends(require_client)],
)
async def v1_forward(request: Request, path: str):
    conn = POOL.pick()
    if conn is None:
        return _no_notebooks()
    qs = f"?{request.url.query}" if request.url.query else ""
    return await relay(request, conn, f"/v1/{path}{qs}")


# ------------------------------ admin --------------------------------------


@app.get("/admin/status", dependencies=[Depends(require_admin)])
async def admin_status():
    return {
        "notebooks": [
            {
                "id": c.id, "name": c.name,
                "connected_at": c.connected_at,
                "uptime_s": round(time.time() - c.connected_at, 1),
                "pending_requests": len(c.pending),
                "relayed_requests": c.relayed,
                "draining": c.draining,
                "username": getattr(c, "username", None),
                "user_id": getattr(c, "user_id", None),
                "models": sorted(c.models),
            }
            for c in POOL.conns.values()
        ],
        "policy": {"mode": POLICY_MODE, "models": sorted(POLICY_SET)},
    }


@app.get("/admin/models", dependencies=[Depends(require_admin)])
async def admin_models():
    merged: dict[str, dict] = {}
    for conn in list(POOL.conns.values()):
        for mid, upstream in (await fetch_models(conn)).items():
            merged.setdefault(mid, {"upstream_model": upstream, "notebooks": []})
            merged[mid]["notebooks"].append(conn.id)
    return {
        "models": [
            {"id": mid, **info, "enabled": _model_enabled(mid)}
            for mid, info in sorted(merged.items())
        ],
        "policy": {"mode": POLICY_MODE, "models": sorted(POLICY_SET)},
    }


async def _set_model_enabled(request: Request, enabled: bool) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    model = (payload.get("model") or "").strip()
    if not model:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "body must include 'model'"}},
        )
    # make sure the index is fresh enough to resolve aliases
    for conn in list(POOL.conns.values()):
        if not conn.models:
            await fetch_models(conn)
    ids = _model_equivalents(model)
    # whitelist: enable adds to the set, disable removes — inverted vs
    # blacklist, where the set is the blocked list.
    add = enabled if POLICY_MODE == "whitelist" else not enabled
    if add:
        POLICY_SET.update(ids)
    else:
        POLICY_SET.difference_update(ids)
    return JSONResponse(status_code=200, content={
        "model": model,
        "matched_ids": sorted(ids),
        "enabled": enabled,
        "policy": {"mode": POLICY_MODE, "models": sorted(POLICY_SET)},
    })


@app.post("/admin/models/disable", dependencies=[Depends(require_admin)])
async def admin_disable(request: Request):
    return await _set_model_enabled(request, enabled=False)


@app.post("/admin/models/enable", dependencies=[Depends(require_admin)])
async def admin_enable(request: Request):
    return await _set_model_enabled(request, enabled=True)


@app.post("/admin/policy", dependencies=[Depends(require_admin)])
async def admin_policy(request: Request):
    global POLICY_MODE
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    mode = (payload.get("mode") or "").strip().lower()
    if mode not in ("whitelist", "blacklist"):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "mode must be 'whitelist' or "
                                          "'blacklist'"}},
        )
    POLICY_MODE = mode
    if "models" in payload:
        models = payload["models"]
        if not isinstance(models, list):
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "'models' must be a list"}},
            )
        POLICY_SET.clear()
        POLICY_SET.update(str(m).strip() for m in models if str(m).strip())
    log.info("policy set: mode=%s models=%s", POLICY_MODE, sorted(POLICY_SET))
    return {"policy": {"mode": POLICY_MODE, "models": sorted(POLICY_SET)}}


@app.post("/admin/notebooks/{notebook_id}/drain",
          dependencies=[Depends(require_admin)])
async def admin_drain(notebook_id: str):
    conn = POOL.conns.get(notebook_id)
    if conn is None:
        return JSONResponse(
            status_code=404, content={"error": {"message": "no such notebook"}}
        )
    conn.draining = True
    try:
        await conn.send_json({"type": "drain"})
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"could not notify notebook: {exc!r}"}},
        )
    return {"draining": notebook_id}


@app.get("/admin/keys", dependencies=[Depends(require_admin)])
async def admin_list_keys():
    return {kind: KEYS.public(kind) for kind in KEY_KINDS}


@app.post("/admin/keys", dependencies=[Depends(require_admin)])
async def admin_create_key(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    kind = (payload.get("kind") or "").strip()
    if kind not in KEY_KINDS:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"kind must be one of {KEY_KINDS}"}},
        )
    key = KEYS.add(kind, label=(payload.get("label") or "").strip())
    log.info("created %s key %s (%s)", kind, key["id"], key["hint"])
    return {**{k: v for k, v in key.items() if k != "value"},
            "value": key["value"]}  # full value returned once, here only


@app.delete("/admin/keys/{key_id}", dependencies=[Depends(require_admin)])
async def admin_revoke_key(key_id: str):
    key = KEYS.keys.get(key_id)
    if key is None:
        return JSONResponse(
            status_code=404, content={"error": {"message": "no such key"}}
        )
    if key["kind"] == "admin" and KEYS.count("admin") == 1:
        return JSONResponse(
            status_code=409,
            content={"error": {"message": "refusing to revoke the last "
                                          "admin key"}},
        )
    KEYS.revoke(key_id)
    log.info("revoked %s key %s (%s)", key["kind"], key_id, key["hint"])
    return {"revoked": {k: v for k, v in key.items() if k != "value"}}


@app.get("/ui")
async def management_ui():
    from fastapi.responses import FileResponse
    if not _UI_PATH.exists():
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "ui.html not found"}},
        )
    return FileResponse(_UI_PATH)


@app.post("/admin/notebooks/{notebook_id}/disconnect",
          dependencies=[Depends(require_admin)])
async def admin_disconnect(notebook_id: str):
    conn = POOL.conns.get(notebook_id)
    if conn is None:
        return JSONResponse(
            status_code=404, content={"error": {"message": "no such notebook"}}
        )
    await conn.ws.close(code=4000, reason="disconnected by admin")
    POOL.remove(conn)
    return {"disconnected": notebook_id}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    if not KEYS.count("rendezvous"):
        generated = KEYS.add("rendezvous", label="generated")
        log.warning("RENDEZVOUS_KEYS unset — generated one-time key: %s",
                    generated["value"])
    if not KEYS.count("client"):
        log.warning("CLIENT_API_KEYS unset — /v1/* endpoints are UNAUTHENTICATED")
    if not KEYS.count("admin"):
        log.warning("ADMIN_API_KEYS unset — /admin/* endpoints are disabled")

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        ws_max_size=WS_MAX_SIZE,
        ws_ping_interval=20,
        ws_ping_timeout=30,
    )


if __name__ == "__main__":
    main()
