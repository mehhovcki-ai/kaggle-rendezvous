"""
kaggle_bridge.py — outbound WebSocket bridge to the rendezvous server.

Run the proxy cell first (serve() must be running). Then:

    bridge = connect(port=8000)          # dials out, relays HTTP over WS
    bridge.public_base_url               # URL remote clients should use
    disconnect(bridge)                   # tear down

Configuration (Kaggle secrets -> env vars):
    RENDEZVOUS_KEY   required bearer key the server authorizes with
    RENDEZVOUS_URL   ws(s):// URL of the server's /connect endpoint
Both may also be passed explicitly as connect(url=..., key=...).

The bridge authenticates to the rendezvous server with the rendezvous key,
then forwards each relayed HTTP request to the local proxy on
http://127.0.0.1:{port} — injecting PROXY_API_KEY itself, so remote clients
never see it.

Failsafes:
  * Session drain — Kaggle sessions die at ~12h; the bridge drains at
    T-10min (env BRIDGE_DRAIN_AFTER_S, default 42600s): it tells the server
    to stop routing, returns 503 for new requests, lets in-flight work
    finish, then disconnects. connect(drain_after=S) to override/0 to off.
  * Reconnect backoff with jitter (thundering-herd safe).
  * Concurrency cap — at most max_concurrent in-flight relays (default 8);
    excess requests get a 429 after queue_timeout instead of piling up.
  * Loopback timeouts — dead local proxy fails fast (connect=5s, pool=5s,
    write=10s); streaming reads stay open (read=None).
The notebook session must stay alive while bridging.
"""

import asyncio
import base64
import json
import os
import random
import threading
import time

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
    "content-length", "content-encoding", "host", "authorization",
}


class Bridge:
    """Handle for a running bridge; returned by connect()."""

    def __init__(self, url, key, port, api_key, name, reconnect,
                 drain_after, max_concurrent, queue_timeout):
        self.url = url
        self.key = key
        self.port = port
        self.api_key = api_key
        self.name = name
        self.reconnect = reconnect
        self.max_concurrent = max_concurrent
        self.queue_timeout = queue_timeout
        self._drain_at = (
            time.monotonic() + drain_after if drain_after else None
        )
        self.draining = False
        self.connected = threading.Event()
        self.notebook_id = None
        self.public_base_url = None
        self.error = None
        self.relayed = 0
        self._should_run = True
        self._tasks = {}
        self._ws = None
        self._loop = None
        self._send_lock = None
        self._sema = None
        self._client = None
        self._drain_task = None
        self.thread = threading.Thread(
            target=self._thread_main, daemon=True, name="kbench-ws-bridge"
        )

    # ------------------------------------------------------------------
    # thread plumbing — same pattern as serve(): own loop in a daemon thread
    # ------------------------------------------------------------------

    def start(self):
        self.thread.start()
        return self

    def _thread_main(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run())
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            asyncio.set_event_loop(None)
            self._loop.close()

    # ------------------------------------------------------------------
    # main loop: connect -> session -> (reconnect with jitter)
    # ------------------------------------------------------------------

    @staticmethod
    def _retry_delay(backoff):
        return min(backoff * random.uniform(0.5, 1.5), 30.0)

    async def _run(self):
        import websockets

        backoff = 1.0
        while self._should_run:
            headers = {
                "Authorization": f"Bearer {self.key}",
                "X-Notebook-Name": self.name,
            }
            connect_kwargs = dict(
                max_size=64 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=30,
                open_timeout=15,
                close_timeout=5,
            )
            try:
                try:
                    ws = await websockets.connect(
                        self.url, additional_headers=headers, **connect_kwargs
                    )
                except TypeError:  # websockets <14: extra_headers
                    ws = await websockets.connect(
                        self.url, extra_headers=headers, **connect_kwargs
                    )
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                self.error = f"connect failed: {exc!r}"
                if status in (401, 403):
                    self.error = (
                        f"rendezvous rejected the key (HTTP {status}) — "
                        "check RENDEZVOUS_KEY"
                    )
                    print(f"[bridge] {self.error}")
                    self._should_run = False
                    self.connected.set()  # unblock ready() callers
                    return
                if not (self.reconnect and self._should_run):
                    self.connected.set()
                    return
                delay = self._retry_delay(backoff)
                print(f"[bridge] {self.error}; retrying in {delay:.0f}s")
                await asyncio.sleep(delay)
                backoff = min(backoff * 2, 30)
                continue

            try:
                import httpx
                self._ws = ws
                self._send_lock = asyncio.Lock()
                self._sema = asyncio.Semaphore(self.max_concurrent)
                self._client = httpx.AsyncClient(
                    # dead loopback fails fast; streaming reads stay open
                    timeout=httpx.Timeout(connect=5, read=None, write=10,
                                          pool=5),
                    limits=httpx.Limits(max_connections=self.max_concurrent),
                )
                self._drain_task = asyncio.ensure_future(self._drain_timer(ws))
                await self._session(ws)
                backoff = 1.0
            except Exception as exc:
                self.error = f"session ended: {exc!r}"
            finally:
                self._ws = None
                self.connected.clear()
                if self._drain_task:
                    self._drain_task.cancel()
                    self._drain_task = None
                for task in self._tasks.values():
                    task.cancel()
                self._tasks.clear()
                if self._client:
                    try:
                        await self._client.aclose()
                    except Exception:
                        pass
                    self._client = None
                try:
                    await ws.close()
                except Exception:
                    pass
            if not (self.reconnect and self._should_run) or self.draining:
                break
            delay = self._retry_delay(backoff)
            print(f"[bridge] disconnected; retrying in {delay:.0f}s")
            await asyncio.sleep(delay)
            backoff = min(backoff * 2, 30)

    async def _drain_timer(self, ws):
        """At the session-limit deadline: tell the server to stop routing,
        refuse new work, let in-flight requests finish, then disconnect."""
        if self._drain_at is None:
            return
        delay = self._drain_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        self.draining = True
        print("[bridge] draining — refusing new requests (session limit)")
        try:
            await self._send(ws, {"type": "drain"})
        except Exception:
            pass
        while self._tasks:  # don't kill mid-run work
            await asyncio.sleep(2)
        self._should_run = False
        try:
            await ws.close()
        except Exception:
            pass

    async def _session(self, ws):
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            mtype = msg.get("type")
            if mtype == "welcome":
                self.notebook_id = msg.get("notebook_id")
                self.public_base_url = msg.get("public_base_url")
                self.error = None
                self.connected.set()
                print(
                    f"[bridge] connected as {self.notebook_id}"
                    + (f" -> {self.public_base_url}"
                       if self.public_base_url else "")
                )
            elif mtype == "http_request":
                if self.draining:
                    await self._send(ws, {
                        "type": "http_error", "id": msg["id"], "status": 503,
                        "message": "notebook is draining (session limit)",
                    })
                else:
                    task = asyncio.ensure_future(self._handle(ws, msg))
                    self._tasks[msg["id"]] = task
            elif mtype == "http_cancel":
                task = self._tasks.pop(msg.get("id"), None)
                if task:
                    task.cancel()
            elif mtype == "drain":  # server-initiated drain
                self.draining = True
            elif mtype == "bye":
                return

    # ------------------------------------------------------------------
    # request relay: WS message -> local HTTP -> WS messages
    # ------------------------------------------------------------------

    async def _send(self, ws, msg):
        async with self._send_lock:
            await ws.send(json.dumps(msg))

    def _local_headers(self, headers):
        out = {
            k: v for k, v in (headers or {}).items()
            if k.lower() not in _HOP_BY_HOP
        }
        if self.api_key:
            out["Authorization"] = f"Bearer {self.api_key}"
        return out

    async def _handle(self, ws, msg):
        rid = msg["id"]
        try:
            await asyncio.wait_for(self._sema.acquire(), self.queue_timeout)
        except Exception:
            try:
                await self._send(ws, {
                    "type": "http_error", "id": rid, "status": 429,
                    "message": "notebook concurrency limit reached",
                })
            finally:
                self._tasks.pop(rid, None)
            return
        self.relayed += 1
        try:
            body = base64.b64decode(msg.get("body_b64") or b"")
            async with self._client.stream(
                msg.get("method", "GET"),
                f"http://127.0.0.1:{self.port}{msg.get('path', '/')}",
                headers=self._local_headers(msg.get("headers")),
                content=body or None,
            ) as resp:
                headers = {
                    k: v for k, v in resp.headers.items()
                    if k.lower() not in _HOP_BY_HOP
                }
                await self._send(ws, {
                    "type": "http_response_start",
                    "id": rid, "status": resp.status_code,
                    "headers": headers,
                })
                async for chunk in resp.aiter_bytes(65536):
                    await self._send(ws, {
                        "type": "http_response_chunk",
                        "id": rid,
                        "data_b64": base64.b64encode(chunk).decode(),
                    })
            await self._send(ws, {"type": "http_response_end", "id": rid})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            try:
                await self._send(ws, {
                    "type": "http_error", "id": rid,
                    "status": 502, "message": f"bridge relay failed: {exc!r}",
                })
            except Exception:
                pass
        finally:
            self._sema.release()
            self._tasks.pop(rid, None)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def ready(self, timeout=30):
        """Block until the first successful connect; raises on failure."""
        if not self.connected.wait(timeout):
            raise TimeoutError(
                f"bridge did not connect within {timeout}s: {self.error}"
            )
        if self.error and self.notebook_id is None:
            raise RuntimeError(self.error)
        return self

    def close(self):
        """Stop reconnecting and drop the current connection."""
        self._should_run = False

        async def _shutdown():
            if self._drain_task:
                self._drain_task.cancel()
            for task in self._tasks.values():
                task.cancel()
            if self._ws is not None:
                try:
                    await self._ws.close()
                except Exception:
                    pass

        loop = self._loop
        if loop and loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
            try:
                fut.result(timeout=10)
            except Exception:
                pass
        self.thread.join(timeout=10)

    def __repr__(self):
        state = "draining" if self.draining else (
            "connected" if self.connected.is_set() else "disconnected"
        )
        return f"<Bridge {self.notebook_id or '-'} {state} url={self.url!r}>"


def connect(url=None, key=None, port=8000, api_key=None, name=None,
            reconnect=True, drain_after=None, max_concurrent=8,
            queue_timeout=30):
    """
    Start the rendezvous bridge in a daemon thread. Returns a Bridge.

    url/key default to the RENDEZVOUS_URL / RENDEZVOUS_KEY env vars (loaded
    from Kaggle secrets). api_key defaults to proxy_api_key() and is sent to
    the LOCAL proxy only — remote clients authenticate to the server with
    their own key.

    drain_after: seconds until the bridge starts draining new requests
    (default BRIDGE_DRAIN_AFTER_S env, else 42600 = 12h session - 10min;
    0 disables). max_concurrent bounds in-flight relays; queue_timeout is
    how long an over-capacity request waits before getting a 429.
    """
    url = url or os.environ.get("RENDEZVOUS_URL")
    key = key or os.environ.get("RENDEZVOUS_KEY")
    if api_key is None:
        api_key = proxy_api_key() if "proxy_api_key" in globals() else None
    if drain_after is None:
        drain_after = float(os.environ.get("BRIDGE_DRAIN_AFTER_S", "42600"))
    if not url:
        raise ValueError("Set RENDEZVOUS_URL (env or secret) or pass url=")
    if not key:
        raise ValueError("Set RENDEZVOUS_KEY (env or secret) or pass key=")
    if url.startswith("https://"):
        url = "wss://" + url[len("https://"):]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://"):]
    bridge = Bridge(url, key, port, api_key, name or "kaggle-notebook",
                    reconnect, drain_after or None, max_concurrent,
                    queue_timeout)
    return bridge.start()


def disconnect(bridge):
    """Stop the bridge returned by connect()."""
    if bridge is not None:
        bridge.close()
