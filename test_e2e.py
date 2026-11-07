"""End-to-end test: stub notebook proxy -> real bridge cell -> rendezvous -> client.

Runs:
  1. a FastAPI stub on :18000 that mimics the notebook's local proxy
     (Bearer PROXY key, /v1/models, /v1/chat/completions with SSE),
  2. rendezvous.py as a subprocess on :18001,
  3. the ACTUAL bridge cell source extracted from the built notebook,
  4. real HTTP client requests through the rendezvous server.

Usage: ../.venv/bin/python test_e2e.py
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import uvicorn

HERE = Path(__file__).parent
NB_PATH = HERE.parent / "notebook" / "bench-testing.ipynb"

STUB_PORT = 18000          # stands in for the notebook's local proxy
RV_PORT = 18001            # rendezvous server under test
PROXY_KEY = "pk-test"      # the notebook-local key (like PROXY_API_KEY)
RENDEZVOUS_KEY = "rk-test" # notebook -> server auth
CLIENT_KEY = "ck-test"     # bench client -> server auth
ADMIN_KEY = "ak-test"      # /admin/* auth

FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        FAILED.append(name)


# ---------------------------------------------------------------------------
# Stub "notebook proxy" — mimics what the notebook serves on 127.0.0.1:8000
# ---------------------------------------------------------------------------

def start_stub():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    stub = FastAPI()
    state = {"last_auth": None}

    @stub.middleware("http")
    async def auth(request: Request, call_next):
        state["last_auth"] = request.headers.get("authorization")
        if request.url.path.startswith("/v1") and state["last_auth"] != f"Bearer {PROXY_KEY}":
            return JSONResponse(401, {"error": {"message": "Invalid proxy API key"}})
        return await call_next(request)

    @stub.get("/v1/models")
    def models():
        return {"object": "list", "data": [
            {"id": "gemini-2.5-flash", "object": "model", "created": 0,
             "owned_by": "kaggle", "upstream_model": "google/gemini-2.5-flash"},
            {"id": "gpt-5.6-luna", "object": "model", "created": 0,
             "owned_by": "kaggle", "upstream_model": "openai/gpt-5.6-luna"},
        ]}

    @stub.get("/v1/slow")
    async def slow():
        import asyncio
        await asyncio.sleep(2)
        return {"ok": True}

    @stub.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        model = payload.get("model", "?")
        if payload.get("stream"):
            def gen():
                for i, tok in enumerate(["Hel", "lo", "!"]):
                    yield f'data: {{"id":"c1","choices":[{{"delta":{{"content":"{tok}"}},"index":{i}}}]}}\n\n'
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return {
            "id": "c1", "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": {
                "role": "assistant", "content": f"stub reply for {model}"},
                "finish_reason": "stop"}],
        }

    config = uvicorn.Config(stub, host="127.0.0.1", port=STUB_PORT,
                            log_level="warning")
    server = uvicorn.Server(config)

    def run():
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    threading.Thread(target=run, daemon=True).start()
    for _ in range(100):
        if server.started:
            return state
        time.sleep(0.05)
    raise RuntimeError("stub did not start")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("== starting stub notebook proxy on :%d" % STUB_PORT)
    stub_state = start_stub()

    print("== starting rendezvous server on :%d" % RV_PORT)
    env = dict(
        __import__("os").environ,
        RENDEZVOUS_KEYS=RENDEZVOUS_KEY,
        CLIENT_API_KEYS=CLIENT_KEY,
        ADMIN_API_KEYS=ADMIN_KEY,
        PORT=str(RV_PORT),
        PUBLIC_BASE_URL=f"http://127.0.0.1:{RV_PORT}",
        REQUEST_TIMEOUT="15",
        MODEL_ALLOWLIST=("gemini-2.5-flash,google/gemini-2.5-flash,"
                         "gpt-5.6-luna,openai/gpt-5.6-luna"),
    )
    proc = subprocess.Popen(
        [sys.executable, str(HERE / "rendezvous.py")],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(100):
            try:
                httpx.get(f"http://127.0.0.1:{RV_PORT}/healthz", timeout=1).raise_for_status()
                break
            except Exception:
                if proc.poll() is not None:
                    print(proc.stdout.read())
                    raise RuntimeError("rendezvous exited early")
                time.sleep(0.1)

        # -- load the ACTUAL bridge cell from the built notebook --------------
        nb = json.loads(NB_PATH.read_text())
        bridge_src = "".join(nb["cells"][4]["source"])
        assert "def connect(" in bridge_src, "cell 4 is not the bridge cell"
        ns = {}
        exec(compile(bridge_src, "bridge_cell", "exec"), ns)

        print("== bridge with WRONG key must be rejected ==")
        bad = ns["connect"](url=f"ws://127.0.0.1:{RV_PORT}/connect",
                            key="rk-wrong", port=STUB_PORT, api_key=PROXY_KEY,
                            reconnect=False)
        try:
            bad.ready(timeout=10)
            check("bad key rejected", False, "ready() did not raise")
        except Exception as exc:
            check("bad key rejected", "403" in str(exc) or "rejected" in str(exc), str(exc)[:80])

        print("== connecting real bridge ==")
        bridge = ns["connect"](url=f"ws://127.0.0.1:{RV_PORT}/connect",
                               key=RENDEZVOUS_KEY, port=STUB_PORT,
                               api_key=PROXY_KEY, max_concurrent=1,
                               queue_timeout=0.2)
        bridge.ready(timeout=10)
        check("bridge connected", bridge.connected.is_set())
        check("notebook_id assigned", bool(bridge.notebook_id), bridge.notebook_id)
        check("public_base_url", bridge.public_base_url == f"http://127.0.0.1:{RV_PORT}/v1",
              str(bridge.public_base_url))

        base = f"http://127.0.0.1:{RV_PORT}"
        ch = {"Authorization": f"Bearer {CLIENT_KEY}"}
        ah = {"Authorization": f"Bearer {ADMIN_KEY}"}

        print("== client-facing API ==")
        r = httpx.get(f"{base}/healthz", timeout=5)
        check("healthz", r.status_code == 200 and r.json()["notebooks"] == 1, r.text[:80])

        r = httpx.get(f"{base}/v1/models", timeout=5)
        check("models requires client key", r.status_code == 401, str(r.status_code))

        r = httpx.get(f"{base}/v1/models", headers=ch, timeout=5)
        ids = [m["id"] for m in r.json().get("data", [])] if r.status_code == 200 else []
        check("models list", r.status_code == 200 and "gemini-2.5-flash" in ids, str(ids))
        check("models enabled flag",
              all(m.get("enabled") for m in r.json().get("data", [])))
        check("proxy key injected (models)", stub_state["last_auth"] == f"Bearer {PROXY_KEY}",
              str(stub_state["last_auth"]))

        r = httpx.post(f"{base}/v1/chat/completions", headers=ch, timeout=10,
                       json={"model": "gemini-2.5-flash",
                             "messages": [{"role": "user", "content": "hi"}]})
        ok = (r.status_code == 200
              and r.json()["choices"][0]["message"]["content"] == "stub reply for gemini-2.5-flash")
        check("chat completion", ok, r.text[:120])
        check("proxy key injected (chat)", stub_state["last_auth"] == f"Bearer {PROXY_KEY}")

        print("== streaming ==")
        chunks = []
        with httpx.stream("POST", f"{base}/v1/chat/completions", headers=ch,
                          timeout=10,
                          json={"model": "gpt-5.6-luna", "stream": True,
                                "messages": [{"role": "user", "content": "hi"}]}) as r:
            check("stream status", r.status_code == 200, str(r.status_code))
            check("stream content-type",
                  "text/event-stream" in r.headers.get("content-type", ""))
            for line in r.iter_lines():
                if line:
                    chunks.append(line)
        check("stream chunks", chunks and chunks[-1] == "data: [DONE]" and len(chunks) == 4,
              f"{len(chunks)} chunks")

        print("== allowlist ==")
        r = httpx.post(f"{base}/v1/chat/completions", headers=ch, timeout=5,
                       json={"model": "evil-model", "messages": []})
        check("non-allowlisted model rejected", r.status_code == 403
              and r.json()["error"]["type"] == "model_not_allowed", r.text[:120])

        print("== concurrency cap (max_concurrent=1, queue_timeout=0.2) ==")
        import concurrent.futures
        def slow_req(_):
            return httpx.get(f"{base}/v1/slow", headers=ch, timeout=10).status_code
        with concurrent.futures.ThreadPoolExecutor(2) as ex:
            codes = sorted(ex.map(slow_req, range(2)))
        check("one 200 one 429", codes == [200, 429], str(codes))

        print("== admin API ==")
        r = httpx.get(f"{base}/admin/status", timeout=5)
        check("admin requires key", r.status_code == 403, str(r.status_code))

        r = httpx.get(f"{base}/admin/status", headers=ah, timeout=5)
        check("admin status", r.status_code == 200
              and r.json()["notebooks"][0]["id"] == bridge.notebook_id, r.text[:120])

        r = httpx.post(f"{base}/admin/models/disable", headers=ah, timeout=5,
                       json={"model": "gemini-2.5-flash"})
        check("disable model", r.status_code == 200
              and "google/gemini-2.5-flash" in r.json()["matched_ids"], r.text[:120])

        r = httpx.post(f"{base}/v1/chat/completions", headers=ch, timeout=5,
                       json={"model": "gemini-2.5-flash", "messages": []})
        check("disabled model rejected", r.status_code == 403
              and r.json()["error"]["type"] in ("model_disabled",
                                                "model_not_allowed"),
              r.text[:120])

        # disabling by alias must also block the upstream id
        r = httpx.post(f"{base}/v1/chat/completions", headers=ch, timeout=5,
                       json={"model": "google/gemini-2.5-flash", "messages": []})
        check("disabled upstream id rejected", r.status_code == 403, str(r.status_code))

        r = httpx.post(f"{base}/admin/models/enable", headers=ah, timeout=5,
                       json={"model": "gemini-2.5-flash"})
        check("enable model", r.status_code == 200, r.text[:120])
        r = httpx.post(f"{base}/v1/chat/completions", headers=ch, timeout=10,
                       json={"model": "gemini-2.5-flash",
                             "messages": [{"role": "user", "content": "hi"}]})
        check("enabled model works", r.status_code == 200, str(r.status_code))

        print("== policy mode switch ==")
        r = httpx.post(f"{base}/admin/policy", headers=ah, timeout=5,
                       json={"mode": "blacklist", "models": []})
        check("switch to blacklist", r.status_code == 200
              and r.json()["policy"]["mode"] == "blacklist", r.text[:120])
        r = httpx.post(f"{base}/v1/chat/completions", headers=ch, timeout=10,
                       json={"model": "evil-model", "messages": []})
        check("blacklist empty allows all", r.status_code == 200,
              str(r.status_code))
        r = httpx.post(f"{base}/admin/policy", headers=ah, timeout=5,
                       json={"mode": "whitelist", "models": [
                           "gemini-2.5-flash", "google/gemini-2.5-flash",
                           "gpt-5.6-luna", "openai/gpt-5.6-luna"]})
        check("switch back to whitelist", r.status_code == 200
              and r.json()["policy"]["mode"] == "whitelist", r.text[:150])
        r = httpx.post(f"{base}/v1/chat/completions", headers=ch, timeout=5,
                       json={"model": "evil-model", "messages": []})
        check("allowlist restored", r.status_code == 403, str(r.status_code))

        print("== key management ==")
        r = httpx.get(f"{base}/admin/keys", headers=ah, timeout=5)
        kinds = r.json()
        check("keys listed by kind", r.status_code == 200
              and len(kinds["rendezvous"]) == 1
              and len(kinds["client"]) == 1
              and len(kinds["admin"]) == 1, str(kinds)[:150])
        check("values never listed",
              all("value" not in k for ks in kinds.values() for k in ks))

        r = httpx.post(f"{base}/admin/keys", headers=ah, timeout=5,
                       json={"kind": "client", "label": "test-key"})
        check("create client key", r.status_code == 200
              and r.json()["value"], r.text[:100])
        newkey, newid = r.json()["value"], r.json()["id"]

        r = httpx.get(f"{base}/v1/models", timeout=5,
                      headers={"Authorization": f"Bearer {newkey}"})
        check("new key works", r.status_code == 200, str(r.status_code))

        r = httpx.delete(f"{base}/admin/keys/{newid}", headers=ah, timeout=5)
        check("revoke key", r.status_code == 200, r.text[:100])
        r = httpx.get(f"{base}/v1/models", timeout=5,
                      headers={"Authorization": f"Bearer {newkey}"})
        check("revoked key rejected", r.status_code == 401, str(r.status_code))

        admin_id = kinds["admin"][0]["id"]
        r = httpx.delete(f"{base}/admin/keys/{admin_id}", headers=ah, timeout=5)
        check("last admin key refused", r.status_code == 409, str(r.status_code))

        print("== admin drain ==")
        r = httpx.post(f"{base}/admin/notebooks/{bridge.notebook_id}/drain",
                       headers=ah, timeout=5)
        check("admin drain accepted", r.status_code == 200, r.text[:100])
        import time as _t
        _t.sleep(0.3)  # let the drain frame reach the bridge
        check("bridge marked draining", bridge.draining)
        r = httpx.get(f"{base}/admin/status", headers=ah, timeout=5)
        check("status shows draining",
              r.json()["notebooks"][0]["draining"] is True, r.text[:150])
        r = httpx.post(f"{base}/v1/chat/completions", headers=ch, timeout=5,
                       json={"model": "gemini-2.5-flash", "messages": []})
        check("drained notebook not routed", r.status_code == 503
              and "draining" in r.json()["error"]["message"], r.text[:120])
        r = httpx.get(f"{base}/healthz", timeout=5)
        check("drained notebook still connected", r.json()["notebooks"] == 1)

        print("== disconnect ==")
        ns["disconnect"](bridge)
        deadline = time.time() + 10
        n = -1
        while time.time() < deadline:
            n = httpx.get(f"{base}/healthz", timeout=5).json()["notebooks"]
            if n == 0:
                break
            time.sleep(0.2)
        check("notebook removed on disconnect", n == 0, f"notebooks={n}")

        r = httpx.post(f"{base}/v1/chat/completions", headers=ch, timeout=5,
                       json={"model": "gemini-2.5-flash", "messages": []})
        check("503 when no notebooks", r.status_code == 503
              and "no notebooks" in r.json()["error"]["message"], r.text[:120])

        print("== drain timer (session limit) ==")
        drainer = ns["connect"](url=f"ws://127.0.0.1:{RV_PORT}/connect",
                                key=RENDEZVOUS_KEY, port=STUB_PORT,
                                api_key=PROXY_KEY, drain_after=0.5)
        drainer.ready(timeout=10)
        deadline = time.time() + 8
        n = -1
        while time.time() < deadline:
            n = httpx.get(f"{base}/healthz", timeout=5).json()["notebooks"]
            if n == 0:
                break
            time.sleep(0.3)
        check("drain-timer disconnects cleanly", n == 0 and drainer.draining,
              f"notebooks={n} draining={drainer.draining}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        out = proc.stdout.read() if proc.stdout else ""
        if FAILED:
            print("\n--- rendezvous log ---")
            print(out[-4000:])

    print()
    if FAILED:
        print("FAILED:", ", ".join(FAILED))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
