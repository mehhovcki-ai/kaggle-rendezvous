# kaggle-rendezvous

WebSocket rendezvous server for Kaggle notebook model proxies — an ngrok-free
way to serve Kaggle Benchmarks models to remote OpenAI-compatible clients.

The notebook dials **out** over one WebSocket; the server relays inbound HTTP
requests to it as serialized HTTP frames (streaming/SSE included). No public
tunnel needed on the notebook side.

```
bench client --HTTPS--> funnel --HTTP--> rendezvous --WS--> notebook --> Kaggle Model Proxy
```

## Components

- `rendezvous.py` — the server (FastAPI + uvicorn). `python rendezvous.py`,
  config via env (`HOST`, `PORT`, `RENDEZVOUS_KEYS`, `CLIENT_API_KEYS`,
  `ADMIN_API_KEYS`, `MODEL_ALLOWLIST`/`MODEL_BLOCKLIST`, `PUBLIC_BASE_URL`,
  `REQUEST_TIMEOUT`).
- `ui.html` — minimal management UI served at `/ui` (notebooks, model
  policy, key management).
- `bridge_cell.py` — notebook-side WS bridge. Embedded verbatim into the
  notebook by `build_notebook.py` (also the code under test in `test_e2e.py`).
- `notebook/` — the Kaggle kernel (FastAPI OpenAI-compatible proxy over
  `kaggle_benchmarks` + the bridge). Rebuild with `python build_notebook.py`,
  push with `kaggle kernels push -p notebook --no-run`.
- `test_e2e.py` — end-to-end suite: stub local proxy + real bridge cell +
  real server (`../.venv/bin/python test_e2e.py`).

## Protocol

Notebook connects to `GET /connect` with `Authorization: Bearer <rendezvous
key>`. Server relays HTTP as JSON frames (`http_request` ->
`http_response_start` / `http_response_chunk` / `http_response_end`, or
`http_error`). `drain` frames flow both ways: a notebook nearing its 12h
session limit drains (stops accepting, finishes in-flight, disconnects);
admins can trigger the same via `POST /admin/notebooks/{id}/drain`.

## API sketch

- `GET /healthz`, `GET /`
- `GET /v1/models`, `POST /v1/chat/completions`, `ANY /v1/{path}` —
  client Bearer keys (`client` kind)
- `/admin/*` — admin Bearer keys: status, models, policy
  (whitelist XOR blacklist), model enable/disable, notebook
  drain/disconnect, and key management (`GET|POST|DELETE /admin/keys`)

Keys are held in memory; `*_KEYS` env vars only seed the store.
