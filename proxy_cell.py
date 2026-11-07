import hmac
import os
import secrets
import threading
import time
from typing import Any

import kaggle_benchmarks as kbench
import openai
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict


def proxy_api_key():
    return os.environ.setdefault("PROXY_API_KEY", secrets.token_urlsafe(32)).strip()


PROXY_API_KEY = proxy_api_key()
MODEL_ALIASES = {
    "gemini-3.8-flash": "google/gemini-3.8-flash",
    "claude-haiku-4-5": "anthropic/claude-haiku-4-5@20251001",
    "claude-haiku-5-5": "anthropic/claude-haiku-5-5",
    "claude-opus-4-5": "anthropic/claude-opus-4-5@20251101",
    "claude-opus-4-6": "anthropic/claude-opus-4-6@default",
    "claude-opus-4-7": "anthropic/claude-opus-4-7@default",
    "claude-opus-4-8": "anthropic/claude-opus-4-8@default",
    "claude-opus-5": "anthropic/claude-opus-5@default",
    "claude-opus-5-5": "anthropic/claude-opus-5-5@default",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4-6@default",
    "claude-sonnet-5": "anthropic/claude-sonnet-5@default",
    "claude-sonnet-5-5": "anthropic/claude-sonnet-5-5@default",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-3-flash-preview": "google/gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview": "google/gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview": "google/gemini-3.1-pro-preview",
    "gemini-3.5-flash-lite": "google/gemini-3.5-flash-lite",
    "gemini-3.6-flash": "google/gemini-3.6-flash",
    "gemini-4-argon": "google/gemini-4-argon",
    "gemma-4-26b-a4b": "google/gemma-4-26b-a4b",
    "gemma-4-31b": "google/gemma-4-31b",
    "gpt-5.4-2026-03-05": "openai/gpt-5.4-2026-03-05",
    "gpt-5.4-mini-2026-03-17": "openai/gpt-5.4-mini-2026-03-17",
    "gpt-5.4-nano-2026-03-17": "openai/gpt-5.4-nano-2026-03-17",
    "gpt-5.5-2026-04-23": "openai/gpt-5.5-2026-04-23",
    "gpt-5.6-luna": "openai/gpt-5.6-luna",
    "gpt-5.6-sol": "openai/gpt-5.6-sol",
    "gpt-5.6-terra": "openai/gpt-5.6-terra",
    "gpt-6-astra": "openai/gpt-6-astra",
    "gpt-6-luna": "openai/gpt-6-luna",
    "gpt-6-sol": "openai/gpt-6-sol",
    "gpt-6.1-sol": "openai/gpt-6.1-sol",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "grok-4.5-0708": "xai/grok-4.5-0708",
    "grok-4.6": "xai/grok-4.6",
}

# ---------------------------------------------------------------------------
# Model Proxy access — every kbench LLMChat wraps the same configured
# openai.OpenAI client (base_url = {MODEL_PROXY_URL}/openapi), so one client
# can serve any model slug the proxy supports.
# ---------------------------------------------------------------------------

def _shared_client():
    for llm in [kbench.llm, *getattr(kbench, "llms", {}).values()]:
        client = getattr(llm, "client", None)
        if client is not None and hasattr(client, "chat"):
            return client
    raise HTTPException(
        503,
        "No OpenAI-compatible Model Proxy client found — set MODEL_PROXY_URL "
        "and MODEL_PROXY_API_KEY, or run inside a Kaggle Benchmarks notebook.",
    )


def _default_model():
    slug = getattr(kbench.llm, "model", None)
    if not slug:
        raise HTTPException(400, "Request must include `model` (no default set).")
    return slug


def _model_aliases():
    slugs = set(getattr(kbench, "llms", None) or {})
    slugs.update(MODEL_ALIASES.values())
    default = getattr(kbench.llm, "model", None)
    if default:
        slugs.add(default)
    aliases = {}
    for slug in sorted(slugs):
        alias = slug.rsplit("/", 1)[-1].removesuffix("@default")
        if alias in aliases and aliases[alias] != slug:
            aliases[alias] = None
        else:
            aliases[alias] = slug
    aliases.update(MODEL_ALIASES)
    return aliases


def _resolve_model(model):
    if "/" in model:
        return model
    aliases = _model_aliases()
    if model in aliases and aliases[model] is None:
        raise HTTPException(
            400, f"Ambiguous model alias {model!r}; use a full upstream model ID"
        )
    return aliases.get(model, model)


def _model_allowed(model):
    """Model gate — set MODEL_ALLOWLIST (only these run) OR MODEL_BLOCKLIST
    (everything but these runs), comma-separated ids/aliases. If both are
    set the allowlist wins. Both empty = allow everything."""
    allow = {
        m.strip() for m in os.environ.get("MODEL_ALLOWLIST", "").split(",")
        if m.strip()
    }
    block = {
        m.strip() for m in os.environ.get("MODEL_BLOCKLIST", "").split(",")
        if m.strip()
    }
    if not allow and not block:
        return True
    aliases = _model_aliases()
    candidates = {model, aliases.get(model)}
    candidates.update(a for a, u in aliases.items() if u == model)
    candidates.discard(None)
    if allow:
        return bool(candidates & allow)
    return not bool(candidates & block)


# ---------------------------------------------------------------------------
# FastAPI app — OpenAI wire format, verbatim passthrough
# ---------------------------------------------------------------------------

def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        HTTPBearer(auto_error=False)
    ),
):
    if credentials is None or not hmac.compare_digest(
        credentials.credentials.encode(), proxy_api_key().encode()
    ):
        raise HTTPException(
            401, "Invalid proxy API key", headers={"WWW-Authenticate": "Bearer"}
        )


router = APIRouter(dependencies=[Depends(require_api_key)])


class ChatCompletionRequest(BaseModel):
    """OpenAI chat.completions.create params. Unknown fields pass through."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[dict[str, Any]]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    user: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    stream_options: dict[str, Any] | None = None
    metadata: dict[str, str] | None = None


@router.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": alias, "object": "model", "created": 0,
                "owned_by": "kaggle", "upstream_model": upstream,
            }
            for alias, upstream in sorted(_model_aliases().items())
            if upstream is not None
        ],
    }


def _sse(upstream):
    """Yield an openai.Stream as OpenAI-style SSE chunks."""
    try:
        for chunk in upstream:
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
    finally:
        close = getattr(upstream, "close", None)
        if callable(close):
            close()
    yield "data: [DONE]\n\n"


@router.post("/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    payload = req.model_dump(exclude_unset=True)
    stream = payload.pop("stream", False)
    model = _resolve_model(payload.pop("model", None) or _default_model())
    if not _model_allowed(model):
        raise HTTPException(403, f"Model {model!r} is blocked by the "
                                 "model allow/block list")

    try:
        upstream = _shared_client().chat.completions.create(
            model=model, stream=stream, **payload
        )
    except openai.APIStatusError as exc:
        if exc.status_code in (401, 403):
            return JSONResponse(
                status_code=502,
                content={"error": {"message": (
                    f"UPSTREAM Kaggle Model Proxy rejected its credentials "
                    f"({exc.status_code}); your PROXY_API_KEY was fine. Check "
                    f"MODEL_PROXY_API_KEY / that this is a Benchmarks notebook.")}},
            )
        body = exc.body
        if not isinstance(body, dict):
            body = {"message": "Upstream model request failed"}
        if "error" not in body:
            body = {"error": body}
        return JSONResponse(status_code=exc.status_code, content=body)
    except openai.APIConnectionError:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "Could not reach Kaggle Model Proxy"}},
        )

    if stream:
        return StreamingResponse(_sse(upstream), media_type="text/event-stream")

    # Real ChatCompletion, serialized faithfully (nulls included, like OpenAI).
    return upstream.model_dump(mode="json")


app = FastAPI(title="Kaggle Benchmarks Model Proxy", version="1.0.0")
app.include_router(router, prefix="/v1")


# ---------------------------------------------------------------------------
# Server + client helpers
# ---------------------------------------------------------------------------

def serve(host="127.0.0.1", port=8000, **uvicorn_kwargs):
    """
    Run the proxy in a daemon thread (notebook-friendly — the cell returns
    immediately). Returns the uvicorn.Server; base URL is
    http://{host}:{port}/v1.
    """
    import asyncio

    import uvicorn

    options = {"log_level": "warning", "loop": "asyncio", **uvicorn_kwargs}
    config = uvicorn.Config(app, host=host, port=port, **options)
    server = uvicorn.Server(config)
    errors = []

    def _run():
        # ipykernel patches asyncio.run without loop_factory support, which
        # breaks uvicorn.Server.run() — drive serve() on our own loop instead.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        except BaseException as exc:
            errors.append(exc)
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            asyncio.set_event_loop(None)
            loop.close()

    server.thread = threading.Thread(
        target=_run, daemon=True, name="kbench-openai-proxy"
    )
    server.thread.start()

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if errors:
            raise RuntimeError("Proxy startup failed") from errors[0]
        if not server.thread.is_alive():
            raise RuntimeError("Proxy thread exited during startup")
        if server.started:
            return server
        time.sleep(0.05)
    server.should_exit = True
    raise TimeoutError("Proxy did not start within 30 seconds")


def stop_server(server):
    server.should_exit = True
    server.thread.join(timeout=10)
    if server.thread.is_alive():
        raise RuntimeError("Proxy is still shutting down")


def openai_client(base_url="http://127.0.0.1:8000/v1", **kwargs):
    """openai.OpenAI pointed at the proxy (local base URL by default)."""
    import openai

    kwargs.setdefault("api_key", proxy_api_key())
    return openai.OpenAI(base_url=base_url, **kwargs)


class ChatSession:
    """
    Minimal stateful multi-turn helper over the proxy.

        chat = ChatSession(model="google/gemini-2.5-flash", system="You are terse.")
        chat.ask("Hi!")        # -> str
        chat.ask("And you?")   # history carried automatically
        chat.messages          # full OpenAI-style history (tool_calls preserved)
        chat.reset()           # clear; chat.reset(system="...") to re-seed
    """

    def __init__(self, model=None, system=None,
                 base_url="http://127.0.0.1:8000/v1", **client_kwargs):
        import os

        if model is None:
            model = (
                getattr(kbench.llm, "model", None)
                or os.environ.get("LLM_DEFAULT")
            )
        if not model:
            raise ValueError("ChatSession requires a model slug.")

        self._oai = openai_client(base_url=base_url, **client_kwargs)
        self.model = model
        self.messages = [{"role": "system", "content": system}] if system else []

    @property
    def history(self):
        return list(self.messages)

    def ask(self, content, **kwargs):
        """Append a user message, get a completion, append the reply, return text."""
        if kwargs.get("stream"):
            raise ValueError("Use create(stream=True) for streaming")
        messages = [*self.messages, {"role": "user", "content": content}]
        resp = self._oai.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        msg = resp.choices[0].message
        # model_dump preserves tool_calls etc. for faithful multi-turn history.
        self.messages = [*messages, msg.model_dump(exclude_none=True)]
        return msg.content or ""

    def send(self, content, **kwargs):
        """Alias for ask()."""
        return self.ask(content, **kwargs)

    def create(self, **kwargs):
        """Raw passthrough on current history WITHOUT appending the exchange."""
        return self._oai.chat.completions.create(
            model=self.model, messages=list(self.messages), **kwargs
        )

    def reset(self, system=None):
        self.messages = [{"role": "system", "content": system}] if system else []
        return self
