"""FastAPI surface: OpenAI-compatible completions plus streaming and metrics."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from inference import Engine
from metrics import Metrics
from models import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    GenerationRequest,
    Request,
    build_request,
)
from scheduler import QueueFullError, Scheduler

metrics = Metrics()
engine: Engine | None = None
scheduler: Scheduler | None = None

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, scheduler
    engine = Engine()
    scheduler = Scheduler(engine, metrics)
    scheduler.start()
    yield
    await scheduler.stop()


app = FastAPI(
    title="mini-vLLM",
    description="A minimal LLM inference engine with continuous batching.",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


# -- helpers ---------------------------------------------------------------


def _submit(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop,
) -> Request:
    assert engine is not None and scheduler is not None
    input_ids = engine.encode(prompt)
    if not input_ids:
        raise HTTPException(status_code=400, detail="prompt encodes to zero tokens")

    request = build_request(
        input_ids=input_ids,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
    )
    try:
        return scheduler.submit(request)
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


async def _drain(request: Request):
    """Yield text deltas until the scheduler signals completion."""
    try:
        while True:
            delta = await request.token_queue.get()
            if delta is None:
                return
            yield delta
    except asyncio.CancelledError:
        request.cancelled = True
        raise
    finally:
        # Covers the client hanging up mid-stream: stop spending compute on a
        # response nobody is reading.
        if not request.finished:
            request.cancelled = True


async def _collect(request: Request) -> str:
    async for _ in _drain(request):
        pass
    return request.text


def _usage(request: Request) -> dict:
    return {
        "prompt_tokens": request.prompt_tokens,
        "completion_tokens": request.completion_tokens,
        "total_tokens": request.prompt_tokens + request.completion_tokens,
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _flatten_chat(messages: list[ChatMessage]) -> str:
    """GPT-2 has no chat template, so use a plain transcript format."""
    lines = []
    for message in messages:
        label = {"system": "System", "user": "User", "assistant": "Assistant"}[
            message.role
        ]
        lines.append(f"{label}: {message.content}")
    lines.append("Assistant:")
    return "\n".join(lines)


# -- OpenAI-compatible endpoints -------------------------------------------


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": config.MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mini-vllm",
            }
        ],
    }


@app.post("/v1/completions")
async def completions(body: CompletionRequest):
    request = _submit(
        body.prompt, body.max_tokens, body.temperature, body.top_p, body.stop
    )
    created = int(time.time())

    if not body.stream:
        text = await _collect(request)
        return {
            "id": request.id,
            "object": "text_completion",
            "created": created,
            "model": body.model,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": request.finish_reason,
                }
            ],
            "usage": _usage(request),
        }

    async def stream():
        async for delta in _drain(request):
            yield _sse(
                {
                    "id": request.id,
                    "object": "text_completion",
                    "created": created,
                    "model": body.model,
                    "choices": [
                        {
                            "text": delta,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
            )
        yield _sse(
            {
                "id": request.id,
                "object": "text_completion",
                "created": created,
                "model": body.model,
                "choices": [
                    {
                        "text": "",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": request.finish_reason,
                    }
                ],
                "usage": _usage(request),
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers=SSE_HEADERS
    )


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    prompt = _flatten_chat(body.messages)
    # Default stops keep the model from writing the user's next turn for them.
    stop = (
        body.stop
        if body.stop is not None
        else ["\nUser:", "\nSystem:", "\nAssistant:"]
    )
    request = _submit(prompt, body.max_tokens, body.temperature, body.top_p, stop)
    created = int(time.time())
    chat_id = request.id.replace("cmpl-", "chatcmpl-")

    if not body.stream:
        text = await _collect(request)
        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": request.finish_reason,
                }
            ],
            "usage": _usage(request),
        }

    async def stream():
        def chunk(delta: dict, finish_reason=None) -> str:
            return _sse(
                {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.model,
                    "choices": [
                        {"index": 0, "delta": delta, "finish_reason": finish_reason}
                    ],
                }
            )

        yield chunk({"role": "assistant"})
        async for delta in _drain(request):
            yield chunk({"content": delta})
        yield chunk({}, request.finish_reason)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers=SSE_HEADERS
    )


# -- simple endpoints -------------------------------------------------------


@app.post("/generate")
async def generate(body: GenerationRequest):
    request = _submit(
        body.prompt, body.max_tokens, body.temperature, body.top_p, None
    )

    async def stream():
        async for delta in _drain(request):
            yield _sse({"token": delta})
        yield _sse(
            {
                "done": True,
                "finish_reason": request.finish_reason,
                "text": request.text,
                "usage": _usage(request),
            }
        )

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers=SSE_HEADERS
    )


@app.get("/metrics")
async def get_metrics():
    return metrics.snapshot()


@app.get("/health")
async def health():
    ready = engine is not None and scheduler is not None
    return {
        "status": "ok" if ready else "starting",
        "model": config.MODEL_NAME,
        "device": config.DEVICE,
        "active_requests": metrics.active_requests,
    }
