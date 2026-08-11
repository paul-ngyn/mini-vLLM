"""Wire schemas and the internal request object the scheduler operates on."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

import config


def _stop_list(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return list(stop)


class CompletionRequest(BaseModel):
    prompt: str
    model: str = config.MODEL_NAME
    max_tokens: int = Field(default=64, ge=1, le=config.MAX_TOKENS_LIMIT)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = config.MODEL_NAME
    max_tokens: int = Field(default=64, ge=1, le=config.MAX_TOKENS_LIMIT)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None


class GenerationRequest(BaseModel):
    """Schema for the simplified /generate endpoint."""

    prompt: str
    max_tokens: int = Field(default=64, ge=1, le=config.MAX_TOKENS_LIMIT)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)


@dataclass
class Request:
    """One in-flight generation, owned by the scheduler once submitted.

    `block_table is None` is the signal that this request still needs
    prefill; everything else in the pool is decoded together each iteration.
    KV storage itself lives in the engine's shared `PagedKVCache` -- a
    request only holds `block_table`, the list of page ids that make up its
    slice of that pool, freed back to the allocator on eviction.
    """

    prompt: str
    input_ids: list[int]
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)

    id: str = field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    block_table: list[int] | None = None
    cache_len: int = 0
    generated_ids: list[int] = field(default_factory=list)

    text: str = ""
    emitted: str = ""
    token_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    finished: bool = False
    finish_reason: str | None = None
    cancelled: bool = False

    created_at: float = field(default_factory=time.monotonic)
    first_token_at: float | None = None
    finished_at: float | None = None

    @property
    def prompt_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def completion_tokens(self) -> int:
        return len(self.generated_ids)

    @property
    def next_token_id(self) -> int:
        return self.generated_ids[-1]

    @property
    def ttft(self) -> float | None:
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.created_at

    def finish(self, reason: str) -> None:
        if self.finished:
            return
        self.finished = True
        self.finish_reason = reason
        self.finished_at = time.monotonic()
        self.token_queue.put_nowait(None)


def build_request(
    input_ids: list[int],
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: str | list[str] | None,
) -> Request:
    return Request(
        prompt=prompt,
        input_ids=input_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=_stop_list(stop),
    )
