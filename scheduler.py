"""The continuous-batching loop: admit, prefill, decode, evict, repeat."""

from __future__ import annotations

import asyncio
import time

import config
from inference import Engine
from metrics import Metrics
from models import Request


class QueueFullError(RuntimeError):
    """Raised when the active pool is already at MAX_ACTIVE_REQUESTS."""


class Scheduler:
    def __init__(
        self,
        engine: Engine,
        metrics: Metrics,
        max_batch_size: int = config.MAX_BATCH_SIZE,
        max_prefills_per_step: int = config.MAX_PREFILLS_PER_STEP,
        max_active: int = config.MAX_ACTIVE_REQUESTS,
    ) -> None:
        self.engine = engine
        self.metrics = metrics
        self.max_batch_size = max_batch_size
        self.max_prefills_per_step = max_prefills_per_step
        self.max_active = max_active

        self.active_requests: list[Request] = []
        self._running = False
        self._task: asyncio.Task | None = None

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # -- admission --------------------------------------------------------

    def submit(self, request: Request) -> Request:
        if len(self.active_requests) >= self.max_active:
            self.metrics.request_rejected()
            raise QueueFullError(
                f"server is at capacity ({self.max_active} concurrent requests)"
            )
        self.active_requests.append(request)
        self.metrics.request_started(request.prompt_tokens)
        self.metrics.active_requests = len(self.active_requests)
        return request

    # -- the loop ---------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            if not self.active_requests:
                await asyncio.sleep(config.IDLE_SLEEP)
                continue

            waiting = [
                r
                for r in self.active_requests
                if r.block_table is None and not r.finished
            ]
            for request in waiting[: self.max_prefills_per_step]:
                self.engine.prefill(request)
                self._emit(request)
                # Inference blocks the event loop, so hand control back before
                # the next prompt: it is the only chance new requests and
                # streaming responses get to make progress.
                await asyncio.sleep(0)
            self.metrics.record_prefill(len(waiting[: self.max_prefills_per_step]))

            decoding = [
                r
                for r in self.active_requests
                if r.block_table is not None and not r.finished
            ][: self.max_batch_size]

            if decoding:
                self.engine.batch_decode(decoding)
                for request in decoding:
                    self._emit(request)
                self.metrics.record_decode(len(decoding))
                await asyncio.sleep(0)

            self._evict()
            await asyncio.sleep(0)

    # -- per-token bookkeeping -------------------------------------------

    def _emit(self, request: Request) -> None:
        """Turn the newest token id into a text delta on the request queue."""
        if request.cancelled:
            request.finish("cancelled")
            return

        if request.first_token_at is None:
            request.first_token_at = time.monotonic()
            self.metrics.record_ttft(request.ttft or 0.0)

        token_id = request.generated_ids[-1]

        if token_id == self.engine.eos_token_id:
            request.generated_ids.pop()
            request.finish("stop")
            return

        # Detokenize the whole completion each time and emit only the new
        # suffix: individual BPE pieces do not always decode to valid text
        # on their own.
        full_text = self.engine.decode(request.generated_ids)

        for stop in request.stop:
            index = full_text.find(stop)
            if index != -1:
                full_text = full_text[:index]
                request.text = full_text
                delta = full_text[len(request.emitted) :]
                if delta:
                    request.emitted = full_text
                    request.token_queue.put_nowait(delta)
                request.finish("stop")
                return

        request.text = full_text
        delta = full_text[len(request.emitted) :]
        if delta:
            request.emitted = full_text
            request.token_queue.put_nowait(delta)

        if request.completion_tokens >= request.max_tokens:
            request.finish("length")

    def _evict(self) -> None:
        """Drop finished requests so the batch shrinks without a barrier."""
        still_running = []
        for request in self.active_requests:
            if request.cancelled and not request.finished:
                request.finish("cancelled")
            if request.finished:
                self.engine.free(request)  # return pages to the pool immediately
                latency = (request.finished_at or time.monotonic()) - request.created_at
                self.metrics.request_finished(latency, request.cancelled)
            else:
                still_running.append(request)

        if len(still_running) != len(self.active_requests):
            self.active_requests = still_running
            self.metrics.active_requests = len(self.active_requests)
