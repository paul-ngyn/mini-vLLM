"""Counters and gauges the scheduler updates as it runs."""

from __future__ import annotations

import time
from collections import deque


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return round(ordered[index], 4)


class Metrics:
    def __init__(self, window: int = 256) -> None:
        self.started_at = time.monotonic()
        self.total_requests = 0
        self.completed_requests = 0
        self.rejected_requests = 0
        self.cancelled_requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.prefill_batches = 0
        self.decode_steps = 0

        self.active_requests = 0
        self.current_batch_size = 0
        self.peak_batch_size = 0

        self._ttft: deque[float] = deque(maxlen=window)
        self._latency: deque[float] = deque(maxlen=window)
        self._batch_sizes: deque[int] = deque(maxlen=window)

    def request_started(self, prompt_tokens: int) -> None:
        self.total_requests += 1
        self.prompt_tokens += prompt_tokens

    def request_rejected(self) -> None:
        self.rejected_requests += 1

    def record_prefill(self, count: int) -> None:
        if count:
            self.prefill_batches += 1

    def record_decode(self, batch_size: int) -> None:
        self.decode_steps += 1
        self.completion_tokens += batch_size
        self.current_batch_size = batch_size
        self.peak_batch_size = max(self.peak_batch_size, batch_size)
        self._batch_sizes.append(batch_size)

    def record_ttft(self, seconds: float) -> None:
        self._ttft.append(seconds)

    def request_finished(self, latency: float, cancelled: bool) -> None:
        self.completed_requests += 1
        self._latency.append(latency)
        if cancelled:
            self.cancelled_requests += 1

    def snapshot(self) -> dict:
        uptime = time.monotonic() - self.started_at
        avg_batch = (
            round(sum(self._batch_sizes) / len(self._batch_sizes), 2)
            if self._batch_sizes
            else 0.0
        )
        return {
            "uptime_seconds": round(uptime, 2),
            "requests": {
                "total": self.total_requests,
                "active": self.active_requests,
                "completed": self.completed_requests,
                "rejected": self.rejected_requests,
                "cancelled": self.cancelled_requests,
            },
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "output_tokens_per_second": round(
                    self.completion_tokens / uptime, 2
                )
                if uptime > 0
                else 0.0,
            },
            "batching": {
                "current_batch_size": self.current_batch_size,
                "average_batch_size": avg_batch,
                "peak_batch_size": self.peak_batch_size,
                "decode_steps": self.decode_steps,
                "prefill_batches": self.prefill_batches,
            },
            "latency_seconds": {
                "ttft_p50": _percentile(list(self._ttft), 50),
                "ttft_p95": _percentile(list(self._ttft), 95),
                "request_p50": _percentile(list(self._latency), 50),
                "request_p95": _percentile(list(self._latency), 95),
            },
        }
