"""Load generator for the running server.

Sends the same workload twice — once one request at a time, once all at once —
so the effect of sharing decode steps across requests is directly visible.

    python bench.py --n 8 --max-tokens 64
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx

PROMPTS = [
    "The future of AI is",
    "In a distant galaxy, a lone engineer",
    "The three rules of good software design are",
    "Once upon a time in a city built on water,",
    "The most surprising thing about language models is",
    "Deep beneath the ocean floor, researchers found",
    "A short history of the printing press begins",
    "The recipe calls for two things nobody expects:",
]


class Result:
    def __init__(self) -> None:
        self.ttft: float = 0.0
        self.total: float = 0.0
        self.tokens: int = 0
        self.error: str | None = None


async def one_request(
    client: httpx.AsyncClient, url: str, prompt: str, max_tokens: int
) -> Result:
    result = Result()
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    started = time.perf_counter()
    try:
        async with client.stream("POST", url, json=body, timeout=600.0) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                chunk = json.loads(payload)
                text = chunk["choices"][0].get("text", "")
                if not text:
                    continue
                if result.tokens == 0:
                    result.ttft = time.perf_counter() - started
                result.tokens += 1
    except Exception as exc:  # noqa: BLE001 - surfaced in the report
        result.error = f"{type(exc).__name__}: {exc}"
    result.total = time.perf_counter() - started
    return result


async def run_sequential(url: str, prompts: list[str], max_tokens: int):
    async with httpx.AsyncClient() as client:
        started = time.perf_counter()
        results = [await one_request(client, url, p, max_tokens) for p in prompts]
        return results, time.perf_counter() - started


async def run_concurrent(url: str, prompts: list[str], max_tokens: int):
    async with httpx.AsyncClient() as client:
        started = time.perf_counter()
        results = await asyncio.gather(
            *(one_request(client, url, p, max_tokens) for p in prompts)
        )
        return results, time.perf_counter() - started


def summarize(name: str, results: list[Result], wall: float) -> dict:
    ok = [r for r in results if r.error is None]
    tokens = sum(r.tokens for r in ok)
    ttfts = [r.ttft for r in ok if r.ttft]
    return {
        "name": name,
        "requests": len(results),
        "failed": len(results) - len(ok),
        "wall_seconds": wall,
        "output_tokens": tokens,
        "throughput": tokens / wall if wall else 0.0,
        "ttft_mean": statistics.mean(ttfts) if ttfts else 0.0,
        "ttft_p95": max(ttfts) if len(ttfts) < 20 else statistics.quantiles(ttfts, n=20)[-1],
        "latency_mean": statistics.mean([r.total for r in ok]) if ok else 0.0,
    }


def report(rows: list[dict]) -> None:
    header = (
        f"{'mode':<12}{'reqs':>6}{'wall(s)':>10}{'tokens':>9}"
        f"{'tok/s':>9}{'ttft(s)':>10}{'latency(s)':>12}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['name']:<12}{row['requests']:>6}{row['wall_seconds']:>10.2f}"
            f"{row['output_tokens']:>9}{row['throughput']:>9.2f}"
            f"{row['ttft_mean']:>10.3f}{row['latency_mean']:>12.2f}"
        )
        if row["failed"]:
            print(f"  {row['failed']} request(s) failed")

    if len(rows) == 2 and rows[0]["throughput"] > 0:
        speedup = rows[1]["throughput"] / rows[0]["throughput"]
        print(f"\nbatched throughput: {speedup:.2f}x sequential")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the mini-vLLM server.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--n", type=int, default=8, help="number of requests")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--mode",
        choices=["sequential", "concurrent", "both"],
        default="both",
    )
    args = parser.parse_args()

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.n)]
    rows = []

    if args.mode in ("sequential", "both"):
        print(f"running {args.n} requests sequentially...")
        results, wall = await run_sequential(args.url, prompts, args.max_tokens)
        rows.append(summarize("sequential", results, wall))

    if args.mode in ("concurrent", "both"):
        print(f"running {args.n} requests concurrently...")
        results, wall = await run_concurrent(args.url, prompts, args.max_tokens)
        rows.append(summarize("concurrent", results, wall))

    report(rows)

    async with httpx.AsyncClient() as client:
        try:
            snapshot = (await client.get(args.url.replace("/v1/completions", "/metrics"))).json()
            print("server metrics:")
            print(json.dumps(snapshot["batching"], indent=2))
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
