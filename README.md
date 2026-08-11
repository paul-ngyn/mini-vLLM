# Mini-vLLM — Build Your Own LLM Inference Engine

A minimal, educational implementation of a production-style LLM inference engine — the kind of system that powers [vLLM](https://github.com/vllm-project/vllm), TGI, and TensorRT-LLM — in a few hundred lines of Python.

It serves an OpenAI-compatible HTTP API backed by a continuous-batching scheduler, so any client that speaks the OpenAI completions protocol can point at it unchanged.

Inspired by the article ["I Built a Mini-vLLM From Scratch"](https://medium.com/@iamuchihadaniel236/i-built-a-mini-vllm-from-scratch-heres-what-i-learned-about-llm-inference-at-scale-4b991342d6c8) by Uchiha Daniel and the reference repository [DanielPopoola/mini-vLLM](https://github.com/DanielPopoola/mini-vLLM).

---

## What This Project Does

Mini-vLLM serves text generation from a small language model (GPT-2, 124M params by default) while demonstrating the **four core optimizations** that make real inference engines fast:

| Optimization | Problem it solves | Result |
|---|---|---|
| **KV Caching** | Naive generation recomputes every previous token on every step — O(n²) work | Store attention Key/Value states once, reuse them — O(n) |
| **Prefill/Decode Separation** | Prompt processing and token generation have incompatible shapes and resource profiles | Prefill runs once over the whole prompt (compute-bound); decode runs one token at a time (memory-bandwidth-bound) |
| **Continuous Batching** | Static batches force fast requests to wait for slow ones | Requests join the batch when they arrive and leave when they finish |
| **PagedAttention** | A contiguous per-request KV tensor forces padding every request up to the longest one in the batch, and wastes memory reserving worst-case length upfront | KV cache lives in fixed-size pages from a shared pool, indexed per-request by a block table; no padding, pages freed and reused the instant a request finishes |

### Why it matters (the motivating math)

Generating 100 tokens from a 5-token prompt *without* caching means reprocessing the growing sequence every step — roughly 5,000 token-processings for what should be ~105. The larger example: a 1,000-token prompt generating 500 tokens costs **375,000** token-processings naively vs. **1,500** with KV caching — a **250× reduction**.

---

## Measured Results

GPT-2 (124M) on CPU, 8 requests × 48 tokens each, greedy decoding:

| Mode | Wall time | Output tokens | Throughput | Mean TTFT |
|---|---|---|---|---|
| Sequential (one at a time) | 7.67 s | 384 | 50.1 tok/s | 0.053 s |
| Concurrent (continuous batching) | 2.82 s | 384 | **136.2 tok/s** | 0.223 s |

**2.72× throughput** for the same work, with peak batch size 8. Note the tradeoff that shows up in every real serving system: batching raises *throughput* but also raises *time-to-first-token*, because a new request waits for the current decode step before it can be prefilled. Reproduce with `python bench.py --n 8 --max-tokens 48`.

---

## Architecture

```
                 ┌────────────────────────────────────────────┐
   HTTP clients  │                 server.py                  │
  /v1/completions─►  FastAPI endpoints + SSE token streaming  │
      /metrics   │  admission control (429 when at capacity)  │
                 └───────────────┬────────────────────────────┘
                                 │ submits Request to the pool
                 ┌───────────────▼────────────────────────────┐
                 │               scheduler.py                 │
                 │  async background loop:                    │
                 │   • new request (block_table is None)?     │
                 │       → run prefill                        │
                 │   • otherwise → add to decode batch        │
                 │   • finished? → evict, free pages           │
                 └───────────────┬────────────────────────────┘
                                 │
                 ┌───────────────▼────────────────────────────┐
                 │               inference.py                 │
                 │  prefill()       — full prompt, once       │
                 │  batch_decode()  — 1 token/step, batched   │
                 │  sample_token()  — greedy / temperature+p  │
                 └───────────────┬────────────────────────────┘
                                 │
                 ┌───────────────▼────────────────────────────┐
                 │  gpt2_paged.py            paged_cache.py    │
                 │  manual GPT-2 forward ──► PagedKVCache pool │
                 │  (reuses HF submodules,   (block allocator, │
                 │   pages K/V per layer)     free list)       │
                 └──────────────────────────────────────────────┘
```

### Request lifecycle

1. Client POSTs to `/v1/completions` → a `Request` is created and submitted to `active_requests` (or rejected with 429 if the pool is full).
2. The scheduler loop sees `block_table is None` → runs **prefill** over the full prompt. Prefill allocates fresh pages from the shared `PagedKVCache` and writes the prompt's K/V into them; the request keeps only the list of page ids (`block_table`). This also produces the first token.
3. On subsequent iterations the request joins the **decode batch** alongside every other in-flight request. Each request's pages can live anywhere in the pool — no padding to a common length is needed.
4. Each decode step produces one token per request; a new page is allocated only when a request's current page fills up. The scheduler detokenizes and pushes the new text delta into that request's `token_queue`.
5. The streaming endpoint drains the queue and yields **Server-Sent Events** to the client.
6. When a request hits `max_tokens`, EOS, or a stop sequence, the scheduler evicts it and returns its pages to the free list immediately — everyone else keeps going uninterrupted. That's continuous batching plus paged memory reuse.

---

## Project Artifacts (File Structure)

```
mini-vllm/
├── server.py         # FastAPI app: OpenAI-compatible endpoints, SSE streaming, metrics
├── scheduler.py      # Async background loop: prefill/decode orchestration, eviction
├── inference.py      # Engine: prefill(), batch_decode(), sampling
├── gpt2_paged.py      # Manual GPT-2 forward pass wired to the paged KV cache
├── paged_cache.py     # PagedKVCache: block pool + free-list allocator
├── models.py         # Pydantic wire schemas + internal Request dataclass
├── metrics.py        # tokens/sec, batch size, TTFT and latency percentiles
├── config.py         # Env-overridable settings (model, device, batch limits, paging)
├── bench.py          # Load generator: sequential vs concurrent comparison
├── main.py           # Entry point
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Construction Steps

Build it in this order — each step is runnable and testable on its own.

### Step 0 — Environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

(Linux/macOS: `source venv/bin/activate`.)

### Step 1 — Naive generation loop (the baseline)

Load GPT-2 with HuggingFace `transformers`. Write the dumbest possible loop: feed the whole sequence in, take the argmax of the last logit, append, repeat. Time it. This is your O(n²) baseline and the thing every later step improves on.

### Step 2 — Add KV caching

Pass `use_cache=True` and capture `past_key_values` from the model output. On each subsequent step, feed **only the newest token** plus the cache. Verify output is identical to Step 1 but dramatically faster for long generations. This single change implicitly commits you to the prefill/decode architecture — the first call (whole prompt) and later calls (one token) are now structurally different.

### Step 3 — Split prefill and decode explicitly (`inference.py`)

- `Engine.prefill(request)` — runs the full prompt once, stores the KV cache, emits the first token.
- `Engine.batch_decode(requests)` — advances every request by exactly one token.
- `sample_token(logits, temperature, top_p)` — `temperature=0` is exact greedy argmax, which keeps the default deterministic; above zero it applies temperature scaling and optional nucleus (top-p) filtering.

### Step 4 — Data structures (`models.py`)

- `CompletionRequest` / `ChatCompletionRequest` (pydantic): the OpenAI-shaped wire schemas, with validation bounds on `max_tokens`, `temperature`, and `top_p`.
- `Request` (dataclass): the internal state — token ids, `block_table` (starts as `None`, the list of KV-cache page ids this request owns once prefilled), `cache_len`, `token_queue` (an `asyncio.Queue`), accumulated text, stop strings, finish reason, and timestamps for TTFT.

### Step 5 — The scheduler (`scheduler.py`)

An async background loop (started by the FastAPI lifespan hook) that, every iteration:

1. Prefills up to `MAX_PREFILLS_PER_STEP` requests whose `kv_cache is None`.
2. Batch-decodes up to `MAX_BATCH_SIZE` of the remaining active requests.
3. Detokenizes and pushes each new text delta into its request's `token_queue`.
4. Evicts finished requests and frees their caches.

**Critical gotcha:** inference is CPU-bound and will starve the asyncio event loop — requests can't even arrive while the model is running. The minimal fix is `await asyncio.sleep(0)` between steps to yield control. (Production systems run inference in a separate thread or process instead.)

Two details worth getting right: throttle prefills, because admitting many prompts at once stalls the decode of everyone already running; and detokenize the **whole completion** each step and emit only the new suffix, because individual BPE pieces don't always decode to valid text on their own.

### Step 6 — Batched decode: PagedAttention instead of padding

The naive fix for "each request's KV cache has a different length" is to pad every request up to the batch's longest sequence, concatenate, and trim afterward. That works (and is what this project did initially — see the git history / the [design tradeoffs write-up](#pagedattention) below) but it wastes memory on padding and forces a worst-case-length reservation per request.

PagedAttention replaces that with fixed-size **pages** and a per-request **block table**:

1. `paged_cache.py` owns one big pool per layer, shape `(num_blocks, block_size, num_heads, head_dim)`, plus a free-list `BlockAllocator`. Nothing is pre-reserved per request — pages are handed out on demand as a request grows past its current page.
2. `gpt2_paged.py` reimplements GPT-2's forward pass layer-by-layer (reusing the pretrained `c_attn`/`c_proj`/`mlp`/layernorm submodules, since HuggingFace's `past_key_values` API has no concept of a paged cache):
   - **Prefill** (`prefill_forward`) processes one full prompt at a time: normal dense causal attention, but every layer's K/V gets scattered into freshly allocated pages as it's computed.
   - **Decode** (`decode_forward`) advances a batch of single new tokens with ragged context lengths: each request writes its new K/V into its next free page slot, then `cache.gather()` reassembles that request's scattered pages into a contiguous view just long enough to score attention against — no padding to the batch max.
3. `inference.py`'s `Engine` owns one shared `PagedKVCache`; a `Request` holds only its `block_table`. Eviction (`scheduler.py`) frees a finished request's pages back to the allocator immediately, so they're available to the very next admitted request.

Verify this step by comparing output against HuggingFace's own `model.generate(do_sample=False)` for several prompts of different lengths. They should match token for token — see [`PagedAttention`](#pagedattention) below for exactly how.

### Step 7 — HTTP server with streaming (`server.py`)

- `POST /v1/completions` and `POST /v1/chat/completions` — OpenAI-shaped request and response bodies, both with `stream: true` support (SSE chunks terminated by `data: [DONE]`).
- `POST /generate` — a simpler endpoint that just streams `{"token": "..."}` events, useful while learning.
- `GET /v1/models`, `GET /metrics`, `GET /health`.

Streaming responses `await` deltas off `token_queue`. If the client disconnects mid-stream, the generator's `finally` block marks the request cancelled so the scheduler stops spending compute on a response nobody is reading.

### Step 8 — Metrics (`metrics.py`)

Track tokens/second, current/average/peak batch size, active and rejected request counts, and TTFT/latency percentiles. This is how you *see* continuous batching working.

### Step 9 — Benchmark it (`bench.py`)

Send the same workload twice — once one request at a time, once all at once — and compare throughput. Without this you're guessing about whether any of the above helped.

---

## Running

```bash
python main.py
```

Or with autoreload during development:

```bash
uvicorn server:app --reload
```

Configuration is environment-driven — for example `MINIVLLM_MODEL=distilgpt2`, `MINIVLLM_MAX_BATCH_SIZE=32`, `MINIVLLM_DEVICE=cuda`, `MINIVLLM_PORT=8080`. See `config.py`.

## API

### Completions

```bash
curl -X POST http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d "{\"prompt\": \"The future of AI is\", \"max_tokens\": 50}"
```

Streaming, with sampling and a stop sequence:

```bash
curl -N -X POST http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d "{\"prompt\": \"The future of AI is\", \"max_tokens\": 50, \"temperature\": 0.8, \"top_p\": 0.9, \"stop\": \"\\n\\n\", \"stream\": true}"
```

| Field | Default | Notes |
|---|---|---|
| `prompt` | — | required |
| `max_tokens` | 64 | capped by `MINIVLLM_MAX_TOKENS_LIMIT` |
| `temperature` | 0.0 | `0.0` = deterministic greedy decoding |
| `top_p` | 1.0 | nucleus sampling; only applies when `temperature > 0` |
| `stop` | none | string or list of strings |
| `stream` | false | SSE when true |

### Chat completions

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\": [{\"role\": \"user\", \"content\": \"Name three colors.\"}], \"max_tokens\": 40}"
```

GPT-2 is not instruction-tuned, so the messages are flattened into a plain `System:/User:/Assistant:` transcript and stop sequences prevent the model from writing the user's next turn. Expect nonsense answers — the point is protocol compatibility, not quality. Point `MINIVLLM_MODEL` at a small instruct model for coherent replies.

### Using the OpenAI client

Because the response shapes match, existing tooling works without modification:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
print(client.completions.create(model="gpt2", prompt="The future of AI is", max_tokens=30))
```

### Observability

```bash
curl http://127.0.0.1:8000/metrics
```

Returns request counts, token counters, batch-size gauges, and TTFT/latency percentiles.

## Benchmarking

```bash
python bench.py --n 8 --max-tokens 48
```

Options: `--n` (number of requests), `--max-tokens`, `--mode sequential|concurrent|both`, `--url`. It reports wall time, output tokens, throughput, mean TTFT, and the batched-vs-sequential speedup, then prints the server's own batching metrics for cross-checking.

**To watch continuous batching live:** run the benchmark in one terminal and poll `GET /metrics` in another — `current_batch_size` grows as requests arrive and shrinks as they finish, with nobody waiting for a "batch boundary."

---

## Key Lessons

- **Caching isn't optional** — recomputation cost is catastrophic (250× in the worked example).
- **KV caching forces architecture** — once you cache, prefill and decode are different operations, and the scheduler, batching, and request lifecycle all follow from that split.
- **Batching trades latency for throughput** — 2.72× more tokens/sec here, but roughly 4× worse time-to-first-token. Which one you optimize is a product decision, not a technical one.
- **Batching overhead is real** — on a small CPU model the win is modest and can invert with tiny generations; it compounds on 7B+ GPU models with many concurrent users.
- **Padding correctness is subtle** — a wrong mask or missing `position_ids` produces output that looks plausible but silently diverges from single-request generation. Diff against `model.generate` rather than eyeballing it.
- **Async Python vs. CPU-bound work is a real conflict** — `asyncio.sleep(0)` is the educational fix; threads or processes are the real one.

## PagedAttention

This project implements vLLM's signature memory-management idea directly, not just KV caching: the KV cache is a pool of fixed-size pages (`paged_cache.py`) indexed per-request by a block table, instead of one contiguous per-request tensor padded to the batch max. Pages are allocated on demand and freed back to the pool the instant a request finishes.

### Running it

Nothing extra to do — it's on by default. Start the server the normal way:

```bash
python main.py
```

or

```bash
uvicorn server:app --reload
```

and hit it with the API examples above. Only GPT-2-family models work (`gpt2`, `distilgpt2`, `gpt2-medium`, ...) — the custom forward pass in `gpt2_paged.py` is hand-written against GPT-2's specific module layout, so a non-GPT2 `MINIVLLM_MODEL` will raise a clear `TypeError` at prefill time rather than silently doing the wrong thing.

### Configuration

Two new env vars control the page pool, in addition to the existing ones in `config.py`:

| Variable | Default | Meaning |
|---|---|---|
| `MINIVLLM_BLOCK_SIZE` | 16 | Tokens per page |
| `MINIVLLM_NUM_BLOCKS` | 2048 | Pages in the shared pool (total capacity = `BLOCK_SIZE × NUM_BLOCKS` tokens, pooled across every layer and every concurrent request) |

If the pool runs out of free pages (too many long concurrent requests for `NUM_BLOCKS`), `paged_cache.OutOfMemoryError` is raised — raise `MINIVLLM_NUM_BLOCKS` or lower `MINIVLLM_MAX_ACTIVE_REQUESTS`/`MAX_TOKENS_LIMIT`.

### Verifying it's correct

Paged attention only changes *how* the KV cache is stored and read, not the math — so a paged run and HuggingFace's own `model.generate(do_sample=False)` should produce byte-identical token streams for the same prompt:

```bash
python -c "
import torch
from inference import Engine
from models import build_request

eng = Engine(model_name='gpt2', device='cpu')
ids = eng.encode('The quick brown fox jumps over the lazy dog and')

r = build_request(ids, 'p', 15, 0.0, 1.0, None)
eng.prefill(r)
for _ in range(14):
    eng.batch_decode([r])
paged_ids = list(ids) + r.generated_ids

ref = eng.model.generate(torch.tensor([ids]), max_new_tokens=15, do_sample=False, pad_token_id=eng.tokenizer.eos_token_id)[0].tolist()
print('MATCH:', paged_ids == ref)
"
```

### The tradeoff

This is a pure-PyTorch *reference* implementation of paged attention's memory-management contract (block tables, free list, gather-then-score), not a fused CUDA/Triton kernel — vLLM's real speed win comes from a kernel that scores attention directly against pages without ever materializing a contiguous gathered copy. What this project gets you: the actual memory-management behavior (no padding, immediate page reuse, no worst-case reservation), verified byte-exact against HuggingFace. What it doesn't get you: the kernel-level throughput — `cache.gather()` copies memory every layer/step, and decode attention loops over the batch in Python rather than one fused, vectorized kernel launch. Good tradeoff for learning the idea; not a production-throughput implementation.

## Not Implemented (deliberately)

- Batched prefill (prompts are prefilled one at a time)
- Prefix/prompt caching across requests
- KV cache eviction, preemption, or memory-aware admission
- Request prioritization or fairness beyond FIFO
- Multi-GPU, quantization, speculative decoding, LoRA adapters

These are the natural next steps if you want to extend the project.

---

## Credits

- Article: [I Built a Mini-vLLM From Scratch — Uchiha Daniel](https://medium.com/@iamuchihadaniel236/i-built-a-mini-vllm-from-scratch-heres-what-i-learned-about-llm-inference-at-scale-4b991342d6c8)
- Reference implementation: [github.com/DanielPopoola/mini-vLLM](https://github.com/DanielPopoola/mini-vLLM) (MIT License)
