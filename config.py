"""Runtime configuration, overridable via environment variables."""

import os

import torch


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


MODEL_NAME = os.environ.get("MINIVLLM_MODEL", "gpt2")

DEVICE = os.environ.get(
    "MINIVLLM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)

DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32

# Most requests decoded in a single forward pass. Extra requests wait a turn.
MAX_BATCH_SIZE = _env_int("MINIVLLM_MAX_BATCH_SIZE", 16)

# Prompts prefilled per scheduler iteration. Prefill is compute-heavy, so
# admitting too many at once stalls the decode of everyone already running.
MAX_PREFILLS_PER_STEP = _env_int("MINIVLLM_MAX_PREFILLS_PER_STEP", 2)

# Hard ceiling on in-flight requests; beyond this the server returns 429.
MAX_ACTIVE_REQUESTS = _env_int("MINIVLLM_MAX_ACTIVE_REQUESTS", 64)

# PagedAttention KV cache: fixed-size pages shared across all requests.
# Total capacity is BLOCK_SIZE * NUM_BLOCKS tokens, pooled across every layer.
PAGED_BLOCK_SIZE = _env_int("MINIVLLM_BLOCK_SIZE", 16)
PAGED_NUM_BLOCKS = _env_int("MINIVLLM_NUM_BLOCKS", 2048)

MAX_TOKENS_LIMIT = _env_int("MINIVLLM_MAX_TOKENS_LIMIT", 512)

# How long the scheduler sleeps when there is nothing to do.
IDLE_SLEEP = _env_float("MINIVLLM_IDLE_SLEEP", 0.005)

HOST = os.environ.get("MINIVLLM_HOST", "127.0.0.1")
PORT = _env_int("MINIVLLM_PORT", 8000)
