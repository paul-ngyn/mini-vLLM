"""Model execution: prefill, batched decode, and PagedAttention KV storage.

The KV cache lives in one shared `PagedKVCache` pool owned by the engine,
not per-request. A request holds only a `block_table` -- the list of pages
that make up its slice of the pool -- so batching never needs the
pad-to-longest-request dance a contiguous per-request tensor forces.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from gpt2_paged import decode_forward, prefill_forward
from models import Request
from paged_cache import PagedKVCache


def sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Pick the next token id from a 1-D logits row.

    `temperature == 0` is exact greedy decoding, which keeps generation
    deterministic and reproducible by default.
    """
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())

    logits = logits.float() / temperature

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        # Keep the smallest prefix whose mass reaches top_p; the first token
        # always survives because its cumulative-before-itself is zero.
        remove = (cumulative - probs) > top_p
        sorted_logits[remove] = float("-inf")
        choice = torch.multinomial(torch.softmax(sorted_logits, dim=-1), 1)
        return int(sorted_idx[choice].item())

    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())


class Engine:
    """Owns the model, tokenizer, and paged KV cache pool."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        device: str = config.DEVICE,
        dtype: torch.dtype = config.DTYPE,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        except TypeError:  # transformers < 4.56 spells it torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype
            )
        self.model = self.model.to(device)
        self.model.eval()
        self.eos_token_id = self.tokenizer.eos_token_id

        num_heads = self.model.config.n_head
        head_dim = self.model.config.n_embd // num_heads
        self.cache = PagedKVCache(
            num_layers=self.model.config.n_layer,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def free(self, request: Request) -> None:
        if request.block_table:
            self.cache.free(request.block_table)
        request.block_table = None

    # -- phase 1: prefill ------------------------------------------------

    def prefill(self, request: Request) -> None:
        """Process the whole prompt in one pass and emit the first token."""
        input_ids = torch.tensor(
            [request.input_ids], dtype=torch.long, device=self.device
        )
        request.block_table = []
        logits = prefill_forward(self.model, self.cache, input_ids, request.block_table)

        token_id = sample_token(logits, request.temperature, request.top_p)
        request.cache_len = len(request.input_ids)
        request.generated_ids.append(token_id)

    # -- phase 2: batched decode -----------------------------------------

    def batch_decode(self, requests: list[Request]) -> None:
        """Advance every request in `requests` by exactly one token."""
        if not requests:
            return

        next_ids = torch.tensor(
            [r.next_token_id for r in requests], dtype=torch.long, device=self.device
        )
        block_tables = [r.block_table for r in requests]
        context_lens = [r.cache_len for r in requests]

        logits = decode_forward(self.model, self.cache, next_ids, block_tables, context_lens)

        for row, request in enumerate(requests):
            token_id = sample_token(
                logits[row], request.temperature, request.top_p
            )
            request.generated_ids.append(token_id)
            request.cache_len += 1
