"""Manual GPT-2 forward pass wired to a PagedKVCache.

HuggingFace's `past_key_values` only understands one contiguous tensor per
request, so there is no way to hand it a paged cache. This reimplements the
GPT-2 forward loop layer-by-layer, reusing the pretrained submodules
(`c_attn`, `c_proj`, `mlp`, layernorms) for everything except the attention
score computation, which reads and writes the paged blocks directly.

Prefill runs one request at a time (batch size 1) over its full prompt, so
attention there is a normal dense causal pass -- only the cache *writes* are
paged. Decode runs a batch of single new tokens with ragged context lengths,
so each request's attention gathers its own blocks before scoring.
"""

from __future__ import annotations

import torch

from paged_cache import PagedKVCache


def _assert_supported(model) -> None:
    if type(model).__name__ != "GPT2LMHeadModel":
        raise TypeError(
            "the paged-attention engine only implements GPT2's architecture; "
            f"got {type(model).__name__}. Use a GPT2-family model "
            "(gpt2, distilgpt2, gpt2-medium, ...)."
        )


@torch.inference_mode()
def prefill_forward(
    model,
    cache: PagedKVCache,
    input_ids: torch.Tensor,
    block_table: list[int],
) -> torch.Tensor:
    """Run the full prompt through the model, paging its K/V into `cache`.

    `input_ids` is `(1, seq_len)`. `block_table` is mutated in place with the
    freshly allocated block ids. Returns logits for the last position, shape
    `(vocab_size,)`.
    """
    _assert_supported(model)
    transformer = model.transformer
    device = input_ids.device
    seq_len = input_ids.shape[1]
    n_embd = model.config.n_embd
    num_heads = model.config.n_head
    head_dim = n_embd // num_heads

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    hidden = transformer.wte(input_ids) + transformer.wpe(position_ids)
    hidden = transformer.drop(hidden)

    for pos in range(seq_len):
        cache.ensure_slot(block_table, pos)
    block_size = cache.block_size
    block_idx = torch.tensor(
        [block_table[pos // block_size] for pos in range(seq_len)], device=device
    )
    offset_idx = torch.tensor(
        [pos % block_size for pos in range(seq_len)], device=device
    )

    causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=device).triu(1)

    for layer_idx, block in enumerate(transformer.h):
        residual = hidden
        h = block.ln_1(hidden)
        q, k, v = block.attn.c_attn(h).split(n_embd, dim=2)
        q = q.reshape(seq_len, num_heads, head_dim)
        k = k.reshape(seq_len, num_heads, head_dim)
        v = v.reshape(seq_len, num_heads, head_dim)

        cache.k_cache[layer_idx, block_idx, offset_idx] = k
        cache.v_cache[layer_idx, block_idx, offset_idx] = v

        q_ = q.transpose(0, 1)  # (heads, seq, dim)
        k_ = k.transpose(0, 1)
        v_ = v.transpose(0, 1)
        scores = torch.matmul(q_, k_.transpose(-2, -1)) / (head_dim**0.5)
        scores = scores + causal_mask
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v_).transpose(0, 1).reshape(1, seq_len, n_embd)
        attn_out = block.attn.c_proj(attn_out)

        hidden = residual + attn_out
        residual = hidden
        hidden = residual + block.mlp(block.ln_2(hidden))

    hidden = transformer.ln_f(hidden)
    logits = model.lm_head(hidden[0, -1, :])
    return logits


@torch.inference_mode()
def decode_forward(
    model,
    cache: PagedKVCache,
    input_ids: torch.Tensor,
    block_tables: list[list[int]],
    context_lens: list[int],
) -> torch.Tensor:
    """Advance a batch of requests by exactly one token each.

    `input_ids` is `(batch,)`, `block_tables[i]` / `context_lens[i]` are the
    block table and current token count for row `i` *before* this step.
    `block_tables` are mutated in place with any newly allocated pages.
    Returns logits, shape `(batch, vocab_size)`.
    """
    _assert_supported(model)
    transformer = model.transformer
    device = input_ids.device
    batch = input_ids.shape[0]
    n_embd = model.config.n_embd
    num_heads = model.config.n_head
    head_dim = n_embd // num_heads
    block_size = cache.block_size

    position_ids = torch.tensor(context_lens, device=device).unsqueeze(1)
    hidden = transformer.wte(input_ids.unsqueeze(1)) + transformer.wpe(position_ids)
    hidden = transformer.drop(hidden)

    for i, ctx_len in enumerate(context_lens):
        cache.ensure_slot(block_tables[i], ctx_len)
    new_block_idx = torch.tensor(
        [block_tables[i][ctx_len // block_size] for i, ctx_len in enumerate(context_lens)],
        device=device,
    )
    new_offset_idx = torch.tensor(
        [ctx_len % block_size for ctx_len in context_lens], device=device
    )

    for layer_idx, block in enumerate(transformer.h):
        residual = hidden
        h = block.ln_1(hidden)
        q, k, v = block.attn.c_attn(h).split(n_embd, dim=2)
        q = q.reshape(batch, num_heads, head_dim)
        k = k.reshape(batch, num_heads, head_dim)
        v = v.reshape(batch, num_heads, head_dim)

        cache.k_cache[layer_idx, new_block_idx, new_offset_idx] = k
        cache.v_cache[layer_idx, new_block_idx, new_offset_idx] = v

        outs = []
        for i in range(batch):
            keys, values = cache.gather(layer_idx, block_tables[i], context_lens[i] + 1)
            keys = keys.transpose(0, 1)  # (heads, total_len, dim)
            values = values.transpose(0, 1)
            qi = q[i].unsqueeze(1)  # (heads, 1, dim)
            scores = torch.matmul(qi, keys.transpose(-2, -1)) / (head_dim**0.5)
            probs = torch.softmax(scores, dim=-1)
            outs.append(torch.matmul(probs, values).squeeze(1))  # (heads, dim)
        attn_out = torch.stack(outs, dim=0).reshape(batch, 1, n_embd)
        attn_out = block.attn.c_proj(attn_out)

        hidden = residual + attn_out
        residual = hidden
        hidden = residual + block.mlp(block.ln_2(hidden))

    hidden = transformer.ln_f(hidden)
    logits = model.lm_head(hidden[:, -1, :])
    return logits
