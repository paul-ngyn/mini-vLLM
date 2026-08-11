"""PagedAttention's actual contribution: fixed-size KV cache pages, allocated
and freed like OS memory pages instead of one growing tensor per request.

Storage is one big pool per layer, shape `(num_blocks, block_size, num_heads,
head_dim)` for keys and values. A request never owns a contiguous span of
that pool -- it owns a `block_table`, a list of block ids that may be
scattered anywhere in the pool. That indirection is what lets many requests
share the pool without pre-reserving worst-case-length spans, and what lets
a finished request's pages come back for reuse immediately.

Attention itself is computed by gathering a request's blocks into a
contiguous view and doing plain dot-product attention -- there is no fused
kernel here, just the block-table memory management vLLM's paged attention
introduced.
"""

from __future__ import annotations

import torch

import config


class OutOfMemoryError(RuntimeError):
    """Raised when the block pool has no free pages left to allocate."""


class BlockAllocator:
    """Free-list allocator handing out page ids from a fixed-size pool."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks))

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise OutOfMemoryError("paged KV cache has no free blocks left")
        return self._free.pop()

    def free(self, block_ids) -> None:
        self._free.extend(block_ids)


class PagedKVCache:
    """The pooled K/V storage plus the allocator that hands out its pages."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = config.PAGED_BLOCK_SIZE,
        num_blocks: int = config.PAGED_NUM_BLOCKS,
        dtype: torch.dtype = config.DTYPE,
        device: str = config.DEVICE,
    ) -> None:
        self.block_size = block_size
        self.num_layers = num_layers
        self.device = device

        shape = (num_layers, num_blocks, block_size, num_heads, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.allocator = BlockAllocator(num_blocks)

    def ensure_slot(self, block_table: list[int], num_tokens: int) -> None:
        """Grow `block_table` with a fresh page if position `num_tokens` needs one."""
        if num_tokens % self.block_size == 0:
            block_table.append(self.allocator.allocate())

    def free(self, block_table: list[int]) -> None:
        self.allocator.free(block_table)

    def gather(self, layer: int, block_table: list[int], num_tokens: int):
        """Read back the `num_tokens` cached K/V rows for one request as
        contiguous `(num_tokens, num_heads, head_dim)` tensors."""
        full_blocks, remainder = divmod(num_tokens, self.block_size)
        block_ids = block_table[:full_blocks]
        keys = [self.k_cache[layer, block_ids]] if block_ids else []
        values = [self.v_cache[layer, block_ids]] if block_ids else []
        if keys:
            keys[0] = keys[0].reshape(-1, *keys[0].shape[2:])
            values[0] = values[0].reshape(-1, *values[0].shape[2:])
        if remainder:
            keys.append(self.k_cache[layer, block_table[full_blocks], :remainder])
            values.append(self.v_cache[layer, block_table[full_blocks], :remainder])
        return torch.cat(keys, dim=0), torch.cat(values, dim=0)
