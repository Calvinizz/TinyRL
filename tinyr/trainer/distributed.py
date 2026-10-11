"""Distributed-training helpers (Phase 8).

Pure functions only — everything here runs identically under pytest on
a laptop and under torchrun on the server. The impure parts (process
groups, NCCL, DDP wrappers) live in scripts/ddp_minimal.py as TODOs.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def shard_for_rank(items: Sequence[T], rank: int, world_size: int) -> list[T]:
    """items[rank::world_size] — the strided shard rank `rank` trains on.

    Strided (not contiguous blocks) on purpose: balanced within one
    item for ANY len(items), and deterministic across runs. Contiguous
    sharding hands rank 0 the short straw whenever len % world != 0 —
    and unbalanced shards are how ranks end up with different numbers
    of backward passes (see even_len).
    """
    return list(items[rank::world_size])


def even_len(n: int, world_size: int) -> int:
    """Largest multiple of world_size that is <= n.

    DDP requires every rank to run the SAME number of backward passes:
    the AllReduce lives in backward's hooks and is COLLECTIVE — a rank
    with one fewer micro-batch hangs the whole world at a bucket that
    never arrives. Trim the batch to a multiple of world_size (or pad,
    if losing rows is more expensive than inventing them) BEFORE
    sharding.
    """
    return n - (n % world_size)
