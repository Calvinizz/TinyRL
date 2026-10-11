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


def group_rows_for_rank(
    n_prompts: int, group_size: int, rank: int, world_size: int
) -> list[int]:
    """Row indices this rank trains on, sharded at the PROMPT level.

    GRPO advantages are computed WITHIN a group (G responses to one
    prompt) — splitting a group across ranks hands each rank a fragment
    of a statistic neither can finish. Shard prompts, keep whole groups:
    rank r's rows = {p*G + g : p in r's prompt shard, g < G}.

    Apply even_len(n_prompts, world_size) first: unequal GROUP counts
    per rank both bias the loss weighting (see ddp_loss_scale) and
    desynchronize backward counts — the DDP hang, again.
    """
    prompts = shard_for_rank(range(n_prompts), rank, world_size)
    return [p * group_size + g for p in prompts for g in range(group_size)]


def ddp_loss_scale(rank_tokens: float, total_tokens: float, world_size: int) -> float:
    """Loss multiplier making DDP's gradient average the EXACT global
    token mean when each rank's loss is a token-mean over its shard.

    DDP averages: grad <- (1/W) * sum_r grad(loss_r). We want grad of
    sum_r (tok_r / total) * mean_r  (= the full-batch token mean), so
    each rank must scale its loss by W * tok_r / total (== 1 when the
    shards happen to be equal — the case naive sharding silently
    assumes). total_tokens comes from one scalar allreduce per step.
    """
    return world_size * rank_tokens / total_tokens
