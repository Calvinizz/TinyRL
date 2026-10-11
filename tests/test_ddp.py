"""Phase 8 — distributed helpers (pure functions; run anywhere, no GPUs).

The torchrun acceptances for scripts/ddp_minimal.py (P8.1) and
scripts/train_ddp.py (P8.2) deliberately live OUTSIDE pytest (they
need N processes, GPUs and a vLLM engine). The invariants those runs
depend on are tested here as pure math.
"""

import pytest
import torch

from tinyr.trainer.distributed import (
    ddp_loss_scale,
    even_len,
    group_rows_for_rank,
    shard_for_rank,
)


class TestShardForRank:
    def test_partition_loses_nothing(self):
        items = list(range(13))
        collected = [x for r in range(4) for x in shard_for_rank(items, r, 4)]
        assert sorted(collected) == items

    def test_strided_assignment(self):
        # balanced for any n: rank 0 never eats a ragged tail alone
        assert shard_for_rank(list(range(10)), 0, 3) == [0, 3, 6, 9]
        assert shard_for_rank(list(range(10)), 1, 3) == [1, 4, 7]
        assert shard_for_rank(list(range(10)), 2, 3) == [2, 5, 8]

    def test_balanced_within_one(self):
        items = list(range(13))
        sizes = [len(shard_for_rank(items, r, 4)) for r in range(4)]
        assert max(sizes) - min(sizes) <= 1

    def test_deterministic(self):
        assert shard_for_rank("abcde", 1, 2) == shard_for_rank("abcde", 1, 2)

    def test_world_of_one_is_identity(self):
        assert shard_for_rank([5, 6, 7], 0, 1) == [5, 6, 7]


class TestEvenLen:
    @pytest.mark.parametrize(
        "n,world,expected",
        [
            (12, 1, 12),
            (12, 2, 12),
            (12, 3, 12),
            (13, 3, 12),  # one row dropped, not one rank shorted
            (12, 4, 12),
            (7, 4, 4),
            (0, 3, 0),
            (2, 5, 0),
        ],
    )
    def test_values(self, n, world, expected):
        assert even_len(n, world) == expected

    def test_always_multiple_of_world(self):
        for n in range(0, 30):
            e = even_len(n, 4)
            assert e % 4 == 0
            assert e <= n

    def test_combined_shards_are_even(self):
        # the actual contract: every rank's shard has the SAME length,
        # else one rank's backward count differs and the world hangs
        for n in range(0, 20):
            e = even_len(n, 3)
            sizes = {len(shard_for_rank(list(range(e)), r, 3)) for r in range(3)}
            assert len(sizes) == 1


class TestGroupRowsForRank:
    def test_groups_never_split(self):
        # GRPO advantages are intra-group; a group across two ranks is
        # two fragments of a statistic neither can finish
        G = 2
        for rank in range(3):
            rows = group_rows_for_rank(9, G, rank, 3)
            groups = {r // G for r in rows}
            assert len(groups) == len(rows) // G

    def test_partition_covers_all_rows(self):
        G = 4
        all_rows = sorted(
            r for rank in range(2) for r in group_rows_for_rank(8, G, rank, 2)
        )
        assert all_rows == list(range(8 * G))

    def test_equal_rows_per_rank_after_even_len(self):
        G = 2
        e = even_len(10, 3)  # 9 — the raw 10 would leave ranks ragged
        sizes = {len(group_rows_for_rank(e, G, r, 3)) for r in range(3)}
        assert len(sizes) == 1


class TestDDPLossScale:
    def test_weighted_average_equals_global_token_mean(self):
        # shards with UNEQUAL token counts; DDP averages rank losses,
        # so exactness needs scale_r = world * tok_r / total
        toks = torch.tensor([10.0, 30.0, 20.0])
        means = torch.tensor([1.0, 2.0, 3.0])
        world, total = 3, toks.sum()
        scales = torch.tensor(
            [ddp_loss_scale(t, total, world) for t in toks]
        )
        ddp_gradient_target = (scales * means).sum() / world
        global_mean = (toks * means).sum() / total
        assert torch.allclose(ddp_gradient_target, global_mean)

    def test_equal_shards_scale_to_one(self):
        # the case naive sharding silently assumes
        assert ddp_loss_scale(100.0, 200.0, 2) == 1.0
