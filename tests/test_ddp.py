"""Phase 8 — distributed helpers (pure functions; run anywhere, no GPUs).

The torchrun acceptance for scripts/ddp_minimal.py deliberately lives
OUTSIDE pytest (it needs N processes and NCCL):

    torchrun --nproc_per_node=3 scripts/ddp_minimal.py

must print identical all-True verdicts on every rank.
"""

import pytest

from tinyr.trainer.distributed import even_len, shard_for_rank


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
