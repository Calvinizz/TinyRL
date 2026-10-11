"""Spec tests for Phase 6 (TODO P6.2 / P6.3): PolicyVersion + policy_lag
+ the policy-version stamp surviving Experience.rows()."""

import threading

import pytest
import torch

from tinyr.experience import Experience
from tinyr.trainer import PolicyVersion, policy_lag

VOCAB, W = 8, 3


class TestPolicyVersion:
    def test_starts_at_zero(self):
        assert PolicyVersion().current == 0
        assert PolicyVersion(start=5).current == 5

    def test_bump_increments_and_returns_new_value(self):
        v = PolicyVersion()
        assert v.bump() == 1
        assert v.current == 1
        assert v.bump() == 2
        assert v.current == 2

    def test_concurrent_bumps_lose_nothing(self):
        # 4 threads x 250 bumps = 1000, no matter the interleaving.
        # A lockless `self._v += 1` is LOAD/ADD/STORE under the GIL and
        # WILL lose increments here — that is the bug the lock exists for.
        v = PolicyVersion()
        barrier = threading.Barrier(4)

        def bump_many():
            barrier.wait()
            for _ in range(250):
                v.bump()

        threads = [threading.Thread(target=bump_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert v.current == 1000


class TestPolicyLag:
    def test_zero_when_on_policy(self):
        assert policy_lag(current_version=5, stamped_version=5) == 0

    def test_positive_when_stale(self):
        assert policy_lag(current_version=8, stamped_version=5) == 3

    def test_negative_left_visible(self):
        # consuming a "future" version is a wiring bug — it must show up
        # as nonsense, not be clamped into a reassuring 0
        assert policy_lag(current_version=2, stamped_version=5) == -3


def make_stamped_exp(version: int, lens: list[int] = (4, 2)) -> Experience:
    n, r = len(lens), max(lens)
    g = torch.Generator().manual_seed(3)
    ids = lambda *s: torch.randint(0, VOCAB, s, generator=g)
    return Experience(
        prompt_ids=ids(n, W),
        response_ids=ids(n, r),
        attention_mask=torch.tensor(
            [[1] * W + [1] * L + [0] * (r - L) for L in lens], dtype=torch.long
        ),
        response_mask=torch.tensor(
            [[0] * W + [1] * L + [0] * (r - L) for L in lens], dtype=torch.bool
        ),
        rewards=torch.ones(n),
        policy_version=version,
    )


class TestStamp:
    def test_default_version_is_zero(self):
        assert make_stamped_exp(0).policy_version == 0

    def test_stamp_survives_rows_untrimmed(self):
        exp = make_stamped_exp(7)
        assert exp.rows([0]).policy_version == 7

    def test_stamp_survives_rows_trimmed(self):
        # the trim branch rebuilds the Experience field by field — forget
        # the stamp there and every bucketed micro-batch looks like v0
        exp = make_stamped_exp(7)
        assert exp.rows([1], trim=True).policy_version == 7
