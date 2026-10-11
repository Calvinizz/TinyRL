"""Spec tests for Phase 5 (TODO P5.1-P5.3): padding metrics, bucketing,
trim, and token-weighted accumulation. Red until implemented."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tinyr.config import TrainingConfig
from tinyr.experience import Experience, padding_ratio, response_lengths
from tinyr.experience.experience import iter_micro_batches_bucketed
from tinyr.trainer import GRPOTrainer

VOCAB = 8
W = 3  # prompt width everywhere below (prompts fully real, no left pad)


def make_mask(lens: list[int]) -> torch.Tensor:
    """response_mask for response lengths `lens`, padded to max(lens)."""
    r = max(lens)
    return torch.tensor(
        [[0] * W + [1] * L + [0] * (r - L) for L in lens], dtype=torch.bool
    )


# ---------------------------------------------------------------- P5.1


class TestMetrics:
    def test_response_lengths(self):
        mask = make_mask([3, 1, 4])
        assert torch.equal(response_lengths(mask), torch.tensor([3, 1, 4]))

    def test_padding_ratio_exact(self):
        # 2 rows x R=4, real tokens 2+2 -> 1 - 4/8 = 0.5 exactly
        mask = make_mask([2, 2])
        assert padding_ratio(mask, 4) == 0.5

    def test_padding_ratio_approx(self):
        mask = make_mask([3, 1])  # real 4 of 2x5
        assert padding_ratio(mask, 5) == pytest.approx(0.6)

    def test_padding_ratio_zero_when_dense(self):
        mask = make_mask([4, 4])
        assert padding_ratio(mask, 4) == 0.0


# ---------------------------------------------------------------- P5.2


class TestBucketing:
    def test_sorted_then_chunked(self):
        # lengths [10,1,5,5,1] -> ascending stable order [1,4,2,3,0]
        # -> chunks of 2: [[1,4],[2,3],[0]]
        assert iter_micro_batches_bucketed(make_mask([10, 1, 5, 5, 1]), 2) == [
            [1, 4], [2, 3], [0],
        ]

    def test_covers_every_row_exactly_once(self):
        chunks = iter_micro_batches_bucketed(make_mask([3, 1, 4, 2]), 3)
        assert sorted(i for c in chunks for i in c) == [0, 1, 2, 3]

    def test_zero_means_one_big_batch(self):
        chunks = iter_micro_batches_bucketed(make_mask([2, 4]), 0)
        assert len(chunks) == 1 and sorted(chunks[0]) == [0, 1]


def make_exp(lens: list[int]) -> Experience:
    """A valid ragged Experience with optional fields filled."""
    n, r = len(lens), max(lens)
    T = W + r
    g = torch.Generator().manual_seed(5)
    ids = lambda *s: torch.randint(0, VOCAB, s, generator=g)
    attention = torch.tensor(
        [[1] * W + [1] * L + [0] * (r - L) for L in lens], dtype=torch.long
    )
    return Experience(
        prompt_ids=ids(n, W),
        response_ids=ids(n, r),
        attention_mask=attention,
        response_mask=make_mask(lens),
        rewards=torch.tensor([1.0, 0.0] * (n // 2)),
        advantages=torch.tensor([1.0, -1.0] * (n // 2)),
        old_logprobs=torch.randn(n, T - 1, generator=g),
        ref_logprobs=torch.randn(n, T - 1, generator=g),
    )


class TestTrim:
    def test_shapes_cut_to_longest_real_response(self):
        exp = make_exp([6, 2, 4])  # W=3, R=6, T=9
        short = exp.rows([1], trim=True)          # new_R=2 -> T'=5
        assert short.response_ids.shape == (1, 2)
        assert short.attention_mask.shape == (1, 5)
        assert short.response_mask.shape == (1, 5)
        assert short.old_logprobs.shape == (1, 4)  # shifted: T'-1
        tall = exp.rows([0, 1], trim=True)        # new_R=6 -> T'=9, no cut
        assert tall.attention_mask.shape == (2, 9)

    def test_content_is_a_prefix_of_the_original(self):
        exp = make_exp([6, 2, 4])
        sub = exp.rows([2, 1], trim=True)         # new_R=4 -> T'=7
        torch.testing.assert_close(
            sub.response_ids, exp.response_ids[[2, 1], :4]
        )
        torch.testing.assert_close(
            sub.response_mask, exp.response_mask[[2, 1], :7]
        )

    def test_real_tokens_survive_the_cut(self):
        exp = make_exp([6, 2, 4])
        assert exp.rows([1, 2], trim=True).n_response_tokens == 2 + 4


# ---------------------------------------------------------------- P5.3
# The mathematical contract: with UNEQUAL token counts per chunk,
# token-weighted accumulation must still equal the full batch — and the
# old 1/n_micro scaling must NOT (that bias is why the weights exist).


class DummyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 4)
        self.head = nn.Linear(4, VOCAB)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


def run_training(micro_batch_size: int, dynamic: bool) -> list[torch.Tensor]:
    torch.manual_seed(7)
    policy, ref = DummyLM(), DummyLM()
    trainer = GRPOTrainer(
        copy.deepcopy(policy),
        copy.deepcopy(ref),
        TrainingConfig(
            lr=1e-3, n_inner_epochs=1, max_grad_norm=10.0, device="cpu",
            micro_batch_size=micro_batch_size, dynamic_batching=dynamic,
        ),
    )
    exp = trainer.prepare(make_exp([4, 4, 2, 2]))
    trainer.train_step(exp)
    return [p.grad.detach().clone() for p in trainer.policy.parameters()]


def test_bucketed_weighted_equals_full_batch():
    full = run_training(micro_batch_size=0, dynamic=False)
    dyn = run_training(micro_batch_size=2, dynamic=True)
    for a, b in zip(full, dyn):
        assert torch.allclose(a, b, atol=1e-6)


def test_unweighted_unequal_chunks_are_biased():
    # same ragged batch, old 1/n_micro scaling (dynamic off): chunks
    # carry 8 and 4 tokens but get equal weight -> grads drift
    full = run_training(micro_batch_size=0, dynamic=False)
    biased = run_training(micro_batch_size=2, dynamic=False)
    assert not torch.allclose(
        torch.cat([g.flatten() for g in full]),
        torch.cat([g.flatten() for g in biased]),
        atol=1e-4,
    )
