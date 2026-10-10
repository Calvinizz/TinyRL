"""Spec tests for gradient accumulation (TODO P3.3).

Mathematical contract: when every micro-batch has the same number of
response tokens, accumulating `(loss / n_micro).backward()` over all
micro-batches must produce EXACTLY the same gradients as one
full-batch `loss.backward()`. The tests below run the real trainer on
a tiny CPU model and compare gradients — no GPU needed.
"""

import copy

import torch
from torch import nn
from types import SimpleNamespace

from tinyr.config import TrainingConfig
from tinyr.experience import Experience
from tinyr.trainer import GRPOTrainer

VOCAB = 8


class DummyLM(nn.Module):
    """Smallest causal-LM stand-in: forward returns an object with .logits."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 4)
        self.head = nn.Linear(4, VOCAB)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


def make_experience(n: int = 4, w: int = 3, r: int = 4) -> Experience:
    g = torch.Generator().manual_seed(123)
    ids = lambda *s: torch.randint(0, VOCAB, s, generator=g)
    return Experience(
        prompt_ids=ids(n, w),
        response_ids=ids(n, r),
        attention_mask=torch.ones(n, w + r, dtype=torch.long),
        response_mask=torch.ones(n, w + r, dtype=torch.bool),  # uniform counts!
        rewards=torch.tensor([1.0, 0.0] * (n // 2)),
    )


def run_training(micro_batch_size: int) -> list[torch.Tensor]:
    """Train identical copies of (policy, ref, experience); return grads."""
    torch.manual_seed(7)
    policy, ref = DummyLM(), DummyLM()
    trainer = GRPOTrainer(
        copy.deepcopy(policy),
        copy.deepcopy(ref),
        TrainingConfig(
            lr=1e-3, n_inner_epochs=1, max_grad_norm=10.0,
            micro_batch_size=micro_batch_size, device="cpu",
        ),
    )
    exp = trainer.prepare(make_experience())
    trainer.train_step(exp)
    return [p.grad.detach().clone() for p in trainer.policy.parameters()]


def test_single_micro_batch_equals_full_batch():
    full = run_training(micro_batch_size=0)      # one big batch (legacy)
    single = run_training(micro_batch_size=4)    # 4 micro-batches of 1 row
    for a, b in zip(full, single):
        assert torch.allclose(a, b, atol=1e-6)


def test_micro_batches_of_two_equal_full_batch():
    full = run_training(micro_batch_size=0)
    pairs = run_training(micro_batch_size=2)     # 2 micro-batches of 2 rows
    for a, b in zip(full, pairs):
        assert torch.allclose(a, b, atol=1e-6)
