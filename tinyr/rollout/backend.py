"""RolloutBackend: the generation contract the trainer is allowed to see.

Phase 7's point: the trainer/pipeline must NOT know how generation
happens — HF `generate()` on the training weights, a dedicated vLLM
engine, or a stub in tests. Everything above this interface (worker,
scripts, benchmarks) talks to `generate(questions) -> RolloutResult`
and nothing else. Everything below it (padding, sampling params, EOS
handling, weight sync) is the backend's private business.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class RolloutResult:
    """What every backend must return — identical layout across backends.

    Layout per row: [left-pad][prompt][response]; rows grouped as
    [q0]*G, [q1]*G, ... which matches the group advantage reshape.
    This dataclass is the CONTRACT: a backend that returns different
    shapes/dtypes breaks everything above it, so unit tests pin the
    layout with stub models (tests/test_backend.py).
    """

    prompt_ids: torch.Tensor      # LongTensor [N, W]
    response_ids: torch.Tensor    # LongTensor [N, R]
    attention_mask: torch.Tensor  # LongTensor [N, W+R]: 0 on prompt padding
                                  # and on tokens generated after first EOS
    responses: list[str]          # decoded response strings, length N
    prompt_width: int             # W


class RolloutBackend(ABC):
    """Abstract generation engine.

    Subclasses decide: which runtime (HF transformers / vLLM / ...),
    which device, how batching and padding happen, how weights stay in
    sync with the trainer. Callers decide only WHAT to sample.

    `questions` are raw user questions (strings, length B); the backend
    applies the chat template, samples `cfg.group_size` responses per
    question, and returns N = B * G rows in question-major order.
    """

    @abstractmethod
    def generate(self, questions: list[str]) -> RolloutResult:
        """Sample `group_size` responses per question."""
        raise NotImplementedError
