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

    # Version of the weights this backend is currently SERVING. None
    # means "tracks the policy object in memory" (the HF case): there
    # is no sync step, so the only honest stamp is the trainer's
    # current version at generate() time. A dedicated engine sets this
    # in sync_policy(); the producer stamps experiences from it, so an
    # engine that lags behind the trainer produces honestly-stamped
    # stale experiences instead of current-stamped lies.
    served_version: int | None = None

    @abstractmethod
    def generate(self, questions: list[str]) -> RolloutResult:
        """Sample `group_size` responses per question."""
        raise NotImplementedError

    def sync_policy(self, version: int, adapter_path: str) -> None:
        """Push a trainer-side LoRA checkpoint into the engine (P7.3).

        `adapter_path` is a peft `save_pretrained` directory
        (adapter_config.json + adapter_model.safetensors, ~9 MiB for
        r16 on q/k/v/o). Default: no-op — HFRolloutBackend shares the
        policy module with the trainer, whose optimizer.step() mutates
        it in place, so the "sync" already happened for free. That free
        lunch is exactly what a separate engine loses and must buy
        back explicitly (vLLM: versioned LoRARequest).
        """

