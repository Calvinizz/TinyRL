"""RolloutWorker: thin orchestrator over a RolloutBackend (Phase 7).

Phases 2-6: this class DID the generation (HF generate() inline).
Phase 7: the logic lives in backends; the worker keeps the constructor
signature (policy, tokenizer, cfg) so existing scripts work unchanged,
and delegates. Swap in any backend — including from tests — without
the pipeline above knowing.
"""

from __future__ import annotations

from tinyr.config import TrainingConfig
from tinyr.rollout.backend import RolloutBackend, RolloutResult  # noqa: F401 (re-export)
from tinyr.rollout.hf_backend import HFRolloutBackend


class RolloutWorker:
    def __init__(
        self,
        policy,
        tokenizer,
        cfg: TrainingConfig,
        backend: RolloutBackend | None = None,
    ):
        self.policy = policy
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.backend = backend or HFRolloutBackend(policy, tokenizer, cfg)

    def generate(self, questions: list[str]) -> RolloutResult:
        return self.backend.generate(questions)
