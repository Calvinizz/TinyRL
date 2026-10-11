"""HFRolloutBackend: transformers generate() behind the Phase-7 interface (TODO P7.1)."""

from __future__ import annotations

import torch

from tinyr.config import SYSTEM_PROMPT, TrainingConfig
from tinyr.rollout.backend import RolloutBackend, RolloutResult


class HFRolloutBackend(RolloutBackend):
    """Group sampling with the in-process HF policy — the Phase 1-6 path.

    This is a PORT, not new code: the Phase-2 `RolloutWorker.generate`
    (~30 lines) moves here almost verbatim, and the worker above
    becomes a thin delegator. The original lives in git history (the
    scaffold deleted it from worker.py on purpose — no copying files,
    actual porting):

        git show 1a1e446:tinyr/rollout/worker.py

    Porting it yourself is the exercise — after this, the trainer/
    pipeline no longer imports or knows about `policy.generate` at all.

    transformers 5.x quirks that still apply (they did in the worker):
      * `apply_chat_template(..., tokenize=True)` returns a
        BatchEncoding — take `["input_ids"]` for the id list.
      * tokenizer must be configured `padding_side="left"` (ModelManager
        does this): `generate()` needs the prompt at the RIGHT edge.
    """

    def __init__(self, policy, tokenizer, cfg: TrainingConfig):
        self.policy = policy
        self.tokenizer = tokenizer
        self.cfg = cfg

    def generate(self, questions: list[str]) -> RolloutResult:
        """TODO P7.1: port RolloutWorker.generate here, verbatim in spirit.

        The 6 steps (source: `git show 1a1e446:tinyr/rollout/worker.py`;
        tests/test_backend.py pins the contract with a stub policy):

          1. chat-template each question -> per-question id lists
             (system + user, add_generation_prompt=True, tokenize=True)
          2. tokenizer.pad into a batch -> input_ids / attention_mask
          3. repeat_interleave(G, dim=0) -> N = B*G rows, question-major
          4. policy.generate(..., max_new_tokens, do_sample=True,
             temperature, top_p, pad_token_id) -> [N, W + new]
          5. post-process: generated = output[:, W:]; per row, zero the
             attention_mask AFTER the first EOS token (EOS itself keeps
             attention 1)
          6. return RolloutResult(prompt_ids, response_ids,
             cat([prompt_attention, gen_attention]), decoded strings, W)

        Return the RolloutResult imported above — the shared contract
        type now lives in backend.py, not worker.py.
        """
        raise NotImplementedError("P7.1: port from tinyr/rollout/worker.py")
