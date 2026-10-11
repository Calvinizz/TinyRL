"""VLLMRolloutBackend: dedicated inference engine behind the Phase-7 interface.

TODO P7.2 (390 server; the 1080 Ti cannot run vLLM — sm_61 < 7.0, see
README Phase 7). Importable nowhere-else: vllm is imported LAZILY inside
methods so this module never breaks environments without it; its unit
tests are `pytest.importorskip("vllm")` — skipped locally, red until
implemented on the server.

Why this backend exists at all (the Phase 4/6 receipts): HF generate()
runs the sampling loop in Python at ~60-80 ms of host overhead per
decode step and ~12 tok/s per sequence, and on a single GPU rollout and
training kernels serialize on one CUDA stream. A dedicated engine
attacks the first (continuous batching, paged KV, no Python per-token
loop) and enables attacking the second (rollout on its own device —
docs/inference_study.md).
"""

from __future__ import annotations

import torch

from tinyr.config import SYSTEM_PROMPT, TrainingConfig
from tinyr.rollout.backend import RolloutBackend, RolloutResult


class VLLMRolloutBackend(RolloutBackend):
    """Group sampling with a vLLM engine over the SAME policy weights.

    Construction (P7.2 (a)):

        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model_path, dtype="float16",
                       gpu_memory_utilization=..., max_model_len=cfg
                       .max_new_tokens + <max prompt len>, seed=cfg.seed)

    Use the tokenizer's chat template OFFLINE — tokenize each question
    to ids, count tokens, and feed vLLM raw `TokensPrompt`s (prompt_
    token_ids=...). vLLM applying the template itself is the other
    option, but offline tokenization gives us the exact prompt ids we
    must reconstruct the contract tensors from. Keep the HF tokenizer
    for pad/eos ids and decoding.

    generate() (P7.2 (b)) — the contract is the WHOLE difficulty; vLLM
    returns per-sequence python lists of ids with NO padding, while
    RolloutResult promises [left-pad][prompt][response] tensors. So:

      1. chat-template + tokenize each question offline -> id lists
         (unequal lengths — vLLM never needed them equal)
      2. SamplingParams(n=cfg.group_size, temperature=cfg.temperature,
         top_p=cfg.top_p, max_tokens=cfg.max_new_tokens) — vLLM samples
         all G responses per prompt in ONE request (no repeat_
         interleave needed)
      3. W = max prompt len; build prompt_ids [N, W] by LEFT-padding
         each row with tokenizer.pad_token_id, attention 0 on the pad —
         byte-for-byte the layout HFRolloutBackend produces
      4. response rows: vLLM's per-sample ids padded RIGHT with pad_
         token_id to the batch max R (padding is trailing — response
         side); attention 1 only up to each row's real length, i.e.
         through its first EOS
      5. decoded strings: tokenizer.batch_decode(response_ids) with
         skip_special_tokens=True
      6. move tensors to cfg.device before returning

    Row order: question-major (all G samples of q0 first) — that is the
    group-advantage layout; n=G preserves it per prompt.

    Weight sync with the trainer is deliberately OUT of scope here —
    see the P7.3 notes in docs/inference_study.md (vLLM LoRA hot-swap
    with versioned adapter names vs. verl-style collective_rpc push).
    First version samples from STATIC weights; enough to measure what
    the engine buys.

    What to verify on the server FIRST (before trusting any numbers):
    one generate() call, print all shapes, and diff a batch_decode of
    your response_ids against vLLM's own output.text for the same rows
    — layout bugs here poison every downstream tensor.
    """

    def __init__(self, model_path: str, tokenizer, cfg: TrainingConfig):
        # TODO P7.2 (a): build the LLM engine (lazy `from vllm import
        # LLM` INSIDE this method — module import must stay vllm-free).
        self.tokenizer = tokenizer
        self.cfg = cfg
        raise NotImplementedError("P7.2 (a): on the 390 server")

    def generate(self, questions: list[str]) -> RolloutResult:
        raise NotImplementedError("P7.2 (b): on the 390 server")
