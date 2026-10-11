"""VLLMRolloutBackend: dedicated inference engine behind the Phase-7 interface.

Implemented on the 390 server (the 1080 Ti cannot run vLLM — sm_61 < 7.0,
see README Phase 7). vllm is imported LAZILY inside methods so this module
never breaks environments without it; its unit tests are
`pytest.importorskip("vllm")` — skipped locally, green on the server
(`tests/test_backend.py::TestVLLMBackend`).

Why this backend exists at all (the Phase 4/6 receipts): HF generate()
runs the sampling loop in Python at ~60-80 ms of host overhead per
decode step and ~12 tok/s per sequence, and on a single GPU rollout and
training kernels serialize on one CUDA stream. A dedicated engine
attacks the first (continuous batching, paged KV, no Python per-token
loop) and enables attacking the second (rollout on its own device —
docs/inference_study.md).

Measured (390 server, 3090, batch 8x2): HF decode 92 tok/s → vLLM
1235 tok/s, 13.4x on identical silicon — `benchmarks/rollout.md`.
"""

from __future__ import annotations

import torch

from tinyr.config import SYSTEM_PROMPT, TrainingConfig
from tinyr.rollout.backend import RolloutBackend, RolloutResult


class VLLMRolloutBackend(RolloutBackend):
    """Group sampling with a vLLM engine over the SAME policy weights.

    __init__: lazy `from vllm import LLM` (module import stays
    vllm-free), engine over the base weights. dtype fp16; memory share
    and context length come from `cfg.gpu_memory_utilization` /
    `cfg.max_model_len` (config.py, phase-7 block).

    generate(): the contract is the WHOLE difficulty; vLLM returns
    per-sequence python lists of ids with NO padding, while
    RolloutResult promises [left-pad][prompt][response] tensors:

      1. chat-template + tokenize each question OFFLINE -> id lists
         (unequal lengths — vLLM never needed them equal); the HF
         tokenizer is kept for pad/eos ids and decoding
      2. SamplingParams(n=G, temperature, top_p, max_tokens) — vLLM
         samples all G responses per prompt in ONE request (no
         repeat_interleave needed); n=G preserves question-major order
      3. W = max prompt len; prompt_ids [N, W] LEFT-padded with
         tokenizer.pad_token_id, attention 0 on the pad —
         byte-for-byte the layout HFRolloutBackend produces
      4. response rows: vLLM's per-sample ids (EOS INCLUDED, verified
         against HF behavior) padded RIGHT with pad_token_id to the
         batch max R; attention 1 only up to each row's real length —
         through its first EOS
      5. decoded strings: tokenizer.batch_decode(skip_special_tokens)
      6. tensors moved to cfg.device before returning

    Weight sync (P7.3): sync_policy(version, adapter_path) swaps the
    engine's adapter via a versioned LoRARequest and stamps
    served_version; the trainer-side save+call lives in
    scripts/train_async.py::sync_weights. Strategy notes and the
    verl-style alternative: docs/inference_study.md §7.
    """

    def __init__(self, model_path: str, tokenizer, cfg: TrainingConfig):
        from vllm import LLM

        self.tokenizer = tokenizer
        self.cfg = cfg
        self.llm = LLM(
            model=model_path,
            dtype="float16",
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=cfg.max_model_len,
            seed=cfg.seed,
            enable_lora=True,          # P7.3: adapters applied per request
            max_loras=1,               # we serve exactly one policy
            max_lora_rank=cfg.lora_rank,
        )
        self._lora = None              # active LoRARequest until P7.3 syncs

    def sync_policy(self, version: int, adapter_path: str) -> None:
        """TODO P7.3 (a): register `adapter_path` as the weights that
        future generate() calls sample from.

          from vllm import LoRARequest          # lazy, as always
          self._lora = LoRARequest(name=f"policy_v{version}",
                                   path=adapter_path)
          self.served_version = version

        Pits, both bite silently:
          * the NAME is the engine's adapter-cache key. Version it —
            an unversioned name ("policy") makes every sync after the
            first a NO-OP (engine reuses the cached adapter forever).
          * set served_version LAST: generate() threads self._lora into
            llm.generate(lora_request=...) already (see below), and
            produce_one() stamps experiences from served_version — the
            two must flip together or a rollout can be sampled at v1
            while stamped v2.
        """
        raise NotImplementedError("P7.3 (a): versioned LoRARequest")

    def generate(self, questions: list[str]) -> RolloutResult:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        # 1. tokenize prompts offline (chat template, generation prompt)
        prompt_id_lists = []
        for q in questions:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if hasattr(encoded, "keys"):  # transformers returned a dict
                encoded = encoded["input_ids"]
            prompt_id_lists.append(list(encoded))

        # 2. one request per prompt, n=G samples each
        prompts = [
            TokensPrompt(prompt_token_ids=ids) for ids in prompt_id_lists
        ]
        sampling_params = SamplingParams(
            n=self.cfg.group_size,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_new_tokens,
        )
        outputs = self.llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=self._lora,   # None until P7.3 syncs an adapter
        )

        # 3. reconstruct left-padded prompts (question-major rows)
        G = self.cfg.group_size
        N = len(questions) * G
        W = max(len(ids) for ids in prompt_id_lists)
        pad_id = self.tokenizer.pad_token_id

        prompt_ids = torch.full((N, W), pad_id, dtype=torch.long)
        prompt_attention = torch.zeros((N, W), dtype=torch.long)

        row = 0
        for ids in prompt_id_lists:
            L = len(ids)
            for _ in range(G):
                prompt_ids[row, W - L:] = torch.tensor(ids)
                prompt_attention[row, W - L:] = 1
                row += 1

        # 4. collect and right-pad responses
        response_id_lists = []
        for request_output in outputs:
            for sample in request_output.outputs:
                response_id_lists.append(list(sample.token_ids))

        R = max(len(ids) for ids in response_id_lists)

        response_ids = torch.full((N, R), pad_id, dtype=torch.long)
        gen_attention = torch.zeros((N, R), dtype=torch.long)

        for i, ids in enumerate(response_id_lists):
            L = len(ids)
            response_ids[i, :L] = torch.tensor(ids)
            gen_attention[i, :L] = 1

        # 5. final contract
        attention_mask = torch.cat([prompt_attention, gen_attention], dim=1)

        responses = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )

        # 6. move tensors
        return RolloutResult(
            prompt_ids=prompt_ids.to(self.cfg.device),
            response_ids=response_ids.to(self.cfg.device),
            attention_mask=attention_mask.to(self.cfg.device),
            responses=responses,
            prompt_width=W,
        )
