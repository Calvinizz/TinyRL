"""RolloutWorker: batched group sampling with the policy model.

Layout per row: [left-pad][prompt][response]; rows are grouped as
[q0]*G, [q1]*G, ... which matches the group advantage reshape.
"""

from dataclasses import dataclass

import torch

from tinyr.config import SYSTEM_PROMPT, TrainingConfig


@dataclass
class RolloutResult:
    prompt_ids: torch.Tensor      # LongTensor [N, W]
    response_ids: torch.Tensor    # LongTensor [N, R]
    attention_mask: torch.Tensor  # LongTensor [N, W+R]: 0 on prompt padding
                                  # and on tokens generated after first EOS
    responses: list[str]          # decoded response strings, length N
    prompt_width: int             # W


class RolloutWorker:
    def __init__(self, policy, tokenizer, cfg: TrainingConfig):
        self.policy = policy
        self.tokenizer = tokenizer
        self.cfg = cfg

    @torch.no_grad()
    def generate(self, questions: list[str]) -> RolloutResult:
        G, cfg = self.cfg.group_size, self.cfg

        encoded = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ],
                add_generation_prompt=True,
                tokenize=True,
            )["input_ids"]
            for q in questions
        ]
        batch = self.tokenizer.pad(
            {"input_ids": encoded}, padding=True, return_tensors="pt"
        )
        input_ids = batch["input_ids"].repeat_interleave(G, dim=0).to(cfg.device)
        attention_mask = (
            batch["attention_mask"].repeat_interleave(G, dim=0).to(cfg.device)
        )
        prompt_width = input_ids.shape[1]

        output_ids = self.policy.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=0,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = output_ids[:, prompt_width:]

        # attention on generated tokens: 1 up to and including first EOS, then 0
        gen_attention = torch.ones_like(generated)
        for i in range(generated.shape[0]):
            eos_pos = (generated[i] == self.tokenizer.eos_token_id).nonzero()
            if len(eos_pos) > 0:
                gen_attention[i, eos_pos[0, 0] + 1 :] = 0

        return RolloutResult(
            prompt_ids=input_ids,
            response_ids=generated,
            attention_mask=torch.cat([attention_mask, gen_attention], dim=1),
            responses=self.tokenizer.batch_decode(
                generated, skip_special_tokens=True
            ),
            prompt_width=prompt_width,
        )
