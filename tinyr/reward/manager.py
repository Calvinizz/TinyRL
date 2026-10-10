"""RewardManager: rule-based rewards (GSM8K exact-answer match)."""

import re

import torch


class RewardManager:
    @staticmethod
    def extract_answer(response: str) -> str | None:
        """TODO 1a: parse the model's final answer from a response string.

        Contract (the unit tests in tests/test_reward.py encode it):
        - '#### <digits>' anywhere in the text wins: return the digits
        - otherwise fall back to the LAST number appearing in the text
          (optionally signed: '-3' counts as '-3')
        - no number at all -> None

        You already wrote this in Phase 1 (scripts/train_grpo.py,
        extract_answer) — port it verbatim.
        """
        match = re.search(r'####\s*(\d+)',response)
        if match:
            return match.group(1)
        numbers = re.findall(r'-?\d+', response)   # fallback: last number
        return numbers[-1] if numbers else None

    @classmethod
    def compute(
        cls, responses: list[str], ground_truths: list[str]
    ) -> torch.Tensor:
        """TODO 1b: rule-based reward — exact-answer match.

        Args:
            responses:      list of N response strings
            ground_truths:  list of N clean answer strings (e.g. '72')
        Returns:
            torch.Tensor[float32] of shape [N]: 1.0 exact match else 0.0
            (extract returned None -> 0.0; CPU tensor is fine, the caller
            moves it to the training device)

        Port your Phase 1 compute_rewards.
        """
        rewards = torch.zeros(len(responses), dtype=torch.float32)
        for i in range(len(responses)):
            # 调用类的方法而不是全局， cls就是本类
            resp = cls.extract_answer(responses[i])
            gt = ground_truths[i]
            rewards[i] = 1.0 if resp is not None and resp == gt else 0.0
        return rewards
