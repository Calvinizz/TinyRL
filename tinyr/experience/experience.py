"""Experience: the unit of data flowing from rollout to trainer."""

from dataclasses import dataclass

import torch

from tinyr.utils.assertions import assert_device, assert_dtype, assert_shape


@dataclass
class Experience:
    """One training batch of group-sampled trajectories.

    Layout per row: [left-pad][prompt][response]; T = W + R.
    advantages / old_logprobs / ref_logprobs start as None and are filled
    by GRPOTrainer.prepare() before the update (mirrors the README's
    planned experience structure).
    """

    prompt_ids: torch.Tensor      # LongTensor  [N, W]
    response_ids: torch.Tensor    # LongTensor  [N, R]
    attention_mask: torch.Tensor  # LongTensor  [N, T]
    response_mask: torch.Tensor   # BoolTensor  [N, T]: True on response only
    rewards: torch.Tensor         # FloatTensor [N]

    advantages: torch.Tensor | None = None    # FloatTensor [N]
    old_logprobs: torch.Tensor | None = None  # FloatTensor [N, T-1]
    ref_logprobs: torch.Tensor | None = None  # FloatTensor [N, T-1]

    @property
    def input_ids(self) -> torch.Tensor:
        """[N, T] full sequence — prompt_ids and response_ids side by side."""
        return torch.cat([self.prompt_ids, self.response_ids], dim=1)

    @property
    def n_sequences(self) -> int:
        return self.prompt_ids.shape[0]

    @property
    def n_response_tokens(self) -> int:
        return int(self.response_mask.sum().item())

    @staticmethod
    def build_response_mask(
        attention_mask: torch.Tensor, prompt_width: int
    ) -> torch.Tensor:
        """TODO 2: mask out everything except response tokens.

        Rows look like [left-pad][prompt][response]; `attention_mask` is
        already 0 on left padding and on tokens generated after the first
        EOS. Response tokens start at column `prompt_width`.

        Returns:
            torch.Tensor[bool] of shape [N, T]: True on response positions
        Port your Phase 1 build_response_mask.
        """
        N,T = attention_mask.shape
        position = torch.arange(T,device=attention_mask.device)
        resp_position = position >= prompt_width
        return resp_position & attention_mask.bool()



    def __post_init__(self) -> None:
        """Validate shapes / dtypes / devices at construction time.

        Raises AssertionError (via the tinyr.utils.assertions helpers)
        naming the offending field. Note advantages/old_logprobs/ref_logprobs
        are usually None here — the trainer fills them after construction.
        """
        # 2D checks first: everything below unpacks these shapes
        assert_shape(self.prompt_ids, (None, None), "prompt_ids")
        assert_shape(self.response_ids, (None, None), "response_ids")

        # reference dims for the consistency checks
        N, W = self.prompt_ids.shape
        R = self.response_ids.shape[1]
        T = W + R

        # shapes
        assert_shape(self.response_ids, (N, R), "response_ids")
        assert_shape(self.attention_mask, (N, T), "attention_mask")
        assert_shape(self.response_mask, (N, T), "response_mask")
        assert_shape(self.rewards, (N,), "rewards")

        # dtypes
        assert_dtype(self.prompt_ids, torch.long, "prompt_ids")
        assert_dtype(self.response_ids, torch.long, "response_ids")
        assert_dtype(self.attention_mask, torch.long, "attention_mask")
        assert_dtype(self.response_mask, torch.bool, "response_mask")
        assert_dtype(self.rewards, torch.float32, "rewards")

        # devices (prompt_ids is the reference)
        device = self.prompt_ids.device
        for name in (
            "response_ids", "attention_mask", "response_mask", "rewards"
        ):
            assert_device(getattr(self, name), device, name)

        # optional fields, validated only when provided
        if self.advantages is not None:
            assert_shape(self.advantages, (N,), "advantages")
        if self.old_logprobs is not None:
            assert_shape(self.old_logprobs, (N, T - 1), "old_logprobs")
        if self.ref_logprobs is not None:
            assert_shape(self.ref_logprobs, (N, T - 1), "ref_logprobs")
