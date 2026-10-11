"""Experience: the unit of data flowing from rollout to trainer."""

from dataclasses import dataclass

import torch

from tinyr.utils.assertions import assert_device, assert_dtype, assert_shape
from tinyr.experience.metrics import response_lengths

@dataclass
class Experience:
    """One training batch of group-sampled trajectories.

    Layout per row: [left-pad][prompt][response]; T = W + R.
    advantages / old_logprobs / ref_logprobs start as None and are filled
    by GRPOTrainer.prepare() before the update (mirrors the README's
    planned experience structure).
    """
    # ids 是 token 本身含有padding， mask 是掩码
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
        """Mask out everything except response tokens.

        Rows look like [left-pad][prompt][response]; `attention_mask` is
        already 0 on left padding and on tokens generated after the first
        EOS. Response tokens start at column `prompt_width`.

        Returns:
            torch.Tensor[bool] of shape [N, T]: True on response positions
        """
        N, T = attention_mask.shape
        position = torch.arange(T, device=attention_mask.device)
        resp_position = position >= prompt_width
        return resp_position & attention_mask.bool()

    def rows(self, idx, trim: bool = False) -> "Experience":
        """A new Experience restricted to rows `idx` (list/slice/tensor).

        Tensors keep their widths — only the batch dim is indexed, so
        prompt padding stays intact. advantages / old_logprobs /
        ref_logprobs are sliced along when present.

        trim=True (TODO P5.2 (a), Phase 5): also cut the response block
        down to the longest REAL response among the selected rows —
        right-side padding beyond that point is dropped, so the sliced
        tensors are exactly as wide as the selected rows need. The
        layout [left-pad][prompt][response][right-pad] means every row's
        real content ENDS at column W + its response length, so:

            new_R = max over selected rows of response_lengths
            new_T = W + new_R

        Slice prompt_ids not at all; response_ids[..., :new_R];
        attention/response_mask[..., :new_T]; old/ref logprobs
        [..., :new_T - 1] (they are shifted — T-1 wide). Left prompt
        padding stays: trimming it would require re-shifting every row
        (a refinement we skip; RoPE makes the left pad harmless).
        """
        if not trim:
            return Experience(
                prompt_ids=self.prompt_ids[idx],
                response_ids=self.response_ids[idx],
                attention_mask=self.attention_mask[idx],
                response_mask=self.response_mask[idx],
                rewards=self.rewards[idx],
                advantages=self.advantages[idx] if self.advantages is not None else None,
                old_logprobs=self.old_logprobs[idx] if self.old_logprobs is not None else None,
                ref_logprobs=self.ref_logprobs[idx] if self.ref_logprobs is not None else None,
            )
        # TODO P5.2 (a): the trimmed branch — new_R / new_T as above,
        # slice every [N, T] (and [N, T-1]) field to the new width and
        # rebuild the Experience. Reuse the untrimmed branch's slicing
        # pattern with the column limits applied.
        else:
             new_r = self.response_mask[idx].sum(dim=-1).max().item()
             W = self.prompt_ids.shape[1]
             new_t = W + new_r
             return Experience(
                 prompt_ids=self.prompt_ids[idx],
                 response_ids=self.response_ids[idx][:,:new_r],
                 attention_mask=self.attention_mask[idx][:,:new_t],
                 response_mask=self.response_mask[idx][:,:new_t],
                 rewards=self.rewards[idx],
                 advantages=self.advantages[idx] if self.advantages is not None else None,
                 old_logprobs=self.old_logprobs[idx][...,:new_t - 1] if self.old_logprobs is not None else None,
                 ref_logprobs=self.ref_logprobs[idx][...,:new_t - 1] if self.ref_logprobs is not None else None,
             )
        
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


def iter_micro_batches(n_rows: int, micro_batch_size: int):
    """Yield row-index lists chunking range(n_rows).

    micro_batch_size <= 0 (or None) means "one big batch" — the Phase 2
    behavior. The last chunk may be smaller than micro_batch_size.
    """
    if micro_batch_size is None or micro_batch_size <= 0:
        micro_batch_size = n_rows
    return (
        list(range(start, min(start + micro_batch_size, n_rows)))
        for start in range(0, n_rows, micro_batch_size)
    )


def iter_micro_batches_bucketed(
    response_mask: torch.Tensor, micro_batch_size: int
) -> list[list[int]]:
    """TODO P5.2 (b): micro-batches of SIMILAR-length rows.

    Sort row indices by real response length ascending — use
    tinyr.experience.metrics.response_lengths (P5.1 (a)); torch.argsort
    is stable, so equal lengths keep their original relative order
    (determinism). Then chunk the sorted order into consecutive groups
    of micro_batch_size (<=0 / None = one big batch, same convention as
    iter_micro_batches).

    Same-length rows land in the same chunk, so rows(..., trim=True)
    cuts almost nothing per chunk — that is the entire Phase-5 win:
    padding is paid per MICRO-BATCH width, not per full-batch width.

    Returns a list of lists (train_step indexes it twice — once for
    iteration, once for len()).
    """
    lengths = response_lengths(response_mask)
    order = torch.argsort(lengths).tolist()
    ans = []
    if micro_batch_size is None or micro_batch_size <= 0:
        return [order]
    for i in range(0, len(order), micro_batch_size):
        ans.append(order[i:i + micro_batch_size])

    return ans