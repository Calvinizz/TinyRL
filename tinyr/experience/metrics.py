"""Padding / length metrics for Phase 5 (TODO P5.1)."""

from __future__ import annotations

import torch


def response_lengths(response_mask: torch.Tensor) -> torch.Tensor:
    """TODO P5.1 (a): real response-token count per sequence.

    Args:  response_mask BoolTensor [N, T], True on response positions
    Returns: LongTensor [N] — row sums of the mask.

    One line. This is the length signal bucketing sorts on.
    """
    raise NotImplementedError


def padding_ratio(response_mask: torch.Tensor, response_width: int) -> float:
    """TODO P5.1 (b): share of the response BLOCK that is padding.

    Args:
        response_mask: BoolTensor [N, T], True on real response tokens
        response_width: R — the padded response width (exp.response_ids
            .shape[1]; the block width the batch is PAYING for)
    Returns:
        float in [0, 1]: 1 - real / (N * R).

    Note the denominator is N*R, NOT N*T — prompts are not padding, we
    only bill the response block (T = W + R includes the prompt block).
    A ratio of 0.6 means 60% of the response compute buys nothing.
    """
    raise NotImplementedError
