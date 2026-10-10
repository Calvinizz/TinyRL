"""Spec tests for Experience.build_response_mask (TODO 2).

Row layout: [left-pad][prompt][response]. attention_mask is 0 on left
padding and on post-EOS generated tokens; response starts at prompt_width.
"""

import torch

from tinyr.experience import Experience


def test_no_padding_row():
    # [prompt][resp][resp] with prompt_width=1 -> mask [0, 1, 1]
    attn = torch.tensor([[1, 1, 1]])
    mask = Experience.build_response_mask(attn, prompt_width=1)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [[False, True, True]]


def test_left_padded_row():
    # [pad][prompt][resp][resp][resp], prompt_width=2 -> [0,0,1,1,1]
    attn = torch.tensor([[0, 1, 1, 1, 1]])
    mask = Experience.build_response_mask(attn, prompt_width=2)
    assert mask.tolist() == [[False, False, True, True, True]]


def test_post_eos_tokens_masked():
    # [pad][prompt][resp][EOS][post-eos][post-eos]: attention is 0 after
    # the first EOS; the EOS token itself still counts as a response
    attn = torch.tensor([[0, 1, 1, 1, 0, 0]])
    mask = Experience.build_response_mask(attn, prompt_width=2)
    assert mask.tolist() == [[False, False, True, True, False, False]]


def test_batch_mixed_padding():
    attn = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
    mask = Experience.build_response_mask(attn, prompt_width=2)
    assert mask.tolist() == [
        [False, False, True, True],
        [False, False, True, True],
    ]


def test_all_prompt_no_response():
    # degenerate: response region empty (prompt_width == T)
    attn = torch.tensor([[1, 1, 1]])
    mask = Experience.build_response_mask(attn, prompt_width=3)
    assert mask.tolist() == [[False, False, False]]
