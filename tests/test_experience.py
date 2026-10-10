"""TODO 8a: unit tests for Experience.__post_init__ validation (my TODO 3).

Spec coverage, one question per test — every reject-test builds a VALID
Experience first, then breaks exactly ONE field:

1. wrong dtype: response_mask as int64 instead of bool     -> test_rejects_non_bool_response_mask
2. wrong dtype: rewards as float64 instead of float32      -> test_rejects_float64_rewards
3. shape mismatch: attention_mask width != W + R           -> test_rejects_wrong_attention_width
4. shape mismatch: response_mask shorter than attention    -> test_rejects_short_response_mask
5. batch mismatch: rewards has N-1 elements               -> test_rejects_wrong_reward_batch
6. device mismatch: rewards on cpu, ids on 'meta' device   -> test_rejects_device_mismatch
7. happy path: valid Experience + properties are correct   -> test_valid_experience_constructs

Use `with pytest.raises(AssertionError):` for 1-6. The valid baseline is:

    prompt_ids ids(2,3) | response_ids ids(2,4) | attention_mask am(2,7)
    | response_mask rm(2,7) | rewards rew(2)      (N=2, W=3, R=4, T=7)
"""

import pytest
import torch

from tinyr.experience import Experience

# CPU-friendly builders (see module docstring for the valid baseline)
ids = lambda *s: torch.zeros(s, dtype=torch.long)
am = lambda *s: torch.ones(s, dtype=torch.long)
rm = lambda *s: torch.ones(s, dtype=torch.bool)
rew = lambda n: torch.zeros(n, dtype=torch.float32)


def test_rejects_non_bool_response_mask():
    """response_mask given as int64 (e.g. the attention_mask tensor)
    must raise AssertionError naming response_mask."""
    with pytest.raises(AssertionError, match="response_mask"):
        Experience(
            prompt_ids=ids(2,3),
            response_ids=ids(2,4),
            attention_mask=am(2,7),
            response_mask=am(2,7),
            rewards=rew(2),
        )


def test_rejects_float64_rewards():
    """rewards given as float64 must raise AssertionError naming rewards.
    (hint: torch.zeros(n, dtype=torch.float64)"""
    with pytest.raises(AssertionError, match="rewards"):
        Experience(
            prompt_ids=ids(2,3),
            response_ids=ids(2,4),
            attention_mask=am(2,7),
            response_mask=rm(2,7),
            rewards=torch.zeros(2,dtype=torch.float64),
        )


def test_rejects_wrong_attention_width():
    """attention_mask [2, 6] while W+R = 7 must raise AssertionError
    naming attention_mask."""
    # attn mask = 3 + 4 namely prompt+resp
    with pytest.raises(AssertionError, match="attention_mask"):
        Experience(
            prompt_ids=ids(2,3),
            response_ids=ids(2,4),
            attention_mask=am(2,6),
            response_mask=rm(2,7),
            rewards=rew(2),
        )

    

def test_rejects_short_response_mask():
    """response_mask [2, 6] while attention_mask is [2, 7] must raise
    AssertionError naming response_mask."""
    with pytest.raises(AssertionError, match="response_mask"):
        Experience(
            prompt_ids=ids(2,3),
            response_ids=ids(2,4),
            attention_mask=am(2,7),
            response_mask=rm(2,6),
            rewards=rew(2),
        )


def test_rejects_wrong_reward_batch():
    """rewards with 1 element while N = 2 must raise AssertionError
    naming rewards."""
    with pytest.raises(AssertionError, match=r"rewards: expected shape \(2,\), got \(1,\)"):
        Experience(
            prompt_ids=ids(2,3),
            response_ids=ids(2,4),
            attention_mask=am(2,7),
            response_mask=rm(2,7),
            rewards=rew(1),
        )


def test_rejects_device_mismatch():
    """ids on the 'meta' device while rewards stays on cpu must raise
    AssertionError naming rewards. (hint: torch.zeros(2, 3,
    dtype=torch.long, device='meta') — meta tensors are free, no GPU
    needed)"""
    with pytest.raises(AssertionError, match="rewards: expected device meta, got cpu"):
        Experience(
            prompt_ids=torch.zeros(2, 3, dtype=torch.long, device='meta'),
            response_ids=torch.zeros(2, 4, dtype=torch.long, device='meta'),
            attention_mask=torch.zeros(2, 7, dtype=torch.long, device='meta'),
            response_mask=torch.zeros(2, 7, dtype=torch.bool, device='meta'),
            rewards=rew(2),
        )


def test_valid_experience_constructs():
    """The valid baseline constructs without raising; input_ids is
    [2, 7] (= W + R), n_sequences == 2, n_response_tokens == 14
    (rm(2,7) is all-True)."""
    exp = Experience(
        prompt_ids=ids(2, 3),
        response_ids=ids(2, 4),
        attention_mask=am(2, 7),
        response_mask=rm(2, 7),
        rewards=rew(2),
    )

    assert exp.input_ids.shape == (2, 7)
    assert exp.n_sequences == 2
    assert exp.n_response_tokens == 14
