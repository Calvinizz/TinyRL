"""Spec tests for RewardManager — these encode the TODO 1 contract."""

import torch

from tinyr.reward import RewardManager


class TestExtractAnswer:
    def test_hash_format(self):
        assert RewardManager.extract_answer("blah blah\n#### 42") == "42"

    def test_hash_with_spaces(self):
        assert RewardManager.extract_answer("####   17") == "17"

    def test_hash_beats_other_numbers(self):
        # '####' answer wins even when other numbers appear later
        assert (
            RewardManager.extract_answer("x = 5 then y = 6\n#### 7\nextra 9")
            == "7"
        )

    def test_fallback_last_number(self):
        assert RewardManager.extract_answer("24 + 11 = 35") == "35"

    def test_fallback_last_number_in_prose(self):
        assert RewardManager.extract_answer("The sum of 24 and 11 is 35.") == "35"

    def test_fallback_signed_number(self):
        assert RewardManager.extract_answer("temperature dropped to -3") == "-3"

    def test_no_numbers_returns_none(self):
        assert RewardManager.extract_answer("no digits here at all") is None

    def test_empty_returns_none(self):
        assert RewardManager.extract_answer("") is None


class TestCompute:
    def test_exact_match_gets_one(self):
        rewards = RewardManager.compute(
            ["#### 72"], ["72"]
        )
        assert rewards.dtype == torch.float32
        assert rewards.shape == (1,)
        assert rewards[0].item() == 1.0

    def test_mismatch_gets_zero(self):
        rewards = RewardManager.compute(["#### 73"], ["72"])
        assert rewards[0].item() == 0.0

    def test_no_answer_gets_zero(self):
        rewards = RewardManager.compute(["I have no idea"], ["72"])
        assert rewards[0].item() == 0.0

    def test_fallback_can_score(self):
        # response without #### but correct last number still scores
        rewards = RewardManager.compute(
            ["24 + 48 = 72"], ["72"]
        )
        assert rewards[0].item() == 1.0

    def test_string_mismatch_is_zero(self):
        # '072' != '72' — extraction must not normalize numerically
        rewards = RewardManager.compute(["#### 072"], ["72"])
        assert rewards[0].item() == 0.0

    def test_batch_shape(self):
        rewards = RewardManager.compute(
            ["#### 1", "#### 2", "wrong"], ["1", "2", "3"]
        )
        assert rewards.shape == (3,)
        assert rewards.tolist() == [1.0, 1.0, 0.0]
