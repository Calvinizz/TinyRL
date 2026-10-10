"""Framework tests for iter_micro_batches + Experience.rows (used by P3.3)."""

import torch

from tinyr.experience import Experience, iter_micro_batches

from tests.test_experience import ids, am, rm, rew


def make_experience(n: int = 4) -> Experience:
    return Experience(
        prompt_ids=ids(n, 3), response_ids=ids(n, 4),
        attention_mask=am(n, 7), response_mask=rm(n, 7), rewards=rew(n),
    )


def test_zero_or_none_means_one_big_batch():
    assert list(iter_micro_batches(4, 0)) == [[0, 1, 2, 3]]
    assert list(iter_micro_batches(4, 99)) == [[0, 1, 2, 3]]


def test_even_chunking():
    assert list(iter_micro_batches(4, 2)) == [[0, 1], [2, 3]]


def test_last_chunk_may_be_short():
    assert list(iter_micro_batches(5, 2)) == [[0, 1], [2, 3], [4]]


def test_rows_keeps_widths_and_slices_all_fields():
    exp = make_experience(4)
    exp.advantages = torch.tensor([1.0, 2.0, 3.0, 4.0])
    exp.old_logprobs = torch.zeros(4, 6)
    exp.ref_logprobs = torch.zeros(4, 6)

    sub = exp.rows([1, 3])

    assert sub.prompt_ids.shape == (2, 3)
    assert sub.attention_mask.shape == (2, 7)
    assert sub.old_logprobs.shape == (2, 6)
    assert sub.advantages.tolist() == [2.0, 4.0]
    assert sub.input_ids.shape == (2, 7)
