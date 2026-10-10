"""Unit tests for ExperienceBuffer (TODO 4 / 8b).

Spec coverage, one question per test:

1. add + get returns the same objects oldest-first      -> test_add_get_returns_oldest_first
2. get() drains: len == 0 and a second get() == []     -> test_get_drains
3. FIFO eviction: max_size=2, add three -> last two    -> test_fifo_eviction
4. len() tracks the number of buffered experiences     -> test_len_tracks
5. identity is preserved (no copying through add/get)  -> test_identity_preserved

Every test builds a fresh buffer — tests never share state.
"""

import torch

from tinyr.experience import Experience, ExperienceBuffer


def make_experience(reward: float = 0.0) -> Experience:
    """Minimal valid CPU Experience (N=2, W=3, R=4, T=7).

    Fresh object per call — pass different `reward` values to tell
    instances apart when asserting FIFO order, and `is` for identity.
    """
    return Experience(
        prompt_ids=torch.zeros(2, 3, dtype=torch.long),
        response_ids=torch.zeros(2, 4, dtype=torch.long),
        attention_mask=torch.ones(2, 7, dtype=torch.long),
        response_mask=torch.ones(2, 7, dtype=torch.bool),
        rewards=torch.full((2,), reward, dtype=torch.float32),
    )


def test_add_get_returns_oldest_first():
    # integer-valued rewards: exactly representable in float32, so the
    # .item() comparison can't be tripped by float precision
    buffer = ExperienceBuffer()
    buffer.add(make_experience(1.0))
    buffer.add(make_experience(2.0))
    buffer.add(make_experience(3.0))

    got = buffer.get()

    assert [e.rewards[0].item() for e in got] == [1.0, 2.0, 3.0]


def test_get_drains():
    buffer = ExperienceBuffer()
    buffer.add(make_experience())

    buffer.get()

    assert len(buffer) == 0
    assert buffer.get() == []


def test_fifo_eviction():
    buffer = ExperienceBuffer(max_size=2)
    buffer.add(make_experience(1.0))
    buffer.add(make_experience(2.0))
    buffer.add(make_experience(3.0))  # 1.0 silently evicted

    got = buffer.get()

    assert [e.rewards[0].item() for e in got] == [2.0, 3.0]


def test_len_tracks():
    buffer = ExperienceBuffer()
    assert len(buffer) == 0

    buffer.add(make_experience())
    assert len(buffer) == 1

    buffer.add(make_experience())
    assert len(buffer) == 2

    buffer.get()
    assert len(buffer) == 0


def test_identity_preserved():
    buffer = ExperienceBuffer()
    exp = make_experience()
    buffer.add(exp)

    (got,) = buffer.get()

    assert got is exp
