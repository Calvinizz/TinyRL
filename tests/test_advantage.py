"""Spec tests for GRPOTrainer.compute_group_advantages (TODO 5)."""

import torch

from tinyr.trainer import GRPOTrainer


def test_two_element_group():
    # mean=0.5, std_pop=0.5 -> [(0.5/0.5), (-0.5/0.5)] ~ [+1, -1]
    rewards = torch.tensor([1.0, 0.0])
    adv = GRPOTrainer.compute_group_advantages(rewards, group_size=2, eps=1e-4)
    assert adv.shape == (2,)
    assert torch.allclose(adv, torch.tensor([1.0, -1.0]), atol=1e-3)


def test_group_of_one_advantage_zero():
    # std=0, mean=r -> 0/(0+eps) = 0
    rewards = torch.tensor([1.0])
    adv = GRPOTrainer.compute_group_advantages(rewards, group_size=1, eps=1e-4)
    assert torch.allclose(adv, torch.zeros(1), atol=1e-6)


def test_uniform_group_is_zero():
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
    adv = GRPOTrainer.compute_group_advantages(rewards, group_size=2, eps=1e-4)
    assert torch.allclose(adv, torch.zeros(4), atol=1e-6)


def test_groups_are_independent():
    # group 0 has signal, group 1 is uniform -> only group 0 nonzero
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])
    adv = GRPOTrainer.compute_group_advantages(rewards, group_size=2, eps=1e-4)
    assert torch.allclose(adv[:2], torch.tensor([1.0, -1.0]), atol=1e-3)
    assert torch.allclose(adv[2:], torch.zeros(2), atol=1e-6)


def test_advantage_is_group_mean_zero():
    # within each group, advantages sum to (approximately) zero
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    adv = GRPOTrainer.compute_group_advantages(rewards, group_size=3, eps=1e-4)
    for g in range(2):
        assert abs(adv[g * 3 : (g + 1) * 3].sum().item()) < 1e-6


def test_returns_float_tensor():
    rewards = torch.tensor([1.0, 0.0])
    adv = GRPOTrainer.compute_group_advantages(rewards, group_size=2, eps=1e-4)
    assert torch.is_floating_point(adv)
