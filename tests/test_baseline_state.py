"""Baseline state, task interfaces, and learning-target checks."""

import importlib
import random

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("method", ["dice", "resfit"])
@pytest.mark.parametrize("base", ["dp", "act", "adroit"])
def test_evaluation_rng_preservation(method, base):
    module = importlib.import_module(f"baselines.{method}.{base}.train")
    module.set_global_seed(17)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    module.set_global_seed(17)
    with module.preserved_rng_state():
        random.random()
        np.random.rand(5)
        torch.rand(9)
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_distributional_critic_targets_and_gradients():
    from model.common.critic import DistributionalCriticObsAct

    model = DistributionalCriticObsAct(
        cond_dim=3,
        action_dim=2,
        action_steps=4,
        mlp_dims=[16, 16],
        critic_ensemble_size=2,
        q_min=-100,
        q_max=0,
        num_bins=21,
    )
    targets = torch.tensor([[-200.0], [-5.0], [0.0]])
    probabilities = model.scalar_target_to_probs(targets)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(3))
    logits = torch.randn(2, 3, 21, requires_grad=True)
    loss = model.cross_entropy_loss_on_scalar(logits, targets)
    loss.mean().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("task", ["can", "lift", "square", "tool_hang", "transport"])
def test_act_parl_observation_contract(task):
    from exrl.paths import ROOT
    import yaml

    config = yaml.safe_load((ROOT / f"configs/baselines/native/{task}_act_parl.yaml").read_text())
    model = config["model"]
    dimensions = model["lowdim_dims"]
    assert sum(dimensions[key] for key in model["lowdim_keys"]) == config["obs_dim"]
    assert model["state_obs_keys"] + model["env_obs_keys"] == model["lowdim_keys"]
    assert len(model["state_obs_keys"]) == (6 if task == "transport" else 3)
