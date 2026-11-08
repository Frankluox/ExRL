"""Numerical checks for the execution-length target and CQL."""

from importlib import import_module
import inspect

import pytest
import torch


@pytest.mark.parametrize("name", ["dp_q", "act_q", "dp_sac", "act_sac", "dp_demo"])
def test_learning_objectives(name):
    core = import_module(f"exrl.algorithms.{name}")
    torch.set_num_threads(1)
    core.run_core_sanity_checks(device=torch.device("cpu"))
    model = core.ExecutionLengthAgent(
        obs_dim=3,
        action_dim=4,
        hidden_dims=(8,),
        device=torch.device("cpu"),
        cql_alpha=0.3,
    )
    q = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0]], [[2.0, 3.0, 4.0, 5.0]]], requires_grad=True
    )
    actions = torch.tensor([2])
    if "batch" in inspect.signature(model._cql_loss).parameters:
        loss, _ = model._cql_loss(q, actions, {})
    else:
        loss, _ = model._cql_loss(q, actions)
    expected = (
        torch.logsumexp(q, dim=-1) - torch.log(torch.tensor(4.0)) - q[:, :, 2]
    ).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(q.grad).all()


@pytest.mark.parametrize("name", ["dp_q", "act_q", "dp_sac", "act_sac", "dp_demo"])
def test_terminal_and_variable_length_discount(name):
    core = import_module(f"exrl.algorithms.{name}")
    result = core.smdp_backup(
        torch.tensor([1.0, 2.0]),
        torch.tensor([0.0, 1.0]),
        0.5,
        torch.tensor([3.0, 5.0]),
        torch.tensor([8.0, 100.0]),
    )
    torch.testing.assert_close(result, torch.tensor([2.0, 2.0]))
