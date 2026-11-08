"""Runner imports and optional execution-length schedules."""

from importlib import import_module
from types import SimpleNamespace

import pytest


@pytest.mark.integration
@pytest.mark.parametrize("name", ["dp_q", "act_q", "dp_sac", "act_sac", "dp_demo"])
def test_runner_import_and_length_floor(name):
    runner = import_module(f"exrl.runners.{name}.train")
    args = SimpleNamespace(
        post_warmup_min_k_until_policy_steps=100, post_warmup_min_k=5
    )
    assert runner._resolve_min_k_floor(20, args, []) == 5
    assert runner._resolve_min_k_floor(100, args, []) == 1
    assert runner._resolve_min_k_floor(100, args, [(0, 3), (50, 2)]) == 2
