"""Episode boundaries, demonstration mixing, and resumable BC scheduling."""

from importlib import import_module

import numpy as np
import pytest
import torch

from exrl.runners.act_q.bc_update_schedule import LinearBCUpdateSchedule
from exrl.runners.act_q.act_offline_bc import split_bc_counts


@pytest.mark.parametrize("base", ["dp", "act"])
def test_success_windows_do_not_cross_episodes(base):
    module = import_module(f"exrl.runners.{base}_q.online_bc")
    replay = module.OnlineSuccessBCReplay(
        capacity=32,
        horizon_steps=4,
        action_dim=2,
        cond_steps=1,
        obs_dim=3,
        device=torch.device("cpu"),
        padded=False,
    )
    for tag in (1.0, 2.0):
        replay.add_success_episode(
            np.full((8, 3), tag, np.float32), np.full((8, 2), tag, np.float32)
        )
    batch = replay.sample(8, sampling="uniform_episode")
    torch.testing.assert_close(
        batch.actions[:, 0, 0], batch.conditions["state"][:, 0, 0]
    )
    assert torch.all(batch.actions == batch.actions[:, :1, :1])


def test_demo_batch_has_requested_fraction():
    assert split_bc_counts(128, 0.5) == (64, 64)


def test_offline_windows_and_resume_data_identity(tmp_path):
    from exrl.runners.act_q.act_offline_bc import (
        ACTOfflineDemoPool,
        validate_resume_bc_contract,
    )
    from exrl.runners.dp_demo.demo_replay import _load_demo_npz

    path = tmp_path / "train.npz"
    states = np.repeat(np.array([0.1, 0.2], dtype=np.float32), 8)[:, None]
    actions = states.copy()
    np.savez(path, states=states, actions=actions, traj_lengths=np.array([8, 8]))
    loaded_states, loaded_actions, lengths = _load_demo_npz(str(path))
    np.testing.assert_array_equal(loaded_states, states)
    np.testing.assert_array_equal(loaded_actions, actions)
    np.testing.assert_array_equal(lengths, [8, 8])
    pool = ACTOfflineDemoPool(
        str(path),
        horizon_steps=4,
        obs_dim=1,
        action_dim=1,
        cond_steps=1,
        device="cpu",
        padded=False,
        observation_space_kind="dsrl_normalized",
        action_space_kind="dsrl_normalized",
    )
    batch = pool.sample(32)
    torch.testing.assert_close(
        batch.actions[:, 0, 0], batch.conditions["state"][:, 0, 0]
    )
    assert torch.all(batch.actions == batch.actions[:, :1, :1])
    validate_resume_bc_contract(pool.contract(), pool.contract())
    with pytest.raises(ValueError, match="mismatch"):
        validate_resume_bc_contract(pool.contract(), {**pool.contract(), "frames": 100})


def test_bc_schedule_restores_fractional_budget():
    schedule = LinearBCUpdateSchedule(5, 1, 600000)
    # The schedule starts when BC becomes eligible, not at environment step zero.
    assert schedule.start_step is None
    assert schedule.next_updates(100000) == 5
    assert schedule.next_updates(475000) == 2
    assert schedule.remainder == 0.5
    state = schedule.state_dict()
    restored = LinearBCUpdateSchedule(5, 1, 600000)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored.next_updates(475000) == 3
    assert restored.next_updates(700000) == 1
