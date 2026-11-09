"""LIBERO and frozen pi0.5 adapter, derived from the formal ExRL-Q runner."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np


TASK_DESCRIPTION = "pick up the cream cheese and put it in the tray"
TASK_NAME = "LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray"
OPENPI_CONFIG_NAME = "pi05_libero_task57_q50_full"
OPENPI_MODEL_TYPE = "pi05"
OPENPI_ACTION_DIM = 32
OPENPI_ACTION_HORIZON = 50
OPENPI_MAX_TOKEN_LEN = 200
OPENPI_DISCRETE_STATE_INPUT = False
OPENPI_EXTRA_DELTA_TRANSFORM = False
OPENPI_POLICY_NORMALIZATION = "quantile_q01_q99"
RL_OBSERVATION_NORMALIZATION = "mean_std_clip10"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_runtime(root, config_dir):
    from exrl.libero.runtime import configure_runtime as configure

    return configure(root, config_dir)


def make_env(seed):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = benchmark.get_benchmark_dict()["libero_90"]().get_task(57)
    if task.language != TASK_DESCRIPTION:
        raise ValueError("task57 language changed")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    # Exactly the pi0/pi0.5 (not FAST) formal constructor. The collector enforces
    # the 400 primitive-step timeout; the native wrapper horizon is 1000.
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(seed)
    return env


def load_policy(checkpoint_dir, **kwargs):
    from exrl.libero.policies import connect_policy

    return connect_policy("pi05", checkpoint_dir, **kwargs)


def variant(norm_stats):
    extra_delta_transform = bool(norm_stats["extra_delta_transform"])
    if extra_delta_transform != OPENPI_EXTRA_DELTA_TRANSFORM:
        raise ValueError("pi0.5 RL conditioning contract changed")
    return SimpleNamespace(
        env="libero",
        task_suite="libero_90",
        task_id=57,
        task_description=TASK_DESCRIPTION,
        query_freq=50,
        max_chunk_len=50,
        max_timesteps=250,
        resize_image=64,
        num_cameras=1,
        add_states=1,
        action_history_len=0,
        condition_on_actions=1,
        discount=0.997,
        reward_offset=1.0,
        env_max_reward=1,
        max_resample_streak=20,
        resample_penalty=-5.0,
        _episode_env_steps=0,
        aac_action_conditioning_extra_delta_transform=extra_delta_transform,
        _aac_obs_norm_stats=norm_stats,
    )


class Pi05Planner:
    def __init__(self, policy, settings):
        self.policy, self.settings = policy, settings
        self.calls = 0

    def plan(self, raw_obs):
        from baselines.resfit.libero.pi05.reference import obs_to_pi_zero_input

        output = self.policy.infer(obs_to_pi_zero_input(raw_obs, self.settings))
        actions = np.asarray(output["actions"], np.float32)
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise ValueError(f"expected finite pi0.5 env actions [50,7], got {actions.shape}")
        self.calls += 1
        return actions.copy()

    def plan_many(self, observations, *, padded_batch_size):
        return [self.plan(observation) for observation in observations]


class LiberoResidualEnv:
    def __init__(self, env, planner, coordinates, norm_stats, *, bootstrap_timeouts=True):
        self.env, self.planner, self.coordinates = env, planner, coordinates
        self.settings = variant(norm_stats)
        self.bootstrap_timeouts = bootstrap_timeouts
        self.raw_obs = self.obs = self.base_raw = None

    def encode_observation(self, raw_obs, base_raw):
        from baselines.resfit.libero.pi05.reference import build_aac_observation, obs_to_qpos

        obs = build_aac_observation(raw_obs, base_raw, [], self.settings)
        return {
            "pixels": np.ascontiguousarray(obs["pixels"][..., 0]),
            "state": np.asarray(obs["state"][:, 0], np.float32),
            "raw_state": np.asarray(obs_to_qpos(raw_obs, self.settings), np.float32),
            "base_conditioning": np.asarray(obs["actions"], np.float32).reshape(-1),
        }

    def set_observation(self, raw_obs, base_raw=None):
        self.raw_obs = raw_obs
        self.base_raw = (
            self.planner.plan(raw_obs)
            if base_raw is None
            else np.asarray(base_raw, np.float32).copy()
        )
        if self.base_raw.shape != (50, 7) or not np.isfinite(self.base_raw).all():
            raise ValueError("invalid supplied pi0.5 plan")
        self.obs = self.encode_observation(raw_obs, self.base_raw)
        return self.obs

    def begin_reset(self, init_state=None):
        raw_obs = self.env.reset()
        if init_state is not None:
            # Same replay initialization as convert_libero_task57_to_lerobot.
            raw_obs = self.env.set_init_state(init_state)
        self.settings._episode_env_steps = 0
        self.raw_obs, self.obs = raw_obs, None
        return raw_obs

    def reset(self, init_state=None):
        return self.set_observation(self.begin_reset(init_state))

    @property
    def base_action(self):
        return self.coordinates.encode(self.base_raw).reshape(-1)

    def act_input(self):
        return {k: v[None] for k, v in self.obs.items()}, self.base_action[None]

    def begin_step(self, residual, *, demo_actions=None):
        from baselines.resfit.libero.pi05.reference import (
            _execute_offline_action_chunk,
            env_actions_to_conditioning_actions,
        )

        if self.obs is None:
            raise RuntimeError("reset is required before stepping")
        old_obs, old_base = self.obs, self.base_action.copy()
        base_raw = self.base_raw.copy()
        requested, executed = self.coordinates.combine(
            base_raw, np.asarray(residual).reshape(50, 7)
        )
        requested_len = 50
        if demo_actions is not None:
            demo_actions = np.asarray(demo_actions, np.float32)
            if (
                demo_actions.ndim != 2
                or demo_actions.shape[1] != 7
                or not 1 <= len(demo_actions) <= 50
            ):
                raise ValueError("demonstration chunk must contain 1..50 seven-dimensional actions")
            requested_len = len(demo_actions)
            requested[:requested_len] = demo_actions
            executed = self.coordinates.encode(requested)
        conditioning = env_actions_to_conditioning_actions(self.raw_obs, requested, self.settings)
        result = _execute_offline_action_chunk(
            self.env, self.raw_obs, requested, conditioning, requested_len, [], self.settings, 0
        )
        duration = result["duration_k"]
        if duration <= 0:
            raise RuntimeError("fixed-TA residual decisions must advance the environment")
        terminal = bool(result["success"])
        truncated = bool(result["episode_done"] and not terminal)
        # A short demonstration ending without observed success is a dataset
        # boundary, not an invented success. Bootstrap from its real endpoint.
        if demo_actions is not None and len(demo_actions) < 50 and not terminal:
            truncated = True
        result["episode_done"] = terminal or truncated
        result["max_zero_residual_command_error"] = (
            float(np.max(np.abs(requested - base_raw)))
            if demo_actions is None and np.count_nonzero(residual) == 0
            else None
        )
        result["controller_saturation_fraction"] = float(
            np.mean(np.abs(requested[:duration]) >= 1.0)
        )
        result["base_max_abs"] = float(np.max(np.abs(base_raw)))
        if demo_actions is not None and duration < 50:
            executed[duration:] = self.coordinates.encode(base_raw[duration:])
        return {
            "old_obs": old_obs,
            "old_base": old_base,
            "executed": executed,
            "terminal": terminal,
            "truncated": truncated,
            "result": result,
            "needs_plan": not (terminal or (truncated and not self.bootstrap_timeouts)),
        }

    def finish_step(self, pending, next_base_raw=None):
        from baselines.resfit.libero.pi05.replay import PixelTransition

        old_obs, old_base, executed = (pending[k] for k in ("old_obs", "old_base", "executed"))
        terminal, truncated, result = (pending[k] for k in ("terminal", "truncated", "result"))
        duration = result["duration_k"]
        if terminal or (truncated and not self.bootstrap_timeouts):
            next_obs = {k: np.zeros_like(v) for k, v in old_obs.items()}
            next_base = np.zeros(350, np.float32)
            self.raw_obs = result["next_raw_obs"]
            self.obs = None
        else:
            next_obs = self.set_observation(result["next_raw_obs"], next_base_raw)
            next_base = self.base_action.copy()
        # The unreachable terminal suffix is the frozen plan (zero residual),
        # as in the DP offline builder. Actual duration drives the backup.
        transition = PixelTransition(
            old_obs,
            old_base,
            executed.reshape(-1),
            float(result["reward"]),
            0.0 if terminal else self.settings.discount**duration,
            next_obs,
            next_base,
            terminal,
            truncated,
        )
        return transition, result

    def step(self, residual, *, demo_actions=None):
        return self.finish_step(self.begin_step(residual, demo_actions=demo_actions))

    def close(self):
        self.env.close()


@contextmanager
def isolated_rng(policy, seed=None):
    """Adapted from the formal clean-RNG eval; additionally preserves Torch."""
    import jax
    import torch

    saved = {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "policy": copy.deepcopy(policy._rng),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        policy._rng = jax.random.key(seed)
    try:
        yield
    finally:
        np.random.set_state(saved["numpy"])
        random.setstate(saved["python"])
        torch.set_rng_state(saved["torch"])
        policy._rng = saved["policy"]
        if saved["cuda"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda"])


def runtime_manifest(root, checkpoint):
    return {
        "base_policy": "pi05",
        "action_horizon": 50,
        "versions": {
            k: importlib.metadata.version(k)
            for k in ("torch", "jax", "numpy", "mujoco", "robosuite")
        },
    }
