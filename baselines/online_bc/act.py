"""Pure ACT online-success BC with a fixed full-chunk execution policy.

This entry intentionally has no RL algorithm, critic, selector, or RL replay.
It reuses the ACT AAC environment adapter so observations, action conversion,
success collection, termination semantics, and ACT inference match the ACT
ExRL-Q + Online-Teacher-BC runs.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import json
import os
import random
import socket
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecMonitor
from torch.utils.tensorboard import SummaryWriter

CURRENT_DIR = str(__import__("exrl.paths", fromlist=["ROOT"]).ROOT)
REPO_ROOT = str(__import__("exrl.paths", fromlist=["ROOT"]).ROOT)
RUNTIME_OVERLAY = os.environ.get("AAC_RUNTIME_OVERLAY", "").strip()

import robomimic.utils.torch_utils as TorchUtils
import exrl.runners.act_q.train as act_train
from exrl.runners.act_q.policy_loader import (
    infer_lowdim_dims_from_stats,
    load_act_policy_wrapper,
    load_env_meta_json,
    parse_lowdim_keys,
)
from exrl.runners.act_q.environment import (
    ExecutionLengthEnv,
)
from exrl.runners.act_q.online_bc import (
    OnlineSuccessBCReplay,
    ema_update_module_,
    select_act_bc_trainable_parameters,
)


PIPELINE_NAME = "act_online_success_bc_fixed_exec_no_rl"
CHECKPOINT_FORMAT_VERSION = 1


def _collect_success_infos(infos, replay: OnlineSuccessBCReplay) -> tuple[int, int]:
    episodes = 0
    windows = 0
    for info in infos or []:
        if not isinstance(info, dict):
            continue
        states = info.get("bc_success_episode_states")
        actions = info.get("bc_success_episode_actions")
        if states is None or actions is None:
            continue
        added = replay.add_success_episode(states, actions)
        if added > 0:
            episodes += 1
            windows += int(added)
    return int(episodes), int(windows)


def _build_task_venv(
    *,
    args,
    policy,
    device,
    lowdim_keys,
    lowdim_dims,
    env_meta,
    n_envs: int,
    seed: int,
    collect_success_episodes: bool,
):
    venv = ExecutionLengthEnv(
        venv=None,
        policy=policy,
        max_chunk_len=args.max_chunk_len,
        device=device,
        reward_offset=args.reward_offset,
        resample_penalty=args.resample_penalty,
        action_history_len=args.action_history_len,
        dp_infer_steps=args.dp_infer_steps,
        dp_sampler=args.dp_sampler,
        dp_amp=args.dp_amp,
        dp_compile=args.dp_compile,
        ipc_chunk_dtype=args.ipc_chunk_dtype,
        dp_infer_gpus=args.dp_infer_gpus,
        dp_grouping=args.dp_grouping,
        dp_infer_parallel=args.dp_infer_parallel,
        ipc_send_mode=args.ipc_send_mode,
        perf_profile=False,
        vendor_root=args.dsrl_official_root,
        env_name=args.env_name,
        env_meta_json=args.env_meta_json,
        normalization_path=args.dsrl_normalization_npz,
        stats_npz_path=args.stats_npz_override,
        lowdim_keys=lowdim_keys,
        lowdim_dims=lowdim_dims,
        n_envs=int(n_envs),
        max_episode_steps=int(
            env_meta.get("env_kwargs", {}).get("horizon", args.env_horizon_override)
        ),
        gamma=args.gamma,
        seed=int(seed),
        min_chunk_len=int(args.min_chunk_len),
        zero_action_fail_streak=int(args.zero_action_fail_streak),
        zero_action_fail_reward=float(args.zero_action_fail_reward),
        terminate_on_success=bool(args.terminate_on_success),
        timeout_on_horizon=bool(args.timeout_on_horizon),
        td_target_next_obs_samples=int(args.td_target_next_obs_samples),
        critic_obs_unnormalize_dp_chunk=bool(args.critic_obs_unnormalize_dp_chunk),
        critic_obs_action_mode=str(args.critic_obs_action_mode),
        critic_obs_state_mode=str(args.critic_obs_state_mode),
        collect_success_episodes=bool(collect_success_episodes),
    )
    return VecMonitor(venv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pure ACT online-success BC baseline with fixed k and no RL components."
    )
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument(
        "--task",
        choices=["lift", "can", "square", "transport", "tool_hang"],
        required=True,
    )
    parser.add_argument("--env_name", type=str, default="")
    parser.add_argument("--dsrl_official_root", type=str, required=True)
    parser.add_argument("--env_meta_json", type=str, default=None)
    parser.add_argument("--dsrl_normalization_npz", type=str, default=None)
    parser.add_argument("--stats_npz_override", type=str, default=None)
    parser.add_argument("--lowdim_keys", type=str, default="")
    parser.add_argument("--lerobot_src", type=str, default="")

    parser.add_argument("--output", type=str, default="./adaptive_rl_results")
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume_ckpt_name", type=str, default="latest_model.pt")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_chunk_len", type=int, required=True)
    parser.add_argument(
        "--fixed_exec_len",
        type=int,
        default=0,
        help="Requested k for every rollout and eval decision; 0 means max_chunk_len.",
    )
    parser.add_argument("--min_chunk_len", type=int, default=1)
    parser.add_argument("--env_horizon_override", type=int, required=True)
    parser.add_argument("--n_envs", type=int, default=48)
    parser.add_argument("--total_policy_steps", type=int, default=100_000_000)
    parser.add_argument(
        "--stop_after_env_steps",
        type=int,
        default=0,
        help="Stop after this many actual primitive env steps; 0 disables the env-step cap.",
    )
    parser.add_argument("--log_every_policy_steps", type=int, default=2048)
    parser.add_argument("--window_size", type=int, default=200)

    parser.add_argument("--offline_eval_episodes", type=int, default=50)
    parser.add_argument("--offline_eval_n_envs", type=int, default=4)
    parser.add_argument("--offline_eval_max_decisions", type=int, default=100_000)
    parser.add_argument("--offline_eval_seed_offset", type=int, default=10_000)
    parser.add_argument("--offline_env_eval_interval_steps", type=int, required=True)
    eval_start = parser.add_mutually_exclusive_group()
    eval_start.add_argument(
        "--offline_env_eval_at_start", dest="offline_env_eval_at_start", action="store_true"
    )
    eval_start.add_argument(
        "--no_offline_env_eval_at_start", dest="offline_env_eval_at_start", action="store_false"
    )
    parser.set_defaults(offline_env_eval_at_start=True)
    parser.add_argument(
        "--async_eval_snapshots",
        action="store_true",
        help=(
            "Write immutable policy-only snapshots at the env-step eval cadence and "
            "skip all in-process evaluation."
        ),
    )
    parser.add_argument(
        "--async_eval_snapshot_dir",
        type=str,
        default="",
        help="Snapshot directory; empty uses RUN_DIR/eval_snapshots.",
    )

    parser.add_argument(
        "--offline_resume_save_interval_steps",
        type=int,
        default=500_000,
        help="Overwrite latest_model.pt at this train-env-step cadence.",
    )
    save_best = parser.add_mutually_exclusive_group()
    save_best.add_argument("--save_best_model", dest="save_best_model", action="store_true")
    save_best.add_argument("--no_save_best_model", dest="save_best_model", action="store_false")
    parser.set_defaults(save_best_model=True)
    parser.add_argument("--disable_checkpoints", action="store_true")

    parser.add_argument("--bc_finetune_mode", choices=["online_success"], default="online_success")
    parser.add_argument("--bc_updates_per_rl_update", type=int, default=1)
    parser.add_argument(
        "--bc_update_every_rl_updates",
        type=int,
        default=32,
        help="Cadence in virtual RL-update ticks, retained for exact +RL density matching.",
    )
    parser.add_argument("--bc_update_every_rl_updates_schedule", type=str, default="")
    parser.add_argument("--bc_schedule_axis", choices=["policy", "env"], default="env")
    parser.add_argument("--bc_start_after_steps", type=int, default=0)
    parser.add_argument("--bc_batch_size", type=int, default=128)
    parser.add_argument("--bc_lr", type=float, default=1e-5)
    parser.add_argument("--bc_lr_schedule", type=str, default="")
    parser.add_argument(
        "--bc_act_trainable_scope",
        choices=["all", "decoder_head", "action_head"],
        default="all",
    )
    parser.add_argument("--bc_grad_clip_norm", type=float, default=10.0)
    parser.add_argument("--bc_success_capacity", type=int, default=25_000)
    parser.add_argument("--bc_online_sampling", choices=["window", "episode"], default="episode")
    parser.add_argument(
        "--bc_tail_padding", choices=["act_native", "unpadded"], default="act_native"
    )
    parser.add_argument("--bc_teacher_constraint_weight", type=float, default=1.0)
    parser.add_argument("--bc_teacher_ema_decay", type=float, default=1.0)
    parser.add_argument(
        "--virtual_rl_updates_per_collector_step",
        type=int,
        default=16,
        help="Virtual UTD used only to reproduce the BC update density of the +RL run.",
    )

    # These are execution-wrapper settings shared with the ACT ExRL-Q path.
    parser.add_argument("--cond_steps", type=int, default=1)
    parser.add_argument("--action_history_len", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--reward_offset", type=float, default=1.0)
    parser.add_argument("--resample_penalty", type=float, default=-5.0)
    parser.add_argument("--zero_action_fail_streak", type=int, default=0)
    parser.add_argument("--zero_action_fail_reward", type=float, default=-1000.0)
    terminate = parser.add_mutually_exclusive_group()
    terminate.add_argument(
        "--terminate_on_success", dest="terminate_on_success", action="store_true"
    )
    terminate.add_argument(
        "--no_terminate_on_success", dest="terminate_on_success", action="store_false"
    )
    parser.set_defaults(terminate_on_success=True)
    parser.add_argument("--timeout_on_horizon", action="store_true")
    parser.add_argument("--td_target_next_obs_samples", type=int, default=1)
    parser.add_argument("--critic_obs_unnormalize_dp_chunk", action="store_true")
    parser.add_argument(
        "--critic_obs_action_mode",
        choices=["stats", "dsrl_unnorm_stats", "dp_raw"],
        default="dp_raw",
    )
    parser.add_argument(
        "--critic_obs_state_mode",
        choices=["stats_clip", "stats_no_clip", "dsrl_raw"],
        default="stats_no_clip",
    )

    # Diffusion-named knobs are accepted solely because the shared ACT wrapper
    # has the legacy AAC constructor signature. They do not change ACT inference.
    parser.add_argument("--decision_runtime", choices=["sync"], default="sync")
    parser.add_argument("--dp_infer_steps", type=int, default=5)
    parser.add_argument("--dp_sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--dp_amp", action="store_true")
    parser.add_argument("--dp_compile", action="store_true")
    parser.add_argument("--dp_infer_parallel", action="store_true")
    parser.add_argument("--dp_infer_gpus", type=str, default="0")
    parser.add_argument(
        "--dp_grouping", choices=["round_robin", "contiguous"], default="round_robin"
    )
    parser.add_argument("--ipc_chunk_dtype", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument(
        "--ipc_send_mode", choices=["full", "executed_only"], default="executed_only"
    )
    parser.add_argument("--infer_microbatch_max", type=int, default=64)
    parser.add_argument("--infer_microbatch_timeout_ms", type=int, default=3)
    parser.add_argument("--infer_queue_size", type=int, default=4096)
    parser.add_argument("--infer_result_queue_size", type=int, default=4096)
    parser.add_argument("--env_poll_timeout_ms", type=int, default=5)
    parser.add_argument("--ready_queue_size", type=int, default=8192)
    parser.add_argument("--max_inflight_per_env", type=int, default=1)
    parser.add_argument("--async_rollout_mode", type=str, default="strict_on_policy")
    parser.add_argument("--force_thread_limits", action="store_true")
    parser.add_argument("--expected_mujoco_version", type=str, default="3.3.5")
    parser.add_argument("--expected_robosuite_version", type=str, default="1.4.1")
    parser.add_argument(
        "--expected_robosuite_root",
        type=str,
        default="",
        help="If set, require the imported robosuite module to reside under this directory.",
    )

    parser.add_argument(
        "--memory_floor_gib",
        type=float,
        default=0.0,
        help="Pause collection before a step whenever Linux MemAvailable is below this hard floor.",
    )
    parser.add_argument("--memory_resume_gib", type=float, default=2.0)
    parser.add_argument("--memory_poll_seconds", type=float, default=30.0)
    return parser


def _available_memory_gib() -> Optional[float]:
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        return None
    return None


def _wait_for_memory(args, *, context: str) -> None:
    available = _available_memory_gib()
    if available is None or available >= float(args.memory_floor_gib):
        return
    print(
        f"[MEMORY][PAUSE] context={context} MemAvailable={available:.2f} GiB "
        f"floor={float(args.memory_floor_gib):.2f} GiB",
        flush=True,
    )
    while available is not None and available < float(args.memory_resume_gib):
        time.sleep(max(float(args.memory_poll_seconds), 1.0))
        available = _available_memory_gib()
    if available is not None:
        print(f"[MEMORY][RESUME] MemAvailable={available:.2f} GiB", flush=True)


def _fixed_action_id(*, fixed_exec_len: int, min_chunk_len: int, action_dim: int) -> int:
    action_id = int(fixed_exec_len) - int(min_chunk_len)
    if not (0 <= action_id < int(action_dim)):
        raise ValueError(
            f"fixed k={fixed_exec_len} maps to invalid action_id={action_id}; "
            f"min_k={min_chunk_len} action_dim={action_dim}"
        )
    return action_id


def _run_fixed_eval(
    *,
    venv,
    action_id: int,
    fixed_exec_len: int,
    episodes: int,
    max_decisions: int,
) -> Dict[str, float]:
    venv.reset()
    successes: List[int] = []
    returns = np.zeros((int(venv.num_envs),), dtype=np.float64)
    episode_returns: List[float] = []
    exec_lens: List[float] = []
    eval_env_steps = 0
    decisions = 0
    actions = np.full((int(venv.num_envs),), int(action_id), dtype=np.int64)
    while len(successes) < int(episodes) and decisions < int(max_decisions):
        _, rewards, dones, infos = venv.step(actions)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        dones = np.asarray(dones, dtype=bool).reshape(-1)
        for i in range(int(venv.num_envs)):
            info = infos[i] if i < len(infos) else {}
            actual_k = int(info.get("exec_len", 0))
            if actual_k < 0 or actual_k > int(fixed_exec_len):
                raise RuntimeError(
                    f"eval returned invalid exec_len={actual_k} for requested k={fixed_exec_len}"
                )
            exec_lens.append(float(actual_k))
            eval_env_steps += actual_k
            returns[i] += float(rewards[i])
            if bool(dones[i]):
                successes.append(1 if act_train._extract_success(info) else 0)
                episode_returns.append(float(returns[i]))
                returns[i] = 0.0
                if len(successes) >= int(episodes):
                    break
        decisions += int(venv.num_envs)
    if len(successes) < int(episodes):
        raise RuntimeError(
            f"fixed-k eval completed only {len(successes)}/{episodes} episodes "
            f"within max_decisions={max_decisions}"
        )
    selected = successes[: int(episodes)]
    return {
        "success_rate": float(np.mean(selected)),
        "success_rate_recent20": float(np.mean(selected[-20:])),
        "avg_exec_len": float(np.mean(exec_lens)) if exec_lens else 0.0,
        "episode_return_mean": float(np.mean(episode_returns[: int(episodes)])),
        "episodes": float(len(selected)),
        "eval_env_steps": float(eval_env_steps),
        "decisions": float(decisions),
    }


def _create_run_dir(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    base = f"BC_{datetime.now().strftime('%m%d_%H%M%S')}"
    for index in range(100):
        name = base if index == 0 else f"{base}_{index:02d}"
        candidate = os.path.join(log_dir, name)
        try:
            os.makedirs(candidate, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"failed to create a unique run directory under {log_dir}")


def _validate_args(args) -> None:
    if int(args.min_chunk_len) != 1:
        raise ValueError("pure fixed-exec ACT BC requires --min_chunk_len 1; k=0 is not exposed")
    if int(args.fixed_exec_len) <= 0:
        args.fixed_exec_len = int(args.max_chunk_len)
    if int(args.fixed_exec_len) != int(args.max_chunk_len):
        raise ValueError(
            "TA32/TA50 pure-BC baseline must execute the full checkpoint chunk: "
            "--fixed_exec_len must equal --max_chunk_len"
        )
    if int(args.n_envs) <= 0 or int(args.offline_eval_n_envs) <= 0:
        raise ValueError("collector and eval env counts must be positive")
    if int(args.total_policy_steps) <= 0:
        raise ValueError("--total_policy_steps must be positive")
    if int(args.stop_after_env_steps) < 0:
        raise ValueError("--stop_after_env_steps must be non-negative")
    if int(args.offline_eval_episodes) <= 0:
        raise ValueError("--offline_eval_episodes must be positive")
    if int(args.offline_env_eval_interval_steps) <= 0:
        raise ValueError("--offline_env_eval_interval_steps must be positive")
    if not (0 < int(args.offline_resume_save_interval_steps) <= 500_000):
        raise ValueError("latest checkpoint env-step interval must be in [1, 500000]")
    if int(args.virtual_rl_updates_per_collector_step) <= 0:
        raise ValueError("--virtual_rl_updates_per_collector_step must be positive")
    if int(args.bc_updates_per_rl_update) <= 0:
        raise ValueError("--bc_updates_per_rl_update must be positive")
    if int(args.bc_update_every_rl_updates) <= 0:
        raise ValueError("--bc_update_every_rl_updates must be positive")
    if float(args.bc_lr) <= 0.0 or int(args.bc_batch_size) <= 0:
        raise ValueError("BC learning rate and batch size must be positive")
    if float(args.bc_grad_clip_norm) < 0.0:
        raise ValueError("--bc_grad_clip_norm must be non-negative")
    if float(args.memory_floor_gib) < 0.0:
        raise ValueError("--memory_floor_gib must be non-negative")
    if float(args.memory_resume_gib) < float(args.memory_floor_gib):
        raise ValueError("--memory_resume_gib must be >= --memory_floor_gib")
    if bool(args.dp_compile):
        raise ValueError("online ACT BC does not support --dp_compile")
    if int(args.td_target_next_obs_samples) != 1:
        raise ValueError("pure BC requires --td_target_next_obs_samples 1")
    if not str(args.expected_mujoco_version).strip():
        raise ValueError("--expected_mujoco_version cannot be empty")
    if not str(args.expected_robosuite_version).strip():
        raise ValueError("--expected_robosuite_version cannot be empty")
    if int(args.bc_success_capacity) <= 0:
        raise ValueError("--bc_success_capacity must be positive")
    if int(args.bc_start_after_steps) < 0:
        raise ValueError("--bc_start_after_steps must be non-negative")
    if not (0.0 <= float(args.bc_teacher_ema_decay) <= 1.0):
        raise ValueError("--bc_teacher_ema_decay must be in [0, 1]")
    if float(args.bc_teacher_constraint_weight) < 0.0:
        raise ValueError("--bc_teacher_constraint_weight must be non-negative")


def _invariants(args, ckpt_meta: Dict) -> Dict:
    return {
        "pipeline": PIPELINE_NAME,
        "task": str(args.task),
        "seed": int(args.seed),
        "max_chunk_len": int(args.max_chunk_len),
        "fixed_exec_len": int(args.fixed_exec_len),
        "min_chunk_len": int(args.min_chunk_len),
        "env_horizon_override": int(args.env_horizon_override),
        "n_envs": int(args.n_envs),
        "bc_batch_size": int(args.bc_batch_size),
        "bc_lr": float(args.bc_lr),
        "bc_lr_schedule": str(args.bc_lr_schedule),
        "bc_grad_clip_norm": float(args.bc_grad_clip_norm),
        "bc_success_capacity": int(args.bc_success_capacity),
        "bc_online_sampling": str(args.bc_online_sampling),
        "bc_tail_padding": str(args.bc_tail_padding),
        "bc_act_trainable_scope": str(args.bc_act_trainable_scope),
        "bc_teacher_constraint_weight": float(args.bc_teacher_constraint_weight),
        "bc_teacher_ema_decay": float(args.bc_teacher_ema_decay),
        "bc_updates_per_rl_update": int(args.bc_updates_per_rl_update),
        "bc_update_every_rl_updates_schedule": str(args.bc_update_every_rl_updates_schedule),
        "bc_schedule_axis": str(args.bc_schedule_axis),
        "bc_start_after_steps": int(args.bc_start_after_steps),
        "virtual_rl_updates_per_collector_step": int(args.virtual_rl_updates_per_collector_step),
        "bc_update_every_rl_updates": int(args.bc_update_every_rl_updates),
        "act_chunk_size": int(ckpt_meta["chunk_size"]),
        "agent": os.path.abspath(str(args.agent)),
        "gamma": float(args.gamma),
        "terminate_on_success": bool(args.terminate_on_success),
        "timeout_on_horizon": bool(args.timeout_on_horizon),
        "critic_obs_action_mode": str(args.critic_obs_action_mode),
        "critic_obs_state_mode": str(args.critic_obs_state_mode),
        "expected_mujoco_version": str(args.expected_mujoco_version),
        "expected_robosuite_version": str(args.expected_robosuite_version),
        "expected_robosuite_root": os.path.realpath(str(args.expected_robosuite_root))
        if str(args.expected_robosuite_root).strip()
        else "",
    }


def _assert_resume_invariants(saved: Dict, current: Dict) -> None:
    if not saved:
        raise ValueError("pure-BC checkpoint is missing invariants")
    mismatches = []
    for key, value in current.items():
        old = saved.get(key, "<missing>")
        if isinstance(value, float):
            equal = isinstance(old, (float, int)) and np.isclose(
                float(old), value, rtol=0.0, atol=1e-12
            )
        else:
            equal = old == value
        if not equal:
            mismatches.append(f"{key}: checkpoint={old!r} current={value!r}")
    if mismatches:
        raise ValueError("resume invariant mismatch: " + "; ".join(mismatches))


def _restore_rng_state_strict(state: Optional[Dict]) -> None:
    """Restore all RNGs without silently accepting a partial resume."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        cpu_state = torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu").contiguous()
        torch.set_rng_state(cpu_state)
    cuda_states = state.get("torch_cuda")
    if torch.cuda.is_available() and cuda_states is not None:
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "resume CUDA RNG count mismatch: "
                f"checkpoint={len(cuda_states)} visible={torch.cuda.device_count()}"
            )
        cpu_cuda_states = [
            torch.as_tensor(item, dtype=torch.uint8, device="cpu").contiguous()
            for item in cuda_states
        ]
        torch.cuda.set_rng_state_all(cpu_cuda_states)


def train(args) -> str:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    _validate_args(args)
    act_train._set_runtime_env(args)
    if str(args.lerobot_src).strip():
        lerobot_src = os.path.abspath(str(args.lerobot_src))
        if lerobot_src not in sys.path:
            sys.path.insert(0, lerobot_src)

    for path_name in (
        "agent",
        "env_meta_json",
        "dsrl_normalization_npz",
        "stats_npz_override",
    ):
        path = str(getattr(args, path_name))
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path_name} not found: {path}")

    run_dir = str(args.resume) if args.resume else _create_run_dir(str(args.log_dir))
    if args.resume and not os.path.isdir(run_dir):
        raise FileNotFoundError(f"resume run directory not found: {run_dir}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    lock = act_train._RunDirLock(run_dir)
    lock.acquire(mode="resume" if args.resume else "new")
    atexit.register(lock.release)

    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    lowdim_keys = parse_lowdim_keys(args.lowdim_keys)
    lowdim_dims = infer_lowdim_dims_from_stats(args.stats_npz_override, lowdim_keys=lowdim_keys)
    policy, ckpt_meta = load_act_policy_wrapper(
        ckpt_path=args.agent,
        device=device,
        task=args.task,
    )
    if int(args.max_chunk_len) != int(ckpt_meta["chunk_size"]):
        raise ValueError(
            f"max_chunk_len={args.max_chunk_len} does not match ACT chunk_size={ckpt_meta['chunk_size']}"
        )

    bc_policy = policy.policy
    if not hasattr(bc_policy, "act_bc_loss_terms"):
        raise RuntimeError("ACT policy adapter does not implement act_bc_loss_terms")
    bc_replay = OnlineSuccessBCReplay(
        capacity=int(args.bc_success_capacity),
        horizon_steps=int(bc_policy.horizon_steps),
        action_dim=int(bc_policy.ac_dim),
        cond_steps=int(bc_policy.cond_steps),
        obs_dim=int(bc_policy.obs_dim),
        device=device,
        padded=str(args.bc_tail_padding) == "act_native",
    )
    trainable_names, trainable_parameters = select_act_bc_trainable_parameters(
        bc_policy.policy_model,
        str(args.bc_act_trainable_scope),
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(args.bc_lr),
        weight_decay=0.0,
    )
    teacher_model = None
    if float(args.bc_teacher_constraint_weight) > 0.0:
        teacher_model = copy.deepcopy(bc_policy.policy_model).to(device).eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)

    invariants = _invariants(args, ckpt_meta)
    policy_steps = 0
    env_steps = 0
    collector_steps = 0
    bc_update_count = 0
    virtual_update_seen_count = 0
    next_eval_axis = int(args.offline_env_eval_interval_steps)
    next_latest_axis = int(args.offline_resume_save_interval_steps)
    next_log_policy = int(args.log_every_policy_steps)
    best_eval_success = -1.0
    success_buffer = deque(maxlen=int(args.window_size))
    success_buffer_20 = deque(maxlen=20)
    exec_buffer = deque(maxlen=2000)
    episode_return_buffer = deque(maxlen=int(args.window_size))
    episode_decision_len_buffer = deque(maxlen=int(args.window_size))
    train_meter: Dict[str, List[float]] = defaultdict(list)
    history = {"timesteps_env": [], "timesteps_policy": [], "metrics": {}}
    metrics_path = os.path.join(run_dir, "training_metrics.json")
    metrics_backup_path = os.path.join(run_dir, "training_metrics.backup.json")
    status_path = os.path.join(run_dir, "status.log")

    resume_rng_state = None
    if args.resume:
        checkpoint_path = os.path.join(ckpt_dir, str(args.resume_ckpt_name))
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        state = act_train._load_ckpt(checkpoint_path, map_location=device)
        if str(state.get("pipeline")) != PIPELINE_NAME:
            raise ValueError(f"not a {PIPELINE_NAME} checkpoint: {checkpoint_path}")
        if int(state.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"unsupported checkpoint format: {state.get('format_version')}")
        _assert_resume_invariants(state.get("invariants", {}), invariants)
        bc_policy.policy_model.load_state_dict(state["act_policy_state"], strict=True)
        optimizer.load_state_dict(state["bc_optimizer_state"])
        bc_replay.load_state_dict(state["bc_replay_state"])
        if teacher_model is not None:
            teacher_model.load_state_dict(state["bc_teacher_state"], strict=True)
        policy_steps = int(state["policy_steps"])
        env_steps = int(state["env_steps"])
        collector_steps = int(state["collector_steps"])
        bc_update_count = int(state["bc_update_count"])
        virtual_update_seen_count = int(state["virtual_update_seen_count"])
        next_eval_axis = int(state["next_eval_axis"])
        next_latest_axis = int(state["next_latest_axis"])
        next_log_policy = int(state["next_log_policy"])
        best_eval_success = float(state.get("best_eval_success", -1.0))
        success_buffer.extend(state.get("success_buffer", []))
        success_buffer_20.extend(state.get("success_buffer_20", []))
        exec_buffer.extend(state.get("exec_buffer", []))
        episode_return_buffer.extend(state.get("episode_return_buffer", []))
        episode_decision_len_buffer.extend(state.get("episode_decision_len_buffer", []))
        loaded_history, loaded_path = act_train._load_json_with_backup(
            metrics_path, metrics_backup_path
        )
        if loaded_history is not None:
            history = loaded_history
        resume_rng_state = state.get("rng_state")
        line = (
            f"[PURE-BC][RESUME] checkpoint={checkpoint_path} metrics={loaded_path} "
            f"policy_steps={policy_steps} env_steps={env_steps} bc_updates={bc_update_count}"
        )
        print(line, flush=True)
        act_train._append_status_line(status_path, line)

    env_meta = load_env_meta_json(args.env_meta_json)
    env_meta.setdefault("env_kwargs", {})
    env_meta["env_kwargs"]["horizon"] = int(args.env_horizon_override)
    _wait_for_memory(args, context="collector construction")
    venv = _build_task_venv(
        args=args,
        policy=policy,
        device=device,
        lowdim_keys=lowdim_keys,
        lowdim_dims=lowdim_dims,
        env_meta=env_meta,
        n_envs=int(args.n_envs),
        seed=int(args.seed),
        collect_success_episodes=True,
    )
    eval_venv = None
    if not bool(args.async_eval_snapshots):
        eval_venv = _build_task_venv(
            args=args,
            policy=policy,
            device=device,
            lowdim_keys=lowdim_keys,
            lowdim_dims=lowdim_dims,
            env_meta=env_meta,
            n_envs=int(args.offline_eval_n_envs),
            seed=int(args.seed) + int(args.offline_eval_seed_offset),
            collect_success_episodes=False,
        )
    venv.reset()
    action_dim = int(venv.action_space.n)
    expected_action_dim = int(args.max_chunk_len) - int(args.min_chunk_len) + 1
    if action_dim != expected_action_dim or (
        eval_venv is not None and int(eval_venv.action_space.n) != action_dim
    ):
        raise RuntimeError(
            f"unexpected fixed-k action spaces: train={action_dim} "
            f"eval={None if eval_venv is None else eval_venv.action_space.n} "
            f"expected={expected_action_dim}"
        )
    action_id = _fixed_action_id(
        fixed_exec_len=int(args.fixed_exec_len),
        min_chunk_len=int(args.min_chunk_len),
        action_dim=action_dim,
    )
    actions = np.full((int(args.n_envs),), action_id, dtype=np.int64)

    collector_wrapper = getattr(venv, "venv", venv)
    eval_wrapper = getattr(eval_venv, "venv", eval_venv) if eval_venv is not None else None
    if hasattr(collector_wrapper, "sync_policy_replicas_from_primary"):
        collector_wrapper.sync_policy_replicas_from_primary()
    if hasattr(collector_wrapper, "refresh_current_policy_plan"):
        collector_wrapper.refresh_current_policy_plan()
    if resume_rng_state is not None:
        # Environment construction performs policy inference. Restore the saved
        # RNG state afterwards so BC replay sampling resumes at the saved point.
        _restore_rng_state_strict(resume_rng_state)

    import mujoco
    import robosuite

    mujoco_package_version = str(mujoco.__version__)
    mujoco_runtime_version = str(mujoco.mj_versionString())
    robosuite_version = str(robosuite.__version__)
    robosuite_file = os.path.realpath(str(robosuite.__file__))
    expected_mujoco = str(args.expected_mujoco_version)
    expected_robosuite = str(args.expected_robosuite_version)
    if mujoco_package_version != expected_mujoco or mujoco_runtime_version != expected_mujoco:
        raise RuntimeError(
            "MuJoCo runtime mismatch: "
            f"package={mujoco_package_version} runtime={mujoco_runtime_version} "
            f"expected={expected_mujoco}"
        )
    if robosuite_version != expected_robosuite:
        raise RuntimeError(
            f"Robosuite runtime mismatch: loaded={robosuite_version} expected={expected_robosuite} "
            f"file={robosuite_file}"
        )
    if str(args.expected_robosuite_root).strip():
        expected_root = os.path.realpath(str(args.expected_robosuite_root))
        try:
            under_expected_root = (
                os.path.commonpath([robosuite_file, expected_root]) == expected_root
            )
        except ValueError:
            under_expected_root = False
        if not under_expected_root:
            raise RuntimeError(
                f"Robosuite loaded from {robosuite_file}, expected a module under {expected_root}"
            )

    writer = SummaryWriter(log_dir=os.path.join(run_dir, "scalars_1"), purge_step=env_steps)
    writer.add_scalar("diagnostic/rl_enabled", 0.0, int(env_steps))
    writer.add_scalar(
        "diagnostic/fixed_requested_exec_len", float(args.fixed_exec_len), int(env_steps)
    )
    writer.add_scalar(
        "diagnostic/virtual_rl_updates_per_collector_step",
        float(args.virtual_rl_updates_per_collector_step),
        int(env_steps),
    )
    episode_returns = np.zeros((int(args.n_envs),), dtype=np.float64)
    episode_decision_lengths = np.zeros((int(args.n_envs),), dtype=np.int64)
    update_every_schedule = act_train._parse_int_schedule(
        args.bc_update_every_rl_updates_schedule,
        min_value=1,
        max_value=None,
    )
    lr_schedule = act_train._parse_float_schedule(
        args.bc_lr_schedule,
        min_value=0.0,
        max_value=None,
    )
    start_time = time.time()

    def record_history(metrics: Dict[str, float], *, axis: int) -> None:
        history.setdefault("timesteps_env", []).append(int(axis))
        history.setdefault("timesteps_policy", []).append(int(policy_steps))
        metric_store = history.setdefault("metrics", {})
        for key, value in metrics.items():
            metric_store.setdefault(key, []).append(float(value))

    def dump_metrics() -> None:
        act_train._atomic_json_dump(history, metrics_path, metrics_backup_path)

    def pack_checkpoint() -> Dict:
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "pipeline": PIPELINE_NAME,
            "invariants": invariants,
            "rng_state": act_train._pack_rng_state(),
            "act_policy_state": bc_policy.policy_model.state_dict(),
            "bc_teacher_state": teacher_model.state_dict() if teacher_model is not None else None,
            "bc_optimizer_state": optimizer.state_dict(),
            "bc_replay_state": bc_replay.state_dict(copy_arrays=False, as_torch=True),
            "policy_steps": int(policy_steps),
            "env_steps": int(env_steps),
            "collector_steps": int(collector_steps),
            "bc_update_count": int(bc_update_count),
            "virtual_update_seen_count": int(virtual_update_seen_count),
            "next_eval_axis": int(next_eval_axis),
            "next_latest_axis": int(next_latest_axis),
            "next_log_policy": int(next_log_policy),
            "best_eval_success": float(best_eval_success),
            "success_buffer": list(success_buffer),
            "success_buffer_20": list(success_buffer_20),
            "exec_buffer": list(exec_buffer),
            "episode_return_buffer": list(episode_return_buffer),
            "episode_decision_len_buffer": list(episode_decision_len_buffer),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

    def save_checkpoint(name: str) -> None:
        if bool(args.disable_checkpoints):
            return
        _wait_for_memory(args, context=f"checkpoint {name}")
        act_train._save_ckpt(os.path.join(ckpt_dir, name), pack_checkpoint())

    snapshot_dir = (
        os.path.abspath(str(args.async_eval_snapshot_dir))
        if str(args.async_eval_snapshot_dir).strip()
        else os.path.join(run_dir, "eval_snapshots")
    )

    def save_async_eval_snapshot(axis: int) -> str:
        if not bool(args.async_eval_snapshots):
            raise RuntimeError("async eval snapshot requested while async mode is disabled")
        os.makedirs(snapshot_dir, exist_ok=True)
        path = os.path.join(snapshot_dir, f"env_{int(axis):010d}.pt")
        payload = {
            "format": "act_pure_online_bc_eval_snapshot_v1",
            "eval_axis": int(axis),
            "actual_env_steps": int(env_steps),
            "policy_steps": int(policy_steps),
            "bc_update_count": int(bc_update_count),
            "act_policy_state": {
                key: value.detach().cpu()
                for key, value in bc_policy.policy_model.state_dict().items()
            },
            "args": dict(vars(args)),
            "invariants": dict(invariants),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        act_train._save_ckpt(path, payload)
        line = (
            f"[PURE-BC][ASYNC-SNAPSHOT] axis={int(axis)} actual_env={int(env_steps)} "
            f"policy={int(policy_steps)} path={path}"
        )
        print(line, flush=True)
        act_train._append_status_line(status_path, line)
        return path

    def run_eval(axis: int) -> float:
        nonlocal best_eval_success
        if eval_venv is None or eval_wrapper is None:
            raise RuntimeError("in-process eval requested while async snapshots are enabled")
        _wait_for_memory(args, context="evaluation")
        if hasattr(eval_wrapper, "sync_policy_replicas_from_primary"):
            eval_wrapper.sync_policy_replicas_from_primary()
        result = _run_fixed_eval(
            venv=eval_venv,
            action_id=action_id,
            fixed_exec_len=int(args.fixed_exec_len),
            episodes=int(args.offline_eval_episodes),
            max_decisions=int(args.offline_eval_max_decisions),
        )
        metrics = {
            "env_step_eval/success_rate": result["success_rate"],
            "env_step_eval/success_rate_recent20": result["success_rate_recent20"],
            "env_step_eval/avg_exec_len": result["avg_exec_len"],
            "env_step_eval/ep_rew_mean": result["episode_return_mean"],
            "env_step_eval/episodes": result["episodes"],
            "env_step_eval/eval_env_steps": result["eval_env_steps"],
            "env_step_eval/decisions": result["decisions"],
            "env_step_eval/train_env_steps_actual": float(env_steps),
            "env_step_eval/train_policy_steps": float(policy_steps),
            "env_step_eval/bc_updates": float(bc_update_count),
            "env_step_eval/fixed_requested_exec_len": float(args.fixed_exec_len),
        }
        for key, value in metrics.items():
            writer.add_scalar(key, float(value), int(axis))
        record_history(metrics, axis=int(axis))
        line = (
            f"[PURE-BC][EVAL] axis={axis} actual_env={env_steps} policy={policy_steps} "
            f"success={result['success_rate']:.4f} success20={result['success_rate_recent20']:.4f} "
            f"avg_k={result['avg_exec_len']:.2f} episodes={int(result['episodes'])}"
        )
        print(line, flush=True)
        act_train._append_status_line(status_path, line)
        if float(result["success_rate"]) > float(best_eval_success):
            best_eval_success = float(result["success_rate"])
            if bool(args.save_best_model):
                save_checkpoint("best_model.pt")
        dump_metrics()
        writer.flush()
        return float(result["success_rate"])

    def run_one_bc_update() -> Optional[Dict[str, float]]:
        nonlocal bc_update_count
        schedule_counter = env_steps if str(args.bc_schedule_axis) == "env" else policy_steps
        if schedule_counter < int(args.bc_start_after_steps):
            return None
        if not bc_replay.can_sample(int(args.bc_batch_size)):
            return None
        current_lr = act_train._resolve_float_schedule(
            int(schedule_counter), lr_schedule, default=float(args.bc_lr)
        )
        for group in optimizer.param_groups:
            group["lr"] = float(current_lr)
        if hasattr(collector_wrapper, "set_primary_policy_trainable"):
            collector_wrapper.set_primary_policy_trainable(True)
        else:
            bc_policy.set_train()
            for parameter in bc_policy.policy_model.parameters():
                parameter.requires_grad_(True)
        trainable_set = set(trainable_names)
        for name, parameter in bc_policy.policy_model.named_parameters():
            parameter.requires_grad_(name in trainable_set)
        optimizer.zero_grad(set_to_none=True)
        sampling_mode = (
            "uniform_episode" if args.bc_online_sampling == "episode" else "uniform_window"
        )
        batch = bc_replay.sample(int(args.bc_batch_size), sampling=sampling_mode)
        online_loss, teacher_loss = bc_policy.act_bc_loss_terms(
            batch.actions,
            batch.conditions,
            action_is_pad=batch.action_is_pad,
            teacher_policy_model=teacher_model,
        )
        loss = online_loss + float(args.bc_teacher_constraint_weight) * teacher_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite ACT BC loss: total={loss} online={online_loss} teacher={teacher_loss}"
            )
        loss.backward()
        grad_norm = 0.0
        if float(args.bc_grad_clip_norm) > 0.0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    float(args.bc_grad_clip_norm),
                )
            )
        optimizer.step()
        if teacher_model is not None and float(args.bc_teacher_ema_decay) < 1.0:
            ema_update_module_(
                teacher_model,
                bc_policy.policy_model,
                float(args.bc_teacher_ema_decay),
            )
        bc_update_count += 1
        if hasattr(collector_wrapper, "sync_policy_replicas_from_primary"):
            collector_wrapper.sync_policy_replicas_from_primary()
        else:
            bc_policy.set_eval()
            for parameter in bc_policy.policy_model.parameters():
                parameter.requires_grad_(False)
        if hasattr(collector_wrapper, "refresh_current_policy_plan"):
            collector_wrapper.refresh_current_policy_plan()
        return {
            "bc/loss": float(loss.detach().cpu()),
            "bc/loss_online": float(online_loss.detach().cpu()),
            "bc/loss_teacher": float(teacher_loss.detach().cpu()),
            "bc/grad_norm_before_clip": float(grad_norm),
            "bc/current_lr": float(current_lr),
        }

    line = (
        f"[PURE-BC] no_rl=1 task={args.task} fixed_k={args.fixed_exec_len} "
        f"n_envs={args.n_envs} virtual_utd={args.virtual_rl_updates_per_collector_step} "
        f"bc_every_virtual={args.bc_update_every_rl_updates} batch={args.bc_batch_size} "
        f"lr={args.bc_lr:.3e} eval_every_env={args.offline_env_eval_interval_steps} "
        f"eval_episodes={args.offline_eval_episodes} async_eval={int(args.async_eval_snapshots)} "
        f"stop_after_env={int(args.stop_after_env_steps)} "
        f"memory_floor={args.memory_floor_gib:.1f}GiB"
    )
    print(line, flush=True)
    act_train._append_status_line(status_path, line)

    try:
        if not args.resume and bool(args.offline_env_eval_at_start):
            if bool(args.async_eval_snapshots):
                save_async_eval_snapshot(0)
            else:
                run_eval(0)

        while policy_steps < int(args.total_policy_steps) and (
            int(args.stop_after_env_steps) <= 0 or env_steps < int(args.stop_after_env_steps)
        ):
            _wait_for_memory(args, context="collector step")
            _, rewards, dones, infos = venv.step(actions)
            rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
            dones = np.asarray(dones, dtype=bool).reshape(-1)
            episodes_added, windows_added = _collect_success_infos(infos, bc_replay)
            train_meter["bc/success_episodes_added_this_step"].append(float(episodes_added))
            train_meter["bc/success_windows_added_this_step"].append(float(windows_added))
            for index in range(int(args.n_envs)):
                info = infos[index] if index < len(infos) else {}
                actual_k = int(info.get("exec_len", 0))
                if actual_k < 0 or actual_k > int(args.fixed_exec_len):
                    raise RuntimeError(
                        f"collector returned invalid exec_len={actual_k} for requested k={args.fixed_exec_len}"
                    )
                env_steps += actual_k
                exec_buffer.append(float(actual_k))
                episode_returns[index] += float(rewards[index])
                episode_decision_lengths[index] += 1
                if bool(dones[index]):
                    success = 1 if act_train._extract_success(info) else 0
                    success_buffer.append(success)
                    success_buffer_20.append(success)
                    episode_return_buffer.append(float(episode_returns[index]))
                    episode_decision_len_buffer.append(float(episode_decision_lengths[index]))
                    episode_returns[index] = 0.0
                    episode_decision_lengths[index] = 0
            policy_steps += int(args.n_envs)
            collector_steps += 1

            for _ in range(int(args.virtual_rl_updates_per_collector_step)):
                virtual_update_seen_count += 1
                schedule_counter = env_steps if args.bc_schedule_axis == "env" else policy_steps
                update_every = act_train._resolve_int_schedule(
                    int(schedule_counter),
                    update_every_schedule,
                    default=int(args.bc_update_every_rl_updates),
                )
                if virtual_update_seen_count % int(update_every) != 0:
                    continue
                for _ in range(int(args.bc_updates_per_rl_update)):
                    _wait_for_memory(args, context="BC update")
                    update_metrics = run_one_bc_update()
                    if update_metrics is not None:
                        for key, value in update_metrics.items():
                            train_meter[key].append(float(value))

            while env_steps >= int(next_eval_axis):
                if bool(args.async_eval_snapshots):
                    save_async_eval_snapshot(int(next_eval_axis))
                else:
                    run_eval(int(next_eval_axis))
                next_eval_axis += int(args.offline_env_eval_interval_steps)

            if env_steps >= int(next_latest_axis):
                while env_steps >= int(next_latest_axis):
                    next_latest_axis += int(args.offline_resume_save_interval_steps)
                save_checkpoint("latest_model.pt")

            if policy_steps >= int(next_log_policy):
                elapsed = max(time.time() - start_time, 1e-6)
                success_recent = float(np.mean(success_buffer)) if success_buffer else 0.0
                success_recent20 = float(np.mean(success_buffer_20)) if success_buffer_20 else 0.0
                metrics = {
                    "rollout/success_rate": success_recent,
                    "rollout/success_rate_recent20": success_recent20,
                    "rollout/ep_rew_mean": (
                        float(np.mean(episode_return_buffer)) if episode_return_buffer else 0.0
                    ),
                    "rollout/ep_len_mean": (
                        float(np.mean(episode_decision_len_buffer))
                        if episode_decision_len_buffer
                        else 0.0
                    ),
                    "adaptive/avg_exec_len": float(np.mean(exec_buffer)) if exec_buffer else 0.0,
                    "time/policy_steps": float(policy_steps),
                    "time/env_steps": float(env_steps),
                    "time/policy_steps_per_sec": float(policy_steps / elapsed),
                    "time/env_steps_per_sec": float(env_steps / elapsed),
                    "bc/update_count": float(bc_update_count),
                    "bc/virtual_update_seen_count": float(virtual_update_seen_count),
                    "bc/success_windows": float(len(bc_replay)),
                    "bc/success_episodes_added": float(bc_replay.success_episodes_added),
                    "bc/active_success_episodes": float(bc_replay.active_episode_count),
                    "diagnostic/rl_enabled": 0.0,
                    "diagnostic/fixed_requested_exec_len": float(args.fixed_exec_len),
                    "system/mem_available_gib": float(_available_memory_gib() or -1.0),
                }
                for key, values in train_meter.items():
                    if values:
                        metrics[key] = float(np.mean(values))
                train_meter.clear()
                for key, value in metrics.items():
                    writer.add_scalar(key, float(value), int(env_steps))
                record_history(metrics, axis=int(env_steps))
                line = (
                    f"[PURE-BC][TRAIN] policy={policy_steps}/{args.total_policy_steps} "
                    f"env={env_steps} success={success_recent:.4f} success20={success_recent20:.4f} "
                    f"avg_k={metrics['adaptive/avg_exec_len']:.2f} "
                    f"bc_updates={bc_update_count} windows={len(bc_replay)}"
                )
                print(line, flush=True)
                act_train._append_status_line(status_path, line)
                dump_metrics()
                writer.flush()
                while policy_steps >= int(next_log_policy):
                    next_log_policy += int(args.log_every_policy_steps)

        save_checkpoint("latest_model.pt")
        dump_metrics()
        line = f"[PURE-BC] training finished policy_steps={policy_steps} env_steps={env_steps}"
        print(line, flush=True)
        act_train._append_status_line(status_path, line)
        return run_dir
    finally:
        try:
            writer.flush()
            writer.close()
        except Exception:
            pass
        try:
            if eval_venv is not None:
                eval_venv.close()
        except Exception:
            pass
        try:
            venv.close()
        except Exception:
            pass


def main() -> None:
    act_train._configure_stdio()
    parser = build_parser()
    args = parser.parse_args()
    act_train._resolve_task_defaults(args)
    act_train._validate_task_alignment(args)
    os.makedirs(args.output, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
