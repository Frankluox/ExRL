"""
Minimal DP-native OGPO online runner.

This is a new entrypoint only. It does not modify AAC/SAC runners. The runner
keeps OGPO_public's action-queue and sequence-replay timing while using the
existing PyTorch DSRL diffusion checkpoint as the trainable actor.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

CURRENT_DIR = __import__("exrl.paths", fromlist=["ROOT"]).ROOT
REPO_ROOT = __import__("exrl.paths", fromlist=["ROOT"]).ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))

from exrl.runners.dp_q.policy_loader import (  # noqa: E402
    infer_action_dim_from_stats,
    infer_lowdim_dims_from_stats,
    load_dsrl_policy_wrapper,
    parse_lowdim_keys,
)
from exrl.runners.dp_q.vector_env import RobomimicExecutor, _extract_state_from_obs  # noqa: E402
from baselines.ogpo.robomimic.dp_core import (
    DPNativeOGPOAgent,
    DPNativeOGPOConfig,
    OGPOSequenceReplayBuffer,
)  # noqa: E402
from baselines.ogpo.robomimic.update_schedule import OGPOUpdateSchedule, parse_utd  # noqa: E402
from baselines.ogpo.robomimic.runtime import validate_runtime  # noqa: E402


def _str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, help="DSRL / DP checkpoint path")
    parser.add_argument("--weight_source", default="ema", choices=["ema", "model"])
    parser.add_argument("--dsrl_official_root", required=True)
    parser.add_argument("--required_robosuite_version", default="")
    parser.add_argument("--required_mujoco_version", default="")
    parser.add_argument("--env_name", default="can")
    parser.add_argument("--env_meta_json", required=True)
    parser.add_argument("--normalization_path", required=True)
    parser.add_argument("--stats_npz", required=True)
    parser.add_argument(
        "--lowdim_keys", default="robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,object"
    )
    parser.add_argument("--cond_steps", type=int, default=1)
    parser.add_argument("--max_episode_steps", type=int, default=300)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--expected_horizon_steps", type=int, default=0)

    parser.add_argument("--output", default="./adaptive_rl_results/ogpo_dp_pytorch")
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--resume_checkpoint", default="", help="Full OGPO checkpoint to resume in place"
    )
    parser.add_argument("--online_steps", type=int, default=2000000)
    parser.add_argument("--start_training", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--success_buffer_batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=2000000)
    parser.add_argument(
        "--utd", type=parse_utd, default="1", help="Updates per eligible vector step, e.g. 1/16"
    )
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--success_bonus", type=float, default=5.0)
    parser.add_argument("--reward_offset", type=float, default=1.0)
    parser.add_argument("--terminate_on_success", action="store_true", default=True)
    parser.add_argument("--timeout_on_horizon", action="store_true", default=False)

    parser.add_argument("--critic_hidden_dims", default="512,512,512,512")
    parser.add_argument("--num_qs", type=int, default=10)
    parser.add_argument("--q_agg", default="mean", choices=["mean", "min"])
    parser.add_argument("--ppo_lr", type=float, default=4.5e-5)
    parser.add_argument("--actor_lr", type=float, default=None)
    parser.add_argument("--critic_lr", type=float, default=None)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--actor_tau", type=float, default=0.005)
    parser.add_argument("--bc_coeff", type=float, default=1.0)
    parser.add_argument("--update_mode", default="ogpo", choices=["ogpo", "bc_only"])
    parser.add_argument("--bc_only_success_only", type=_str2bool, default=True)
    parser.add_argument(
        "--rollout_policy", default="auto", choices=["auto", "bon", "sde", "deterministic"]
    )
    parser.add_argument("--group_num_samples", type=int, default=32)
    parser.add_argument(
        "--actor_microbatch_size",
        type=int,
        default=0,
        help="Chain rows per backward; 0 disables splitting",
    )
    parser.add_argument(
        "--inference_microbatch_size",
        type=int,
        default=0,
        help="Rows per target actor forward; 0 disables splitting",
    )
    parser.add_argument("--fuse_log_prob_steps", type=_str2bool, default=False)
    parser.add_argument("--vectorized_critic", type=_str2bool, default=False)
    parser.add_argument(
        "--matmul_precision", choices=["highest", "high", "medium"], default="highest"
    )
    parser.add_argument("--compile_models", type=_str2bool, default=False)
    parser.add_argument(
        "--compile_mode", choices=["default", "reduce-overhead", "max-autotune"], default="default"
    )
    parser.add_argument("--clip_epsilon", type=float, default=0.01)
    parser.add_argument("--best_of_n", type=int, default=8)
    parser.add_argument(
        "--train_denoising_steps",
        type=int,
        default=0,
        help="0 means infer from checkpoint metadata",
    )
    parser.add_argument(
        "--eval_ddim_steps", type=int, default=0, help="0 means infer from checkpoint metadata"
    )
    parser.add_argument("--ft_denoising_steps", type=int, default=0, help="0 means full DDIM path")
    parser.add_argument("--use_ogpo_sde", type=_str2bool, default=True)
    parser.add_argument("--use_constant_noise", type=_str2bool, default=False)
    parser.add_argument("--use_tapered_noise", type=_str2bool, default=True)
    parser.add_argument("--error_correct_sde_to_ode", type=_str2bool, default=True)
    parser.add_argument("--constant_noise_std", type=float, default=0.05)
    parser.add_argument("--min_noise_std", type=float, default=0.01)
    parser.add_argument("--denoised_clip_value", type=float, default=1.0)
    parser.add_argument("--randn_clip_value", type=float, default=3.0)
    parser.add_argument("--final_action_clip_value", type=float, default=1.0)

    parser.add_argument("--log_interval", type=int, default=5000)
    parser.add_argument("--save_interval", type=int, default=30000)
    parser.add_argument("--eval_snapshot_interval", type=int, default=0)
    parser.add_argument("--eval_interval", type=int, default=30000)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--n_eval_envs", type=int, default=5)
    parser.add_argument("--eval_sde", type=_str2bool, default=True)
    parser.add_argument(
        "--profile_timing",
        type=_str2bool,
        default=False,
        help="Synchronize updates for wall-clock timing",
    )
    return parser


def _parse_ints(spec: str) -> List[int]:
    vals = [int(x.strip()) for x in str(spec).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"empty int list: {spec}")
    return vals


def _effective_rollout_policy(update_mode: str, rollout_policy: str) -> str:
    if rollout_policy != "auto":
        return rollout_policy
    return "sde" if update_mode == "bc_only" else "bon"


def _select_action_chunks(
    agent: DPNativeOGPOAgent, states: np.ndarray, rollout_policy: str, best_of_n: int
) -> np.ndarray:
    if rollout_policy == "deterministic":
        return agent.select_action_chunk_np(states, deterministic=True, best_of_n=1)
    if rollout_policy == "sde":
        return agent.select_action_chunk_np(states, deterministic=False, best_of_n=1)
    if rollout_policy == "bon":
        return agent.select_action_chunk_np(states, deterministic=False, best_of_n=int(best_of_n))
    raise ValueError(f"unknown rollout_policy={rollout_policy!r}")


def _make_run_dir(args) -> str:
    if args.resume_checkpoint:
        checkpoint = os.path.abspath(os.path.expanduser(args.resume_checkpoint))
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
        run_dir = os.path.dirname(checkpoint)
        if args.run_name is not None:
            expected = os.path.abspath(os.path.join(args.output, args.run_name))
            if expected != run_dir:
                raise ValueError(
                    f"resume run directory mismatch: expected={expected}, checkpoint={run_dir}"
                )
        return run_dir
    run_name = args.run_name or f"OGPO_DP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _capture_rng_state() -> Dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = state["torch_cuda"]
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                f"CUDA RNG device count mismatch: checkpoint={len(cuda_states)}, current={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _last_train_progress(run_dir: str, max_step: int) -> Dict:
    path = os.path.join(run_dir, "metrics.jsonl")
    latest = {}
    if not os.path.isfile(path):
        return latest
    with open(path, errors="replace") as handle:
        for line in handle:
            if not line.endswith("\n") or not line.strip():
                continue
            row = json.loads(line)
            if row.get("phase") == "train" and int(row.get("env_steps", -1)) <= int(max_step):
                latest = row
    return latest


def _validate_resume_config(
    checkpoint_config: DPNativeOGPOConfig, config: DPNativeOGPOConfig
) -> None:
    operational_fields = {
        "actor_microbatch_size",
        "inference_microbatch_size",
        "fuse_log_prob_steps",
        "compile_models",
        "compile_mode",
    }
    current = config.__dict__
    checkpoint = checkpoint_config.__dict__
    mismatches = {}
    for key, old_value in checkpoint.items():
        if key in operational_fields:
            continue
        new_value = current.get(key)
        if new_value != old_value:
            mismatches[key] = {"checkpoint": old_value, "current": new_value}
    if mismatches:
        raise ValueError(f"resume algorithm config mismatch: {mismatches}")


def _restore_update_schedule(
    update_schedule: OGPOUpdateSchedule, checkpoint: Dict, run_dir: str
) -> str:
    training_state = checkpoint.get("training_state", {})
    if "update_schedule" in training_state:
        update_schedule.load_state_dict(training_state["update_schedule"])
        return "checkpoint"

    progress = _last_train_progress(run_dir, int(checkpoint["step"]))
    if not progress or "eligible_vector_steps" not in progress:
        raise ValueError("legacy checkpoint resume requires eligible_vector_steps in metrics.jsonl")
    eligible_steps = int(progress["eligible_vector_steps"])
    state = update_schedule.state_dict()
    state.update(
        {
            "eligible_steps": eligible_steps,
            "credit": (eligible_steps * update_schedule.rate.numerator)
            % update_schedule.rate.denominator,
        }
    )
    update_schedule.load_state_dict(state)
    expected_updates = (
        eligible_steps * update_schedule.rate.numerator // update_schedule.rate.denominator
    )
    actual_updates = int(checkpoint["agent"].get("update_step", 0))
    if actual_updates != expected_updates:
        raise ValueError(
            f"legacy update count mismatch: schedule={expected_updates}, checkpoint={actual_updates}"
        )
    return "metrics_legacy"


def _training_state(
    update_schedule: OGPOUpdateSchedule,
    *,
    updates: int,
    skipped_updates: int,
    success_batch_updates: int,
    recent_successes,
    recent_returns,
    recent_lengths,
) -> Dict:
    return {
        "format": "ogpo_training_state_v1",
        "update_schedule": update_schedule.state_dict(),
        "updates": int(updates),
        "skipped_updates": int(skipped_updates),
        "success_batch_updates": int(success_batch_updates),
        "recent_successes": list(recent_successes),
        "recent_returns": list(recent_returns),
        "recent_lengths": list(recent_lengths),
        "rng": _capture_rng_state(),
        "environment_state_saved": False,
    }


def _save_checkpoint(
    run_dir: str,
    agent: DPNativeOGPOAgent,
    replay: OGPOSequenceReplayBuffer,
    step: int,
    meta: Dict,
    training_state: Dict,
) -> None:
    path = os.path.join(run_dir, "latest_model.pt")
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(
        {
            "step": int(step),
            "agent": agent.state_dict(),
            "replay": replay,
            "meta": dict(meta),
            "training_state": training_state,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def _cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    module = getattr(module, "_orig_mod", module)
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _save_eval_snapshot(run_dir: str, agent: DPNativeOGPOAgent, step: int, meta: Dict) -> str:
    snapshot_dir = os.path.join(run_dir, "eval_snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    path = os.path.join(snapshot_dir, f"eval_step_{int(step):012d}.pt")
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite existing eval snapshot: {path}")
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(
        {
            "format": "ogpo_target_actor_eval_snapshot_v1",
            "step": int(step),
            "agent": {"target_actor": _cpu_state_dict(agent.target_actor)},
            "meta": {
                "horizon_steps": int(meta["horizon_steps"]),
                "obs_dim": int(meta["obs_dim"]),
                "action_dim": int(meta["action_dim"]),
            },
        },
        tmp_path,
    )
    os.replace(tmp_path, path)
    return path


def _write_json(path: str, payload: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _append_jsonl(path: str, payload: Dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _terminal_state_from_info(info_i: Dict, fallback: np.ndarray) -> np.ndarray:
    terminal_observation = info_i.get("terminal_observation", None)
    if terminal_observation is None:
        return fallback.copy()
    if isinstance(terminal_observation, dict) and "state" in terminal_observation:
        arr = np.asarray(terminal_observation["state"], dtype=np.float32)
        if arr.shape == fallback.shape:
            return arr.copy()
        if arr.ndim >= 2 and arr.shape[-1] == fallback.shape[-1]:
            return arr.reshape(-1, arr.shape[-1])[-1].copy()
    try:
        return _extract_state_from_obs(terminal_observation)[0].astype(np.float32)
    except Exception:
        arr = np.asarray(terminal_observation, dtype=np.float32)
        if arr.shape == fallback.shape:
            return arr.copy()
        if arr.ndim >= 2 and arr.shape[-1] == fallback.shape[-1]:
            return arr.reshape(-1, arr.shape[-1])[-1].copy()
    return fallback.copy()


def _enqueue_full_action_chunks(
    action_queues: List[deque],
    env_indices: List[int],
    chunks: np.ndarray,
    expected_horizon_steps: int,
) -> None:
    chunks = np.asarray(chunks, dtype=np.float32)
    if chunks.ndim != 3 or int(chunks.shape[0]) != len(env_indices):
        raise ValueError(
            f"action chunk batch mismatch: shape={chunks.shape}, env_indices={len(env_indices)}"
        )
    if int(chunks.shape[1]) != int(expected_horizon_steps):
        raise ValueError(
            f"full action chunk required: expected {expected_horizon_steps}, got {chunks.shape[1]}"
        )
    for local_j, env_i in enumerate(env_indices):
        if action_queues[env_i]:
            raise ValueError(f"refusing to replace non-empty action queue for env {env_i}")
        action_queues[env_i].extend([action.astype(np.float32) for action in chunks[local_j]])


@torch.no_grad()
def _evaluate_executor(
    agent: DPNativeOGPOAgent,
    executor: RobomimicExecutor,
    *,
    deterministic: bool,
    best_of_n: int,
    eval_episodes: int,
    seed: int,
) -> Dict[str, float]:
    states = executor.reset_all(base_seed=int(seed))
    action_queues: List[deque] = [deque() for _ in range(int(executor.n_envs))]
    returns = np.zeros((int(executor.n_envs),), dtype=np.float32)
    lengths = np.zeros((int(executor.n_envs),), dtype=np.int64)
    successes: List[float] = []
    ep_returns: List[float] = []
    ep_lengths: List[float] = []
    horizon_steps = int(agent.config.horizon_steps)

    while len(successes) < int(eval_episodes):
        empty = [i for i, q in enumerate(action_queues) if len(q) == 0]
        if empty:
            chunks = agent.select_action_chunk_np(
                states[empty],
                deterministic=bool(deterministic),
                best_of_n=int(best_of_n),
            )
            _enqueue_full_action_chunks(action_queues, empty, chunks, horizon_steps)

        actions = np.stack(
            [action_queues[i].popleft() for i in range(int(executor.n_envs))], axis=0
        ).astype(np.float32)
        step = executor.step_variable_k(
            action_chunks=actions[:, None, :],
            exec_lens=np.ones((int(executor.n_envs),), dtype=np.int64),
        )
        states = np.asarray(step.next_state, dtype=np.float32)
        rewards = np.asarray(step.rewards, dtype=np.float32).reshape(-1)
        dones = np.asarray(step.dones, dtype=bool).reshape(-1)

        returns += rewards
        lengths += 1
        for env_i in range(int(executor.n_envs)):
            if dones[env_i]:
                info_i = step.infos[env_i] if env_i < len(step.infos) else {}
                successes.append(
                    float(bool(info_i.get("is_success", info_i.get("success", False))))
                )
                ep_returns.append(float(returns[env_i]))
                ep_lengths.append(float(lengths[env_i]))
                returns[env_i] = 0.0
                lengths[env_i] = 0
                action_queues[env_i].clear()
                if len(successes) >= int(eval_episodes):
                    break

    return {
        "success": float(np.mean(successes)) if successes else 0.0,
        "return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        "episodes": float(len(successes)),
    }


def main() -> None:
    args = build_parser().parse_args()
    torch.set_float32_matmul_precision(str(args.matmul_precision))
    runtime_meta = validate_runtime(
        args.dsrl_official_root, args.required_robosuite_version, args.required_mujoco_version
    )
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lowdim_keys = parse_lowdim_keys(args.lowdim_keys)
    lowdim_dims = infer_lowdim_dims_from_stats(args.stats_npz, lowdim_keys)
    action_dim = infer_action_dim_from_stats(args.stats_npz)
    obs_dim = int(sum(int(lowdim_dims[k]) for k in lowdim_keys))

    policy_wrapper, ckpt_meta = load_dsrl_policy_wrapper(
        ckpt_path=args.agent,
        device=device,
        stats_npz=args.stats_npz,
        lowdim_keys=lowdim_keys,
        lowdim_dims=lowdim_dims,
        action_dim=action_dim,
        cond_steps=int(args.cond_steps),
        weight_source=args.weight_source,
    )
    actor = policy_wrapper.policy.noise_pred_net
    actor.train()
    horizon_steps = int(policy_wrapper.policy.horizon_steps)
    update_schedule = OGPOUpdateSchedule(
        args.utd, args.start_training, args.batch_size, horizon_steps
    )
    args.utd = float(update_schedule.rate)
    if int(args.expected_horizon_steps) > 0 and horizon_steps != int(args.expected_horizon_steps):
        raise ValueError(
            f"checkpoint horizon mismatch: expected {args.expected_horizon_steps}, got {horizon_steps}"
        )
    schedule_defaults = ckpt_meta.get("schedule_defaults", {})
    train_denoising_steps = int(
        args.train_denoising_steps or schedule_defaults.get("train_denoising_steps", 5)
    )
    eval_ddim_steps = int(
        args.eval_ddim_steps
        or schedule_defaults.get("default_inference_steps", train_denoising_steps)
    )

    cfg = DPNativeOGPOConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        horizon_steps=horizon_steps,
        cond_steps=int(args.cond_steps),
        discount=float(args.discount),
        tau=float(args.tau),
        actor_tau=float(args.actor_tau),
        num_qs=int(args.num_qs),
        q_agg=str(args.q_agg),
        critic_hidden_dims=tuple(_parse_ints(args.critic_hidden_dims)),
        ppo_lr=float(args.ppo_lr),
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        bc_coeff=float(args.bc_coeff),
        update_mode=str(args.update_mode),
        group_num_samples=int(args.group_num_samples),
        actor_microbatch_size=int(args.actor_microbatch_size),
        inference_microbatch_size=int(args.inference_microbatch_size),
        fuse_log_prob_steps=bool(args.fuse_log_prob_steps),
        vectorized_critic=bool(args.vectorized_critic),
        compile_models=bool(args.compile_models),
        compile_mode=str(args.compile_mode),
        clip_epsilon=float(args.clip_epsilon),
        best_of_n=int(args.best_of_n),
        train_denoising_steps=train_denoising_steps,
        eval_ddim_steps=eval_ddim_steps,
        ft_denoising_steps=int(args.ft_denoising_steps),
        use_ogpo_sde=bool(args.use_ogpo_sde),
        use_constant_noise=bool(args.use_constant_noise),
        use_tapered_noise=bool(args.use_tapered_noise),
        error_correct_sde_to_ode=bool(args.error_correct_sde_to_ode),
        constant_noise_std=float(args.constant_noise_std),
        min_noise_std=float(args.min_noise_std),
        denoised_clip_value=float(args.denoised_clip_value),
        randn_clip_value=float(args.randn_clip_value),
        final_action_clip_value=float(args.final_action_clip_value),
    )
    agent = DPNativeOGPOAgent(actor, cfg, device=device)
    run_dir = _make_run_dir(args)
    resume_checkpoint = None
    resume_training_state = {}
    resume_progress = {}
    resume_schedule_source = None
    resume_rng_state = None
    if args.resume_checkpoint:
        resume_checkpoint = torch.load(
            os.path.abspath(os.path.expanduser(args.resume_checkpoint)),
            map_location="cpu",
        )
        resume_step = int(resume_checkpoint["step"])
        if resume_step >= int(args.online_steps):
            raise ValueError(
                f"resume step {resume_step} must be below online_steps={int(args.online_steps)}"
            )
        _validate_resume_config(resume_checkpoint["agent"]["config"], cfg)
        agent.load_state_dict(resume_checkpoint["agent"])
        replay = resume_checkpoint["replay"]
        if (int(replay.obs_dim), int(replay.action_dim), int(replay.capacity)) != (
            obs_dim,
            action_dim,
            int(args.buffer_size),
        ):
            raise ValueError(
                "resume replay mismatch: "
                f"checkpoint={(replay.obs_dim, replay.action_dim, replay.capacity)}, "
                f"current={(obs_dim, action_dim, int(args.buffer_size))}"
            )
        replay.device = device
        resume_schedule_source = _restore_update_schedule(
            update_schedule, resume_checkpoint, run_dir
        )
        resume_training_state = resume_checkpoint.get("training_state", {})
        resume_progress = _last_train_progress(run_dir, resume_step)
        resume_rng_state = resume_training_state.get("rng")
    else:
        replay = OGPOSequenceReplayBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            capacity=int(args.buffer_size),
            device=device,
            seed=int(args.seed) + 30000,
        )

    executor = RobomimicExecutor(
        vendor_root=args.dsrl_official_root,
        env_name=args.env_name,
        env_meta_json=args.env_meta_json,
        normalization_path=args.normalization_path,
        lowdim_keys=lowdim_keys,
        n_envs=int(args.n_envs),
        max_episode_steps=int(args.max_episode_steps),
        seed=int(args.seed),
        asynchronous=True,
        terminate_on_success=bool(args.terminate_on_success),
        timeout_on_horizon=bool(args.timeout_on_horizon),
    )
    eval_executor = None
    if int(args.eval_interval) > 0 and int(args.eval_episodes) > 0:
        eval_executor = RobomimicExecutor(
            vendor_root=args.dsrl_official_root,
            env_name=args.env_name,
            env_meta_json=args.env_meta_json,
            normalization_path=args.normalization_path,
            lowdim_keys=lowdim_keys,
            n_envs=int(args.n_eval_envs),
            max_episode_steps=int(args.max_episode_steps),
            seed=int(args.seed) + 50000,
            asynchronous=True,
            terminate_on_success=bool(args.terminate_on_success),
            timeout_on_horizon=bool(args.timeout_on_horizon),
        )

    meta = {
        "args": dict(vars(args)),
        "ckpt_meta": ckpt_meta,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "horizon_steps": horizon_steps,
        "config": cfg.__dict__,
        "runtime": runtime_meta,
        "update_schedule": {
            "unit": "eligible_vector_step",
            "numerator": update_schedule.rate.numerator,
            "denominator": update_schedule.rate.denominator,
            "min_replay_size": update_schedule.min_replay_size,
            "updates_per_env_step": float(update_schedule.rate) / int(args.n_envs),
        },
    }
    if resume_checkpoint is not None:
        old_meta_path = os.path.join(run_dir, "run_meta.json")
        old_meta = json.load(open(old_meta_path)) if os.path.isfile(old_meta_path) else {}
        resume_history = list(old_meta.get("resume_history", []))
        resume_history.append(
            {
                "checkpoint": os.path.abspath(os.path.expanduser(args.resume_checkpoint)),
                "source_step": int(resume_checkpoint["step"]),
                "target_step": int(args.online_steps),
                "schedule_source": str(resume_schedule_source),
                "rng_source": "checkpoint" if resume_rng_state is not None else "derived_legacy",
                "environment_state_restored": False,
                "wall_time_unix": time.time(),
            }
        )
        meta["initial_args"] = old_meta.get("initial_args", old_meta.get("args", {}))
        meta["resume_history"] = resume_history
    _write_json(os.path.join(run_dir, "run_meta.json"), meta)
    if resume_checkpoint is None and int(args.eval_snapshot_interval) > 0:
        snapshot_path = _save_eval_snapshot(run_dir, agent, 0, meta)
        print(
            json.dumps(
                {"phase": "eval_snapshot", "env_steps": 0, "path": snapshot_path},
                sort_keys=True,
            ),
            flush=True,
        )

    env_steps = int(resume_checkpoint["step"]) if resume_checkpoint is not None else 0
    states = executor.reset_all(base_seed=int(args.seed) + 10000 + env_steps)
    if resume_rng_state is not None:
        _restore_rng_state(resume_rng_state)
    elif resume_checkpoint is not None:
        legacy_seed = int(args.seed) + 900000 + env_steps
        random.seed(legacy_seed)
        np.random.seed(legacy_seed % (2**32))
        torch.manual_seed(legacy_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(legacy_seed)
    action_queues: List[deque] = [deque() for _ in range(int(args.n_envs))]
    traj_buffers: List[List[Dict]] = [[] for _ in range(int(args.n_envs))]
    recent_successes = deque(resume_training_state.get("recent_successes", []), maxlen=100)
    recent_returns = deque(resume_training_state.get("recent_returns", []), maxlen=100)
    recent_lengths = deque(resume_training_state.get("recent_lengths", []), maxlen=100)
    episode_returns = np.zeros((int(args.n_envs),), dtype=np.float32)
    episode_lengths = np.zeros((int(args.n_envs),), dtype=np.int64)

    updates = (
        int(resume_training_state.get("updates", agent.update_step)) if resume_checkpoint else 0
    )
    if resume_checkpoint is not None and updates != int(agent.update_step):
        raise ValueError(
            f"resume update counter mismatch: training_state={updates}, agent={agent.update_step}"
        )
    skipped_updates = int(
        resume_training_state.get("skipped_updates", resume_progress.get("skipped_updates", 0))
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    last_log_time = training_started
    last_log_env_steps = env_steps
    policy_seconds = env_seconds = replay_seconds = update_seconds = 0.0
    last_update = dict(resume_progress.get("last_update", {}))
    success_batch_updates = int(
        resume_training_state.get(
            "success_batch_updates", resume_progress.get("success_batch_updates", 0)
        )
    )
    rollout_policy = _effective_rollout_policy(str(args.update_mode), str(args.rollout_policy))
    eval_sde_best_of_n = 1 if rollout_policy in {"sde", "deterministic"} else int(args.best_of_n)
    if resume_checkpoint is not None:
        resume_event = {
            "phase": "resume",
            "env_steps": int(env_steps),
            "target_env_steps": int(args.online_steps),
            "updates": int(updates),
            "eligible_vector_steps": int(update_schedule.eligible_steps),
            "schedule_credit": int(update_schedule.credit),
            "schedule_source": str(resume_schedule_source),
            "rng_source": "checkpoint" if resume_rng_state is not None else "derived_legacy",
            "environment_state_restored": False,
            "replay_size": int(len(replay)),
            "wall_time_unix": time.time(),
        }
        print(json.dumps(resume_event, sort_keys=True), flush=True)
        _append_jsonl(os.path.join(run_dir, "metrics.jsonl"), resume_event)
        del resume_checkpoint
    try:
        while env_steps < int(args.online_steps):
            section_start = time.perf_counter()
            empty = [i for i, q in enumerate(action_queues) if len(q) == 0]
            if empty:
                chunks = _select_action_chunks(
                    agent, states[empty], rollout_policy, int(args.best_of_n)
                )
                _enqueue_full_action_chunks(action_queues, empty, chunks, horizon_steps)

            policy_seconds += time.perf_counter() - section_start
            section_start = time.perf_counter()

            actions = np.stack(
                [action_queues[i].popleft() for i in range(int(args.n_envs))], axis=0
            ).astype(np.float32)
            step = executor.step_variable_k(
                action_chunks=actions[:, None, :],
                exec_lens=np.ones((int(args.n_envs),), dtype=np.int64),
                gamma=float(args.discount),
                reward_offset=float(args.reward_offset),
                discount_within_chunk=True,
            )
            next_states = np.asarray(step.next_state, dtype=np.float32)
            rewards = np.asarray(step.rewards, dtype=np.float32).reshape(-1)
            dones = np.asarray(step.dones, dtype=bool).reshape(-1)
            env_seconds += time.perf_counter() - section_start
            section_start = time.perf_counter()

            for env_i in range(int(args.n_envs)):
                info_i = step.infos[env_i] if env_i < len(step.infos) else {}
                is_success = float(bool(info_i.get("is_success", info_i.get("success", False))))
                if is_success > 0:
                    rewards[env_i] += float(args.success_bonus)
                    dones[env_i] = True

                truncated = bool(info_i.get("TimeLimit.truncated", False))
                terminated = bool(dones[env_i]) and not truncated
                transition_next_state = next_states[env_i].copy()
                if bool(dones[env_i]):
                    transition_next_state = _terminal_state_from_info(info_i, transition_next_state)
                transition = {
                    "observation": states[env_i].copy(),
                    "action": actions[env_i].copy(),
                    "reward": float(rewards[env_i]),
                    "terminal": float(dones[env_i]),
                    "mask": 1.0 - float(terminated),
                    "next_observation": transition_next_state,
                }
                traj_buffers[env_i].append(transition)
                episode_returns[env_i] += float(rewards[env_i])
                episode_lengths[env_i] += 1

                if dones[env_i]:
                    score = is_success if is_success != 0 else 1e-9
                    for tr in traj_buffers[env_i]:
                        replay.add_transition(
                            tr["observation"],
                            tr["action"],
                            tr["reward"],
                            tr["terminal"],
                            tr["mask"],
                            tr["next_observation"],
                            is_success=is_success,
                            score=score,
                        )
                    recent_successes.append(is_success)
                    recent_returns.append(float(episode_returns[env_i]))
                    recent_lengths.append(float(episode_lengths[env_i]))
                    traj_buffers[env_i] = []
                    episode_returns[env_i] = 0.0
                    episode_lengths[env_i] = 0
                    action_queues[env_i].clear()

            states = next_states
            env_steps += int(args.n_envs)
            replay_seconds += time.perf_counter() - section_start

            updates_due = update_schedule.advance(env_steps, len(replay))
            if updates_due:
                section_start = time.perf_counter()
                for _ in range(updates_due):
                    batch = replay.sample_sequence(
                        int(args.batch_size), horizon_steps, float(args.discount)
                    )
                    success_batch = None
                    success_flag = False
                    try:
                        success_batch = replay.sample_sequence(
                            int(args.success_buffer_batch_size),
                            horizon_steps,
                            float(args.discount),
                            success_only=True,
                        )
                        success_flag = True
                    except ValueError:
                        success_batch = batch
                    if str(args.update_mode) == "bc_only":
                        if not success_flag and bool(args.bc_only_success_only):
                            skipped_updates += 1
                            continue
                        info = agent.bc_only_update(success_batch)
                    else:
                        info = agent.batch_update(
                            batch, success_batch=success_batch, success_flag=success_flag
                        )
                    last_update = info
                    success_batch_updates += int(success_flag)
                    updates += 1
                if args.profile_timing and device.type == "cuda":
                    torch.cuda.synchronize(device)
                update_seconds += time.perf_counter() - section_start

            if int(args.log_interval) > 0 and env_steps % int(args.log_interval) < int(args.n_envs):
                now = time.perf_counter()
                payload = {
                    "env_steps": int(env_steps),
                    "updates": int(updates),
                    "skipped_updates": int(skipped_updates),
                    "replay_size": int(len(replay)),
                    "update_mode": str(args.update_mode),
                    "rollout_policy": rollout_policy,
                    "recent_success_rate": float(np.mean(recent_successes))
                    if recent_successes
                    else 0.0,
                    "recent_return": float(np.mean(recent_returns)) if recent_returns else 0.0,
                    "recent_length": float(np.mean(recent_lengths)) if recent_lengths else 0.0,
                    "wall_time_unix": time.time(),
                    "elapsed_seconds": now - training_started,
                    "interval_env_steps_per_second": (env_steps - last_log_env_steps)
                    / (now - last_log_time),
                    "policy_seconds": policy_seconds,
                    "env_seconds": env_seconds,
                    "replay_write_seconds": replay_seconds,
                    "update_seconds": update_seconds,
                    "eligible_vector_steps": update_schedule.eligible_steps,
                    "utd": args.utd,
                    "updates_per_env_step": float(update_schedule.rate) / int(args.n_envs),
                    "last_update": last_update,
                    "success_batch_updates": success_batch_updates,
                }
                print(json.dumps(payload, sort_keys=True))
                _append_jsonl(os.path.join(run_dir, "metrics.jsonl"), {"phase": "train", **payload})
                last_log_time = now
                last_log_env_steps = env_steps

            if (
                eval_executor is not None
                and int(args.eval_interval) > 0
                and env_steps % int(args.eval_interval) < int(args.n_envs)
            ):
                eval_payload = _evaluate_executor(
                    agent,
                    eval_executor,
                    deterministic=True,
                    best_of_n=int(args.best_of_n),
                    eval_episodes=int(args.eval_episodes),
                    seed=int(args.seed) + 60000 + int(env_steps),
                )
                eval_payload = {"phase": "eval", "env_steps": int(env_steps), **eval_payload}
                print(json.dumps(eval_payload, sort_keys=True))
                _append_jsonl(os.path.join(run_dir, "metrics.jsonl"), eval_payload)
                if bool(args.eval_sde):
                    eval_sde_payload = _evaluate_executor(
                        agent,
                        eval_executor,
                        deterministic=False,
                        best_of_n=eval_sde_best_of_n,
                        eval_episodes=int(args.eval_episodes),
                        seed=int(args.seed) + 70000 + int(env_steps),
                    )
                    eval_sde_payload = {
                        "phase": "eval_sde",
                        "env_steps": int(env_steps),
                        **eval_sde_payload,
                    }
                    print(json.dumps(eval_sde_payload, sort_keys=True))
                    _append_jsonl(os.path.join(run_dir, "metrics.jsonl"), eval_sde_payload)

            if int(args.eval_snapshot_interval) > 0 and env_steps % int(
                args.eval_snapshot_interval
            ) < int(args.n_envs):
                snapshot_path = _save_eval_snapshot(run_dir, agent, env_steps, meta)
                print(
                    json.dumps(
                        {
                            "phase": "eval_snapshot",
                            "env_steps": int(env_steps),
                            "path": snapshot_path,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            if int(args.save_interval) > 0 and env_steps % int(args.save_interval) < int(
                args.n_envs
            ):
                _save_checkpoint(
                    run_dir,
                    agent,
                    replay,
                    env_steps,
                    meta,
                    _training_state(
                        update_schedule,
                        updates=updates,
                        skipped_updates=skipped_updates,
                        success_batch_updates=success_batch_updates,
                        recent_successes=recent_successes,
                        recent_returns=recent_returns,
                        recent_lengths=recent_lengths,
                    ),
                )
    finally:
        _save_checkpoint(
            run_dir,
            agent,
            replay,
            env_steps,
            meta,
            _training_state(
                update_schedule,
                updates=updates,
                skipped_updates=skipped_updates,
                success_batch_updates=success_batch_updates,
                recent_successes=recent_successes,
                recent_returns=recent_returns,
                recent_lengths=recent_lengths,
            ),
        )
        executor.close()
        if eval_executor is not None:
            eval_executor.close()


if __name__ == "__main__":
    main()
