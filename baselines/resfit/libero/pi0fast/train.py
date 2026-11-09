"""Train fixed-Q50 ResFiT on audited LIBERO-90 task57 and frozen pi0-FAST.

The stage order, update scheduler and TD3 algorithm come from the formal DP
ResFiT run. Task observations/augmentation and base model follow the formal
task57-Q50 pi0-FAST ExRL-Q contract.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import time

# Set before importing either framework; respect the launcher's GPU selection.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

from baselines.resfit.libero.pi0fast.runtime import (
    configure_runtime,
    make_env,
    load_policy,
    variant,
    Pi0FastPlanner,
    LiberoResidualEnv,
    isolated_rng,
    runtime_manifest,
    sha256_file,
    disable_tensorflow_gpu_if_available,
)
from baselines.resfit.libero.pi0fast.model import PixelResFiTLearner, ActionCoordinates
from baselines.resfit.libero.pi0fast.replay import PixelReplayBuffer, sample_training_batch
from baselines.resfit.libero.pi0fast.offline import build_offline_replay
from baselines.resfit.libero.pi0fast.collector import SerialCollector, ParallelCollector
from baselines.resfit.libero.pi0fast.evaluation import (
    AsyncSerialEvaluator,
    AsyncEpisodeEvaluator,
    TraceEnv,
)
from baselines.resfit.dp.td3_core import ResFiTLearnerConfig, TransitionUpdateScheduler


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-root", default=None)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--demo-hdf5")
    p.add_argument("--offline-cache-dir")
    p.add_argument("--data-mode", choices=("mixed", "online"), default="mixed")
    p.add_argument("--mode", choices=("train", "eval", "cache", "base-check"), default="train")
    p.add_argument("--resume")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--eval-gpu", type=int, help="physical GPU for persistent evaluation processes")
    p.add_argument(
        "--eval-gpus",
        type=int,
        nargs="+",
        help="physical GPUs for episode-seeded evaluators; workers are assigned round-robin",
    )
    p.add_argument(
        "--eval-workers",
        type=int,
        help="split each snapshot's episodes across processes with fixed per-episode seeds",
    )
    p.add_argument(
        "--eval-trace",
        action="store_true",
        help="validation hashes of every evaluation action and sim state",
    )
    p.add_argument("--learning-starts", type=int, default=2048)
    p.add_argument("--critic-warmup-updates", type=int, default=2000)
    p.add_argument("--train-decisions", type=int, default=100000000)
    p.add_argument(
        "--train-env-steps",
        type=int,
        default=3000000,
        help="post-warmup primitive step budget, as in DP ResFiT",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--online-replay-capacity", type=int, default=60000)
    p.add_argument("--offline-replay-capacity", type=int, default=3000)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=30000)
    p.add_argument("--checkpoint-interval", type=int, default=100000)
    p.add_argument("--max-demo-episodes", type=int, default=0)
    p.add_argument(
        "--test-mode",
        action="store_true",
        help="mark reduced-budget validation, never a formal run",
    )
    return p


def evaluation_layout(args):
    if args.eval_gpu is not None and args.eval_gpus is not None:
        raise ValueError("eval-gpu and eval-gpus are mutually exclusive")
    gpus = (
        list(args.eval_gpus)
        if args.eval_gpus is not None
        else ([args.eval_gpu] if args.eval_gpu is not None else [])
    )
    workers = args.eval_workers
    if args.eval_gpus is not None and workers is None:
        workers = len(gpus)
    return gpus, workers


def validate_args(args):
    for name in (
        "learning_starts",
        "train_decisions",
        "train_env_steps",
        "batch_size",
        "online_replay_capacity",
        "offline_replay_capacity",
        "eval_episodes",
        "eval_interval",
        "checkpoint_interval",
        "num_envs",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(name + " must be positive")
    if args.critic_warmup_updates < 0 or args.max_demo_episodes < 0:
        raise ValueError("negative warmup or demo limit")
    eval_gpus, eval_workers = evaluation_layout(args)
    if any(gpu < 0 for gpu in eval_gpus) or len(set(eval_gpus)) != len(eval_gpus):
        raise ValueError("evaluation GPUs must be distinct nonnegative physical indices")
    if eval_workers is not None:
        if eval_workers <= 0 or not eval_gpus or args.mode == "cache":
            raise ValueError(
                "positive eval-workers requires evaluation GPU(s) and an evaluation-capable mode"
            )
    if args.data_mode == "mixed" and not args.demo_hdf5:
        raise ValueError(
            "mixed mode requires the task57 demonstration HDF5; no rollout substitution"
        )
    if args.data_mode == "online" and args.demo_hdf5:
        raise ValueError("online mode must not be given a demonstration dataset")
    if args.data_mode == "mixed" and args.batch_size % 2:
        raise ValueError("mixed batch size must be even")
    required = args.batch_size // 2 if args.data_mode == "mixed" else args.batch_size
    if min(args.learning_starts, args.online_replay_capacity) < required:
        raise ValueError("warmup/replay is too small for the batch")
    if args.mode == "eval" and not args.resume:
        raise ValueError("eval requires --resume; use base-check to test the zero actor")
    if args.resume and args.mode not in ("eval", "train"):
        raise ValueError("resume is only valid for train/eval")
    if args.mode == "cache" and args.data_mode != "mixed":
        raise ValueError("cache mode requires mixed data")


def source_hashes():
    return {"replay_schema": 1}


OFFLINE_CACHE_SOURCE_FILES = ("replay_schema",)


def offline_cache_provenance(contract, *, demo_sha256, seed, capacity, max_demo_episodes):
    """Keep reusable demo replay independent of learner/evaluator settings."""
    source_files = contract["source_files"]
    selected = {name: source_files[name] for name in OFFLINE_CACHE_SOURCE_FILES}
    return {
        "schema": "libero_task57_pi0fast_offline_replay_v1",
        "task": contract["task"],
        "data_mode": contract["data_mode"],
        "action_coordinates": contract["action_coordinates"],
        "runtime": contract["runtime"],
        "source_files": selected,
        "demo_sha256": demo_sha256,
        "seed": int(seed),
        "capacity": int(capacity),
        "max_demo_episodes": int(max_demo_episodes),
    }


def capture_rng(policy):
    import jax

    return {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "policy": np.asarray(jax.random.key_data(policy._rng)),
    }


def restore_rng(policy, rng):
    import jax

    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    policy._rng = jax.random.wrap_key_data(rng["policy"])


def evaluate(adapter, learner, episodes, seed, *, trace=False):
    success, lengths, errors, traces = [], [], [], []
    with isolated_rng(adapter.planner.policy, seed):
        for _ in range(episodes):
            adapter.reset()
            length = 0
            while True:
                obs, base = adapter.act_input()
                residual = learner.act(obs, base, eval_mode=True)[0]
                _, result = adapter.step(residual)
                length += result["duration_k"]
                if result["max_zero_residual_command_error"] is not None:
                    errors.append(result["max_zero_residual_command_error"])
                if result["episode_done"]:
                    success.append(float(result["success"]))
                    lengths.append(length)
                    if trace:
                        traces.append(adapter.env.trace())
                        if len(traces) % 10 == 0:
                            print(
                                json.dumps(
                                    {
                                        "evaluation_episode": len(traces),
                                        "episodes": episodes,
                                        "successes": int(sum(success)),
                                    }
                                ),
                                flush=True,
                            )
                    break
    metrics = {
        "env_step_eval/success_rate": float(np.mean(success)),
        "env_step_eval/episodes": episodes,
        "env_step_eval/episode_length_mean": float(np.mean(lengths)),
        "env_step_eval/zero_residual_command_error": max(errors) if errors else None,
    }
    if trace:
        metrics["evaluation_trace"] = traces
    return metrics


def check_zero_equivalence(adapter, learner, seed):
    """Compare one original pi0-FAST macro step with the actual residual path."""
    from baselines.resfit.libero.pi0fast.reference import (
        _execute_offline_action_chunk,
        env_actions_to_conditioning_actions,
    )

    policy = adapter.planner.policy
    with isolated_rng(policy, seed):
        raw = adapter.env.reset()
        plan = adapter.planner.plan(raw)
        adapter.settings._episode_env_steps = 0
        baseline = _execute_offline_action_chunk(
            adapter.env,
            raw,
            plan,
            env_actions_to_conditioning_actions(raw, plan, adapter.settings),
            50,
            [],
            adapter.settings,
            0,
        )
        reference_sim = adapter.env.sim.get_state().flatten().copy()
    with isolated_rng(policy, seed):
        adapter.reset()
        np.testing.assert_array_equal(adapter.base_raw, plan)
        obs, base = adapter.act_input()
        residual = learner.act(obs, base, eval_mode=True)[0]
        np.testing.assert_array_equal(residual, np.zeros(350, np.float32))
        _, actual = adapter.step(residual)
        actual_sim = adapter.env.sim.get_state().flatten().copy()
        np.testing.assert_array_equal(actual_sim, reference_sim)
        for key in ("reward", "duration_k", "success", "done_for_backup", "episode_done"):
            if actual[key] != baseline[key]:
                raise AssertionError(f"zero-residual baseline mismatch: {key}")
    return {
        "passed": True,
        "primitive_steps": actual["duration_k"],
        "max_sim_state_error": float(np.max(np.abs(reference_sim - actual_sim))),
        "max_command_error": actual["max_zero_residual_command_error"],
        "base_action_max_abs": float(np.max(np.abs(plan))),
    }


def planner_timing_delta(planner, before):
    """Return successful policy inference work since a prior snapshot."""
    after = planner.timing_snapshot()
    delta = {
        "calls": after["calls"] - before["calls"],
        "seconds": after["seconds"] - before["seconds"],
    }
    if delta["calls"] < 0 or delta["seconds"] < 0:
        raise RuntimeError("pi0-FAST planner timing counters moved backwards")
    return delta


def add_policy_timing(bucket, delta):
    bucket["policy_inference_calls"] += int(delta["calls"])
    bucket["policy_inference_seconds"] += float(delta["seconds"])


def run(args):
    validate_args(args)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        raise FileExistsError("use a new output directory, including for resumes: " + str(out))
    root = configure_runtime(args.runtime_root, out / "libero_config")
    disable_tensorflow_gpu_if_available()
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    policy, norm_stats = load_policy(args.checkpoint_dir)
    import jax

    policy._rng = jax.random.key(args.seed)
    coordinates = ActionCoordinates(0.1)
    learner_config = ResFiTLearnerConfig(
        state_dim=8,
        action_dim=350,
        actor_hidden_dims=(1024, 1024, 1024),
        critic_hidden_dims=(512, 512, 512),
        critic_ensemble_size=10,
        target_critic_subsample_size=2,
        critic_use_layernorm=True,
        action_scale=0.1,
        actor_lr=5e-6,
        critic_lr=1e-4,
        tau=0.01,
        actor_grad_clip=1.0,
        critic_grad_clip=1.0,
        target_noise_std=0.01,
        target_noise_clip=0.3,
        td_loss_mode="per_head_mse",
    )
    learner = PixelResFiTLearner(learner_config, norm_stats, args.device, seed=args.seed)
    planner = Pi0FastPlanner(policy, variant(norm_stats))
    manifest = {
        "schema": 1,
        "hostname": socket.gethostname(),
        "test_mode": args.test_mode,
        "algorithm": "ResFiT_fixed_chunk_TD3",
        "learner": asdict(learner_config),
        "data_mode": args.data_mode,
        "offline_fraction": 0.5 if args.data_mode == "mixed" else 0.0,
        "updates_per_transition": {"critic": 1.0, "actor": 0.25},
        "action_coordinates": coordinates.manifest(),
        "source_files": source_hashes(),
        "runtime": runtime_manifest(root, Path(args.checkpoint_dir)),
        "task": {
            "suite": "libero_90",
            "id": 57,
            "horizon": 250,
            "gamma": 0.997,
            "reward_offset": 1.0,
            "query_freq": 50,
            "libero_hard_reset": False,
            "base_policy_max_decoding_steps": 320,
            "base_policy_temperature": 0.0,
        },
        "algorithm_exclusions": [
            "variable_k",
            "cql",
            "entropy_alpha",
            "bc_actor_loss",
            "q_target_clip",
        ],
        "resume_rollout_policy": "restore_replay_optimizers_and_rng_then_reset_environment",
        "augmentation": "verbatim_formal_pi_crop4_color_jitter_current_and_next",
    }

    def write_manifest():
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def log(step, metrics):
        row = {"step": int(step), "wall_time": time.time(), **metrics}
        with (out / "training_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    write_manifest()
    training = evaluation = demo_env = async_evaluation = None
    online, offline = PixelReplayBuffer(args.online_replay_capacity, args.seed), None
    scheduler = TransitionUpdateScheduler(1.0, 0.25)
    counters = {
        "total_env_steps": 0,
        "warmup_decisions": 0,
        "train_decisions": 0,
        "train_env_steps": 0,
        "critic_warmup_updates": 0,
        "critic_updates": 0,
        "actor_updates": 0,
        "episodes": 0,
        "successes": 0,
    }
    # Contract is independent of output path, budgets, and execution mode.
    contract = {
        k: manifest[k]
        for k in ("learner", "task", "data_mode", "action_coordinates", "runtime", "source_files")
    }
    contract.update(
        {
            "batch_size": args.batch_size,
            "learning_starts": args.learning_starts,
            "critic_warmup_updates": args.critic_warmup_updates,
            "max_demo_episodes": args.max_demo_episodes,
            "test_mode": args.test_mode,
        }
    )
    contract = json.loads(json.dumps(contract))
    try:
        with isolated_rng(policy):
            eval_env = make_env(args.seed + 123)
            evaluation = LiberoResidualEnv(
                TraceEnv(eval_env) if args.eval_trace else eval_env,
                planner,
                coordinates,
                norm_stats,
                bootstrap_timeouts=False,
            )
        if args.mode in ("train", "cache") and args.data_mode == "mixed":
            offline = PixelReplayBuffer(args.offline_replay_capacity, args.seed + 1)
            cache_provenance = offline_cache_provenance(
                contract,
                demo_sha256=sha256_file(args.demo_hdf5),
                seed=args.seed,
                capacity=args.offline_replay_capacity,
                max_demo_episodes=args.max_demo_episodes,
            )
            cache_key = hashlib.sha256(
                json.dumps(cache_provenance, sort_keys=True).encode()
            ).hexdigest()[:20]
            cache_dir = Path(args.offline_cache_dir or out / "offline_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"seed{args.seed}_{cache_key}.pt"
            with isolated_rng(policy, args.seed + 20000):
                if cache_path.exists():
                    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                    if cached["provenance"] != cache_provenance:
                        raise ValueError("offline cache provenance mismatch")
                    offline.load_state_dict(cached["replay"])
                    summary = cached["summary"]
                else:
                    cached = None
                    for legacy_path in sorted(cache_dir.glob(f"seed{args.seed}_*.pt")):
                        legacy = torch.load(legacy_path, map_location="cpu", weights_only=False)
                        legacy_provenance = legacy.get("provenance", {})
                        try:
                            normalized = offline_cache_provenance(
                                legacy_provenance,
                                demo_sha256=legacy_provenance["demo_sha256"],
                                seed=legacy_provenance["seed"],
                                capacity=legacy_provenance["capacity"],
                                max_demo_episodes=legacy_provenance["max_demo_episodes"],
                            )
                        except KeyError:
                            continue
                        if normalized == cache_provenance:
                            cached = {**legacy, "provenance": cache_provenance}
                            torch.save(cached, cache_path)
                            manifest["offline_cache_migrated_from"] = str(legacy_path)
                            break
                    if cached is not None:
                        offline.load_state_dict(cached["replay"])
                        summary = cached["summary"]
                    else:
                        demo_env = LiberoResidualEnv(
                            make_env(args.seed + 20000), planner, coordinates, norm_stats
                        )
                        summary = build_offline_replay(
                            args.demo_hdf5, demo_env, offline, max_episodes=args.max_demo_episodes
                        )
                        torch.save(
                            {
                                "provenance": cache_provenance,
                                "replay": offline.state_dict(),
                                "summary": summary,
                            },
                            cache_path,
                        )
            if len(offline) < args.batch_size // 2:
                raise ValueError("demo replay too small for a half batch")
            manifest["offline_summary"] = summary
            manifest["offline_cache"] = str(cache_path)
            contract["demo_sha256"] = summary["sha256"]
            write_manifest()
        if args.mode == "cache":
            return out
        # Collection/evaluation execution does not change demonstration replay.
        eval_gpus, eval_workers = evaluation_layout(args)
        contract["collector"] = {
            "num_envs": args.num_envs,
            "protocol": "serial"
            if args.num_envs == 1
            else "sync_env_spawn_serial_policy_pi0fast_v1",
        }
        contract["evaluation_mode"] = "serial_inline" if not eval_gpus else "serial_worker"
        if eval_workers is not None:
            contract["evaluation_mode"] = "episode_seed_workers_v1"
        manifest["collector"] = contract["collector"]
        manifest["evaluation_execution"] = {
            "mode": contract["evaluation_mode"],
            "gpu": eval_gpus[0] if len(eval_gpus) == 1 else None,
            "gpus": eval_gpus,
            "workers": eval_workers or 1,
            "worker_gpus": (
                [eval_gpus[index % len(eval_gpus)] for index in range(eval_workers)]
                if eval_workers is not None
                else eval_gpus
            ),
            "seed_protocol": "seed_plus_10000000_plus_episode_id"
            if eval_workers is not None
            else "continuous_serial_rng",
        }
        write_manifest()
        restored = None
        if args.resume:
            restored = torch.load(args.resume, map_location="cpu", weights_only=False)
            if args.mode == "eval":
                # Evaluating a mixed-policy checkpoint needs no demo replay.
                for k in ("learner", "task", "action_coordinates", "runtime", "source_files"):
                    if restored["contract"][k] != contract[k]:
                        raise ValueError("checkpoint contract changed: " + k)
            elif restored["contract"] != contract:
                raise ValueError(
                    "resume contract changed (algorithm, source, task, runtime or data)"
                )
            learner.load_state_dict(restored["learner"])
            if args.mode == "train":
                online.load_state_dict(restored["online"])
                scheduler.load_state_dict(restored["scheduler"])
            counters = restored["counters"]
            if offline is not None:
                offline.rng.set_state(restored["offline_rng"])
            restore_rng(policy, restored["rng"])
            manifest["resume_from"] = str(Path(args.resume).resolve())
            write_manifest()
        if not args.resume:
            manifest["zero_residual_equivalence"] = check_zero_equivalence(
                evaluation, learner, args.seed + 10_000_000
            )
            write_manifest()

        def record_evaluations(results):
            for result in results:
                job = result["job"]
                log(
                    job["step"],
                    {
                        **result["metrics"],
                        "eval_trigger_step": job["trigger"],
                        "evaluation_seconds": result["seconds"],
                        "evaluation_snapshot": job["snapshot"],
                    },
                )

        if eval_gpus:
            evaluator_class = (
                AsyncSerialEvaluator if eval_workers is None else AsyncEpisodeEvaluator
            )
            evaluator_options = (
                {"gpu": eval_gpus[0]}
                if eval_workers is None
                else {"gpus": eval_gpus, "workers": eval_workers}
            )
            async_evaluation = evaluator_class(
                runtime_root=root,
                checkpoint_dir=args.checkpoint_dir,
                learner_config=asdict(learner_config),
                seed=args.seed,
                output=out / "evaluation_snapshots",
                episodes=args.eval_episodes,
                trace=args.eval_trace,
                **evaluator_options,
            )
            if restored is not None and args.mode == "train":
                async_evaluation.restore_pending(restored.get("pending_evaluations", []))

        def run_evaluation(trigger):
            if async_evaluation is not None:
                async_evaluation.submit(learner.actor, counters["total_env_steps"], trigger)
                record_evaluations(async_evaluation.poll())
            else:
                metrics = evaluate(
                    evaluation,
                    learner,
                    args.eval_episodes,
                    args.seed + 10_000_000,
                    trace=args.eval_trace,
                )
                log(counters["total_env_steps"], {**metrics, "eval_trigger_step": trigger})

        if args.mode in ("eval", "base-check") or not args.resume:
            run_evaluation(counters["total_env_steps"])
            if args.mode in ("eval", "base-check"):
                if async_evaluation is not None:
                    record_evaluations(async_evaluation.finish())
                return out
        if args.num_envs == 1:
            training = SerialCollector(
                LiberoResidualEnv(make_env(args.seed), planner, coordinates, norm_stats)
            )
        else:
            training = ParallelCollector(
                planner,
                norm_stats,
                n_envs=args.num_envs,
                runtime_root=root,
                config_dir=out / "collector_config",
                seed=args.seed,
            )
        if restored is not None:
            # make_env seeds NumPy; restoration must happen AFTER construction.
            restore_rng(policy, restored["rng"])
            training.load_state_dict(restored["collector"])
        initial_reset_policy = planner.timing_snapshot()
        initial_reset_started = time.perf_counter()
        training.reset()
        initial_reset_seconds = time.perf_counter() - initial_reset_started
        initial_reset_policy = planner_timing_delta(planner, initial_reset_policy)
        manifest["initial_training_reset_timing"] = {
            "wall_seconds": initial_reset_seconds,
            "policy_inference_seconds": initial_reset_policy["seconds"],
            "policy_inference_calls": initial_reset_policy["calls"],
        }
        warmup_started = time.perf_counter()
        warmup_timing = {
            "collector_step_seconds": 0.0,
            "reset_seconds": 0.0,
            "policy_inference_seconds": 0.0,
            "policy_inference_calls": 0,
        }
        while counters["warmup_decisions"] < args.learning_starts:
            count = min(args.num_envs, args.learning_starts - counters["warmup_decisions"])
            indices = list(range(count))
            residual = np.random.uniform(-0.1, 0.1, size=(count, 50, 7)).astype(np.float32)
            policy_before = planner.timing_snapshot()
            started = time.perf_counter()
            rows = training.step(residual, indices)
            warmup_timing["collector_step_seconds"] += time.perf_counter() - started
            add_policy_timing(warmup_timing, planner_timing_delta(planner, policy_before))
            for _, transition, result in rows:
                online.add(transition)
                counters["warmup_decisions"] += 1
                counters["total_env_steps"] += result["duration_k"]
            done_indices = [i for i, _, result in rows if result["episode_done"]]
            if done_indices:
                policy_before = planner.timing_snapshot()
                started = time.perf_counter()
                training.reset(done_indices)
                warmup_timing["reset_seconds"] += time.perf_counter() - started
                add_policy_timing(warmup_timing, planner_timing_delta(planner, policy_before))
            if counters["warmup_decisions"] % (4 if args.test_mode else 32) == 0:
                log(counters["total_env_steps"], {"phase": "warmup", **counters})
            if async_evaluation is not None:
                record_evaluations(async_evaluation.poll())
        warmup_timing["total_seconds"] = time.perf_counter() - warmup_started
        warmup_timing["collector_environment_seconds"] = max(
            0.0,
            warmup_timing["collector_step_seconds"]
            + warmup_timing["reset_seconds"]
            - warmup_timing["policy_inference_seconds"],
        )
        warmup_timing["policy_inference_mean_seconds"] = (
            warmup_timing["policy_inference_seconds"] / warmup_timing["policy_inference_calls"]
            if warmup_timing["policy_inference_calls"]
            else 0.0
        )
        manifest["warmup_collection_seconds"] = warmup_timing["total_seconds"]
        manifest["warmup_timing"] = warmup_timing
        critic_warmup_started = time.perf_counter()
        while counters["critic_warmup_updates"] < args.critic_warmup_updates:
            batch = sample_training_batch(
                online, offline, args.data_mode, args.batch_size, learner.device
            )
            learner.update(batch, update_actor=False)
            counters["critic_warmup_updates"] += 1
        manifest["critic_warmup_seconds"] = time.perf_counter() - critic_warmup_started
        next_eval = (counters["total_env_steps"] // args.eval_interval + 1) * args.eval_interval
        next_save = (
            counters["total_env_steps"] // args.checkpoint_interval + 1
        ) * args.checkpoint_interval
        timing = {
            "learner_act_seconds": 0.0,
            "collector_step_seconds": 0.0,
            "collection_seconds": 0.0,
            "update_seconds": 0.0,
            "reset_seconds": 0.0,
            "policy_inference_seconds": 0.0,
            "policy_inference_calls": 0,
            "evaluation_call_seconds": 0.0,
            "checkpoint_seconds": 0.0,
        }

        def save():
            path = out / f"checkpoint_env{counters['total_env_steps']:09d}_{time.time_ns()}.pt"
            torch.save(
                {
                    "contract": contract,
                    "learner": learner.state_dict(),
                    "online": online.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "counters": dict(counters),
                    "rng": capture_rng(policy),
                    "offline_rng": None if offline is None else offline.rng.get_state(),
                    "collector": training.state_dict(),
                    "pending_evaluations": []
                    if async_evaluation is None
                    else async_evaluation.state_dict(),
                },
                path,
            )
            (out / "latest_checkpoint.json").write_text(
                json.dumps({"path": str(path), "counters": counters}, indent=2) + "\n"
            )
            return path

        loop_started, starting_steps = time.perf_counter(), counters["train_env_steps"]
        while (
            counters["train_decisions"] < args.train_decisions
            and counters["train_env_steps"] < args.train_env_steps
        ):
            count = min(args.num_envs, args.train_decisions - counters["train_decisions"])
            # Keep final primitive-budget overshoot bounded by one Q50 chunk,
            # as in the original serial runner.
            if args.train_env_steps - counters["train_env_steps"] < count * 50:
                count = 1
            indices = list(range(count))
            started = time.perf_counter()
            residual = learner.act(*training.act_input(indices), eval_mode=False)
            timing["learner_act_seconds"] += time.perf_counter() - started
            policy_before = planner.timing_snapshot()
            started = time.perf_counter()
            rows = training.step(residual, indices)
            timing["collector_step_seconds"] += time.perf_counter() - started
            add_policy_timing(timing, planner_timing_delta(planner, policy_before))
            timing["collection_seconds"] = (
                timing["learner_act_seconds"] + timing["collector_step_seconds"]
            )
            for _, transition, result in rows:
                online.add(transition)
                counters["train_decisions"] += 1
                counters["train_env_steps"] += result["duration_k"]
                counters["total_env_steps"] += result["duration_k"]
            metrics = {}
            started = time.perf_counter()
            for update_actor in scheduler.plan(len(rows)):
                batch = sample_training_batch(
                    online, offline, args.data_mode, args.batch_size, learner.device
                )
                metrics.update(learner.update(batch, update_actor))
                counters["critic_updates"] += 1
                counters["actor_updates"] += int(update_actor)
            timing["update_seconds"] += time.perf_counter() - started
            done_rows = [(i, result) for i, _, result in rows if result["episode_done"]]
            if done_rows:
                counters["episodes"] += len(done_rows)
                counters["successes"] += sum(int(result["success"]) for _, result in done_rows)
                policy_before = planner.timing_snapshot()
                started = time.perf_counter()
                training.reset([i for i, _ in done_rows])
                timing["reset_seconds"] += time.perf_counter() - started
                add_policy_timing(timing, planner_timing_delta(planner, policy_before))
            if args.test_mode or counters["train_decisions"] % 100 == 0:
                log(
                    counters["total_env_steps"],
                    {
                        **metrics,
                        **counters,
                        "phase": "train",
                        "buffer/online_size": len(online),
                        "buffer/offline_size": 0 if offline is None else len(offline),
                        "perf/learner_act_seconds": timing["learner_act_seconds"],
                        "perf/collector_step_seconds": timing["collector_step_seconds"],
                        "perf/policy_inference_seconds": timing["policy_inference_seconds"],
                        "perf/policy_inference_calls": timing["policy_inference_calls"],
                        "perf/update_seconds": timing["update_seconds"],
                        "perf/reset_seconds": timing["reset_seconds"],
                        "action/controller_saturation_fraction": float(
                            np.mean([r["controller_saturation_fraction"] for _, _, r in rows])
                        ),
                        "action/residual_abs_mean": float(np.mean(np.abs(residual))),
                    },
                )
            if counters["total_env_steps"] >= next_eval:
                started = time.perf_counter()
                run_evaluation(next_eval)
                timing["evaluation_call_seconds"] += time.perf_counter() - started
                next_eval = (
                    counters["total_env_steps"] // args.eval_interval + 1
                ) * args.eval_interval
            if counters["total_env_steps"] >= next_save:
                started = time.perf_counter()
                save()
                timing["checkpoint_seconds"] += time.perf_counter() - started
                next_save = (
                    counters["total_env_steps"] // args.checkpoint_interval + 1
                ) * args.checkpoint_interval
            if async_evaluation is not None:
                record_evaluations(async_evaluation.poll())
        timing["training_loop_seconds"] = time.perf_counter() - loop_started
        timing["measured_train_env_steps"] = counters["train_env_steps"] - starting_steps
        timing["train_env_steps_per_second"] = (
            timing["measured_train_env_steps"] / timing["training_loop_seconds"]
        )
        timing["policy_inference_mean_seconds"] = (
            timing["policy_inference_seconds"] / timing["policy_inference_calls"]
            if timing["policy_inference_calls"]
            else 0.0
        )
        timing["collector_environment_seconds"] = max(
            0.0,
            timing["collector_step_seconds"]
            + timing["reset_seconds"]
            - timing["policy_inference_seconds"],
        )
        accounted = (
            timing["collection_seconds"]
            + timing["update_seconds"]
            + timing["reset_seconds"]
            + timing["evaluation_call_seconds"]
            + timing["checkpoint_seconds"]
        )
        timing["training_loop_overhead_seconds"] = max(
            0.0, timing["training_loop_seconds"] - accounted
        )
        started = time.perf_counter()
        if async_evaluation is not None:
            record_evaluations(async_evaluation.finish())
        timing["final_evaluation_wait_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        checkpoint = save()
        timing["final_checkpoint_seconds"] = time.perf_counter() - started
        manifest["base_policy_timing_total"] = planner.timing_snapshot()
        manifest["timing"] = timing
        manifest["final_counters"] = counters
        manifest["final_checkpoint"] = str(checkpoint)
        write_manifest()
        log(
            counters["total_env_steps"],
            {"status": "complete", "checkpoint": str(checkpoint), **counters},
        )
        return out
    finally:
        if async_evaluation is not None:
            async_evaluation.close()
        for env in (training, evaluation, demo_env):
            if env is not None:
                env.close()


def main():
    out = run(build_parser().parse_args())
    print(f"[DONE] {out}", flush=True)


if __name__ == "__main__":
    main()
