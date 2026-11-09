"""Persistent LIBERO evaluation of immutable actor snapshots."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback

import numpy as np


class TraceEnv:
    """Optional validation trace of every command and primitive sim state."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        raw = self.env.reset()
        self.actions, self.physics = hashlib.sha256(), hashlib.sha256()
        self.physics.update(np.asarray(self.env.get_sim_state(), np.float64).tobytes())
        self.steps = 0
        return raw

    def step(self, action):
        result = self.env.step(action)
        self.actions.update(np.asarray(action, np.float32).tobytes())
        self.physics.update(np.asarray(self.env.get_sim_state(), np.float64).tobytes())
        self.physics.update(np.asarray([result[1], result[2]], np.float64).tobytes())
        self.steps += 1
        return result

    def trace(self):
        return {
            "steps": self.steps,
            "action_sha256": self.actions.hexdigest(),
            "physics_reward_done_sha256": self.physics.hexdigest(),
        }


def _evaluation_worker(jobs, results, runtime_root, checkpoint_dir, gpu, config, seed, trace):
    # Spawn imports modules, but no CUDA context has been created at this point.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    adapter = None
    try:
        import torch
        from baselines.resfit.libero.pi05.runtime import (
            configure_runtime,
            load_policy,
            make_env,
            Pi05Planner,
            LiberoResidualEnv,
            variant,
            isolated_rng,
        )
        from baselines.resfit.libero.pi05.model import (
            ActionCoordinates,
            PixelResidualActor,
            PixelResFiTLearner,
        )
        from baselines.resfit.dp.td3_core import ResFiTLearnerConfig
        from baselines.resfit.libero.pi05.train import evaluate

        # Parent already created a private LIBERO config before spawning.
        configure_runtime(runtime_root, os.environ["LIBERO_CONFIG_PATH"])
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        torch.set_num_threads(1)
        policy, norm_stats = load_policy(checkpoint_dir)

        class ActorPolicy:
            act = PixelResFiTLearner.act

        learner = ActorPolicy()
        learner.device = torch.device("cuda:0")
        learner.actor = PixelResidualActor(ResFiTLearnerConfig(**config)).to(learner.device)
        with isolated_rng(policy):
            env = make_env(seed + 123)
            adapter = LiberoResidualEnv(
                TraceEnv(env) if trace else env,
                Pi05Planner(policy, variant(norm_stats)),
                ActionCoordinates(),
                norm_stats,
                bootstrap_timeouts=False,
            )
        results.put({"ready": True, "gpu": gpu})
        while True:
            job = jobs.get()
            if job is None:
                break
            path = Path(job["snapshot"])
            result_path = Path(str(path) + ".result.json")
            if result_path.exists():
                result = json.loads(result_path.read_text())
            else:
                snapshot = torch.load(path, map_location="cpu", weights_only=False)
                if snapshot["job"] != job:
                    raise ValueError("evaluation snapshot/job mismatch")
                learner.actor.load_state_dict(snapshot["actor"])
                started = time.perf_counter()
                metrics = evaluate(adapter, learner, job["episodes"], job["seed"], trace=trace)
                result = {"job": job, "metrics": metrics, "seconds": time.perf_counter() - started}
                with result_path.open("x") as stream:
                    json.dump(result, stream)
            results.put(result)
    except BaseException:
        results.put({"error": traceback.format_exc()})
    finally:
        if adapter is not None:
            adapter.close()


class AsyncSerialEvaluator:
    def __init__(
        self,
        *,
        runtime_root,
        checkpoint_dir,
        gpu,
        learner_config,
        seed,
        output,
        episodes,
        max_pending=2,
        trace=False,
    ):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episodes, self.seed, self.max_pending = episodes, seed + 10_000_000, max_pending
        self.pending, self.completed = {}, []
        self.ready = False
        context = mp.get_context("spawn")
        self.jobs, self.results = context.Queue(), context.Queue()
        self.process = context.Process(
            target=_evaluation_worker,
            args=(
                self.jobs,
                self.results,
                str(runtime_root),
                str(checkpoint_dir),
                gpu,
                learner_config,
                seed,
                trace,
            ),
            daemon=True,
        )
        self.process.start()

    def _receive(self, wait=False):
        try:
            result = self.results.get(timeout=1) if wait else self.results.get_nowait()
        except queue.Empty:
            if not self.process.is_alive():
                raise RuntimeError(f"evaluation worker exited with code {self.process.exitcode}")
            return False
        if "error" in result:
            raise RuntimeError("evaluation worker failed:\n" + result["error"])
        if "ready" in result:
            self.ready = True
        else:
            self.pending.pop(result["job"]["snapshot"], None)
            self.completed.append(result)
        return True

    def wait_ready(self):
        while not self.ready:
            self._receive(wait=True)

    def submit(self, actor, step, trigger):
        import torch

        while len(self.pending) >= self.max_pending:
            self._receive(wait=True)
        path = self.output / f"actor_env{step:09d}_{time.time_ns()}.pt"
        job = {
            "snapshot": str(path),
            "step": int(step),
            "trigger": int(trigger),
            "episodes": self.episodes,
            "seed": self.seed,
        }
        # Clones form a consistent, durable snapshot before training continues.
        weights = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
        torch.save({"job": job, "actor": weights}, path)
        self.pending[str(path)] = job
        self.jobs.put(job)

    def restore_pending(self, jobs):
        for job in jobs:
            if job["episodes"] != self.episodes or job["seed"] != self.seed:
                raise ValueError("pending evaluation settings changed")
            while len(self.pending) >= self.max_pending:
                self._receive(wait=True)
            self.pending[job["snapshot"]] = job
            self.jobs.put(job)

    def state_dict(self):
        return list(self.pending.values())

    def poll(self):
        while self._receive():
            pass
        completed, self.completed = self.completed, []
        return completed

    def finish(self):
        while self.pending:
            self._receive(wait=True)
        completed, self.completed = self.completed, []
        return completed

    def close(self):
        # Normal callers finish and record every result before closing.
        self.jobs.put(None)
        self.process.join(timeout=10)


EPISODE_SEED_PROTOCOL = "episode_seed_v1"


def assign_worker_gpus(gpus, workers):
    """Assign persistent workers round-robin across distinct physical GPUs."""
    selected = tuple(int(gpu) for gpu in gpus)
    if workers <= 0 or not selected or any(gpu < 0 for gpu in selected):
        raise ValueError("evaluation workers and physical GPUs must be positive/nonnegative")
    if len(set(selected)) != len(selected):
        raise ValueError("evaluation physical GPUs must be distinct")
    return tuple(selected[index % len(selected)] for index in range(workers))


def evaluate_seeded_episode(adapter, learner, episode_id, seed, *, trace=False):
    """Use the original evaluator with an RNG stream owned by this episode."""
    from baselines.resfit.libero.pi05.train import evaluate

    started = time.perf_counter()
    episode_seed = int(seed) + int(episode_id)
    metrics = evaluate(adapter, learner, 1, episode_seed, trace=trace)
    return {
        "episode_id": int(episode_id),
        "seed": episode_seed,
        "metrics": metrics,
        "seconds": time.perf_counter() - started,
    }


def merge_episode_results(episodes, count, seed, *, trace=False):
    """Aggregate each expected episode exactly once, independent of completion order."""
    ordered = sorted(episodes, key=lambda row: row["episode_id"])
    if len(ordered) != count or [row["episode_id"] for row in ordered] != list(range(count)):
        raise ValueError("missing or duplicate evaluation episode")
    if any(
        row["seed"] != seed + row["episode_id"] or row["metrics"]["env_step_eval/episodes"] != 1
        for row in ordered
    ):
        raise ValueError("evaluation episode seed/count mismatch")
    metrics = [row["metrics"] for row in ordered]
    errors = [
        row["env_step_eval/zero_residual_command_error"]
        for row in metrics
        if row["env_step_eval/zero_residual_command_error"] is not None
    ]
    merged = {
        "env_step_eval/success_rate": float(
            np.mean([row["env_step_eval/success_rate"] for row in metrics])
        ),
        "env_step_eval/episodes": count,
        "env_step_eval/episode_length_mean": float(
            np.mean([row["env_step_eval/episode_length_mean"] for row in metrics])
        ),
        "env_step_eval/zero_residual_command_error": max(errors) if errors else None,
    }
    if trace:
        if any(len(row["evaluation_trace"]) != 1 for row in metrics):
            raise ValueError("each evaluation episode must have one trace")
        merged["evaluation_trace"] = [row["evaluation_trace"][0] for row in metrics]
    return merged


def _episode_evaluation_worker(
    jobs,
    results,
    runtime_root,
    checkpoint_dir,
    gpu,
    config,
    seed,
    trace,
    worker_index,
    workers,
    config_dir,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    adapter = None
    try:
        import torch
        from baselines.resfit.libero.pi05.runtime import (
            configure_runtime,
            load_policy,
            make_env,
            Pi05Planner,
            LiberoResidualEnv,
            variant,
            isolated_rng,
        )
        from baselines.resfit.libero.pi05.model import (
            ActionCoordinates,
            PixelResidualActor,
            PixelResFiTLearner,
        )
        from baselines.resfit.dp.td3_core import ResFiTLearnerConfig

        configure_runtime(runtime_root, config_dir)
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        torch.set_num_threads(1)
        policy, norm_stats = load_policy(checkpoint_dir)

        class ActorPolicy:
            act = PixelResFiTLearner.act

        learner = ActorPolicy()
        learner.device = torch.device("cuda:0")
        learner.actor = PixelResidualActor(ResFiTLearnerConfig(**config)).to(learner.device)
        with isolated_rng(policy):
            env = make_env(seed + 123)
            adapter = LiberoResidualEnv(
                TraceEnv(env) if trace else env,
                Pi05Planner(policy, variant(norm_stats)),
                ActionCoordinates(),
                norm_stats,
                bootstrap_timeouts=False,
            )
            # Pay batch-one compilation before advertising worker readiness.
            adapter.reset()
            learner.act(*adapter.act_input(), eval_mode=True)
        results.put({"ready": True, "worker": worker_index, "gpu": gpu})
        loaded = None
        while True:
            job = jobs.get()
            if job is None:
                break
            if job["protocol"] != EPISODE_SEED_PROTOCOL:
                raise ValueError("evaluation RNG protocol mismatch")
            path = Path(job["snapshot"])
            for episode_id in range(worker_index, job["episodes"], workers):
                result_path = Path(str(path) + f".episode{episode_id:04d}.json")
                if result_path.exists():
                    record = json.loads(result_path.read_text())
                    if record["job"] != job or record["episode"]["episode_id"] != episode_id:
                        raise ValueError("saved evaluation episode/job mismatch")
                else:
                    if loaded != path:
                        snapshot = torch.load(path, map_location="cpu", weights_only=False)
                        if snapshot["job"] != job:
                            raise ValueError("evaluation snapshot/job mismatch")
                        learner.actor.load_state_dict(snapshot["actor"])
                        loaded = path
                    episode = evaluate_seeded_episode(
                        adapter, learner, episode_id, job["seed"], trace=trace
                    )
                    record = {"job": job, "episode": episode}
                    with result_path.open("x") as stream:
                        json.dump(record, stream)
                results.put({**record, "worker": worker_index})
    except BaseException:
        results.put({"error": traceback.format_exc(), "worker": worker_index})
    finally:
        if adapter is not None:
            adapter.close()


class AsyncEpisodeEvaluator(AsyncSerialEvaluator):
    """Several persistent processes evaluate episodes of the same snapshot.

    Each process holds the same frozen pi0.5 and actor on an assigned physical GPU.
    Episode seeds and durable episode files are independent of worker count/order
    and GPU placement.
    """

    def __init__(
        self,
        *,
        runtime_root,
        checkpoint_dir,
        learner_config,
        seed,
        output,
        episodes,
        workers,
        gpu=None,
        gpus=None,
        max_pending=2,
        trace=False,
    ):
        if workers <= 0 or episodes <= 0 or max_pending <= 0:
            raise ValueError("evaluation workers, episodes and queue capacity must be positive")
        if (gpu is None) == (gpus is None):
            raise ValueError("provide exactly one of gpu or gpus")
        selected_gpus = [gpu] if gpu is not None else gpus
        self.worker_gpus = assign_worker_gpus(selected_gpus, workers)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episodes, self.seed, self.max_pending = episodes, seed + 10_000_000, max_pending
        self.workers, self.trace = workers, trace
        self.pending, self.completed, self.partial, self.started = {}, [], {}, {}
        self.ready, self.ready_workers = False, set()
        self.jobs, self.processes = [], []
        context = mp.get_context("spawn")
        self.results = context.Queue()
        try:
            for index in range(workers):
                jobs = context.Queue()
                worker_gpu = self.worker_gpus[index]
                process = context.Process(
                    target=_episode_evaluation_worker,
                    args=(
                        jobs,
                        self.results,
                        str(runtime_root),
                        str(checkpoint_dir),
                        worker_gpu,
                        learner_config,
                        seed,
                        trace,
                        index,
                        workers,
                        str(self.output / f"worker{index}_config"),
                    ),
                    daemon=True,
                )
                process.start()
                self.jobs.append(jobs)
                self.processes.append(process)
        except BaseException:
            self.close()
            raise

    def _receive(self, wait=False):
        try:
            result = self.results.get(timeout=1) if wait else self.results.get_nowait()
        except queue.Empty:
            for index, process in enumerate(self.processes):
                if not process.is_alive():
                    raise RuntimeError(
                        f"evaluation worker {index} exited with code {process.exitcode}"
                    )
            return False
        if "error" in result:
            raise RuntimeError(f"evaluation worker {result['worker']} failed:\n" + result["error"])
        if "ready" in result:
            if result["gpu"] != self.worker_gpus[result["worker"]]:
                raise ValueError("evaluation worker started on an unexpected physical GPU")
            self.ready_workers.add(result["worker"])
            self.ready = len(self.ready_workers) == self.workers
            return True
        job, episode = result["job"], result["episode"]
        path = job["snapshot"]
        if self.pending.get(path) != job:
            raise ValueError("unexpected evaluation job result")
        episode_id = episode["episode_id"]
        if episode_id in self.partial[path] or not 0 <= episode_id < self.episodes:
            raise ValueError("duplicate or invalid evaluation episode")
        self.partial[path][episode_id] = episode
        if len(self.partial[path]) == self.episodes:
            episodes = [self.partial[path][index] for index in range(self.episodes)]
            metrics = merge_episode_results(episodes, self.episodes, self.seed, trace=self.trace)
            completed = {
                "job": job,
                "metrics": metrics,
                "episodes": episodes,
                "seconds": time.perf_counter() - self.started[path],
                "workers": self.workers,
                "worker_gpus": list(self.worker_gpus),
            }
            with Path(path + ".result.json").open("x") as stream:
                json.dump(completed, stream)
            self.completed.append(completed)
            self.pending.pop(path)
            self.partial.pop(path)
            self.started.pop(path)
        return True

    def _enqueue(self, job):
        path = job["snapshot"]
        result_path = Path(path + ".result.json")
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result["job"] != job:
                raise ValueError("saved evaluation result/job mismatch")
            self.completed.append(result)
            return
        self.pending[path], self.partial[path], self.started[path] = job, {}, time.perf_counter()
        for jobs in self.jobs:
            jobs.put(job)

    def submit(self, actor, step, trigger):
        import torch

        while len(self.pending) >= self.max_pending:
            self._receive(wait=True)
        path = self.output / f"actor_env{step:09d}_{time.time_ns()}.pt"
        job = {
            "snapshot": str(path),
            "step": int(step),
            "trigger": int(trigger),
            "episodes": self.episodes,
            "seed": self.seed,
            "protocol": EPISODE_SEED_PROTOCOL,
            "trace": self.trace,
        }
        weights = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
        torch.save({"job": job, "actor": weights}, path)
        self._enqueue(job)

    def restore_pending(self, jobs):
        for job in jobs:
            if (
                job["episodes"] != self.episodes
                or job["seed"] != self.seed
                or job.get("protocol") != EPISODE_SEED_PROTOCOL
                or job.get("trace") != self.trace
            ):
                raise ValueError("pending evaluation settings changed")
            while len(self.pending) >= self.max_pending:
                self._receive(wait=True)
            self._enqueue(job)

    def close(self):
        for jobs in self.jobs:
            jobs.put(None)
        for process in self.processes:
            process.join(timeout=10)
