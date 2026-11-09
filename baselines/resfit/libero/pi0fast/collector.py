"""Fixed-chunk process collector, following ResFiT DP's vector adapter.

Workers reuse LiberoResidualEnv's exact transition construction. Frozen pi0-FAST
planning is deferred to the parent, which intentionally issues one batch-one
autoregressive model call per active observation.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import random
import traceback

import numpy as np


def _worker(connection, runtime_root, config_dir, norm_stats, action_scale, seed):
    from baselines.resfit.libero.pi0fast.runtime import (
        configure_runtime,
        make_env,
        LiberoResidualEnv,
    )
    from baselines.resfit.libero.pi0fast.model import ActionCoordinates

    adapter = None
    try:
        configure_runtime(runtime_root, config_dir)
        random.seed(seed)
        np.random.seed(seed)
        adapter = LiberoResidualEnv(
            make_env(seed), None, ActionCoordinates(action_scale), norm_stats
        )
        pending = None
        connection.send((True, "ready"))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                if pending is not None:
                    raise RuntimeError("cannot reset an unfinished chunk")
                value = adapter.begin_reset(payload)
            elif command == "reset_plan":
                adapter.set_observation(adapter.raw_obs, payload)
                value = adapter.act_input()
            elif command == "step":
                if pending is not None:
                    raise RuntimeError("previous chunk is unfinished")
                pending = adapter.begin_step(payload)
                value = {
                    "needs_plan": pending["needs_plan"],
                    "raw_obs": pending["result"]["next_raw_obs"] if pending["needs_plan"] else None,
                }
            elif command == "step_plan":
                if pending is None:
                    raise RuntimeError("no pending chunk")
                if pending["needs_plan"] and payload is None:
                    raise ValueError("missing endpoint pi0-FAST plan")
                transition, result = adapter.finish_step(pending, payload)
                pending = None
                # Full camera observations have already been sent for planning.
                value = (transition, {k: v for k, v in result.items() if k != "next_raw_obs"})
            elif command == "get_rng":
                value = {"numpy": np.random.get_state(), "python": random.getstate()}
            elif command == "get_sim_state":
                value = np.asarray(adapter.env.get_sim_state(), np.float64).copy()
            elif command == "set_rng":
                np.random.set_state(payload["numpy"])
                random.setstate(payload["python"])
                value = None
            else:
                raise ValueError("unknown collector command: " + command)
            connection.send((True, value))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        if adapter is not None:
            adapter.close()
        connection.close()


class SerialCollector:
    n_envs = 1

    def __init__(self, adapter):
        self.adapter = adapter

    def reset(self, indices=None):
        self.adapter.reset()

    def act_input(self, indices=None):
        return self.adapter.act_input()

    def step(self, residuals, indices=None):
        transition, result = self.adapter.step(np.asarray(residuals)[0])
        return [(0, transition, result)]

    def close(self):
        self.adapter.close()

    def state_dict(self):
        return {"n_envs": 1}

    def load_state_dict(self, state):
        if state != {"n_envs": 1}:
            raise ValueError("serial collector configuration changed")


class ParallelCollector:
    def __init__(
        self, planner, norm_stats, *, n_envs, runtime_root, config_dir, seed, action_scale=0.1
    ):
        if n_envs <= 1:
            raise ValueError("ParallelCollector requires at least two environments")
        self.planner, self.n_envs = planner, int(n_envs)
        self.connections, self.processes = [], []
        self.inputs = [None] * self.n_envs
        context = mp.get_context("spawn")  # Never fork a parent with live JAX/CUDA.
        try:
            for index in range(self.n_envs):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker,
                    args=(
                        child,
                        str(runtime_root),
                        str(Path(config_dir) / f"env{index}"),
                        norm_stats,
                        action_scale,
                        seed + index,
                    ),
                    daemon=True,
                )
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for index in range(self.n_envs):
                self._receive(index)
        except BaseException:
            self.close()
            raise

    def _receive(self, index):
        connection = self.connections[index]
        if not connection.poll(180):
            raise TimeoutError(
                f"LIBERO worker {index} did not respond; alive={self.processes[index].is_alive()}"
            )
        ok, result = connection.recv()
        if not ok:
            raise RuntimeError(f"LIBERO worker {index}:\n{result}")
        return result

    def _indices(self, indices):
        indices = list(range(self.n_envs)) if indices is None else list(indices)
        if len(set(indices)) != len(indices) or any(i < 0 or i >= self.n_envs for i in indices):
            raise ValueError("invalid worker indices")
        return indices

    def reset(self, indices=None, init_states=None):
        indices = self._indices(indices)
        if not indices:
            return
        if init_states is None:
            init_states = [None] * len(indices)
        if len(init_states) != len(indices):
            raise ValueError("initial state count mismatch")
        for index, initial in zip(indices, init_states):
            self.connections[index].send(("reset", initial))
        raw = [self._receive(index) for index in indices]
        plans = self.planner.plan_many(raw, padded_batch_size=self.n_envs)
        for index, plan in zip(indices, plans):
            self.connections[index].send(("reset_plan", plan))
        for index in indices:
            self.inputs[index] = self._receive(index)

    def act_input(self, indices=None):
        inputs = [self.inputs[i] for i in self._indices(indices)]
        if not inputs or any(value is None for value in inputs):
            raise RuntimeError("active workers must be reset before acting")
        return (
            {k: np.concatenate([value[0][k] for value in inputs]) for k in inputs[0][0]},
            np.concatenate([value[1] for value in inputs]),
        )

    def step(self, residuals, indices=None):
        indices = self._indices(indices)
        if len(residuals) != len(indices):
            raise ValueError("residual batch size mismatch")
        for index, residual in zip(indices, residuals):
            self.connections[index].send(("step", residual))
        pending = {index: self._receive(index) for index in indices}
        need = [index for index in indices if pending[index]["needs_plan"]]
        plans = self.planner.plan_many(
            [pending[i]["raw_obs"] for i in need], padded_batch_size=self.n_envs
        )
        by_index = dict(zip(need, plans))
        for index in indices:
            self.connections[index].send(("step_plan", by_index.get(index)))
        rows = []
        for index in indices:
            transition, result = self._receive(index)
            self.inputs[index] = (
                None
                if result["episode_done"]
                else (
                    {k: value[None] for k, value in transition.next_state.items()},
                    transition.next_base_action[None],
                )
            )
            rows.append((index, transition, result))
        return rows

    def close(self):
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, OSError):
                    pass
        for connection, process in zip(self.connections, self.processes):
            process.join(timeout=10)
            connection.close()
        # Processes are owned simulation workers. No external jobs are touched.

    def state_dict(self):
        for connection in self.connections:
            connection.send(("get_rng", None))
        return {"n_envs": self.n_envs, "worker_rng": [self._receive(i) for i in range(self.n_envs)]}

    def load_state_dict(self, state):
        if state["n_envs"] != self.n_envs or len(state["worker_rng"]) != self.n_envs:
            raise ValueError("parallel collector configuration changed")
        for connection, rng in zip(self.connections, state["worker_rng"]):
            connection.send(("set_rng", rng))
        for index in range(self.n_envs):
            self._receive(index)
