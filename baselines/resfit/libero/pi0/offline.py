"""Build ResFiT demo replay using the existing task57 HDF5 replay workflow.

Derived from openpi/examples/libero/convert_libero_task57_to_lerobot.py and
resfit_offline_chunk_dataset_dsrl_ckpt_compat.py. Recorded images in older
LIBERO HDF5s are post-action observations: render BEFORE executing each chunk
by replaying from states[0], exactly as the pi0.5 finetuning converter does.
"""

from __future__ import annotations
import json
from pathlib import Path
import time

import numpy as np

from baselines.resfit.libero.pi0.runtime import TASK_NAME, sha256_file


def build_offline_replay(hdf5_path, adapter, replay, *, max_episodes=0, emit=print):
    import h5py

    path = Path(hdf5_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.name != TASK_NAME + "_demo.hdf5":
        raise ValueError("demo filename must identify the audited LIBERO task57")
    summary = {
        "path": str(path),
        "sha256": sha256_file(path),
        "episodes": 0,
        "success_episodes": 0,
        "transitions": 0,
        "primitive_steps": 0,
        "terminal_partial_chunks": 0,
        "timeouts": 0,
        "dataset_boundaries": 0,
        "observation_source": "replay_from_states0_render_before_action",
        "success_source": "replayed_environment_sparse_reward",
        "stored_hdf5_images_used": False,
        "max_episodes": max_episodes,
    }
    with h5py.File(path, "r") as handle:
        names = sorted(handle["data"], key=lambda n: int(n.split("_")[-1]))
        summary["dataset_episodes"] = len(names)
        if max_episodes:
            names = names[:max_episodes]
        # Protect the fixed demo buffer from silently discarding episodes.
        max_chunks = sum((min(len(handle["data"][n]["actions"]), 400) + 49) // 50 for n in names)
        if replay.capacity < max_chunks:
            raise ValueError(
                f"offline capacity {replay.capacity} < required upper bound {max_chunks}"
            )
        for name in names:
            started = time.monotonic()
            group = handle["data"][name]
            actions = np.asarray(group["actions"], np.float32)
            if (
                actions.ndim != 2
                or actions.shape[1] != 7
                or not len(actions)
                or not np.isfinite(actions).all()
            ):
                raise ValueError("invalid demonstration action array: " + name)
            adapter.reset(init_state=np.asarray(group["states"][0]))
            success = False
            for start in range(0, min(len(actions), 400), 50):
                transition, result = adapter.step(
                    np.zeros((50, 7), np.float32), demo_actions=actions[start : start + 50]
                )
                # Exact full-length dataset endings also truncate. The adapter
                # already planned the real endpoint for the bootstrap.
                if start + result["duration_k"] >= len(actions) and not transition.terminated:
                    transition.truncated = True
                    result["episode_done"] = True
                    summary["dataset_boundaries"] += 1
                replay.add(transition)
                summary["transitions"] += 1
                summary["primitive_steps"] += result["duration_k"]
                summary["terminal_partial_chunks"] += int(
                    transition.terminated and result["duration_k"] < 50
                )
                summary["timeouts"] += int(result["horizon_truncate"])
                success |= bool(result["success"])
                if result["episode_done"]:
                    break
            summary["episodes"] += 1
            summary["success_episodes"] += int(success)
            emit(
                json.dumps(
                    {
                        "offline_demo": name,
                        "success": success,
                        "transitions_so_far": len(replay),
                        "seconds": round(time.monotonic() - started, 2),
                    }
                )
            )
    if not summary["success_episodes"]:
        raise RuntimeError(
            "no replayed demonstration succeeded; check dataset/environment alignment"
        )
    return summary
