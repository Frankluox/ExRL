#!/usr/bin/env python3
"""Convert Minari D4RL Adroit datasets to robomimic low-dim HDF5."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


TASK_TO_ENV = {
    "door": "AdroitHandDoor-v1",
    "hammer": "AdroitHandHammer-v1",
    "pen": "AdroitHandPen-v1",
    "relocate": "AdroitHandRelocate-v1",
}
QUALITIES = ("human", "cloned", "expert")


def dataset_ids(tasks: Iterable[str], qualities: Iterable[str]) -> list[str]:
    return [f"D4RL/{task}/{quality}-v2" for task in tasks for quality in qualities]


def parse_dataset_id(dataset_id: str) -> tuple[str, str]:
    parts = dataset_id.split("/")
    if len(parts) != 3 or parts[0] != "D4RL":
        raise ValueError(f"Expected D4RL/<task>/<quality>-v2, got {dataset_id}")
    task = parts[1].lower()
    quality = parts[2].split("-")[0].lower()
    if task not in TASK_TO_ENV:
        raise ValueError(f"Unsupported Adroit task {task}")
    return task, quality


def as_array(x, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == object:
        arr = np.stack(list(x), axis=0)
    return arr.astype(dtype, copy=False)


def flatten_observations(observations) -> np.ndarray:
    if isinstance(observations, dict):
        if "observation" in observations:
            return as_array(observations["observation"])
        pieces = []
        for key in sorted(observations):
            pieces.append(as_array(observations[key]).reshape(len(observations[key]), -1))
        return np.concatenate(pieces, axis=-1)
    return as_array(observations)


def env_id_from_dataset(dataset, task: str) -> str:
    for attr in ("env_spec", "eval_env_spec"):
        spec = getattr(dataset, attr, None)
        if spec is not None and getattr(spec, "id", None):
            return str(spec.id)
    return TASK_TO_ENV[task]


def max_steps_from_dataset(dataset):
    spec = getattr(dataset, "env_spec", None)
    if spec is not None and getattr(spec, "max_episode_steps", None) is not None:
        return int(spec.max_episode_steps)
    return None


def write_mask(root, name: str, keys: list[str]) -> None:
    root[f"mask/{name}"] = np.array(keys, dtype="S")


def convert_one(
    dataset_id: str, output_path: Path, valid_ratio: float, seed: int, download: bool
) -> None:
    import minari

    task, quality = parse_dataset_id(dataset_id)
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists; move it aside or pass a new --output-root"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = minari.load_dataset(dataset_id, download=download)
    env_id = env_id_from_dataset(dataset, task)
    max_episode_steps = max_steps_from_dataset(dataset)
    env_kwargs = {}
    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = max_episode_steps

    demo_keys = []
    total = 0
    returns = []

    with h5py.File(output_path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(
            {
                "env_name": env_id,
                "type": 2,
                "env_kwargs": env_kwargs,
            }
        )
        data.attrs["source_dataset_id"] = dataset_id
        data.attrs["task"] = task
        data.attrs["quality"] = quality

        for idx, episode in enumerate(dataset.iterate_episodes()):
            actions = as_array(episode.actions)
            rewards = np.asarray(episode.rewards, dtype=np.float32).reshape(-1)
            terminations = np.asarray(episode.terminations, dtype=bool).reshape(-1)
            truncations = np.asarray(episode.truncations, dtype=bool).reshape(-1)
            observations = flatten_observations(episode.observations)

            steps = actions.shape[0]
            if observations.shape[0] == steps + 1:
                obs = observations[:-1]
                next_obs = observations[1:]
            elif observations.shape[0] == steps:
                obs = observations
                next_obs = np.concatenate([observations[1:], observations[-1:]], axis=0)
            else:
                raise ValueError(
                    f"{dataset_id} episode {idx}: obs length {observations.shape[0]} "
                    f"does not match action length {steps}"
                )

            dones = np.logical_or(terminations, truncations).astype(np.uint8)
            if len(dones) > 0:
                dones[-1] = 1

            demo_key = f"demo_{idx}"
            demo_keys.append(demo_key)
            demo = data.create_group(demo_key)
            demo.create_dataset("actions", data=actions.astype(np.float32, copy=False))
            demo.create_dataset("rewards", data=rewards)
            demo.create_dataset("dones", data=dones)
            obs_group = demo.create_group("obs")
            next_obs_group = demo.create_group("next_obs")
            obs_group.create_dataset("flat", data=obs.astype(np.float32, copy=False))
            next_obs_group.create_dataset("flat", data=next_obs.astype(np.float32, copy=False))
            demo.attrs["num_samples"] = int(steps)
            total += int(steps)
            returns.append(float(np.sum(rewards)))

        data.attrs["total"] = int(total)
        data.attrs["num_demos"] = len(demo_keys)
        data.attrs["return_mean"] = float(np.mean(returns)) if returns else 0.0

        if 0.0 < valid_ratio < 1.0 and len(demo_keys) > 1:
            rng = np.random.default_rng(seed)
            shuffled = list(demo_keys)
            rng.shuffle(shuffled)
            n_valid = max(1, int(round(valid_ratio * len(shuffled))))
            valid_keys = sorted(shuffled[:n_valid])
            train_keys = sorted(shuffled[n_valid:])
            write_mask(f, "train", train_keys)
            write_mask(f, "valid", valid_keys)

    print(f"Wrote {output_path}")
    print(f"  source={dataset_id} env={env_id} demos={len(demo_keys)} steps={total}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-id", action="append", help="Explicit Minari id, e.g. D4RL/door/human-v2"
    )
    parser.add_argument(
        "--task", action="append", choices=tuple(TASK_TO_ENV) + ("all",), default=None
    )
    parser.add_argument("--quality", action="append", choices=QUALITIES + ("all",), default=None)
    parser.add_argument(
        "--output-root",
        default=os.environ.get("ADROIT_ROBOMIMIC_DATASETS", "datasets/adroit_dp/robomimic"),
    )
    parser.add_argument("--minari-root", default=os.environ.get("MINARI_DATASETS_PATH"))
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--no-download", action="store_true", help="Require datasets to exist locally"
    )
    args = parser.parse_args()

    if args.minari_root:
        os.environ["MINARI_DATASETS_PATH"] = args.minari_root

    if args.dataset_id:
        ids = args.dataset_id
    else:
        tasks = list(TASK_TO_ENV) if args.task is None or "all" in args.task else args.task
        qualities = (
            list(QUALITIES) if args.quality is None or "all" in args.quality else args.quality
        )
        ids = dataset_ids(tasks, qualities)

    output_root = Path(args.output_root)
    for dataset_id in ids:
        task, quality = parse_dataset_id(dataset_id)
        out = output_root / task / f"{quality}_v2_robomimic.hdf5"
        convert_one(dataset_id, out, args.valid_ratio, args.seed, download=not args.no_download)


if __name__ == "__main__":
    main()
