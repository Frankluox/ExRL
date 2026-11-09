"""Convert RGB/state/action HDF5 demonstrations to OpenPI's LeRobot format."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def episode_arrays(group, crop_size=None):
    arrays = {key: np.asarray(group[key]) for key in ("state", "actions", "image", "wrist_image")}
    length = len(arrays["state"])
    if not length or any(len(value) != length for value in arrays.values()):
        raise ValueError("All four streams must have the same nonzero length")
    for key in ("state", "actions"):
        value = arrays[key].astype(np.float32)
        if value.shape != (length, 7) or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain finite N x 7 joint/gripper vectors")
        if np.any((value[:, 6] < 0) | (value[:, 6] > 1)):
            raise ValueError("The gripper coordinate must be closed fraction in [0, 1]")
        arrays[key] = value
    for key in ("image", "wrist_image"):
        value = arrays[key]
        if value.ndim != 4 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise ValueError(f"{key} must contain uint8 RGB images shaped N x H x W x 3")
        if crop_size is not None:
            height, width = value.shape[1:3]
            if not 0 < crop_size <= min(height, width):
                raise ValueError("The center crop must fit inside both cameras")
            y, x = (height - crop_size) // 2, (width - crop_size) // 2
            arrays[key] = value[:, y : y + crop_size, x : x + crop_size]
    return arrays


def main():
    import h5py

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--crop-size", type=int)
    args = parser.parse_args()
    repo = Path(args.repo_id)
    if repo.is_absolute() or len(repo.parts) != 2 or ".." in repo.parts:
        raise ValueError("repo-id must be namespace/dataset")
    target = args.output_root.expanduser().resolve() / repo
    if target.exists():
        raise FileExistsError(target)
    os.environ["HF_LEROBOT_HOME"] = str(target.parent.parent)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    with h5py.File(args.dataset, "r") as source:
        episodes = source["data"]
        names = sorted(episodes, key=lambda key: int(key.rsplit("_", 1)[-1]))
        if not names:
            raise ValueError("No demonstrations under data/demo_N")
        first = episode_arrays(episodes[names[0]], args.crop_size)
        features = {
            key: dict(
                dtype="image", shape=first[key].shape[1:], names=["height", "width", "channel"]
            )
            for key in ("image", "wrist_image")
        }
        features.update(
            {key: dict(dtype="float32", shape=(7,), names=[key]) for key in ("state", "actions")}
        )
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=target,
            fps=20,
            robot_type="piper",
            features=features,
            use_videos=False,
            image_writer_threads=4,
            image_writer_processes=0,
        )
        for name in names:
            arrays = episode_arrays(episodes[name], args.crop_size)
            for key in ("image", "wrist_image"):
                if arrays[key].shape[1:] != first[key].shape[1:]:
                    raise ValueError("Camera dimensions must remain constant across episodes")
            for i in range(len(arrays["state"])):
                dataset.add_frame(
                    {**{key: value[i] for key, value in arrays.items()}, "task": args.prompt}
                )
            dataset.save_episode()
            print(f"Saved {name}: {len(arrays['state'])} frames")
    print(f"Prepared {target}")


if __name__ == "__main__":
    main()
