"""Replay a LIBERO task dataset and export RGB/state/action episodes for OpenPI."""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import math
import os
import pathlib

import h5py
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import tyro


@dataclasses.dataclass
class Args:
    raw_hdf5_root: pathlib.Path = pathlib.Path("data/libero_raw")
    repo_id: str = "local/libero_task57"
    output_root: pathlib.Path | None = None
    suite: str = "libero_90"
    task_id: int = 57
    image_size: int = 256
    fps: int = 20
    seed: int = 7
    max_demos: int = 0
    stop_on_success: bool = False
    keep_only_success: bool = False
    image_writer_threads: int = 10
    image_writer_processes: int = 5


def main(args: Args) -> None:
    if args.output_root is not None:
        os.environ["HF_LEROBOT_HOME"] = str(args.output_root.expanduser().resolve())

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    logging.basicConfig(level=logging.INFO)
    np.random.seed(args.seed)

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task = task_suite.get_task(args.task_id)

    output_path = _resolve_lerobot_output_path(HF_LEROBOT_HOME, args.repo_id, args.output_root)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing LeRobot dataset: {output_path}. "
            "Move it aside explicitly if you want to regenerate it."
        )

    hdf5_path = _find_task_hdf5(args.raw_hdf5_root, task_suite, args.task_id, args.suite)
    logging.info("Converting %s -> %s", hdf5_path, output_path)

    dataset = _create_dataset(LeRobotDataset, args, output_path)
    env, task_description = _make_env(task, args.image_size, args.seed)

    converted_episodes = 0
    converted_frames = 0
    success_episodes = 0

    with h5py.File(hdf5_path, "r") as hdf5_file:
        demos = sorted(hdf5_file["data"].keys(), key=_demo_sort_key)
        if args.max_demos > 0:
            demos = demos[: args.max_demos]

        for demo_key in demos:
            actions = np.asarray(hdf5_file[f"data/{demo_key}/actions"], dtype=np.float32)
            states = np.asarray(hdf5_file[f"data/{demo_key}/states"])
            frames, episode_success = _replay_demo(
                env=env,
                init_state=states[0],
                actions=actions,
                task_description=task_description,
                stop_on_success=args.stop_on_success,
            )
            if args.keep_only_success and not episode_success:
                logging.warning("Skipping %s because replay did not report success.", demo_key)
                continue

            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()

            converted_episodes += 1
            converted_frames += len(frames)
            success_episodes += int(episode_success)
            logging.info(
                "Saved %s: frames=%d success=%s total_episodes=%d",
                demo_key,
                len(frames),
                episode_success,
                converted_episodes,
            )

    env.close()
    metadata = {
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task.name,
        "task": task.language,
        "repo_id": args.repo_id,
        "raw_hdf5": str(hdf5_path),
        "output_path": str(output_path),
        "image_size": args.image_size,
        "fps": args.fps,
        "episodes": converted_episodes,
        "frames": converted_frames,
        "success_episodes": success_episodes,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "conversion.json").write_text(json.dumps(metadata, indent=2))
    logging.info("Wrote metadata: %s", output_path / "conversion.json")


def _resolve_lerobot_output_path(
    hf_lerobot_home: pathlib.Path,
    repo_id: str,
    output_root: pathlib.Path | None,
) -> pathlib.Path:
    root = output_root.expanduser().resolve() if output_root is not None else hf_lerobot_home
    return pathlib.Path(root) / repo_id


def _create_dataset(le_robot_dataset_cls, args: Args, output_path: pathlib.Path):
    kwargs = {
        "repo_id": args.repo_id,
        "robot_type": "panda",
        "fps": args.fps,
        "features": {
            "image": {
                "dtype": "image",
                "shape": (args.image_size, args.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (args.image_size, args.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        "image_writer_threads": args.image_writer_threads,
        "image_writer_processes": args.image_writer_processes,
    }
    if "root" in inspect.signature(le_robot_dataset_cls.create).parameters:
        kwargs["root"] = output_path
    return le_robot_dataset_cls.create(**kwargs)


def _find_task_hdf5(
    raw_hdf5_root: pathlib.Path, task_suite, task_id: int, suite: str
) -> pathlib.Path:
    raw_hdf5_root = raw_hdf5_root.expanduser().resolve()
    demo_relpath = pathlib.Path(task_suite.get_task_demonstration(task_id))
    candidates = [
        raw_hdf5_root / demo_relpath,
        raw_hdf5_root / suite / demo_relpath,
        raw_hdf5_root / "libero_100" / demo_relpath,
        raw_hdf5_root / demo_relpath.name,
        raw_hdf5_root / suite / demo_relpath.name,
        raw_hdf5_root / "libero_100" / suite / demo_relpath.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(raw_hdf5_root.rglob(demo_relpath.name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Found multiple candidate demos for {demo_relpath.name}: {matches}"
        )
    raise FileNotFoundError(f"Could not find {demo_relpath.name} under {raw_hdf5_root}")


def _make_env(task, resolution: int, seed: int):
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


def _replay_demo(
    *,
    env: OffScreenRenderEnv,
    init_state: np.ndarray,
    actions: np.ndarray,
    task_description: str,
    stop_on_success: bool,
) -> tuple[list[dict], bool]:
    env.reset()
    obs = env.set_init_state(init_state)
    frames = []
    success = False

    for action in actions:
        frames.append(
            {
                "image": _parse_image(obs["agentview_image"]),
                "wrist_image": _parse_image(obs["robot0_eye_in_hand_image"]),
                "state": _parse_state(obs),
                "actions": np.asarray(action, dtype=np.float32),
                "task": task_description,
            }
        )
        obs, _reward, done, _info = env.step(np.asarray(action, dtype=np.float32).tolist())
        success = success or bool(done)
        if done and stop_on_success:
            break

    return frames, success


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.ascontiguousarray(image[::-1, ::-1])
    return image.astype(np.uint8, copy=False)


def _parse_state(obs: dict) -> np.ndarray:
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)


def _demo_sort_key(demo_key: str) -> int:
    try:
        return int(demo_key.split("_")[-1])
    except ValueError:
        return 0


if __name__ == "__main__":
    main(tyro.cli(Args))
