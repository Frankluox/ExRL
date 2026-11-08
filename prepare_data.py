#!/usr/bin/env python3
"""Convert local Robomimic low-dimensional HDF5 demonstrations to ExRL assets."""
import argparse, json
from pathlib import Path
import numpy as np

TASKS = ("lift", "can", "square", "transport", "tool_hang")


def observation_keys(task):
    keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    if task == "transport":
        keys += ["robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"]
    return keys + ["object"]


def prepare(source, task, output, filter_key=None, max_episodes=0):
    import h5py

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty asset directory")
    keys = observation_keys(task)
    with h5py.File(source, "r") as f:
        if filter_key:
            names = [
                x.decode() if isinstance(x, bytes) else str(x)
                for x in f["mask"][filter_key][:]
            ]
        else:
            names = list(f["data"])
        names = sorted(names, key=lambda s: int(s.split("_")[-1]))
        if max_episodes > 0:
            names = names[:max_episodes]
        if not names:
            raise ValueError("No demonstrations selected")
        per_key = {k: [] for k in keys}
        actions = []
        rewards = []
        lengths = []
        for name in names:
            d = f["data"][name]
            a = np.asarray(d["actions"], dtype=np.float64)
            if len(a) == 0:
                raise ValueError(f"Empty episode: {name}")
            actions.append(a)
            lengths.append(len(a))
            rewards.append(np.asarray(d["rewards"], dtype=np.float32).reshape(-1))
            for k in keys:
                v = np.asarray(d["obs"][k], dtype=np.float64)
                if v.ndim != 2 or len(v) != len(a):
                    raise ValueError(f"Invalid observations: {name}/{k}")
                per_key[k].append(v)
        arrays = {k: np.concatenate(v) for k, v in per_key.items()}
        states = np.concatenate([arrays[k] for k in keys], axis=1)
        actions = np.concatenate(actions)
        if task == "transport" and (states.shape[1], actions.shape[1]) != (59, 14):
            raise ValueError(
                "Transport must include both robots: 59 state and 14 action dimensions"
            )
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("Non-finite data")
        norm = dict(
            obs_min=states.min(0),
            obs_max=states.max(0),
            action_min=actions.min(0),
            action_max=actions.max(0),
        )
        normalized_states = (
            2 * (states - norm["obs_min"]) / (norm["obs_max"] - norm["obs_min"] + 1e-6)
            - 1
        )
        normalized_actions = (
            2
            * (actions - norm["action_min"])
            / (norm["action_max"] - norm["action_min"] + 1e-6)
            - 1
        )
        stats = {}
        for k, a in {**arrays, "action": actions}.items():
            stats[k + "_mean"] = a.mean(0).astype(np.float32)
            stats[k + "_std"] = np.maximum(a.std(0), 1e-6).astype(np.float32)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output / "train.npz",
            states=normalized_states.astype(np.float32),
            actions=normalized_actions.astype(np.float32),
            rewards=np.concatenate(rewards),
            terminals=np.zeros(len(states), dtype=bool),
            traj_lengths=np.asarray(lengths, dtype=np.int64),
        )
        np.savez_compressed(output / "normalization.npz", **norm)
        np.savez_compressed(output / f"{task}_lowdim_stats.npz", **stats)
        env = f["data"].attrs.get("env_args")
        if env is None:
            raise ValueError("HDF5 data is missing env_args metadata")
        env = json.loads(env)
        (output / f"{task}.json").write_text(json.dumps(env, indent=2) + "\n")
    summary = dict(
        task=task,
        episodes=len(lengths),
        transitions=len(states),
        observation_keys=keys,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        filter_key=filter_key,
        observation_space="dsrl_normalized",
        action_space="dsrl_normalized",
    )
    (output / "dataset_info.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument(
        "--dataset",
        required=True,
        help="Local HDF5 file with low-dimensional observations",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--filter-key", help="Optional HDF5 mask key, such as train")
    p.add_argument("--max-episodes", type=int, default=0)
    a = p.parse_args()
    print(
        json.dumps(
            prepare(a.dataset, a.task, a.output, a.filter_key, a.max_episodes), indent=2
        )
    )


if __name__ == "__main__":
    main()
