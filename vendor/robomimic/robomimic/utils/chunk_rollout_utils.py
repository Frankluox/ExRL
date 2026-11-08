"""Chunk rollout helpers for chunk-policy algorithms."""

from collections import OrderedDict
from copy import deepcopy
import json
import os
import time

import imageio
import numpy as np

import robomimic.utils.log_utils as LogUtils
from robomimic.algo import RolloutPolicy
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper


def run_chunk_rollout(
    policy,
    env,
    horizon,
    chunk_exec_len,
    pred_chunk_len=None,
    use_goals=False,
    render=False,
    video_writer=None,
    video_skip=5,
    terminate_on_success=False,
):
    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)

    policy.start_episode()
    ob_dict = env.reset()
    goal_dict = env.get_goal() if use_goals else None

    rews = []
    success = None
    end_step = None
    video_count = 0
    video_frames = []

    decisions = 0
    actual_exec_lens = []

    try:
        step_i = -1
        while step_i + 1 < horizon:
            chunk = policy(ob=ob_dict, goal=goal_dict)
            if chunk.ndim == 1:
                chunk = chunk[None, :]
            if chunk.ndim != 2:
                raise RuntimeError(f"Expected chunk action with ndim 2, got shape {chunk.shape}")

            if not np.isfinite(chunk).all():
                raise RuntimeError("Chunk action contains NaN/Inf")

            pred_len = int(chunk.shape[0])
            if pred_chunk_len is not None and pred_len != int(pred_chunk_len):
                raise RuntimeError(f"Predicted chunk length {pred_len} != configured pred_len {pred_chunk_len}")
            if chunk_exec_len > pred_len:
                raise RuntimeError(f"chunk_exec_len ({chunk_exec_len}) > predicted chunk length ({pred_len})")

            decisions += 1
            executed_this_decision = 0

            for local_idx in range(chunk_exec_len):
                if step_i + 1 >= horizon:
                    break

                action = chunk[local_idx]
                ob_dict, r, done, _ = env.step(action)
                step_i += 1
                executed_this_decision += 1
                rews.append(r)

                cur_success = env.is_success()
                if success is None:
                    success = deepcopy(cur_success)
                else:
                    for k in success:
                        success[k] = success[k] | cur_success[k]

                if render:
                    env.render(mode="human")

                if video_writer is not None:
                    if video_count % video_skip == 0:
                        frame = env.render(mode="rgb_array", height=512, width=512)
                        video_frames.append(frame)
                    video_count += 1

                if done or (terminate_on_success and success["task"]):
                    end_step = step_i
                    break

            actual_exec_lens.append(executed_this_decision)

            if end_step is not None:
                break

    except env.rollout_exceptions as e:
        print(f"WARNING: got rollout exception {e}")

    if video_writer is not None:
        for frame in video_frames:
            video_writer.append_data(frame)

    if end_step is None:
        end_step = step_i

    if success is None:
        success = {"task": False}

    total_reward = np.sum(rews[: end_step + 1]) if len(rews) > 0 else 0.0
    results = {
        "Return": float(total_reward),
        "Horizon": int(end_step + 1),
        "Success_Rate": float(success["task"]),
        "pred_chunk_len": float(pred_chunk_len) if pred_chunk_len is not None else float(chunk.shape[0]),
        "exec_chunk_len_requested": float(chunk_exec_len),
        "exec_chunk_len_actual_mean": float(np.mean(actual_exec_lens)) if actual_exec_lens else 0.0,
        "decisions_per_episode": float(decisions),
    }
    return results


def rollout_with_chunk_stats(
    policy,
    envs,
    horizon,
    use_goals=False,
    num_episodes=None,
    render=False,
    video_dir=None,
    video_path=None,
    epoch=None,
    video_skip=5,
    terminate_on_success=False,
    verbose=False,
    chunk_exec_len=None,
):
    assert isinstance(policy, RolloutPolicy)

    all_rollout_logs = OrderedDict()

    assert (video_path is None) or (video_dir is None), "rollout_with_chunk_stats: can't specify both video path and dir"
    write_video = (video_path is not None) or (video_dir is not None)
    video_paths = OrderedDict()
    video_writers = OrderedDict()

    if video_path is not None:
        video_paths = {k: video_path for k in envs}
        video_writer = imageio.get_writer(video_path, fps=20)
        video_writers = {k: video_writer for k in envs}

    if video_dir is not None:
        video_str = f"_epoch_{epoch}.mp4" if epoch is not None else ".mp4"
        video_paths = {k: os.path.join(video_dir, f"{k}{video_str}") for k in envs}
        video_writers = {k: imageio.get_writer(video_paths[k], fps=20) for k in envs}

    pred_chunk_len = int(policy.policy.algo_config.chunk.pred_len)
    req_exec_len = int(chunk_exec_len if chunk_exec_len is not None else policy.policy.algo_config.chunk.exec_len)

    if req_exec_len > pred_chunk_len:
        raise RuntimeError(f"chunk_exec_len ({req_exec_len}) > pred_chunk_len ({pred_chunk_len})")

    for env_key, env in envs.items():
        env_video_writer = video_writers.get(env_key, None) if write_video else None
        env_name = env.name

        print(
            "chunk rollout: env={}, horizon={}, use_goals={}, num_episodes={}, pred_len={}, exec_len={}".format(
                env_name, horizon, use_goals, num_episodes, pred_chunk_len, req_exec_len
            )
        )

        rollout_logs = []
        iterator = range(num_episodes)
        if not verbose:
            iterator = LogUtils.custom_tqdm(iterator, total=num_episodes)

        num_success = 0
        for ep_i in iterator:
            ts = time.time()
            rollout_info = run_chunk_rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                chunk_exec_len=req_exec_len,
                pred_chunk_len=pred_chunk_len,
                render=render,
                use_goals=use_goals,
                video_writer=env_video_writer,
                video_skip=video_skip,
                terminate_on_success=terminate_on_success,
            )
            rollout_info["time"] = time.time() - ts
            rollout_logs.append(rollout_info)
            num_success += rollout_info["Success_Rate"]

            if verbose:
                print(f"Episode {ep_i + 1}, horizon={horizon}, num_success={num_success}")
                print(json.dumps(rollout_info, sort_keys=True, indent=4))

        if video_dir is not None and env_video_writer is not None:
            env_video_writer.close()

        rollout_logs = {k: [rollout_logs[i][k] for i in range(len(rollout_logs))] for k in rollout_logs[0]}
        rollout_logs_mean = {k: float(np.mean(v)) for k, v in rollout_logs.items()}
        rollout_logs_mean["Time_Episode"] = float(np.sum(rollout_logs["time"]) / 60.0)
        all_rollout_logs[env_key] = rollout_logs_mean

    if video_path is not None:
        video_writer.close()

    return all_rollout_logs, video_paths
