"""Observation and action transforms for the frozen LIBERO base policy."""

import math
from typing import Optional, Sequence
import numpy as np
import PIL.Image
from openpi_client import image_tools


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def obs_to_img(obs, variant):
    if variant.env == "libero":
        curr_image = obs["agentview_image"][::-1, ::-1]
    elif variant.env == "aloha_cube":
        curr_image = obs["pixels"]["top"]
    else:
        raise NotImplementedError()
    if variant.resize_image > 0:
        curr_image = np.array(
            PIL.Image.fromarray(curr_image).resize((variant.resize_image, variant.resize_image))
        )
    return curr_image


def obs_to_pi_zero_input(obs, variant):
    if variant.env == "libero":
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))
        return {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ),
            "prompt": str(variant.task_description),
        }
    if variant.env == "aloha_cube":
        img = np.ascontiguousarray(obs["pixels"]["top"])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        pi_obs = {
            "state": obs["agent_pos"],
            "images": {"cam_high": np.transpose(img, (2, 0, 1))},
        }
        if getattr(variant, "task_description", ""):
            pi_obs["prompt"] = str(variant.task_description)
        return pi_obs
    raise NotImplementedError()


def obs_to_cosmos_input(obs, variant):
    """Match the current Cosmos ExRL-Q server's LIBERO request contract."""
    if variant.env != "libero":
        raise NotImplementedError("Cosmos ResFiT currently supports LIBERO only")
    proprio = np.concatenate(
        (
            obs["robot0_gripper_qpos"],
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"],
        )
    ).astype(np.float32)
    if proprio.shape != (9,) or not np.isfinite(proprio).all():
        raise ValueError(f"expected finite Cosmos proprio [9], got {proprio.shape}")
    return {
        "primary_image": np.ascontiguousarray(np.flipud(obs["agentview_image"])),
        "wrist_image": np.ascontiguousarray(np.flipud(obs["robot0_eye_in_hand_image"])),
        "proprio": proprio,
        "prompt": str(variant.task_description),
    }


def obs_to_qpos(obs, variant):
    if variant.env == "libero":
        return np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
    if variant.env == "aloha_cube":
        return obs["agent_pos"]
    raise NotImplementedError()


def obs_to_conditioning_state(obs, variant):
    if variant.env == "libero":
        return obs_to_qpos(obs, variant)
    if variant.env == "aloha_cube":
        # Match OpenPI's ALOHA training space before normalization.
        from openpi.policies import aloha_policy as openpi_aloha_policy

        return openpi_aloha_policy._decode_state(np.asarray(obs["agent_pos"]), adapt_to_pi=True)
    raise NotImplementedError()


def env_actions_to_conditioning_actions(obs, actions, variant):
    actions = np.asarray(actions, dtype=np.float32)
    squeeze = actions.ndim == 1
    if squeeze:
        actions = actions[None, :]

    if variant.env == "libero":
        state = obs_to_conditioning_state(obs, variant)
        conditioning_actions = actions.copy()
        # Cosmos follows the pi0.5 task57 convention. The switch stays explicit
        # so each checkpoint's action convention remains part of the contract.
        if bool(variant.aac_action_conditioning_extra_delta_transform):
            conditioning_actions[:, :6] -= state[:6][None, :]
    elif variant.env == "aloha_cube":
        # ALOHA runtime actions are absolute joint/gripper targets. Convert
        # them to pi-internal absolute actions first, then expose true temporal
        # deltas to the AAC learner: first target - current state, then
        # target_t - target_{t-1}.
        from openpi.policies import aloha_policy as openpi_aloha_policy

        absolute_actions = openpi_aloha_policy._encode_actions_inv(
            actions, adapt_to_pi=True
        ).astype(np.float32)
        current_state = np.asarray(obs_to_conditioning_state(obs, variant), dtype=np.float32)
        conditioning_actions = np.empty_like(absolute_actions, dtype=np.float32)
        conditioning_actions[0] = absolute_actions[0] - current_state
        if actions.shape[0] > 1:
            conditioning_actions[1:] = absolute_actions[1:] - absolute_actions[:-1]
    else:
        raise NotImplementedError()

    if squeeze:
        return conditioning_actions[0]
    return conditioning_actions


def _normalize_rl_obs_array(
    values: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    normalized = (values - mean) / (std + 1e-6)
    np.clip(normalized, -float(clip), float(clip), out=normalized)
    return normalized.astype(np.float32)


def _pad_plan(actions: np.ndarray, max_chunk_len: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected planned actions shape [T, action_dim], got {actions.shape}")
    action_dim = actions.shape[-1]
    max_chunk_len = int(max_chunk_len)
    padded = np.zeros((max_chunk_len, action_dim), dtype=np.float32)
    take = min(max_chunk_len, actions.shape[0])
    if take > 0:
        padded[:take] = actions[:take]
        if take < max_chunk_len:
            padded[take:] = actions[take - 1]
    return padded


def _build_prev_action(
    action_history: Sequence[np.ndarray], history_len: int, action_dim: int
) -> np.ndarray:
    history_len = int(history_len)
    prev_action = np.zeros((history_len, action_dim), dtype=np.float32)
    if history_len <= 0:
        return prev_action
    if len(action_history) <= 0:
        return prev_action
    take = list(action_history)[-history_len:]
    prev_action[-len(take) :] = np.asarray(take, dtype=np.float32)
    return prev_action


def build_aac_observation(
    raw_obs,
    planned_actions,
    action_history,
    variant,
    conditioning_actions: Optional[np.ndarray] = None,
):
    curr_image = obs_to_img(raw_obs, variant)
    obs_norm_stats = getattr(variant, "_aac_obs_norm_stats", None)
    if conditioning_actions is None:
        conditioning_actions = env_actions_to_conditioning_actions(
            raw_obs, planned_actions, variant
        )
    conditioning_actions = np.asarray(conditioning_actions, dtype=np.float32)
    observation = {
        "pixels": curr_image[..., np.newaxis],
    }
    if bool(int(getattr(variant, "condition_on_actions", 1))):
        observation["actions"] = _pad_plan(conditioning_actions, variant.max_chunk_len)
        if obs_norm_stats is not None:
            observation["actions"] = _normalize_rl_obs_array(
                observation["actions"],
                obs_norm_stats["action_mean"],
                obs_norm_stats["action_std"],
                obs_norm_stats["clip"],
            )
    if variant.add_states:
        qpos = obs_to_conditioning_state(raw_obs, variant)
        if obs_norm_stats is not None:
            qpos = _normalize_rl_obs_array(
                qpos,
                obs_norm_stats["state_mean"],
                obs_norm_stats["state_std"],
                obs_norm_stats["clip"],
            )
        observation["state"] = qpos[:, np.newaxis].astype(np.float32)
    if int(variant.action_history_len) > 0:
        prev_action = _build_prev_action(
            action_history,
            history_len=variant.action_history_len,
            action_dim=planned_actions.shape[-1],
        )
        if obs_norm_stats is not None:
            prev_action = _normalize_rl_obs_array(
                prev_action,
                obs_norm_stats["action_mean"],
                obs_norm_stats["action_std"],
                obs_norm_stats["clip"],
            )
        observation["prev_action"] = prev_action
    return observation


def _aac_step_reward(env_reward: float, variant) -> float:
    if variant.env == "aloha_cube":
        step_reward_mode = str(getattr(variant, "step_reward_mode", "sparse_success_step"))
        if step_reward_mode == "progressive_minus_offset_step":
            return float(env_reward) - float(variant.reward_offset)
        if step_reward_mode == "progressive_scaled_success_step":
            # Keep the progressive 0..env_max_reward signal but normalize it to
            # the same rough scale as sparse -1/0 rewards before discounting.
            reward_scale = max(float(getattr(variant, "env_max_reward", 1.0)), 1e-6)
            return float(env_reward) / reward_scale - float(variant.reward_offset)
        # pi DSRL ALOHA ignores the progressive 0/1/2/3/4 reward for
        # training and uses sparse time-to-success reward: -1 until the
        # success step, then 0. Keep AAC aligned while still logging the
        # raw progressive reward separately as env_reward_total.
        return (
            0.0
            if float(env_reward) == float(variant.env_max_reward)
            else -float(variant.reward_offset)
        )
    return float(env_reward) - float(variant.reward_offset)


def _execute_offline_action_chunk(
    env,
    raw_obs,
    planned_actions,
    conditioning_actions,
    exec_len: int,
    action_history,
    variant,
    resample_streak: int,
    *,
    count_env_steps_for_horizon: bool = True,
    catch_terminated_episode: bool = False,
):
    env_reward_total = 0.0
    backup_reward_total = 0.0
    current_step_gamma = 1.0
    env_steps = 0
    success = False
    forced_truncate = False
    env_terminal = False
    horizon_truncate = False
    episode_done = False
    done_for_backup = False
    next_raw_obs = raw_obs
    updated_episode_env_steps = int(getattr(variant, "_episode_env_steps", 0))

    if int(exec_len) == 0:
        reward = float(variant.resample_penalty)
        resample_streak += 1
        if int(resample_streak) >= int(variant.max_resample_streak):
            forced_truncate = True
            episode_done = True
            done_for_backup = False
        return {
            "reward": reward,
            "backup_reward_total": reward,
            "duration_k": 0,
            "next_raw_obs": next_raw_obs,
            "env_steps": 0,
            "env_reward_total": 0.0,
            "success": False,
            "episode_done": bool(episode_done),
            "done_for_backup": bool(done_for_backup),
            "forced_truncate": bool(forced_truncate),
            "env_terminal": bool(env_terminal),
            "horizon_truncate": bool(horizon_truncate),
            "resample_streak": int(resample_streak),
            "updated_episode_env_steps": int(updated_episode_env_steps),
        }

    resample_streak = 0
    max_env_steps_remaining = (
        max(int(variant.max_timesteps) - int(updated_episode_env_steps), 0)
        if bool(count_env_steps_for_horizon)
        else int(exec_len)
    )
    effective_exec_len = min(int(exec_len), int(max_env_steps_remaining))
    if effective_exec_len <= 0:
        forced_truncate = True
        horizon_truncate = True
        episode_done = True
        reward = 0.0
        return {
            "reward": reward,
            "backup_reward_total": reward,
            "duration_k": 0,
            "next_raw_obs": raw_obs,
            "env_steps": 0,
            "env_reward_total": 0.0,
            "success": False,
            "episode_done": bool(episode_done),
            "done_for_backup": False,
            "forced_truncate": bool(forced_truncate),
            "env_terminal": bool(env_terminal),
            "horizon_truncate": bool(horizon_truncate),
            "resample_streak": int(resample_streak),
            "updated_episode_env_steps": int(updated_episode_env_steps),
        }

    for step_idx in range(effective_exec_len):
        action_t = np.asarray(planned_actions[step_idx], dtype=np.float32)
        conditioning_action_t = np.asarray(conditioning_actions[step_idx], dtype=np.float32)
        try:
            if "libero" in variant.env:
                next_raw_obs, env_reward, env_done, _ = env.step(action_t)
            else:
                raise NotImplementedError("AAC offline v1 is LIBERO-only")
        except ValueError as exc:
            if (not bool(catch_terminated_episode)) or "terminated episode" not in str(exc):
                raise
            episode_done = True
            done_for_backup = False
            env_terminal = True
            break

        action_history.append(conditioning_action_t.copy())
        env_reward_total += float(env_reward)
        step_reward = _aac_step_reward(env_reward, variant)
        backup_reward_total += current_step_gamma * step_reward
        current_step_gamma *= float(variant.discount)
        env_steps += 1
        if bool(count_env_steps_for_horizon):
            updated_episode_env_steps += 1
            variant._episode_env_steps = int(updated_episode_env_steps)
        success = bool(env_reward == variant.env_max_reward)
        if success:
            episode_done = True
            done_for_backup = True
            env_terminal = True
            break
        if env_done:
            episode_done = True
            done_for_backup = False
            env_terminal = True
            break
        if bool(count_env_steps_for_horizon) and int(updated_episode_env_steps) >= int(
            variant.max_timesteps
        ):
            episode_done = True
            done_for_backup = False
            horizon_truncate = True
            break

    reward = float(backup_reward_total)
    return {
        "reward": reward,
        "backup_reward_total": float(backup_reward_total),
        "duration_k": int(env_steps),
        "next_raw_obs": next_raw_obs,
        "env_steps": int(env_steps),
        "env_reward_total": float(env_reward_total),
        "success": bool(success),
        "episode_done": bool(episode_done),
        "done_for_backup": bool(done_for_backup),
        "forced_truncate": bool(forced_truncate),
        "env_terminal": bool(env_terminal),
        "horizon_truncate": bool(horizon_truncate),
        "resample_streak": int(resample_streak),
        "updated_episode_env_steps": int(updated_episode_env_steps),
    }
