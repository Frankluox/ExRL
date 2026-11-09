"""Piper coordinates and the inputs used by the execution-length critic."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

RAW_PER_RAD = 180_000.0 / np.pi
GRIPPER_OPEN_RAW = 70_000.0


def state_from_raw(raw, joint_offset=None):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (7,) or not np.isfinite(raw).all():
        raise ValueError("Expected seven finite Piper joint/gripper coordinates")
    state = raw.copy()
    state[:6] /= RAW_PER_RAD
    if joint_offset is not None:
        state[:6] += np.asarray(joint_offset, dtype=np.float32)
    state[6] = 1.0 - np.clip(raw[6] / GRIPPER_OPEN_RAW, 0.0, 1.0)
    return state


def action_to_raw(action, joint_offset=None):
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("Expected a finite seven-dimensional policy action")
    joints = action[:6].copy()
    if joint_offset is not None:
        joints -= np.asarray(joint_offset, dtype=np.float32)
    return np.rint(
        np.r_[joints * RAW_PER_RAD, (1.0 - np.clip(action[6], 0, 1)) * GRIPPER_OPEN_RAW]
    ).astype(np.int64)


def resize(image, size):
    import cv2

    return cv2.resize(np.asarray(image), (size, size), interpolation=cv2.INTER_AREA)


def load_normalization(path):
    payload = json.loads(Path(path).read_text())
    source = payload.get("norm_stats", payload)
    if "actions" in source:
        return {
            f"{prefix}_{field}": np.asarray(source[key][field], np.float32)[:7]
            for key, prefix in (("state", "state"), ("actions", "action"))
            for field in ("mean", "std")
        }
    return {
        key: np.asarray(source[key], np.float32)
        for key in ("state_mean", "state_std", "action_mean", "action_std")
    }


def critic_observation(frame, chunk, stats, image_size=128, action_mode="state_delta"):
    actions = np.asarray(chunk, dtype=np.float32).copy()
    state = np.asarray(frame["state"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError("The critic expects an H by 7 action chunk")
    if action_mode == "state_delta":
        actions[:, :6] -= state[None, :6]
    elif action_mode != "absolute":
        raise ValueError("action_mode must be state_delta or absolute")

    def normalize(value, prefix):
        return np.clip(
            (value - stats[prefix + "_mean"]) / (stats[prefix + "_std"] + 1e-6), -10, 10
        ).astype(np.float32)

    pixels = np.concatenate(
        [resize(frame[k], image_size) for k in ("image", "wrist_image")], axis=-1
    )
    return {
        "pixels": pixels[..., None].astype(np.uint8),
        "state": normalize(state, "state")[:, None],
        "actions": normalize(actions, "action"),
    }
