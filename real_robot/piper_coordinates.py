"""Coordinate conversion for the optional Piper adapter."""

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
