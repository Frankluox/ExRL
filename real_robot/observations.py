"""Robot-independent image, state, and action-chunk inputs for ExRL."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def resize(image, size):
    import cv2

    return cv2.resize(np.asarray(image), (size, size), interpolation=cv2.INTER_AREA)


def load_normalization(path):
    payload = json.loads(Path(path).read_text())
    source = payload.get("norm_stats", payload)
    if "actions" in source:
        return {
            f"{prefix}_{field}": np.asarray(source[key][field], np.float32)
            for key, prefix in (("state", "state"), ("actions", "action"))
            for field in ("mean", "std")
        }
    return {
        key: np.asarray(source[key], np.float32)
        for key in ("state_mean", "state_std", "action_mean", "action_std")
    }


def critic_observation(
    frame,
    chunk,
    stats,
    image_size=128,
    action_mode="absolute",
    camera_keys=("image",),
    action_state_indices=(),
):
    actions = np.asarray(chunk, dtype=np.float32).copy()
    state = np.asarray(frame["state"], dtype=np.float32)
    if (
        state.ndim != 1
        or actions.ndim != 2
        or not np.isfinite(state).all()
        or not np.isfinite(actions).all()
    ):
        raise ValueError("Expected a finite state vector and an H x action_dim chunk")
    if action_mode == "state_delta":
        if not action_state_indices:
            raise ValueError("state_delta requires explicit [action_index, state_index] pairs")
        for action_index, state_index in action_state_indices:
            actions[:, action_index] -= state[state_index]
    elif action_mode != "absolute":
        raise ValueError("action_mode must be state_delta or absolute")

    def normalize(value, prefix):
        size = value.shape[-1]
        mean = stats[prefix + "_mean"][:size]
        std = stats[prefix + "_std"][:size]
        if (
            mean.shape != (size,)
            or std.shape != (size,)
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std < 0)
        ):
            raise ValueError(f"Invalid {prefix} normalization dimensions or values")
        return np.clip((value - mean) / (std + 1e-6), -10, 10).astype(np.float32)

    images = []
    for key in camera_keys:
        image = np.asarray(frame[key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{key} must be a uint8 RGB image")
        images.append(resize(image, image_size))
    if not images:
        raise ValueError("At least one camera is required by the image-based critic")
    return {
        "pixels": np.concatenate(images, axis=-1)[..., None],
        "state": normalize(state, "state")[:, None],
        "actions": normalize(actions, "action"),
    }
