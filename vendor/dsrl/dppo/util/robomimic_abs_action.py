"""Utilities for Cartesian absolute actions used by robomimic policies.

The training representation is ``[position(3), rotation_6d(6), gripper(1)]``.
The executable robosuite OSC_POSE representation is
``[position(3), axis_angle(3), gripper(1)]``.  Rotation-6D follows the
PyTorch3D / Zhou et al. convention: the first two rows of a rotation matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation


ArrayLike = Union[np.ndarray, list, tuple]


def matrix_to_rotation_6d(matrix: ArrayLike) -> np.ndarray:
    """Convert rotation matrices to the first-two-rows 6D representation."""
    matrix = np.asarray(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3), got {matrix.shape}")
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,))


def rotation_6d_to_matrix(rotation_6d: ArrayLike, eps: float = 1e-8) -> np.ndarray:
    """Project arbitrary 6D vectors onto SO(3) using Gram-Schmidt."""
    rotation_6d = np.asarray(rotation_6d)
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Expected (..., 6), got {rotation_6d.shape}")

    a1 = rotation_6d[..., :3]
    a2 = rotation_6d[..., 3:]
    a1_norm = np.linalg.norm(a1, axis=-1, keepdims=True)
    if np.any(a1_norm < eps):
        raise ValueError("Degenerate rotation-6D first vector")
    b1 = a1 / a1_norm

    a2_orthogonal = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    a2_norm = np.linalg.norm(a2_orthogonal, axis=-1, keepdims=True)
    if np.any(a2_norm < eps):
        raise ValueError("Degenerate rotation-6D second vector")
    b2 = a2_orthogonal / a2_norm
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)


def axis_angle_to_rotation_6d(axis_angle: ArrayLike) -> np.ndarray:
    """Convert axis-angle vectors to rotation-6D."""
    axis_angle = np.asarray(axis_angle)
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3), got {axis_angle.shape}")
    flat = axis_angle.reshape(-1, 3)
    matrices = Rotation.from_rotvec(flat).as_matrix().reshape(
        axis_angle.shape[:-1] + (3, 3)
    )
    return matrix_to_rotation_6d(matrices)


def rotation_6d_to_axis_angle(rotation_6d: ArrayLike) -> np.ndarray:
    """Convert rotation-6D vectors to principal axis-angle vectors."""
    rotation_6d = np.asarray(rotation_6d)
    matrices = rotation_6d_to_matrix(rotation_6d)
    flat = matrices.reshape(-1, 3, 3)
    return Rotation.from_matrix(flat).as_rotvec().reshape(
        rotation_6d.shape[:-1] + (3,)
    )


def axis_angle_action_to_rotation_6d(action: ArrayLike) -> np.ndarray:
    """Convert ``[..., 7]`` absolute OSC actions to ``[..., 10]``."""
    action = np.asarray(action)
    if action.shape[-1] != 7:
        raise ValueError(f"Expected (..., 7), got {action.shape}")
    return np.concatenate(
        (action[..., :3], axis_angle_to_rotation_6d(action[..., 3:6]), action[..., 6:7]),
        axis=-1,
    )


def rotation_6d_action_to_axis_angle(action: ArrayLike) -> np.ndarray:
    """Convert ``[..., 10]`` training actions to executable ``[..., 7]``."""
    action = np.asarray(action)
    if action.shape[-1] != 10:
        raise ValueError(f"Expected (..., 10), got {action.shape}")
    return np.concatenate(
        (action[..., :3], rotation_6d_to_axis_angle(action[..., 3:9]), action[..., 9:10]),
        axis=-1,
    )


def normalize(values: ArrayLike, minimum: ArrayLike, maximum: ArrayLike) -> np.ndarray:
    """Match DSRL's min-max mapping to [-1, 1]."""
    values = np.asarray(values)
    minimum = np.asarray(minimum)
    maximum = np.asarray(maximum)
    return 2.0 * (values - minimum) / (maximum - minimum + 1e-6) - 1.0


def unnormalize(values: ArrayLike, minimum: ArrayLike, maximum: ArrayLike) -> np.ndarray:
    """Invert DSRL's min-max action normalization."""
    values = np.asarray(values)
    minimum = np.asarray(minimum)
    maximum = np.asarray(maximum)
    return (values + 1.0) * 0.5 * (maximum - minimum) + minimum


class Rot6dAbsoluteActionAdapter:
    """Turn normalized per-robot 10D DP outputs into 7D robosuite actions.

    The environment's regular low-dimensional wrapper still owns the final
    unnormalization.  This adapter therefore decodes each raw 10D robot action
    and then normalizes the concatenated 7D-per-robot actions with the
    environment normalization.  Single-arm tasks use 10D -> 7D; Transport uses
    20D -> 14D with the original robot ordering preserved.
    """

    def __init__(
        self,
        source_normalization_path: Union[str, Path],
        target_normalization_path: Union[str, Path],
        clip_source: bool = True,
        clip_target: bool = False,
    ) -> None:
        source = np.load(source_normalization_path, allow_pickle=False)
        target = np.load(target_normalization_path, allow_pickle=False)
        self.source_min = np.asarray(source["action_min"])
        self.source_max = np.asarray(source["action_max"])
        self.target_min = np.asarray(target["action_min"])
        self.target_max = np.asarray(target["action_max"])
        if self.source_min.ndim != 1 or self.source_max.shape != self.source_min.shape:
            raise ValueError("Source normalization must contain matching 1D arrays")
        if self.target_min.ndim != 1 or self.target_max.shape != self.target_min.shape:
            raise ValueError("Target normalization must contain matching 1D arrays")
        if self.source_min.size % 10 != 0:
            raise ValueError(
                "Source normalization must describe concatenated 10D rot6d actions"
            )
        self.num_robots = self.source_min.size // 10
        if self.target_min.size != 7 * self.num_robots:
            raise ValueError(
                "Target normalization must describe matching 7D axis-angle actions"
            )
        self.clip_source = bool(clip_source)
        self.clip_target = bool(clip_target)

    def __call__(self, normalized_action: ArrayLike) -> np.ndarray:
        normalized_action = np.asarray(normalized_action)
        expected_dim = 10 * self.num_robots
        if normalized_action.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected normalized (..., {expected_dim}) action, "
                f"got {normalized_action.shape}"
            )
        if self.clip_source:
            normalized_action = np.clip(normalized_action, -1.0, 1.0)
        raw_rot6d = unnormalize(normalized_action, self.source_min, self.source_max)
        raw_axis_angle = np.concatenate(
            [
                rotation_6d_action_to_axis_angle(
                    raw_rot6d[..., 10 * index : 10 * (index + 1)]
                )
                for index in range(self.num_robots)
            ],
            axis=-1,
        )
        normalized_axis_angle = normalize(
            raw_axis_angle, self.target_min, self.target_max
        )
        if self.clip_target:
            normalized_axis_angle = np.clip(normalized_axis_angle, -1.0, 1.0)
        return normalized_axis_angle.astype(normalized_action.dtype, copy=False)
