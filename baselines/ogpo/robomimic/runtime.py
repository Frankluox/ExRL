"""Validate the isolated simulator stack used by an OGPO process."""

import os

from exrl.runners.dp_q.vector_env import check_environment_dependencies


def validate_runtime(dsrl_root, required_robosuite="", required_mujoco=""):
    if not required_robosuite and not required_mujoco:
        return {}
    check_environment_dependencies(dsrl_root)
    import mujoco
    import robosuite
    import torch

    loaded_path = os.path.realpath(robosuite.__file__)
    if required_robosuite and robosuite.__version__ != required_robosuite:
        raise RuntimeError(
            f"robosuite version mismatch: {robosuite.__version__} != {required_robosuite}"
        )
    if required_mujoco and (
        mujoco.__version__ != required_mujoco or mujoco.mj_versionString() != required_mujoco
    ):
        raise RuntimeError(
            f"MuJoCo Python/native mismatch: {mujoco.__version__}/{mujoco.mj_versionString()}"
        )
    return {
        "robosuite_version": robosuite.__version__,
        "robosuite_path": loaded_path,
        "mujoco_version": mujoco.__version__,
        "mujoco_native": mujoco.mj_versionString(),
        "torch_version": torch.__version__,
    }
