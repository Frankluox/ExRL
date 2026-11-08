from __future__ import annotations

CONFIG_NAME = "config.json"
MODEL_FILE_NAME = "model.safetensors"

OBS_STR = "observation"
OBS_PREFIX = OBS_STR + "."
OBS_ENV_STATE = OBS_STR + ".environment_state"
OBS_STATE = OBS_STR + ".state"
OBS_IMAGE = OBS_STR + ".image"
OBS_IMAGES = OBS_STR + ".images"

ACTION = "action"
REWARD = "next.reward"
DONE = "next.done"
TRUNCATED = "next.truncated"
INFO = "info"

POLICY_PREPROCESSOR_DEFAULT_NAME = "policy_preprocessor"
POLICY_POSTPROCESSOR_DEFAULT_NAME = "policy_postprocessor"
