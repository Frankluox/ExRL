"""Task dimensions and default hyperparameters for Adroit."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class AdroitTaskContract:
    env_name: str
    env_id: str
    state_dim: int
    action_dim: int
    checkpoint_epoch: int = 2
    env_horizon: int = 200
    base_horizon_steps: int = 32
    dp_infer_steps: int = 5
    gamma: float = 0.997
    residual_state_mode: str = "dp_raw"
    actor_hidden_dims: tuple = (1024, 1024, 1024)
    critic_hidden_dims: tuple = (1024, 1024, 1024)
    critic_use_layernorm: bool = False


TASK_CONTRACTS = {
    "door": AdroitTaskContract(
        "door",
        "AdroitHandDoor-v1",
        39,
        28,
    ),
    "hammer": AdroitTaskContract(
        "hammer",
        "AdroitHandHammer-v1",
        46,
        26,
    ),
    "pen": AdroitTaskContract(
        "pen",
        "AdroitHandPen-v1",
        45,
        24,
    ),
    "relocate": AdroitTaskContract("relocate", "AdroitHandRelocate-v1", 39, 30, 10),
}


def get_task_contract(env_name):
    for task, contract in TASK_CONTRACTS.items():
        if str(env_name).lower() in (task, contract.env_id.lower()):
            return contract
    raise ValueError("expected Adroit task: " + ", ".join(TASK_CONTRACTS))
