"""Task dimensions and default hyperparameters for Adroit."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AdroitTaskContract:
    env_name: str
    env_id: str
    state_dim: int
    action_dim: int
    checkpoint_epoch: int = 2
    online_steps: int = 10_000_000
    env_horizon: int = 200
    base_horizon_steps: int = 32
    dp_infer_steps: int = 5
    gamma: float = 0.997
    critic_hidden_dims: tuple = (512, 512, 512)
    critic_use_layernorm: bool = True


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
        online_steps=9_800_000,
    ),
    "pen": AdroitTaskContract(
        "pen",
        "AdroitHandPen-v1",
        45,
        24,
        online_steps=9_000_000,
    ),
    "relocate": AdroitTaskContract(
        "relocate",
        "AdroitHandRelocate-v1",
        39,
        30,
        checkpoint_epoch=10,
        online_steps=9_000_000,
    ),
}


def get_task_contract(env_name):
    for task, contract in TASK_CONTRACTS.items():
        if str(env_name).lower() in (task, contract.env_id.lower()):
            return contract
    raise ValueError("expected Adroit task: " + ", ".join(TASK_CONTRACTS))
