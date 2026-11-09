"""Task dimensions and default hyperparameters for Adroit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class AdroitDICETaskContract:
    env_name: str
    env_id: str
    state_dim: int
    action_dim: int
    offline_episodes: int
    offline_primitive_transitions: int
    offline_success_episodes: int
    offline_timeout_episodes: int
    offline_failure_terminal_episodes: int
    offline_chunk_transitions: Dict[int, int]
    offline_partial_terminal_chunks: Dict[int, int]
    checkpoint_epoch: int = 2
    lowdim_keys: Tuple[str, ...] = ("flat",)
    env_horizon: int = 200
    dp_infer_steps: int = 5
    gamma: float = 0.997
    eval_interval_primitive_steps: int = 200_000
    actor_hidden_dims: Tuple[int, ...] = (1024, 1024, 1024)
    critic_hidden_dims: Tuple[int, ...] = (1024, 1024, 1024)
    n_step: int = 3
    reward_offset: float = 0.0
    n_envs: int = 4
    batch_size: int = 256
    gradient_steps_per_vector_step: int = 10
    actor_update_frequency: int = 2
    critic_ensemble_size: int = 10
    critic_use_layernorm: bool = False
    learning_rate: float = 1e-4
    tau: float = 0.01
    beta: float = 50.0
    q_filter_epsilon: float = -0.5
    actor_num_noise_samples: int = 16
    target_num_noise_samples: int = 4
    rollout_num_noise_samples: int = 16
    offline_ratio_start: float = 0.7
    offline_ratio_end: float = 0.1
    offline_ratio_decay_primitive_steps: int = 160_000
    q_filter_warmup_primitive_steps: int = 32_000
    best_of_n_warmup_primitive_steps: int = 64_000
    offline_success_mode: str = "minari_infos"

    @property
    def periodic_eval_warmup_primitive_steps(self) -> int:
        return max(
            int(self.q_filter_warmup_primitive_steps),
            int(self.best_of_n_warmup_primitive_steps),
        )


TASK_CONTRACTS: Dict[str, AdroitDICETaskContract] = {
    "door": AdroitDICETaskContract(
        env_name="door",
        env_id="AdroitHandDoor-v1",
        state_dim=39,
        action_dim=28,
        offline_episodes=5000,
        offline_primitive_transitions=286_657,
        offline_success_episodes=4897,
        offline_timeout_episodes=103,
        offline_failure_terminal_episodes=0,
        offline_chunk_transitions={32: 10_717},
        offline_partial_terminal_chunks={32: 4_983},
    ),
    "hammer": AdroitDICETaskContract(
        env_name="hammer",
        env_id="AdroitHandHammer-v1",
        state_dim=46,
        action_dim=26,
        offline_episodes=5000,
        offline_primitive_transitions=403_815,
        offline_success_episodes=3876,
        offline_timeout_episodes=1124,
        offline_failure_terminal_episodes=0,
        offline_chunk_transitions={32: 15_998},
        offline_partial_terminal_chunks={32: 4_990},
    ),
    "pen": AdroitDICETaskContract(
        env_name="pen",
        env_id="AdroitHandPen-v1",
        state_dim=45,
        action_dim=24,
        offline_episodes=4958,
        offline_primitive_transitions=233_686,
        offline_success_episodes=4130,
        offline_timeout_episodes=828,
        offline_failure_terminal_episodes=0,
        offline_chunk_transitions={32: 9_952},
        offline_partial_terminal_chunks={32: 4_856},
    ),
    "relocate": AdroitDICETaskContract(
        env_name="relocate",
        env_id="AdroitHandRelocate-v1",
        state_dim=39,
        action_dim=30,
        offline_episodes=5000,
        offline_primitive_transitions=249_624,
        offline_success_episodes=4934,
        offline_timeout_episodes=66,
        offline_failure_terminal_episodes=0,
        offline_chunk_transitions={32: 10_437},
        offline_partial_terminal_chunks={32: 4_970},
        checkpoint_epoch=10,
    ),
}


def is_adroit_task(env_name: str) -> bool:
    key = str(env_name).strip().lower()
    return key in TASK_CONTRACTS or any(
        key == contract.env_id.lower() for contract in TASK_CONTRACTS.values()
    )


def get_adroit_task_contract(env_name: str) -> AdroitDICETaskContract:
    key = str(env_name).strip().lower()
    if key in TASK_CONTRACTS:
        return TASK_CONTRACTS[key]
    for contract in TASK_CONTRACTS.values():
        if key == contract.env_id.lower():
            return contract
    raise ValueError("unsupported Adroit DICE task %r" % env_name)


def validate_adroit_horizon(contract: AdroitDICETaskContract, horizon_steps: int) -> int:
    horizon = int(horizon_steps)
    if horizon not in contract.offline_chunk_transitions:
        raise ValueError(
            "%s Adroit DICE supports only audited TA%s"
            % (contract.env_name, sorted(contract.offline_chunk_transitions))
        )
    return horizon


def expected_adroit_offline_transitions(
    contract: AdroitDICETaskContract, horizon_steps: int
) -> int:
    return int(contract.offline_chunk_transitions[validate_adroit_horizon(contract, horizon_steps)])
