"""Strict task contracts for DICE-RL around frozen ACT base policies.

The learner remains task agnostic. These contracts are copied from the
audited state-based ACT ResFiT transfer and extended only with the DICE-RL
algorithm/evaluation settings used by this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


SINGLE_ARM_LOWDIM_KEYS: Tuple[str, ...] = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
)

TRANSPORT_LOWDIM_KEYS: Tuple[str, ...] = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot1_eef_pos",
    "robot1_eef_quat",
    "robot1_gripper_qpos",
    "object",
)


@dataclass(frozen=True)
class DICEACTTaskContract:
    env_name: str
    state_dim: int
    action_dim: int
    lowdim_keys: Tuple[str, ...]
    env_horizon: int
    gamma: float
    eval_interval_primitive_steps: int
    actor_hidden_dims: Tuple[int, ...]
    critic_hidden_dims: Tuple[int, ...]
    offline_success_mode: str
    offline_primitive_transitions: int
    late_success_episodes: int
    offline_chunk_transitions: Dict[int, int]
    offline_partial_terminal_chunks: Dict[int, int]
    n_step: int = 3
    reward_offset: float = 1.0
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
    offline_episodes: int = 300

    @property
    def periodic_eval_warmup_primitive_steps(self) -> int:
        """Last primitive step at which periodic evaluation is suppressed."""

        return max(
            int(self.q_filter_warmup_primitive_steps),
            int(self.best_of_n_warmup_primitive_steps),
        )


TASK_CONTRACTS: Dict[str, DICEACTTaskContract] = {
    "lift": DICEACTTaskContract(
        env_name="lift",
        state_dim=19,
        action_dim=7,
        lowdim_keys=SINGLE_ARM_LOWDIM_KEYS,
        env_horizon=300,
        gamma=0.997,
        eval_interval_primitive_steps=30_000,
        actor_hidden_dims=(1024, 1024, 1024),
        critic_hidden_dims=(1024, 1024, 1024),
        offline_success_mode="last_frame",
        offline_primitive_transitions=31_127,
        late_success_episodes=2,
        offline_chunk_transitions={32: 1_115, 50: 753},
        offline_partial_terminal_chunks={32: 289, 50: 288},
    ),
    "can": DICEACTTaskContract(
        env_name="can",
        state_dim=23,
        action_dim=7,
        lowdim_keys=SINGLE_ARM_LOWDIM_KEYS,
        env_horizon=300,
        gamma=0.997,
        eval_interval_primitive_steps=30_000,
        actor_hidden_dims=(1024, 1024, 1024),
        critic_hidden_dims=(1024, 1024, 1024),
        offline_success_mode="last_frame",
        offline_primitive_transitions=62_756,
        late_success_episodes=39,
        offline_chunk_transitions={32: 2_109, 50: 1_403},
        offline_partial_terminal_chunks={32: 290, 50: 291},
    ),
    "square": DICEACTTaskContract(
        env_name="square",
        state_dim=23,
        action_dim=7,
        lowdim_keys=SINGLE_ARM_LOWDIM_KEYS,
        env_horizon=400,
        gamma=0.997,
        eval_interval_primitive_steps=100_000,
        actor_hidden_dims=(1024, 1024, 1024),
        critic_hidden_dims=(1024, 1024, 1024),
        offline_success_mode="reward",
        offline_primitive_transitions=79_520,
        late_success_episodes=30,
        offline_chunk_transitions={32: 2_633, 50: 1_744},
        offline_partial_terminal_chunks={32: 289, 50: 296},
    ),
    "transport": DICEACTTaskContract(
        env_name="transport",
        state_dim=59,
        action_dim=14,
        lowdim_keys=TRANSPORT_LOWDIM_KEYS,
        env_horizon=800,
        gamma=0.999,
        eval_interval_primitive_steps=200_000,
        actor_hidden_dims=(2048, 2048, 2048),
        critic_hidden_dims=(2048, 2048, 2048),
        offline_success_mode="reward",
        offline_primitive_transitions=194_425,
        late_success_episodes=36,
        offline_chunk_transitions={32: 6_225, 50: 4_033},
        offline_partial_terminal_chunks={32: 295, 50: 292},
    ),
    "tool_hang": DICEACTTaskContract(
        env_name="tool_hang",
        state_dim=53,
        action_dim=7,
        lowdim_keys=SINGLE_ARM_LOWDIM_KEYS,
        env_horizon=800,
        gamma=0.999,
        eval_interval_primitive_steps=200_000,
        actor_hidden_dims=(1024, 1024, 1024),
        critic_hidden_dims=(1024, 1024, 1024),
        offline_success_mode="reward",
        offline_primitive_transitions=94_591,
        late_success_episodes=0,
        offline_chunk_transitions={32: 3_054, 50: 1_993},
        offline_partial_terminal_chunks={32: 193, 50: 198},
        offline_episodes=200,
    ),
}


def get_task_contract(env_name: str) -> DICEACTTaskContract:
    key = str(env_name).strip().lower()
    try:
        return TASK_CONTRACTS[key]
    except KeyError as exc:
        raise ValueError(
            "unsupported strict DICE ACT task %r; expected one of %s"
            % (env_name, sorted(TASK_CONTRACTS))
        ) from exc


def validate_horizon(contract: DICEACTTaskContract, horizon_steps: int) -> int:
    horizon = int(horizon_steps)
    if horizon not in contract.offline_chunk_transitions:
        raise ValueError(
            "%s DICE supports only audited TA%s"
            % (contract.env_name, sorted(contract.offline_chunk_transitions))
        )
    return horizon


def expected_offline_transitions(contract: DICEACTTaskContract, horizon_steps: int) -> int:
    return int(contract.offline_chunk_transitions[validate_horizon(contract, horizon_steps)])
