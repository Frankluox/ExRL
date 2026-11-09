"""Strict task contracts for state-based ACT ResFiT experiments.

The learner remains task agnostic.  These contracts keep the task-dependent
base-policy settings and the requested residual-network widths in one
auditable location.
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
class ResFiTACTTaskContract:
    env_name: str
    state_dim: int
    action_dim: int
    lowdim_keys: Tuple[str, ...]
    env_horizon: int
    gamma: float
    residual_state_mode: str
    actor_hidden_dims: Tuple[int, ...]
    critic_hidden_dims: Tuple[int, ...]
    critic_use_layernorm: bool
    offline_success_mode: str
    offline_primitive_transitions: int
    late_success_episodes: int
    offline_chunk_transitions: Dict[int, int]
    offline_partial_terminal_chunks: Dict[int, int]
    offline_episodes: int = 300
    offline_replay_capacity_floor: int = 3_000


TASK_CONTRACTS: Dict[str, ResFiTACTTaskContract] = {
    "lift": ResFiTACTTaskContract(
        env_name="lift",
        state_dim=19,
        action_dim=7,
        lowdim_keys=SINGLE_ARM_LOWDIM_KEYS,
        env_horizon=300,
        gamma=0.997,
        residual_state_mode="stats_no_clip",
        actor_hidden_dims=(1024, 1024, 1024),
        critic_hidden_dims=(1024, 1024, 1024),
        critic_use_layernorm=False,
        # The authoritative robomimic v1.4.1 MH source has no reward/done
        # arrays.  ResFiT's converter explicitly marks its last frame done.
        offline_success_mode="last_frame",
        offline_primitive_transitions=31_127,
        late_success_episodes=2,
        offline_chunk_transitions={32: 1_115, 50: 753},
        offline_partial_terminal_chunks={32: 289, 50: 288},
    ),
    "can": ResFiTACTTaskContract(
        env_name="can",
        state_dim=23,
        action_dim=7,
        lowdim_keys=SINGLE_ARM_LOWDIM_KEYS,
        env_horizon=300,
        gamma=0.997,
        residual_state_mode="stats_no_clip",
        actor_hidden_dims=(1024, 1024, 1024),
        critic_hidden_dims=(1024, 1024, 1024),
        critic_use_layernorm=False,
        offline_success_mode="last_frame",
        offline_primitive_transitions=62_756,
        late_success_episodes=39,
        offline_chunk_transitions={32: 2_109, 50: 1_403},
        offline_partial_terminal_chunks={32: 290, 50: 291},
    ),
    "square": ResFiTACTTaskContract(
        env_name="square",
        state_dim=23,
        action_dim=7,
        lowdim_keys=SINGLE_ARM_LOWDIM_KEYS,
        env_horizon=400,
        gamma=0.997,
        residual_state_mode="stats_no_clip",
        actor_hidden_dims=(1024, 1024, 1024),
        critic_hidden_dims=(1024, 1024, 1024),
        critic_use_layernorm=False,
        offline_success_mode="reward",
        offline_primitive_transitions=79_520,
        late_success_episodes=30,
        offline_chunk_transitions={32: 2_633, 50: 1_744},
        offline_partial_terminal_chunks={32: 289, 50: 296},
    ),
    "transport": ResFiTACTTaskContract(
        env_name="transport",
        state_dim=59,
        action_dim=14,
        lowdim_keys=TRANSPORT_LOWDIM_KEYS,
        env_horizon=800,
        gamma=0.999,
        residual_state_mode="stats_no_clip",
        actor_hidden_dims=(2048, 2048, 2048),
        critic_hidden_dims=(2048, 2048, 2048),
        critic_use_layernorm=False,
        offline_success_mode="reward",
        offline_primitive_transitions=194_425,
        late_success_episodes=36,
        offline_chunk_transitions={32: 6_225, 50: 4_033},
        offline_partial_terminal_chunks={32: 295, 50: 292},
    ),
    "tool_hang": ResFiTACTTaskContract(
        env_name="tool_hang",
        state_dim=53,
        action_dim=7,
        lowdim_keys=SINGLE_ARM_LOWDIM_KEYS,
        env_horizon=800,
        gamma=0.999,
        residual_state_mode="dsrl_raw",
        actor_hidden_dims=(2048, 2048, 2048),
        critic_hidden_dims=(2048, 2048, 2048),
        critic_use_layernorm=True,
        offline_success_mode="reward",
        offline_primitive_transitions=94_591,
        late_success_episodes=0,
        offline_chunk_transitions={32: 3_054, 50: 1_993},
        offline_partial_terminal_chunks={32: 193, 50: 198},
        offline_episodes=200,
        # Match the audited DP Tool Hang ResFiT capacities exactly:
        # TA32=4054 and TA50=2993.
        offline_replay_capacity_floor=0,
    ),
}


def get_task_contract(env_name: str) -> ResFiTACTTaskContract:
    key = str(env_name).strip().lower()
    try:
        return TASK_CONTRACTS[key]
    except KeyError as exc:
        raise ValueError(
            "unsupported strict ACT ResFiT task %r; expected one of %s"
            % (env_name, sorted(TASK_CONTRACTS))
        ) from exc


def expected_offline_transitions(
    contract: ResFiTACTTaskContract,
    residual_granularity: str,
    base_horizon_steps: int,
) -> int:
    if str(residual_granularity).lower() == "primitive":
        return int(contract.offline_primitive_transitions)
    try:
        return int(contract.offline_chunk_transitions[int(base_horizon_steps)])
    except KeyError as exc:
        raise ValueError(
            "%s has no audited offline count for TA%d"
            % (contract.env_name, int(base_horizon_steps))
        ) from exc
