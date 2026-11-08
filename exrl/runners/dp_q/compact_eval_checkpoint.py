"""Build and validate compact, evaluation-only ExRL-Q checkpoints.

Full training checkpoints intentionally contain replay buffers, optimizer states,
target networks, and RNG state so that training can resume exactly.  None of
those objects are required by the immutable critic evaluator.  This module
extracts the policy and critic weights that evaluator consumes while keeping
the full ``latest_model.pt`` format unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional

import torch


FORMAT = "dp_exrlq_critic_eval_snapshot_v1"
CHECKPOINT_KIND = "eval_only_compact_v1"


def _first_present(state: Mapping[str, Any], *keys: str):
    for key in keys:
        value = state.get(key)
        if value is not None:
            return value
    return None


def _phase_id(state: Mapping[str, Any]) -> int:
    value = state.get("phase_id")
    if value is None and isinstance(state.get("offline_phase_state"), Mapping):
        value = state["offline_phase_state"].get("phase_id")
    return int(value or 0)


def is_compactable_training_checkpoint(state: Any) -> bool:
    """Return whether *state* is a full discrete ExRL-Q training checkpoint."""

    if not isinstance(state, Mapping):
        return False
    if state.get("checkpoint_kind") == CHECKPOINT_KIND or state.get("format") == FORMAT:
        return False
    sac_state = state.get("sac_state")
    return (
        isinstance(sac_state, Mapping)
        and isinstance(sac_state.get("critic"), Mapping)
        and "replay_state" in state
    )


def make_compact_eval_checkpoint(
    state: Mapping[str, Any],
    args: Mapping[str, Any],
    *,
    source_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract the exact weights and metadata needed by the critic evaluator."""

    if not is_compactable_training_checkpoint(state):
        raise ValueError(
            "checkpoint is not a supported full ExRL-Q training checkpoint"
        )
    sac_state = state["sac_state"]
    base_policy_state = _first_present(
        state, "bc_live_policy_state", "base_policy_state"
    )
    env_steps = int(state.get("env_steps", state.get("actual_env_steps", 0)) or 0)
    policy_steps = int(state.get("policy_steps", 0) or 0)
    critic_updates = int(
        state.get("num_updates_total", state.get("critic_updates", 0)) or 0
    )
    compact = {
        "format": FORMAT,
        "checkpoint_kind": CHECKPOINT_KIND,
        "args": dict(args),
        "base_policy_state": base_policy_state,
        "critic_state": sac_state["critic"],
        "eval_axis": env_steps,
        "actual_env_steps": env_steps,
        "policy_steps": policy_steps,
        "critic_updates": critic_updates,
        "bc_update_count": int(state.get("bc_update_count", 0) or 0),
        "phase_id": _phase_id(state),
        "best_success": float(state.get("best_success", 0.0) or 0.0),
        "best_checkpoint_metric": state.get("best_checkpoint_metric"),
        "run_dir": state.get("run_dir"),
        "created_at": state.get("created_at"),
    }
    if source_checkpoint is not None:
        compact["source_checkpoint"] = str(source_checkpoint)
    return compact


def _assert_state_dict_equal(expected: Any, actual: Any, path: str) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(
                f"{path}: expected tensor, got {type(actual).__name__}"
            )
        if expected.dtype != actual.dtype or tuple(expected.shape) != tuple(
            actual.shape
        ):
            raise AssertionError(
                f"{path}: tensor metadata differs: "
                f"{expected.dtype}{tuple(expected.shape)} != {actual.dtype}{tuple(actual.shape)}"
            )
        if not torch.equal(expected.detach().cpu(), actual.detach().cpu()):
            raise AssertionError(f"{path}: tensor values differ")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise AssertionError(
                f"{path}: expected mapping, got {type(actual).__name__}"
            )
        if set(expected) != set(actual):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise AssertionError(
                f"{path}: keys differ; missing={missing}, extra={extra}"
            )
        for key in expected:
            _assert_state_dict_equal(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(expected) != len(actual):
            raise AssertionError(f"{path}: sequence differs")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _assert_state_dict_equal(expected_item, actual_item, f"{path}[{index}]")
        return
    if expected != actual:
        raise AssertionError(f"{path}: {expected!r} != {actual!r}")


def verify_compact_checkpoint(
    source: Mapping[str, Any], compact: Mapping[str, Any], args: Mapping[str, Any]
) -> None:
    """Verify that a serialized compact checkpoint preserved every eval tensor."""

    if (
        compact.get("format") != FORMAT
        or compact.get("checkpoint_kind") != CHECKPOINT_KIND
    ):
        raise AssertionError("compact checkpoint format marker is missing")
    if compact.get("args") != dict(args):
        raise AssertionError("compact checkpoint args differ from config.json")
    expected_base = _first_present(source, "bc_live_policy_state", "base_policy_state")
    _assert_state_dict_equal(
        expected_base, compact.get("base_policy_state"), "base_policy_state"
    )
    _assert_state_dict_equal(
        source["sac_state"]["critic"], compact.get("critic_state"), "critic_state"
    )
    forbidden = {
        "replay_state",
        "rng_state",
        "bc_optimizer_state",
        "bc_replay_state",
        "bc_teacher_state",
        "bc_fixed_teacher_state",
        "sac_state",
    }
    present = forbidden.intersection(compact)
    if present:
        raise AssertionError(
            f"training-only fields leaked into compact checkpoint: {sorted(present)}"
        )
