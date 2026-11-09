from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from exrl.pixel.data.dataset import DatasetDict
from exrl.pixel.types import Params


def _reduce_q(qs, critic_reduction: str):
    if critic_reduction == "min":
        return qs.min(axis=0)
    if critic_reduction == "mean":
        return qs.mean(axis=0)
    raise ValueError(f"Invalid critic reduction: {critic_reduction}")


def _transition_weights(durations: jnp.ndarray, mode: str) -> jnp.ndarray:
    durations = jnp.asarray(durations, dtype=jnp.float32).reshape(-1)
    if mode == "none":
        weights = jnp.ones_like(durations)
    elif mode == "duration":
        weights = durations
    elif mode == "sqrt_duration":
        weights = jnp.sqrt(durations)
    else:
        raise ValueError(f"Invalid transition weight mode: {mode}")
    return weights / jnp.clip(weights.mean(), min=1e-6)


def _validate_action_id_bounds(
    legal_min_action_id: int,
    actor_prior_min_action_id: int,
    action_dim: int,
) -> None:
    legal_min_action_id = int(legal_min_action_id)
    actor_prior_min_action_id = int(actor_prior_min_action_id)
    action_dim = int(action_dim)
    if legal_min_action_id < 0 or legal_min_action_id >= action_dim:
        if legal_min_action_id != 0:
            raise RuntimeError(f"legal_min_action_id out of range: {legal_min_action_id}")
    if not (0 <= actor_prior_min_action_id < action_dim):
        raise RuntimeError(f"actor_prior_min_action_id out of range: {actor_prior_min_action_id}")


def _actor_stats(
    actor: TrainState,
    critic: TrainState,
    temp: TrainState,
    observations,
    durations,
    critic_reduction: str,
    transition_weight_mode: str,
    legal_min_action_id: int,
    actor_prior_coeff: float,
    actor_prior_min_action_id: int,
    action_id_offset: int,
    actor_params: Params,
    training: bool,
):
    input_collections = {"params": actor_params}
    if hasattr(actor, "batch_stats") and actor.batch_stats is not None:
        input_collections["batch_stats"] = actor.batch_stats
        if training:
            logits, new_model_state = actor.apply_fn(
                input_collections,
                observations,
                training=True,
                min_action_id=legal_min_action_id,
                return_logits=True,
                mutable=["batch_stats"],
            )
        else:
            logits = actor.apply_fn(
                input_collections,
                observations,
                training=False,
                min_action_id=legal_min_action_id,
                return_logits=True,
                mutable=False,
            )
            new_model_state = {}
    else:
        logits = actor.apply_fn(
            input_collections,
            observations,
            training=training,
            min_action_id=legal_min_action_id,
            return_logits=True,
        )
        new_model_state = {}

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)

    critic_inputs = {"params": critic.params}
    if hasattr(critic, "batch_stats") and critic.batch_stats is not None:
        critic_inputs["batch_stats"] = critic.batch_stats
    qs = critic.apply_fn(critic_inputs, observations, training=False)
    q = _reduce_q(qs, critic_reduction=critic_reduction)
    weights = _transition_weights(durations, transition_weight_mode)
    alpha = temp.apply_fn({"params": temp.params})
    actor_loss_per_state = jnp.sum(probs * (alpha * log_probs - q), axis=-1)
    actor_loss = (weights * actor_loss_per_state).mean()

    action_dim = logits.shape[-1]
    _validate_action_id_bounds(legal_min_action_id, actor_prior_min_action_id, action_dim)
    min_id = jnp.asarray(actor_prior_min_action_id, dtype=jnp.int32)
    action_ids = jnp.arange(action_dim, dtype=jnp.int32)
    prior_mask = action_ids >= min_id
    prior = prior_mask.astype(logits.dtype)
    prior = prior / jnp.clip(prior.sum(), min=1.0)
    actor_prior_ce_per_state = -(prior * log_probs).sum(axis=-1)
    actor_prior_ce = (weights * actor_prior_ce_per_state).mean()
    actor_loss = actor_loss + jnp.asarray(actor_prior_coeff, dtype=logits.dtype) * actor_prior_ce

    entropy_per_state = -(probs * log_probs).sum(axis=-1)
    entropy = (weights * entropy_per_state).mean()
    action_offset = jnp.asarray(action_id_offset, dtype=jnp.float32)
    if legal_min_action_id > 0:
        illegal_value = jnp.full_like(q, -1e9)
        masked_q = jnp.where(
            jnp.arange(q.shape[-1], dtype=jnp.int32)[None, :] < legal_min_action_id,
            illegal_value,
            q,
        )
        q_span_slice = q[:, legal_min_action_id:]
    else:
        masked_q = q
        q_span_slice = q
    q_argmax = jnp.argmax(masked_q, axis=-1).astype(jnp.float32) + action_offset
    q_span = q_span_slice.max(axis=-1) - q_span_slice.min(axis=-1)

    info = {
        "actor_loss": actor_loss,
        "entropy": entropy,
        "q_pi_in_actor": (weights * jnp.sum(probs * q, axis=-1)).mean(),
        "policy_exec_len_mean": (
            weights
            * jnp.sum(
                probs * (jnp.arange(logits.shape[-1], dtype=jnp.float32) + action_offset), axis=-1
            )
        ).mean(),
        "q_argmax_k_mean": q_argmax.mean(),
        "q_argmax_k_std": q_argmax.std(),
        "q_argmax_k_min": q_argmax.min(),
        "q_argmax_k_max": q_argmax.max(),
        "q_span_mean": q_span.mean(),
        "q_span_std": q_span.std(),
        "actor_prior_ce": actor_prior_ce,
        "logits_mean": logits.mean(),
        "logits_max": logits.max(),
        "logits_min": logits.min(),
    }
    return actor_loss, info, new_model_state


def update_actor(
    actor: TrainState,
    critic: TrainState,
    temp: TrainState,
    batch: DatasetDict,
    critic_reduction: str = "min",
    transition_weight_mode: str = "none",
    legal_min_action_id: int = 0,
    actor_prior_coeff: float = 0.0,
    actor_prior_min_action_id: int = 0,
    action_id_offset: int = 0,
) -> Tuple[TrainState, Dict[str, float]]:
    def actor_loss_fn(actor_params: Params):
        actor_loss, info, new_model_state = _actor_stats(
            actor=actor,
            critic=critic,
            temp=temp,
            observations=batch["observations"],
            durations=batch["durations"],
            critic_reduction=critic_reduction,
            transition_weight_mode=transition_weight_mode,
            legal_min_action_id=legal_min_action_id,
            actor_prior_coeff=actor_prior_coeff,
            actor_prior_min_action_id=actor_prior_min_action_id,
            action_id_offset=action_id_offset,
            actor_params=actor_params,
            training=True,
        )
        return actor_loss, (info, new_model_state)

    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    if "batch_stats" in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state["batch_stats"])
    else:
        new_actor = actor.apply_gradients(grads=grads)
    return new_actor, info


def evaluate_actor(
    actor: TrainState,
    critic: TrainState,
    temp: TrainState,
    batch: DatasetDict,
    critic_reduction: str = "min",
    transition_weight_mode: str = "none",
    legal_min_action_id: int = 0,
    actor_prior_coeff: float = 0.0,
    actor_prior_min_action_id: int = 0,
    action_id_offset: int = 0,
) -> Dict[str, float]:
    _, info, _ = _actor_stats(
        actor=actor,
        critic=critic,
        temp=temp,
        observations=batch["observations"],
        durations=batch["durations"],
        critic_reduction=critic_reduction,
        transition_weight_mode=transition_weight_mode,
        legal_min_action_id=legal_min_action_id,
        actor_prior_coeff=actor_prior_coeff,
        actor_prior_min_action_id=actor_prior_min_action_id,
        action_id_offset=action_id_offset,
        actor_params=actor.params,
        training=False,
    )
    return info
