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


def _subsample_critic_values_with_replacement(
    rng,
    qs: jnp.ndarray,
    paired_qs,
    subsample_size: int,
):
    ensemble_size = int(qs.shape[0])
    subsample_size = int(subsample_size)
    if subsample_size <= 0 or subsample_size >= ensemble_size:
        return qs, paired_qs
    indices = jax.random.randint(rng, (subsample_size,), 0, ensemble_size)
    qs = jnp.take(qs, indices, axis=0)
    if paired_qs is not None:
        paired_qs = jnp.take(paired_qs, indices, axis=0)
    return qs, paired_qs


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


def _validate_legal_min_action_id(legal_min_action_id: int, action_dim: int) -> None:
    legal_min_action_id = int(legal_min_action_id)
    action_dim = int(action_dim)
    if legal_min_action_id < 0 or legal_min_action_id >= action_dim:
        if legal_min_action_id != 0:
            raise RuntimeError(f"legal_min_action_id out of range: {legal_min_action_id}")


def _mask_q_values(q: jnp.ndarray, min_action_id: int) -> jnp.ndarray:
    min_action_id = int(min_action_id)
    if min_action_id <= 0:
        return q
    action_ids = jnp.arange(q.shape[-1], dtype=jnp.int32)
    illegal = action_ids < min_action_id
    return jnp.where(illegal, jnp.full_like(q, -1e9), q)


def _critic_value_from_q(
    q: jnp.ndarray,
    mode: str,
    min_action_id: int,
    critic_backup_temp: float,
) -> jnp.ndarray:
    q = _mask_q_values(q, min_action_id)
    mode = str(mode)
    if mode == "q_greedy":
        return q.max(axis=-1)
    if mode == "q_soft":
        temp = jnp.asarray(critic_backup_temp, dtype=q.dtype)
        return temp * jax.nn.logsumexp(q / temp, axis=-1)
    raise ValueError(f"Invalid target policy mode: {mode}")


def _cql_loss(
    rng,
    qs: jnp.ndarray,
    actions: jnp.ndarray,
    batch: DatasetDict,
    cql_temp: float,
    cql_ensemble_mode: str,
    target_critic_subsample_size: int,
    use_mc_lower_bound: bool,
    mc_lower_bound_scope: str,
    mc_lower_bound_topk: int,
):
    action_idx = actions[None, :, None]
    q_data = jnp.take_along_axis(qs, action_idx, axis=-1).squeeze(-1)
    q_for_lse = qs
    mc_bound_rate = jnp.asarray(0.0, dtype=qs.dtype)
    mc_bound_actions_per_valid = jnp.asarray(0.0, dtype=qs.dtype)

    if use_mc_lower_bound:
        if "mc_returns" not in batch or "mc_return_valid" not in batch:
            raise RuntimeError("use_mc_lower_bound=True requires replay mc_returns")
        valid = jnp.asarray(batch["mc_return_valid"], dtype=bool)[None, :, None]
        mc = jnp.asarray(batch["mc_returns"], dtype=qs.dtype)[None, :, None]
        action_ids = jnp.arange(qs.shape[-1], dtype=jnp.int32)[None, None, :]
        data_action = actions[None, :, None]
        is_data_action = action_ids == data_action

        if mc_lower_bound_scope == "all":
            scope_mask = jnp.ones_like(is_data_action, dtype=bool)
        elif mc_lower_bound_scope == "non_data":
            scope_mask = jnp.logical_not(is_data_action)
        elif mc_lower_bound_scope == "data":
            scope_mask = is_data_action
        elif mc_lower_bound_scope == "topk_non_data":
            k = min(int(mc_lower_bound_topk), int(qs.shape[-1]))
            rank_q = qs.mean(axis=0)
            threshold = jnp.sort(rank_q, axis=-1)[:, -k][:, None]
            topk_mask = rank_q >= threshold
            scope_mask = topk_mask[None, :, :] & jnp.logical_not(is_data_action)
        else:
            raise RuntimeError(f"Unexpected mc_lower_bound_scope: {mc_lower_bound_scope}")

        apply_mask = valid & scope_mask
        bound_events = (q_for_lse < mc) & apply_mask
        denom = jnp.clip(apply_mask.astype(qs.dtype).sum(), min=1.0)
        mc_bound_rate = bound_events.astype(qs.dtype).sum() / denom
        valid_count = jnp.clip(valid.astype(qs.dtype).sum(), min=1.0)
        mc_bound_actions_per_valid = apply_mask.astype(qs.dtype).sum() / valid_count
        q_for_lse = jnp.where(apply_mask, jnp.maximum(q_for_lse, mc), q_for_lse)

    if cql_ensemble_mode == "target_subsample_with_replacement":
        q_for_lse, q_data = _subsample_critic_values_with_replacement(
            rng,
            q_for_lse,
            q_data,
            target_critic_subsample_size,
        )
    elif cql_ensemble_mode != "all":
        raise ValueError(f"Invalid CQL ensemble mode: {cql_ensemble_mode}")

    temp = jnp.asarray(cql_temp, dtype=qs.dtype)
    cql_ood_unnormalized = temp * jax.nn.logsumexp(q_for_lse / temp, axis=-1)
    cql_ood = cql_ood_unnormalized - temp * jnp.log(
        jnp.asarray(q_for_lse.shape[-1], dtype=qs.dtype)
    )
    cql_diff = cql_ood - q_data
    cql_loss = cql_diff.mean()
    return cql_loss, {
        "cql_loss": cql_loss,
        "cql_loss_unnormalized": (cql_ood_unnormalized - q_data).mean(),
        "cql_ood_q_mean": cql_ood.mean(),
        "cql_ood_q_mean_unnormalized": cql_ood_unnormalized.mean(),
        "cql_data_q_mean": q_data.mean(),
        "cql_ensemble_size": jnp.asarray(q_for_lse.shape[0], dtype=qs.dtype),
        "mc_bound_rate": mc_bound_rate,
        "mc_bound_actions_per_valid": mc_bound_actions_per_valid,
    }


def _td_element_loss(
    td_error: jnp.ndarray, td_loss_type: str, td_huber_delta: float
) -> jnp.ndarray:
    if td_loss_type == "mse":
        return jnp.square(td_error)
    if td_loss_type == "huber":
        beta = jnp.clip(jnp.asarray(td_huber_delta, dtype=td_error.dtype), min=1e-6)
        abs_error = jnp.abs(td_error)
        quadratic = 0.5 * jnp.square(td_error) / beta
        linear = abs_error - 0.5 * beta
        return jnp.where(abs_error < beta, quadratic, linear)
    raise ValueError(f"Invalid td_loss_type: {td_loss_type}")


def update_critic(
    rng,
    actor: TrainState,
    critic: TrainState,
    target_critic: TrainState,
    temp: TrainState,
    batch: DatasetDict,
    critic_reduction: str = "min",
    legal_min_action_id: int = 0,
    transition_weight_mode: str = "none",
    target_policy_mode: str = "actor",
    critic_backup_temp: float = 1.0,
    cql_alpha: float = 0.0,
    cql_temp: float = 1.0,
    target_q_clip_min: float = -1.0e9,
    target_q_clip_max: float = 1.0e9,
    target_ensemble_mode: str = "configured_without_replacement",
    cql_ensemble_mode: str = "all",
    td_loss_type: str = "mse",
    td_huber_delta: float = 10.0,
    use_mc_lower_bound: bool = False,
    mc_lower_bound_scope: str = "topk_non_data",
    mc_lower_bound_topk: int = 10,
    critic_bootstrap_mask_prob: float = 1.0,
    target_critic_subsample_size: int = 0,
) -> Tuple[TrainState, Dict[str, float]]:
    target_critic_inputs = {"params": target_critic.params}
    if hasattr(target_critic, "batch_stats") and target_critic.batch_stats is not None:
        target_critic_inputs["batch_stats"] = target_critic.batch_stats
    next_qs = target_critic.apply_fn(
        target_critic_inputs,
        batch["next_observations"],
        training=False,
    )
    target_critic_subsample_size = int(target_critic_subsample_size)
    target_ensemble_size = int(next_qs.shape[0])
    target_ensemble_mode = str(target_ensemble_mode)
    if target_ensemble_mode == "min_with_replacement":
        target_key, rng = jax.random.split(rng)
        next_qs_for_target, _ = _subsample_critic_values_with_replacement(
            target_key,
            next_qs,
            None,
            target_critic_subsample_size,
        )
        next_q = next_qs_for_target.min(axis=0)
    elif target_ensemble_mode == "configured_without_replacement":
        if 0 < target_critic_subsample_size < target_ensemble_size:
            target_key, rng = jax.random.split(rng)
            target_indices = jax.random.permutation(target_key, target_ensemble_size)[
                :target_critic_subsample_size
            ]
            next_qs_for_target = jnp.take(next_qs, target_indices, axis=0)
        else:
            next_qs_for_target = next_qs
        next_q = _reduce_q(next_qs_for_target, critic_reduction=critic_reduction)
    else:
        raise ValueError(f"Invalid target ensemble mode: {target_ensemble_mode}")

    target_policy_mode = str(target_policy_mode)
    _validate_legal_min_action_id(legal_min_action_id, next_q.shape[-1])
    if target_policy_mode == "actor":
        actor_inputs = {"params": actor.params}
        if hasattr(actor, "batch_stats") and actor.batch_stats is not None:
            actor_inputs["batch_stats"] = actor.batch_stats
        logits = actor.apply_fn(
            actor_inputs,
            batch["next_observations"],
            training=False,
            min_action_id=legal_min_action_id,
            return_logits=True,
        )
        next_log_probs = jax.nn.log_softmax(logits, axis=-1)
        next_probs = jnp.exp(next_log_probs)
        alpha = temp.apply_fn({"params": temp.params})
        next_v = jnp.sum(next_probs * (next_q - alpha * next_log_probs), axis=-1)
        target_entropy = -(next_probs * next_log_probs).sum(axis=-1).mean()
    else:
        next_v = _critic_value_from_q(
            next_q,
            mode=target_policy_mode,
            min_action_id=legal_min_action_id,
            critic_backup_temp=critic_backup_temp,
        )
        target_entropy = jnp.asarray(0.0, dtype=next_v.dtype)

    target_q_unclipped = batch["rewards"] + batch["discount"] * batch["masks"] * next_v
    clip_min = jnp.asarray(target_q_clip_min, dtype=target_q_unclipped.dtype)
    clip_max = jnp.asarray(target_q_clip_max, dtype=target_q_unclipped.dtype)
    target_q = jnp.clip(target_q_unclipped, clip_min, clip_max)
    target_q_clip_fraction = jnp.mean(
        (jnp.abs(target_q - target_q_unclipped) > 1e-6).astype(target_q.dtype)
    )
    weights = _transition_weights(batch["durations"], transition_weight_mode)
    actions = jnp.asarray(batch["actions"], dtype=jnp.int32).reshape(-1)

    def critic_loss_fn(critic_params: Params):
        critic_inputs = {"params": critic_params}
        if hasattr(critic, "batch_stats") and critic.batch_stats is not None:
            critic_inputs["batch_stats"] = critic.batch_stats
            qs, new_model_state = critic.apply_fn(
                critic_inputs,
                batch["observations"],
                training=True,
                mutable=["batch_stats"],
            )
        else:
            qs = critic.apply_fn(
                critic_inputs,
                batch["observations"],
                training=True,
            )
            new_model_state = {}

        chosen_qs = jnp.take_along_axis(qs, actions[None, :, None], axis=-1).squeeze(-1)
        td_error = chosen_qs - target_q[None, :]
        member_weights = jnp.broadcast_to(weights[None, :], chosen_qs.shape)
        mask_key, force_key, cql_key = jax.random.split(rng, 3)
        bootstrap_mask = jax.random.bernoulli(
            mask_key,
            p=jnp.clip(jnp.asarray(critic_bootstrap_mask_prob), 0.0, 1.0),
            shape=chosen_qs.shape,
        ).astype(chosen_qs.dtype)
        forced_indices = jax.random.randint(force_key, (chosen_qs.shape[0],), 0, chosen_qs.shape[1])
        forced_mask = jax.nn.one_hot(forced_indices, chosen_qs.shape[1], dtype=chosen_qs.dtype)
        empty_members = bootstrap_mask.sum(axis=1, keepdims=True) <= 0.0
        bootstrap_mask = jnp.where(empty_members, forced_mask, bootstrap_mask)
        member_weights = jnp.where(
            jnp.asarray(critic_bootstrap_mask_prob) < 1.0,
            member_weights * bootstrap_mask,
            member_weights,
        )
        td_element_loss = _td_element_loss(td_error, td_loss_type, td_huber_delta)
        td_loss = (member_weights * td_element_loss).sum() / jnp.clip(member_weights.sum(), min=1.0)
        cql_info = {
            "cql_loss": jnp.asarray(0.0, dtype=td_loss.dtype),
            "cql_loss_unnormalized": jnp.asarray(0.0, dtype=td_loss.dtype),
            "cql_ood_q_mean": jnp.asarray(0.0, dtype=td_loss.dtype),
            "cql_ood_q_mean_unnormalized": jnp.asarray(0.0, dtype=td_loss.dtype),
            "cql_data_q_mean": jnp.asarray(0.0, dtype=td_loss.dtype),
            "cql_ensemble_size": jnp.asarray(0.0, dtype=td_loss.dtype),
            "mc_bound_rate": jnp.asarray(0.0, dtype=td_loss.dtype),
            "mc_bound_actions_per_valid": jnp.asarray(0.0, dtype=td_loss.dtype),
        }
        cql_loss = jnp.asarray(0.0, dtype=td_loss.dtype)
        if float(cql_alpha) > 0.0:
            cql_loss, cql_info = _cql_loss(
                cql_key,
                qs,
                actions,
                batch,
                cql_temp=cql_temp,
                cql_ensemble_mode=cql_ensemble_mode,
                target_critic_subsample_size=target_critic_subsample_size,
                use_mc_lower_bound=use_mc_lower_bound,
                mc_lower_bound_scope=mc_lower_bound_scope,
                mc_lower_bound_topk=mc_lower_bound_topk,
            )
        critic_loss = td_loss + jnp.asarray(cql_alpha, dtype=td_loss.dtype) * cql_loss
        info = {
            "critic_loss": critic_loss,
            "td_loss": td_loss,
            "q": chosen_qs.mean(),
            "target_q": target_q.mean(),
            "target_q_unclipped": target_q_unclipped.mean(),
            "target_q_clip_fraction": target_q_clip_fraction,
            "target_v_next": next_v.mean(),
            "target_actor_entropy": target_entropy,
            "duration_mean": jnp.asarray(batch["durations"], dtype=jnp.float32).mean(),
            "discount_mean": jnp.asarray(batch["discount"], dtype=jnp.float32).mean(),
            "reward_mean": jnp.asarray(batch["rewards"], dtype=jnp.float32).mean(),
            "mask_mean": jnp.asarray(batch["masks"], dtype=jnp.float32).mean(),
            "transition_weight_mean": weights.mean(),
            "critic_bootstrap_mask_prob": jnp.asarray(
                critic_bootstrap_mask_prob, dtype=jnp.float32
            ),
            "critic_bootstrap_mask_mean": bootstrap_mask.mean(),
            "target_critic_ensemble_size": jnp.asarray(
                next_qs_for_target.shape[0], dtype=jnp.float32
            ),
            "target_critic_reduction_id": jnp.asarray(
                0.0
                if target_ensemble_mode == "min_with_replacement"
                else {"min": 0.0, "mean": 1.0}[critic_reduction],
                dtype=jnp.float32,
            ),
            "target_q_clip_min": clip_min.astype(jnp.float32),
            "target_q_clip_max": clip_max.astype(jnp.float32),
            "td_loss_type_id": jnp.asarray({"mse": 0, "huber": 1}[td_loss_type], dtype=jnp.float32),
            "td_huber_delta": jnp.asarray(td_huber_delta, dtype=jnp.float32),
            "action_id_mean": actions.astype(jnp.float32).mean(),
            "target_policy_mode_id": jnp.asarray(
                {"actor": 0, "q_soft": 1, "q_greedy": 2}[target_policy_mode], dtype=jnp.float32
            ),
        }
        info.update(cql_info)
        return critic_loss, (info, new_model_state)

    grads, (info, new_model_state) = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    if "batch_stats" in new_model_state:
        new_critic = critic.apply_gradients(grads=grads, batch_stats=new_model_state["batch_stats"])
    else:
        new_critic = critic.apply_gradients(grads=grads)
    return new_critic, info
