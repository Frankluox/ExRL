"""Image-based execution-length Q-learning and discrete SAC."""

import copy
import functools
import pathlib
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from flax.training import checkpoints
from flax.training import train_state
from flax.core.frozen_dict import FrozenDict
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn

from exrl.pixel.agents.agent import Agent
from exrl.pixel.agents.discrete.actor_updater import evaluate_actor, update_actor
from exrl.pixel.agents.discrete.critic_updater import update_critic
from exrl.pixel.agents.discrete.temperature import Temperature
from exrl.pixel.agents.discrete.temperature_updater import update_temperature
from exrl.pixel.data.augmentations import batched_random_crop, color_transform
from exrl.pixel.data.dataset import DatasetDict
from exrl.pixel.networks.categorical_policy import CategoricalPolicy
from exrl.pixel.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from exrl.pixel.networks.encoders.networks import Encoder, PixelMultiplexer
from exrl.pixel.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from exrl.pixel.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from exrl.pixel.networks.values.discrete_action_value import DiscreteActionValueEnsemble
from exrl.pixel.types import Params, PRNGKey
from exrl.pixel.utils.target_update import soft_target_update


class TrainState(train_state.TrainState):
    batch_stats: Any


def _resolve_activation_fn(activation_fn: Union[str, Any]):
    if callable(activation_fn):
        return activation_fn
    name = str(activation_fn).lower()
    if name == "relu":
        return nn.relu
    if name == "swish":
        return nn.swish
    if name == "tanh":
        return nn.tanh
    raise ValueError(f"Unsupported activation_fn: {activation_fn}")


def _maybe_color_transform(rng: PRNGKey, pixels: jnp.ndarray, num_cameras: int) -> jnp.ndarray:
    if num_cameras > 1:
        out = pixels
        for camera_idx in range(num_cameras):
            out = out.at[:, :, :, camera_idx * 3 : (camera_idx + 1) * 3].set(
                (
                    color_transform(
                        rng,
                        pixels[:, :, :, camera_idx * 3 : (camera_idx + 1) * 3].astype(jnp.float32)
                        / 255.0,
                    )
                    * 255
                ).astype(jnp.uint8)
            )
        return out
    return (color_transform(rng, pixels.astype(jnp.float32) / 255.0) * 255).astype(jnp.uint8)


@functools.partial(
    jax.jit,
    static_argnames=(
        "critic_reduction",
        "random_crop",
        "color_jitter",
        "aug_next",
        "num_cameras",
        "update_actor_flag",
        "update_alpha_flag",
        "legal_min_action_id",
        "actor_prior_min_action_id",
        "transition_weight_mode",
        "target_policy_mode",
        "cql_alpha",
        "target_ensemble_mode",
        "cql_ensemble_mode",
        "td_loss_type",
        "use_mc_lower_bound",
        "mc_lower_bound_scope",
        "mc_lower_bound_topk",
        "target_critic_subsample_size",
    ),
)
def _update_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    temp: TrainState,
    batch: FrozenDict,
    tau: float,
    target_entropy: float,
    critic_reduction: str,
    random_crop: bool,
    color_jitter: bool,
    aug_next: bool,
    num_cameras: int,
    update_actor_flag: bool,
    update_alpha_flag: bool,
    legal_min_action_id: int,
    actor_prior_coeff: float,
    actor_prior_min_action_id: int,
    action_id_offset: int,
    transition_weight_mode: str,
    max_temperature: float,
    target_policy_mode: str,
    critic_backup_temp: float,
    cql_alpha: float,
    cql_temp: float,
    target_q_clip_min: float,
    target_q_clip_max: float,
    target_ensemble_mode: str,
    cql_ensemble_mode: str,
    td_loss_type: str,
    td_huber_delta: float,
    use_mc_lower_bound: bool,
    mc_lower_bound_scope: str,
    mc_lower_bound_topk: int,
    critic_bootstrap_mask_prob: float,
    target_critic_subsample_size: int,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:
    aug_pixels = batch["observations"]["pixels"]
    aug_next_pixels = batch["next_observations"]["pixels"]

    if random_crop and batch["observations"]["pixels"].squeeze().ndim != 2:
        rng, key = jax.random.split(rng)
        aug_pixels = batched_random_crop(key, batch["observations"]["pixels"])
        if color_jitter:
            rng, key = jax.random.split(rng)
            aug_pixels = _maybe_color_transform(key, aug_pixels, num_cameras)

    observations = batch["observations"].copy(add_or_replace={"pixels": aug_pixels})
    batch = batch.copy(add_or_replace={"observations": observations})

    if aug_next:
        if random_crop:
            rng, key = jax.random.split(rng)
            aug_next_pixels = batched_random_crop(key, batch["next_observations"]["pixels"])
        if color_jitter:
            rng, key = jax.random.split(rng)
            aug_next_pixels = _maybe_color_transform(key, aug_next_pixels, num_cameras)
        next_observations = batch["next_observations"].copy(
            add_or_replace={"pixels": aug_next_pixels}
        )
        batch = batch.copy(add_or_replace={"next_observations": next_observations})

    target_critic = critic.replace(params=target_critic_params)
    rng, critic_key = jax.random.split(rng)
    new_critic, critic_info = update_critic(
        critic_key,
        actor,
        critic,
        target_critic,
        temp,
        batch,
        critic_reduction=critic_reduction,
        legal_min_action_id=legal_min_action_id,
        transition_weight_mode=transition_weight_mode,
        target_policy_mode=target_policy_mode,
        critic_backup_temp=critic_backup_temp,
        cql_alpha=cql_alpha,
        cql_temp=cql_temp,
        target_q_clip_min=target_q_clip_min,
        target_q_clip_max=target_q_clip_max,
        target_ensemble_mode=target_ensemble_mode,
        cql_ensemble_mode=cql_ensemble_mode,
        td_loss_type=td_loss_type,
        td_huber_delta=td_huber_delta,
        use_mc_lower_bound=use_mc_lower_bound,
        mc_lower_bound_scope=mc_lower_bound_scope,
        mc_lower_bound_topk=mc_lower_bound_topk,
        critic_bootstrap_mask_prob=critic_bootstrap_mask_prob,
        target_critic_subsample_size=target_critic_subsample_size,
    )
    new_target_critic_params = soft_target_update(new_critic.params, target_critic_params, tau)

    if update_actor_flag:
        new_actor, actor_info = update_actor(
            actor,
            new_critic,
            temp,
            batch,
            critic_reduction=critic_reduction,
            transition_weight_mode=transition_weight_mode,
            legal_min_action_id=legal_min_action_id,
            actor_prior_coeff=actor_prior_coeff,
            actor_prior_min_action_id=actor_prior_min_action_id,
            action_id_offset=action_id_offset,
        )
    else:
        new_actor = actor
        actor_info = evaluate_actor(
            actor,
            new_critic,
            temp,
            batch,
            critic_reduction=critic_reduction,
            transition_weight_mode=transition_weight_mode,
            legal_min_action_id=legal_min_action_id,
            actor_prior_coeff=actor_prior_coeff,
            actor_prior_min_action_id=actor_prior_min_action_id,
            action_id_offset=action_id_offset,
        )

    if update_alpha_flag:
        new_temp, alpha_info = update_temperature(
            temp,
            actor_info["entropy"],
            target_entropy,
            max_temperature=max_temperature,
        )
    else:
        new_temp = temp
        temperature = temp.apply_fn({"params": temp.params})
        alpha_info = {
            "temperature": temperature,
            "temperature_loss": jnp.asarray(0.0, dtype=temperature.dtype),
        }

    info = {}
    info.update(critic_info)
    info.update(actor_info)
    info.update(alpha_info)
    return rng, new_actor, new_critic, new_target_critic_params, new_temp, info


@functools.partial(jax.jit, static_argnames=("actor_apply_fn", "min_action_id"))
def _sample_action_ids_jit(
    rng: PRNGKey,
    actor_apply_fn,
    actor_params: Params,
    actor_batch_stats: Any,
    observations,
    min_action_id: int,
) -> Tuple[PRNGKey, jnp.ndarray]:
    input_collections = {"params": actor_params}
    if actor_batch_stats is not None:
        input_collections["batch_stats"] = actor_batch_stats
    dist = actor_apply_fn(
        input_collections,
        observations,
        training=False,
        min_action_id=min_action_id,
        mutable=False,
    )
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


@functools.partial(jax.jit, static_argnames=("actor_apply_fn", "min_action_id"))
def _eval_action_ids_jit(
    actor_apply_fn,
    actor_params: Params,
    actor_batch_stats: Any,
    observations,
    min_action_id: int,
) -> jnp.ndarray:
    input_collections = {"params": actor_params}
    if actor_batch_stats is not None:
        input_collections["batch_stats"] = actor_batch_stats
    dist = actor_apply_fn(
        input_collections,
        observations,
        training=False,
        min_action_id=min_action_id,
        mutable=False,
    )
    return dist.mode()


@functools.partial(jax.jit, static_argnames=("actor_apply_fn", "min_action_id"))
def _policy_logits_jit(
    actor_apply_fn,
    actor_params: Params,
    actor_batch_stats: Any,
    observations,
    min_action_id: int,
) -> jnp.ndarray:
    input_collections = {"params": actor_params}
    if actor_batch_stats is not None:
        input_collections["batch_stats"] = actor_batch_stats
    return actor_apply_fn(
        input_collections,
        observations,
        training=False,
        min_action_id=min_action_id,
        return_logits=True,
        mutable=False,
    )


def _reduce_q_values(qs: jnp.ndarray, critic_reduction: str) -> jnp.ndarray:
    if critic_reduction == "min":
        return qs.min(axis=0)
    if critic_reduction == "mean":
        return qs.mean(axis=0)
    raise ValueError(f"Unsupported critic_reduction: {critic_reduction}")


def _mask_q_values(q: jnp.ndarray, min_action_id: int) -> jnp.ndarray:
    if int(min_action_id) <= 0:
        return q
    action_ids = jnp.arange(q.shape[-1], dtype=jnp.int32)
    return jnp.where(action_ids < int(min_action_id), jnp.full_like(q, -1e9), q)


@functools.partial(
    jax.jit,
    static_argnames=("critic_apply_fn", "critic_reduction", "mode", "min_action_id"),
)
def _critic_action_ids_jit(
    rng: PRNGKey,
    critic_apply_fn,
    critic_params: Params,
    critic_batch_stats: Any,
    observations,
    critic_reduction: str,
    mode: str,
    min_action_id: int,
    critic_backup_temp: float,
) -> Tuple[PRNGKey, jnp.ndarray]:
    input_collections = {"params": critic_params}
    if critic_batch_stats is not None:
        input_collections["batch_stats"] = critic_batch_stats
    qs = critic_apply_fn(input_collections, observations, training=False, mutable=False)
    q = _mask_q_values(_reduce_q_values(qs, critic_reduction), min_action_id)
    if mode == "q_greedy":
        return rng, jnp.argmax(q, axis=-1)
    if mode == "q_soft":
        probs = jax.nn.softmax(q / jnp.asarray(critic_backup_temp, dtype=q.dtype), axis=-1)
        rng, key = jax.random.split(rng)
        actions = jax.random.categorical(key, jnp.log(jnp.clip(probs, min=1e-12)), axis=-1)
        return rng, actions
    raise ValueError(f"Unsupported critic action mode: {mode}")


@functools.partial(
    jax.jit,
    static_argnames=("critic_apply_fn", "critic_reduction", "min_action_id"),
)
def _critic_q_values_jit(
    critic_apply_fn,
    critic_params: Params,
    critic_batch_stats: Any,
    observations,
    critic_reduction: str,
    min_action_id: int,
) -> jnp.ndarray:
    input_collections = {"params": critic_params}
    if critic_batch_stats is not None:
        input_collections["batch_stats"] = critic_batch_stats
    qs = critic_apply_fn(input_collections, observations, training=False, mutable=False)
    return _mask_q_values(_reduce_q_values(qs, critic_reduction), min_action_id)


class PixelExRL(Agent):
    def __init__(
        self,
        seed: int,
        observations: Union[jnp.ndarray, DatasetDict],
        actions: jnp.ndarray,
        num_discrete_actions: int,
        action_id_offset: int = 0,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        decay_steps: Optional[int] = None,
        hidden_dims: Sequence[int] = (256, 256),
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        discount: float = 0.99,
        tau: float = 0.005,
        critic_reduction: str = "mean",
        dropout_rate: Optional[float] = None,
        encoder_type: str = "resnet_34_v1",
        encoder_norm: str = "group",
        random_crop: bool = True,
        color_jitter: bool = True,
        use_spatial_softmax: bool = True,
        softmax_temperature: float = 1,
        aug_next: bool = True,
        use_bottleneck: bool = True,
        init_temperature: float = 1.0,
        num_qs: int = 2,
        target_entropy: float = None,
        target_entropy_scale: float = 0.98,
        num_cameras: int = 1,
        activation_fn: Union[str, Any] = "relu",
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = True,
        actor_head_init_method: str = "default",
        actor_head_init_scale: float = 1e-2,
        max_alpha: float = -1.0,
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
    ):
        self.aug_next = aug_next
        self.random_crop = random_crop
        self.color_jitter = color_jitter
        self.num_cameras = num_cameras
        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction
        self.num_discrete_actions = int(num_discrete_actions)
        self.action_id_offset = int(action_id_offset)
        self.action_chunk_shape = (
            tuple(actions.shape[-2:]) if actions.ndim >= 3 else tuple(actions.shape[-1:])
        )
        self.target_entropy_scale = float(target_entropy_scale)
        self.max_alpha = float(max_alpha)
        self.target_policy_mode = str(target_policy_mode)
        if self.target_policy_mode not in ("actor", "q_soft", "q_greedy"):
            raise ValueError(f"Unsupported target_policy_mode: {target_policy_mode}")
        self.critic_backup_temp = float(critic_backup_temp)
        self.cql_alpha = float(cql_alpha)
        self.cql_temp = float(cql_temp)
        self.target_q_clip_min = float(target_q_clip_min)
        self.target_q_clip_max = float(target_q_clip_max)
        self.target_ensemble_mode = str(target_ensemble_mode)
        self.cql_ensemble_mode = str(cql_ensemble_mode)
        self.td_loss_type = str(td_loss_type).lower()
        self.td_huber_delta = float(td_huber_delta)
        self.use_mc_lower_bound = bool(use_mc_lower_bound)
        self.mc_lower_bound_scope = str(mc_lower_bound_scope)
        self.mc_lower_bound_topk = int(mc_lower_bound_topk)
        self.critic_bootstrap_mask_prob = float(critic_bootstrap_mask_prob)
        self.target_critic_subsample_size = int(target_critic_subsample_size)
        if self.critic_backup_temp <= 0.0:
            raise ValueError(f"critic_backup_temp must be > 0, got {critic_backup_temp}")
        if self.cql_alpha < 0.0:
            raise ValueError(f"cql_alpha must be >= 0, got {cql_alpha}")
        if self.cql_temp <= 0.0:
            raise ValueError(f"cql_temp must be > 0, got {cql_temp}")
        if self.target_q_clip_min > self.target_q_clip_max:
            raise ValueError(
                f"target_q_clip_min ({self.target_q_clip_min}) cannot exceed "
                f"target_q_clip_max ({self.target_q_clip_max})"
            )
        if self.target_ensemble_mode not in (
            "configured_without_replacement",
            "min_with_replacement",
        ):
            raise ValueError(f"Unsupported target_ensemble_mode: {target_ensemble_mode}")
        if self.cql_ensemble_mode not in ("all", "target_subsample_with_replacement"):
            raise ValueError(f"Unsupported cql_ensemble_mode: {cql_ensemble_mode}")
        if self.td_loss_type not in ("mse", "huber"):
            raise ValueError(f"td_loss_type must be mse or huber, got {td_loss_type}")
        if self.td_huber_delta <= 0.0:
            raise ValueError(f"td_huber_delta must be > 0, got {td_huber_delta}")
        if self.mc_lower_bound_scope not in ("all", "non_data", "data", "topk_non_data"):
            raise ValueError(
                "mc_lower_bound_scope must be one of all/non_data/data/topk_non_data, "
                f"got {mc_lower_bound_scope}"
            )
        if self.mc_lower_bound_topk <= 0:
            raise ValueError(f"mc_lower_bound_topk must be positive, got {mc_lower_bound_topk}")
        if not (0.0 < self.critic_bootstrap_mask_prob <= 1.0):
            raise ValueError(
                f"critic_bootstrap_mask_prob must be in (0, 1], got {critic_bootstrap_mask_prob}"
            )
        if self.target_critic_subsample_size < 0:
            raise ValueError(
                f"target_critic_subsample_size must be >= 0, got {target_critic_subsample_size}"
            )

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        if encoder_type == "small":
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == "impala":
            encoder_def = ImpalaEncoder()
        elif encoder_type == "impala_small":
            encoder_def = SmallerImpalaEncoder()
        elif encoder_type == "resnet_small":
            encoder_def = ResNetSmall(
                norm=encoder_norm,
                use_spatial_softmax=use_spatial_softmax,
                softmax_temperature=softmax_temperature,
            )
        elif encoder_type == "resnet_18_v1":
            encoder_def = ResNet18(
                norm=encoder_norm,
                use_spatial_softmax=use_spatial_softmax,
                softmax_temperature=softmax_temperature,
            )
        elif encoder_type == "resnet_34_v1":
            encoder_def = ResNet34(
                norm=encoder_norm,
                use_spatial_softmax=use_spatial_softmax,
                softmax_temperature=softmax_temperature,
            )
        elif encoder_type == "resnet_small_v2":
            encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif encoder_type == "resnet_18_v2":
            encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif encoder_type == "resnet_34_v2":
            encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        else:
            raise ValueError(f"encoder type not found: {encoder_type}")

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0], hidden_dims[0])
        activation = _resolve_activation_fn(activation_fn)

        policy_def = CategoricalPolicy(
            hidden_dims,
            self.num_discrete_actions,
            activations=activation,
            dropout_rate=dropout_rate,
            use_layer_norm=bool(actor_use_layer_norm),
            init_method=str(actor_head_init_method),
            head_init_scale=float(actor_head_init_scale),
        )
        actor_def = PixelMultiplexer(
            encoder=encoder_def,
            network=policy_def,
            latent_dim=latent_dim,
            use_bottleneck=use_bottleneck,
        )
        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init["params"]
        actor_batch_stats = (
            actor_def_init["batch_stats"] if "batch_stats" in actor_def_init else None
        )
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
            batch_stats=actor_batch_stats,
        )

        critic_def = DiscreteActionValueEnsemble(
            hidden_dims,
            self.num_discrete_actions,
            activations=activation,
            use_layer_norm=bool(critic_use_layer_norm),
            num_qs=num_qs,
        )
        critic_def = PixelMultiplexer(
            encoder=encoder_def,
            network=critic_def,
            latent_dim=latent_dim,
            use_bottleneck=use_bottleneck,
        )
        critic_def_init = critic_def.init(critic_key, observations)
        critic_params = critic_def_init["params"]
        critic_batch_stats = (
            critic_def_init["batch_stats"] if "batch_stats" in critic_def_init else None
        )
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
            batch_stats=critic_batch_stats,
        )
        target_critic_params = copy.deepcopy(critic_params)

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
            batch_stats=None,
        )

        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params
        self._temp = temp

        if target_entropy is None or target_entropy == "auto":
            self.target_entropy = float(
                self.target_entropy_scale * np.log(self.num_discrete_actions)
            )
        else:
            self.target_entropy = float(target_entropy)

    def _validate_min_action_id(self, min_action_id: int, name: str = "min_action_id") -> int:
        min_action_id = int(min_action_id)
        if not (0 <= min_action_id < int(self.num_discrete_actions)):
            raise ValueError(
                f"{name} out of range: {min_action_id} for num_discrete_actions={self.num_discrete_actions}"
            )
        return min_action_id

    def update(
        self,
        batch: FrozenDict,
        update_actor: bool = True,
        update_alpha: bool = True,
        transition_weight_mode: str = "none",
        actor_prior_coeff: float = 0.0,
        actor_prior_min_action_id: int = 0,
        legal_min_action_id: int = 0,
    ) -> Dict[str, float]:
        legal_min_action_id = self._validate_min_action_id(
            legal_min_action_id, "legal_min_action_id"
        )
        actor_prior_min_action_id = self._validate_min_action_id(
            actor_prior_min_action_id, "actor_prior_min_action_id"
        )
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self._rng,
            self._actor,
            self._critic,
            self._target_critic_params,
            self._temp,
            batch,
            self.tau,
            self.target_entropy,
            self.critic_reduction,
            self.random_crop,
            self.color_jitter,
            self.aug_next,
            self.num_cameras,
            bool(update_actor),
            bool(update_alpha),
            legal_min_action_id,
            float(actor_prior_coeff),
            actor_prior_min_action_id,
            int(self.action_id_offset),
            transition_weight_mode,
            float(self.max_alpha),
            str(self.target_policy_mode),
            float(self.critic_backup_temp),
            float(self.cql_alpha),
            float(self.cql_temp),
            float(self.target_q_clip_min),
            float(self.target_q_clip_max),
            str(self.target_ensemble_mode),
            str(self.cql_ensemble_mode),
            str(self.td_loss_type),
            float(self.td_huber_delta),
            bool(self.use_mc_lower_bound),
            str(self.mc_lower_bound_scope),
            int(self.mc_lower_bound_topk),
            float(self.critic_bootstrap_mask_prob),
            int(self.target_critic_subsample_size),
        )
        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        self._temp = new_temp
        return info

    def sample_action_ids(self, observations: np.ndarray, min_action_id: int = 0) -> np.ndarray:
        min_action_id = self._validate_min_action_id(min_action_id)
        rng, actions = _sample_action_ids_jit(
            self._rng,
            self._actor.apply_fn,
            self._actor.params,
            getattr(self._actor, "batch_stats", None),
            observations,
            min_action_id,
        )
        self._rng = rng
        return np.asarray(actions)

    def eval_action_ids(self, observations: np.ndarray, min_action_id: int = 0) -> np.ndarray:
        min_action_id = self._validate_min_action_id(min_action_id)
        actions = _eval_action_ids_jit(
            self._actor.apply_fn,
            self._actor.params,
            getattr(self._actor, "batch_stats", None),
            observations,
            min_action_id,
        )
        return np.asarray(actions)

    def get_policy_probs(self, observations: np.ndarray, min_action_id: int = 0) -> np.ndarray:
        min_action_id = self._validate_min_action_id(min_action_id)
        logits = _policy_logits_jit(
            self._actor.apply_fn,
            self._actor.params,
            getattr(self._actor, "batch_stats", None),
            observations,
            min_action_id,
        )
        return np.asarray(jax.nn.softmax(logits, axis=-1))

    def eval_q_action_ids(
        self,
        observations: np.ndarray,
        mode: str = "q_greedy",
        critic_reduction: Optional[str] = None,
        min_action_id: int = 0,
    ) -> np.ndarray:
        min_action_id = self._validate_min_action_id(min_action_id)
        mode = str(mode)
        if mode not in ("q_greedy", "q_soft"):
            raise ValueError(f"Unsupported critic action mode: {mode}")
        reduction = str(self.critic_reduction if critic_reduction is None else critic_reduction)
        rng, actions = _critic_action_ids_jit(
            self._rng,
            self._critic.apply_fn,
            self._critic.params,
            getattr(self._critic, "batch_stats", None),
            observations,
            reduction,
            mode,
            min_action_id,
            float(self.critic_backup_temp),
        )
        self._rng = rng
        return np.asarray(actions)

    def get_q_values(
        self,
        observations: np.ndarray,
        critic_reduction: Optional[str] = None,
        min_action_id: int = 0,
    ) -> np.ndarray:
        min_action_id = self._validate_min_action_id(min_action_id)
        reduction = str(self.critic_reduction if critic_reduction is None else critic_reduction)
        q = _critic_q_values_jit(
            self._critic.apply_fn,
            self._critic.params,
            getattr(self._critic, "batch_stats", None),
            observations,
            reduction,
            min_action_id,
        )
        return np.asarray(q)

    @property
    def _save_dict(self):
        return {
            "critic": self._critic,
            "target_critic_params": self._target_critic_params,
            "actor": self._actor,
            "temp": self._temp,
            "rng": self._rng,
            "num_discrete_actions": self.num_discrete_actions,
            "action_id_offset": self.action_id_offset,
            "target_entropy": self.target_entropy,
            "target_entropy_scale": self.target_entropy_scale,
            "max_alpha": self.max_alpha,
            "target_policy_mode": self.target_policy_mode,
            "critic_backup_temp": self.critic_backup_temp,
            "cql_alpha": self.cql_alpha,
            "cql_temp": self.cql_temp,
            "target_q_clip_min": self.target_q_clip_min,
            "target_q_clip_max": self.target_q_clip_max,
            "target_ensemble_mode": self.target_ensemble_mode,
            "cql_ensemble_mode": self.cql_ensemble_mode,
            "td_loss_type": self.td_loss_type,
            "td_huber_delta": self.td_huber_delta,
            "use_mc_lower_bound": self.use_mc_lower_bound,
            "mc_lower_bound_scope": self.mc_lower_bound_scope,
            "mc_lower_bound_topk": self.mc_lower_bound_topk,
            "critic_bootstrap_mask_prob": self.critic_bootstrap_mask_prob,
            "target_critic_subsample_size": self.target_critic_subsample_size,
        }

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} does not exist."
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict, prefix="checkpoint")
        self._actor = output_dict["actor"]
        self._critic = output_dict["critic"]
        self._target_critic_params = output_dict["target_critic_params"]
        self._temp = output_dict["temp"]
        self._rng = output_dict.get("rng", self._rng)
        self.num_discrete_actions = int(
            output_dict.get("num_discrete_actions", self.num_discrete_actions)
        )
        self.action_id_offset = int(output_dict.get("action_id_offset", self.action_id_offset))
        self.target_entropy = float(output_dict.get("target_entropy", self.target_entropy))
        self.target_entropy_scale = float(
            output_dict.get("target_entropy_scale", self.target_entropy_scale)
        )
        self.max_alpha = float(output_dict.get("max_alpha", self.max_alpha))
        self.target_policy_mode = str(
            output_dict.get("target_policy_mode", self.target_policy_mode)
        )
        self.critic_backup_temp = float(
            output_dict.get("critic_backup_temp", self.critic_backup_temp)
        )
        self.cql_alpha = float(output_dict.get("cql_alpha", self.cql_alpha))
        self.cql_temp = float(output_dict.get("cql_temp", self.cql_temp))
        self.target_q_clip_min = float(output_dict.get("target_q_clip_min", self.target_q_clip_min))
        self.target_q_clip_max = float(output_dict.get("target_q_clip_max", self.target_q_clip_max))
        self.target_ensemble_mode = str(
            output_dict.get("target_ensemble_mode", self.target_ensemble_mode)
        )
        self.cql_ensemble_mode = str(output_dict.get("cql_ensemble_mode", self.cql_ensemble_mode))
        self.td_loss_type = str(output_dict.get("td_loss_type", self.td_loss_type)).lower()
        self.td_huber_delta = float(output_dict.get("td_huber_delta", self.td_huber_delta))
        self.use_mc_lower_bound = bool(
            output_dict.get("use_mc_lower_bound", self.use_mc_lower_bound)
        )
        self.mc_lower_bound_scope = str(
            output_dict.get("mc_lower_bound_scope", self.mc_lower_bound_scope)
        )
        self.mc_lower_bound_topk = int(
            output_dict.get("mc_lower_bound_topk", self.mc_lower_bound_topk)
        )
        self.critic_bootstrap_mask_prob = float(
            output_dict.get("critic_bootstrap_mask_prob", self.critic_bootstrap_mask_prob)
        )
        self.target_critic_subsample_size = int(
            output_dict.get("target_critic_subsample_size", self.target_critic_subsample_size)
        )
        print("restored from ", dir)
