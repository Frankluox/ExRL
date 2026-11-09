from copy import deepcopy
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_schedule_fn, polyak_update, update_learning_rate
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.sac.sac import SAC

SelfResidualSAC = TypeVar("SelfResidualSAC", bound="ResidualSAC")


class ResidualSAC(SAC):
    """
    Residual SAC on top of a frozen diffusion policy.

    The frozen base policy samples an action chunk in environment action space.
    SAC learns a stochastic residual policy in normalized action space and we
    execute the clipped sum:

        a_final = clip(a_base + residual_scale * a_residual, low, high)

    The replay buffer stores the executed final action (normalized to [-1, 1])
    together with the rollout-time base action chunk used for that transition.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        actor_gradient_steps: int = -1,
        action_norm_regularization: float = 0.0,
        base_policy=None,
        diffusion_act_dim=None,
        base_policy_act_dim=None,
        residual_scale: float = 0.1,
        progressive_explore_steps: int = 0,
        critic_learning_rate: Optional[Union[float, Schedule]] = None,
        ent_coef_learning_rate: Optional[Union[float, Schedule]] = None,
        max_grad_norm: Optional[float] = None,
        actor_observation_mode: str = "obs",
        action_feature_mean: Optional[Union[np.ndarray, th.Tensor]] = None,
        action_feature_std: Optional[Union[np.ndarray, th.Tensor]] = None,
        fixed_ent_coef_final: Optional[float] = None,
        fixed_ent_coef_fraction: float = 1.0,
        fixed_ent_coef_steps: Optional[int] = None,
        pre_learning_residual_mode: str = "random_residual",
        residual_temporal_group_size: int = 1,
        residual_active_steps: Optional[int] = None,
    ):
        self.base_policy = base_policy
        loading_without_saved_dims = diffusion_act_dim is None and not _init_setup_model
        if diffusion_act_dim is None:
            if _init_setup_model:
                raise ValueError("ResidualSAC requires diffusion_act_dim=(action_chunk, action_dim).")
            flat_action_dim = int(np.prod(env.action_space.shape))
            diffusion_act_dim = (1, flat_action_dim)
        self.diffusion_act_chunk = int(diffusion_act_dim[0])
        self.diffusion_act_dim = int(diffusion_act_dim[1])
        if base_policy_act_dim is None:
            base_policy_act_dim = diffusion_act_dim
        self.base_policy_act_chunk = int(base_policy_act_dim[0])
        self.base_policy_act_dim = int(base_policy_act_dim[1])
        if self.base_policy_act_dim != self.diffusion_act_dim:
            raise ValueError(
                "ResidualSAC requires base_policy_act_dim and diffusion_act_dim to share action_dim."
            )
        self.residual_scale = float(residual_scale)
        self.residual_temporal_group_size = max(int(residual_temporal_group_size), 1)
        if loading_without_saved_dims and residual_active_steps is None:
            self.residual_active_steps = None
        elif residual_active_steps is None:
            self.residual_active_steps = self.diffusion_act_chunk
        else:
            self.residual_active_steps = int(residual_active_steps)
        if self.residual_active_steps is not None and (
            self.residual_active_steps < 0 or self.residual_active_steps > self.diffusion_act_chunk
        ):
            raise ValueError(
                "ResidualSAC residual_active_steps must be in "
                f"[0, {self.diffusion_act_chunk}], got {self.residual_active_steps}."
            )
        self.progressive_explore_steps = int(progressive_explore_steps)
        if actor_observation_mode not in {"obs", "obs_base_action"}:
            raise ValueError("ResidualSAC actor_observation_mode must be 'obs' or 'obs_base_action'.")
        self.actor_observation_mode = actor_observation_mode
        self.action_feature_mean_np = None if action_feature_mean is None else np.asarray(action_feature_mean, dtype=np.float32).reshape(-1)
        self.action_feature_std_np = None if action_feature_std is None else np.asarray(action_feature_std, dtype=np.float32).reshape(-1)
        self.critic_learning_rate = learning_rate if critic_learning_rate is None else critic_learning_rate
        if ent_coef_learning_rate is None:
            ent_coef_learning_rate = self.critic_learning_rate
        self.ent_coef_learning_rate = ent_coef_learning_rate
        self.max_grad_norm = None if max_grad_norm is None or float(max_grad_norm) <= 0 else float(max_grad_norm)
        self.fixed_ent_coef_initial = None if isinstance(ent_coef, str) else float(ent_coef)
        self.fixed_ent_coef_final = None if fixed_ent_coef_final is None else float(fixed_ent_coef_final)
        self.fixed_ent_coef_fraction = max(float(fixed_ent_coef_fraction), 1e-6)
        self.fixed_ent_coef_steps = None if fixed_ent_coef_steps is None else max(int(fixed_ent_coef_steps), 1)
        if pre_learning_residual_mode not in {"random_residual", "base_only", "actor"}:
            raise ValueError(
                "ResidualSAC pre_learning_residual_mode must be one of "
                "{'random_residual', 'base_only', 'actor'}."
            )
        self.pre_learning_residual_mode = pre_learning_residual_mode
        self._last_base_actions: Optional[np.ndarray] = None

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            actor_gradient_steps=actor_gradient_steps,
            action_norm_regularization=action_norm_regularization,
        )

        self._action_low_np = np.asarray(self.action_space.low, dtype=np.float32)
        self._action_high_np = np.asarray(self.action_space.high, dtype=np.float32)
        expected_action_dim = self.diffusion_act_chunk * self.diffusion_act_dim
        actual_action_dim = int(np.prod(self.action_space.shape))
        if expected_action_dim != actual_action_dim:
            raise ValueError(
                "ResidualSAC diffusion_act_dim does not match environment action space: "
                f"expected {expected_action_dim}, got {actual_action_dim}."
            )
        action_center = 0.5 * (self._action_high_np + self._action_low_np)
        if not np.allclose(action_center, 0.0, atol=1e-6):
            raise ValueError(
                "ResidualSAC currently expects a zero-centered continuous action space."
            )
        self._action_half_range_np = 0.5 * (self._action_high_np - self._action_low_np)
        if self.actor_observation_mode == "obs_base_action":
            if self.action_feature_mean_np is None or self.action_feature_std_np is None:
                raise ValueError(
                    "ResidualSAC base-action conditioning requires action_feature_mean/std."
                )
            feature_dim = self.action_feature_mean_np.shape[0]
            feature_std_dim = self.action_feature_std_np.shape[0]
            if feature_dim == self.diffusion_act_dim and feature_std_dim == self.diffusion_act_dim:
                self.action_feature_mean_np = np.tile(self.action_feature_mean_np, self.diffusion_act_chunk)
                self.action_feature_std_np = np.tile(self.action_feature_std_np, self.diffusion_act_chunk)
            elif feature_dim != actual_action_dim or feature_std_dim != actual_action_dim:
                raise ValueError(
                    "ResidualSAC action_feature_mean/std must match per-step action_dim or flattened action dimension."
                )
            self.action_feature_std_np = np.where(self.action_feature_std_np < 1e-6, 1.0, self.action_feature_std_np)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        if not hasattr(self, "residual_temporal_group_size"):
            self.residual_temporal_group_size = 1
        if getattr(self, "residual_active_steps", None) is None:
            self.residual_active_steps = self.diffusion_act_chunk
        super()._setup_model()
        self.action_low = th.as_tensor(self._action_low_np, device=self.device, dtype=th.float32)
        self.action_high = th.as_tensor(self._action_high_np, device=self.device, dtype=th.float32)
        self.action_half_range = th.as_tensor(
            self._action_half_range_np, device=self.device, dtype=th.float32
        )
        self.action_feature_mean = None
        self.action_feature_std = None
        if self.action_feature_mean_np is not None and self.action_feature_std_np is not None:
            self.action_feature_mean = th.as_tensor(self.action_feature_mean_np, device=self.device, dtype=th.float32)
            self.action_feature_std = th.as_tensor(self.action_feature_std_np, device=self.device, dtype=th.float32)
        self.critic_lr_schedule = get_schedule_fn(self.critic_learning_rate)
        self.ent_coef_lr_schedule = get_schedule_fn(self.ent_coef_learning_rate)
        self._update_optimizer_learning_rates(record=False)

    def _update_optimizer_learning_rates(self, record: bool = True) -> None:
        actor_lr = self.lr_schedule(self._current_progress_remaining)
        critic_lr = self.critic_lr_schedule(self._current_progress_remaining)
        ent_coef_lr = self.ent_coef_lr_schedule(self._current_progress_remaining)
        update_learning_rate(self.actor.optimizer, actor_lr)
        update_learning_rate(self.critic.optimizer, critic_lr)
        if self.ent_coef_optimizer is not None:
            update_learning_rate(self.ent_coef_optimizer, ent_coef_lr)
        if record:
            self.logger.record("train/learning_rate", actor_lr)
            self.logger.record("train/actor_learning_rate", actor_lr)
            self.logger.record("train/critic_learning_rate", critic_lr)
            if self.ent_coef_optimizer is not None:
                self.logger.record("train/alpha_learning_rate", ent_coef_lr)

    def _get_current_fixed_ent_coef_ratio(self) -> float:
        if self.fixed_ent_coef_steps is not None:
            return min(float(self.env_steps_exact) / float(self.fixed_ent_coef_steps), 1.0)
        progress_done = 1.0 - float(self._current_progress_remaining)
        return min(max(progress_done / self.fixed_ent_coef_fraction, 0.0), 1.0)

    def _sample_base_actions(
        self,
        observations: Union[np.ndarray, th.Tensor],
        deterministic: bool = False,
    ) -> th.Tensor:
        if self.base_policy is None:
            raise ValueError("ResidualSAC requires a frozen base_policy.")
        if isinstance(observations, dict):
            raise ValueError("ResidualSAC only supports array observations.")

        obs_tensor = th.as_tensor(observations, device=self.device, dtype=th.float32)
        if obs_tensor.ndim == len(self.observation_space.shape):
            obs_tensor = obs_tensor.unsqueeze(0)

        batch_size = obs_tensor.shape[0]
        noise = th.randn(
            (batch_size, self.base_policy_act_chunk, self.base_policy_act_dim),
            device=self.device,
            dtype=th.float32,
        )

        with th.no_grad():
            base_actions = self.base_policy(obs_tensor, noise, return_numpy=False)
        base_actions = base_actions.reshape(batch_size, self.base_policy_act_chunk, self.base_policy_act_dim)
        if self.base_policy_act_chunk < self.diffusion_act_chunk:
            raise ValueError(
                "ResidualSAC requires base_policy_act_chunk >= diffusion_act_chunk."
            )
        base_actions = base_actions[:, : self.diffusion_act_chunk, :]
        return base_actions.reshape(batch_size, self.diffusion_act_chunk * self.diffusion_act_dim)

    def _combine_actions(
        self, base_env_actions: th.Tensor, residual_actions_norm: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        residual_actions_norm = self._project_residual_actions(residual_actions_norm)
        residual_env_actions = residual_actions_norm * self.action_half_range
        final_env_actions = base_env_actions + self.residual_scale * residual_env_actions
        final_env_actions = th.max(th.min(final_env_actions, self.action_high), self.action_low)
        final_norm_actions = self._normalize_env_actions(final_env_actions)
        return final_env_actions, final_norm_actions

    def _normalize_env_actions(self, env_actions: th.Tensor) -> th.Tensor:
        denom = (self.action_high - self.action_low).clamp_min(1e-6)
        return 2.0 * ((env_actions - self.action_low) / denom) - 1.0

    def _normalize_action_features(self, env_actions: th.Tensor) -> th.Tensor:
        if self.action_feature_mean is None or self.action_feature_std is None:
            raise ValueError("ResidualSAC action feature stats are required for base-action conditioning.")
        return (env_actions - self.action_feature_mean) / self.action_feature_std

    def _project_residual_actions(self, residual_actions_norm: th.Tensor) -> th.Tensor:
        if self.residual_temporal_group_size <= 1 and self.residual_active_steps >= self.diffusion_act_chunk:
            return residual_actions_norm

        flat_shape = residual_actions_norm.shape
        residual = residual_actions_norm.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim)

        if self.residual_temporal_group_size > 1:
            grouped = []
            group_size = self.residual_temporal_group_size
            for start in range(0, self.diffusion_act_chunk, group_size):
                end = min(start + group_size, self.diffusion_act_chunk)
                group_mean = residual[:, start:end, :].mean(dim=1, keepdim=True)
                grouped.append(group_mean.expand(-1, end - start, -1))
            residual = th.cat(grouped, dim=1)

        if self.residual_active_steps < self.diffusion_act_chunk:
            residual = residual.clone()
            residual[:, self.residual_active_steps :, :] = 0.0

        return residual.reshape(flat_shape)

    def _build_conditioned_observations(
        self,
        observations: Union[np.ndarray, th.Tensor],
        observation_mode: str,
        base_actions: Optional[Union[np.ndarray, th.Tensor]] = None,
    ) -> th.Tensor:
        if isinstance(observations, dict):
            raise ValueError("ResidualSAC only supports array observations.")
        obs_tensor = th.as_tensor(observations, device=self.device, dtype=th.float32)
        if obs_tensor.ndim == len(self.observation_space.shape):
            obs_tensor = obs_tensor.unsqueeze(0)
        if observation_mode == "obs":
            return obs_tensor
        if base_actions is None:
            raise ValueError("ResidualSAC *_observation_mode='obs_base_action' requires base actions.")
        base_tensor = th.as_tensor(base_actions, device=self.device, dtype=th.float32)
        if base_tensor.ndim == len(self.action_space.shape):
            base_tensor = base_tensor.unsqueeze(0)
        base_tensor = self._normalize_action_features(base_tensor)
        return th.cat([obs_tensor.reshape(obs_tensor.shape[0], -1), base_tensor.reshape(base_tensor.shape[0], -1)], dim=1)

    def _build_actor_observations(
        self,
        observations: Union[np.ndarray, th.Tensor],
        base_actions: Optional[Union[np.ndarray, th.Tensor]] = None,
    ) -> th.Tensor:
        return self._build_conditioned_observations(
            observations,
            observation_mode=self.actor_observation_mode,
            base_actions=base_actions,
        )

    def _sample_residual_actions(
        self,
        observations: Union[np.ndarray, th.Tensor],
        deterministic: bool,
        base_actions: Optional[Union[np.ndarray, th.Tensor]] = None,
    ) -> th.Tensor:
        actor_obs = self._build_actor_observations(observations, base_actions=base_actions)
        with th.no_grad():
            residual_actions = self.actor(actor_obs, deterministic=deterministic)
        return residual_actions.reshape(actor_obs.shape[0], *self.action_space.shape)

    def _get_scheduled_fixed_ent_coef(self) -> th.Tensor:
        if self.fixed_ent_coef_initial is None:
            return self.ent_coef_tensor
        ent_coef = self.ent_coef_tensor
        if self.fixed_ent_coef_final is None:
            return ent_coef
        schedule_ratio = self._get_current_fixed_ent_coef_ratio()
        current_value = self.fixed_ent_coef_initial + schedule_ratio * (self.fixed_ent_coef_final - self.fixed_ent_coef_initial)
        return th.tensor(current_value, device=self.device, dtype=th.float32)

    def _apply_progressive_residual_mask(self, residual_actions: th.Tensor) -> th.Tensor:
        if self.progressive_explore_steps <= 0:
            return residual_actions
        policy_step_count = self.num_timesteps + residual_actions.shape[0]
        residual_ratio = min(float(policy_step_count) / float(self.progressive_explore_steps), 1.0)
        if residual_ratio >= 1.0:
            return residual_actions
        mask = (
            th.rand((residual_actions.shape[0], 1), device=residual_actions.device, dtype=residual_actions.dtype)
            < residual_ratio
        )
        return residual_actions * mask

    def _sample_pre_learning_residual_actions(
        self,
        observations: Union[np.ndarray, th.Tensor],
        base_actions: th.Tensor,
    ) -> th.Tensor:
        batch_size = base_actions.shape[0]
        if self.pre_learning_residual_mode == "random_residual":
            return th.empty(
                (batch_size, *self.action_space.shape), device=self.device, dtype=th.float32
            ).uniform_(-1.0, 1.0)
        if self.pre_learning_residual_mode == "base_only":
            return th.zeros((batch_size, *self.action_space.shape), device=self.device, dtype=th.float32)
        return self._sample_residual_actions(observations, deterministic=False, base_actions=base_actions)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self._update_optimizer_learning_rates()

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        actor_grad_norms, critic_grad_norms = [], []
        base_action_norms, residual_env_norms = [], []
        residual_delta_norms, action_clip_fracs = [], []
        actor_log_std_means, actor_log_std_mins, actor_log_std_maxs = [], [], []

        if self.actor_gradient_steps < 0:
            actor_gradient_idx = set(range(gradient_steps))
        else:
            actor_gradient_idx = set(
                np.linspace(
                    int(gradient_steps / self.actor_gradient_steps) - 1,
                    gradient_steps - 1,
                    self.actor_gradient_steps,
                    dtype=int,
                ).tolist()
            )

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            if self.use_sde:
                self.actor.reset_noise()

            update_actor = gradient_step in actor_gradient_idx
            update_ent_coef = self.ent_coef_optimizer is not None and self.log_ent_coef is not None
            residual_actions_pi = None
            log_prob = None
            base_actions = replay_data.base_actions

            # With a fixed alpha, critic-only steps do not use the current-policy
            # sample. Avoiding it is mathematically identical and matters at high UTD.
            if update_actor or update_ent_coef:
                actor_obs = self._build_actor_observations(
                    replay_data.observations, base_actions=base_actions
                )
                mean_actions_pi, log_std_pi, dist_kwargs = self.actor.get_action_dist_params(actor_obs)
                residual_actions_pi, log_prob = self.actor.action_dist.log_prob_from_params(
                    mean_actions_pi, log_std_pi, **dist_kwargs
                )
                log_prob = log_prob.reshape(-1, 1)
                actor_log_std_means.append(log_std_pi.mean().detach())
                actor_log_std_mins.append(log_std_pi.min().detach())
                actor_log_std_maxs.append(log_std_pi.max().detach())

                if base_actions is not None:
                    projected_residual_actions_pi = self._project_residual_actions(residual_actions_pi)
                    residual_env_actions = projected_residual_actions_pi * self.action_half_range
                    final_env_actions_pre_clip = base_actions + self.residual_scale * residual_env_actions
                    final_env_actions = th.max(
                        th.min(final_env_actions_pre_clip, self.action_high), self.action_low
                    )
                    base_action_norms.append(base_actions.norm(dim=1).mean().detach())
                    residual_env_norms.append(residual_env_actions.norm(dim=1).mean().detach())
                    residual_delta_norms.append(
                        (final_env_actions - base_actions).norm(dim=1).mean().detach()
                    )
                    action_clip_fracs.append(
                        (
                            (final_env_actions_pre_clip < self.action_low)
                            | (final_env_actions_pre_clip > self.action_high)
                        )
                        .float()
                        .mean()
                        .detach()
                    )

            ent_coef_loss = None
            if update_ent_coef:
                assert log_prob is not None
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.detach())
            else:
                ent_coef = self._get_scheduled_fixed_ent_coef()

            ent_coefs.append(ent_coef.detach())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                base_next_actions = replay_data.next_base_actions
                if base_next_actions is None:
                    raise ValueError("ResidualSAC requires replayed base next actions for training.")
                next_actor_obs = self._build_actor_observations(replay_data.next_observations, base_actions=base_next_actions)
                next_residual_actions, next_log_prob = self.actor.action_log_prob(next_actor_obs)
                _, next_final_actions = self._combine_actions(base_next_actions, next_residual_actions)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_final_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.detach())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            if self.max_grad_norm is not None:
                q_network_grad_norms = []
                for q_network in self.critic.q_networks:
                    q_grad_norm = th.nn.utils.clip_grad_norm_(q_network.parameters(), self.max_grad_norm)
                    q_network_grad_norms.append(q_grad_norm.detach())
                critic_grad_norms.append(th.stack(q_network_grad_norms).mean())
            self.critic.optimizer.step()

            if update_actor:
                if base_actions is None:
                    raise ValueError("ResidualSAC requires replayed base actions for training.")
                assert residual_actions_pi is not None and log_prob is not None
                _, final_actions_pi = self._combine_actions(base_actions, residual_actions_pi)
                q_values_pi = th.cat(self.critic(replay_data.observations, final_actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                if self.action_norm_regularization > 0:
                    actor_loss = (ent_coef * log_prob - min_qf_pi).mean() + 0.5 * self.action_norm_regularization * th.mean(
                        residual_actions_pi ** 2
                    )
                else:
                    actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.detach())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                if self.max_grad_norm is not None:
                    actor_grad_norm = th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    actor_grad_norms.append(actor_grad_norm.detach())
                self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        def metric_mean(values: list[th.Tensor]) -> float:
            return float(th.stack(values).mean().cpu())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", metric_mean(ent_coefs))
        self.logger.record("train/actor_loss", metric_mean(actor_losses))
        self.logger.record("train/critic_loss", metric_mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", metric_mean(ent_coef_losses))
        if len(actor_grad_norms) > 0:
            self.logger.record("train/actor_grad_norm", metric_mean(actor_grad_norms))
        if len(critic_grad_norms) > 0:
            self.logger.record("train/critic_grad_norm", metric_mean(critic_grad_norms))
        if len(base_action_norms) > 0:
            self.logger.record("train/base_action_norm", metric_mean(base_action_norms))
        if len(residual_env_norms) > 0:
            self.logger.record("train/residual_env_action_norm", metric_mean(residual_env_norms))
        if len(residual_delta_norms) > 0:
            self.logger.record("train/residual_delta_norm", metric_mean(residual_delta_norms))
        if len(action_clip_fracs) > 0:
            self.logger.record("train/action_clip_frac", metric_mean(action_clip_fracs))
        if len(actor_log_std_means) > 0:
            self.logger.record("train/actor_log_std_mean", metric_mean(actor_log_std_means))
            self.logger.record("train/actor_log_std_min", metric_mean(actor_log_std_mins))
            self.logger.record("train/actor_log_std_max", metric_mean(actor_log_std_maxs))
        self.logger.record("train/residual_temporal_group_size", self.residual_temporal_group_size)
        self.logger.record("train/residual_active_steps", self.residual_active_steps)
        if self.ent_coef_optimizer is None and self.fixed_ent_coef_final is not None:
            self.logger.record("train/ent_coef_schedule_progress", self._get_current_fixed_ent_coef_ratio())

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        del action_noise  # Residual exploration is handled by the stochastic actor and progressive masking.

        assert self._last_obs is not None, "self._last_obs was not set"
        base_actions = self._sample_base_actions(self._last_obs)
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            residual_actions = self._sample_pre_learning_residual_actions(self._last_obs, base_actions)
        else:
            residual_actions = self._sample_residual_actions(self._last_obs, deterministic=False, base_actions=base_actions)

        residual_actions = self._apply_progressive_residual_mask(residual_actions)
        self._last_base_actions = base_actions.detach().cpu().numpy()
        final_env_actions, final_norm_actions = self._combine_actions(base_actions, residual_actions)
        return final_env_actions.detach().cpu().numpy(), final_norm_actions.detach().cpu().numpy()

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        if self._last_base_actions is None:
            raise ValueError("ResidualSAC expected cached base actions before storing transitions.")

        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        next_obs = deepcopy(new_obs_)
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])  # type: ignore[assignment]

        next_base_actions = self._sample_base_actions(next_obs).detach().cpu().numpy()
        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            next_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
            infos,
            base_action=self._last_base_actions,
            next_base_action=next_base_actions,
        )

        self._last_obs = new_obs
        self._last_base_actions = next_base_actions
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        del episode_start
        self.policy.set_training_mode(False)
        if isinstance(observation, dict):
            raise ValueError("ResidualSAC only supports array observations.")

        obs_tensor = th.as_tensor(observation, device=self.device, dtype=th.float32)
        vectorized_env = obs_tensor.ndim > len(self.observation_space.shape)
        if not vectorized_env:
            obs_tensor = obs_tensor.unsqueeze(0)

        with th.no_grad():
            base_actions = self._sample_base_actions(obs_tensor, deterministic=deterministic)
            residual_actions = self._sample_residual_actions(obs_tensor, deterministic=deterministic, base_actions=base_actions)
            final_env_actions, _ = self._combine_actions(base_actions, residual_actions)

        actions = final_env_actions.cpu().numpy().reshape((-1, *self.action_space.shape))
        if not vectorized_env:
            actions = actions.squeeze(axis=0)
        return actions, state

    def predict_base(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        del state
        if isinstance(observation, dict):
            raise ValueError("ResidualSAC only supports array observations.")

        obs_tensor = th.as_tensor(observation, device=self.device, dtype=th.float32)
        vectorized_env = obs_tensor.ndim > len(self.observation_space.shape)
        if not vectorized_env:
            obs_tensor = obs_tensor.unsqueeze(0)

        with th.no_grad():
            base_actions = self._sample_base_actions(obs_tensor)
        actions = base_actions.cpu().numpy().reshape((-1, *self.action_space.shape))
        if not vectorized_env:
            actions = actions.squeeze(axis=0)
        return actions, None

    def learn(
        self: SelfResidualSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "ResidualSAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfResidualSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["base_policy"]  # noqa: RUF005
