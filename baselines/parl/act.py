"""
Policy-Agnostic RL model for ACT action-chunk policies.

This class reuses the PA-RL critic/action-optimization implementation from the
diffusion PA-RL model, but swaps the base policy and actor distillation loss to
ACT's checkpoint-native forward path.
"""

import copy
import logging
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from baselines.parl.diffusion import PARLDQLDiffusion

log = logging.getLogger(__name__)

ACTION_KEY = "action"


class ACTPARLDQL(PARLDQLDiffusion):
    """PA-RL wrapper around a deterministic lowdim ACT policy."""

    DEFAULT_LOWDIM_DIMS = {
        "robot0_eef_pos": 3,
        "robot0_eef_quat": 4,
        "robot0_gripper_qpos": 2,
        "robot1_eef_pos": 3,
        "robot1_eef_quat": 4,
        "robot1_gripper_qpos": 2,
    }

    def __init__(
        self,
        critic,
        network_path=None,
        actor_checkpoint_path=None,
        actor_wrapper=None,
        task=None,
        state_obs_keys=None,
        env_obs_keys=None,
        lowdim_keys=None,
        lowdim_dims=None,
        horizon_steps=32,
        obs_dim=19,
        action_dim=7,
        device="cuda:0",
        final_action_clip_value=1.0,
        # Conservative Q-Learning regularization. Keep names aligned with
        # PARLDQLDiffusion so configs can be copied with minimal edits.
        cql_alpha=0.0,
        cql_n_actions=10,
        cql_temp=1.0,
        cql_action_sample_method="uniform",
        cql_include_random_actions=True,
        cql_include_current_actions=True,
        cql_include_next_actions=True,
        cql_include_data_action=True,
        cql_importance_sample=False,
        cql_clip_diff_min=None,
        cql_clip_diff_max=None,
        cql_action_low=-1.0,
        cql_action_high=1.0,
        distributional_critic=False,
        critic_subsample_size=None,
        use_calql=False,
        use_calql_on_random_actions=False,
        cql_use_parl_policy_actions=True,
        cql_num_base_policy_actions=1,
        cql_num_actions_to_keep=1,
        cql_local_optimization_steps=10,
        cql_local_optimization_step_size=3e-4,
        cql_optimize_critic_ensemble_min=False,
        cql_use_target_critic_for_optimization=False,
        cql_policy_argmax=True,
        cql_policy_temperature=1.0,
        cql_policy_use_min_q=False,
        backup_use_parl_policy_actions=True,
        backup_policy_argmax=True,
        **kwargs,
    ):
        del kwargs
        nn.Module.__init__(self)
        self.device = torch.device(device)
        self.horizon_steps = int(horizon_steps)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.final_action_clip_value = final_action_clip_value

        self.actor_metadata = {}
        if actor_wrapper is None:
            checkpoint_path = actor_checkpoint_path or network_path
            if checkpoint_path is None:
                raise ValueError("ACTPARLDQL requires actor_checkpoint_path or network_path.")
            from model.act.act_ckpt_loader import load_act_policy_wrapper

            wrapper, metadata = load_act_policy_wrapper(
                ckpt_path=checkpoint_path,
                device=self.device,
                task=task,
                state_obs_keys=self._parse_key_list(state_obs_keys),
                env_obs_keys=self._parse_key_list(env_obs_keys),
            )
            actor_wrapper = wrapper.policy
            self.actor_metadata = metadata

        self.actor = actor_wrapper.to(self.device)
        self.critic = critic.to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.state_obs_keys = (
            self._parse_key_list(state_obs_keys)
            or list(self.actor_metadata.get("state_obs_keys") or [])
            or ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        )
        self.env_obs_keys = (
            self._parse_key_list(env_obs_keys)
            or list(self.actor_metadata.get("env_obs_keys") or [])
            or ["object"]
        )
        self.lowdim_keys = (
            self._parse_key_list(lowdim_keys)
            or list(self.actor_metadata.get("lowdim_keys") or [])
            or list(dict.fromkeys(self.state_obs_keys + self.env_obs_keys))
        )
        self.lowdim_dims = self._resolve_lowdim_dims(lowdim_dims)
        self.lowdim_slices = self._build_lowdim_slices()

        self.cql_alpha = float(cql_alpha)
        self.cql_n_actions = int(cql_n_actions)
        self.cql_temp = float(cql_temp)
        self.cql_action_sample_method = cql_action_sample_method
        self.cql_include_random_actions = bool(cql_include_random_actions)
        self.cql_include_current_actions = bool(cql_include_current_actions)
        self.cql_include_next_actions = bool(cql_include_next_actions)
        self.cql_include_data_action = bool(cql_include_data_action)
        self.cql_importance_sample = bool(cql_importance_sample)
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.cql_action_low = float(cql_action_low)
        self.cql_action_high = float(cql_action_high)
        self.distributional_critic = bool(distributional_critic)
        self.critic_subsample_size = critic_subsample_size
        self.use_calql = bool(use_calql)
        self.use_calql_on_random_actions = bool(use_calql_on_random_actions)
        self.cql_use_parl_policy_actions = bool(cql_use_parl_policy_actions)
        self.cql_num_base_policy_actions = int(cql_num_base_policy_actions)
        self.cql_num_actions_to_keep = int(cql_num_actions_to_keep)
        self.cql_local_optimization_steps = int(cql_local_optimization_steps)
        self.cql_local_optimization_step_size = float(cql_local_optimization_step_size)
        self.cql_optimize_critic_ensemble_min = bool(cql_optimize_critic_ensemble_min)
        self.cql_use_target_critic_for_optimization = bool(cql_use_target_critic_for_optimization)
        self.cql_policy_argmax = bool(cql_policy_argmax)
        self.cql_policy_temperature = float(cql_policy_temperature)
        self.cql_policy_use_min_q = bool(cql_policy_use_min_q)
        self.backup_use_parl_policy_actions = bool(backup_use_parl_policy_actions)
        self.backup_policy_argmax = bool(backup_policy_argmax)
        self._candidate_repeat_warned = False
        self.last_actor_loss_dict = {}
        log.info(
            "Loaded ACT PA-RL actor with lowdim keys %s and dims %s",
            self.lowdim_keys,
            self.lowdim_dims,
        )

    @staticmethod
    def _parse_key_list(spec):
        if spec is None:
            return []
        if isinstance(spec, str):
            return [part.strip() for part in spec.split(",") if part.strip()]
        return [str(item).strip() for item in spec if str(item).strip()]

    def _resolve_lowdim_dims(self, lowdim_dims):
        if lowdim_dims is None:
            lowdim_dims = self.actor_metadata.get("lowdim_dims")
        if isinstance(lowdim_dims, Mapping):
            dims = {str(key): int(value) for key, value in lowdim_dims.items()}
        elif lowdim_dims is not None:
            values = [int(value) for value in lowdim_dims]
            if len(values) != len(self.lowdim_keys):
                raise ValueError("lowdim_dims list must match lowdim_keys length")
            dims = dict(zip(self.lowdim_keys, values))
        else:
            dims = {}

        for key in self.lowdim_keys:
            if key in dims:
                continue
            if key in self.DEFAULT_LOWDIM_DIMS:
                dims[key] = self.DEFAULT_LOWDIM_DIMS[key]

        missing = [key for key in self.lowdim_keys if key not in dims]
        if missing:
            known_total = sum(dims.get(key, 0) for key in self.lowdim_keys)
            if len(missing) == 1 and self.obs_dim > known_total:
                dims[missing[0]] = self.obs_dim - known_total
            else:
                raise ValueError(f"Cannot infer lowdim dims for {missing}; pass model.lowdim_dims.")
        return dims

    def _build_lowdim_slices(self):
        slices = {}
        offset = 0
        for key in self.lowdim_keys:
            dim = int(self.lowdim_dims[key])
            slices[key] = slice(offset, offset + dim)
            offset += dim
        if offset != self.obs_dim:
            raise ValueError(f"lowdim dims sum to {offset}, but obs_dim={self.obs_dim}")
        return slices

    def _cond_state_latest(self, cond):
        if "state" not in cond:
            raise KeyError("ACTPARLDQL expects cond['state']")
        state = cond["state"].to(self.device, dtype=torch.float32)
        if state.ndim == 3:
            state = state[:, -1]
        elif state.ndim > 3:
            state = state.reshape(state.shape[0], -1)
        if state.ndim != 2 or state.shape[-1] != self.obs_dim:
            raise ValueError(
                f"cond['state'] must end with obs_dim={self.obs_dim}, got {tuple(state.shape)}"
            )
        return state

    def _split_cond_to_obs_dict(self, cond):
        state = self._cond_state_latest(cond)
        return {key: state[:, self.lowdim_slices[key]] for key in self.lowdim_keys}

    def _actor_policy_obs(self, cond):
        obs_dict = self._split_cond_to_obs_dict(cond)
        if hasattr(self.actor, "_build_policy_obs"):
            return self.actor._build_policy_obs(obs_dict)
        return obs_dict

    def _act_action_chunk(self, cond):
        obs_dict = self._split_cond_to_obs_dict(cond)
        was_training = self.actor.training
        if hasattr(self.actor, "_get_action_trajectory"):
            actions = self.actor._get_action_trajectory(obs_dict)
        else:
            actions = self.actor(obs_dict)
        if was_training:
            self.actor.train()
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions = actions.to(self.device, dtype=torch.float32)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.shape[1:] != (self.horizon_steps, self.action_dim):
            raise ValueError(
                f"ACT action chunk shape {tuple(actions.shape[1:])} does not match "
                f"{(self.horizon_steps, self.action_dim)}"
            )
        if self.final_action_clip_value is not None:
            clip = float(self.final_action_clip_value)
            actions = torch.clamp(actions, -clip, clip)
        return actions

    def _sample_random_cql_actions(self, batch_size, num_actions, action_steps):
        shape = (batch_size, num_actions, action_steps, self.action_dim)
        if self.cql_action_sample_method == "uniform":
            actions = torch.empty(shape, device=self.device)
            actions.uniform_(self.cql_action_low, self.cql_action_high)
            return actions
        if self.cql_action_sample_method == "normal":
            return torch.randn(shape, device=self.device)
        raise ValueError(f"Unsupported cql_action_sample_method={self.cql_action_sample_method}")

    @torch.no_grad()
    def forward(self, cond, deterministic=False):
        del deterministic
        return self._act_action_chunk(cond)

    @torch.no_grad()
    def sample_candidate_actions(self, cond, num_base_policy_actions):
        """Return deterministic ACT action chunks, repeated without noise."""
        num_base_policy_actions = int(num_base_policy_actions)
        if num_base_policy_actions <= 0:
            raise ValueError("num_base_policy_actions must be positive")
        if num_base_policy_actions > 1 and not self._candidate_repeat_warned:
            log.warning(
                "ACT actor is deterministic; repeating the same base action %d times "
                "without artificial noise.",
                num_base_policy_actions,
            )
            self._candidate_repeat_warned = True
        actions = self.forward(cond, deterministic=True)
        return actions[:, None].repeat(1, num_base_policy_actions, 1, 1)

    def loss(self, target_actions, cond):
        """ACT checkpoint-native BC loss on PA-RL optimized action chunks."""
        if hasattr(self.actor, "loss_on_targets"):
            return self.actor.loss_on_targets(self._split_cond_to_obs_dict(cond), target_actions)

        target_actions = target_actions.to(self.device, dtype=torch.float32)
        if target_actions.shape[1:] != (self.horizon_steps, self.action_dim):
            raise ValueError(
                f"target_actions must have shape (B, {self.horizon_steps}, {self.action_dim})"
            )
        batch = dict(self._actor_policy_obs(cond))
        batch[ACTION_KEY] = target_actions
        batch["action_is_pad"] = torch.zeros(
            target_actions.shape[:2],
            dtype=torch.bool,
            device=target_actions.device,
        )
        if hasattr(self.actor, "preprocessor"):
            batch = self.actor.preprocessor(batch)
        policy = getattr(self.actor, "policy_model", self.actor)
        loss, loss_dict = policy(batch)
        self.last_actor_loss_dict = {
            f"parl/act_{key}": float(value)
            for key, value in (loss_dict or {}).items()
            if isinstance(value, (int, float))
        }
        return loss

    def loss_actor_distill(
        self,
        obs,
        replay_actions,
        num_base_policy_actions,
        num_actions_to_keep,
        act_steps,
        distill_argmax=False,
        softmax_temperature=1.0,
        consider_replay_action=True,
        use_min_q=True,
        local_optimization_steps=10,
        local_optimization_step_size=3e-4,
        optimize_critic_ensemble_min=False,
        use_target_critic=False,
    ):
        candidate_actions = self.sample_candidate_actions(
            obs,
            num_base_policy_actions=num_base_policy_actions,
        )
        target_actions, metrics = self.rank_candidate_actions(
            obs,
            candidate_actions,
            act_steps=act_steps,
            replay_actions=replay_actions,
            num_actions_to_keep=num_actions_to_keep,
            distill_argmax=distill_argmax,
            softmax_temperature=softmax_temperature,
            consider_replay_action=consider_replay_action,
            use_min_q=use_min_q,
            local_optimization_steps=local_optimization_steps,
            local_optimization_step_size=local_optimization_step_size,
            optimize_critic_ensemble_min=optimize_critic_ensemble_min,
            use_target_critic=use_target_critic,
        )
        actor_loss = self.loss(target_actions, obs)
        metrics["parl/loss_actor_distill"] = actor_loss.detach().item()
        metrics.update(getattr(self, "last_actor_loss_dict", {}))
        return actor_loss, metrics
