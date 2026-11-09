"""
Policy-Agnostic RL on top of DSRL Diffusion Q-Learning.

Phase A keeps DSRL's online rollout, replay buffer, and twin-Q critic training,
but freezes the diffusion actor. Phase B hooks are present behind a disabled
flag and use supervised distillation on Q-reranked action candidates instead of
the DQL actor gradient.
"""

import json
import os
import pickle
import random
import numpy as np
import torch
import logging

try:
    import wandb
except ImportError:
    wandb = None
from omegaconf import OmegaConf

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - tensorboard is optional at runtime
    SummaryWriter = None

log = logging.getLogger(__name__)
from util.timer import Timer
from collections import deque
from baselines.parl.training import TrainAgent
from util.scheduler import CosineAnnealingWarmupRestarts


PARL_ASYNC_EVAL_SNAPSHOT_FORMAT = "parl_dql_diffusion_eval_v1"


class _NumpyRingBuffer:
    """Growable ring buffer with O(batch) sampling and checkpoint conversion."""

    def __init__(self, maxlen, dtype, values=None):
        self.maxlen = int(maxlen)
        if self.maxlen <= 0:
            raise ValueError("Replay buffer maxlen must be positive")
        self.dtype = np.dtype(dtype)
        self._data = None
        self._size = 0
        self._start = 0
        if values is not None:
            self._load_values(values)

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter(self.to_numpy())

    def __array__(self, dtype=None, copy=None):
        array = self.to_numpy()
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array.copy() if copy else array

    @property
    def allocated(self):
        return 0 if self._data is None else len(self._data)

    def _allocation_size(self, required):
        allocated = min(self.maxlen, 1024)
        while allocated < required:
            allocated = min(self.maxlen, allocated * 2)
        return allocated

    def _load_values(self, values):
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, dtype=self.dtype)
        if values.ndim == 0:
            values = values.reshape(1)
        values = values[-self.maxlen :]
        if len(values) == 0:
            return
        allocated = self._allocation_size(len(values))
        self._data = np.empty((allocated, *values.shape[1:]), dtype=self.dtype)
        self._data[: len(values)] = values
        self._size = len(values)

    def _grow(self, value_shape):
        required = self._size + 1
        allocated = self._allocation_size(required)
        new_data = np.empty((allocated, *value_shape), dtype=self.dtype)
        if self._size:
            new_data[: self._size] = self.to_numpy()
        self._data = new_data
        self._start = 0

    def append(self, value):
        value = np.asarray(value, dtype=self.dtype)
        if self._data is not None and value.shape != self._data.shape[1:]:
            raise ValueError(
                f"Replay value shape {value.shape} does not match {self._data.shape[1:]}"
            )
        if self._data is None or (self._size == self.allocated and self.allocated < self.maxlen):
            self._grow(value.shape)

        if self._size < self.allocated:
            index = (self._start + self._size) % self.allocated
            self._size += 1
        else:
            index = self._start
            self._start = (self._start + 1) % self.allocated
        self._data[index] = value

    def take(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            return self.to_numpy()[:0]
        if np.any(indices < 0) or np.any(indices >= self._size):
            raise IndexError("Replay sample index out of range")
        physical = (self._start + indices) % self.allocated
        return self._data[physical]

    def to_numpy(self):
        if self._size == 0:
            if self._data is None:
                return np.empty((0,), dtype=self.dtype)
            return self._data[:0]
        end = self._start + self._size
        if end <= self.allocated:
            return self._data[self._start : end]
        return np.concatenate(
            [self._data[self._start :], self._data[: end - self.allocated]],
            axis=0,
        )


class TrainPARLDQLDiffusionAgent(TrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # note the discount factor gamma here is applied to reward every act_steps, instead of every env step
        self.gamma = cfg.train.gamma

        # Fallback warmup in train iterations if train.parl.critic_warmup_updates
        # is not set.
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        parl_cfg = cfg.train.get("parl", {})
        self.actor_distill_enabled = parl_cfg.get("enable_actor_distill", False)
        self.actor_update_freq = max(1, int(parl_cfg.get("actor_update_freq", 4)))
        self.num_base_policy_actions = parl_cfg.get("num_base_policy_actions", 32)
        self.num_actions_to_keep = parl_cfg.get("num_actions_to_keep", 10)
        self.distill_argmax = parl_cfg.get("distill_argmax", False)
        self.softmax_temperature = parl_cfg.get("softmax_temperature", 1.0)
        self.consider_replay_action = parl_cfg.get("consider_replay_action", True)
        self.use_min_q_for_rerank = parl_cfg.get("use_min_q", True)
        self.local_optimization_steps = int(parl_cfg.get("local_optimization_steps", 10))
        self.local_optimization_step_size = float(
            parl_cfg.get("local_optimization_step_size", 3e-4)
        )
        distill_local_steps = parl_cfg.get("distill_local_optimization_steps", None)
        self.distill_local_optimization_steps = (
            self.local_optimization_steps
            if distill_local_steps is None
            else int(distill_local_steps)
        )
        distill_local_step_size = parl_cfg.get(
            "distill_local_optimization_step_size",
            None,
        )
        self.distill_local_optimization_step_size = (
            self.local_optimization_step_size
            if distill_local_step_size is None
            else float(distill_local_step_size)
        )
        self.optimize_critic_ensemble_min = bool(
            parl_cfg.get("optimize_critic_ensemble_min", False)
        )
        self.use_target_critic_for_optimization = bool(
            parl_cfg.get("use_target_critic_for_optimization", False)
        )
        self.actor_distill_start_updates = parl_cfg.get(
            "actor_distill_start_updates",
            None,
        )
        self.actor_distill_start_env_steps = parl_cfg.get(
            "actor_distill_start_env_steps",
            None,
        )
        self.optimized_eval_freq_env_steps = int(parl_cfg.get("optimized_eval_freq_env_steps", 0))
        self.optimized_eval_n_episodes = int(parl_cfg.get("optimized_eval_n_episodes", 200))
        self.optimized_eval_max_steps = parl_cfg.get(
            "optimized_eval_max_steps",
            None,
        )
        self.optimized_eval_at_start = bool(parl_cfg.get("optimized_eval_at_start", True))
        self.optimized_eval_async = bool(parl_cfg.get("optimized_eval_async", False))
        self.optimized_eval_snapshot_slots = int(parl_cfg.get("optimized_eval_snapshot_slots", 64))
        if self.optimized_eval_async:
            if self.optimized_eval_freq_env_steps <= 0:
                raise ValueError("optimized_eval_async requires optimized_eval_freq_env_steps > 0")
            if self.optimized_eval_snapshot_slots <= 0:
                raise ValueError("optimized_eval_snapshot_slots must be positive")
        self.override_actor_lr_after_resume = parl_cfg.get(
            "override_actor_lr_after_resume",
            None,
        )
        self.critic_warmup_updates = parl_cfg.get("critic_warmup_updates", None)
        self.critic_warmup_env_steps = parl_cfg.get("critic_warmup_env_steps", None)
        self.rollout_policy = parl_cfg.get("rollout_policy", "raw")
        self.eval_policy = parl_cfg.get("eval_policy", "raw")
        self.optimized_rollout_warmup_updates = parl_cfg.get(
            "optimized_rollout_warmup_updates",
            None,
        )
        self.optimized_policy_argmax = bool(parl_cfg.get("optimized_policy_argmax", True))
        self.optimized_policy_temperature = float(
            parl_cfg.get("optimized_policy_temperature", self.softmax_temperature)
        )
        valid_rollout_policies = {"raw", "optimized", "optimized_after_warmup"}
        valid_eval_policies = {"raw", "optimized"}
        if self.rollout_policy not in valid_rollout_policies:
            raise ValueError(
                f"rollout_policy must be one of {valid_rollout_policies}, got {self.rollout_policy}"
            )
        if self.eval_policy not in valid_eval_policies:
            raise ValueError(
                f"eval_policy must be one of {valid_eval_policies}, got {self.eval_policy}"
            )

        # Optimizer
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )

        # Buffer size
        self.buffer_size = cfg.train.buffer_size

        # Perturbation scale retained from DQL config for compatibility. PA-RL
        # actor distillation does not use DQL's Q-gradient actor loss.
        self.eta = cfg.train.eta

        # Reward factor - scale down mujoco reward for better critic training
        self.scale_reward_factor = cfg.train.scale_reward_factor
        self.reward_offset = cfg.train.get("reward_offset", 0.0)
        self.store_mc_returns = bool(getattr(self.model, "use_calql", False))
        self.calql_push_failed_to_min = cfg.train.get("calql_push_failed_to_min", False)
        self.calql_min_reward = cfg.train.get("calql_min_reward", -1.0)
        self.offline_dataset_path = cfg.train.get(
            "offline_dataset_path",
            cfg.get("offline_dataset_path", None),
        )
        self.offline_batch_ratio = float(cfg.train.get("offline_batch_ratio", 0.0))
        self.offline_max_n_episodes = int(cfg.train.get("offline_max_n_episodes", 1000000))
        self.offline_success_reward_threshold = float(
            cfg.train.get("offline_success_reward_threshold", 0.5)
        )
        if self.offline_batch_ratio < 0 or self.offline_batch_ratio > 1:
            raise ValueError("offline_batch_ratio must be in [0, 1]")
        if self.offline_batch_ratio > 0 and self.offline_dataset_path is None:
            raise ValueError("train.offline_dataset_path is required when offline_batch_ratio > 0")
        self.offline_replay = None
        if self.offline_dataset_path is not None and self.offline_batch_ratio > 0:
            self.offline_replay = self.build_offline_replay_from_npz(
                dataset_path=self.offline_dataset_path,
                act_steps=self.act_steps,
                cond_steps=self.n_cond_step,
                gamma=self.gamma,
                reward_offset=self.reward_offset,
                scale_reward_factor=self.scale_reward_factor,
                max_n_episodes=self.offline_max_n_episodes,
                success_reward_threshold=self.offline_success_reward_threshold,
            )
            log.info(
                "Loaded PA-RL offline replay from %s with %d chunk transitions",
                self.offline_dataset_path,
                len(self.offline_replay["obs"]),
            )

        # Updates
        self.replay_ratio = cfg.train.replay_ratio
        self.save_replay_buffer = bool(cfg.train.get("save_replay_buffer", True))
        self.save_model_freq_env_steps = int(cfg.train.get("save_model_freq_env_steps", 500000))
        self.max_env_steps = cfg.train.get("max_env_steps", None)
        if self.max_env_steps is not None:
            self.max_env_steps = int(self.max_env_steps)
            if self.max_env_steps <= 0:
                raise ValueError("train.max_env_steps must be positive when set")
        self._loaded_replay_state = None
        self._online_replay_buffers = None
        self._pending_episode_buffers = None
        self.discarded_resume_pending_transitions = 0
        if self.critic_warmup_updates is None:
            updates_per_train_itr = int(
                self.n_steps * self.n_envs / self.batch_size * self.replay_ratio
            )
            self.critic_warmup_updates = self.n_critic_warmup_itr * updates_per_train_itr
        if self.optimized_rollout_warmup_updates is None:
            self.optimized_rollout_warmup_updates = self.critic_warmup_updates
        if self.actor_distill_start_updates is None:
            self.actor_distill_start_updates = self.critic_warmup_updates
        self.actor_distill_start_updates = int(self.actor_distill_start_updates)
        if self.actor_distill_start_env_steps is not None:
            self.actor_distill_start_env_steps = int(self.actor_distill_start_env_steps)
        if self.optimized_eval_max_steps is not None:
            self.optimized_eval_max_steps = int(self.optimized_eval_max_steps)
        self.next_optimized_eval_step = (
            0
            if self.optimized_eval_freq_env_steps > 0 and self.optimized_eval_at_start
            else self.optimized_eval_freq_env_steps
            if self.optimized_eval_freq_env_steps > 0
            else None
        )

        # critic target update rate
        self.target_ema_rate = cfg.train.target_ema_rate

        self.cnt_train_step = 0
        self.cnt_critic_update = 0
        self.rollout_success_history = deque(maxlen=200)
        self.next_model_save_step = 0 if self.save_model_freq_env_steps > 0 else None
        self.resume_ckpt = cfg.train.get("resume_ckpt", cfg.get("resume_ckpt", None))
        if self.resume_ckpt is not None:
            self.load(self.resume_ckpt)
            if self.override_actor_lr_after_resume is not None:
                self._override_actor_lr(float(self.override_actor_lr_after_resume))

        self.use_tensorboard = cfg.train.get("tensorboard", True)
        self.tensorboard_dir = cfg.train.get("tensorboard_dir", "scalars_1")
        self.tb_writer = None
        if self.use_tensorboard:
            if SummaryWriter is None:
                log.warning("TensorBoard requested but torch.utils.tensorboard is unavailable.")
            else:
                self.tb_writer = SummaryWriter(
                    log_dir=os.path.join(self.logdir, self.tensorboard_dir)
                )

    def _override_actor_lr(self, actor_lr):
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = actor_lr
            param_group["initial_lr"] = actor_lr
        self.actor_lr_scheduler.min_lr = actor_lr
        self.actor_lr_scheduler.max_lr = actor_lr
        self.actor_lr_scheduler.base_max_lr = actor_lr
        self.actor_lr_scheduler.base_lrs = [actor_lr for _ in self.actor_optimizer.param_groups]
        log.info("Overrode actor optimizer and scheduler lr to %s after resume", actor_lr)

    def _record_rollout_successes(self, episode_successes):
        if not hasattr(self, "rollout_success_history"):
            self.rollout_success_history = deque(maxlen=200)
        self.rollout_success_history.extend(float(success) for success in episode_successes)

    def _recent_rollout_success_rate(self, window):
        if not hasattr(self, "rollout_success_history"):
            self.rollout_success_history = deque(maxlen=200)
        recent = list(self.rollout_success_history)[-window:]
        if len(recent) == 0:
            return 0.0
        return float(np.mean(recent))

    def _recent_rollout_success_metrics(self):
        return {
            "success_rate_recent20": self._recent_rollout_success_rate(20),
            "success_rate_recent100": self._recent_rollout_success_rate(100),
            "success_rate_recent200": self._recent_rollout_success_rate(200),
            "success_history_size": len(self.rollout_success_history),
        }

    @staticmethod
    def _as_scalar(value):
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            value = value.detach().cpu().item()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
        return None

    def log_tensorboard(self, scalars, step):
        if self.tb_writer is None:
            return
        for key, value in scalars.items():
            scalar = self._as_scalar(value)
            if scalar is not None:
                self.tb_writer.add_scalar(key, scalar, step)

    def _critic_warmup_done(self):
        warmup_done = self.cnt_critic_update >= self.critic_warmup_updates
        if self.critic_warmup_env_steps is not None:
            warmup_done = warmup_done and self.cnt_train_step >= self.critic_warmup_env_steps
        return warmup_done

    def _optimized_rollout_warmup_done(self):
        warmup_done = self.cnt_critic_update >= self.optimized_rollout_warmup_updates
        if self.critic_warmup_env_steps is not None:
            warmup_done = warmup_done and self.cnt_train_step >= self.critic_warmup_env_steps
        return warmup_done

    def _actor_distill_active(self):
        if not self.actor_distill_enabled:
            return False
        active = self.cnt_critic_update >= self.actor_distill_start_updates
        if self.actor_distill_start_env_steps is not None:
            active = active and self.cnt_train_step >= self.actor_distill_start_env_steps
        return active

    def _action_policy_for_mode(self, eval_mode):
        if eval_mode:
            return self.eval_policy
        if self.rollout_policy == "optimized_after_warmup":
            if self._optimized_rollout_warmup_done():
                return "optimized"
            return "raw"
        return self.rollout_policy

    def _select_action_chunks(self, cond, eval_mode, action_policy_override=None):
        action_policy = (
            action_policy_override
            if action_policy_override is not None
            else self._action_policy_for_mode(eval_mode)
        )
        if action_policy == "optimized":
            actions, metrics = self.model.sample_optimized_actions(
                cond,
                act_steps=self.act_steps,
                num_base_policy_actions=self.num_base_policy_actions,
                num_actions_to_keep=self.num_actions_to_keep,
                argmax=self.optimized_policy_argmax,
                softmax_temperature=self.optimized_policy_temperature,
                use_min_q=self.use_min_q_for_rerank,
                local_optimization_steps=self.local_optimization_steps,
                local_optimization_step_size=self.local_optimization_step_size,
                optimize_critic_ensemble_min=self.optimize_critic_ensemble_min,
                use_target_critic=self.use_target_critic_for_optimization,
            )
            metrics["policy/is_optimized"] = 1.0
            metrics["policy/critic_update_at_action"] = float(self.cnt_critic_update)
            return actions, metrics

        samples = self.model(
            cond=cond,
            deterministic=eval_mode,
        )
        return samples[:, : self.act_steps], {
            "policy/is_optimized": 0.0,
            "policy/critic_update_at_action": float(self.cnt_critic_update),
        }

    def _average_scalar_metrics(self, metrics_by_key):
        return {
            key: float(np.mean(values)) for key, values in metrics_by_key.items() if len(values) > 0
        }

    def _optimized_eval_due(self):
        return (
            self.next_optimized_eval_step is not None
            and self.cnt_train_step >= self.next_optimized_eval_step
        )

    def _advance_next_optimized_eval_step(self):
        if self.optimized_eval_freq_env_steps <= 0:
            self.next_optimized_eval_step = None
            return
        while self.next_optimized_eval_step <= self.cnt_train_step:
            self.next_optimized_eval_step += self.optimized_eval_freq_env_steps

    @staticmethod
    def _module_state_dict_to_cpu(module):
        return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}

    @staticmethod
    def _atomic_torch_save(payload, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f"{path}.tmp.{os.getpid()}"
        torch.save(payload, temporary)
        os.replace(temporary, path)

    @staticmethod
    def _atomic_json_write(path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f"{path}.tmp.{os.getpid()}"
        with open(temporary, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def _save_async_optimized_eval_snapshot(self):
        if self.next_optimized_eval_step is None:
            raise RuntimeError("No optimized evaluation is scheduled")
        scheduled_step = int(self.next_optimized_eval_step)
        interval = int(self.optimized_eval_freq_env_steps)
        slot_index = (scheduled_step // interval) % self.optimized_eval_snapshot_slots
        snapshot_dir = os.path.join(self.logdir, "eval_snapshots")
        snapshot_path = os.path.join(snapshot_dir, f"eval_slot_{slot_index:02d}.pt")
        sidecar_path = os.path.join(snapshot_dir, f"eval_slot_{slot_index:02d}.json")

        if os.path.isfile(sidecar_path):
            with open(sidecar_path) as stream:
                previous = json.load(stream)
            previous_step = int(previous["scheduled_eval_step"])
            if previous_step == scheduled_step:
                log.info(
                    "Reusing queued PA-RL eval snapshot for step %d after resume",
                    scheduled_step,
                )
                return snapshot_path
            acknowledgement = os.path.join(
                snapshot_dir,
                f"ack_step_{previous_step:012d}.json",
            )
            if not os.path.isfile(acknowledgement):
                raise RuntimeError(
                    "PA-RL async eval queue exhausted: "
                    f"slot {slot_index} step {previous_step} has no watcher ack"
                )

        payload = {
            "snapshot_format": PARL_ASYNC_EVAL_SNAPSHOT_FORMAT,
            "scheduled_eval_step": scheduled_step,
            "train_env_step": int(self.cnt_train_step),
            "critic_update": int(self.cnt_critic_update),
            "itr": int(self.itr),
            "seed": int(self.seed),
            "actor": self._module_state_dict_to_cpu(self.model.actor),
            "critic": self._module_state_dict_to_cpu(self.model.critic),
        }
        if getattr(self, "use_target_critic_for_optimization", False):
            payload["critic_target"] = self._module_state_dict_to_cpu(self.model.critic_target)
        self._atomic_torch_save(payload, snapshot_path)
        self._atomic_json_write(
            sidecar_path,
            {
                "snapshot_format": PARL_ASYNC_EVAL_SNAPSHOT_FORMAT,
                "scheduled_eval_step": scheduled_step,
                "train_env_step": int(self.cnt_train_step),
                "critic_update": int(self.cnt_critic_update),
                "slot_index": slot_index,
                "snapshot_file": os.path.basename(snapshot_path),
            },
        )
        log.info(
            "Queued PA-RL optimized eval snapshot: scheduled=%d actual=%d slot=%d",
            scheduled_step,
            self.cnt_train_step,
            slot_index,
        )
        return snapshot_path

    def _model_save_due(self):
        if self.itr == self.n_train_itr - 1:
            return True
        max_env_steps = getattr(self, "max_env_steps", None)
        if max_env_steps is not None and self.cnt_train_step >= max_env_steps:
            return True
        if self.save_model_freq_env_steps > 0:
            return (
                self.next_model_save_step is not None
                and self.cnt_train_step >= self.next_model_save_step
            )
        return self.itr % self.save_model_freq == 0

    def _advance_next_model_save_step(self):
        if self.save_model_freq_env_steps <= 0:
            self.next_model_save_step = None
            return
        while self.next_model_save_step <= self.cnt_train_step:
            self.next_model_save_step += self.save_model_freq_env_steps

    @staticmethod
    def offset_rewards(raw_rewards, reward_offset, exec_lens=None):
        raw_rewards = np.asarray(raw_rewards)
        if exec_lens is None:
            exec_lens = 1.0
        return raw_rewards - float(reward_offset) * np.asarray(exec_lens)

    @staticmethod
    def calc_return_to_go(
        rewards,
        masks,
        gamma,
        push_failed_to_min=False,
        min_reward=-1.0,
    ):
        rewards = np.asarray(rewards, dtype=np.float32)
        masks = np.asarray(masks, dtype=np.float32)
        if len(rewards) == 0:
            return np.array([], dtype=np.float32)
        if rewards[-1] == min_reward and push_failed_to_min:
            assert np.all(rewards <= 0)
            return np.full_like(rewards, min_reward / (1 - gamma), dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        prev_return = 0.0
        for i in reversed(range(len(rewards))):
            returns[i] = rewards[i] + gamma * prev_return * masks[i]
            prev_return = returns[i]
        return returns

    @staticmethod
    def _state_history(states, index, cond_steps):
        history = [states[max(index - t, 0)] for t in reversed(range(cond_steps))]
        return np.stack(history, axis=0).astype(np.float32)

    @staticmethod
    def _lowlevel_return_to_success(
        rewards,
        start,
        gamma,
        act_steps,
        reward_offset,
        scale_reward_factor,
        success_reward_threshold,
    ):
        gamma_low = float(gamma) ** (1.0 / max(int(act_steps), 1))
        future_rewards = rewards[start:]
        success_offsets = np.flatnonzero(future_rewards >= float(success_reward_threshold))
        end = len(rewards)
        if success_offsets.size > 0:
            end = start + int(success_offsets[0]) + 1
        adjusted = (rewards[start:end].astype(np.float32) - float(reward_offset)) * float(
            scale_reward_factor
        )
        discounts = gamma_low ** np.arange(len(adjusted), dtype=np.float32)
        return float(np.sum(adjusted * discounts))

    @classmethod
    def build_offline_replay_from_npz(
        cls,
        dataset_path,
        act_steps,
        cond_steps,
        gamma,
        reward_offset,
        scale_reward_factor=1.0,
        max_n_episodes=1000000,
        success_reward_threshold=0.5,
    ):
        dataset = np.load(os.path.expanduser(dataset_path), allow_pickle=False)
        required_keys = {"states", "actions", "rewards", "traj_lengths"}
        missing_keys = required_keys.difference(dataset.files)
        if missing_keys:
            raise ValueError(f"Offline PA-RL dataset {dataset_path} is missing keys {missing_keys}")
        states = dataset["states"].astype(np.float32)
        actions = dataset["actions"].astype(np.float32)
        rewards = dataset["rewards"].astype(np.float32)
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]

        obs_buffer = []
        next_obs_buffer = []
        action_buffer = []
        reward_buffer = []
        terminated_buffer = []
        mc_return_buffer = []

        offset = 0
        for traj_len in traj_lengths:
            traj_len = int(traj_len)
            ep_states = states[offset : offset + traj_len]
            ep_actions = actions[offset : offset + traj_len]
            ep_rewards = rewards[offset : offset + traj_len]
            offset += traj_len
            if traj_len < act_steps:
                continue

            for start in range(0, traj_len - act_steps + 1):
                action_chunk = ep_actions[start : start + act_steps]
                if action_chunk.shape[0] != act_steps:
                    continue
                reward_chunk = ep_rewards[start : start + act_steps]
                success_offsets = np.flatnonzero(reward_chunk >= float(success_reward_threshold))
                if success_offsets.size > 0:
                    exec_len = int(success_offsets[0]) + 1
                    terminated = True
                else:
                    exec_len = int(act_steps)
                    terminated = False
                    if start + exec_len >= traj_len:
                        continue
                raw_reward = float(np.sum(reward_chunk[:exec_len]))
                shifted_reward = (raw_reward - float(reward_offset) * float(exec_len)) * float(
                    scale_reward_factor
                )
                next_index = min(start + exec_len, traj_len - 1)

                obs_buffer.append(cls._state_history(ep_states, start, cond_steps))
                next_obs_buffer.append(cls._state_history(ep_states, next_index, cond_steps))
                action_buffer.append(action_chunk.astype(np.float32))
                reward_buffer.append(float(shifted_reward))
                terminated_buffer.append(bool(terminated))
                mc_return_buffer.append(
                    cls._lowlevel_return_to_success(
                        ep_rewards,
                        start=start,
                        gamma=gamma,
                        act_steps=act_steps,
                        reward_offset=reward_offset,
                        scale_reward_factor=scale_reward_factor,
                        success_reward_threshold=success_reward_threshold,
                    )
                )

        if len(obs_buffer) == 0:
            raise ValueError(f"Offline PA-RL dataset {dataset_path} produced no chunk transitions")
        return {
            "obs": np.asarray(obs_buffer, dtype=np.float32),
            "next_obs": np.asarray(next_obs_buffer, dtype=np.float32),
            "actions": np.asarray(action_buffer, dtype=np.float32),
            "rewards": np.asarray(reward_buffer, dtype=np.float32),
            "terminated": np.asarray(terminated_buffer, dtype=np.float32),
            "mc_returns": np.asarray(mc_return_buffer, dtype=np.float32),
        }

    @staticmethod
    def _extract_success(info):
        raw = info.get("is_success", False) if isinstance(info, dict) else False
        if isinstance(raw, dict):
            raw = raw.get("task", False)
        if isinstance(raw, (list, tuple, np.ndarray)):
            if len(raw) == 0:
                return False
            raw = raw[-1]
            if isinstance(raw, dict):
                raw = raw.get("task", False)
        return bool(raw)

    @classmethod
    def extract_success_flags(cls, info_venv, n_envs):
        return np.array(
            [cls._extract_success(info_venv[i]) for i in range(n_envs)],
            dtype=bool,
        )

    @staticmethod
    def _extract_exec_len(info, default_exec_len):
        if not isinstance(info, dict):
            return default_exec_len
        raw = info.get("exec_len", default_exec_len)
        if isinstance(raw, (list, tuple, np.ndarray)):
            if len(raw) == 0:
                return default_exec_len
            raw = raw[-1]
        return int(raw)

    @classmethod
    def extract_exec_lengths(cls, info_venv, n_envs, default_exec_len):
        return np.array(
            [cls._extract_exec_len(info_venv[i], default_exec_len) for i in range(n_envs)],
            dtype=np.float32,
        )

    def _resolve_checkpoint_path(self, checkpoint):
        if isinstance(checkpoint, int):
            return os.path.join(self.checkpoint_dir, f"state_{checkpoint}.pt")
        checkpoint = os.path.expanduser(str(checkpoint))
        if checkpoint.isdigit():
            return os.path.join(self.checkpoint_dir, f"state_{checkpoint}.pt")
        return checkpoint

    @staticmethod
    def _replay_buffer_to_tensor(buffer, dtype=None):
        if buffer is None or len(buffer) == 0:
            return None
        if isinstance(buffer, _NumpyRingBuffer):
            array = buffer.to_numpy()
        else:
            array = np.asarray(list(buffer))
        tensor = torch.from_numpy(array)
        return tensor.to(dtype=dtype) if dtype is not None else tensor

    @staticmethod
    def _tensor_to_replay_buffer(tensor_value, maxlen, dtype):
        return _NumpyRingBuffer(maxlen=maxlen, dtype=dtype, values=tensor_value)

    @staticmethod
    def _serialize_pending_episode_buffers(pending_episode_buffers):
        serialized = []
        for pending in pending_episode_buffers or []:
            serialized_episode = []
            for transition in pending:
                serialized_episode.append(
                    {
                        "obs": torch.from_numpy(np.asarray(transition["obs"], dtype=np.float32)),
                        "next_obs": torch.from_numpy(
                            np.asarray(transition["next_obs"], dtype=np.float32)
                        ),
                        "action": torch.from_numpy(
                            np.asarray(transition["action"], dtype=np.float32)
                        ),
                        "reward": float(transition["reward"]),
                        "terminated": bool(transition["terminated"]),
                    }
                )
            serialized.append(serialized_episode)
        return serialized

    @staticmethod
    def _deserialize_pending_episode_buffers(serialized, n_envs):
        pending_episode_buffers = [[] for _ in range(n_envs)]
        if serialized is None:
            return pending_episode_buffers
        for env_ind, pending in enumerate(serialized[:n_envs]):
            restored_episode = []
            for transition in pending:
                restored_episode.append(
                    {
                        "obs": transition["obs"].detach().cpu().numpy().astype(np.float32),
                        "next_obs": transition["next_obs"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32),
                        "action": transition["action"].detach().cpu().numpy().astype(np.float32),
                        "reward": float(transition["reward"]),
                        "terminated": bool(transition["terminated"]),
                    }
                )
            pending_episode_buffers[env_ind] = restored_episode
        return pending_episode_buffers

    @staticmethod
    def _pack_numpy_rng_state(state):
        name, keys, pos, has_gauss, cached_gaussian = state
        return {
            "name": name,
            "keys": torch.from_numpy(keys.astype(np.int64, copy=True)),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        }

    @staticmethod
    def _unpack_numpy_rng_state(state):
        return (
            state["name"],
            state["keys"].detach().cpu().numpy().astype(np.uint32),
            int(state["pos"]),
            int(state["has_gauss"]),
            float(state["cached_gaussian"]),
        )

    def _capture_rng_state(self):
        rng_state = {
            "python": random.getstate(),
            "numpy": self._pack_numpy_rng_state(np.random.get_state()),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
            device_index = self._cuda_device_index(self.device)
            rng_state["torch_cuda_training_device_index"] = device_index
            rng_state["torch_cuda_training_device"] = str(self.device)
            rng_state["torch_cuda_training_state"] = torch.cuda.get_rng_state(device_index)
        return rng_state

    @staticmethod
    def _cuda_device_index(device):
        device = torch.device(device)
        if device.type != "cuda":
            return None
        return torch.cuda.current_device() if device.index is None else device.index

    @staticmethod
    def _infer_checkpoint_source_device(loadpath):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(loadpath)),
            ".hydra",
            "config.yaml",
        )
        if not os.path.isfile(config_path):
            return None
        try:
            return str(OmegaConf.load(config_path).get("device", "")) or None
        except Exception as exc:
            log.warning("Could not read checkpoint device from %s: %s", config_path, exc)
            return None

    def _restore_rng_state(self, rng_state, source_device=None):
        if not rng_state:
            return
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(self._unpack_numpy_rng_state(rng_state["numpy"]))
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"].detach().cpu())
        if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
            target_index = self._cuda_device_index(self.device)
            training_state = rng_state.get("torch_cuda_training_state")
            if training_state is None and "torch_cuda" in rng_state:
                saved_states = rng_state["torch_cuda"]
                source_index = rng_state.get("torch_cuda_training_device_index")
                if source_index is None and source_device is not None:
                    try:
                        parsed_source = torch.device(source_device)
                        if parsed_source.type == "cuda":
                            source_index = parsed_source.index
                    except (RuntimeError, TypeError):
                        log.warning(
                            "Ignoring invalid checkpoint CUDA device %r",
                            source_device,
                        )
                if source_index is None and len(saved_states) == 1:
                    source_index = 0
                if source_index is not None and 0 <= int(source_index) < len(saved_states):
                    training_state = saved_states[int(source_index)]
                elif len(saved_states) > 0:
                    log.warning(
                        "Could not identify source CUDA RNG device %s; using state 0.",
                        source_device,
                    )
                    training_state = saved_states[0]
            if training_state is not None:
                torch.cuda.set_rng_state(training_state.detach().cpu(), target_index)

    def _capture_replay_state(self):
        if not self.save_replay_buffer or self._online_replay_buffers is None:
            return None
        buffers = self._online_replay_buffers
        return {
            "obs": self._replay_buffer_to_tensor(buffers["obs"], dtype=torch.float32),
            "next_obs": self._replay_buffer_to_tensor(
                buffers["next_obs"],
                dtype=torch.float32,
            ),
            "actions": self._replay_buffer_to_tensor(buffers["actions"], dtype=torch.float32),
            "rewards": self._replay_buffer_to_tensor(buffers["rewards"], dtype=torch.float32),
            "terminated": self._replay_buffer_to_tensor(
                buffers["terminated"],
                dtype=torch.float32,
            ),
            "mc_returns": (
                self._replay_buffer_to_tensor(buffers["mc_returns"], dtype=torch.float32)
                if self.store_mc_returns
                else None
            ),
            "pending_episode_buffers": self._serialize_pending_episode_buffers(
                self._pending_episode_buffers
            ),
            "buffer_size": int(self.buffer_size),
            "store_mc_returns": bool(self.store_mc_returns),
        }

    def _init_replay_from_loaded_state(self):
        replay_state = self._loaded_replay_state
        if replay_state is None:
            obs_buffer = _NumpyRingBuffer(self.buffer_size, np.float32)
            next_obs_buffer = _NumpyRingBuffer(self.buffer_size, np.float32)
            action_buffer = _NumpyRingBuffer(self.buffer_size, np.float32)
            reward_buffer = _NumpyRingBuffer(self.buffer_size, np.float64)
            terminated_buffer = _NumpyRingBuffer(self.buffer_size, np.bool_)
            mc_return_buffer = _NumpyRingBuffer(self.buffer_size, np.float64)
            pending_episode_buffers = [[] for _ in range(self.n_envs)]
        else:
            obs_buffer = self._tensor_to_replay_buffer(
                replay_state.get("obs"),
                self.buffer_size,
                np.float32,
            )
            next_obs_buffer = self._tensor_to_replay_buffer(
                replay_state.get("next_obs"),
                self.buffer_size,
                np.float32,
            )
            action_buffer = self._tensor_to_replay_buffer(
                replay_state.get("actions"),
                self.buffer_size,
                np.float32,
            )
            reward_buffer = self._tensor_to_replay_buffer(
                replay_state.get("rewards", torch.empty(0)),
                self.buffer_size,
                np.float64,
            )
            terminated_buffer = self._tensor_to_replay_buffer(
                replay_state.get("terminated", torch.empty(0)),
                self.buffer_size,
                np.bool_,
            )
            mc_return_buffer = _NumpyRingBuffer(self.buffer_size, np.float64)
            if self.store_mc_returns and replay_state.get("mc_returns") is not None:
                mc_return_buffer = self._tensor_to_replay_buffer(
                    replay_state["mc_returns"],
                    self.buffer_size,
                    np.float64,
                )
            serialized_pending = replay_state.get("pending_episode_buffers") or []
            self.discarded_resume_pending_transitions = sum(
                len(pending) for pending in serialized_pending
            )
            pending_episode_buffers = [[] for _ in range(self.n_envs)]
            if self.discarded_resume_pending_transitions:
                log.warning(
                    "Discarded %d incomplete PA-RL transitions on resume because "
                    "the environment state is reset and cannot continue those episodes.",
                    self.discarded_resume_pending_transitions,
                )
            log.info(
                "Restored PA-RL replay buffer with online=%d discarded_pending=%d",
                len(obs_buffer),
                self.discarded_resume_pending_transitions,
            )
            self._loaded_replay_state = None
        self._online_replay_buffers = {
            "obs": obs_buffer,
            "next_obs": next_obs_buffer,
            "actions": action_buffer,
            "rewards": reward_buffer,
            "terminated": terminated_buffer,
            "mc_returns": mc_return_buffer,
        }
        self._pending_episode_buffers = pending_episode_buffers
        return (
            obs_buffer,
            next_obs_buffer,
            action_buffer,
            reward_buffer,
            terminated_buffer,
            mc_return_buffer,
            pending_episode_buffers,
        )

    def save_model(self):
        data = {
            "itr": self.itr + 1,
            "saved_itr": self.itr,
            "model": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
            "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
            "cnt_train_step": self.cnt_train_step,
            "cnt_critic_update": self.cnt_critic_update,
            "rollout_success_history": list(getattr(self, "rollout_success_history", [])),
            "next_optimized_eval_step": self.next_optimized_eval_step,
            "next_model_save_step": self.next_model_save_step,
            "device": str(self.device),
            "rng_state": self._capture_rng_state(),
        }
        replay_state = self._capture_replay_state()
        if replay_state is not None:
            data["replay_state"] = replay_state
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
        torch.save(data, savepath)
        log.info(f"Saved PA-RL checkpoint to {savepath}")

    def load(self, checkpoint):
        loadpath = self._resolve_checkpoint_path(checkpoint)
        # Replay is CPU resident and can be hundreds of MB. Load the checkpoint
        # on CPU first; model/optimizer restoration moves only their own state to
        # the parameter device and avoids a replay-sized GPU allocation/copy.
        data = torch.load(loadpath, map_location="cpu", weights_only=True)

        self.itr = data.get("itr", self.itr)
        incompatible = self.model.load_state_dict(data["model"], strict=False)
        if incompatible.missing_keys:
            log.warning("Missing keys while loading %s: %s", loadpath, incompatible.missing_keys)
            if any(key.startswith("critic") for key in incompatible.missing_keys):
                log.warning("Checkpoint %s did not restore the full critic.", loadpath)
        if incompatible.unexpected_keys:
            log.warning(
                "Unexpected keys while loading %s: %s",
                loadpath,
                incompatible.unexpected_keys,
            )

        if "actor_optimizer" in data:
            self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        else:
            log.warning("Checkpoint %s has no actor optimizer state.", loadpath)
        if "critic_optimizer" in data:
            self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        else:
            log.warning("Checkpoint %s has no critic optimizer state.", loadpath)
        if "actor_lr_scheduler" in data:
            self.actor_lr_scheduler.load_state_dict(data["actor_lr_scheduler"])
        else:
            log.warning("Checkpoint %s has no actor scheduler state.", loadpath)
        if "critic_lr_scheduler" in data:
            self.critic_lr_scheduler.load_state_dict(data["critic_lr_scheduler"])
        else:
            log.warning("Checkpoint %s has no critic scheduler state.", loadpath)

        self.cnt_train_step = data.get("cnt_train_step", self.cnt_train_step)
        self.cnt_critic_update = data.get(
            "cnt_critic_update",
            self.cnt_critic_update,
        )
        self.rollout_success_history = deque(
            [float(success) for success in data.get("rollout_success_history", [])][-200:],
            maxlen=200,
        )
        if self.optimized_eval_freq_env_steps > 0:
            self.next_optimized_eval_step = data.get("next_optimized_eval_step", None)
            if self.next_optimized_eval_step is None:
                self.next_optimized_eval_step = (
                    0
                    if self.optimized_eval_at_start and self.cnt_train_step == 0
                    else self.optimized_eval_freq_env_steps
                )
            while self.next_optimized_eval_step <= self.cnt_train_step:
                self.next_optimized_eval_step += self.optimized_eval_freq_env_steps
        else:
            self.next_optimized_eval_step = None
        if self.save_model_freq_env_steps > 0:
            self.next_model_save_step = data.get("next_model_save_step", None)
            if self.next_model_save_step is None:
                self.next_model_save_step = 0
            while self.next_model_save_step <= self.cnt_train_step:
                self.next_model_save_step += self.save_model_freq_env_steps
        else:
            self.next_model_save_step = None
        self._loaded_replay_state = data.get("replay_state", None)
        if self.save_replay_buffer and "replay_state" not in data:
            log.warning(
                "Checkpoint %s has no replay_state. Phase B resume will "
                "start with an empty online replay buffer, so it is not "
                "equivalent to a single uninterrupted Phase A+B run.",
                loadpath,
            )
        source_device = data.get("device")
        if source_device is None:
            source_device = self._infer_checkpoint_source_device(loadpath)
        self._restore_rng_state(
            data.get("rng_state", None),
            source_device=source_device,
        )
        log.info(
            "Loaded PA-RL checkpoint from %s at itr=%s, env_step=%s, critic_update=%s",
            loadpath,
            self.itr,
            self.cnt_train_step,
            self.cnt_critic_update,
        )

    def _run_optimized_action_success_eval(self):
        n_episodes = int(self.optimized_eval_n_episodes)
        if n_episodes <= 0:
            return {}

        chunks_per_episode = max(
            1,
            int(np.ceil(self.max_episode_steps / max(1, self.act_steps))),
        )
        episodes_per_env = int(np.ceil(n_episodes / max(1, self.n_envs)))
        max_steps = self.optimized_eval_max_steps
        if max_steps is None:
            max_steps = chunks_per_episode * (episodes_per_env + 1)

        was_training = self.model.training
        self.model.eval()
        prev_obs_venv = self.reset_env_all(options_venv=[{} for _ in range(self.n_envs)])
        running_success = np.zeros(self.n_envs, dtype=bool)
        completed_successes = []
        eval_env_steps = 0
        action_metrics_by_key = {}

        for _ in range(max_steps):
            with torch.no_grad():
                cond = {"state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)}
                action_tensor, action_metrics = self._select_action_chunks(
                    cond=cond,
                    eval_mode=True,
                    action_policy_override="optimized",
                )
                action_venv = action_tensor.cpu().numpy()
            for key, value in action_metrics.items():
                scalar = self._as_scalar(value)
                if scalar is not None:
                    action_metrics_by_key.setdefault(key, []).append(scalar)

            obs_venv, _, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
            done_venv = terminated_venv | truncated_venv
            exec_len_venv = self.extract_exec_lengths(
                info_venv,
                self.n_envs,
                default_exec_len=self.act_steps,
            )
            eval_env_steps += int(np.sum(exec_len_venv))
            running_success |= self.extract_success_flags(info_venv, self.n_envs)
            for env_ind, done in enumerate(done_venv):
                if done:
                    completed_successes.append(float(running_success[env_ind]))
                    running_success[env_ind] = False
                    if len(completed_successes) >= n_episodes:
                        break
            prev_obs_venv = obs_venv
            if len(completed_successes) >= n_episodes:
                break

        if was_training:
            self.model.train()

        successes = completed_successes[:n_episodes]
        success_rate = float(np.mean(successes)) if len(successes) > 0 else 0.0
        metrics = {
            "eval_optimized/success_rate": success_rate,
            "eval_optimized/num_episode": len(successes),
            "eval_optimized/env_steps": eval_env_steps,
            "eval_optimized/max_steps": max_steps,
            "eval_optimized/target_num_episode": n_episodes,
            "eval_optimized/critic_update": self.cnt_critic_update,
            "eval_optimized/train_env_step": self.cnt_train_step,
        }
        policy_metrics = self._average_scalar_metrics(action_metrics_by_key)
        for key, value in policy_metrics.items():
            metrics[f"eval_optimized/{key}"] = value
        log.info(
            "optimized eval: step %s | critic update %s | success %.4f | episodes %s/%s",
            self.cnt_train_step,
            self.cnt_critic_update,
            success_rate,
            len(successes),
            n_episodes,
        )
        self.log_tensorboard(metrics, self.cnt_train_step)
        if self.use_wandb:
            wandb.log(metrics, step=self.itr, commit=False)
        return metrics

    def run(self):

        # make a FIFO replay buffer for obs, action, and reward
        (
            obs_buffer,
            next_obs_buffer,
            action_buffer,
            reward_buffer,
            terminated_buffer,
            mc_return_buffer,
            pending_episode_buffers,
        ) = self._init_replay_from_loaded_state()

        def append_transition_to_replay(transition, mc_return=None):
            obs_buffer.append(transition["obs"])
            next_obs_buffer.append(transition["next_obs"])
            action_buffer.append(transition["action"])
            reward_buffer.append(float(transition["reward"]))
            terminated_buffer.append(bool(transition["terminated"]))
            if self.store_mc_returns:
                mc_return_buffer.append(float(mc_return))

        def flush_pending_episode(env_ind):
            pending = pending_episode_buffers[env_ind]
            if len(pending) == 0:
                return 0
            rewards = np.array([transition["reward"] for transition in pending])
            masks = np.array([1.0 - float(transition["terminated"]) for transition in pending])
            mc_returns = self.calc_return_to_go(
                rewards,
                masks,
                self.gamma,
                push_failed_to_min=self.calql_push_failed_to_min,
                min_reward=self.calql_min_reward,
            )
            for transition, mc_return in zip(pending, mc_returns):
                append_transition_to_replay(transition, mc_return=mc_return)
            pending.clear()
            return len(mc_returns)

        # Start training loop
        timer = Timer()
        run_results = []
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        needs_env_reset = True
        while self.itr < self.n_train_itr and (
            self.max_env_steps is None or self.cnt_train_step < self.max_env_steps
        ):
            if self.itr == 0 and self._optimized_eval_due():
                if self.optimized_eval_async:
                    self._save_async_optimized_eval_snapshot()
                else:
                    self._run_optimized_action_success_eval()
                    needs_env_reset = True
                self._advance_next_optimized_eval_step()

            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval - all envs restart
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.store_mc_returns and (self.reset_at_iteration or eval_mode or last_itr_eval):
                for env_ind in range(self.n_envs):
                    flush_pending_episode(env_ind)
            if (
                needs_env_reset
                or self.itr == 0
                or self.reset_at_iteration
                or eval_mode
                or last_itr_eval
            ):
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
                needs_env_reset = False
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            raw_reward_trajs = np.zeros((self.n_steps, self.n_envs))
            success_trajs = np.zeros((self.n_steps, self.n_envs), dtype=bool)
            exec_len_trajs = np.zeros((self.n_steps, self.n_envs))
            action_metrics_by_key = {}

            # Collect a set of trajectories from env
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Select action
                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                    }
                    action_tensor, action_metrics = self._select_action_chunks(
                        cond,
                        eval_mode=eval_mode,
                    )
                    action_venv = action_tensor.cpu().numpy()
                    for key, value in action_metrics.items():
                        scalar = self._as_scalar(value)
                        if scalar is not None:
                            action_metrics_by_key.setdefault(key, []).append(scalar)

                # Apply multi-step action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(
                    action_venv
                )
                done_venv = terminated_venv | truncated_venv
                raw_reward_venv = np.asarray(reward_venv)
                exec_len_venv = self.extract_exec_lengths(
                    info_venv,
                    self.n_envs,
                    default_exec_len=self.act_steps,
                )
                reward_venv = self.offset_rewards(
                    raw_reward_venv,
                    self.reward_offset,
                    exec_len_venv,
                )
                success_venv = self.extract_success_flags(info_venv, self.n_envs)
                raw_reward_trajs[step] = raw_reward_venv
                reward_trajs[step] = reward_venv
                success_trajs[step] = success_venv
                exec_len_trajs[step] = exec_len_venv
                firsts_trajs[step + 1] = done_venv

                # add to buffer
                if not eval_mode:
                    for i in range(self.n_envs):
                        if truncated_venv[i]:  # truncated
                            next_obs_i = info_venv[i]["final_obs"]["state"]
                        else:
                            next_obs_i = obs_venv["state"][i]
                        transition = {
                            "obs": np.array(prev_obs_venv["state"][i], copy=True),
                            "next_obs": np.array(next_obs_i, copy=True),
                            "action": np.array(action_venv[i], copy=True),
                            "reward": float(reward_venv[i] * self.scale_reward_factor),
                            "terminated": bool(terminated_venv[i]),
                        }
                        if self.store_mc_returns:
                            pending_episode_buffers[i].append(transition)
                            if done_venv[i]:
                                flush_pending_episode(i)
                        else:
                            append_transition_to_replay(transition)

                # update for next step
                prev_obs_venv = obs_venv

                # Count actual low-level env steps. This matters once
                # terminate_on_success can stop inside an action chunk.
                if not eval_mode:
                    self.cnt_train_step += int(np.sum(exec_len_venv))

            # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                raw_reward_trajs_split = [
                    raw_reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                success_trajs_split = [
                    success_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_successes = np.array(
                    [np.any(success_traj) for success_traj in success_trajs_split],
                    dtype=np.float32,
                )
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                raw_episode_reward = np.array(
                    [np.sum(raw_reward_traj) for raw_reward_traj in raw_reward_trajs_split]
                )
                episode_best_reward = np.array(
                    [np.max(reward_traj) for reward_traj in reward_trajs_split]
                )
                raw_episode_best_reward = np.array(
                    [
                        np.max(raw_reward_traj) / self.act_steps
                        for raw_reward_traj in raw_reward_trajs_split
                    ]
                )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                avg_raw_episode_reward = np.mean(raw_episode_reward)
                avg_raw_best_reward = np.mean(raw_episode_best_reward)
                success_rate = float(np.mean(episode_successes))
            else:
                episode_successes = np.array([], dtype=np.float32)
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                avg_raw_episode_reward = 0
                avg_raw_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")
            policy_metrics_this_itr = self._average_scalar_metrics(action_metrics_by_key)
            rollout_success_rate_itr = success_rate
            if not eval_mode:
                self._record_rollout_successes(episode_successes)
            recent_success_metrics = self._recent_rollout_success_metrics()
            rollout_success_rate_recent20 = recent_success_metrics["success_rate_recent20"]
            rollout_success_rate_recent100 = recent_success_metrics["success_rate_recent100"]
            rollout_success_rate_recent200 = recent_success_metrics["success_rate_recent200"]

            loss_actor = None
            loss_critic = None
            actor_updates_this_itr = 0
            actor_metrics_this_itr = {}
            critic_metrics_this_itr = {}
            offline_samples_this_itr = 0
            online_samples_this_itr = 0

            # Update models
            if not eval_mode:
                num_batch = int(self.n_steps * self.n_envs / self.batch_size * self.replay_ratio)
                # only worth converting first with parallel envs - large number of updates below
                online_replay_size = len(obs_buffer)
                offline_replay_size = (
                    0
                    if self.offline_replay is None or self.offline_batch_ratio <= 0
                    else len(self.offline_replay["obs"])
                )
                has_replay_batch = num_batch > 0 and (
                    online_replay_size > 0 or offline_replay_size > 0
                )
                if not has_replay_batch:
                    log.info(
                        "[WARNING] Replay has online=%s offline=%s samples, skipping critic update.",
                        online_replay_size,
                        offline_replay_size,
                    )

                def sample_training_batch():
                    def merge_parts(parts):
                        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

                    target_offline = 0
                    if self.offline_replay is not None and self.offline_batch_ratio > 0:
                        target_offline = int(round(self.batch_size * self.offline_batch_ratio))
                    target_offline = min(target_offline, self.batch_size)
                    target_online = self.batch_size - target_offline
                    if online_replay_size == 0:
                        target_offline = self.batch_size
                        target_online = 0
                    if offline_replay_size == 0:
                        target_offline = 0
                        target_online = self.batch_size
                    if target_offline == 0 and target_online == 0:
                        raise ValueError("No replay samples available")

                    obs_parts = []
                    next_obs_parts = []
                    action_parts = []
                    reward_parts = []
                    terminated_parts = []
                    mc_return_parts = []
                    if target_offline > 0:
                        offline_inds = np.random.choice(
                            offline_replay_size,
                            target_offline,
                            replace=True,
                        )
                        offline = self.offline_replay
                        obs_parts.append(offline["obs"][offline_inds])
                        next_obs_parts.append(offline["next_obs"][offline_inds])
                        action_parts.append(offline["actions"][offline_inds])
                        reward_parts.append(offline["rewards"][offline_inds])
                        terminated_parts.append(offline["terminated"][offline_inds])
                        if self.store_mc_returns:
                            mc_return_parts.append(offline["mc_returns"][offline_inds])
                    if target_online > 0:
                        online_inds = np.random.choice(
                            online_replay_size,
                            target_online,
                            replace=True,
                        )
                        obs_parts.append(obs_buffer.take(online_inds))
                        next_obs_parts.append(next_obs_buffer.take(online_inds))
                        action_parts.append(action_buffer.take(online_inds))
                        reward_parts.append(reward_buffer.take(online_inds))
                        terminated_parts.append(terminated_buffer.take(online_inds))
                        if self.store_mc_returns:
                            mc_return_parts.append(mc_return_buffer.take(online_inds))
                    return (
                        merge_parts(obs_parts),
                        merge_parts(next_obs_parts),
                        merge_parts(action_parts),
                        merge_parts(reward_parts),
                        merge_parts(terminated_parts),
                        (merge_parts(mc_return_parts) if self.store_mc_returns else None),
                        target_offline,
                        target_online,
                    )

                # Critic learning
                num_updates = num_batch if has_replay_batch else 0
                for update_ind in range(num_updates):
                    (
                        obs_sample,
                        next_obs_sample,
                        action_sample,
                        reward_sample,
                        terminated_sample,
                        mc_return_sample,
                        num_offline_sampled,
                        num_online_sampled,
                    ) = sample_training_batch()
                    offline_samples_this_itr += num_offline_sampled
                    online_samples_this_itr += num_online_sampled
                    obs_b = torch.from_numpy(obs_sample).float().to(self.device)
                    next_obs_b = torch.from_numpy(next_obs_sample).float().to(self.device)
                    actions_b = torch.from_numpy(action_sample).float().to(self.device)
                    rewards_b = torch.from_numpy(reward_sample).float().to(self.device)
                    terminated_b = torch.from_numpy(terminated_sample).float().to(self.device)
                    mc_returns_b = (
                        torch.from_numpy(mc_return_sample).float().to(self.device)
                        if self.store_mc_returns
                        else None
                    )

                    # Only the final update's diagnostics are logged. Avoiding
                    # GPU scalar synchronization on earlier updates leaves the
                    # optimization sequence unchanged.
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    loss_critic = self.model.loss_critic(
                        {"state": obs_b},
                        {"state": next_obs_b},
                        actions_b,
                        rewards_b,
                        terminated_b,
                        self.gamma,
                        mc_returns=mc_returns_b,
                        collect_metrics=update_ind == num_updates - 1,
                    )
                    loss_critic.backward()
                    self.critic_optimizer.step()
                    critic_metrics_this_itr = getattr(
                        self.model,
                        "last_critic_metrics",
                        {},
                    )

                    # Optional PA-RL Phase B: distill actor from Q-reranked
                    # candidate actions. The start update lets Phase A and
                    # Phase B live in one run without a checkpoint handoff.
                    actor_distill_active = self._actor_distill_active()
                    should_update_actor = (
                        actor_distill_active
                        and self.cnt_critic_update % self.actor_update_freq == 0
                    )
                    if should_update_actor:
                        self.actor_optimizer.zero_grad(set_to_none=True)
                        loss_actor, actor_metrics_this_itr = self.model.loss_actor_distill(
                            {"state": obs_b},
                            replay_actions=actions_b,
                            num_base_policy_actions=self.num_base_policy_actions,
                            num_actions_to_keep=self.num_actions_to_keep,
                            act_steps=self.act_steps,
                            distill_argmax=self.distill_argmax,
                            softmax_temperature=self.softmax_temperature,
                            consider_replay_action=self.consider_replay_action,
                            use_min_q=self.use_min_q_for_rerank,
                            local_optimization_steps=(self.distill_local_optimization_steps),
                            local_optimization_step_size=(
                                self.distill_local_optimization_step_size
                            ),
                            optimize_critic_ensemble_min=(self.optimize_critic_ensemble_min),
                            use_target_critic=(self.use_target_critic_for_optimization),
                        )
                        loss_actor.backward()
                        if self.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.actor.parameters(), self.max_grad_norm
                            )
                        self.actor_optimizer.step()
                        actor_updates_this_itr += 1

                    # update target
                    self.model.update_target_critic(self.target_ema_rate)
                    self.cnt_critic_update += 1

            # Update lr
            if self.actor_distill_enabled and actor_updates_this_itr > 0:
                self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

            # Queue asynchronous evaluation before the full checkpoint so a
            # resumed run cannot advance past an unsaved evaluation boundary.
            if (not eval_mode) and self.optimized_eval_async and self._optimized_eval_due():
                self._save_async_optimized_eval_snapshot()
                self._advance_next_optimized_eval_step()

            # Save model
            if self._model_save_due():
                self.save_model()
                self._advance_next_model_save_step()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": self.cnt_train_step,
                    "critic_update": self.cnt_critic_update,
                }
            )
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | raw reward {avg_raw_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        eval_log_data = {
                            "success rate - eval": success_rate,
                            "avg episode reward - eval": avg_episode_reward,
                            "avg best reward - eval": avg_best_reward,
                            "avg raw episode reward - eval": avg_raw_episode_reward,
                            "avg raw best reward - eval": avg_raw_best_reward,
                            "num episode - eval": num_episode_finished,
                        }
                        eval_log_data.update(policy_metrics_this_itr)
                        wandb.log(eval_log_data, step=self.itr, commit=False)
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                    run_results[-1]["eval_raw_episode_reward"] = avg_raw_episode_reward
                    run_results[-1]["eval_raw_best_reward"] = avg_raw_best_reward
                    run_results[-1].update(policy_metrics_this_itr)
                    eval_tb_data = {
                        "eval/success_rate": success_rate,
                        "eval/episode_reward": avg_episode_reward,
                        "eval/best_reward": avg_best_reward,
                        "eval/raw_episode_reward": avg_raw_episode_reward,
                        "eval/raw_best_reward": avg_raw_best_reward,
                        "eval/num_episode": num_episode_finished,
                        "time/itr": self.itr,
                        "time/env_steps": self.cnt_train_step,
                    }
                    eval_tb_data.update(policy_metrics_this_itr)
                    self.log_tensorboard(
                        eval_tb_data,
                        self.cnt_train_step,
                    )
                else:
                    loss_actor_value = loss_actor.detach().item() if loss_actor is not None else 0.0
                    loss_critic_value = (
                        loss_critic.detach().item() if loss_critic is not None else 0.0
                    )
                    log.info(
                        f"{self.itr}: step {self.cnt_train_step:8d} | actor distill {int(self.actor_distill_enabled)} | loss actor {loss_actor_value:8.4f} | loss critic {loss_critic_value:8.4f} | reward {avg_episode_reward:8.4f} | raw reward {avg_raw_episode_reward:8.4f} | success {rollout_success_rate_recent200:8.4f} | itr success {rollout_success_rate_itr:8.4f} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        log_data = {
                            "total env step": self.cnt_train_step,
                            "loss - actor": loss_actor_value,
                            "loss - critic": loss_critic_value,
                            "actor distill enabled": int(self.actor_distill_enabled),
                            "actor updates": actor_updates_this_itr,
                            "critic warmup updates": self.critic_warmup_updates,
                            "critic update": self.cnt_critic_update,
                            "avg episode reward - train": avg_episode_reward,
                            "avg raw episode reward - train": avg_raw_episode_reward,
                            "avg best reward - train": avg_best_reward,
                            "avg raw best reward - train": avg_raw_best_reward,
                            "success rate - train": rollout_success_rate_recent200,
                            "success rate recent20 - train": rollout_success_rate_recent20,
                            "success rate recent100 - train": rollout_success_rate_recent100,
                            "success rate recent200 - train": rollout_success_rate_recent200,
                            "success rate itr - train": rollout_success_rate_itr,
                            "reward offset": self.reward_offset,
                            "num episode - train": num_episode_finished,
                        }
                        log_data.update(actor_metrics_this_itr)
                        log_data.update(critic_metrics_this_itr)
                        log_data.update(policy_metrics_this_itr)
                        wandb.log(log_data, step=self.itr, commit=True)
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                    run_results[-1]["train_raw_episode_reward"] = avg_raw_episode_reward
                    run_results[-1]["train_best_reward"] = avg_best_reward
                    run_results[-1]["train_raw_best_reward"] = avg_raw_best_reward
                    run_results[-1]["train_success_rate"] = rollout_success_rate_recent200
                    run_results[-1]["train_success_rate_itr"] = rollout_success_rate_itr
                    run_results[-1]["train_success_rate_recent20"] = rollout_success_rate_recent20
                    run_results[-1]["train_success_rate_recent100"] = rollout_success_rate_recent100
                    run_results[-1]["train_success_rate_recent200"] = rollout_success_rate_recent200
                    run_results[-1]["loss_actor"] = loss_actor_value
                    run_results[-1]["loss_critic"] = loss_critic_value
                    run_results[-1]["actor_updates"] = actor_updates_this_itr
                    run_results[-1]["actor_distill_enabled"] = self.actor_distill_enabled
                    run_results[-1]["actor_distill_active"] = self._actor_distill_active()
                    run_results[-1]["actor_distill_start_updates"] = (
                        self.actor_distill_start_updates
                    )
                    run_results[-1]["distill_local_optimization_steps"] = (
                        self.distill_local_optimization_steps
                    )
                    run_results[-1]["offline_batch_ratio"] = self.offline_batch_ratio
                    run_results[-1]["offline_replay_size"] = (
                        0 if self.offline_replay is None else len(self.offline_replay["obs"])
                    )
                    run_results[-1]["offline_samples"] = offline_samples_this_itr
                    run_results[-1]["online_samples"] = online_samples_this_itr
                    run_results[-1].update(actor_metrics_this_itr)
                    run_results[-1].update(critic_metrics_this_itr)
                    run_results[-1].update(policy_metrics_this_itr)
                    tb_data = {
                        "time/itr": self.itr,
                        "time/env_steps": self.cnt_train_step,
                        "time/iteration_wall_sec": time,
                        "train/loss_actor": loss_actor_value,
                        "train/loss_critic": loss_critic_value,
                        "train/actor_distill_enabled": int(self.actor_distill_enabled),
                        "train/actor_distill_active": int(self._actor_distill_active()),
                        "train/actor_updates": actor_updates_this_itr,
                        "train/critic_update": self.cnt_critic_update,
                        "train/critic_warmup_updates": self.critic_warmup_updates,
                        "train/actor_distill_start_updates": (self.actor_distill_start_updates),
                        "train/distill_local_optimization_steps": (
                            self.distill_local_optimization_steps
                        ),
                        "rollout/episode_reward": avg_episode_reward,
                        "rollout/raw_episode_reward": avg_raw_episode_reward,
                        "rollout/best_reward": avg_best_reward,
                        "rollout/raw_best_reward": avg_raw_best_reward,
                        "rollout/success_rate": rollout_success_rate_recent200,
                        "rollout/success_rate_recent20": rollout_success_rate_recent20,
                        "rollout/success_rate_recent100": rollout_success_rate_recent100,
                        "rollout/success_rate_recent200": rollout_success_rate_recent200,
                        "rollout/success_rate_itr": rollout_success_rate_itr,
                        "rollout/num_episode": num_episode_finished,
                        "rollout/recent_success_history_size": recent_success_metrics[
                            "success_history_size"
                        ],
                        "replay/size": len(obs_buffer),
                        "replay/offline_size": (
                            0 if self.offline_replay is None else len(self.offline_replay["obs"])
                        ),
                        "replay/offline_batch_ratio": self.offline_batch_ratio,
                        "replay/offline_samples": offline_samples_this_itr,
                        "replay/online_samples": online_samples_this_itr,
                        "replay/pending": sum(len(pending) for pending in pending_episode_buffers),
                        "reward/offset": self.reward_offset,
                    }
                    tb_data.update(actor_metrics_this_itr)
                    tb_data.update(critic_metrics_this_itr)
                    tb_data.update(policy_metrics_this_itr)
                    self.log_tensorboard(tb_data, self.cnt_train_step)
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            if (not eval_mode) and (not self.optimized_eval_async) and self._optimized_eval_due():
                if self.store_mc_returns:
                    for env_ind in range(self.n_envs):
                        flush_pending_episode(env_ind)
                optimized_eval_metrics = self._run_optimized_action_success_eval()
                if len(run_results) > 0:
                    run_results[-1].update(optimized_eval_metrics)
                    with open(self.result_path, "wb") as f:
                        pickle.dump(run_results, f)
                self._advance_next_optimized_eval_step()
                needs_env_reset = True
            self.itr += 1
        if self.max_env_steps is not None and self.cnt_train_step >= self.max_env_steps:
            log.info(
                "Reached train.max_env_steps=%s at env_step=%s, stopping training loop.",
                self.max_env_steps,
                self.cnt_train_step,
            )
        if self.tb_writer is not None:
            self.tb_writer.close()
