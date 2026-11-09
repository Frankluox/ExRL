import hydra
import abc
import json
import os
import sys
import types
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None
from stable_baselines3.common.callbacks import BaseCallback


def _extract_success(info):
    raw = info.get("is_success", None)
    if raw is None and isinstance(info.get("episode"), dict):
        raw = info["episode"].get("is_success", None)
    if raw is None:
        return False
    if isinstance(raw, dict):
        return bool(raw.get("task", False))
    if isinstance(raw, (list, tuple, np.ndarray)):
        if len(raw) == 0:
            return False
        raw = raw[-1]
        if isinstance(raw, dict):
            return bool(raw.get("task", False))
    return bool(raw)


class DPPOBasePolicyWrapper:
    def __init__(self, base_policy):
        self.base_policy = base_policy

    def __call__(self, obs, initial_noise, return_numpy=True):
        cond = {
            "state": obs,
            "noise_action": initial_noise,
        }
        with torch.no_grad():
            samples = self.base_policy(cond=cond, deterministic=True)
        diffused_actions = samples.trajectories.detach()
        if return_numpy:
            diffused_actions = diffused_actions.cpu().numpy()
        return diffused_actions


def _cfg_get(cfg, key, default=None):
    try:
        return getattr(cfg, key)
    except Exception:
        return default


def _parse_csv_or_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [part.strip() for part in text.split(",") if part.strip()]
    try:
        return [str(item).strip() for item in value if str(item).strip()]
    except TypeError:
        text = str(value).strip()
        return [text] if text else []


def _read_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _resolve_act_checkpoint_dir(path_or_dir):
    raw = Path(str(path_or_dir)).expanduser()
    base = raw.resolve() if raw.is_absolute() else (Path.cwd() / raw).resolve()
    if (base / "model.safetensors").exists() and (base / "config.json").exists():
        return base
    final_dir = base / "final"
    if (final_dir / "model.safetensors").exists() and (final_dir / "config.json").exists():
        return final_dir
    checkpoints_dir = base / "checkpoints"
    if checkpoints_dir.exists():
        candidates = [
            item
            for item in checkpoints_dir.iterdir()
            if item.is_dir()
            and (item / "model.safetensors").exists()
            and (item / "config.json").exists()
        ]
        if candidates:
            return sorted(candidates)[-1]
    raise FileNotFoundError(f"unable to resolve ACT checkpoint dir from: {path_or_dir}")


def _infer_act_train_output_dir(ckpt_dir):
    if ckpt_dir.parent.name == "final":
        return ckpt_dir.parent.parent
    if ckpt_dir.parent.parent.name == "checkpoints":
        return ckpt_dir.parent.parent.parent
    return ckpt_dir.parent


def _load_act_metadata(ckpt_dir):
    metadata = {}
    output_dir = _infer_act_train_output_dir(ckpt_dir)
    for candidate in (
        output_dir / "train_config.json",
        output_dir / "final" / "run_summary.json",
        ckpt_dir / "run_summary.json",
    ):
        metadata.update(_read_json(candidate))
    return metadata


def _ensure_lerobot_importable(act_lerobot_path=None):
    candidates = []
    if act_lerobot_path:
        candidates.append(Path(str(act_lerobot_path)).expanduser())
    here = Path(__file__).resolve().parent
    candidates.extend(
        [
            here / "third_party" / "lerobot" / "src",
            Path("assets/src"),
            Path("assets/src"),
            Path("assets/src"),
            Path("assets/src"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            candidate_str = str(candidate.resolve())
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate_str
    raise FileNotFoundError(
        "ACT base policy requires LeRobot sources. Set cfg.act_lerobot_path to a lerobot/src directory."
    )


def _install_lerobot_py39_optimizer_stub():
    if sys.version_info >= (3, 10):
        return
    module_name = "lerobot.optim.optimizers"
    if module_name in sys.modules:
        return

    @dataclass
    class OptimizerConfig(abc.ABC):
        lr: float
        weight_decay: float
        grad_clip_norm: float

        @abc.abstractmethod
        def build(self, params):
            raise NotImplementedError

    @dataclass
    class AdamWConfig(OptimizerConfig):
        lr: float = 1e-3
        betas: tuple = (0.9, 0.999)
        eps: float = 1e-8
        weight_decay: float = 1e-2
        grad_clip_norm: float = 10.0

        def build(self, params):
            return torch.optim.AdamW(
                params,
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )

    stub = types.ModuleType(module_name)
    stub.OptimizerConfig = OptimizerConfig
    stub.AdamWConfig = AdamWConfig
    sys.modules[module_name] = stub


def _infer_lowdim_dims_from_stats(stats_path, lowdim_keys):
    if not stats_path:
        return {}
    if not os.path.exists(str(stats_path)):
        return {}
    stats = np.load(str(stats_path))
    dims = {}
    for key in lowdim_keys:
        mean_key = f"{key}_mean"
        if mean_key in stats:
            dims[key] = int(np.asarray(stats[mean_key]).reshape(-1).shape[0])
    return dims


_ACT_DEFAULT_STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
_ACT_DEFAULT_ENV_KEYS = ("object",)
_ACT_TASK_OBS_PRESETS = {
    "square": (_ACT_DEFAULT_STATE_KEYS, _ACT_DEFAULT_ENV_KEYS),
    "can": (_ACT_DEFAULT_STATE_KEYS, _ACT_DEFAULT_ENV_KEYS),
    "lift": (_ACT_DEFAULT_STATE_KEYS, _ACT_DEFAULT_ENV_KEYS),
    "transport": (
        (
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot1_eef_pos",
            "robot1_eef_quat",
            "robot1_gripper_qpos",
        ),
        _ACT_DEFAULT_ENV_KEYS,
    ),
}


def _act_obs_key_preset(task):
    task_name = str(task or "lift").strip().lower()
    return _ACT_TASK_OBS_PRESETS.get(task_name, (_ACT_DEFAULT_STATE_KEYS, _ACT_DEFAULT_ENV_KEYS))


class ACTBasePolicyWrapper:
    def __init__(self, cfg):
        lerobot_src = _ensure_lerobot_importable(_cfg_get(cfg, "act_lerobot_path", None))
        _install_lerobot_py39_optimizer_stub()
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            policy_action_to_transition,
            transition_to_policy_action,
        )
        from lerobot.utils.constants import (
            OBS_ENV_STATE,
            OBS_STATE,
            POLICY_POSTPROCESSOR_DEFAULT_NAME,
            POLICY_PREPROCESSOR_DEFAULT_NAME,
        )

        self.device = torch.device(str(cfg.device))
        self.ckpt_dir = _resolve_act_checkpoint_dir(cfg.base_policy_path)
        self.metadata = _load_act_metadata(self.ckpt_dir)
        self.obs_state_key = OBS_STATE
        self.obs_env_state_key = OBS_ENV_STATE
        self.task = (
            str(_cfg_get(cfg, "env_name", None) or self.metadata.get("task") or "lift")
            .strip()
            .lower()
        )
        preset_state_keys, preset_env_keys = _act_obs_key_preset(self.task)
        cfg_act = PreTrainedConfig.from_pretrained(self.ckpt_dir)
        if not isinstance(cfg_act, ACTConfig):
            raise TypeError(
                f"base_policy_type=act expected ACT config, got {type(cfg_act).__name__}"
            )
        cfg_act.device = str(self.device)
        if getattr(cfg_act, "temporal_ensemble_coeff", None) is not None:
            raise ValueError(
                "Residual ACT base uses direct chunk prediction and does not support temporal ensemble."
            )

        self.policy = (
            ACTPolicy.from_pretrained(self.ckpt_dir, config=cfg_act, strict=True)
            .to(self.device)
            .eval()
        )
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            self.ckpt_dir,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            overrides={"device_processor": {"device": str(self.device)}},
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            self.ckpt_dir,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

        self.state_obs_keys = _parse_csv_or_list(_cfg_get(cfg, "act_state_obs_keys", None))
        if not self.state_obs_keys:
            self.state_obs_keys = _parse_csv_or_list(self.metadata.get("state_obs_keys")) or list(
                preset_state_keys
            )
        self.env_obs_keys = _parse_csv_or_list(_cfg_get(cfg, "act_env_obs_keys", None))
        if not self.env_obs_keys:
            self.env_obs_keys = _parse_csv_or_list(self.metadata.get("env_obs_keys")) or list(
                preset_env_keys
            )
        self.lowdim_keys = _parse_csv_or_list(_cfg_get(cfg, "act_lowdim_keys", None))
        if not self.lowdim_keys:
            self.lowdim_keys = (
                _parse_csv_or_list(self.metadata.get("lowdim_keys"))
                or _parse_csv_or_list(cfg.env.wrappers.robomimic_lowdim.low_dim_keys)
                or list(dict.fromkeys(self.state_obs_keys + self.env_obs_keys))
            )

        lowdim_dims_cfg = _cfg_get(cfg, "act_lowdim_dims", None)
        lowdim_dims_override = dict(lowdim_dims_cfg) if lowdim_dims_cfg is not None else {}
        if lowdim_dims_override:
            self.lowdim_dims = {str(k): int(v) for k, v in lowdim_dims_override.items()}
        else:
            self.lowdim_dims = {
                str(k): int(v) for k, v in dict(self.metadata.get("lowdim_dims") or {}).items()
            }
        if not self.lowdim_dims:
            self.lowdim_dims = _infer_lowdim_dims_from_stats(
                _cfg_get(cfg, "action_stats_path", None) or self.metadata.get("stats_npz"),
                self.lowdim_keys,
            )
        missing_dims = [key for key in self.lowdim_keys if key not in self.lowdim_dims]
        if missing_dims:
            raise ValueError(f"ACT base policy is missing lowdim dims for keys: {missing_dims}")

        self.observation_space_kind = (
            str(
                _cfg_get(cfg, "act_observation_space_kind", None)
                or self.metadata.get("observation_space_kind")
                or "dsrl_normalized"
            )
            .strip()
            .lower()
        )
        self.action_space_kind = (
            str(
                _cfg_get(cfg, "act_action_space_kind", None)
                or self.metadata.get("action_space_kind")
                or "dsrl_normalized"
            )
            .strip()
            .lower()
        )
        self.normalization = None
        if self.observation_space_kind == "raw_lowdim" or self.action_space_kind == "raw_action":
            normalization = np.load(str(cfg.normalization_path))
            self.normalization = {
                "obs_min": torch.as_tensor(
                    normalization["obs_min"], device=self.device, dtype=torch.float32
                ).reshape(1, -1),
                "obs_max": torch.as_tensor(
                    normalization["obs_max"], device=self.device, dtype=torch.float32
                ).reshape(1, -1),
                "action_min": torch.as_tensor(
                    normalization["action_min"], device=self.device, dtype=torch.float32
                ).reshape(1, 1, -1),
                "action_max": torch.as_tensor(
                    normalization["action_max"], device=self.device, dtype=torch.float32
                ).reshape(1, 1, -1),
            }
            self.normalization["obs_span"] = (
                self.normalization["obs_max"] - self.normalization["obs_min"] + 1e-6
            )
            self.normalization["action_span"] = (
                self.normalization["action_max"] - self.normalization["action_min"] + 1e-6
            )

        action_feature = self.policy.config.action_feature
        if action_feature is None:
            raise ValueError("ACT checkpoint is missing action feature metadata.")
        self.horizon_steps = int(self.policy.config.chunk_size)
        self.ac_dim = int(action_feature.shape[0])
        self.cond_steps = int(self.policy.config.n_obs_steps)
        self.obs_dim = int(
            (
                self.policy.config.robot_state_feature.shape[0]
                if self.policy.config.robot_state_feature
                else 0
            )
            + (
                self.policy.config.env_state_feature.shape[0]
                if self.policy.config.env_state_feature
                else 0
            )
        )
        self.use_state_as_env_when_missing = bool(
            self.policy.config.env_state_feature is not None and len(self.env_obs_keys) == 0
        )
        self.obs_shapes = {key: (int(self.lowdim_dims[key]),) for key in self.lowdim_keys}
        self.use_action_scheduler = False
        self.use_ood_monitor = False
        self.ema = None
        self.lerobot_src = lerobot_src
        print(
            f"[ACT base] ckpt={self.ckpt_dir} horizon={self.horizon_steps} action_dim={self.ac_dim} "
            f"obs_space={self.observation_space_kind} action_space={self.action_space_kind}",
            flush=True,
        )

    def reset(self):
        reset_fn = getattr(self.policy, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def _split_obs(self, obs):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if obs.ndim != 2:
            raise ValueError(f"ACT base expected flat obs [B, D], got {tuple(obs.shape)}")
        if self.observation_space_kind == "raw_lowdim":
            if self.normalization is None:
                raise ValueError("raw_lowdim ACT base requires normalization stats.")
            obs = (
                ((obs + 1.0) * 0.5) * self.normalization["obs_span"] + self.normalization["obs_min"]
            ).to(torch.float32)
        elif self.observation_space_kind != "dsrl_normalized":
            raise ValueError(
                f"unsupported ACT observation_space_kind={self.observation_space_kind}"
            )

        total_dim = sum(int(self.lowdim_dims[key]) for key in self.lowdim_keys)
        if obs.shape[1] != total_dim:
            raise ValueError(f"ACT base obs dim mismatch: got {obs.shape[1]}, expected {total_dim}")
        out = {}
        start = 0
        for key in self.lowdim_keys:
            dim = int(self.lowdim_dims[key])
            out[key] = obs[:, start : start + dim]
            start += dim
        return out

    def _concat_obs_parts(self, obs_parts, keys):
        pieces = [
            obs_parts[key].reshape(obs_parts[key].shape[0], -1) for key in keys if key in obs_parts
        ]
        if not pieces:
            batch_size = next(iter(obs_parts.values())).shape[0] if obs_parts else 1
            return torch.zeros((batch_size, 0), device=self.device, dtype=torch.float32)
        return torch.cat(pieces, dim=-1)

    def _build_policy_obs(self, obs):
        obs_parts = self._split_obs(obs)
        state_vec = self._concat_obs_parts(obs_parts, self.state_obs_keys)
        policy_obs = {self.obs_state_key: state_vec}
        if self.policy.config.env_state_feature is not None:
            if self.env_obs_keys:
                policy_obs[self.obs_env_state_key] = self._concat_obs_parts(
                    obs_parts, self.env_obs_keys
                )
            elif self.use_state_as_env_when_missing:
                policy_obs[self.obs_env_state_key] = state_vec.clone()
            else:
                raise ValueError(
                    "ACT checkpoint expects environment_state but act_env_obs_keys is empty."
                )
        return policy_obs

    def _actions_to_dsrl_space(self, action_chunk):
        action_chunk = torch.as_tensor(action_chunk, device=self.device, dtype=torch.float32)
        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(0)
        if action_chunk.ndim != 3:
            raise ValueError(
                f"ACT base expected action chunk [B, H, A], got {tuple(action_chunk.shape)}"
            )
        if self.action_space_kind == "dsrl_normalized":
            return action_chunk.clamp(-1.0, 1.0)
        if self.action_space_kind != "raw_action":
            raise ValueError(f"unsupported ACT action_space_kind={self.action_space_kind}")
        if self.normalization is None:
            raise ValueError("raw_action ACT base requires normalization stats.")
        normalized = 2.0 * (
            (action_chunk - self.normalization["action_min"]) / self.normalization["action_span"]
            - 0.5
        )
        return normalized.clamp(-1.0, 1.0)

    def __call__(self, obs, initial_noise=None, return_numpy=True):
        del initial_noise
        with torch.no_grad():
            policy_obs = self._build_policy_obs(obs)
            processed = self.preprocessor(policy_obs)
            action_chunk = self.policy.predict_action_chunk(processed)
            action_chunk = self.postprocessor(action_chunk)
            action_chunk = self._actions_to_dsrl_space(action_chunk)
        if return_numpy:
            return action_chunk.detach().cpu().numpy()
        return action_chunk.detach()


def load_base_policy(cfg):
    base_policy_type = str(_cfg_get(cfg, "base_policy_type", "diffusion")).strip().lower()
    if base_policy_type == "vqbet":
        from vqbet_base_policy import VQBETBasePolicyWrapper

        return VQBETBasePolicyWrapper(cfg)
    if base_policy_type == "act":
        return ACTBasePolicyWrapper(cfg)
    if base_policy_type not in {"diffusion", "dppo"}:
        raise ValueError(f"unknown base_policy_type={base_policy_type!r}")
    base_policy = hydra.utils.instantiate(cfg.model)
    base_policy = base_policy.eval()
    return DPPOBasePolicyWrapper(base_policy)


class LoggingCallback(BaseCallback):
    def __init__(
        self,
        action_chunk=4,
        log_freq=1000,
        use_wandb=True,
        eval_env=None,
        eval_freq=70,
        eval_episodes=2,
        verbose=0,
        rew_offset=0,
        num_train_env=1,
        num_eval_env=1,
        algorithm="dsrl_sac",
        max_steps=-1,
        deterministic_eval=False,
        success_window=200,
        eval_base_policy=False,
    ):
        super().__init__(verbose)
        self.action_chunk = action_chunk
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.use_wandb = use_wandb
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.log_count = 0
        self.total_reward = 0
        self.rew_offset = rew_offset
        self.total_timesteps = 0
        self.num_train_env = num_train_env
        self.num_eval_env = num_eval_env
        self.episode_success = np.zeros(self.num_train_env)
        self.episode_completed = np.zeros(self.num_train_env)
        self.algorithm = algorithm
        self.max_steps = max_steps
        self.deterministic_eval = deterministic_eval
        self.success_window = int(success_window)
        self.eval_base_policy = bool(eval_base_policy)
        self.completed_success_recent = deque(maxlen=max(1, int(success_window)))
        self.completed_success_since_log = []

    def _on_step(self):
        for env_idx, info in enumerate(self.locals["infos"]):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
            if _extract_success(info):
                self.episode_success[env_idx] = 1
        rew = self.locals["rewards"]
        self.total_reward += np.mean(rew)
        dones = self.locals["dones"]
        for env_idx, done_i in enumerate(dones):
            if bool(done_i):
                success_i = float(self.episode_success[env_idx] > 0)
                self.completed_success_recent.append(success_i)
                self.completed_success_since_log.append(success_i)
                self.episode_success[env_idx] = 0
                self.episode_completed[env_idx] = 0
        # Exact env-step accumulation when wrapper provides per-decision executed steps.
        # Fallback to old estimate for compatibility.
        chunk_steps = 0
        for info in self.locals["infos"]:
            step_i = info.get("chunk_env_steps", self.action_chunk)
            try:
                step_i = int(step_i)
            except Exception:
                step_i = self.action_chunk
            chunk_steps += max(step_i, 1)
        self.total_timesteps += chunk_steps
        if self.n_calls % self.log_freq == 0:
            if len(self.episode_rewards) > 0:
                success_rate_raw = (
                    float(np.mean(self.completed_success_since_log))
                    if len(self.completed_success_since_log) > 0
                    else 0.0
                )
                success_rate_recent = (
                    float(np.mean(self.completed_success_recent))
                    if len(self.completed_success_recent) > 0
                    else success_rate_raw
                )
                success_rate_recent20 = (
                    float(np.mean(list(self.completed_success_recent)[-20:]))
                    if len(self.completed_success_recent) > 0
                    else success_rate_raw
                )
                # Keep both AAC-prefixed tags and rollout/* tags for consistent dashboards.
                ep_rew_mean = float(np.mean(self.episode_rewards))
                ep_len_mean = float(np.mean(self.episode_lengths))
                self.logger.record("aac/train_success_rate", float(success_rate_recent))
                self.logger.record("aac/train_success_rate_recent20", float(success_rate_recent20))
                self.logger.record("aac/train_success_rate_window_raw", float(success_rate_raw))
                self.logger.record("aac/train_ep_rew_mean_cb", ep_rew_mean)
                self.logger.record("aac/train_ep_len_mean_cb", ep_len_mean)
                self.logger.record("aac/train_rew_mean_cb", float(np.mean(self.total_reward)))
                self.logger.record("aac/train_env_steps_est", float(self.total_timesteps))
                self.logger.record("rollout/success_rate", float(success_rate_recent))
                self.logger.record("rollout/success_rate_recent20", float(success_rate_recent20))
                self.logger.record("rollout/success_rate_window_raw", float(success_rate_raw))
                self.logger.record("rollout/ep_rew_mean", ep_rew_mean)
                self.logger.record("rollout/ep_len_mean", ep_len_mean)
                if self.use_wandb:
                    self.log_count += 1
                    wandb.log(
                        {
                            "train/ep_len_mean": np.mean(self.episode_lengths),
                            "train/success_rate": success_rate_recent,
                            "train/success_rate_recent20": success_rate_recent20,
                            "train/success_rate_window_raw": success_rate_raw,
                            "train/ep_rew_mean": np.mean(self.episode_rewards),
                            "train/rew_mean": np.mean(self.total_reward),
                            "train/timesteps": self.total_timesteps,
                            "train/ent_coef": self.locals["self"].logger.name_to_value[
                                "train/ent_coef"
                            ],
                            "train/actor_loss": self.locals["self"].logger.name_to_value[
                                "train/actor_loss"
                            ],
                            "train/critic_loss": self.locals["self"].logger.name_to_value[
                                "train/critic_loss"
                            ],
                            "train/ent_coef_loss": self.locals["self"].logger.name_to_value[
                                "train/ent_coef_loss"
                            ],
                        },
                        step=self.log_count,
                    )
                    if len(self.completed_success_since_log) > 0:
                        wandb.log(
                            {
                                "train/success_rate": success_rate_recent,
                            },
                            step=self.log_count,
                        )
                    if self.algorithm == "dsrl_na":
                        wandb.log(
                            {
                                "train/noise_critic_loss": self.locals["self"].logger.name_to_value[
                                    "train/noise_critic_loss"
                                ],
                            },
                            step=self.log_count,
                        )
                self.episode_rewards = []
                self.episode_lengths = []
                self.total_reward = 0
                self.completed_success_since_log = []

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.evaluate(self.locals["self"], deterministic=False)
            if self.deterministic_eval:
                self.evaluate(self.locals["self"], deterministic=True)
            if self.eval_base_policy:
                self.evaluate(self.locals["self"], deterministic=False, eval_mode="base")
        return True

    def evaluate(self, agent, deterministic=False, eval_mode="agent"):
        if self.eval_episodes <= 0:
            return
        env = self.eval_env
        # NOTE:
        # evaluate() is called once before model.learn(...) in baselines.policy_decorator.train.py.
        # At that moment callback internals (self.model/self.logger) are not initialized yet.
        # Use logger only when available; otherwise perform print-only pre-eval.
        logger = None
        if hasattr(self, "logger") and hasattr(self, "model"):
            logger = self.logger
        elif hasattr(agent, "_logger"):
            logger = agent.logger
        with torch.no_grad():
            success = []
            rew_total, total_ep = 0, 0
            rew_ep = np.zeros(self.num_eval_env)
            for i in range(self.eval_episodes):
                obs = env.reset()
                success_i = np.zeros(obs.shape[0])
                for _ in range(self.max_steps):
                    if eval_mode == "base":
                        if not hasattr(agent, "base_policy") or agent.base_policy is None:
                            raise ValueError(
                                "Base-policy evaluation requested but agent.base_policy is missing."
                            )
                        if hasattr(agent, "predict_base"):
                            action, _ = agent.predict_base(obs)
                        else:
                            if not hasattr(agent, "diffusion_act_chunk") or not hasattr(
                                agent, "diffusion_act_dim"
                            ):
                                raise ValueError(
                                    "Base-policy evaluation requires diffusion_act_chunk/diffusion_act_dim."
                                )
                            obs_tensor = torch.as_tensor(
                                obs, device=agent.device, dtype=torch.float32
                            )
                            noise = torch.randn(
                                (
                                    obs_tensor.shape[0],
                                    int(agent.diffusion_act_chunk),
                                    int(agent.diffusion_act_dim),
                                ),
                                device=agent.device,
                                dtype=torch.float32,
                            )
                            action = agent.base_policy(obs_tensor, noise, return_numpy=False)
                            action = action.reshape(obs_tensor.shape[0], -1).detach().cpu().numpy()
                    elif self.algorithm in ["dsrl_sac", "residual_sac"]:
                        action, _ = agent.predict(obs, deterministic=deterministic)
                    elif self.algorithm == "dsrl_na":
                        action, _ = agent.predict_diffused(obs, deterministic=deterministic)
                    next_obs, reward, done, info = env.step(action)
                    obs = next_obs
                    rew_ep += reward
                    rew_total += sum(rew_ep[done])
                    rew_ep[done] = 0
                    total_ep += np.sum(done)
                    for env_idx, info_i in enumerate(info):
                        if _extract_success(info_i):
                            success_i[env_idx] = 1
                success.append(success_i.mean())
                print(f"eval episode {i} at timestep {self.total_timesteps}")

            success_rate = np.mean(success)
            if total_ep > 0:
                avg_rew = rew_total / total_ep
            else:
                avg_rew = 0
            if eval_mode == "base":
                tag_prefix = "aac/eval_base"
            else:
                tag_prefix = "aac/eval_deterministic" if deterministic else "aac/eval"
            if logger is not None:
                logger.record(f"{tag_prefix}_success_rate", float(success_rate))
                logger.record(f"{tag_prefix}_avg_reward", float(avg_rew))
                logger.record(f"{tag_prefix}_env_steps_est", float(self.total_timesteps))
            print(
                f"[eval-summary] mode={eval_mode} deterministic={deterministic} "
                f"success_rate={success_rate:.4f} avg_reward={avg_rew:.4f} "
                f"env_steps_est={self.total_timesteps}"
            )
            if self.use_wandb:
                name = "eval_base" if eval_mode == "base" else "eval"
                if eval_mode == "base":
                    wandb.log(
                        {
                            f"{name}/success_rate": success_rate,
                            f"{name}/reward": avg_rew,
                        },
                        step=self.log_count,
                    )
                elif deterministic:
                    wandb.log(
                        {
                            f"{name}/success_rate_deterministic": success_rate,
                            f"{name}/reward_deterministic": avg_rew,
                        },
                        step=self.log_count,
                    )
                else:
                    wandb.log(
                        {
                            f"{name}/success_rate": success_rate,
                            f"{name}/reward": avg_rew,
                            f"{name}/timesteps": self.total_timesteps,
                        },
                        step=self.log_count,
                    )

    def set_timesteps(self, timesteps):
        self.total_timesteps = timesteps


def evaluate_residual_success_rate(
    model, eval_env, eval_episodes, deterministic=False, max_steps=-1
):
    env = eval_env
    num_envs = int(getattr(env, "num_envs", 1))
    episode_successes = []
    episode_success = np.zeros(num_envs, dtype=np.float32)
    episode_steps = np.zeros(num_envs, dtype=np.int32)
    obs = env.reset()
    was_training = bool(getattr(model.policy, "training", False))
    model.policy.set_training_mode(False)
    try:
        with torch.no_grad():
            # The env wrapper should terminate at max_steps; retain a safety cap
            # so an invalid environment cannot hang either training or a watcher.
            if max_steps > 0:
                max_vec_steps = max_steps * max(
                    1, int(np.ceil(eval_episodes / max(num_envs, 1))) + 1
                )
            else:
                max_vec_steps = max(1000, eval_episodes * 100)
            vec_steps = 0
            while len(episode_successes) < eval_episodes and vec_steps < max_vec_steps:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, _, dones, infos = env.step(action)
                vec_steps += 1
                episode_steps += 1
                for env_idx, info in enumerate(infos):
                    if _extract_success(info):
                        episode_success[env_idx] = 1.0
                    done_i = bool(dones[env_idx])
                    timeout_i = max_steps > 0 and episode_steps[env_idx] >= max_steps
                    if done_i or timeout_i:
                        episode_successes.append(float(episode_success[env_idx] > 0))
                        episode_success[env_idx] = 0.0
                        episode_steps[env_idx] = 0
                        if len(episode_successes) >= eval_episodes:
                            break
    finally:
        model.policy.set_training_mode(was_training)

    if len(episode_successes) == 0:
        return 0.0
    return float(np.mean(episode_successes[:eval_episodes]))


class EnvStepEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_interval=30000,
        eval_episodes=50,
        deterministic=False,
        max_steps=-1,
        verbose=0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_interval = int(eval_interval)
        self.eval_episodes = int(eval_episodes)
        self.deterministic = bool(deterministic)
        self.max_steps = int(max_steps)
        self.next_eval_step = 0

    def _on_training_start(self) -> None:
        if self.eval_interval > 0 and self.eval_episodes > 0:
            env_step = int(getattr(self.model, "env_steps_exact", self.num_timesteps))
            self._evaluate_and_log(env_step=env_step)
            self.next_eval_step = ((env_step // self.eval_interval) + 1) * self.eval_interval

    def _on_step(self) -> bool:
        if self.eval_interval <= 0 or self.eval_episodes <= 0:
            return True
        env_step = int(getattr(self.model, "env_steps_exact", self.num_timesteps))
        if env_step >= self.next_eval_step:
            self._evaluate_and_log(env_step=env_step)
            while self.next_eval_step <= env_step:
                self.next_eval_step += self.eval_interval
        return True

    def _evaluate_and_log(self, env_step):
        success_rate = self._evaluate_success_rate()
        self.logger.record("env_step_eval/success_rate", float(success_rate))
        self.logger.record("env_step_eval/episodes", self.eval_episodes)
        self.logger.dump(step=int(env_step))
        print(
            f"[env-step-eval] env_step={int(env_step)} episodes={self.eval_episodes} "
            f"deterministic={self.deterministic} success_rate={success_rate:.4f}",
            flush=True,
        )

    def _evaluate_success_rate(self):
        return evaluate_residual_success_rate(
            self.model,
            self.eval_env,
            self.eval_episodes,
            deterministic=self.deterministic,
            max_steps=self.max_steps,
        )


def publish_latest_checkpoint(model_path):
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    latest_path = os.path.join(os.path.dirname(model_path), "latest.zip")
    temp_path = latest_path + "." + uuid.uuid4().hex + ".tmp"
    # Publish only after the model and its paired replay have finished saving.
    os.symlink(os.path.basename(model_path), temp_path)
    os.replace(temp_path, latest_path)


def publish_async_eval_reference(model, reference_path):
    """Atomically publish the full model used to reconstruct async eval actors."""
    reference_path = hydra.utils.to_absolute_path(str(reference_path))
    if not reference_path.endswith(".zip"):
        reference_path += ".zip"
    os.makedirs(os.path.dirname(reference_path), exist_ok=True)
    if os.path.lexists(reference_path):
        raise FileExistsError(f"async eval reference already exists: {reference_path}")
    temp_path = reference_path[:-4] + f".tmp-{uuid.uuid4().hex}.zip"
    try:
        model.save(temp_path)
        if not os.path.isfile(temp_path):
            raise RuntimeError(f"model.save did not create async eval reference: {temp_path}")
        os.replace(temp_path, reference_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    return reference_path


class PostRolloutCheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_freq,
        save_path,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=0,
        save_latest=False,
    ):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = bool(save_replay_buffer)
        self.save_vecnormalize = bool(save_vecnormalize)
        self.save_latest = bool(save_latest)
        self._next_save_env_step = self.save_freq

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        current_env_step = self._current_env_step()
        self._next_save_env_step = (
            (current_env_step // self.save_freq + 1) * self.save_freq if self.save_freq > 0 else 0
        )

    def _current_env_step(self):
        return int(getattr(self.model, "env_steps_exact", self.num_timesteps))

    def _checkpoint_path(self, step_value, checkpoint_type="", extension=""):
        return os.path.join(
            self.save_path,
            f"{self.name_prefix}_{checkpoint_type}{step_value}_steps.{extension}",
        )

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        current_env_step = self._current_env_step()
        if self.save_freq <= 0 or current_env_step < self._next_save_env_step:
            return
        if self.save_replay_buffer and getattr(self.model, "replay_buffer", None) is None:
            raise RuntimeError("Cannot save a resumable checkpoint without its replay buffer.")
        model_path = self._checkpoint_path(current_env_step, extension="zip")
        self.model.save(model_path)
        if self.verbose >= 2:
            print(f"Saving model checkpoint to {model_path}")

        if (
            self.save_replay_buffer
            and hasattr(self.model, "replay_buffer")
            and self.model.replay_buffer is not None
        ):
            replay_buffer_path = self._checkpoint_path(
                current_env_step, "replay_buffer_", extension="pkl"
            )
            self.model.save_replay_buffer(replay_buffer_path)
            if self.verbose > 1:
                print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

        if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
            vec_normalize_path = self._checkpoint_path(
                current_env_step, "vecnormalize_", extension="pkl"
            )
            self.model.get_vec_normalize_env().save(vec_normalize_path)
            if self.verbose >= 2:
                print(f"Saving model VecNormalize to {vec_normalize_path}")

        if self.save_latest:
            publish_latest_checkpoint(model_path)

        while self._next_save_env_step <= current_env_step:
            self._next_save_env_step += self.save_freq


class ResidualActorSnapshotCallback(BaseCallback):
    """Publish lightweight actor snapshots for an out-of-process evaluator."""

    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        self._next_save_env_step = self.save_freq

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)
        current_env_step = int(getattr(self.model, "env_steps_exact", self.num_timesteps))
        self._next_save_env_step = (
            (current_env_step // self.save_freq + 1) * self.save_freq if self.save_freq > 0 else 0
        )

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        current_env_step = int(getattr(self.model, "env_steps_exact", self.num_timesteps))
        if self.save_freq <= 0 or current_env_step < self._next_save_env_step:
            return

        path = os.path.join(self.save_path, f"actor_{current_env_step}_steps.pth")
        temp_path = path + "." + uuid.uuid4().hex + ".tmp"
        payload = {
            "actor": {
                name: value.detach().cpu() for name, value in self.model.actor.state_dict().items()
            },
            "env_steps_exact": current_env_step,
            "num_timesteps": int(self.model.num_timesteps),
            "n_updates": int(getattr(self.model, "_n_updates", 0)),
        }
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
        if self.verbose > 0:
            print(f"[async-eval-snapshot] env_step={current_env_step} path={path}", flush=True)

        while self._next_save_env_step <= current_env_step:
            self._next_save_env_step += self.save_freq


class StopTrainingOnEnvStepCallback(BaseCallback):
    def __init__(self, max_env_steps, verbose=0):
        super().__init__(verbose)
        self.max_env_steps = int(max_env_steps)

    def _on_step(self):
        current_env_step = int(getattr(self.model, "env_steps_exact", self.num_timesteps))
        keep_training = current_env_step < self.max_env_steps
        if not keep_training and self.verbose > 0:
            print(
                f"[env-step-stop] env_step={current_env_step} target={self.max_env_steps}",
                flush=True,
            )
        return keep_training


def collect_rollouts(model, env, num_steps, base_policy, cfg):
    obs = env.reset()
    diffusion_horizon = int(getattr(cfg, "diffusion_horizon_steps", cfg.act_steps))
    base_policy_horizon = int(getattr(cfg, "base_policy_horizon_steps", diffusion_horizon))
    num_envs = int(getattr(env, "num_envs", cfg.env.n_envs))
    env_steps_exact = 0
    for i in range(num_steps):
        noise_horizon = (
            base_policy_horizon if cfg.algorithm == "residual_sac" else diffusion_horizon
        )
        noise = torch.randn(num_envs, noise_horizon, cfg.action_dim).to(device=cfg.device)
        if cfg.algorithm == "dsrl_sac":
            noise[noise < -cfg.train.action_magnitude] = -cfg.train.action_magnitude
            noise[noise > cfg.train.action_magnitude] = cfg.train.action_magnitude
        action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)
        if cfg.algorithm == "residual_sac":
            action = action[:, :diffusion_horizon]
        next_obs, reward, done, info = env.step(action)
        stored_next_obs = next_obs
        for info_i in info:
            step_i = info_i.get("chunk_env_steps", cfg.act_steps)
            try:
                step_i = int(step_i)
            except Exception:
                step_i = cfg.act_steps
            env_steps_exact += max(step_i, 1)
        if cfg.algorithm == "dsrl_na":
            action_store = action
        elif cfg.algorithm == "dsrl_sac":
            action_store = noise.detach().cpu().numpy()
        elif cfg.algorithm == "residual_sac":
            action_store = action
        action_store = action_store.reshape(-1, action_store.shape[1] * action_store.shape[2])
        if cfg.algorithm in ["dsrl_sac", "residual_sac"]:
            action_store = model.policy.scale_action(action_store)
        if cfg.algorithm == "residual_sac":
            stored_next_obs = np.array(next_obs, copy=True)
            for idx, done_i in enumerate(done):
                if done_i and info[idx].get("terminal_observation") is not None:
                    stored_next_obs[idx] = info[idx]["terminal_observation"]
            next_noise = torch.randn(num_envs, base_policy_horizon, cfg.action_dim).to(
                device=cfg.device
            )
            base_next_action = base_policy(
                torch.tensor(stored_next_obs, device=cfg.device, dtype=torch.float32), next_noise
            )
            base_next_action = base_next_action[:, :diffusion_horizon]
            base_next_action = base_next_action.reshape(
                -1, base_next_action.shape[1] * base_next_action.shape[2]
            )
            base_action = action.reshape(-1, action.shape[1] * action.shape[2])
        else:
            base_action = None
            base_next_action = None
        model.replay_buffer.add(
            obs=obs,
            next_obs=stored_next_obs,
            action=action_store,
            reward=reward,
            done=done,
            infos=info,
            base_action=base_action,
            next_base_action=base_next_action,
        )
        obs = next_obs
    model.replay_buffer.final_offline_step()
    policy_steps = int(num_steps * num_envs)
    print(
        f"[init-rollout-summary] policy_steps={policy_steps} env_steps={env_steps_exact}",
        flush=True,
    )
    return {
        "policy_steps": policy_steps,
        "env_steps_exact": int(env_steps_exact),
    }


def load_offline_data(model, offline_data_path, n_env):
    # this function should only be applied with dsrl_na
    offline_data = np.load(offline_data_path)
    obs = offline_data["states"]
    next_obs = offline_data["states_next"]
    actions = offline_data["actions"]
    rewards = offline_data["rewards"]
    terminals = offline_data["terminals"]
    for i in range(int(obs.shape[0] / n_env)):
        model.replay_buffer.add(
            obs=obs[n_env * i : n_env * i + n_env],
            next_obs=next_obs[n_env * i : n_env * i + n_env],
            action=actions[n_env * i : n_env * i + n_env],
            reward=rewards[n_env * i : n_env * i + n_env],
            done=terminals[n_env * i : n_env * i + n_env],
            infos=[{}] * n_env,
        )
    model.replay_buffer.final_offline_step()
