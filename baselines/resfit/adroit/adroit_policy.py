"""Native Adroit DP loading and coordinates, adapted from Adroit ExRL-Q.

The ResFiT planner's historical ``state_dp`` field contains raw Adroit flat
observations here. Both the frozen DP and residual actor use the checkpoint's
observation normalization, exactly once. Action expansion and residual
composition remain the shared ResFiT implementation.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import torch

from baselines.resfit.adroit.td3_core import ResFiTActionScaler


def checkpoint_stats(ckpt):
    """Same offset/scale extraction and small-scale rule as Adroit_Q."""
    result = {}
    for name, key, prefix in (
        ("obs_normalization_stats", "flat", "flat"),
        ("action_normalization_stats", "actions", "action"),
    ):
        stats = ckpt[name][key]
        offset = np.asarray(stats["offset"], dtype=np.float32).reshape(-1)
        scale = np.asarray(stats["scale"], dtype=np.float32).reshape(-1)
        if (
            offset.shape != scale.shape
            or not np.isfinite(offset).all()
            or not np.isfinite(scale).all()
        ):
            raise ValueError("invalid checkpoint normalization: " + name)
        result[prefix + "_mean"] = offset
        result[prefix + "_std"] = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
    return result


class AdroitStatePreprocessor:
    mode = "dp_raw"

    def __init__(self, stats):
        self.mean = stats["flat_mean"].copy()
        self.std = stats["flat_std"].copy()
        self.state_dim = int(self.mean.size)

    def transform(self, raw_state):
        state = np.asarray(raw_state, dtype=np.float32)
        if state.shape[-1] != self.state_dim or not np.isfinite(state).all():
            raise ValueError("invalid raw Adroit state")
        return ((state - self.mean) / self.std).astype(np.float32)

    def manifest(self):
        return {
            "mode": self.mode,
            "state_dim": self.state_dim,
            "offset": self.mean.tolist(),
            "scale": self.std.tolist(),
            "clip": False,
            "source": "checkpoint.obs_normalization_stats.flat",
        }


class AdroitActionScaler(ResFiTActionScaler):
    """Use the native checkpoint affine map, retaining ResFiT expansion.

    Formal Adroit checkpoints use identity action normalization and the native
    [-1, 1] command domain. The environment retains its own actuator clipping;
    replay stores the complete command after ResFiT's outer clamp, just as DP.
    """

    def __init__(self, stats, action_scale=0.1, min_range_per_dim=0.1):
        self.offset = stats["action_mean"].copy()
        self.scale = stats["action_std"].copy()
        super().__init__(
            self.offset - self.scale,
            self.offset + self.scale,
            action_scale=action_scale,
            min_range_per_dim=min_range_per_dim,
        )

    def dp_to_raw(self, action_dp):
        self._check_last_dim(action_dp, "action_dp")
        return action_dp * self._like(self.scale, action_dp) + self._like(self.offset, action_dp)

    def raw_to_dp(self, action_raw):
        self._check_last_dim(action_raw, "action_raw")
        return (action_raw - self._like(self.offset, action_raw)) / self._like(
            self.scale, action_raw
        )

    def manifest(self):
        return {
            **super().manifest(),
            "dp_action_offset": self.offset.tolist(),
            "dp_action_scale": self.scale.tolist(),
            "source": "checkpoint.action_normalization_stats.actions",
        }


def configure_adroit_policy(policy, infer_steps):
    """Copied from train_adaptive_sac_adroit_q._configure_adroit_policy."""
    algo = policy.policy if hasattr(policy, "policy") else policy
    pred_h = int(algo.algo_config.horizon.prediction_horizon)
    if int(algo.algo_config.horizon.observation_horizon) != 1:
        raise ValueError("Adroit state ResFiT requires the formal one-frame DP")
    with algo.algo_config.values_unlocked():
        algo.algo_config.horizon.action_horizon = pred_h
        algo.algo_config.horizon.prediction_horizon = pred_h
        algo.algo_config.ddpm.enabled = False
        algo.algo_config.ddim.enabled = True
        algo.algo_config.ddim.num_inference_timesteps = int(infer_steps)
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    cfg = algo.algo_config.ddim
    algo.noise_scheduler = DDIMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        clip_sample=cfg.clip_sample,
        set_alpha_to_one=cfg.set_alpha_to_one,
        steps_offset=cfg.steps_offset,
        prediction_type=cfg.prediction_type,
    )
    if hasattr(algo, "ac_horizon"):
        algo.ac_horizon = pred_h
    if hasattr(algo, "use_action_scheduler"):
        algo.use_action_scheduler = False
    if hasattr(algo, "use_ood_monitor"):
        algo.use_ood_monitor = False
    algo.set_eval()
    # Native trajectory inference uses EMA, so freeze both modules explicitly.
    for module in (algo.nets, getattr(getattr(algo, "ema", None), "averaged_model", None)):
        if module is not None:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    policy.start_episode()
    return {
        "prediction_horizon": pred_h,
        "action_horizon": pred_h,
        "observation_horizon": 1,
        "dp_sampler": "ddim",
        "dp_infer_steps": int(infer_steps),
        "weight_source": "ema",
        "base_policy_frozen": True,
    }


class AdroitTrajectoryPolicy:
    def __init__(self, rollout_policy, state_preprocessor, env_meta):
        self.rollout_policy = rollout_policy
        self.algo = rollout_policy.policy
        self.state_preprocessor = state_preprocessor
        self.env_meta = env_meta

    @torch.no_grad()
    def _get_action_trajectory(self, obs_dict):
        raw = np.asarray(obs_dict["flat"], dtype=np.float32)
        normalized = self.state_preprocessor.transform(raw)
        obs = {"flat": torch.as_tensor(normalized, dtype=torch.float32, device=self.algo.device)}
        return self.algo._get_action_trajectory(obs_dict=obs)


def load_adroit_policy(args, device, contract):
    import robomimic.utils.file_utils as FileUtils

    ckpt = FileUtils.maybe_dict_from_checkpoint(ckpt_path=args.agent)
    shape = ckpt["shape_metadata"]
    state_dim = int(np.prod(shape["all_shapes"]["flat"]))
    action_dim = int(shape["ac_dim"])
    if (state_dim, action_dim) != (contract.state_dim, contract.action_dim):
        raise ValueError("checkpoint dimensions do not match the selected Adroit task")
    env_meta = copy.deepcopy(ckpt["env_metadata"])
    if env_meta["env_name"] != contract.env_id:
        raise ValueError("checkpoint environment does not match the selected Adroit task")
    env_meta.setdefault("env_kwargs", {})["max_episode_steps"] = int(args.env_horizon)
    env_meta["env_kwargs"].pop("horizon", None)
    stats = checkpoint_stats(ckpt)
    if stats["flat_mean"].size != state_dim or stats["action_mean"].size != action_dim:
        raise ValueError("checkpoint stats dimensions disagree with shape_metadata")
    # The registered checkpoints all use unnormalized, bounded native actions.
    if not np.array_equal(stats["action_mean"], np.zeros(action_dim)) or not np.array_equal(
        stats["action_std"], np.ones(action_dim)
    ):
        raise ValueError("formal Adroit action contract requires identity normalization")
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt, device=device, verbose=False)
    policy_meta = configure_adroit_policy(policy, args.dp_infer_steps)
    if policy_meta["prediction_horizon"] != args.base_horizon_steps:
        raise ValueError("checkpoint chunk horizon disagrees with ResFiT")
    if getattr(policy.policy, "ema", None) is None:
        raise ValueError("formal Adroit policy requires EMA weights")
    preprocessor = AdroitStatePreprocessor(stats)
    scaler = AdroitActionScaler(stats, args.action_scale, args.min_action_range)
    adapter = AdroitTrajectoryPolicy(policy, preprocessor, env_meta)
    metadata = {
        **policy_meta,
        "env_meta": env_meta,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "checkpoint": args.agent,
    }
    return SimpleNamespace(policy=adapter), metadata, scaler, preprocessor
