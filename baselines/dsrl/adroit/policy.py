from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils


def _extract_ckpt_stat(stats: Dict, key: str) -> np.ndarray:
    if key not in stats:
        raise KeyError(f"normalization stat missing key={key}; available={list(stats.keys())}")
    return np.asarray(stats[key], dtype=np.float32).reshape(-1)


def _extract_adroit_stats(ckpt_dict: Dict, obs_key: str = "flat") -> Dict[str, np.ndarray]:
    obs_stats = ckpt_dict.get("obs_normalization_stats", {}).get(obs_key)
    if obs_stats is None:
        raise KeyError(f"ckpt missing obs_normalization_stats[{obs_key!r}]")
    action_stats = ckpt_dict.get("action_normalization_stats", {}).get("actions")
    if action_stats is None:
        raise KeyError("ckpt missing action_normalization_stats['actions']")
    stats = {
        "obs_mean": _extract_ckpt_stat(obs_stats, "offset"),
        "obs_std": _extract_ckpt_stat(obs_stats, "scale"),
        "action_mean": _extract_ckpt_stat(action_stats, "offset"),
        "action_std": _extract_ckpt_stat(action_stats, "scale"),
    }
    stats["obs_std"] = np.where(stats["obs_std"] < 1e-8, 1.0, stats["obs_std"]).astype(np.float32)
    stats["action_std"] = np.where(stats["action_std"] < 1e-8, 1.0, stats["action_std"]).astype(
        np.float32
    )
    return stats


def _configure_adroit_policy(policy, dp_sampler: str, dp_infer_steps: int) -> Dict:
    algo = policy.policy if hasattr(policy, "policy") else policy
    pred_h = int(algo.algo_config.horizon.prediction_horizon)
    sampler = str(dp_sampler).lower()
    if sampler not in {"ddim", "ddpm"}:
        raise ValueError(f"unsupported diffusion sampler {dp_sampler!r}")
    if int(dp_infer_steps) <= 0:
        raise ValueError(f"dp_infer_steps must be positive, got {dp_infer_steps}")
    with algo.algo_config.values_unlocked():
        algo.algo_config.horizon.action_horizon = pred_h
        algo.algo_config.horizon.prediction_horizon = pred_h
        if sampler == "ddim":
            algo.algo_config.ddpm.enabled = False
            algo.algo_config.ddim.enabled = True
            algo.algo_config.ddim.num_inference_timesteps = int(dp_infer_steps)
        else:
            algo.algo_config.ddpm.enabled = True
            algo.algo_config.ddim.enabled = False
            algo.algo_config.ddpm.num_inference_timesteps = int(dp_infer_steps)
    if sampler == "ddim":
        scheduler_cfg = algo.algo_config.ddim
        algo.noise_scheduler = DDIMScheduler(
            num_train_timesteps=int(scheduler_cfg.num_train_timesteps),
            beta_schedule=str(scheduler_cfg.beta_schedule),
            clip_sample=bool(scheduler_cfg.clip_sample),
            set_alpha_to_one=bool(scheduler_cfg.set_alpha_to_one),
            steps_offset=int(scheduler_cfg.steps_offset),
            prediction_type=str(scheduler_cfg.prediction_type),
        )
    else:
        scheduler_cfg = algo.algo_config.ddpm
        algo.noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(scheduler_cfg.num_train_timesteps),
            beta_schedule=str(scheduler_cfg.beta_schedule),
            clip_sample=bool(scheduler_cfg.clip_sample),
            prediction_type=str(scheduler_cfg.prediction_type),
        )
    if hasattr(algo, "ac_horizon"):
        algo.ac_horizon = pred_h
    if hasattr(algo, "use_action_scheduler"):
        algo.use_action_scheduler = False
    if hasattr(algo, "use_ood_monitor"):
        algo.use_ood_monitor = False
    try:
        algo.nets.eval()
        for p in algo.nets.parameters():
            p.requires_grad = False
    except Exception:
        pass
    try:
        policy.start_episode()
    except Exception:
        pass
    return {
        "prediction_horizon": int(pred_h),
        "action_horizon": int(getattr(algo, "ac_horizon", pred_h)),
        "dp_sampler": sampler,
        "dp_infer_steps": int(dp_infer_steps),
        "noise_scheduler_class": type(algo.noise_scheduler).__name__,
        "use_action_scheduler": bool(getattr(algo, "use_action_scheduler", False)),
        "use_ood_monitor": bool(getattr(algo, "use_ood_monitor", False)),
    }


@dataclass
class AdroitPolicyBundle:
    policy: "AdroitDiffusionPolicyWrapper"
    env_meta: Dict
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    shape_metadata: Dict
    policy_meta: Dict


class AdroitDiffusionPolicyWrapper:
    """
    DSRL adapter for robomimic Adroit diffusion checkpoints.

    The DSRL actor emits the initial diffusion noise. This wrapper runs the same
    reverse diffusion path as robomimic, starting from that supplied noise, and
    returns the normalized DP sampler output used by the DSRL main critic. The
    environment wrapper alone converts it to physical Adroit actions.
    """

    def __init__(
        self,
        policy,
        *,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        device,
        obs_key: str = "flat",
    ):
        self.policy = policy
        self.algo = policy.policy if hasattr(policy, "policy") else policy
        self.device = torch.device(device)
        self.obs_key = str(obs_key)
        self.obs_mean = torch.as_tensor(obs_mean, device=self.device, dtype=torch.float32).reshape(
            1, -1
        )
        self.obs_std = torch.as_tensor(obs_std, device=self.device, dtype=torch.float32).reshape(
            1, -1
        )
        self.action_mean = torch.as_tensor(
            action_mean, device=self.device, dtype=torch.float32
        ).reshape(1, 1, -1)
        self.action_std = torch.as_tensor(
            action_std, device=self.device, dtype=torch.float32
        ).reshape(1, 1, -1)
        self.obs_dim = int(self.obs_mean.shape[-1])
        self.action_dim = int(self.action_mean.shape[-1])
        self.prediction_horizon = int(self.algo.algo_config.horizon.prediction_horizon)
        self.action_horizon = int(self.algo.algo_config.horizon.action_horizon)

    def normalize_obs_np(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        mean = self.obs_mean.detach().cpu().numpy().reshape(-1)
        std = self.obs_std.detach().cpu().numpy().reshape(-1)
        out = (obs - mean) / std
        return out.astype(np.float32)

    def __call__(self, obs, initial_noise, return_numpy: bool = True):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            noise_t = torch.as_tensor(initial_noise, device=self.device, dtype=torch.float32)
            if obs_t.ndim == 1:
                obs_t = obs_t.reshape(1, -1)
            if noise_t.ndim == 2:
                noise_t = noise_t.reshape(1, noise_t.shape[0], noise_t.shape[1])
            if noise_t.shape[1] < self.prediction_horizon:
                raise ValueError(
                    f"initial_noise horizon {noise_t.shape[1]} < policy horizon {self.prediction_horizon}"
                )
            if noise_t.shape[2] != self.action_dim:
                raise ValueError(
                    f"initial_noise action_dim {noise_t.shape[2]} != {self.action_dim}"
                )
            noise_t = noise_t[:, : self.prediction_horizon, :]
            normalized_actions = self._get_action_trajectory_from_noise(obs_t, noise_t)
        if return_numpy:
            return normalized_actions.detach().cpu().numpy().astype(np.float32)
        return normalized_actions

    def _get_action_trajectory_from_noise(
        self, normalized_obs: torch.Tensor, initial_noise: torch.Tensor
    ):
        algo = self.algo
        To = int(algo.algo_config.horizon.observation_horizon)
        Ta = int(algo.algo_config.horizon.action_horizon)
        Tp = int(algo.algo_config.horizon.prediction_horizon)
        if algo.algo_config.ddpm.enabled is True:
            num_inference_timesteps = int(algo.algo_config.ddpm.num_inference_timesteps)
        elif algo.algo_config.ddim.enabled is True:
            num_inference_timesteps = int(algo.algo_config.ddim.num_inference_timesteps)
        else:
            raise ValueError("Adroit diffusion policy has neither DDPM nor DDIM enabled")

        nets = algo.nets
        if getattr(algo, "ema", None) is not None:
            nets = algo.ema.averaged_model

        obs_dict = {self.obs_key: normalized_obs}
        inputs = {"obs": obs_dict, "goal": None}
        for key in algo.obs_shapes:
            if inputs["obs"][key].ndim - 1 == len(algo.obs_shapes[key]):
                inputs["obs"][key] = inputs["obs"][key].unsqueeze(1)
            assert inputs["obs"][key].ndim - 2 == len(algo.obs_shapes[key])

        obs_features = TensorUtils.time_distributed(
            inputs,
            nets["policy"]["obs_encoder"],
            inputs_as_kwargs=True,
        )
        obs_cond = obs_features.flatten(start_dim=1)
        naction = initial_noise[:, :Tp, :]
        algo.noise_scheduler.set_timesteps(num_inference_timesteps)

        for timestep in algo.noise_scheduler.timesteps:
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction,
                timestep=timestep,
                global_cond=obs_cond,
            )
            naction = algo.noise_scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=naction,
            ).prev_sample

        start = To - 1
        end = start + Ta
        return naction[:, start:end, :]


def load_adroit_policy_bundle(cfg) -> AdroitPolicyBundle:
    device = torch.device(str(cfg.device))
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(cfg.base_policy_path))
    shape_meta = ckpt_dict.get("shape_metadata", {})
    flat_shape = shape_meta.get("all_shapes", {}).get(str(cfg.obs_key))
    if flat_shape is None:
        raise KeyError(f"ckpt shape_metadata missing all_shapes[{str(cfg.obs_key)!r}]")
    obs_dim = int(np.prod(flat_shape))
    action_dim = int(shape_meta.get("ac_dim", 0))
    if int(cfg.obs_dim) != obs_dim:
        raise ValueError(f"cfg.obs_dim={cfg.obs_dim} but ckpt flat dim={obs_dim}")
    if int(cfg.action_dim) != action_dim:
        raise ValueError(f"cfg.action_dim={cfg.action_dim} but ckpt action dim={action_dim}")

    stats = _extract_adroit_stats(ckpt_dict, obs_key=str(cfg.obs_key))
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=False)
    policy_meta = _configure_adroit_policy(
        policy,
        dp_sampler=str(cfg.dp_sampler),
        dp_infer_steps=int(cfg.dp_infer_steps),
    )
    adapter = AdroitDiffusionPolicyWrapper(
        policy,
        obs_mean=stats["obs_mean"],
        obs_std=stats["obs_std"],
        action_mean=stats["action_mean"],
        action_std=stats["action_std"],
        device=device,
        obs_key=str(cfg.obs_key),
    )

    env_meta = copy.deepcopy(ckpt_dict.get("env_metadata", {}))
    env_meta["env_name"] = str(cfg.env_name or env_meta.get("env_name", "AdroitHandDoor-v1"))
    env_meta.setdefault("env_kwargs", {})
    env_meta["env_kwargs"]["max_episode_steps"] = int(cfg.env.max_episode_steps)
    env_meta["env_kwargs"].pop("horizon", None)

    return AdroitPolicyBundle(
        policy=adapter,
        env_meta=env_meta,
        obs_mean=stats["obs_mean"],
        obs_std=stats["obs_std"],
        action_mean=stats["action_mean"],
        action_std=stats["action_std"],
        shape_metadata=shape_meta,
        policy_meta=policy_meta,
    )
