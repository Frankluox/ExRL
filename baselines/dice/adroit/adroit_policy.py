"""Native Adroit diffusion-policy loading for DICE-RL.

Checkpoint loading and coordinate handling are copied from the validated
Adroit ExRL-Q / ResFiT transfer.  The only DICE extension is explicit initial
diffusion noise: a fixed ``(state, z)`` deterministically produces the same
frozen-DP chunk everywhere it is used by the actor and critic.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

import robomimic.utils.tensor_utils as TensorUtils


def _checkpoint_stats(checkpoint: Dict[str, object]) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for name, key, prefix in (
        ("obs_normalization_stats", "flat", "state"),
        ("action_normalization_stats", "actions", "action"),
    ):
        stats = checkpoint[name][key]
        offset = np.asarray(stats["offset"], dtype=np.float32).reshape(-1)
        scale = np.asarray(stats["scale"], dtype=np.float32).reshape(-1)
        if (
            offset.shape != scale.shape
            or not np.isfinite(offset).all()
            or not np.isfinite(scale).all()
        ):
            raise ValueError("invalid Adroit checkpoint normalization: %s" % name)
        result[prefix + "_offset"] = offset
        result[prefix + "_scale"] = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
    return result


class AdroitDICEPolicyAdapter(nn.Module):
    """Frozen native robomimic DP with an explicit-noise sampling interface."""

    def __init__(self, rollout_policy, checkpoint: Dict[str, object], infer_steps: int):
        super().__init__()
        self.rollout_policy = rollout_policy
        self.algo = rollout_policy.policy
        self.device = torch.device(self.algo.device)
        self.horizon_steps = int(self.algo.algo_config.horizon.prediction_horizon)
        self.cond_steps = int(self.algo.algo_config.horizon.observation_horizon)
        self.ac_dim = int(self.algo.ac_dim)
        self.obs_dim = int(np.prod(self.algo.obs_shapes["flat"]))
        if self.cond_steps != 1:
            raise ValueError("Adroit DICE requires a one-frame diffusion policy")

        stats = _checkpoint_stats(checkpoint)
        if stats["state_offset"].shape != (self.obs_dim,):
            raise ValueError("Adroit state normalization dimension mismatch")
        if stats["action_offset"].shape != (self.ac_dim,):
            raise ValueError("Adroit action normalization dimension mismatch")
        # All registered Adroit policies use the native bounded action domain.
        if not np.array_equal(
            stats["action_offset"], np.zeros((self.ac_dim,), dtype=np.float32)
        ) or not np.array_equal(stats["action_scale"], np.ones((self.ac_dim,), dtype=np.float32)):
            raise ValueError("formal Adroit DICE requires identity checkpoint action normalization")
        self.register_buffer(
            "state_offset",
            torch.as_tensor(stats["state_offset"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "state_scale",
            torch.as_tensor(stats["state_scale"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "action_offset",
            torch.as_tensor(stats["action_offset"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "action_scale",
            torch.as_tensor(stats["action_scale"], dtype=torch.float32),
            persistent=False,
        )

        ema = getattr(self.algo, "ema", None)
        averaged_model = getattr(ema, "averaged_model", None)
        if averaged_model is None:
            raise ValueError("formal Adroit DICE checkpoint requires EMA weights")
        # object.__setattr__ avoids registering another alias of the same frozen
        # module in this adapter's state_dict.
        object.__setattr__(self, "_inference_nets", averaged_model)
        self._runtime_optimization = "baseline"
        self._autocast_dtype: Optional[torch.dtype] = None
        self._configure_sampler(int(infer_steps))
        self.set_eval()

    @property
    def inference_nets(self):
        return object.__getattribute__(self, "_inference_nets")

    def _configure_sampler(self, infer_steps: int) -> None:
        with self.algo.algo_config.values_unlocked():
            self.algo.algo_config.horizon.action_horizon = self.horizon_steps
            self.algo.algo_config.horizon.prediction_horizon = self.horizon_steps
            self.algo.algo_config.ddpm.enabled = False
            self.algo.algo_config.ddim.enabled = True
            self.algo.algo_config.ddim.num_inference_timesteps = int(infer_steps)
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        cfg = self.algo.algo_config.ddim
        self.algo.noise_scheduler = DDIMScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_schedule=cfg.beta_schedule,
            clip_sample=cfg.clip_sample,
            set_alpha_to_one=cfg.set_alpha_to_one,
            steps_offset=cfg.steps_offset,
            prediction_type=cfg.prediction_type,
        )
        if hasattr(self.algo, "ac_horizon"):
            self.algo.ac_horizon = self.horizon_steps
        if hasattr(self.algo, "use_action_scheduler"):
            self.algo.use_action_scheduler = False
        if hasattr(self.algo, "use_ood_monitor"):
            self.algo.use_ood_monitor = False

    def set_eval(self) -> None:
        self.eval()
        self.algo.set_eval()
        self.inference_nets.eval()
        for module in (self.algo.nets, self.inference_nets):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.rollout_policy.start_episode()

    def configure_runtime_optimization(
        self,
        level: str,
        phase2_amp: bool = True,
        phase2_compile: bool = False,
    ) -> None:
        if level not in ("baseline", "phase1", "phase2"):
            raise ValueError("unknown DICE runtime optimization level: %s" % level)
        self._runtime_optimization = str(level)
        self._autocast_dtype = None
        if level == "phase2" and phase2_amp and self.device.type == "cuda":
            major, _minor = torch.cuda.get_device_capability(self.device)
            self._autocast_dtype = torch.bfloat16 if major >= 8 else torch.float16
        if phase2_compile:
            raise ValueError(
                "native Adroit DP compilation is not audited; use the measured AMP/eager path"
            )

    def normalize_raw_state(self, state) -> np.ndarray:
        value = np.asarray(state, dtype=np.float32)
        if value.shape[-1] != self.obs_dim or not np.isfinite(value).all():
            raise ValueError("invalid raw Adroit state")
        offset = self.state_offset.detach().cpu().numpy()
        scale = self.state_scale.detach().cpu().numpy()
        return ((value - offset) / scale).astype(np.float32)

    def normalize_raw_action(self, action) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        if value.shape[-1] != self.ac_dim or not np.isfinite(value).all():
            raise ValueError("invalid raw Adroit action")
        offset = self.action_offset.detach().cpu().numpy()
        scale = self.action_scale.detach().cpu().numpy()
        return ((value - offset) / scale).astype(np.float32)

    def denormalize_action(self, action) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        if value.shape[-1] != self.ac_dim or not np.isfinite(value).all():
            raise ValueError("invalid normalized Adroit action")
        offset = self.action_offset.detach().cpu().numpy()
        scale = self.action_scale.detach().cpu().numpy()
        return (value * scale + offset).astype(np.float32)

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self._autocast_dtype,
            enabled=self._autocast_dtype is not None and self.device.type == "cuda",
        )

    @torch.no_grad()
    def sample_from_state_noise(
        self, state: torch.Tensor, noise_action: torch.Tensor
    ) -> torch.Tensor:
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        noise_action = torch.as_tensor(noise_action, dtype=torch.float32, device=self.device)
        expected_state = (int(state.shape[0]), self.obs_dim)
        expected_noise = (int(state.shape[0]), self.horizon_steps, self.ac_dim)
        if tuple(state.shape) != expected_state:
            raise ValueError(
                "Adroit DICE state must be [B,%d], got %s" % (self.obs_dim, tuple(state.shape))
            )
        if tuple(noise_action.shape) != expected_noise:
            raise ValueError(
                "Adroit DICE noise must be [B,%d,%d], got %s"
                % (self.horizon_steps, self.ac_dim, tuple(noise_action.shape))
            )
        if not torch.isfinite(state).all() or not torch.isfinite(noise_action).all():
            raise ValueError("Adroit DICE state/noise contains NaN or Inf")

        obs = {"flat": state[:, None, :]}
        inputs = {"obs": obs, "goal": None}
        with self._autocast():
            features = TensorUtils.time_distributed(
                inputs,
                self.inference_nets["policy"]["obs_encoder"],
                inputs_as_kwargs=True,
            )
            if features.ndim != 3:
                raise RuntimeError(
                    "unexpected Adroit DP encoded state shape: %s" % (tuple(features.shape),)
                )
            condition = features.flatten(start_dim=1)
            action = noise_action
            scheduler = self.algo.noise_scheduler
            infer_steps = int(self.algo.algo_config.ddim.num_inference_timesteps)
            scheduler.set_timesteps(infer_steps)
            for timestep in scheduler.timesteps:
                prediction = self.inference_nets["policy"]["noise_pred_net"](
                    sample=action,
                    timestep=timestep,
                    global_cond=condition,
                )
                action = scheduler.step(
                    model_output=prediction,
                    timestep=timestep,
                    sample=action,
                ).prev_sample
        action = action.to(dtype=torch.float32)
        if tuple(action.shape) != expected_noise or not torch.isfinite(action).all():
            raise RuntimeError("native Adroit DP returned an invalid action chunk")
        # Observation horizon is one, so the native start offset is zero and
        # the complete TA32 prediction is the action horizon.
        return action


@dataclass
class AdroitDICEPolicyBundle:
    policy: AdroitDICEPolicyAdapter
    checkpoint: Dict[str, object]
    env_meta: Dict[str, object]
    metadata: Dict[str, object]


def load_adroit_dice_policy(args, device: torch.device, contract):
    """Load and validate one registered native Adroit diffusion checkpoint."""

    import robomimic.utils.file_utils as FileUtils

    checkpoint = FileUtils.maybe_dict_from_checkpoint(ckpt_path=args.agent)
    shape = checkpoint["shape_metadata"]
    state_dim = int(np.prod(shape["all_shapes"]["flat"]))
    action_dim = int(shape["ac_dim"])
    if (state_dim, action_dim) != (contract.state_dim, contract.action_dim):
        raise ValueError(
            "Adroit checkpoint dimensions %d/%d do not match %s %d/%d"
            % (
                state_dim,
                action_dim,
                contract.env_name,
                contract.state_dim,
                contract.action_dim,
            )
        )
    env_meta = copy.deepcopy(checkpoint["env_metadata"])
    if str(env_meta.get("env_name")) != str(contract.env_id):
        raise ValueError("Adroit checkpoint environment does not match task")
    env_meta.setdefault("env_kwargs", {})["max_episode_steps"] = int(contract.env_horizon)
    env_meta["env_kwargs"].pop("horizon", None)

    rollout_policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_dict=checkpoint, device=device, verbose=False
    )
    policy = AdroitDICEPolicyAdapter(
        rollout_policy, checkpoint, infer_steps=contract.dp_infer_steps
    )
    policy.env_meta = copy.deepcopy(env_meta)
    if policy.horizon_steps != int(args.base_horizon_steps):
        raise ValueError(
            "Adroit checkpoint horizon %d does not match requested TA%d"
            % (policy.horizon_steps, int(args.base_horizon_steps))
        )
    metadata = {
        "checkpoint": args.agent,
        "env_meta": env_meta,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "prediction_horizon": policy.horizon_steps,
        "observation_horizon": policy.cond_steps,
        "dp_sampler": "ddim_eta0_explicit_initial_noise",
        "dp_infer_steps": int(contract.dp_infer_steps),
        "weight_source": "ema",
        "base_policy_frozen": True,
        "state_coordinates": "checkpoint_normalized_flat",
        "action_coordinates": "checkpoint_normalized_identity_native",
    }
    return AdroitDICEPolicyBundle(policy, checkpoint, env_meta, metadata)
