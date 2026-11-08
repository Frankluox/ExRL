"""
Diffusion policy variant with training-time action perturbation.
"""

import torch

import robomimic.utils.tensor_utils as TensorUtils
from robomimic.algo import register_algo_factory_func
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet


@register_algo_factory_func("diffusion_policy_action_perturb")
def algo_config_to_class(algo_config):
    if algo_config.unet.enabled:
        return DiffusionPolicyActionPerturbUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


class DiffusionPolicyActionPerturbUNet(DiffusionPolicyUNet):
    """
    Add per-sample constant epsilon perturbation on training labels.
    """

    def process_batch_for_training(self, batch):
        input_batch = super(DiffusionPolicyActionPerturbUNet, self).process_batch_for_training(batch)

        ap_cfg = self.algo_config.action_perturb
        if not ap_cfg.enabled:
            return input_batch

        scale = float(ap_cfg.scale)
        constant = float(ap_cfg.constant)
        if (scale == 0.0) and (constant == 0.0):
            return input_batch

        actions = input_batch["actions"]
        batch_size, _, act_dim = actions.shape
        n_dims = min(max(int(ap_cfg.dims), 0), int(act_dim))
        if n_dims == 0:
            return input_batch

        if constant != 0.0:
            eps_scalar = torch.full(
                (batch_size, 1, 1), constant, dtype=actions.dtype, device=actions.device
            )
        else:
            eps_scalar = torch.empty(
                (batch_size, 1, 1), dtype=actions.dtype, device=actions.device
            ).uniform_(-scale, scale)

        eps = torch.zeros((batch_size, 1, act_dim), dtype=actions.dtype, device=actions.device)
        eps[:, :, :n_dims] = eps_scalar

        perturbed_actions = actions + eps
        if bool(ap_cfg.clip):
            perturbed_actions = torch.clamp(perturbed_actions, -1.0, 1.0)

        input_batch["actions"] = perturbed_actions
        return TensorUtils.to_device(input_batch, self.device)
