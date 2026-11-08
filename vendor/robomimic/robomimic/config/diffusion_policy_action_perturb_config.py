"""
Config for Diffusion Policy with training-time action perturbation.
"""

from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig


class DiffusionPolicyActionPerturbConfig(DiffusionPolicyConfig):
    ALGO_NAME = "diffusion_policy_action_perturb"

    def algo_config(self):
        """
        Extend diffusion policy config with action perturbation controls.
        """
        super(DiffusionPolicyActionPerturbConfig, self).algo_config()

        self.algo.action_perturb.enabled = True
        self.algo.action_perturb.scale = 0.05
        self.algo.action_perturb.constant = 0.0
        self.algo.action_perturb.dims = 1
        self.algo.action_perturb.clip = True
