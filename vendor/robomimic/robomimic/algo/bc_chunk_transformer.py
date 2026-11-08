"""BC Chunk Transformer algorithm.

Predicts an action chunk starting from the current timestep and returns the full
chunk during inference.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributions as D

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo


@register_algo_factory_func("bc_chunk_transformer")
def algo_config_to_class(algo_config):
    return BCChunkTransformer, {}


class BCChunkTransformer(PolicyAlgo):
    """Transformer chunk policy with deterministic or GMM head."""

    def _create_networks(self):
        assert self.algo_config.transformer.enabled, "bc_chunk_transformer requires transformer.enabled=True"
        assert self.algo_config.chunk.enabled, "bc_chunk_transformer requires chunk.enabled=True"
        self._set_params_from_config()

        self.nets = nn.ModuleDict()
        if self.gmm_enabled:
            self.nets["policy"] = PolicyNets.TransformerGMMActorNetwork(
                obs_shapes=self.obs_shapes,
                goal_shapes=self.goal_shapes,
                # One GMM sample is a full chunk.
                ac_dim=self.ac_dim * self.pred_len,
                num_modes=self.algo_config.gmm.num_modes,
                min_std=self.algo_config.gmm.min_std,
                std_activation=self.algo_config.gmm.std_activation,
                low_noise_eval=self.algo_config.gmm.low_noise_eval,
                encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
                **BaseNets.transformer_args_from_config(self.algo_config.transformer),
            )
        else:
            self.nets["policy"] = PolicyNets.TransformerActorNetwork(
                obs_shapes=self.obs_shapes,
                goal_shapes=self.goal_shapes,
                # Predict a full chunk from the current context.
                ac_dim=self.ac_dim * self.pred_len,
                encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
                **BaseNets.transformer_args_from_config(self.algo_config.transformer),
            )
        self.nets = self.nets.float().to(self.device)

    def _set_params_from_config(self):
        self.context_length = int(self.algo_config.transformer.context_length)
        self.pred_len = int(self.algo_config.chunk.pred_len)
        self.loss_reduction = str(self.algo_config.chunk.loss_reduction)
        self.gmm_enabled = bool(getattr(self.algo_config.gmm, "enabled", False))

        assert self.context_length >= 1
        assert self.pred_len >= 1

    def process_batch_for_training(self, batch):
        input_batch = dict()
        h = self.context_length
        p = self.pred_len

        input_batch["obs"] = {k: batch["obs"][k][:, :h, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        # Core rule: supervise from current timestep (no h-1 shift).
        input_batch["actions"] = batch["actions"][:, 0:p, :]

        # Explicit correctness assertion for the first target element.
        if batch["actions"].shape[1] >= 1:
            assert torch.allclose(input_batch["actions"][:, 0, :], batch["actions"][:, 0, :])

        input_batch = TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        return input_batch

    def train_on_batch(self, batch, epoch, validate=False):
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(BCChunkTransformer, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                info.update(self._train_step(losses))

        return info

    def _forward_training(self, batch, epoch=None):
        TensorUtils.assert_size_at_dim(
            batch["obs"],
            size=(self.context_length),
            dim=1,
            msg=(
                "Error: expect temporal dimension of obs batch to match "
                "transformer context length {}".format(self.context_length)
            ),
        )

        if self.gmm_enabled:
            dists = self.nets["policy"].forward_train(
                obs_dict=batch["obs"],
                actions=None,
                goal_dict=batch["goal_obs"],
                low_noise_eval=False,
            )
            assert len(dists.batch_shape) == 2  # [B, T]
            # Use final timestep distribution to match current-context prediction.
            component_distribution = D.Normal(
                loc=dists.component_distribution.base_dist.loc[:, -1],
                scale=dists.component_distribution.base_dist.scale[:, -1],
            )
            component_distribution = D.Independent(component_distribution, 1)
            mixture_distribution = D.Categorical(logits=dists.mixture_distribution.logits[:, -1])
            dists_last = D.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                component_distribution=component_distribution,
            )
            target_flat = batch["actions"].reshape(batch["actions"].shape[0], -1)
            predictions = OrderedDict()
            predictions["log_probs"] = dists_last.log_prob(target_flat)
            return predictions

        full_actions = self.nets["policy"](obs_dict=batch["obs"], actions=None, goal_dict=batch["goal_obs"])
        # Use the last temporal token to predict the whole chunk.
        last_token = full_actions[:, -1, :]
        pred_actions = last_token.view(last_token.shape[0], self.pred_len, self.ac_dim)

        predictions = OrderedDict()
        predictions["actions"] = pred_actions
        return predictions

    def _compute_losses(self, predictions, batch):
        losses = OrderedDict()
        if self.gmm_enabled:
            action_loss = -predictions["log_probs"].mean()
            losses["log_probs"] = -action_loss
            losses["action_loss"] = action_loss
            return losses

        a_target = batch["actions"]
        actions = predictions["actions"]

        assert tuple(actions.shape[1:]) == (self.pred_len, self.ac_dim), (
            f"Expected predictions [B, {self.pred_len}, {self.ac_dim}], got {tuple(actions.shape)}"
        )
        assert tuple(a_target.shape[1:]) == (self.pred_len, self.ac_dim), (
            f"Expected targets [B, {self.pred_len}, {self.ac_dim}], got {tuple(a_target.shape)}"
        )

        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_loss = (
            self.algo_config.loss.l2_weight * losses["l2_loss"]
            + self.algo_config.loss.l1_weight * losses["l1_loss"]
            + self.algo_config.loss.cos_weight * losses["cos_loss"]
        )
        losses["action_loss"] = action_loss
        return losses

    def _train_step(self, losses):
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
            max_grad_norm=self.global_config.train.max_grad_norm,
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def get_action(self, obs_dict, goal_dict=None):
        """Return chunk action tensor [B, pred_len, ac_dim]."""
        assert not self.nets.training
        full_actions = self.nets["policy"](obs_dict, actions=None, goal_dict=goal_dict)
        last_token = full_actions[:, -1, :]
        return last_token.view(last_token.shape[0], self.pred_len, self.ac_dim)

    def log_info(self, info):
        log = super(BCChunkTransformer, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "log_probs" in info["losses"]:
            log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
