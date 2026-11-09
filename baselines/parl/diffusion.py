"""
Policy-Agnostic RL model based on DSRL Diffusion Q-Learning.
"""

import torch
import logging
import copy
import math

log = logging.getLogger(__name__)

from model.diffusion.diffusion import DiffusionModel
from model.diffusion.sampling import make_timesteps


class PARLDQLDiffusion(DiffusionModel):
    def __init__(
        self,
        actor,
        critic,
        use_ddim=False,
        # modifying denoising schedule
        min_sampling_denoising_std=0.1,
        # Conservative Q-Learning regularization. Disabled by default so this
        # class can still reproduce the previous DQL critic exactly.
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
        cql_num_base_policy_actions=32,
        cql_num_actions_to_keep=10,
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
        super().__init__(network=actor, use_ddim=use_ddim, **kwargs)
        # assert not self.use_ddim, "DQL does not support DDIM"
        self.critic = critic.to(self.device)

        # target critic
        self.critic_target = copy.deepcopy(self.critic)

        # reassign actor
        self.actor = self.network

        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std = min_sampling_denoising_std
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

    # ---------- RL training ----------#

    @staticmethod
    def _metrics_to_python(metrics):
        """Move scalar metric tensors to CPU with one device synchronization."""
        converted = dict(metrics)
        tensor_items = [
            (key, value)
            for key, value in metrics.items()
            if torch.is_tensor(value) and value.numel() == 1
        ]
        if not tensor_items:
            return converted
        tensor_values = torch.stack([value.detach().reshape(()) for _, value in tensor_items]).cpu()
        for (key, _), value in zip(tensor_items, tensor_values.tolist()):
            converted[key] = value
        return converted

    def _critic_output(self, critic, obs, actions):
        critic_output = critic(obs, actions)
        if isinstance(critic_output, tuple) and len(critic_output) == 2:
            first, second = critic_output
            if torch.is_tensor(second) and second.ndim == 3:
                return first, second
            return torch.stack([first, second], dim=0), None
        if torch.is_tensor(critic_output):
            if critic_output.ndim == 1:
                return critic_output.unsqueeze(0), None
            return critic_output, None
        raise TypeError(f"Unsupported critic output type: {type(critic_output)}")

    def _subsample_critic_values(self, q_values, q_logits=None):
        if self.critic_subsample_size is None:
            return q_values, q_logits
        subsample_size = int(self.critic_subsample_size)
        if subsample_size <= 0 or subsample_size >= q_values.shape[0]:
            return q_values, q_logits
        indices = torch.randint(
            low=0,
            high=q_values.shape[0],
            size=(subsample_size,),
            device=q_values.device,
        )
        q_values = q_values.index_select(0, indices)
        if q_logits is not None:
            q_logits = q_logits.index_select(0, indices)
        return q_values, q_logits

    def _reduce_critic_values(self, q_values, use_min=True):
        if use_min:
            return q_values.min(dim=0).values
        return q_values.mean(dim=0)

    def _distributional_td_loss(self, current_logits, target_q):
        if not hasattr(self.critic, "cross_entropy_loss_on_scalar"):
            raise TypeError(
                "distributional_critic=True requires a critic with cross_entropy_loss_on_scalar()."
            )
        return self.critic.cross_entropy_loss_on_scalar(
            current_logits,
            target_q,
        ).mean()

    def loss_critic(
        self,
        obs,
        next_obs,
        actions,
        rewards,
        terminated,
        gamma,
        mc_returns=None,
        collect_metrics=True,
    ):

        # get current Q-function
        current_q_values, current_q_logits = self._critic_output(
            self.critic,
            obs,
            actions,
        )

        # get next Q-function
        with torch.no_grad():
            if self.backup_use_parl_policy_actions:
                next_actions = self._sample_policy_cql_actions(
                    next_obs,
                    num_actions=1,
                    action_steps=actions.shape[1],
                    argmax=self.backup_policy_argmax,
                ).squeeze(1)
            else:
                next_actions = self.forward(
                    cond=next_obs,
                    deterministic=False,
                )  # forward() has no gradient, which is desired here.
            next_q_values, _ = self._critic_output(
                self.critic_target,
                next_obs,
                next_actions,
            )
            next_q_values, _ = self._subsample_critic_values(next_q_values)
            next_q = next_q_values.min(dim=0).values

            # terminal state mask
            mask = 1 - terminated

            # flatten
            rewards = rewards.view(-1)
            next_q = next_q.view(-1)
            mask = mask.view(-1)

            # target value
            target_q = rewards + gamma * next_q * mask

        # Update critic
        if self.distributional_critic or current_q_logits is not None:
            td_loss = self._distributional_td_loss(current_q_logits, target_q)
        else:
            td_loss = torch.mean((current_q_values - target_q.view(1, -1)) ** 2)
        loss_critic = td_loss
        cql_metrics = {}
        if self.cql_alpha > 0:
            cql_loss, cql_metrics = self.loss_cql(
                obs=obs,
                next_obs=next_obs,
                replay_actions=actions,
                current_q_values=current_q_values,
                mc_returns=mc_returns,
                collect_metrics=collect_metrics,
            )
            loss_critic = loss_critic + self.cql_alpha * cql_loss

        if collect_metrics:
            current_q = current_q_values.mean(dim=0).detach()
            current_q_min = current_q_values.min(dim=0).values.detach()
            target_q_detached = target_q.detach()
            critic_metrics = {
                "critic/td_loss": td_loss.detach(),
                "critic/cql_alpha": self.cql_alpha,
                "critic/distributional": int(current_q_logits is not None),
                "critic/q_pred_mean": current_q.mean(),
                "critic/q_pred_min": current_q_values.detach().min(),
                "critic/q_pred_max": current_q_values.detach().max(),
                "critic/q_pred_min_ensemble_mean": current_q_min.mean(),
                "critic/q_target_mean": target_q_detached.mean(),
                "critic/q_target_min": target_q_detached.min(),
                "critic/q_target_max": target_q_detached.max(),
                "critic/q_over_target_mean": (current_q - target_q_detached).mean(),
            }
            if mc_returns is not None:
                mc_returns_detached = mc_returns.detach()
                critic_metrics.update(
                    {
                        "critic/mc_return_mean": mc_returns_detached.mean(),
                        "critic/mc_return_min": mc_returns_detached.min(),
                        "critic/mc_return_max": mc_returns_detached.max(),
                    }
                )
            critic_metrics.update(cql_metrics)
            self.last_critic_metrics = self._metrics_to_python(critic_metrics)
        return loss_critic

    def _sample_random_cql_actions(self, batch_size, num_actions, action_steps):
        shape = (batch_size, num_actions, action_steps, self.action_dim)
        if self.cql_action_sample_method == "uniform":
            actions = torch.empty(shape, device=self.betas.device)
            actions.uniform_(self.cql_action_low, self.cql_action_high)
            return actions
        if self.cql_action_sample_method == "normal":
            return torch.randn(shape, device=self.betas.device)
        raise ValueError(f"Unsupported cql_action_sample_method={self.cql_action_sample_method}")

    def _local_optimization_steps(
        self,
        cond,
        base_policy_actions,
        num_steps,
        step_size,
        optimize_critic_ensemble_min=False,
        use_target_critic=False,
        keep_action_with_max_value=True,
        collect_metrics=True,
    ):
        """Official PA-RL local action optimization in PyTorch.

        `base_policy_actions` is shaped (B, K, H, A). We optimize the flattened
        action chunks, clip to the configured action bounds, and return the
        best action seen during the ascent path, matching PA-RL's
        `keep_action_with_max_value=True` behavior.
        """
        if base_policy_actions.ndim != 4:
            raise ValueError("base_policy_actions must have shape (B, K, H, A)")
        B, K, H, A = base_policy_actions.shape
        flat_actions = base_policy_actions.reshape(B * K, H, A).detach()
        repeated_cond = self._repeat_cond(cond, K)
        critic = self.critic_target if use_target_critic else self.critic

        best_actions = flat_actions.clone()
        best_values = torch.full(
            (B * K,),
            -torch.inf,
            device=flat_actions.device,
            dtype=flat_actions.dtype,
        )
        best_indices = (
            torch.zeros(B * K, device=flat_actions.device, dtype=torch.long)
            if collect_metrics
            else None
        )
        last_gradient_norm = torch.zeros((), device=flat_actions.device)

        num_steps = max(0, int(num_steps))
        for step in range(num_steps):
            actions_for_grad = flat_actions.detach().requires_grad_(True)
            with torch.enable_grad():
                q_values, _ = self._critic_output(critic, repeated_cond, actions_for_grad)
                values = self._reduce_critic_values(
                    q_values,
                    use_min=optimize_critic_ensemble_min,
                )
                gradients = torch.autograd.grad(
                    values.sum(),
                    actions_for_grad,
                    only_inputs=True,
                )[0]

            if keep_action_with_max_value:
                improved = values.detach() >= best_values
                best_values = torch.where(improved, values.detach(), best_values)
                best_actions = torch.where(
                    improved[:, None, None],
                    actions_for_grad.detach(),
                    best_actions,
                )
                if collect_metrics:
                    best_indices = torch.where(
                        improved,
                        torch.full_like(best_indices, step),
                        best_indices,
                    )

            if collect_metrics and step == num_steps - 1:
                last_gradient_norm = gradients.reshape(B * K, -1).norm(dim=1).mean().detach()
            flat_actions = actions_for_grad.detach() + float(step_size) * gradients.detach()
            flat_actions = torch.clamp(
                flat_actions,
                min=self.cql_action_low,
                max=self.cql_action_high,
            )

        with torch.no_grad():
            q_values, _ = self._critic_output(critic, repeated_cond, flat_actions)
            final_values = self._reduce_critic_values(
                q_values,
                use_min=optimize_critic_ensemble_min,
            )
            if keep_action_with_max_value:
                improved = final_values >= best_values
                best_values = torch.where(improved, final_values, best_values)
                best_actions = torch.where(
                    improved[:, None, None],
                    flat_actions,
                    best_actions,
                )
                if collect_metrics:
                    best_indices = torch.where(
                        improved,
                        torch.full_like(best_indices, num_steps),
                        best_indices,
                    )
                optimized_actions = best_actions
                optimized_values = best_values
            else:
                optimized_actions = flat_actions
                optimized_values = final_values

        results = {
            "actions": optimized_actions.reshape(B, K, H, A).detach(),
            "values": optimized_values.reshape(B, K).detach(),
        }
        if collect_metrics:
            results.update(
                {
                    "last_gradient_norm": last_gradient_norm.detach(),
                    "action_with_max_value_index": best_indices.reshape(B, K).detach(),
                }
            )
        return results

    def _optimize_policy_actions(
        self,
        cond,
        candidate_actions,
        num_actions_to_keep,
        local_optimization_steps,
        local_optimization_step_size,
        optimize_critic_ensemble_min=False,
        use_target_critic=False,
        dataset_actions_to_consider=None,
        global_rank_use_min_q=False,
        action_steps=None,
        collect_metrics=True,
        metrics_as_tensors=False,
    ):
        if candidate_actions.ndim != 4:
            raise ValueError("candidate_actions must have shape (B, K, horizon_steps, action_dim)")
        B, K, H, A = candidate_actions.shape
        if action_steps is None:
            action_steps = H
        if H != self.horizon_steps or A != self.action_dim:
            raise ValueError(
                f"candidate shape {(H, A)} does not match model "
                f"{(self.horizon_steps, self.action_dim)}"
            )
        if action_steps != self.horizon_steps:
            raise ValueError(
                "PA-RL action optimization currently requires act_steps == "
                "horizon_steps so optimized chunks are valid diffusion targets."
            )

        critic_cond = self._repeat_cond(cond, K)
        critic_actions = candidate_actions[:, :, :action_steps].reshape(
            B * K,
            action_steps,
            A,
        )
        with torch.no_grad():
            critic_q_values, _ = self._critic_output(
                self.critic,
                critic_cond,
                critic_actions,
            )
            # Official PA-RL ranks global candidates by ensemble mean. Keeping
            # this configurable lets us reproduce earlier conservative runs.
            q_values = self._reduce_critic_values(
                critic_q_values,
                use_min=global_rank_use_min_q,
            ).view(B, K)

            keep = min(int(num_actions_to_keep), K)
            if keep <= 0:
                raise ValueError("num_actions_to_keep must be positive")
            top_values, top_indices = torch.topk(q_values, keep, dim=1, sorted=False)
            sorted_order = torch.argsort(top_values, dim=1)
            top_indices = torch.gather(top_indices, dim=1, index=sorted_order)
            top_values = torch.gather(top_values, dim=1, index=sorted_order)
            gather_indices = top_indices[:, :, None, None].expand(B, keep, H, A)
            top_actions = torch.gather(candidate_actions, dim=1, index=gather_indices)

            used_replay_action = 0
            replay_q = None
            if dataset_actions_to_consider is not None:
                if dataset_actions_to_consider.shape[1:] != (
                    self.horizon_steps,
                    self.action_dim,
                ):
                    raise ValueError(
                        "dataset_actions_to_consider must match (horizon_steps, action_dim)."
                    )
                # Official action_optimization_sample_actions drops the worst
                # globally kept action and appends the dataset action.
                top_actions = torch.cat(
                    [top_actions[:, 1:], dataset_actions_to_consider[:, None]],
                    dim=1,
                )
                replay_q_values, _ = self._critic_output(
                    self.critic,
                    cond,
                    dataset_actions_to_consider,
                )
                replay_q = self._reduce_critic_values(
                    replay_q_values,
                    use_min=global_rank_use_min_q,
                )
                top_values = torch.cat([top_values[:, 1:], replay_q[:, None]], dim=1)
                used_replay_action = 1

        local_results = self._local_optimization_steps(
            cond,
            top_actions,
            num_steps=local_optimization_steps,
            step_size=local_optimization_step_size,
            optimize_critic_ensemble_min=optimize_critic_ensemble_min,
            use_target_critic=use_target_critic,
            keep_action_with_max_value=True,
            collect_metrics=collect_metrics,
        )
        optimized_actions = local_results["actions"]
        optimized_values_for_grad = local_results["values"]

        if not use_target_critic and not optimize_critic_ensemble_min:
            eval_values = optimized_values_for_grad
        else:
            with torch.no_grad():
                eval_cond = self._repeat_cond(cond, optimized_actions.shape[1])
                eval_actions = optimized_actions.reshape(
                    B * optimized_actions.shape[1],
                    H,
                    A,
                )
                eval_q_values, _ = self._critic_output(self.critic, eval_cond, eval_actions)
                eval_values = self._reduce_critic_values(
                    eval_q_values,
                    use_min=False,
                ).view(B, optimized_actions.shape[1])

        if not collect_metrics:
            return optimized_actions.detach(), eval_values.detach(), {}

        with torch.no_grad():
            base_l1 = torch.sum(torch.abs(top_actions), dim=(-2, -1))
            opt_l1 = torch.sum(torch.abs(optimized_actions), dim=(-2, -1))
            action_delta_l1 = torch.sum(
                torch.abs(optimized_actions - top_actions),
                dim=(-2, -1),
            )

        metrics = {
            "parl/base_policy_actions_l1_norm": base_l1.mean().detach(),
            "parl/base_policy_q_mean": q_values.mean().detach(),
            "parl/base_policy_top_q_mean": top_values.mean().detach(),
            "parl/used_replay_action": used_replay_action,
            "parl/local_gradient_norm": local_results["last_gradient_norm"],
            "parl/local_optimization_actions_l1_norm": opt_l1.mean().detach(),
            "parl/q_values_after_steps": eval_values.mean().detach(),
            "parl/total_value_increase": (eval_values - top_values).mean().detach(),
            "parl/l1_q_diff_actions_base_policy_actions": (action_delta_l1.mean().detach()),
            "parl/index_of_max_value_action": (
                local_results["action_with_max_value_index"].float().mean()
            ),
            "parl/local_optimization_steps": float(local_optimization_steps),
            "parl/local_step_size": float(local_optimization_step_size),
            "parl/optimized_value_for_gradient_mean": (optimized_values_for_grad.mean().detach()),
        }
        if replay_q is not None:
            metrics["parl/replay_action_q_mean"] = replay_q.mean().detach()
        if not metrics_as_tensors:
            metrics = self._metrics_to_python(metrics)
        return optimized_actions.detach(), eval_values.detach(), metrics

    def _sample_policy_cql_actions(self, cond, num_actions, action_steps, argmax=None):
        # CQL only trains the critic here; policy samples are treated as action
        # proposals and must not route gradients into the diffusion actor.
        if not self.cql_use_parl_policy_actions:
            with torch.no_grad():
                return self.sample_candidate_actions(
                    cond,
                    num_base_policy_actions=num_actions,
                )[:, :, :action_steps]

        B = len(cond["state"])
        with torch.no_grad():
            candidates = self.sample_candidate_actions(
                cond,
                num_base_policy_actions=self.cql_num_base_policy_actions,
            )
        optimized_actions, logits, _ = self._optimize_policy_actions(
            cond,
            candidates,
            num_actions_to_keep=self.cql_num_actions_to_keep,
            local_optimization_steps=self.cql_local_optimization_steps,
            local_optimization_step_size=self.cql_local_optimization_step_size,
            optimize_critic_ensemble_min=self.cql_optimize_critic_ensemble_min,
            use_target_critic=self.cql_use_target_critic_for_optimization,
            dataset_actions_to_consider=None,
            global_rank_use_min_q=self.cql_policy_use_min_q,
            action_steps=action_steps,
            collect_metrics=False,
        )
        argmax = self.cql_policy_argmax if argmax is None else bool(argmax)
        if argmax:
            best_indices = torch.argmax(logits, dim=1)
            sampled_indices = best_indices[:, None].expand(B, num_actions)
        else:
            categorical = torch.distributions.Categorical(
                logits=logits / max(self.cql_policy_temperature, 1e-6)
            )
            sampled_indices = categorical.sample((num_actions,)).transpose(0, 1)
        sampled_actions = torch.gather(
            optimized_actions,
            dim=1,
            index=sampled_indices[:, :, None, None].expand(
                B,
                num_actions,
                optimized_actions.shape[2],
                optimized_actions.shape[3],
            ),
        )
        return sampled_actions[:, :, :action_steps]

    def loss_cql(
        self,
        obs,
        next_obs,
        replay_actions,
        current_q_values,
        mc_returns=None,
        collect_metrics=True,
    ):
        if self.cql_importance_sample:
            raise NotImplementedError(
                "PARL CQL currently implements the official "
                "cql_importance_sample=False path used by BASE_PARL_CALQL_CONFIG."
            )
        if self.use_calql and mc_returns is None:
            raise ValueError("use_calql=True requires mc_returns in the replay batch.")
        if self.cql_n_actions <= 0:
            raise ValueError("cql_n_actions must be positive when cql_alpha > 0")
        if self.cql_temp <= 0:
            raise ValueError("cql_temp must be positive")

        B = replay_actions.shape[0]
        action_steps = replay_actions.shape[1]
        sampled_action_blocks = []
        block_names = []
        calql_bound_block_mask = []

        if self.cql_include_random_actions:
            sampled_action_blocks.append(
                self._sample_random_cql_actions(
                    batch_size=B,
                    num_actions=self.cql_n_actions,
                    action_steps=action_steps,
                )
            )
            block_names.append("random")
            calql_bound_block_mask.append(self.use_calql_on_random_actions)
        if self.cql_include_next_actions:
            sampled_action_blocks.append(
                self._sample_policy_cql_actions(
                    next_obs,
                    num_actions=self.cql_n_actions,
                    action_steps=action_steps,
                )
            )
            block_names.append("next_policy")
            calql_bound_block_mask.append(True)
        if self.cql_include_current_actions:
            sampled_action_blocks.append(
                self._sample_policy_cql_actions(
                    obs,
                    num_actions=self.cql_n_actions,
                    action_steps=action_steps,
                )
            )
            block_names.append("current_policy")
            calql_bound_block_mask.append(True)
        if not sampled_action_blocks and not self.cql_include_data_action:
            raise ValueError("CQL needs at least one sampled action or the data action")

        q_chunks = []
        metrics = {}
        calql_bound_rate = None
        if sampled_action_blocks:
            sampled_actions = torch.cat(sampled_action_blocks, dim=1)
            K = sampled_actions.shape[1]
            repeated_obs = self._repeat_cond(obs, K)
            critic_actions = sampled_actions.reshape(
                B * K,
                action_steps,
                self.action_dim,
            )
            sampled_q_values, _ = self._critic_output(
                self.critic,
                repeated_obs,
                critic_actions,
            )
            sampled_q_values = sampled_q_values.view(
                sampled_q_values.shape[0],
                B,
                K,
            )

            start = 0
            for block_name, block_actions in zip(block_names, sampled_action_blocks):
                end = start + block_actions.shape[1]
                block_q = sampled_q_values[:, :, start:end]
                if collect_metrics:
                    metrics[f"critic/cql_{block_name}_q_mean"] = block_q.detach().mean()
                start = end

            if self.use_calql:
                lower_bound_parts = []
                start = 0
                for block_actions, should_bound in zip(
                    sampled_action_blocks,
                    calql_bound_block_mask,
                ):
                    end = start + block_actions.shape[1]
                    if should_bound:
                        lower_bound_parts.append((start, end))
                    start = end
                if lower_bound_parts:
                    mc_lower_bound = mc_returns.view(1, B, 1)
                    bounded_chunks = []
                    total_bound_events = 0.0
                    total_bound_count = 0
                    start = 0
                    for block_actions, should_bound in zip(
                        sampled_action_blocks,
                        calql_bound_block_mask,
                    ):
                        end = start + block_actions.shape[1]
                        q_slice = sampled_q_values[:, :, start:end]
                        if should_bound:
                            if collect_metrics:
                                total_bound_events = (
                                    total_bound_events + (q_slice < mc_lower_bound).float().sum()
                                )
                                total_bound_count += q_slice.numel()
                            q_slice = torch.maximum(q_slice, mc_lower_bound)
                        bounded_chunks.append(q_slice)
                        start = end
                    if collect_metrics and total_bound_count > 0:
                        calql_bound_rate = (total_bound_events / float(total_bound_count)).detach()
                    sampled_q_values = torch.cat(bounded_chunks, dim=-1)
            q_chunks.append(sampled_q_values)

        if self.cql_include_data_action:
            q_chunks.append(current_q_values.unsqueeze(-1))
            if collect_metrics:
                metrics["critic/cql_data_q_mean"] = current_q_values.detach().mean()

        cql_q_values = torch.cat(q_chunks, dim=-1)
        cql_q_values, current_q_for_cql = self._subsample_critic_values(
            cql_q_values,
            current_q_values,
        )
        num_values = cql_q_values.shape[-1]
        log_num_values = math.log(float(num_values))

        cql_ood_values = (
            torch.logsumexp(cql_q_values / self.cql_temp, dim=-1) * self.cql_temp
            - log_num_values * self.cql_temp
        )
        cql_diff = cql_ood_values - current_q_for_cql
        if self.cql_clip_diff_min is not None or self.cql_clip_diff_max is not None:
            cql_diff = torch.clamp(
                cql_diff,
                min=self.cql_clip_diff_min,
                max=self.cql_clip_diff_max,
            )
        cql_loss = cql_diff.mean()

        if collect_metrics:
            metrics.update(
                {
                    "critic/cql_loss": cql_loss.detach(),
                    "critic/cql_diff": cql_diff.detach().mean(),
                    "critic/cql_ood_q_mean": cql_ood_values.detach().mean(),
                    "critic/cql_num_sampled_values": float(num_values),
                }
            )
            if calql_bound_rate is not None:
                metrics["critic/calql_bound_rate"] = calql_bound_rate
        return cql_loss, metrics

    def _repeat_cond(self, cond, repeat):
        repeated = {}
        for key, value in cond.items():
            B = value.shape[0]
            repeat_shape = (1, repeat) + tuple(1 for _ in value.shape[1:])
            repeated[key] = (
                value[:, None].repeat(*repeat_shape).reshape(B * repeat, *value.shape[1:])
            )
        return repeated

    @torch.no_grad()
    def sample_candidate_actions(self, cond, num_base_policy_actions):
        """Sample K action chunks from the current/base diffusion policy."""
        B = len(cond["state"])
        repeated_cond = self._repeat_cond(cond, num_base_policy_actions)
        actions = self.forward(
            cond=repeated_cond,
            deterministic=False,
        )
        return actions.view(
            B,
            num_base_policy_actions,
            self.horizon_steps,
            self.action_dim,
        )

    @torch.no_grad()
    def rank_candidate_actions(
        self,
        cond,
        candidate_actions,
        act_steps,
        replay_actions=None,
        num_actions_to_keep=10,
        distill_argmax=False,
        softmax_temperature=1.0,
        consider_replay_action=True,
        use_min_q=True,
        local_optimization_steps=10,
        local_optimization_step_size=3e-4,
        optimize_critic_ensemble_min=False,
        use_target_critic=False,
    ):
        """Official PA-RL action search: global top-k, local optimize, sample."""
        optimized_actions, top_values, metrics = self._optimize_policy_actions(
            cond,
            candidate_actions,
            num_actions_to_keep=num_actions_to_keep,
            local_optimization_steps=local_optimization_steps,
            local_optimization_step_size=local_optimization_step_size,
            optimize_critic_ensemble_min=optimize_critic_ensemble_min,
            use_target_critic=use_target_critic,
            dataset_actions_to_consider=(
                replay_actions if consider_replay_action and replay_actions is not None else None
            ),
            global_rank_use_min_q=use_min_q,
            action_steps=act_steps,
            metrics_as_tensors=True,
        )
        B, keep, H, A = optimized_actions.shape

        if distill_argmax:
            best_indices = torch.argmax(top_values, dim=1)
            distill_actions = torch.gather(
                optimized_actions,
                dim=1,
                index=best_indices[:, None, None, None].expand(B, 1, H, A),
            ).squeeze(1)
            entropy_value = torch.zeros((), device=top_values.device)
        else:
            temperature = max(float(softmax_temperature), 1e-6)
            categorical = torch.distributions.Categorical(logits=top_values / temperature)
            sampled_indices = categorical.sample()
            distill_actions = torch.gather(
                optimized_actions,
                dim=1,
                index=sampled_indices[:, None, None, None].expand(B, 1, H, A),
            ).squeeze(1)
            entropy_value = categorical.entropy().mean().detach()

        if replay_actions is not None:
            target_replay_l1 = torch.sum(
                torch.abs(distill_actions - replay_actions),
                dim=(-2, -1),
            )
            target_replay_l1_mean = target_replay_l1.mean().detach()

        metrics.update(
            {
                "parl/top_q_mean": top_values.mean().detach(),
                "parl/top_q_max": top_values.max(dim=1).values.mean().detach(),
                "parl/top_q_min": top_values.min(dim=1).values.mean().detach(),
                "parl/categorical_entropy": entropy_value,
                "parl/categorical_logits_min": top_values.min(dim=1).values.mean(),
                "parl/categorical_logits_max": top_values.max(dim=1).values.mean(),
                "parl/categorical_logits_mean": top_values.mean(),
                "parl/categorical_logits_std": (top_values.std(dim=1, unbiased=False).mean()),
            }
        )
        if replay_actions is not None:
            metrics["parl/l1_target_action_replay_action"] = target_replay_l1_mean
        return distill_actions.detach(), self._metrics_to_python(metrics)

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
        return actor_loss, metrics

    @torch.no_grad()
    def sample_optimized_actions(
        self,
        cond,
        act_steps,
        num_base_policy_actions=None,
        num_actions_to_keep=None,
        argmax=True,
        softmax_temperature=1.0,
        use_min_q=False,
        local_optimization_steps=None,
        local_optimization_step_size=None,
        optimize_critic_ensemble_min=False,
        use_target_critic=False,
    ):
        """Sample the deployed PA-RL policy by optimizing base-policy actions."""
        num_base_policy_actions = (
            self.cql_num_base_policy_actions
            if num_base_policy_actions is None
            else int(num_base_policy_actions)
        )
        num_actions_to_keep = (
            self.cql_num_actions_to_keep
            if num_actions_to_keep is None
            else int(num_actions_to_keep)
        )
        local_optimization_steps = (
            self.cql_local_optimization_steps
            if local_optimization_steps is None
            else int(local_optimization_steps)
        )
        local_optimization_step_size = (
            self.cql_local_optimization_step_size
            if local_optimization_step_size is None
            else float(local_optimization_step_size)
        )
        candidate_actions = self.sample_candidate_actions(
            cond,
            num_base_policy_actions=num_base_policy_actions,
        )
        actions, metrics = self.rank_candidate_actions(
            cond,
            candidate_actions,
            act_steps=act_steps,
            replay_actions=None,
            num_actions_to_keep=num_actions_to_keep,
            distill_argmax=argmax,
            softmax_temperature=softmax_temperature,
            consider_replay_action=False,
            use_min_q=use_min_q,
            local_optimization_steps=local_optimization_steps,
            local_optimization_step_size=local_optimization_step_size,
            optimize_critic_ensemble_min=optimize_critic_ensemble_min,
            use_target_critic=use_target_critic,
        )
        metrics["parl/optimized_policy_argmax"] = int(bool(argmax))
        return actions[:, :act_steps].detach(), metrics

    def loss_actor(self, *args, **kwargs):
        raise RuntimeError(
            "PARLDQLDiffusion does not use DQL's Q-gradient actor loss. "
            "Use loss_actor_distill for PA-RL actor updates."
        )

    def update_target_critic(self, tau):
        with torch.no_grad():
            for target_param, source_param in zip(
                self.critic_target.parameters(), self.critic.parameters()
            ):
                target_param.lerp_(source_param, float(tau))

    # ---------- Sampling ----------#``

    # override
    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
    ):
        device = self.betas.device
        B = len(cond["state"])

        # Loop
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
            )
            std = torch.exp(0.5 * logvar)

            # Determine the noise level
            if self.use_ddim:
                if deterministic:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=self.min_sampling_denoising_std)
            else:
                if deterministic and t == 0:
                    std = torch.zeros_like(std)
                elif deterministic:
                    std = torch.clip(std, min=1e-3)
                else:
                    std = torch.clip(std, min=self.min_sampling_denoising_std)
            noise = torch.randn_like(x).clamp_(-self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(x, -self.final_action_clip_value, self.final_action_clip_value)
        return x

    def forward_train(
        self,
        cond,
        deterministic=False,
    ):
        """
        Differentiable forward pass used in actor training.
        """
        device = self.betas.device
        B = len(cond["state"])

        # Loop
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
            )
            std = torch.exp(0.5 * logvar)

            # Determine the noise level
            if self.use_ddim:
                if deterministic:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=self.min_sampling_denoising_std)
            else:
                if deterministic and t == 0:
                    std = torch.zeros_like(std)
                elif deterministic:  # For DDPM, sample with noise
                    std = torch.clip(std, min=1e-3)
                else:
                    std = torch.clip(std, min=self.min_sampling_denoising_std)
            noise = torch.randn_like(x).clamp_(-self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value and i == len(t_all) - 1:
                x = torch.clamp(x, -1, 1)
        return x
