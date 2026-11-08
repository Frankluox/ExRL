"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.models.diffusion_policy_nets as DPNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo, algo_factory

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

import sys
sys.path.append("assets/local") # 确保能找到模块
from f_control_as_a_function import ActionScheduler
from OOD_detect_as_a_function import OODMonitor

import numpy as np

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@register_algo_factory_func("diffusion_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.unet.enabled:
        return DiffusionPolicyUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


class DiffusionPolicyUNet(PolicyAlgo):
    def __init__(self, algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device, **kwargs):
        """
        构造函数。参数名必须与 algo_factory 传递的关键字一致。
        """
        # 调用父类初始化。注意这里必须是 obs_key_shapes
        super().__init__(
            algo_config=algo_config, 
            obs_config=obs_config, 
            global_config=global_config, 
            obs_key_shapes=obs_key_shapes, 
            ac_dim=ac_dim, 
            device=device
        )

        # [新增] 初始化调度器和日志变量
        try:
            # 确保 path 能找到你写的那个文件
            # 初始化调度器
            self.action_scheduler = ActionScheduler(safety_lambda=1.0, power_threshold=0.95)
            self.ood_monitor = OODMonitor(
                    max_prediction_len=self.algo_config.horizon.action_horizon, # 模型输出长度 (T)
                    max_ood_bound=self.algo_config.horizon.action_horizon # OOD 严重时的强制执行长度
                )
            print(self.ood_monitor.max_len)
            print("Successfully initialized ActionScheduler and OODMonitor in DiffusionPolicyUNet.")
        except Exception as e:
            print(f"Warning: Could not initialize ActionScheduler or OODMonitor: {e}")
            self.action_scheduler = None
            self.ood_monitor = None
            raise e
            
        self.last_execution_info = None

        # 3. 状态管理变量
        self.action_queue = deque()      # 存储当前正在执行的动作片段
        self.last_plan_executed_len = 0  # 上一次规划实际执行了多少步 (传给 OODMonitor 用)
        self.current_raw_action = None   # 缓存当前的完整预测

        self.use_action_scheduler = False 
        self.use_ood_monitor = False

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "noise_pred_net": noise_pred_net
            })
        })

        nets = nets.float().to(self.device)
        
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)
                
        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
    
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"] if k != "pad_mask"}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]
        # Optional validity mask for padded sequence tail.
        # When enabled (train.get_pad_mask=True), this keeps padded tokens out of the loss.
        pad_mask = batch.get("pad_mask", None)
        if pad_mask is None and isinstance(batch.get("obs", None), dict):
            pad_mask = batch["obs"].get("pad_mask", None)
        if pad_mask is not None:
            input_batch["action_pad_mask"] = pad_mask[:, :Tp, :]
        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError("'actions' must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.")
            self.action_check_done = True
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch["actions"].shape[0]
        
        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]
            
            # encode obs
            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"]
            }
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
            
            obs_features = TensorUtils.time_distributed(inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]

            obs_cond = obs_features.flatten(start_dim=1)
            
            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=self.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=self.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            noise_pred = self.nets["policy"]["noise_pred_net"](
                noisy_actions, timesteps, global_cond=obs_cond)
            
            # L2 loss (optionally mask padded sequence tail)
            action_pad_mask = batch.get("action_pad_mask", None)
            if action_pad_mask is not None:
                # action_pad_mask: [B, Tp, 1] -> [B, Tp, D]
                valid = action_pad_mask.float().expand_as(noise_pred)
                sq_err = (noise_pred - noise) ** 2
                denom = torch.clamp(valid.sum(), min=1.0)
                loss = (sq_err * valid).sum() / denom
            else:
                loss = F.mse_loss(noise_pred, noise)
            
            # logging
            losses = {
                "l2_loss": loss
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)
                
                step_info = {
                    "policy_grad_norms": policy_grad_norms
                }
                info.update(step_info)

        return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue

        # 2. [新增] 重置自定义执行记录
        self.last_plan_executed_len = 0
        self.last_execution_info = None

        # 3. [新增] 重置 OODMonitor 内部缓存
        if hasattr(self, 'ood_monitor') and self.ood_monitor is not None:
            self.ood_monitor.prev_prediction_buffer = None
            self.ood_monitor.loss_history.clear()
            self.ood_monitor.current_ood_score = 0.0

        # [新增] 重置 ActionScheduler
        if self.action_scheduler is not None:
            self.action_scheduler.reset()
    

    
    def get_action(self, obs_dict, goal_dict=None):
        # print("Use?", self.use_action_scheduler, self.use_ood_monitor)
        if len(self.action_queue) == 0:
            # 1. 执行模型推理
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)
            action_sequence_2d = action_sequence.squeeze(0)
            action_sequence_np = action_sequence_2d.cpu().detach().numpy()

            # 2. 初始默认值（对应“都不开”的情况）
            # 默认执行长度为模型输出的全长
            final_len = action_sequence_np.shape[0]
            target_len = final_len
            lower_bound = 1
            ood_score = 0.0
            reason = "default_full_horizon"
            ood_metrics = {}

            # 3. [模式：只开 Action Scheduler 或 两个都开] 
            # 计算基于复杂度的截断长度（上限控制）
            # print(self.action_scheduler)
            if self.use_action_scheduler and self.action_scheduler is not None:
                _, _, target_len, reason = self.action_scheduler.step(action_sequence_np, None)
                final_len = target_len

                # print("length after scheduler",final_len)
            # 4. [模式：两个都开]
            # 计算基于 OOD 的执行下界（下界控制/平滑）
            if self.use_ood_monitor and self.ood_monitor is not None:
                ood_result = self.ood_monitor.step(
                    current_action_raw=action_sequence_np,
                    last_execution_len=self.last_plan_executed_len
                )
                lower_bound = ood_result["suggested_bound"]
                ood_score = ood_result["ood_score"]
                ood_metrics = ood_result["metrics"]
                
                # 融合逻辑：如果 OOD 严重，强制拉长执行步数以覆盖不稳定的过渡期
                # 如果 use_action_scheduler 为 False，final_len 默认为最大长度，max 不起作用
                final_len = max(final_len, lower_bound)

            # 5. 安全检查与状态更新
            final_len = int(np.clip(final_len, 1, action_sequence_np.shape[0]))
            self.last_plan_executed_len = final_len

            # 6. 填充动作队列
            chunk = action_sequence_2d[:final_len]
            self.action_queue.extend(chunk)

            # print("chunk_size", chunk.shape)

            # 7. 日志记录
            self.last_execution_info = {
                "final_len": final_len,
                "full_action_chunk": action_sequence_np,
                "use_sched": self.use_action_scheduler,
                "use_ood": self.use_ood_monitor,
                "ood_score": ood_score,
                "ood_suggested_bound": lower_bound,
                "reason": reason,
                "step_log": True
            }
            
            mode_str = f"Sched={self.use_action_scheduler}, OOD={self.use_ood_monitor}"
            print(f"[Policy Replan] {mode_str} | Final Len: {final_len} | Reason: {reason}")

        else:
            if self.last_execution_info is not None:
                self.last_execution_info["step_log"] = False
        # print("Action queue length:", len(self.action_queue))
        return self.action_queue.popleft().unsqueeze(0)


    # def get_action(self, obs_dict, goal_dict=None):
        # """
        # Get policy action outputs.

        # Args:
        #     obs_dict (dict): current observation [1, Do]
        #     goal_dict (dict): (optional) goal

        # Returns:
        #     action (torch.Tensor): action tensor [1, Da]
        # """
        # # obs_dict: key: [1,D]
        # To = self.algo_config.horizon.observation_horizon
        # Ta = self.algo_config.horizon.action_horizon

        # # [修改] 如果队列为空，进行推理并记录信息
        # if len(self.action_queue) == 0:
        #     # no actions left, run inference
        #     # [1,T,Da]
        #     action_sequence = self._get_action_trajectory(obs_dict=obs_dict)

        #     # [新增] --- 自适应截断逻辑开始 ---
        #     action_sequence_2d = action_sequence.squeeze(0) # [T, Da]
        #     action_sequence_np = action_sequence_2d.cpu().detach().numpy() # 转为 numpy

        #     # 调用调度器
        #     # 注意：这里传入 None 作为 prev_action，如果需要更精细的控制，
        #     # 你可能需要维护一个 prev_action buffer
        #     chunk, is_truncated, reason = self.action_scheduler.step(action_sequence_np, None)

        #     truncation_length = len(chunk)

        #     print(f"Action sequence length: {action_sequence_np.shape[0]}, Truncated length: {truncation_length}, Reason: {reason}")

        #     # 将截断后的 chunk 转回 tensor 并放入队列
        #     chunk_tensor = torch.as_tensor(chunk, dtype=action_sequence.dtype, device=action_sequence.device)
        #     self.action_queue.extend(chunk_tensor)

        #     # [新增] --- 记录日志 ---
        #     # 我们记录完整的原始输出、截断后的长度、以及原因
        #     self.last_execution_info = {
        #         "full_action_chunk": action_sequence_np,  # 原始完整输出
        #         "truncation_length": truncation_length,   # 实际执行长度
        #         "reason": reason,
        #         "step_log": True # 标记这是一个新的推理步
        #     }
        #     # import sys
        #     # sys.path.append("assets/local")
        #     # from f_control_as_a_function import ActionScheduler
        #     # action_scheduler = ActionScheduler(safety_lambda=3.3, power_threshold=0.95, dct_scale = 10.0, trend_ratio = 0.2, freq_smoothing_factor = 2.0)
        #     # action_sequence_2d = action_sequence.squeeze(0)
        #     # action_sequence_np = action_sequence_2d.cpu().numpy()
        #     # chunk, _, __ = action_scheduler.step(action_sequence_np, None)
        #     # chunk_tensor = torch.as_tensor(chunk, dtype=action_sequence.dtype)
        #     # chunk_tensor = chunk_tensor.to(action_sequence.device)
        #     # action_sequence = chunk_tensor.unsqueeze(0)
        #     # print(chunk.shape)


        #     # # put actions into the queue
        #     # self.action_queue.extend(action_sequence[0])
        # else:
        #     # [新增] 如果是从队列里取动作，标记这不是新的推理步
        #      if self.last_execution_info is not None:
        #          self.last_execution_info["step_log"] = False
        


        # # has action, execute from left to right
        # # [Da]
        # action = self.action_queue.popleft()
        
        # # [1,Da]
        # action = action.unsqueeze(0)
        # return action
        
    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # encode obs
        inputs = {
            "obs": obs_dict,
            "goal": goal_dict
        }
        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                # adding time dimension if not present -- this is required as
                # frame stacking is not invoked when sequence length is 1
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action

        # np.set_printoptions(suppress=True)

        # print("=================noise=============")
        # print(np.round(naction.cpu().numpy(), 4))
        # print("===============")
        
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # print("=================noise_pred=============")
            # print(np.round(noise_pred.cpu().numpy(), 4))
            # print("===============")

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # process action using Ta
        # Ta = 16
        start = To - 1
        end = start + Ta
        # print(Ta)
        action = naction[:,start:end]
        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "optimizers": { k : self.optimizers[k].state_dict() for k in self.optimizers },
            "lr_schedulers": { k : self.lr_schedulers[k].state_dict() if self.lr_schedulers[k] is not None else None for k in self.lr_schedulers },
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            load_optimizers (bool): whether to load optimizers and lr_schedulers from the model_dict;
                used when resuming training from a checkpoint
        """
        self.nets.load_state_dict(model_dict["nets"])

        # for backwards compatibility
        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if model_dict.get("ema", None) is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])

        if load_optimizers:
            for k in model_dict["optimizers"]:
                self.optimizers[k].load_state_dict(model_dict["optimizers"][k])
            for k in model_dict["lr_schedulers"]:
                if model_dict["lr_schedulers"][k] is not None:
                    self.lr_schedulers[k].load_state_dict(model_dict["lr_schedulers"][k])


def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module
