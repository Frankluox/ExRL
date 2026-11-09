import os
import warnings

warnings.filterwarnings("ignore")
import math
import torch
import random
import numpy as np
import hydra
from omegaconf import OmegaConf

try:
    import wandb
except Exception:
    wandb = None
import gym

try:
    import d4rl  # noqa: F401
except Exception:
    d4rl = None
# Optional: gym-mujoco dependencies are not required for robomimic tasks.
# Keep import best-effort so robomimic runs do not crash when mujoco_py is absent.
try:
    import d4rl.gym_mujoco  # noqa: F401
except Exception:
    pass
import sys


from stable_baselines3 import SAC, DSRL
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from baselines.dsrl.environment import (
    DiffusionPolicyEnvWrapper,
    ObservationWrapperRobomimic,
    ObservationWrapperGym,
    ActionChunkWrapper,
    make_robomimic_env,
)
from baselines.dsrl.utils import (
    load_base_policy,
    load_offline_data,
    collect_rollouts,
    LoggingCallback,
)
from baselines.dsrl.checkpoint import (
    AtomicLatestCheckpointCallback,
    DEFAULT_INTERVAL_ENV_STEPS,
    load_training_checkpoint,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
OmegaConf.register_new_resolver("round_down", math.floor, replace=True)

base_path = os.path.dirname(os.path.abspath(__file__))


@hydra.main(
    config_path=os.path.join(base_path, "cfg/robomimic"),
    config_name="dsrl_can.yaml",
    version_base=None,
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.use_wandb:
        if wandb is None:
            raise ImportError("use_wandb=True but wandb is not installed")
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.name,
            group=cfg.wandb.group,
            monitor_gym=True,
            save_code=True,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    diffusion_horizon = int(getattr(cfg, "diffusion_horizon_steps", cfg.act_steps))
    rl_action_horizon = int(cfg.act_steps)
    execute_steps = int(getattr(cfg, "execute_steps", cfg.act_steps))
    execute_steps = max(1, min(execute_steps, diffusion_horizon))
    MAX_STEPS = int(math.ceil(cfg.env.max_episode_steps / execute_steps))

    num_env = cfg.env.n_envs

    def make_env():
        if cfg.env_name in ["halfcheetah-medium-v2", "hopper-medium-v2", "walker2d-medium-v2"]:
            env = gym.make(cfg.env_name)
            env = ObservationWrapperGym(env, cfg.normalization_path)
        elif cfg.env_name in ["lift", "can", "square", "transport", "tool_hang"]:
            env = make_robomimic_env(
                env=cfg.env_name,
                normalization_path=cfg.normalization_path,
                low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys,
                dppo_path=cfg.dppo_path,
            )
            env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
        env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
        return env

    base_policy = load_base_policy(cfg)
    env = make_vec_env(make_env, n_envs=num_env, vec_env_cls=SubprocVecEnv)
    if cfg.algorithm == "dsrl_sac":
        env = DiffusionPolicyEnvWrapper(env, cfg, base_policy)
    env.seed(cfg.seed + 1)
    post_linear_modules = None
    if cfg.train.use_layer_norm:
        post_linear_modules = [torch.nn.LayerNorm]

    net_arch = []
    for _ in range(cfg.train.num_layers):
        net_arch.append(cfg.train.layer_size)
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=torch.nn.Tanh,
        log_std_init=0.0,
        post_linear_modules=post_linear_modules,
        n_critics=cfg.train.n_critics,
    )
    if cfg.algorithm == "dsrl_sac":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=cfg.train.actor_lr,
            buffer_size=int(OmegaConf.select(cfg, "train.buffer_size", default=20000000)),
            learning_starts=1,  # How many steps before learning starts (total steps for all env combined)
            batch_size=cfg.train.batch_size,
            tau=cfg.train.tau,  # Target network update rate
            gamma=cfg.train.discount,  # Discount factor
            train_freq=cfg.train.train_freq,  # Update the model every train_freq steps
            gradient_steps=cfg.train.utd,  # How many gradient steps to do at each update
            action_noise=None,  # No additional action noise
            optimize_memory_usage=False,
            ent_coef="auto"
            if cfg.train.ent_coef == -1
            else cfg.train.ent_coef,  # Automatic entropy tuning
            target_update_interval=1,  # Update target network every interval
            target_entropy="auto"
            if cfg.train.target_ent == -1
            else cfg.train.target_ent,  # Automatic target entropy
            use_sde=False,
            sde_sample_freq=-1,
            tensorboard_log=cfg.logdir,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )
    elif cfg.algorithm == "dsrl_na":
        model = DSRL(
            "MlpPolicy",
            env,
            learning_rate=cfg.train.actor_lr,
            buffer_size=int(OmegaConf.select(cfg, "train.buffer_size", default=10000000)),
            learning_starts=1,  # How many steps before learning starts (total steps for all env combined)
            batch_size=cfg.train.batch_size,
            tau=cfg.train.tau,  # Target network update rate
            gamma=cfg.train.discount,  # Discount factor
            train_freq=cfg.train.train_freq,  # Update the model every train_freq steps
            gradient_steps=cfg.train.utd,  # How many gradient steps to do at each update
            action_noise=None,  # No additional action noise
            optimize_memory_usage=False,
            ent_coef="auto"
            if cfg.train.ent_coef == -1
            else cfg.train.ent_coef,  # Automatic entropy tuning
            target_update_interval=1,  # Update target network every interval
            target_entropy="auto"
            if cfg.train.target_ent == -1
            else cfg.train.target_ent,  # Automatic target entropy
            use_sde=False,
            sde_sample_freq=-1,
            tensorboard_log=cfg.logdir,
            verbose=1,
            policy_kwargs=policy_kwargs,
            diffusion_policy=base_policy,
            diffusion_act_dim=(rl_action_horizon, cfg.action_dim),
            noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
            critic_backup_combine_type=cfg.train.critic_backup_combine_type,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.save_model_interval,
        save_path=cfg.logdir + "/checkpoint/",
        name_prefix="ft_policy",
        save_replay_buffer=cfg.save_replay_buffer,
        save_vecnormalize=True,
    )

    num_env_eval = cfg.env.n_eval_envs
    online_eval_interval_env_steps = int(
        OmegaConf.select(cfg, "online_eval.interval_env_steps", default=30000)
    )
    online_eval_total_episodes = int(
        OmegaConf.select(cfg, "online_eval.total_episodes", default=50)
    )
    online_eval_async = bool(OmegaConf.select(cfg, "online_eval.async_eval", default=True))
    eval_env = None
    if not online_eval_async:
        eval_env = make_vec_env(make_env, n_envs=num_env_eval, vec_env_cls=SubprocVecEnv)
        if cfg.algorithm == "dsrl_sac":
            eval_env = DiffusionPolicyEnvWrapper(eval_env, cfg, base_policy)
        eval_env.seed(cfg.seed + num_env + 1)

    logging_callback = LoggingCallback(
        action_chunk=execute_steps,
        log_freq=MAX_STEPS,
        use_wandb=cfg.use_wandb,
        eval_env=eval_env,
        num_train_env=num_env,
        num_eval_env=num_env_eval,
        rew_offset=cfg.env.reward_offset,
        algorithm=cfg.algorithm,
        max_steps=MAX_STEPS,
        online_eval_interval_env_steps=online_eval_interval_env_steps,
        online_eval_total_episodes=online_eval_total_episodes,
        online_eval_async=online_eval_async,
        cfg=cfg,
        tb_log_dir=cfg.logdir,
    )

    total_timesteps_cfg = OmegaConf.select(cfg, "total_timesteps", default=None)
    configured_train_policy_steps = (
        int(total_timesteps_cfg) if total_timesteps_cfg is not None else 20000000
    )
    latest_checkpoint_interval = int(
        OmegaConf.select(
            cfg,
            "latest_checkpoint.interval_env_steps",
            default=DEFAULT_INTERVAL_ENV_STEPS,
        )
    )
    latest_checkpoint_path = OmegaConf.select(
        cfg,
        "latest_checkpoint.path",
        default=os.path.join(cfg.logdir, "checkpoint", "latest.ckpt"),
    )
    training_state = {
        "warmup_policy_steps": 0,
        "warmup_env_steps": 0,
        "configured_train_policy_steps": configured_train_policy_steps,
    }
    latest_checkpoint_callback = AtomicLatestCheckpointCallback(
        path=latest_checkpoint_path,
        logging_callback=logging_callback,
        numbered_checkpoint_callback=checkpoint_callback,
        cfg=cfg,
        training_state=training_state,
        interval_env_steps=latest_checkpoint_interval,
    )
    resume_checkpoint_path = OmegaConf.select(
        cfg,
        "resume.path",
        default=OmegaConf.select(cfg, "baselines.dsrl.checkpoint", default=None),
    )
    allow_resume_config_mismatch = bool(
        OmegaConf.select(cfg, "resume.allow_config_mismatch", default=False)
    )
    resuming = resume_checkpoint_path is not None and str(resume_checkpoint_path).strip() != ""

    warmup_policy_steps = 0
    warmup_env_steps = 0
    if resuming:
        training_state, resume_manifest = load_training_checkpoint(
            resume_checkpoint_path,
            model,
            logging_callback,
            checkpoint_callback,
            latest_checkpoint_callback,
            cfg,
            allow_config_mismatch=allow_resume_config_mismatch,
        )
        latest_checkpoint_callback.training_state = training_state
        warmup_policy_steps = int(training_state["warmup_policy_steps"])
        warmup_env_steps = int(training_state["warmup_env_steps"])
        print(
            f"[resume] loaded path={resume_checkpoint_path} "
            f"policy_step={model.num_timesteps} env_step={model.env_steps_exact} "
            f"replay_rows={resume_manifest['replay']['size']}",
            flush=True,
        )
    else:
        print(
            f"[phase] launching online-eval stochastic async={online_eval_async} "
            f"eval_envs={num_env_eval} total_episodes={online_eval_total_episodes} "
            f"env_step=0 max_steps={MAX_STEPS}",
            flush=True,
        )
        logging_callback.run_online_eval(model, env_step=0)
        print("[phase] online-eval request submitted", flush=True)
        logging_callback.log_count += 1

        if cfg.load_offline_data:
            load_offline_data(model, cfg.offline_data_path, num_env)
        if cfg.train.init_rollout_steps > 0:
            print(
                f"[phase] starting init-rollout policy_steps={cfg.train.init_rollout_steps} "
                f"envs={num_env} act_steps={cfg.act_steps} execute_steps={execute_steps}",
                flush=True,
            )
            warmup_stats = collect_rollouts(
                model, env, cfg.train.init_rollout_steps, base_policy, cfg
            )
            warmup_policy_steps = int(warmup_stats["policy_steps"])
            warmup_env_steps = int(warmup_stats["env_steps_exact"])
            print(
                f"[phase] finished init-rollout policy_steps={warmup_policy_steps} "
                f"env_steps={warmup_env_steps}",
                flush=True,
            )
            logging_callback.set_timesteps(warmup_env_steps, reset_online_eval_schedule=True)
            model.num_timesteps = warmup_policy_steps
            model.env_steps_exact = warmup_env_steps
        training_state["warmup_policy_steps"] = warmup_policy_steps
        training_state["warmup_env_steps"] = warmup_env_steps

    training_state["configured_train_policy_steps"] = configured_train_policy_steps
    latest_checkpoint_callback.training_state = training_state
    callbacks = [checkpoint_callback, logging_callback, latest_checkpoint_callback]
    primitive_budget = int(OmegaConf.select(cfg, "train.max_env_steps", default=0))
    if primitive_budget > 0:
        from baselines.dsrl.budget import PrimitiveEnvBudgetCallback

        callbacks.append(
            PrimitiveEnvBudgetCallback(
                primitive_budget,
                latest_checkpoint_callback,
                logging_callback,
                os.path.join(cfg.logdir, "final_status.json"),
            )
        )
    target_policy_step = warmup_policy_steps + configured_train_policy_steps
    remaining_policy_steps = max(0, target_policy_step - int(model.num_timesteps))
    if primitive_budget > 0 and int(model.env_steps_exact) >= primitive_budget:
        # A completed full-state resume only needs final saving/evaluation. Never
        # collect another rollout just to make the budget callback observe it.
        remaining_policy_steps = 0
    print(
        f"[train-budget] current_policy_step={model.num_timesteps} "
        f"target_policy_step={target_policy_step} remaining_policy_steps={remaining_policy_steps}",
        flush=True,
    )
    # Train the agent
    model.learn(
        total_timesteps=remaining_policy_steps,
        callback=callbacks,
        reset_num_timesteps=False,
    )

    # Save the final model
    if len(cfg.name) > 0:
        model.save(cfg.logdir + "/checkpoint/final")

    # Close environment and wandb
    env.close()
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
