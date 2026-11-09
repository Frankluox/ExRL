import os
import warnings

warnings.filterwarnings("ignore")
import math
import json
import torch
import random
import numpy as np
import hydra
from omegaconf import OmegaConf
from gymnasium import spaces

try:
    import wandb
except Exception:
    wandb = None
import gym

try:
    import d4rl
except Exception:
    d4rl = None
# Optional: gym-mujoco dependencies are not required for robomimic tasks.
# Keep import best-effort so robomimic runs do not crash when mujoco_py is absent.
try:
    import d4rl.gym_mujoco  # noqa: F401
except Exception:
    pass
import sys


from stable_baselines3 import SAC, DSRL, ResidualSAC
from stable_baselines3.sac.policies import ActorConditionedMlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from baselines.policy_decorator.environment import (
    DiffusionPolicyEnvWrapper,
    ObservationWrapperRobomimic,
    ObservationWrapperGym,
    ActionChunkWrapper,
    make_robomimic_env,
)
from baselines.policy_decorator.utils import (
    load_base_policy,
    load_offline_data,
    collect_rollouts,
    LoggingCallback,
    EnvStepEvalCallback,
    PostRolloutCheckpointCallback,
    ResidualActorSnapshotCallback,
    StopTrainingOnEnvStepCallback,
    evaluate_residual_success_rate,
    publish_async_eval_reference,
    publish_latest_checkpoint,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
OmegaConf.register_new_resolver("round_down", math.floor, replace=True)

base_path = os.path.dirname(os.path.abspath(__file__))


def _cfg_path_or_none(path_value):
    if path_value is None:
        return None
    path_text = str(path_value).strip()
    if path_text == "" or path_text.lower() in {"none", "null"}:
        return None
    return hydra.utils.to_absolute_path(path_text)


def _resolve_checkpoint_path(path_value):
    path = _cfg_path_or_none(path_value)
    if path is None:
        return None
    if os.path.exists(path):
        # Pin latest.zip to one generation before inferring the paired replay path.
        return os.path.realpath(path)
    if not path.endswith(".zip") and os.path.exists(path + ".zip"):
        return os.path.realpath(path + ".zip")
    raise FileNotFoundError(f"resume checkpoint not found: {path}")


def _resolve_resume_replay_buffer_path(cfg, checkpoint_path):
    raw_path = getattr(cfg, "resume_replay_buffer_path", "auto")
    if raw_path is None:
        raise ValueError(
            "resume_from requires a replay buffer; set resume_replay_buffer_path or use 'auto'."
        )
    path_text = str(raw_path).strip()
    if path_text.lower() in {"", "none", "null"}:
        raise ValueError(
            "resume_from requires a replay buffer; set resume_replay_buffer_path or use 'auto'."
        )
    if path_text.lower() == "auto":
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_name = os.path.basename(checkpoint_path)
        stem = checkpoint_name[:-4] if checkpoint_name.endswith(".zip") else checkpoint_name
        if stem == "final":
            replay_name = "final_replay_buffer.pkl"
        elif stem.startswith("ft_policy_"):
            replay_name = stem.replace("ft_policy_", "ft_policy_replay_buffer_", 1) + ".pkl"
        else:
            raise ValueError(
                "resume_replay_buffer_path=auto only supports checkpoint names like "
                "ft_policy_<steps>_steps.zip or final.zip; pass resume_replay_buffer_path explicitly."
            )
        replay_path = os.path.join(checkpoint_dir, replay_name)
    else:
        replay_path = hydra.utils.to_absolute_path(path_text)
    if not os.path.exists(replay_path):
        raise FileNotFoundError(f"resume replay buffer not found: {replay_path}")
    return replay_path


@hydra.main(
    config_path=os.path.join(base_path, "cfg/robomimic"),
    config_name="dsrl_can.yaml",
    version_base=None,
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    eval_only = bool(getattr(cfg, "eval_only", False))

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
    base_policy_horizon = int(getattr(cfg, "base_policy_horizon_steps", diffusion_horizon))
    MAX_STEPS = max(1, math.ceil(cfg.env.max_episode_steps / cfg.act_steps))

    train_num_env = cfg.env.n_envs
    num_env = cfg.env.n_eval_envs if eval_only else train_num_env

    def make_env():
        if cfg.env_name in ["halfcheetah-medium-v2", "hopper-medium-v2", "walker2d-medium-v2"]:
            if d4rl is None:
                raise ImportError(f"d4rl is required for gym task {cfg.env_name!r}.")
            env = gym.make(cfg.env_name)
            env = ObservationWrapperGym(env, cfg.normalization_path)
        elif cfg.env_name in ["lift", "can", "square", "tool_hang", "transport"]:
            env = make_robomimic_env(
                env=cfg.env_name,
                normalization_path=cfg.normalization_path,
                low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys,
                dppo_path=cfg.dppo_path,
                reward_shaping=getattr(cfg.env, "reward_shaping", False),
            )
            env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
        env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
        return env

    base_policy = load_base_policy(cfg)
    env = make_vec_env(make_env, n_envs=num_env, vec_env_cls=SubprocVecEnv)
    residual_policy = "MlpPolicy"
    resume_from = _resolve_checkpoint_path(getattr(cfg, "resume_from", None))
    if resume_from is not None and cfg.algorithm != "residual_sac":
        raise ValueError("resume_from is currently implemented only for algorithm=residual_sac.")
    base_policy_type = str(getattr(cfg, "base_policy_type", "diffusion")).strip().lower()
    actor_obs_mode = str(getattr(cfg.train, "actor_obs_mode", "auto")).strip().lower()
    if actor_obs_mode == "auto":
        actor_obs_mode = "obs_base_action" if base_policy_type == "act" else "obs"
    print(
        f"[residual-config] base_policy_type={base_policy_type} actor_obs_mode={actor_obs_mode}",
        flush=True,
    )
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
        log_std_init=getattr(cfg.train, "log_std_init", 0.0),
        post_linear_modules=post_linear_modules,
        n_critics=cfg.train.n_critics,
        standard_gauss_init=getattr(cfg.train, "standard_gauss_init", False),
    )
    action_feature_mean = None
    action_feature_std = None
    if cfg.algorithm == "residual_sac" and actor_obs_mode == "obs_base_action":
        if not getattr(cfg, "action_stats_path", None):
            raise ValueError(
                "residual_sac with actor base-action conditioning requires cfg.action_stats_path "
                "pointing to a lowdim stats npz with action_mean/action_std."
            )
        action_stats = np.load(cfg.action_stats_path)
        action_feature_mean = np.asarray(action_stats["action_mean"], dtype=np.float32).reshape(-1)
        action_feature_std = np.asarray(action_stats["action_std"], dtype=np.float32).reshape(-1)
        obs_low = np.asarray(env.observation_space.low, dtype=np.float32).reshape(-1)
        obs_high = np.asarray(env.observation_space.high, dtype=np.float32).reshape(-1)
        action_low = np.full(int(np.prod(env.action_space.shape)), -np.inf, dtype=np.float32)
        action_high = np.full(int(np.prod(env.action_space.shape)), np.inf, dtype=np.float32)
        conditioned_observation_space = spaces.Box(
            low=np.concatenate([obs_low, action_low], axis=0),
            high=np.concatenate([obs_high, action_high], axis=0),
            dtype=np.float32,
        )
        policy_kwargs["actor_observation_space"] = conditioned_observation_space
        residual_policy = ActorConditionedMlpPolicy
    if cfg.algorithm == "dsrl_sac":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=cfg.train.actor_lr,
            buffer_size=20000000,  # Replay buffer size
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
    elif cfg.algorithm == "residual_sac":
        if resume_from is not None:
            model = ResidualSAC.load(
                resume_from,
                env=env,
                device=cfg.device,
                force_reset=True,
                base_policy=base_policy,
                tensorboard_log=cfg.logdir,
            )
            model.base_policy = base_policy
            model.tensorboard_log = cfg.logdir
            model._last_base_actions = None
            if eval_only:
                actor_snapshot_path = _cfg_path_or_none(getattr(cfg, "eval_actor_snapshot", None))
                if actor_snapshot_path is not None:
                    payload = torch.load(actor_snapshot_path, map_location=cfg.device)
                    model.actor.load_state_dict(payload["actor"], strict=True)
                    model.env_steps_exact = int(payload["env_steps_exact"])
                    model.num_timesteps = int(payload["num_timesteps"])
                print(
                    f"[eval-only-load] checkpoint={resume_from} actor_snapshot={actor_snapshot_path} "
                    f"policy_steps={int(getattr(model, 'num_timesteps', 0))} "
                    f"env_steps={int(getattr(model, 'env_steps_exact', 0))}",
                    flush=True,
                )
            else:
                resume_replay_buffer_path = _resolve_resume_replay_buffer_path(cfg, resume_from)
                model.load_replay_buffer(resume_replay_buffer_path, truncate_last_traj=True)
                if getattr(
                    model, "actor_observation_mode", "obs"
                ) == "obs_base_action" and not getattr(
                    model.replay_buffer, "use_base_actions", False
                ):
                    raise ValueError(
                        "Loaded replay buffer does not contain base_action arrays required by "
                        "actor_obs_mode=obs_base_action."
                    )
                print(
                    f"[resume] checkpoint={resume_from} replay_buffer={resume_replay_buffer_path} "
                    f"policy_steps={int(getattr(model, 'num_timesteps', 0))} "
                    f"env_steps={int(getattr(model, 'env_steps_exact', 0))} "
                    f"updates={int(getattr(model, '_n_updates', 0))} new_logdir={cfg.logdir}",
                    flush=True,
                )
        else:
            model = ResidualSAC(
                residual_policy,
                env,
                learning_rate=cfg.train.actor_lr,
                buffer_size=getattr(cfg.train, "buffer_size", 20000000),
                learning_starts=getattr(cfg.train, "learning_starts", 8000),
                batch_size=cfg.train.batch_size,
                tau=cfg.train.tau,
                gamma=cfg.train.discount,
                train_freq=cfg.train.train_freq,
                gradient_steps=cfg.train.utd,
                action_noise=None,
                optimize_memory_usage=False,
                ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,
                target_update_interval=1,
                target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,
                use_sde=False,
                sde_sample_freq=-1,
                tensorboard_log=cfg.logdir,
                verbose=1,
                policy_kwargs=policy_kwargs,
                base_policy=base_policy,
                diffusion_act_dim=(diffusion_horizon, cfg.action_dim),
                base_policy_act_dim=(base_policy_horizon, cfg.action_dim),
                residual_scale=cfg.train.res_scale,
                progressive_explore_steps=getattr(cfg.train, "prog_explore_steps", 0),
                critic_learning_rate=getattr(cfg.train, "critic_lr", cfg.train.actor_lr),
                ent_coef_learning_rate=getattr(
                    cfg.train, "alpha_lr", getattr(cfg.train, "critic_lr", cfg.train.actor_lr)
                ),
                max_grad_norm=getattr(cfg.train, "max_grad_norm", None),
                actor_gradient_steps=getattr(cfg.train, "actor_gradient_steps", -1),
                action_norm_regularization=getattr(cfg.train, "action_norm_regularization", 0.0),
                actor_observation_mode=actor_obs_mode,
                action_feature_mean=action_feature_mean,
                action_feature_std=action_feature_std,
                fixed_ent_coef_final=getattr(cfg.train, "ent_coef_final", None),
                fixed_ent_coef_fraction=getattr(cfg.train, "ent_coef_fraction", 1.0),
                fixed_ent_coef_steps=getattr(cfg.train, "ent_coef_steps", None),
                pre_learning_residual_mode=getattr(
                    cfg.train, "pre_learning_residual_mode", "random_residual"
                ),
                residual_temporal_group_size=getattr(cfg.train, "residual_temporal_group_size", 1),
                residual_active_steps=getattr(cfg.train, "residual_active_steps", None),
            )
    elif cfg.algorithm == "dsrl_na":
        model = DSRL(
            "MlpPolicy",
            env,
            learning_rate=cfg.train.actor_lr,
            buffer_size=10000000,  # Replay buffer size
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
            diffusion_act_dim=(diffusion_horizon, cfg.action_dim),
            noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
            critic_backup_combine_type=cfg.train.critic_backup_combine_type,
        )

    async_eval_snapshot_interval = int(
        OmegaConf.select(cfg, "train.async_eval_snapshot_interval", default=0)
    )
    if (
        cfg.algorithm == "residual_sac"
        and async_eval_snapshot_interval > 0
        and resume_from is None
        and not eval_only
    ):
        reference_path = publish_async_eval_reference(
            model, cfg.logdir + "/async_eval/eval_reference.zip"
        )
        print(
            f"[async-eval-reference] checkpoint={reference_path} "
            f"policy_steps={int(getattr(model, 'num_timesteps', 0))} "
            f"env_steps={int(getattr(model, 'env_steps_exact', 0))}",
            flush=True,
        )

    checkpoint_callback_cls = (
        PostRolloutCheckpointCallback if cfg.algorithm == "residual_sac" else CheckpointCallback
    )
    checkpoint_kwargs = {}
    if cfg.algorithm == "residual_sac":
        checkpoint_kwargs["save_latest"] = bool(getattr(cfg, "save_latest_checkpoint", True))
    checkpoint_callback = checkpoint_callback_cls(
        save_freq=cfg.save_model_interval,
        save_path=cfg.logdir + "/checkpoint/",
        name_prefix="ft_policy",
        save_replay_buffer=cfg.save_replay_buffer,
        save_vecnormalize=True,
        **checkpoint_kwargs,
    )

    env_step_eval_interval = int(
        OmegaConf.select(
            cfg,
            "train.env_step_eval_interval",
            default=30000 if cfg.algorithm == "residual_sac" else 0,
        )
    )
    env_step_eval_episodes = int(OmegaConf.select(cfg, "train.env_step_eval_episodes", default=50))
    num_env_eval = cfg.env.n_eval_envs
    eval_env = None
    if eval_only:
        eval_env = env
        eval_env.seed(cfg.seed + train_num_env + 1)
    elif cfg.algorithm != "residual_sac" or (
        env_step_eval_interval > 0 and env_step_eval_episodes > 0
    ):
        eval_env = make_vec_env(make_env, n_envs=num_env_eval, vec_env_cls=SubprocVecEnv)
        if cfg.algorithm == "dsrl_sac":
            eval_env = DiffusionPolicyEnvWrapper(eval_env, cfg, base_policy)
        eval_env.seed(cfg.seed + train_num_env + 1)

    legacy_eval_freq = 0 if cfg.algorithm == "residual_sac" else cfg.eval_interval
    logging_callback = LoggingCallback(
        action_chunk=cfg.act_steps,
        eval_episodes=int(cfg.num_evals / num_env_eval),
        log_freq=MAX_STEPS,
        use_wandb=cfg.use_wandb,
        eval_env=eval_env,
        eval_freq=legacy_eval_freq,
        num_train_env=num_env,
        num_eval_env=num_env_eval,
        rew_offset=cfg.env.reward_offset,
        algorithm=cfg.algorithm,
        max_steps=MAX_STEPS,
        deterministic_eval=cfg.deterministic_eval,
        eval_base_policy=False,
    )

    if cfg.algorithm != "residual_sac":
        logging_callback.evaluate(model, deterministic=False)
    if cfg.algorithm != "residual_sac" and cfg.deterministic_eval:
        logging_callback.evaluate(model, deterministic=True)
    logging_callback.log_count += 1

    if eval_only:
        if cfg.algorithm != "residual_sac" or resume_from is None or eval_env is None:
            raise ValueError("eval_only requires algorithm=residual_sac and resume_from.")
        eval_episodes = int(getattr(cfg, "eval_only_episodes", 50))
        deterministic = bool(getattr(cfg, "eval_only_deterministic", False))
        env_step = int(getattr(model, "env_steps_exact", getattr(model, "num_timesteps", 0)))
        expected_env_step = int(getattr(cfg, "eval_expected_env_step", env_step))
        if env_step != expected_env_step:
            raise ValueError(
                f"eval snapshot env_step={env_step} does not match expected {expected_env_step}."
            )
        success_rate = evaluate_residual_success_rate(
            model, eval_env, eval_episodes, deterministic=deterministic, max_steps=MAX_STEPS
        )
        output_logdir = _cfg_path_or_none(getattr(cfg, "eval_output_logdir", None))
        if output_logdir is None:
            raise ValueError("eval_only requires eval_output_logdir.")
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=output_logdir, filename_suffix=".async_eval")
        writer.add_scalar("env_step_eval/success_rate", success_rate, env_step)
        writer.add_scalar("env_step_eval/episodes", eval_episodes, env_step)
        writer.flush()
        writer.close()
        result_path = _cfg_path_or_none(getattr(cfg, "eval_result_path", None))
        result = {
            "env_step": env_step,
            "episodes": eval_episodes,
            "deterministic": deterministic,
            "success_rate": success_rate,
            "checkpoint": resume_from,
            "actor_snapshot": _cfg_path_or_none(getattr(cfg, "eval_actor_snapshot", None)),
        }
        if result_path is not None:
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            temp_result_path = result_path + ".tmp"
            with open(temp_result_path, "w") as stream:
                json.dump(result, stream, sort_keys=True)
            os.replace(temp_result_path, result_path)
        print(f"[eval-only-result] {json.dumps(result, sort_keys=True)}", flush=True)
        eval_env.close()
        if cfg.use_wandb:
            wandb.finish()
        return

    if resume_from is not None and cfg.load_offline_data:
        raise ValueError(
            "resume_from cannot be combined with load_offline_data; resume replay buffer is loaded instead."
        )
    if cfg.load_offline_data:
        load_offline_data(model, cfg.offline_data_path, num_env)
    warmup_policy_steps = 0
    warmup_env_steps = 0
    if resume_from is not None and cfg.train.init_rollout_steps > 0:
        raise ValueError(
            "resume_from cannot be combined with train.init_rollout_steps; replay buffer is already resumed."
        )
    if cfg.train.init_rollout_steps > 0:
        warmup_stats = collect_rollouts(model, env, cfg.train.init_rollout_steps, base_policy, cfg)
        warmup_policy_steps = int(warmup_stats["policy_steps"])
        warmup_env_steps = int(warmup_stats["env_steps_exact"])
        logging_callback.set_timesteps(warmup_env_steps)
        model.num_timesteps = warmup_policy_steps
        model.env_steps_exact = warmup_env_steps
    elif resume_from is not None:
        logging_callback.set_timesteps(
            int(getattr(model, "env_steps_exact", getattr(model, "num_timesteps", 0)))
        )

    callbacks = [checkpoint_callback, logging_callback]
    if cfg.algorithm == "residual_sac" and async_eval_snapshot_interval > 0:
        callbacks.append(
            ResidualActorSnapshotCallback(
                save_freq=async_eval_snapshot_interval,
                save_path=cfg.logdir + "/async_eval/snapshots",
            )
        )
    max_env_steps = int(OmegaConf.select(cfg, "train.max_env_steps", default=0))
    if cfg.algorithm == "residual_sac" and max_env_steps > 0:
        callbacks.append(StopTrainingOnEnvStepCallback(max_env_steps=max_env_steps, verbose=1))
    if (
        cfg.algorithm == "residual_sac"
        and env_step_eval_interval > 0
        and env_step_eval_episodes > 0
    ):
        assert eval_env is not None
        callbacks.append(
            EnvStepEvalCallback(
                eval_env=eval_env,
                eval_interval=env_step_eval_interval,
                eval_episodes=env_step_eval_episodes,
                deterministic=bool(
                    OmegaConf.select(cfg, "train.env_step_eval_deterministic", default=False)
                ),
                max_steps=MAX_STEPS,
            )
        )
    total_timesteps = int(OmegaConf.select(cfg, "train.total_timesteps", default=100000000))
    learn_timesteps = total_timesteps
    if resume_from is not None:
        resumed_policy_steps = int(getattr(model, "num_timesteps", 0))
        if total_timesteps <= resumed_policy_steps:
            raise ValueError(
                f"train.total_timesteps={total_timesteps} is not larger than resumed "
                f"policy_steps={resumed_policy_steps}; increase train.total_timesteps to continue."
            )
        learn_timesteps = total_timesteps - resumed_policy_steps
        print(
            f"[resume] continuing to policy_step={total_timesteps} "
            f"(additional_policy_steps={learn_timesteps})",
            flush=True,
        )
    # Train the agent
    model.learn(
        total_timesteps=learn_timesteps,
        callback=callbacks,
        reset_num_timesteps=False,
    )

    # Save the final model
    if len(cfg.name) > 0:
        final_checkpoint_path = cfg.logdir + "/checkpoint/final"
        model.save(final_checkpoint_path)
        if (
            cfg.save_replay_buffer
            and hasattr(model, "save_replay_buffer")
            and model.replay_buffer is not None
        ):
            model.save_replay_buffer(cfg.logdir + "/checkpoint/final_replay_buffer.pkl")
        if cfg.algorithm == "residual_sac" and bool(getattr(cfg, "save_latest_checkpoint", True)):
            publish_latest_checkpoint(final_checkpoint_path + ".zip")
        if cfg.algorithm == "residual_sac":
            final_step_path = cfg.logdir + "/checkpoint/final_env_step.txt"
            with open(final_step_path + ".tmp", "w") as stream:
                stream.write(
                    str(int(getattr(model, "env_steps_exact", model.num_timesteps))) + "\n"
                )
            os.replace(final_step_path + ".tmp", final_step_path)

    # Close environment and wandb
    env.close()
    if eval_env is not None and eval_env is not env:
        eval_env.close()
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
