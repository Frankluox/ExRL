import json
import math
import os
import random
import sys
import warnings
from datetime import datetime
from types import SimpleNamespace

warnings.filterwarnings("ignore")

import numpy as np
import torch
from stable_baselines3 import DSRL
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

try:
    import wandb
except Exception:
    wandb = None

from baselines.dsrl.adroit.policy import load_adroit_policy_bundle
from baselines.dsrl.adroit.environment import make_adroit_dsrl_env
from baselines.dsrl.adroit.utils import collect_rollouts, load_offline_data, LoggingCallback

base_path = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(base_path, "../.."))


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _as_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: _as_dict(v) for k, v in vars(obj).items()}
    return obj


def _parse_scalar(text: str):
    if text in ("True", "true"):
        return True
    if text in ("False", "false"):
        return False
    if text in ("None", "none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _set_nested(cfg, key: str, value):
    parts = key.split(".")
    cur = cfg
    for part in parts[:-1]:
        if not hasattr(cur, part):
            setattr(cur, part, _ns())
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)


def _default_cfg():
    return _ns(
        name="adroit_door_dsrl_seed42",
        log_dir="./logs",
        logdir="",
        base_policy_path="assets/model_epoch_2.pth",
        offline_data_path="",
        algorithm="dsrl_na",
        seed=42,
        device="cuda:0",
        use_wandb=False,
        env_name="AdroitHandDoor-v1",
        obs_key="flat",
        obs_dim=39,
        action_dim=28,
        cond_steps=1,
        act_steps=32,
        diffusion_horizon_steps=32,
        execute_steps=32,
        load_offline_data=False,
        eval_at_start=True,
        online_eval_interval_env_steps=30000,
        online_eval_total_episodes=50,
        total_timesteps=20000000,
        dp_sampler="ddim",
        dp_infer_steps=5,
        save_model_interval=10000000,
        save_replay_buffer=False,
        save_final_model=True,
        env=_ns(
            n_envs=4,
            n_eval_envs=4,
            name="AdroitHandDoor-v1",
            max_episode_steps=200,
            reward_offset=0,
            sparse_step_reward=-1.0,
            sparse_success_reward=0.0,
            terminate_on_success=False,
            timeout_on_horizon=False,
        ),
        wandb=_ns(project="dsrl", run="adroit_door_dsrl", group="adroit-door"),
        train=_ns(
            tau=0.005,
            actor_lr=3e-4,
            batch_size=256,
            train_freq=1,
            utd=1,
            noise_critic_grad_steps=1,
            actor_gradient_steps=-1,
            use_layer_norm=True,
            layer_size=512,
            num_layers=3,
            discount=0.997,
            ent_coef=-1,
            target_ent=0.0,
            init_rollout_steps=512,
            action_magnitude=1.5,
            n_critics=2,
            critic_backup_combine_type="mean",
            target_q_clip_min=None,
            target_q_clip_max=None,
            buffer_size=10000000,
        ),
    )


def _load_cfg_from_cli(argv):
    cfg = _default_cfg()
    for item in argv:
        if "=" not in item:
            raise ValueError(f"expected key=value override, got {item!r}")
        key, value = item.split("=", 1)
        _set_nested(cfg, key, _parse_scalar(value))
    if not str(cfg.logdir).strip():
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cfg.logdir = os.path.join(
            str(cfg.log_dir),
            "adroit-dsrl",
            str(cfg.name),
            f"{stamp}_{int(cfg.seed)}",
        )
    cfg.env.name = cfg.env_name
    return cfg


def main(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if cfg.use_wandb:
        if wandb is None:
            raise ImportError("use_wandb=True but wandb is not installed")
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.name,
            group=cfg.wandb.group,
            monitor_gym=True,
            save_code=True,
            config=_as_dict(cfg),
        )

    diffusion_horizon = int(getattr(cfg, "diffusion_horizon_steps", cfg.act_steps))
    execute_steps = int(getattr(cfg, "execute_steps", cfg.act_steps))
    execute_steps = max(1, min(execute_steps, diffusion_horizon))
    max_decisions = int(math.ceil(float(cfg.env.max_episode_steps) / float(execute_steps)))

    policy_bundle = load_adroit_policy_bundle(cfg)
    base_policy = policy_bundle.policy
    print(
        "[ADROIT-DSRL] "
        f"env={cfg.env_name} obs_dim={cfg.obs_dim} action_dim={cfg.action_dim} "
        f"horizon={diffusion_horizon} execute_steps={execute_steps} "
        f"policy_meta={policy_bundle.policy_meta}",
        flush=True,
    )
    os.makedirs(cfg.logdir, exist_ok=True)
    with open(os.path.join(cfg.logdir, "run_metadata.json"), "w") as f:
        json.dump(
            {
                "config": _as_dict(cfg),
                "env_name": str(cfg.env_name),
                "base_policy_path": str(cfg.base_policy_path),
                "shape_metadata": policy_bundle.shape_metadata,
                "policy_meta": policy_bundle.policy_meta,
                "env_meta": policy_bundle.env_meta,
                "obs_dim": int(cfg.obs_dim),
                "action_dim": int(cfg.action_dim),
                "diffusion_horizon_steps": int(diffusion_horizon),
                "execute_steps": int(execute_steps),
                "max_policy_decisions_per_episode": int(max_decisions),
                "evaluation_protocol": {
                    "metric": "env_step_eval/success_rate",
                    "policy_mode": "stochastic",
                    "eval_at_start": bool(cfg.eval_at_start),
                    "step_zero_timing": "before warmup and before actor updates",
                    "warmup_evaluation": False,
                    "interval_training_primitive_env_steps": int(
                        cfg.online_eval_interval_env_steps
                    ),
                    "episodes_per_eval": int(cfg.online_eval_total_episodes),
                    "num_parallel_eval_envs": int(cfg.env.n_eval_envs),
                    "details_file": "online_eval_details.jsonl",
                },
            },
            f,
            indent=2,
        )

    num_env = int(cfg.env.n_envs)
    env_meta_for_workers = json.loads(json.dumps(policy_bundle.env_meta))
    obs_mean_for_workers = np.asarray(policy_bundle.obs_mean, dtype=np.float32).copy()
    obs_std_for_workers = np.asarray(policy_bundle.obs_std, dtype=np.float32).copy()
    action_mean_for_workers = np.asarray(policy_bundle.action_mean, dtype=np.float32).copy()
    action_std_for_workers = np.asarray(policy_bundle.action_std, dtype=np.float32).copy()

    def make_env_factory(env_id):
        def _init():
            env_i = make_adroit_dsrl_env(
                env_meta_for_workers,
                obs_mean_for_workers,
                obs_std_for_workers,
                action_mean_for_workers,
                action_std_for_workers,
                cfg,
                env_id=env_id,
            )
            return Monitor(env_i, info_keywords=("is_success", "success"))

        return _init

    env_fns = [make_env_factory(i) for i in range(num_env)]
    env = DummyVecEnv(env_fns) if num_env <= 1 else SubprocVecEnv(env_fns, start_method="fork")
    env.seed(cfg.seed + 1)

    post_linear_modules = [torch.nn.LayerNorm] if cfg.train.use_layer_norm else None
    net_arch = [int(cfg.train.layer_size) for _ in range(int(cfg.train.num_layers))]
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=torch.nn.Tanh,
        log_std_init=0.0,
        post_linear_modules=post_linear_modules,
        n_critics=int(cfg.train.n_critics),
    )
    model = DSRL(
        "MlpPolicy",
        env,
        learning_rate=cfg.train.actor_lr,
        buffer_size=int(cfg.train.buffer_size),
        learning_starts=1,
        batch_size=int(cfg.train.batch_size),
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
        device=cfg.device,
        diffusion_policy=base_policy,
        diffusion_act_dim=(diffusion_horizon, cfg.action_dim),
        noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
        actor_gradient_steps=cfg.train.actor_gradient_steps,
        critic_backup_combine_type=cfg.train.critic_backup_combine_type,
        target_q_clip_min=cfg.train.target_q_clip_min,
        target_q_clip_max=cfg.train.target_q_clip_max,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=int(cfg.save_model_interval),
        save_path=os.path.join(cfg.logdir, "checkpoint"),
        name_prefix="ft_policy",
        save_replay_buffer=cfg.save_replay_buffer,
        save_vecnormalize=True,
    )

    num_env_eval = int(cfg.env.n_eval_envs)
    eval_fns = [make_env_factory(10000 + i) for i in range(num_env_eval)]
    eval_env = (
        DummyVecEnv(eval_fns) if num_env_eval <= 1 else SubprocVecEnv(eval_fns, start_method="fork")
    )
    eval_env.seed(cfg.seed + num_env + 1)

    logging_callback = LoggingCallback(
        action_chunk=execute_steps,
        log_freq=max(1, max_decisions),
        use_wandb=cfg.use_wandb,
        eval_env=eval_env,
        num_train_env=num_env,
        num_eval_env=num_env_eval,
        rew_offset=cfg.env.reward_offset,
        algorithm=cfg.algorithm,
        max_steps=max_decisions,
        online_eval_interval_env_steps=int(cfg.online_eval_interval_env_steps),
        online_eval_total_episodes=int(cfg.online_eval_total_episodes),
        tb_log_dir=cfg.logdir,
    )

    if int(cfg.online_eval_total_episodes) > 0 and bool(cfg.eval_at_start):
        print(
            f"[phase] starting pre-warmup online-eval eval_envs={num_env_eval} "
            f"total_episodes={cfg.online_eval_total_episodes} env_step=0 max_decisions={max_decisions}",
            flush=True,
        )
        logging_callback.run_online_eval(
            model,
            env_step=0,
            observed_total_env_step=0,
            phase="pre_warmup_untrained",
        )
        print("[phase] finished pre-warmup online-eval", flush=True)
        logging_callback.log_count += 1

    if cfg.load_offline_data:
        load_offline_data(model, cfg.offline_data_path, num_env)
    warmup_policy_steps = 0
    warmup_env_steps = 0
    if int(cfg.train.init_rollout_steps) > 0:
        print(
            f"[phase] starting init-rollout policy_steps={cfg.train.init_rollout_steps} "
            f"envs={num_env} execute_steps={execute_steps}",
            flush=True,
        )
        warmup_stats = collect_rollouts(
            model, env, int(cfg.train.init_rollout_steps), base_policy, cfg
        )
        warmup_policy_steps = int(warmup_stats["policy_steps"])
        warmup_env_steps = int(warmup_stats["env_steps_exact"])
        print(
            f"[phase] finished init-rollout policy_steps={warmup_policy_steps} "
            f"env_steps={warmup_env_steps}",
            flush=True,
        )
        with open(os.path.join(cfg.logdir, "warmup_summary.json"), "w") as f:
            json.dump(
                {
                    "policy_steps": int(warmup_policy_steps),
                    "primitive_env_steps": int(warmup_env_steps),
                    "agent_updates": int(getattr(model, "_n_updates", 0)),
                    "evaluated_during_warmup": False,
                },
                f,
                indent=2,
            )
        logging_callback.set_timesteps(warmup_env_steps, reset_online_eval_schedule=True)
        model.num_timesteps = warmup_policy_steps
        model.env_steps_exact = warmup_env_steps

    callbacks = [checkpoint_callback, logging_callback]
    total_timesteps = int(cfg.total_timesteps)
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=False,
    )

    if len(cfg.name) > 0 and bool(cfg.save_final_model):
        model.save(os.path.join(cfg.logdir, "checkpoint", "final"))

    env.close()
    eval_env.close()
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main(_load_cfg_from_cli(sys.argv[1:]))
