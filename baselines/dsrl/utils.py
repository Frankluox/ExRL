import torch

try:
    import wandb
except Exception:
    wandb = None
import numpy as np
import time
import math
import multiprocessing as mp
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import OmegaConf


def _extract_success(info):
    raw = info.get("is_success", None)
    if raw is None and isinstance(info.get("episode"), dict):
        raw = info["episode"].get("is_success", None)
    if raw is None:
        return False
    if isinstance(raw, dict):
        return bool(raw.get("task", False))
    if isinstance(raw, (list, tuple, np.ndarray)):
        if len(raw) == 0:
            return False
        raw = raw[-1]
        if isinstance(raw, dict):
            return bool(raw.get("task", False))
    return bool(raw)


class DPPOBasePolicyWrapper:
    def __init__(self, base_policy, action_magnitude=None):
        self.base_policy = base_policy
        self.action_magnitude = action_magnitude

    def __call__(self, obs, initial_noise, return_numpy=True):
        source_horizon = int(initial_noise.shape[1])
        target_horizon = int(getattr(self.base_policy, "horizon_steps", source_horizon))
        noise_action = initial_noise
        if source_horizon != target_horizon:
            noise_action = torch.randn(
                initial_noise.shape[0],
                target_horizon,
                initial_noise.shape[2],
                device=initial_noise.device,
                dtype=initial_noise.dtype,
            )
            if self.action_magnitude is not None:
                noise_action = torch.clamp(
                    noise_action, -self.action_magnitude, self.action_magnitude
                )
            copy_steps = min(source_horizon, target_horizon)
            noise_action[:, :copy_steps, :] = initial_noise[:, :copy_steps, :]
        cond = {
            "state": obs,
            "noise_action": noise_action,
        }
        with torch.no_grad():
            samples = self.base_policy(cond=cond, deterministic=True)
            diffused_actions = samples.trajectories.detach()
        if source_horizon != target_horizon:
            diffused_actions = diffused_actions[:, :source_horizon, :]
        if return_numpy:
            diffused_actions = diffused_actions.cpu().numpy()
        return diffused_actions


def load_base_policy(cfg):
    base_policy = hydra.utils.instantiate(cfg.model)
    base_policy = base_policy.eval()
    return DPPOBasePolicyWrapper(base_policy, action_magnitude=cfg.train.action_magnitude)


def _build_policy_kwargs(cfg):
    post_linear_modules = None
    if cfg.train.use_layer_norm:
        post_linear_modules = [torch.nn.LayerNorm]
    net_arch = [cfg.train.layer_size for _ in range(cfg.train.num_layers)]
    return dict(
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=torch.nn.Tanh,
        log_std_init=0.0,
        post_linear_modules=post_linear_modules,
        n_critics=cfg.train.n_critics,
    )


def _build_eval_agent(cfg, eval_env, base_policy):
    from stable_baselines3 import SAC, DSRL

    policy_kwargs = _build_policy_kwargs(cfg)
    if cfg.algorithm == "dsrl_sac":
        return SAC(
            "MlpPolicy",
            eval_env,
            learning_rate=cfg.train.actor_lr,
            buffer_size=1,
            learning_starts=1,
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
            tensorboard_log=None,
            verbose=0,
            policy_kwargs=policy_kwargs,
        )
    if cfg.algorithm == "dsrl_na":
        return DSRL(
            "MlpPolicy",
            eval_env,
            learning_rate=cfg.train.actor_lr,
            buffer_size=1,
            learning_starts=1,
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
            tensorboard_log=None,
            verbose=0,
            policy_kwargs=policy_kwargs,
            diffusion_policy=base_policy,
            diffusion_act_dim=(int(cfg.act_steps), cfg.action_dim),
            noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
            critic_backup_combine_type=cfg.train.critic_backup_combine_type,
        )
    raise ValueError(f"Unsupported DSRL algorithm: {cfg.algorithm}")


def _async_online_eval_worker(
    cfg_container,
    policy_state,
    env_step,
    total_episodes,
    max_steps,
    num_eval_env,
    tb_log_dir,
    deterministic=False,
):
    try:
        import random
        import gym
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv
        from baselines.dsrl.environment import (
            DiffusionPolicyEnvWrapper,
            ObservationWrapperRobomimic,
            ObservationWrapperGym,
            ActionChunkWrapper,
            make_robomimic_env,
        )

        torch.set_num_threads(1)
        cfg = OmegaConf.create(cfg_container)
        OmegaConf.resolve(cfg)
        seed = int(cfg.seed) + int(env_step) + 100000
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

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
            else:
                raise ValueError(f"Unsupported env_name: {cfg.env_name}")
            return ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)

        print(
            f"[async-eval-worker] start env_step={env_step} episodes={total_episodes} "
            f"eval_envs={num_eval_env} max_steps={max_steps}",
            flush=True,
        )
        base_policy = load_base_policy(cfg)
        eval_env = make_vec_env(make_env, n_envs=num_eval_env, vec_env_cls=DummyVecEnv)
        if cfg.algorithm == "dsrl_sac":
            eval_env = DiffusionPolicyEnvWrapper(eval_env, cfg, base_policy)
        eval_env.seed(seed + 1)
        agent = _build_eval_agent(cfg, eval_env, base_policy)
        agent.policy.load_state_dict(policy_state)
        agent.policy.set_training_mode(False)

        completed_success = []
        completed_rewards = []
        with torch.no_grad():
            while len(completed_success) < total_episodes:
                remaining = total_episodes - len(completed_success)
                batch_episodes = min(num_eval_env, remaining)
                active = np.zeros(num_eval_env, dtype=bool)
                active[:batch_episodes] = True
                episode_success = np.zeros(num_eval_env, dtype=np.float32)
                episode_reward = np.zeros(num_eval_env, dtype=np.float32)
                obs = eval_env.reset()

                for _ in range(max(1, int(max_steps))):
                    if cfg.algorithm == "dsrl_sac":
                        action, _ = agent.predict(obs, deterministic=deterministic)
                    elif cfg.algorithm == "dsrl_na":
                        action, _ = agent.predict_diffused(obs, deterministic=deterministic)
                    obs, reward, done, info = eval_env.step(action)
                    episode_reward[active] += np.asarray(reward)[active]
                    for env_idx, info_i in enumerate(info):
                        if active[env_idx] and _extract_success(info_i):
                            episode_success[env_idx] = 1
                    for env_idx, done_i in enumerate(done):
                        if active[env_idx] and bool(done_i):
                            active[env_idx] = False
                    if not active.any():
                        break

                for env_idx in range(batch_episodes):
                    completed_success.append(float(episode_success[env_idx] > 0))
                    completed_rewards.append(float(episode_reward[env_idx]))

        success_rate = (
            float(np.mean(completed_success[:total_episodes])) if completed_success else 0.0
        )
        avg_rew = float(np.mean(completed_rewards[:total_episodes])) if completed_rewards else 0.0
        writer = SummaryWriter(tb_log_dir)
        writer.add_scalar("env_step_eval/success_rate", success_rate, int(env_step))
        writer.flush()
        writer.close()
        eval_env.close()
        print(
            f"[async-eval-summary] env_step={env_step} success_rate={success_rate:.4f} "
            f"avg_reward={avg_rew:.4f}",
            flush=True,
        )
    except Exception:
        import traceback

        print("[async-eval-error]", flush=True)
        traceback.print_exc()
        raise


class LoggingCallback(BaseCallback):
    def __init__(
        self,
        action_chunk=4,
        log_freq=1000,
        use_wandb=True,
        eval_env=None,
        verbose=0,
        rew_offset=0,
        num_train_env=1,
        num_eval_env=1,
        algorithm="dsrl_sac",
        max_steps=-1,
        success_window=200,
        online_eval_interval_env_steps=30000,
        online_eval_total_episodes=50,
        online_eval_async=True,
        cfg=None,
        tb_log_dir=None,
    ):
        super().__init__(verbose)
        self.action_chunk = action_chunk
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.use_wandb = use_wandb
        self.eval_env = eval_env
        self.log_count = 0
        self.total_reward = 0
        self.rew_offset = rew_offset
        self.total_timesteps = 0
        self.num_train_env = num_train_env
        self.num_eval_env = num_eval_env
        self.episode_success = np.zeros(self.num_train_env)
        self.episode_completed = np.zeros(self.num_train_env)
        self.algorithm = algorithm
        self.max_steps = max_steps
        self.success_window = int(success_window)
        self.completed_success_recent = deque(maxlen=max(1, int(success_window)))
        self.completed_success_recent20 = deque(maxlen=20)
        self.completed_success_since_log = []
        self.online_eval_interval_env_steps = int(online_eval_interval_env_steps)
        self.online_eval_total_episodes = int(online_eval_total_episodes)
        self.online_eval_async = bool(online_eval_async)
        self.online_eval_process = None
        self.online_eval_cfg = (
            OmegaConf.to_container(cfg, resolve=True) if cfg is not None else None
        )
        self.tb_log_dir = tb_log_dir
        self.next_online_eval_env_step = self.online_eval_interval_env_steps
        self.online_eval_writer = (
            SummaryWriter(tb_log_dir)
            if tb_log_dir is not None and not self.online_eval_async
            else None
        )

    def _on_step(self):
        for env_idx, info in enumerate(self.locals["infos"]):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
            if _extract_success(info):
                self.episode_success[env_idx] = 1
        rew = self.locals["rewards"]
        self.total_reward += np.mean(rew)
        dones = self.locals["dones"]
        for env_idx, done_i in enumerate(dones):
            if bool(done_i):
                success_i = float(self.episode_success[env_idx] > 0)
                self.completed_success_recent.append(success_i)
                self.completed_success_recent20.append(success_i)
                self.completed_success_since_log.append(success_i)
                self.episode_success[env_idx] = 0
                self.episode_completed[env_idx] = 0
        # Exact env-step accumulation when wrapper provides per-decision executed steps.
        # Fallback to old estimate for compatibility.
        chunk_steps = 0
        for info in self.locals["infos"]:
            step_i = info.get("chunk_env_steps", self.action_chunk)
            try:
                step_i = int(step_i)
            except Exception:
                step_i = self.action_chunk
            chunk_steps += max(step_i, 1)
        self.total_timesteps += chunk_steps
        if self.n_calls % self.log_freq == 0:
            if len(self.episode_rewards) > 0:
                success_rate_raw = (
                    float(np.mean(self.completed_success_since_log))
                    if len(self.completed_success_since_log) > 0
                    else 0.0
                )
                success_rate_recent = (
                    float(np.mean(self.completed_success_recent))
                    if len(self.completed_success_recent) > 0
                    else success_rate_raw
                )
                success_rate_recent20 = (
                    float(np.mean(self.completed_success_recent20))
                    if len(self.completed_success_recent20) > 0
                    else success_rate_recent
                )
                # Keep both AAC-prefixed tags and rollout/* tags for consistent dashboards.
                ep_rew_mean = float(np.mean(self.episode_rewards))
                ep_len_mean = float(np.mean(self.episode_lengths))
                self.logger.record("aac/train_success_rate", float(success_rate_recent))
                self.logger.record("aac/train_success_rate_recent20", float(success_rate_recent20))
                self.logger.record("aac/train_success_rate_window_raw", float(success_rate_raw))
                self.logger.record("aac/train_ep_rew_mean_cb", ep_rew_mean)
                self.logger.record("aac/train_ep_len_mean_cb", ep_len_mean)
                self.logger.record("aac/train_rew_mean_cb", float(np.mean(self.total_reward)))
                self.logger.record("aac/train_env_steps_est", float(self.total_timesteps))
                self.logger.record("rollout/success_rate", float(success_rate_recent))
                self.logger.record("rollout/success_rate_recent20", float(success_rate_recent20))
                self.logger.record("rollout/success_rate_window_raw", float(success_rate_raw))
                self.logger.record("rollout/ep_rew_mean", ep_rew_mean)
                self.logger.record("rollout/ep_len_mean", ep_len_mean)
                if self.use_wandb:
                    self.log_count += 1
                    wandb.log(
                        {
                            "train/ep_len_mean": np.mean(self.episode_lengths),
                            "train/success_rate": success_rate_recent,
                            "train/success_rate_recent20": success_rate_recent20,
                            "train/success_rate_window_raw": success_rate_raw,
                            "train/ep_rew_mean": np.mean(self.episode_rewards),
                            "train/rew_mean": np.mean(self.total_reward),
                            "train/timesteps": self.total_timesteps,
                            "train/ent_coef": self.locals["self"].logger.name_to_value[
                                "train/ent_coef"
                            ],
                            "train/actor_loss": self.locals["self"].logger.name_to_value[
                                "train/actor_loss"
                            ],
                            "train/critic_loss": self.locals["self"].logger.name_to_value[
                                "train/critic_loss"
                            ],
                            "train/ent_coef_loss": self.locals["self"].logger.name_to_value[
                                "train/ent_coef_loss"
                            ],
                        },
                        step=self.log_count,
                    )
                    if len(self.completed_success_since_log) > 0:
                        wandb.log(
                            {
                                "train/success_rate": success_rate_recent,
                            },
                            step=self.log_count,
                        )
                    if self.algorithm == "dsrl_na":
                        wandb.log(
                            {
                                "train/noise_critic_loss": self.locals["self"].logger.name_to_value[
                                    "train/noise_critic_loss"
                                ],
                            },
                            step=self.log_count,
                        )
            self.episode_rewards = []
            self.episode_lengths = []
            self.total_reward = 0
            self.completed_success_since_log = []

        if (
            self.online_eval_interval_env_steps > 0
            and self.total_timesteps >= self.next_online_eval_env_step
        ):
            self.run_online_eval(self.locals["self"], env_step=self.next_online_eval_env_step)
            while self.next_online_eval_env_step <= self.total_timesteps:
                self.next_online_eval_env_step += self.online_eval_interval_env_steps
        return True

    def run_online_eval(self, agent, env_step=None):
        env_step = self.total_timesteps if env_step is None else int(env_step)
        if self.online_eval_async:
            return self.launch_async_online_eval(agent, env_step=env_step)
        return self.evaluate(
            agent,
            deterministic=False,
            total_episodes=self.online_eval_total_episodes,
            env_step=env_step,
            log_env_step_metric=True,
        )

    def launch_async_online_eval(self, agent, env_step=None):
        env_step = self.total_timesteps if env_step is None else int(env_step)
        if self.online_eval_cfg is None or self.tb_log_dir is None:
            print(
                "[async-eval] fallback to blocking eval because cfg/logdir is missing",
                flush=True,
            )
            return self.evaluate(
                agent,
                deterministic=False,
                total_episodes=self.online_eval_total_episodes,
                env_step=env_step,
                log_env_step_metric=True,
            )
        if self.online_eval_process is not None:
            if self.online_eval_process.is_alive():
                print(
                    f"[async-eval] skip env_step={env_step} because previous "
                    f"pid={self.online_eval_process.pid} is still running",
                    flush=True,
                )
                return None
            print(
                f"[async-eval] previous pid={self.online_eval_process.pid} "
                f"exitcode={self.online_eval_process.exitcode}",
                flush=True,
            )
            self.online_eval_process = None
        policy_state = {
            k: v.detach().cpu().clone() if torch.is_tensor(v) else v
            for k, v in agent.policy.state_dict().items()
        }
        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_async_online_eval_worker,
            args=(
                self.online_eval_cfg,
                policy_state,
                int(env_step),
                int(self.online_eval_total_episodes),
                int(self.max_steps),
                int(self.num_eval_env),
                self.tb_log_dir,
                False,
            ),
        )
        proc.daemon = True
        proc.start()
        self.online_eval_process = proc
        print(
            f"[async-eval] launched env_step={env_step} pid={proc.pid} "
            f"episodes={self.online_eval_total_episodes} eval_envs={self.num_eval_env}",
            flush=True,
        )
        return proc

    def evaluate(
        self,
        agent,
        deterministic=False,
        total_episodes=None,
        env_step=None,
        log_env_step_metric=False,
    ):
        total_episodes = (
            self.online_eval_total_episodes if total_episodes is None else int(total_episodes)
        )
        env_step = self.total_timesteps if env_step is None else int(env_step)
        if total_episodes > 0:
            env = self.eval_env
            num_eval_env = max(1, int(self.num_eval_env))
            print(
                f"[eval] start deterministic={deterministic} "
                f"episodes={total_episodes} eval_envs={self.num_eval_env} "
                f"max_steps={self.max_steps}",
                flush=True,
            )
            with torch.no_grad():
                completed_success = []
                completed_rewards = []
                while len(completed_success) < total_episodes:
                    remaining = total_episodes - len(completed_success)
                    batch_episodes = min(num_eval_env, remaining)
                    active = np.zeros(num_eval_env, dtype=bool)
                    active[:batch_episodes] = True
                    episode_success = np.zeros(num_eval_env, dtype=np.float32)
                    episode_reward = np.zeros(num_eval_env, dtype=np.float32)
                    obs = env.reset()

                    # Evaluate fixed-horizon episodes. Some robomimic training configs do not
                    # set done on horizon/success, so done alone is not a reliable episode count.
                    for _ in range(max(1, self.max_steps)):
                        if self.algorithm == "dsrl_sac":
                            action, _ = agent.predict(obs, deterministic=deterministic)
                        elif self.algorithm == "dsrl_na":
                            action, _ = agent.predict_diffused(obs, deterministic=deterministic)
                        next_obs, reward, done, info = env.step(action)
                        obs = next_obs
                        episode_reward[active] += np.asarray(reward)[active]
                        for env_idx, info_i in enumerate(info):
                            if active[env_idx] and _extract_success(info_i):
                                episode_success[env_idx] = 1
                        for env_idx, done_i in enumerate(done):
                            if active[env_idx] and bool(done_i):
                                active[env_idx] = False
                        if not active.any():
                            break

                    for env_idx in range(batch_episodes):
                        completed_success.append(float(episode_success[env_idx] > 0))
                        completed_rewards.append(float(episode_reward[env_idx]))

                if len(completed_success) == 0:
                    success_rate = 0.0
                    avg_rew = 0.0
                else:
                    success_rate = float(np.mean(completed_success[:total_episodes]))
                    avg_rew = float(np.mean(completed_rewards[:total_episodes]))
                if len(completed_success) < total_episodes:
                    print(
                        f"[eval-warning] completed_episodes={len(completed_success)} "
                        f"requested_episodes={total_episodes}",
                        flush=True,
                    )
                if (
                    log_env_step_metric
                    and not deterministic
                    and self.online_eval_writer is not None
                ):
                    self.online_eval_writer.add_scalar(
                        "env_step_eval/success_rate", float(success_rate), env_step
                    )
                    self.online_eval_writer.flush()
                print(
                    f"[eval-summary] deterministic={deterministic} "
                    f"success_rate={success_rate:.4f} avg_reward={avg_rew:.4f} "
                    f"env_steps_est={env_step}",
                    flush=True,
                )

    def set_timesteps(self, timesteps, reset_online_eval_schedule=False):
        self.total_timesteps = int(timesteps)
        if reset_online_eval_schedule and self.online_eval_interval_env_steps > 0:
            interval = self.online_eval_interval_env_steps
            self.next_online_eval_env_step = ((self.total_timesteps // interval) + 1) * interval

    def _on_training_end(self) -> None:
        if self.online_eval_process is not None and self.online_eval_process.is_alive():
            print(
                f"[async-eval] training ended while pid={self.online_eval_process.pid} "
                "is still running; not waiting",
                flush=True,
            )
        if self.online_eval_writer is not None:
            self.online_eval_writer.flush()
            self.online_eval_writer.close()


def collect_rollouts(model, env, num_steps, base_policy, cfg):
    obs = env.reset()
    diffusion_horizon = int(getattr(cfg, "diffusion_horizon_steps", cfg.act_steps))
    rl_action_horizon = int(cfg.act_steps)
    num_envs = int(getattr(env, "num_envs", cfg.env.n_envs))
    env_steps_exact = 0
    start_time = time.time()
    progress_every = max(1, num_steps // 20)
    for i in range(num_steps):
        noise = torch.randn(cfg.env.n_envs, rl_action_horizon, cfg.action_dim).to(device=cfg.device)
        if cfg.algorithm == "dsrl_sac":
            noise[noise < -cfg.train.action_magnitude] = -cfg.train.action_magnitude
            noise[noise > cfg.train.action_magnitude] = cfg.train.action_magnitude
        action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)
        next_obs, reward, done, info = env.step(action)
        for info_i in info:
            step_i = info_i.get("chunk_env_steps", cfg.act_steps)
            try:
                step_i = int(step_i)
            except Exception:
                step_i = cfg.act_steps
            env_steps_exact += max(step_i, 1)
        if cfg.algorithm == "dsrl_na":
            action_store = action
        elif cfg.algorithm == "dsrl_sac":
            action_store = noise.detach().cpu().numpy()
        action_store = action_store.reshape(-1, action_store.shape[1] * action_store.shape[2])
        if cfg.algorithm == "dsrl_sac":
            action_store = model.policy.scale_action(action_store)
        model.replay_buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=action_store,
            reward=reward,
            done=done,
            infos=info,
        )
        obs = next_obs
        step_idx = i + 1
        if step_idx == 1 or step_idx % progress_every == 0 or step_idx == num_steps:
            elapsed = max(time.time() - start_time, 1e-6)
            steps_per_sec = step_idx / elapsed
            remaining = max(num_steps - step_idx, 0)
            eta_sec = remaining / steps_per_sec if steps_per_sec > 0 else 0.0
            print(
                f"[init-rollout] step={step_idx}/{num_steps} "
                f"pct={100.0 * step_idx / max(num_steps, 1):.1f} "
                f"env_steps={env_steps_exact} "
                f"elapsed_min={elapsed / 60.0:.1f} eta_min={eta_sec / 60.0:.1f}",
                flush=True,
            )
    model.replay_buffer.final_offline_step()
    policy_steps = int(num_steps * num_envs)
    print(
        f"[init-rollout-summary] policy_steps={policy_steps} env_steps={env_steps_exact}",
        flush=True,
    )
    return {
        "policy_steps": policy_steps,
        "env_steps_exact": int(env_steps_exact),
    }


def load_offline_data(model, offline_data_path, n_env):
    # this function should only be applied with dsrl_na
    offline_data = np.load(offline_data_path)
    obs = offline_data["states"]
    next_obs = offline_data["states_next"]
    actions = offline_data["actions"]
    rewards = offline_data["rewards"]
    terminals = offline_data["terminals"]
    for i in range(int(obs.shape[0] / n_env)):
        model.replay_buffer.add(
            obs=obs[n_env * i : n_env * i + n_env],
            next_obs=next_obs[n_env * i : n_env * i + n_env],
            action=actions[n_env * i : n_env * i + n_env],
            reward=rewards[n_env * i : n_env * i + n_env],
            done=terminals[n_env * i : n_env * i + n_env],
            infos=[{}] * n_env,
        )
    model.replay_buffer.final_offline_step()
