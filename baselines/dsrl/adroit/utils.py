import copy
import json
import os
import random

import torch

try:
    import wandb
except Exception:
    wandb = None
import numpy as np
import time
import math
from collections import deque
from contextlib import contextmanager
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

try:
    import hydra
except Exception:
    hydra = None


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


def _replace_terminal_observations(next_obs, dones, infos):
    replay_next_obs = copy.deepcopy(next_obs)
    for env_idx, done_i in enumerate(dones):
        if not bool(done_i):
            continue
        terminal_obs = infos[env_idx].get("terminal_observation")
        if terminal_obs is None:
            continue
        if isinstance(replay_next_obs, dict):
            for key in replay_next_obs:
                replay_next_obs[key][env_idx] = terminal_obs[key]
        else:
            replay_next_obs[env_idx] = terminal_obs
    return replay_next_obs


@contextmanager
def _preserve_training_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        try:
            cuda_states = torch.cuda.get_rng_state_all()
        except Exception:
            cuda_states = None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class DPPOBasePolicyWrapper:
    def __init__(self, base_policy):
        self.base_policy = base_policy

    def __call__(self, obs, initial_noise, return_numpy=True):
        cond = {
            "state": obs,
            "noise_action": initial_noise,
        }
        with torch.no_grad():
            samples = self.base_policy(cond=cond, deterministic=True)
        diffused_actions = samples.trajectories.detach()
        if return_numpy:
            diffused_actions = diffused_actions.cpu().numpy()
        return diffused_actions


def load_base_policy(cfg):
    if hydra is None:
        raise ImportError("hydra is required for DPPO base policies but is not installed")
    base_policy = hydra.utils.instantiate(cfg.model)
    base_policy = base_policy.eval()
    return DPPOBasePolicyWrapper(base_policy)


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
        tb_log_dir=None,
    ):
        super().__init__(verbose)
        self.action_chunk = action_chunk
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.use_wandb = use_wandb
        if self.use_wandb and wandb is None:
            raise ImportError("use_wandb=True but wandb is not installed")
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
        self.online_eval_step_offset = 0
        self.next_online_eval_env_step = self.online_eval_interval_env_steps
        self.online_eval_count = 0
        self.online_eval_writer = SummaryWriter(tb_log_dir) if tb_log_dir is not None else None
        self.online_eval_details_path = (
            os.path.join(tb_log_dir, "online_eval_details.jsonl")
            if tb_log_dir is not None
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
            scheduled_total_env_step = self.next_online_eval_env_step
            scheduled_training_env_step = scheduled_total_env_step - self.online_eval_step_offset
            self.run_online_eval(
                self.locals["self"],
                env_step=scheduled_training_env_step,
                observed_total_env_step=self.total_timesteps,
                phase="periodic_training",
            )
            while self.next_online_eval_env_step <= self.total_timesteps:
                self.next_online_eval_env_step += self.online_eval_interval_env_steps
        return True

    def run_online_eval(
        self,
        agent,
        env_step=None,
        observed_total_env_step=None,
        phase="manual",
        metric_prefix="env_step_eval",
        step_axis="training_primitive_env_steps",
    ):
        env_step = (
            max(0, self.total_timesteps - self.online_eval_step_offset)
            if env_step is None
            else int(env_step)
        )
        with _preserve_training_rng():
            return self.evaluate(
                agent,
                deterministic=False,
                total_episodes=self.online_eval_total_episodes,
                env_step=env_step,
                log_env_step_metric=True,
                observed_total_env_step=observed_total_env_step,
                phase=phase,
                metric_prefix=metric_prefix,
                step_axis=step_axis,
            )

    def evaluate(
        self,
        agent,
        deterministic=False,
        total_episodes=None,
        env_step=None,
        log_env_step_metric=False,
        observed_total_env_step=None,
        phase="manual",
        metric_prefix="env_step_eval",
        step_axis="training_primitive_env_steps",
    ):
        total_episodes = (
            self.online_eval_total_episodes if total_episodes is None else int(total_episodes)
        )
        env_step = (
            max(0, self.total_timesteps - self.online_eval_step_offset)
            if env_step is None
            else int(env_step)
        )
        observed_total_env_step = (
            self.total_timesteps
            if observed_total_env_step is None
            else int(observed_total_env_step)
        )
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
                completed_env_steps = []
                completed_decisions = []
                initial_noise_samples = []
                while len(completed_success) < total_episodes:
                    remaining = total_episodes - len(completed_success)
                    batch_episodes = min(num_eval_env, remaining)
                    active = np.zeros(num_eval_env, dtype=bool)
                    active[:batch_episodes] = True
                    episode_success = np.zeros(num_eval_env, dtype=np.float32)
                    episode_reward = np.zeros(num_eval_env, dtype=np.float32)
                    episode_env_steps = np.zeros(num_eval_env, dtype=np.int64)
                    episode_decisions = np.zeros(num_eval_env, dtype=np.int64)
                    obs = env.reset()

                    # Evaluate fixed-horizon episodes. Some robomimic training configs do not
                    # set done on horizon/success, so done alone is not a reliable episode count.
                    for _ in range(max(1, self.max_steps)):
                        active_before_step = active.copy()
                        if self.algorithm == "dsrl_sac":
                            action, _ = agent.predict(obs, deterministic=deterministic)
                        elif self.algorithm == "dsrl_na":
                            if hasattr(agent, "predict_diffused_with_initial_noise"):
                                action, _, initial_noise = (
                                    agent.predict_diffused_with_initial_noise(
                                        obs,
                                        deterministic=deterministic,
                                    )
                                )
                                initial_noise_samples.append(
                                    np.asarray(initial_noise)[active_before_step].reshape(-1)
                                )
                            else:
                                action, _ = agent.predict_diffused(obs, deterministic=deterministic)
                        next_obs, reward, done, info = env.step(action)
                        obs = next_obs
                        episode_reward[active_before_step] += np.asarray(reward)[active_before_step]
                        for env_idx, info_i in enumerate(info):
                            if not active_before_step[env_idx]:
                                continue
                            step_i = info_i.get("chunk_env_steps", self.action_chunk)
                            try:
                                step_i = int(step_i)
                            except Exception:
                                step_i = self.action_chunk
                            episode_env_steps[env_idx] += max(step_i, 1)
                            episode_decisions[env_idx] += 1
                            if _extract_success(info_i):
                                episode_success[env_idx] = 1
                        for env_idx, done_i in enumerate(done):
                            if active_before_step[env_idx] and bool(done_i):
                                active[env_idx] = False
                        if not active.any():
                            break

                    for env_idx in range(batch_episodes):
                        completed_success.append(float(episode_success[env_idx] > 0))
                        completed_rewards.append(float(episode_reward[env_idx]))
                        completed_env_steps.append(int(episode_env_steps[env_idx]))
                        completed_decisions.append(int(episode_decisions[env_idx]))

                if len(completed_success) == 0:
                    success_rate = 0.0
                    avg_rew = 0.0
                    avg_env_steps = 0.0
                else:
                    success_rate = float(np.mean(completed_success[:total_episodes]))
                    avg_rew = float(np.mean(completed_rewards[:total_episodes]))
                    avg_env_steps = float(np.mean(completed_env_steps[:total_episodes]))
                initial_noise_stats = None
                if len(initial_noise_samples) > 0:
                    noise_flat = np.concatenate(initial_noise_samples).astype(
                        np.float64, copy=False
                    )
                    noise_abs = np.abs(noise_flat)
                    noise_bound = float(
                        np.max(
                            np.abs(
                                np.concatenate(
                                    [
                                        np.asarray(agent.action_space.low).reshape(-1),
                                        np.asarray(agent.action_space.high).reshape(-1),
                                    ]
                                )
                            )
                        )
                    )
                    quantiles = np.quantile(noise_flat, [0.01, 0.05, 0.5, 0.95, 0.99])
                    abs_quantiles = np.quantile(noise_abs, [0.5, 0.9, 0.95, 0.99])
                    initial_noise_stats = {
                        "count": int(noise_flat.size),
                        "action_bound": float(noise_bound),
                        "mean": float(np.mean(noise_flat)),
                        "std": float(np.std(noise_flat)),
                        "min": float(np.min(noise_flat)),
                        "max": float(np.max(noise_flat)),
                        "q01_q05_q50_q95_q99": [float(v) for v in quantiles],
                        "abs_q50_q90_q95_q99": [float(v) for v in abs_quantiles],
                        "fraction_abs_ge_95pct_bound": float(
                            np.mean(noise_abs >= 0.95 * noise_bound)
                        ),
                        "fraction_abs_ge_bound": float(np.mean(noise_abs >= noise_bound)),
                    }
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
                        f"{metric_prefix}/success_rate", float(success_rate), env_step
                    )
                    self.online_eval_writer.add_scalar(
                        f"{metric_prefix}/avg_reward", float(avg_rew), env_step
                    )
                    self.online_eval_writer.add_scalar(
                        f"{metric_prefix}/avg_env_steps", float(avg_env_steps), env_step
                    )
                    self.online_eval_writer.add_scalar(
                        f"{metric_prefix}/success_count",
                        float(np.sum(completed_success[:total_episodes])),
                        env_step,
                    )
                    if initial_noise_stats is not None:
                        self.online_eval_writer.add_scalar(
                            f"{metric_prefix}/initial_noise_std",
                            initial_noise_stats["std"],
                            env_step,
                        )
                        self.online_eval_writer.add_scalar(
                            f"{metric_prefix}/initial_noise_abs_q95",
                            initial_noise_stats["abs_q50_q90_q95_q99"][2],
                            env_step,
                        )
                        self.online_eval_writer.add_scalar(
                            f"{metric_prefix}/initial_noise_fraction_abs_ge_95pct_bound",
                            initial_noise_stats["fraction_abs_ge_95pct_bound"],
                            env_step,
                        )
                    self.online_eval_writer.flush()
                record = {
                    "eval_index": int(self.online_eval_count),
                    "phase": str(phase),
                    "metric_prefix": str(metric_prefix),
                    "step_axis": str(step_axis),
                    "deterministic": bool(deterministic),
                    "scheduled_axis_steps": int(env_step),
                    "observed_axis_steps": int(
                        observed_total_env_step
                        if step_axis == "warmup_primitive_env_steps"
                        else max(0, observed_total_env_step - self.online_eval_step_offset)
                    ),
                    "observed_total_env_steps_including_warmup": int(observed_total_env_step),
                    "warmup_env_steps": (
                        int(self.online_eval_step_offset)
                        if self.online_eval_step_offset > 0
                        else None
                    ),
                    "agent_policy_timesteps": int(getattr(agent, "num_timesteps", 0)),
                    "agent_updates": int(getattr(agent, "_n_updates", 0)),
                    "episodes_requested": int(total_episodes),
                    "episodes_completed": int(min(len(completed_success), total_episodes)),
                    "success_count": int(np.sum(completed_success[:total_episodes])),
                    "success_rate": float(success_rate),
                    "avg_reward": float(avg_rew),
                    "avg_env_steps": float(avg_env_steps),
                    "initial_noise_stats": initial_noise_stats,
                    "episodes": [
                        {
                            "episode_index": int(episode_idx),
                            "success": int(completed_success[episode_idx]),
                            "reward": float(completed_rewards[episode_idx]),
                            "env_steps": int(completed_env_steps[episode_idx]),
                            "policy_decisions": int(completed_decisions[episode_idx]),
                        }
                        for episode_idx in range(min(len(completed_success), total_episodes))
                    ],
                }
                if self.online_eval_details_path is not None:
                    with open(self.online_eval_details_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, sort_keys=True) + "\n")
                        f.flush()
                self.online_eval_count += 1
                initial_noise_std_text = (
                    "none" if initial_noise_stats is None else f"{initial_noise_stats['std']:.4f}"
                )
                print(
                    f"[eval-summary] deterministic={deterministic} "
                    f"success={record['success_count']}/{record['episodes_completed']} "
                    f"success_rate={success_rate:.4f} avg_reward={avg_rew:.4f} "
                    f"avg_env_steps={avg_env_steps:.1f} "
                    f"step_axis={step_axis} scheduled_axis_steps={env_step} "
                    f"observed_axis_steps={record['observed_axis_steps']} "
                    f"initial_noise_std={initial_noise_std_text}",
                    flush=True,
                )
                return record

    def set_timesteps(self, timesteps, reset_online_eval_schedule=False):
        self.total_timesteps = int(timesteps)
        if reset_online_eval_schedule and self.online_eval_interval_env_steps > 0:
            self.online_eval_step_offset = self.total_timesteps
            self.next_online_eval_env_step = (
                self.online_eval_step_offset + self.online_eval_interval_env_steps
            )

    def _on_training_end(self) -> None:
        if self.online_eval_writer is not None:
            self.online_eval_writer.flush()
            self.online_eval_writer.close()


def collect_rollouts(model, env, num_steps, base_policy, cfg):
    obs = env.reset()
    diffusion_horizon = int(getattr(cfg, "diffusion_horizon_steps", cfg.act_steps))
    num_envs = int(getattr(env, "num_envs", cfg.env.n_envs))
    env_steps_exact = 0
    start_time = time.time()
    progress_every = max(1, num_steps // 20)
    for i in range(num_steps):
        noise = torch.randn(cfg.env.n_envs, diffusion_horizon, cfg.action_dim).to(device=cfg.device)
        if cfg.algorithm == "dsrl_sac":
            noise[noise < -cfg.train.action_magnitude] = -cfg.train.action_magnitude
            noise[noise > cfg.train.action_magnitude] = cfg.train.action_magnitude
        action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)
        next_obs, reward, done, info = env.step(action)
        replay_next_obs = _replace_terminal_observations(next_obs, done, info)
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
            next_obs=replay_next_obs,
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
