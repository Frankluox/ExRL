import copy
import os
import pickle

import numpy as np
import pytest
import torch
import torch.nn as nn

from baselines.ogpo.robomimic.dp_core import (
    DPDiffusionChainSampler,
    DPDiffusionSamplerConfig,
    DPReverseChain,
    DPNativeOGPOAgent,
    DPNativeOGPOConfig,
    OGPOSequenceReplayBuffer,
    aggregate_q_values,
    compute_td_loss,
    compute_td_target,
    conservative_group_advantages,
    ppo_clipped_loss,
    select_best_of_n_actions,
)
from exrl.runners.dp_q.diffusion import DiffusionSampler, DiffusionSamplerConfig
from baselines.ogpo.robomimic.train import _terminal_state_from_info


class ZeroNoiseNet(nn.Module):
    def __init__(self, horizon_steps=2, action_dim=1, cond_steps=1, obs_dim=3):
        super().__init__()
        self.horizon_steps = int(horizon_steps)
        self.action_dim = int(action_dim)
        self.cond_steps = int(cond_steps)
        self.obs_dim = int(obs_dim)

    def forward(self, x, time, cond, **kwargs):
        return torch.zeros_like(x)


class TinyNoiseNet(nn.Module):
    def __init__(self, horizon_steps=2, action_dim=1, cond_steps=1, obs_dim=3):
        super().__init__()
        self.horizon_steps = int(horizon_steps)
        self.action_dim = int(action_dim)
        self.cond_steps = int(cond_steps)
        self.obs_dim = int(obs_dim)
        in_dim = self.horizon_steps * self.action_dim + self.cond_steps * self.obs_dim + 1
        self.net = nn.Sequential(nn.Linear(in_dim, 16), nn.Mish(), nn.Linear(16, self.horizon_steps * self.action_dim))

    def forward(self, x, time, cond, **kwargs):
        b = x.shape[0]
        state = cond["state"].reshape(b, -1)
        t = time.reshape(b, 1).to(dtype=x.dtype) / 10.0
        y = self.net(torch.cat([x.reshape(b, -1), state, t], dim=-1))
        return y.reshape_as(x)


def test_sequence_replay_matches_ogpo_prefix_semantics():
    replay = OGPOSequenceReplayBuffer(obs_dim=2, action_dim=1, capacity=8, device=torch.device("cpu"), seed=0)
    for i in range(5):
        replay.add_transition(
            observation=np.array([i, i + 0.5], dtype=np.float32),
            action=np.array([i + 10], dtype=np.float32),
            reward=float(i + 1),
            terminal=float(i == 2),
            mask=0.0 if i == 2 else 1.0,
            next_observation=np.array([i + 1, i + 1.5], dtype=np.float32),
            is_success=float(i <= 2),
        )

    batch = replay.sample_sequence(batch_size=1, sequence_length=3, discount=0.99, idxs=np.array([0]))
    np.testing.assert_allclose(batch.actions.squeeze(-1).numpy(), [[10.0, 11.0, 12.0]])
    np.testing.assert_allclose(batch.rewards.numpy(), [[1.0, 1.0 + 0.99 * 2.0, 1.0 + 0.99 * 2.0 + 0.99**2 * 3.0]])
    np.testing.assert_allclose(batch.masks.numpy(), [[1.0, 1.0, 0.0]])
    np.testing.assert_allclose(batch.terminals.numpy(), [[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(batch.valid.numpy(), [[1.0, 1.0, 1.0]])
    np.testing.assert_allclose(batch.next_observations.numpy(), [[3.0, 3.5]])


def _add_labeled_trajectory(replay, label, length, success):
    for step in range(length):
        value = float(label * 10 + step)
        terminal = float(step == length - 1)
        replay.add_transition(
            observation=np.array([value, value + 0.5], dtype=np.float32),
            action=np.array([value], dtype=np.float32),
            reward=value,
            terminal=terminal,
            mask=1.0 - terminal,
            next_observation=np.array([value + 1.0, value + 1.5], dtype=np.float32),
            is_success=float(success),
        )


def test_sequence_replay_never_crosses_episode_boundary():
    replay = OGPOSequenceReplayBuffer(obs_dim=2, action_dim=1, capacity=16, device=torch.device("cpu"), seed=3)
    _add_labeled_trajectory(replay, label=1, length=2, success=False)
    _add_labeled_trajectory(replay, label=2, length=3, success=True)

    with pytest.raises(ValueError, match="crosses a trajectory boundary"):
        replay.sample_sequence(batch_size=1, sequence_length=3, discount=0.99, idxs=np.array([0]))

    batch = replay.sample_sequence(batch_size=32, sequence_length=3, discount=0.99)
    np.testing.assert_allclose(batch.actions.squeeze(-1).numpy(), np.tile([20.0, 21.0, 22.0], (32, 1)))
    np.testing.assert_allclose(batch.valid.numpy(), np.ones((32, 3), dtype=np.float32))

    success_batch = replay.sample_sequence(batch_size=8, sequence_length=3, discount=0.99, success_only=True)
    np.testing.assert_allclose(success_batch.actions.squeeze(-1).numpy(), np.tile([20.0, 21.0, 22.0], (8, 1)))


def test_sequence_replay_preserves_trajectory_boundaries_after_ring_wrap():
    replay = OGPOSequenceReplayBuffer(obs_dim=2, action_dim=1, capacity=6, device=torch.device("cpu"), seed=7)
    _add_labeled_trajectory(replay, label=1, length=2, success=False)
    _add_labeled_trajectory(replay, label=2, length=2, success=False)
    _add_labeled_trajectory(replay, label=3, length=2, success=False)
    _add_labeled_trajectory(replay, label=4, length=2, success=True)

    batch = replay.sample_sequence(batch_size=128, sequence_length=2, discount=0.99)
    actions = batch.actions.squeeze(-1).numpy()
    np.testing.assert_allclose(actions[:, 1], actions[:, 0] + 1.0)
    assert set((actions[:, 0] // 10).astype(np.int64)).issubset({2, 3, 4})

    wrapped = replay.sample_sequence(batch_size=1, sequence_length=2, discount=0.99, idxs=np.array([0]))
    np.testing.assert_allclose(wrapped.actions.squeeze(-1).numpy(), [[40.0, 41.0]])


def test_sequence_replay_reconstructs_trajectory_ids_for_legacy_checkpoint():
    replay = OGPOSequenceReplayBuffer(obs_dim=2, action_dim=1, capacity=8, device=torch.device("cpu"), seed=11)
    _add_labeled_trajectory(replay, label=1, length=2, success=False)
    _add_labeled_trajectory(replay, label=2, length=3, success=True)
    del replay.trajectory_ids
    del replay._next_trajectory_id

    batch = replay.sample_sequence(batch_size=1, sequence_length=3, discount=0.99, idxs=np.array([2]))
    np.testing.assert_allclose(batch.actions.squeeze(-1).numpy(), [[20.0, 21.0, 22.0]])
    with pytest.raises(ValueError, match="crosses a trajectory boundary"):
        replay.sample_sequence(batch_size=1, sequence_length=3, discount=0.99, idxs=np.array([0]))


def _reference_sequence_starts(replay, sequence_length, success_only=None):
    logical = replay._chronological_indices()
    t = int(sequence_length)
    candidate_count = len(logical) - t + 1
    if candidate_count <= 0:
        return np.empty((0,), dtype=np.int64)
    first = logical[:candidate_count]
    last = logical[t - 1 : t - 1 + candidate_count]
    valid = replay.trajectory_ids[first] == replay.trajectory_ids[last]
    if success_only is not None:
        successful = replay.is_success[first] > 0.0
        valid &= successful if success_only else ~successful
    return np.asarray(first[valid], dtype=np.int64)


def _assert_sequence_caches_match_reference(replay, sequence_lengths):
    for sequence_length in sequence_lengths:
        cache = replay._get_sequence_cache(sequence_length)
        np.testing.assert_array_equal(
            cache.all_starts.active_items(),
            _reference_sequence_starts(replay, sequence_length),
        )
        np.testing.assert_array_equal(
            cache.success_starts.active_items(),
            _reference_sequence_starts(replay, sequence_length, success_only=True),
        )
        np.testing.assert_array_equal(
            cache.failure_starts.active_items(),
            _reference_sequence_starts(replay, sequence_length, success_only=False),
        )


def test_sequence_replay_incrementally_maintains_start_pools_through_ring_wrap():
    replay = OGPOSequenceReplayBuffer(obs_dim=2, action_dim=1, capacity=9, device=torch.device("cpu"), seed=19)
    _add_labeled_trajectory(replay, label=1, length=4, success=False)
    _assert_sequence_caches_match_reference(replay, (2, 3))

    trajectories = [(2, 2, True), (3, 5, False), (4, 3, True), (5, 4, True)]
    for label, length, success in trajectories:
        for step in range(length):
            value = float(label * 10 + step)
            terminal = float(step == length - 1)
            replay.add_transition(
                observation=np.array([value, value + 0.5], dtype=np.float32),
                action=np.array([value], dtype=np.float32),
                reward=value,
                terminal=terminal,
                mask=1.0 - terminal,
                next_observation=np.array([value + 1.0, value + 1.5], dtype=np.float32),
                is_success=float(success),
                score=float(label),
            )
            _assert_sequence_caches_match_reference(replay, (2, 3))


def test_sequence_replay_cached_sample_and_add_do_not_rescan_full_replay(monkeypatch):
    replay = OGPOSequenceReplayBuffer(obs_dim=2, action_dim=1, capacity=12, device=torch.device("cpu"), seed=23)
    _add_labeled_trajectory(replay, label=1, length=4, success=False)
    _add_labeled_trajectory(replay, label=2, length=4, success=True)
    replay.sample_sequence(batch_size=4, sequence_length=3, discount=0.99)

    def fail_rescan():
        raise AssertionError("cached sequence sampling must not rebuild chronological replay")

    monkeypatch.setattr(replay, "_chronological_indices", fail_rescan)
    replay.sample_sequence(batch_size=8, sequence_length=3, discount=0.99)
    replay.sample_sequence(batch_size=8, sequence_length=3, discount=0.99, success_only=True)
    _add_labeled_trajectory(replay, label=3, length=4, success=False)
    replay.sample_sequence(batch_size=8, sequence_length=3, discount=0.99)


def test_sequence_replay_score_weighted_cache_matches_expected_distribution():
    replay = OGPOSequenceReplayBuffer(obs_dim=2, action_dim=1, capacity=20, device=torch.device("cpu"), seed=29)
    for label, score in ((1, 1.0), (2, 2.0), (3, 4.0)):
        for step in range(4):
            value = float(label * 10 + step)
            terminal = float(step == 3)
            replay.add_transition(
                observation=np.array([value, value + 0.5], dtype=np.float32),
                action=np.array([value], dtype=np.float32),
                reward=value,
                terminal=terminal,
                mask=1.0 - terminal,
                next_observation=np.array([value + 1.0, value + 1.5], dtype=np.float32),
                is_success=float(label > 1),
                score=score,
            )

    batch = replay.sample_sequence(batch_size=40000, sequence_length=2, discount=0.99, by_score=True)
    labels = (batch.actions[:, 0, 0].numpy() // 10).astype(np.int64)
    observed = np.array([(labels == label).mean() for label in (1, 2, 3)])
    expected = np.array([1.0, 2.0, 4.0]) / 7.0
    np.testing.assert_allclose(observed, expected, atol=0.015)


def test_sequence_replay_uniform_sampling_matches_legacy_oracle_after_wrap():
    replay = OGPOSequenceReplayBuffer(obs_dim=2, action_dim=1, capacity=23, device=torch.device("cpu"), seed=37)
    for label, length, success in ((1, 3, False), (2, 5, True), (3, 2, False), (4, 6, True), (5, 4, False), (6, 5, True)):
        _add_labeled_trajectory(replay, label=label, length=length, success=success)

    sequence_length = 3
    for success_only in (None, True, False):
        valid_starts = _reference_sequence_starts(replay, sequence_length, success_only=success_only)
        oracle_rng = np.random.default_rng()
        oracle_rng.bit_generator.state = copy.deepcopy(replay.rng.bit_generator.state)
        expected_starts = oracle_rng.choice(valid_starts, size=256)
        expected_indices = (
            expected_starts[:, None] + np.arange(sequence_length, dtype=np.int64)[None, :]
        ) % replay.capacity
        expected_actions = replay.actions[expected_indices]

        actual = replay.sample_sequence(
            batch_size=256,
            sequence_length=sequence_length,
            discount=0.99,
            success_only=success_only,
        )
        np.testing.assert_array_equal(actual.actions.numpy(), expected_actions)


def test_sequence_replay_checkpoint_omits_and_lazily_rebuilds_start_caches():
    replay = OGPOSequenceReplayBuffer(obs_dim=2, action_dim=1, capacity=12, device=torch.device("cpu"), seed=31)
    _add_labeled_trajectory(replay, label=1, length=4, success=False)
    _add_labeled_trajectory(replay, label=2, length=4, success=True)
    replay.sample_sequence(batch_size=4, sequence_length=3, discount=0.99)
    expected = replay.sample_sequence(batch_size=1, sequence_length=3, discount=0.99, idxs=np.array([4]))
    assert 3 in replay._sequence_caches

    restored = pickle.loads(pickle.dumps(replay))
    assert restored._sequence_caches == {}
    actual = restored.sample_sequence(batch_size=1, sequence_length=3, discount=0.99, idxs=np.array([4]))
    torch.testing.assert_close(actual.actions, expected.actions)
    restored.sample_sequence(batch_size=4, sequence_length=3, discount=0.99, success_only=True)
    assert 3 in restored._sequence_caches


def test_terminal_state_from_info_prefers_terminal_observation():
    fallback = np.array([0.0, 0.0], dtype=np.float32)

    state_dict = {"state": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
    np.testing.assert_allclose(_terminal_state_from_info({"terminal_observation": state_dict}, fallback), [3.0, 4.0])

    flat_stack = np.array([[5.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(_terminal_state_from_info({"terminal_observation": flat_stack}, fallback), [5.0, 6.0])

    out = _terminal_state_from_info({}, fallback)
    np.testing.assert_allclose(out, fallback)
    assert out is not fallback


def test_ddpm_chain_logprob_recomputes_sampled_logprob():
    torch.manual_seed(0)
    net = ZeroNoiseNet()
    sampler = DPDiffusionChainSampler(horizon_steps=2, action_dim=1)
    cfg = DPDiffusionSamplerConfig(denoising_steps=4, use_ddim=False, include_initial_logprob=True)
    obs = torch.zeros((3, 3), dtype=torch.float32)
    chain = sampler.sample_chain(net, obs, cfg, cond_steps=1, obs_dim=3, deterministic=False)
    recomputed = sampler.log_prob_chain(net, obs, chain, cfg, cond_steps=1, obs_dim=3)
    assert chain.log_prob is not None
    torch.testing.assert_close(recomputed, chain.log_prob)
    assert torch.isfinite(recomputed).all()


def test_ddim_ogpo_sde_chain_logprob_recomputes_sampled_logprob():
    torch.manual_seed(0)
    net = TinyNoiseNet(horizon_steps=2, action_dim=1, cond_steps=1, obs_dim=3)
    sampler = DPDiffusionChainSampler(horizon_steps=2, action_dim=1)
    cfg = DPDiffusionSamplerConfig(
        denoising_steps=5,
        use_ddim=True,
        ddim_steps=5,
        use_ogpo_sde=True,
        use_constant_noise=False,
        use_tapered_noise=True,
        error_correct_sde_to_ode=True,
        constant_noise_std=0.05,
        denoised_clip_value=1.0,
        randn_clip_value=3.0,
        final_action_clip_value=None,
        include_initial_logprob=True,
    )
    obs = torch.zeros((3, 3), dtype=torch.float32)
    chain = sampler.sample_chain(net, obs, cfg, cond_steps=1, obs_dim=3, deterministic=False)
    recomputed = sampler.log_prob_chain(net, obs, chain, cfg, cond_steps=1, obs_dim=3)
    assert chain.log_prob is not None
    torch.testing.assert_close(recomputed, chain.log_prob)
    assert torch.isfinite(recomputed).all()
    assert chain.xs.shape[1] == cfg.ddim_steps + 1


def test_ddim_ogpo_sde_first_step_matches_flow_style_formula():
    net = ZeroNoiseNet()
    sampler = DPDiffusionChainSampler(horizon_steps=2, action_dim=1)
    cfg = DPDiffusionSamplerConfig(
        denoising_steps=4,
        use_ddim=True,
        ddim_steps=4,
        use_ogpo_sde=True,
        use_tapered_noise=True,
        error_correct_sde_to_ode=True,
        constant_noise_std=0.2,
        denoised_clip_value=None,
        final_action_clip_value=None,
    )
    obs = torch.zeros((2, 3), dtype=torch.float32)
    cond = sampler._cond_from_obs(obs, cond_steps=1, obs_dim=3)
    sched = sampler._build_schedule(4, True, 4, torch.device("cpu"), torch.float32)
    x = torch.tensor([[[0.25], [-0.5]], [[0.1], [0.2]]], dtype=torch.float32)
    t_val = int(sched["ddim_t"][0].item())
    mean, std = sampler._step_params(net, x, t_val, 0, cond, sched, cfg, num_steps=4)

    alpha = sched["ddim_alphas"][0]
    alpha_prev = sched["ddim_alphas_prev"][0]
    mean_det = x * torch.sqrt(alpha_prev / alpha)
    dt = 1.0 / 4.0
    drift = (mean_det - x) / dt
    correction = 0.5 * cfg.constant_noise_std * cfg.constant_noise_std * (-x)
    expected_mean = x + (drift + correction) * dt
    expected_std = torch.full_like(x, cfg.constant_noise_std)

    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(std, expected_std)


def test_ddim_ogpo_sde_logprob_matches_jax_flow_oracle():
    oracle_path = os.environ.get("OGPO_JAX_ORACLE_NPZ")
    if not oracle_path:
        return
    data = np.load(oracle_path)
    net = ZeroNoiseNet(horizon_steps=2, action_dim=2, cond_steps=1, obs_dim=3)
    sampler = DPDiffusionChainSampler(horizon_steps=2, action_dim=2)
    xs = torch.as_tensor(data["flow_chain_flat"], dtype=torch.float32).reshape(2, 5, 2, 2)
    obs = torch.as_tensor(data["flow_observations"], dtype=torch.float32)
    sched = sampler._build_schedule(4, True, 4, torch.device("cpu"), torch.float32)
    chain = DPReverseChain(xs=xs, t_values=sched["ddim_t"], log_prob=None, actions=xs[:, -1])

    cfg_full = DPDiffusionSamplerConfig(
        denoising_steps=4,
        use_ddim=True,
        ddim_steps=4,
        ft_denoising_steps=0,
        use_ogpo_sde=True,
        use_constant_noise=False,
        use_tapered_noise=True,
        error_correct_sde_to_ode=True,
        constant_noise_std=0.05,
        denoised_clip_value=None,
        final_action_clip_value=None,
        include_initial_logprob=True,
    )
    lp_full = sampler.log_prob_chain(net, obs, chain, cfg_full, cond_steps=1, obs_dim=3)
    np.testing.assert_allclose(lp_full.numpy(), data["flow_logprob_tapered_full"], rtol=2e-5, atol=2e-5)
    assert int(data["flow_logprob_steps_tapered_full"]) == 5

    cfg_partial = DPDiffusionSamplerConfig(
        denoising_steps=4,
        use_ddim=True,
        ddim_steps=4,
        ft_denoising_steps=2,
        use_ogpo_sde=True,
        use_constant_noise=False,
        use_tapered_noise=True,
        error_correct_sde_to_ode=True,
        constant_noise_std=0.05,
        denoised_clip_value=None,
        final_action_clip_value=None,
        include_initial_logprob=True,
    )
    lp_partial = sampler.log_prob_chain(net, obs, chain, cfg_partial, cond_steps=1, obs_dim=3)
    np.testing.assert_allclose(lp_partial.numpy(), data["flow_logprob_tapered_partial"], rtol=2e-5, atol=2e-5)
    assert int(data["flow_logprob_steps_tapered_partial"]) == 2


def test_ddim_ogpo_constant_noise_skip_last_matches_jax_flow_oracle():
    oracle_path = os.environ.get("OGPO_JAX_ORACLE_NPZ")
    if not oracle_path:
        return
    data = np.load(oracle_path)
    net = ZeroNoiseNet(horizon_steps=2, action_dim=2, cond_steps=1, obs_dim=3)
    sampler = DPDiffusionChainSampler(horizon_steps=2, action_dim=2)
    xs = torch.as_tensor(data["flow_chain_flat"], dtype=torch.float32).reshape(2, 5, 2, 2)
    obs = torch.as_tensor(data["flow_observations"], dtype=torch.float32)
    sched = sampler._build_schedule(4, True, 4, torch.device("cpu"), torch.float32)
    chain = DPReverseChain(xs=xs, t_values=sched["ddim_t"], log_prob=None, actions=xs[:, -1])
    cfg = DPDiffusionSamplerConfig(
        denoising_steps=4,
        use_ddim=True,
        ddim_steps=4,
        ft_denoising_steps=0,
        use_ogpo_sde=True,
        use_constant_noise=True,
        use_tapered_noise=False,
        error_correct_sde_to_ode=True,
        constant_noise_std=0.05,
        denoised_clip_value=None,
        final_action_clip_value=None,
        include_initial_logprob=True,
    )
    lp = sampler.log_prob_chain(net, obs, chain, cfg, cond_steps=1, obs_dim=3)
    np.testing.assert_allclose(lp.numpy(), data["flow_logprob_constant_full"], rtol=2e-5, atol=2e-5)
    assert int(data["flow_logprob_steps_constant_full"]) == 4


def test_dp_native_config_defaults_match_ogpo_eval_and_sde_semantics():
    cfg = DPNativeOGPOConfig(obs_dim=3, action_dim=2, horizon_steps=2, train_denoising_steps=10, eval_ddim_steps=10)
    rollout = cfg.rollout_sampler_config()
    assert rollout.use_ddim is True
    assert rollout.use_ogpo_sde is True
    assert rollout.use_tapered_noise is True
    assert rollout.use_constant_noise is False
    assert rollout.error_correct_sde_to_ode is True
    assert rollout.constant_noise_std == 0.05
    assert rollout.include_initial_logprob is True
    assert rollout.final_action_clip_value == 1.0

    eval_cfg = cfg.eval_sampler_config()
    assert eval_cfg.use_ddim is True
    assert eval_cfg.use_ogpo_sde is False
    assert eval_cfg.include_initial_logprob is False
    assert eval_cfg.final_action_clip_value == 1.0


def test_final_action_clip_order_matches_ogpo_source_behavior():
    net = ZeroNoiseNet(horizon_steps=2, action_dim=1, cond_steps=1, obs_dim=3)
    sampler = DPDiffusionChainSampler(horizon_steps=2, action_dim=1)
    cfg = DPDiffusionSamplerConfig(
        denoising_steps=2,
        use_ddim=True,
        ddim_steps=2,
        use_ogpo_sde=True,
        use_tapered_noise=True,
        error_correct_sde_to_ode=False,
        constant_noise_std=0.01,
        denoised_clip_value=None,
        randn_clip_value=0.0,
        final_action_clip_value=1.0,
        include_initial_logprob=True,
    )
    obs = torch.zeros((1, 3), dtype=torch.float32)
    noise_action = torch.full((1, 2, 1), 2.0, dtype=torch.float32)
    chain = sampler.sample_chain(
        net,
        obs,
        cfg,
        cond_steps=1,
        obs_dim=3,
        deterministic=False,
        noise_action=noise_action,
    )
    assert chain.actions.abs().max() <= 1.0
    recomputed_on_clipped_chain = sampler.log_prob_chain(net, obs, chain, cfg, cond_steps=1, obs_dim=3)
    assert chain.log_prob is not None
    assert not torch.allclose(chain.log_prob, recomputed_on_clipped_chain)


def test_dp_chain_sampler_deterministic_matches_existing_dsrl_sampler():
    torch.manual_seed(0)
    net = TinyNoiseNet(horizon_steps=2, action_dim=1, cond_steps=1, obs_dim=3)
    obs = torch.randn((4, 3), dtype=torch.float32)
    cond = {"state": obs.reshape(4, 1, 3)}
    noise_action = torch.randn((4, 2, 1), dtype=torch.float32)

    chain_sampler = DPDiffusionChainSampler(horizon_steps=2, action_dim=1)
    legacy_sampler = DiffusionSampler(horizon_steps=2, action_dim=1)

    for use_ddim in (False, True):
        cfg = DPDiffusionSamplerConfig(
            denoising_steps=5,
            use_ddim=use_ddim,
            ddim_steps=3,
            denoised_clip_value=1.0,
            randn_clip_value=3.0,
            final_action_clip_value=None,
            eps_clip_value=None,
        )
        legacy_cfg = DiffusionSamplerConfig(
            denoising_steps=5,
            use_ddim=use_ddim,
            ddim_steps=3,
            denoised_clip_value=1.0,
            randn_clip_value=3.0,
            final_action_clip_value=None,
            eps_clip_value=None,
        )
        ours = chain_sampler.sample_chain(
            net,
            obs,
            cfg,
            cond_steps=1,
            obs_dim=3,
            deterministic=True,
            noise_action=noise_action,
        ).actions
        legacy = legacy_sampler.sample(net, cond, legacy_cfg, deterministic=True, noise_action=noise_action)
        torch.testing.assert_close(ours, legacy)


def test_conservative_group_advantages_safe_sign_agreement():
    q = torch.tensor(
        [
            [
                [3.0, 3.0, 3.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ],
            [
                [3.0, 1.0, 2.0],
                [1.0, 3.0, 2.0],
                [2.0, 2.0, 2.0],
            ],
        ]
    )
    adv = conservative_group_advantages(q)
    torch.testing.assert_close(adv[0], torch.tensor([1.0, -1.0, 0.0]))
    torch.testing.assert_close(adv[1], torch.tensor([0.0, 0.0, 0.0]))


def test_best_of_n_subsample_matches_jax_take_along_axis_semantics():
    actions = torch.tensor(
        [
            [[0.0], [1.0], [2.0]],
            [[10.0], [11.0], [12.0]],
        ]
    )
    q_values = torch.tensor(
        [
            [[1.0, 9.0, 3.0], [2.0, 8.0, 5.0]],
            [[2.0, 1.0, 4.0], [3.0, 7.0, 4.0]],
            [[3.0, 2.0, 5.0], [4.0, 6.0, 3.0]],
        ]
    )  # [M, B, N]
    idx = torch.tensor(
        [
            [[0, 1, 2], [0, 1, 2]],
            [[1, 2, 0], [1, 2, 0]],
        ]
    )
    best, best_idx = select_best_of_n_actions(actions, q_values, subsample_bon=True, subsample_size=2, subsample_indices=idx)
    torch.testing.assert_close(best_idx, torch.tensor([2, 1]))
    torch.testing.assert_close(best, torch.tensor([[2.0], [11.0]]))


def test_jax_oracle_matches_shared_ogpo_math():
    oracle_path = os.environ.get("OGPO_JAX_ORACLE_NPZ")
    if not oracle_path:
        return
    data = np.load(oracle_path)
    oracle_version = int(np.asarray(data["oracle_version"]).item()) if "oracle_version" in data else 1

    if oracle_version >= 2:
        replay = OGPOSequenceReplayBuffer(obs_dim=2, action_dim=1, capacity=8, device=torch.device("cpu"), seed=123)
        obs = np.arange(16, dtype=np.float32).reshape(8, 2)
        next_obs = obs + 0.25
        actions = (np.arange(8, dtype=np.float32).reshape(8, 1) - 3.0) / 10.0
        rewards = np.array([-1.0, -0.5, 2.0, -1.0, -0.25, 0.75, -1.0, 1.5], dtype=np.float32)
        terminals = np.array([0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
        masks = np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=np.float32)
        is_success = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
        for i in range(8):
            replay.add_transition(obs[i], actions[i], rewards[i], terminals[i], masks[i], next_obs[i], is_success=is_success[i])
        seq = replay.sample_sequence(
            batch_size=4,
            sequence_length=3,
            discount=0.97,
            idxs=np.zeros((4,), dtype=np.int64),
        )
        np.testing.assert_allclose(seq.observations.numpy(), data["seq_observations"])
        np.testing.assert_allclose(seq.actions.numpy(), data["seq_actions"])
        np.testing.assert_allclose(seq.rewards.numpy(), data["seq_rewards"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(seq.masks.numpy(), data["seq_masks"])
        np.testing.assert_allclose(seq.terminals.numpy(), data["seq_terminals"])
        np.testing.assert_allclose(seq.valid.numpy(), data["seq_valid"])
        np.testing.assert_allclose(seq.next_observations.numpy(), data["seq_next_observations"])
        seq_success = replay.sample_sequence(
            batch_size=2,
            sequence_length=2,
            discount=0.97,
            success_only=True,
            idxs=np.zeros((2,), dtype=np.int64),
        )
        np.testing.assert_allclose(seq_success.observations.numpy(), data["seq_success_observations"])
        np.testing.assert_allclose(seq_success.actions.numpy(), data["seq_success_actions"])

    q_values_t = torch.as_tensor(data["q_values"].T, dtype=torch.float32)
    torch.testing.assert_close(aggregate_q_values(q_values_t, "mean"), torch.as_tensor(data["q_mean"], dtype=torch.float32))
    torch.testing.assert_close(aggregate_q_values(q_values_t, "min"), torch.as_tensor(data["q_min"], dtype=torch.float32))

    target = compute_td_target(
        torch.as_tensor(data["rewards_2d"], dtype=torch.float32),
        torch.as_tensor(data["masks_2d"], dtype=torch.float32),
        torch.as_tensor(data["next_q"], dtype=torch.float32),
        discount=0.9,
        horizon_length=3,
    )
    torch.testing.assert_close(target, torch.as_tensor(data["td_target"], dtype=torch.float32))
    td_loss = compute_td_loss(
        torch.as_tensor(data["q_pred"].T, dtype=torch.float32),
        target,
        torch.as_tensor(data["valid"], dtype=torch.float32),
    )
    torch.testing.assert_close(td_loss, torch.as_tensor(data["td_loss"], dtype=torch.float32))

    q_full_t = torch.as_tensor(np.transpose(data["q_full"], (2, 0, 1)), dtype=torch.float32)
    adv = conservative_group_advantages(q_full_t)
    torch.testing.assert_close(adv.T, torch.as_tensor(data["advantages"], dtype=torch.float32))

    ppo_loss, ppo_stats = ppo_clipped_loss(
        torch.as_tensor(data["log_probs"], dtype=torch.float32),
        torch.as_tensor(data["old_log_probs"], dtype=torch.float32),
        torch.as_tensor(data["ppo_adv"], dtype=torch.float32),
        clip_epsilon=0.1,
    )
    torch.testing.assert_close(ppo_loss, torch.as_tensor(data["ppo_loss"], dtype=torch.float32))
    torch.testing.assert_close(ppo_stats["ratio"], torch.as_tensor(data["ppo_ratio"], dtype=torch.float32))
    torch.testing.assert_close(ppo_stats["ratio_std"], torch.as_tensor(data["ppo_ratio_std"], dtype=torch.float32))
    torch.testing.assert_close(ppo_stats["approx_kl"], torch.as_tensor(data["ppo_approx_kl"], dtype=torch.float32))
    torch.testing.assert_close(
        ppo_stats["ratio_clipped_total"],
        torch.as_tensor(data["ppo_ratio_clipped_total"], dtype=torch.float32),
    )


def test_dp_native_ogpo_batch_update_smoke():
    torch.manual_seed(1)
    actor = TinyNoiseNet(horizon_steps=2, action_dim=1, cond_steps=1, obs_dim=3)
    cfg = DPNativeOGPOConfig(
        obs_dim=3,
        action_dim=1,
        horizon_steps=2,
        cond_steps=1,
        critic_hidden_dims=(32, 32),
        num_qs=3,
        group_num_samples=2,
        best_of_n=2,
        train_denoising_steps=4,
        eval_ddim_steps=2,
        bc_coeff=0.1,
    )
    agent = DPNativeOGPOAgent(actor, cfg, device=torch.device("cpu"))
    replay = OGPOSequenceReplayBuffer(obs_dim=3, action_dim=1, capacity=16, device=torch.device("cpu"), seed=2)
    for i in range(8):
        replay.add_transition(
            observation=np.full((3,), i / 10.0, dtype=np.float32),
            action=np.array([0.1 * i], dtype=np.float32),
            reward=-1.0,
            terminal=0.0,
            mask=1.0,
            next_observation=np.full((3,), (i + 1) / 10.0, dtype=np.float32),
            is_success=float(i >= 4),
        )
    batch = replay.sample_sequence(batch_size=4, sequence_length=2, discount=0.99)
    success_batch = replay.sample_sequence(batch_size=4, sequence_length=2, discount=0.99, success_only=True)
    info = agent.batch_update(batch, success_batch=success_batch, success_flag=True)
    assert "actor/loss" in info
    assert "rb_critic/loss" in info
    assert "sb_critic/loss" in info
    assert np.isfinite(info["actor/loss"])


def test_bc_only_update_skips_critic_and_uses_raw_bc_loss():
    torch.manual_seed(3)
    actor = TinyNoiseNet(horizon_steps=2, action_dim=1, cond_steps=1, obs_dim=3)
    cfg = DPNativeOGPOConfig(
        obs_dim=3,
        action_dim=1,
        horizon_steps=2,
        cond_steps=1,
        critic_hidden_dims=(32, 32),
        num_qs=2,
        train_denoising_steps=4,
        eval_ddim_steps=2,
        update_mode="bc_only",
    )
    agent = DPNativeOGPOAgent(actor, cfg, device=torch.device("cpu"))
    replay = OGPOSequenceReplayBuffer(obs_dim=3, action_dim=1, capacity=16, device=torch.device("cpu"), seed=4)
    for i in range(8):
        replay.add_transition(
            observation=np.full((3,), i / 10.0, dtype=np.float32),
            action=np.array([0.05 * i], dtype=np.float32),
            reward=-1.0,
            terminal=0.0,
            mask=1.0,
            next_observation=np.full((3,), (i + 1) / 10.0, dtype=np.float32),
            is_success=1.0,
        )
    batch = replay.sample_sequence(batch_size=4, sequence_length=2, discount=0.99, success_only=True)
    critic_before = {k: v.detach().clone() for k, v in agent.critic.state_dict().items()}
    actor_before = {k: v.detach().clone() for k, v in agent.actor.state_dict().items()}

    info = agent.bc_only_update(batch)

    assert info["update_mode"] == "bc_only"
    assert "rb_critic/loss" not in info
    assert info["actor/pg_loss"] == 0.0
    assert np.isfinite(info["actor/bc_loss"])
    for k, v in agent.critic.state_dict().items():
        torch.testing.assert_close(v, critic_before[k])
    assert any(not torch.equal(v, actor_before[k]) for k, v in agent.actor.state_dict().items())


def test_best_of_one_rollout_bypasses_critic_bon():
    class RaisingCritic(nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("critic should not be used for best_of_n=1 rollout")

    torch.manual_seed(5)
    actor = TinyNoiseNet(horizon_steps=2, action_dim=1, cond_steps=1, obs_dim=3)
    cfg = DPNativeOGPOConfig(
        obs_dim=3,
        action_dim=1,
        horizon_steps=2,
        cond_steps=1,
        critic_hidden_dims=(32, 32),
        train_denoising_steps=4,
        eval_ddim_steps=2,
    )
    agent = DPNativeOGPOAgent(actor, cfg, device=torch.device("cpu"))
    agent.target_critic = RaisingCritic()
    actions = agent.select_action_chunk_np(np.zeros((2, 3), dtype=np.float32), deterministic=False, best_of_n=1)
    assert actions.shape == (2, 2, 1)
