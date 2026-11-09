import numpy as np
import torch

from baselines.dice.dp.core import (
    CompleteEpisodeAccumulator,
    DICEQMember,
    DICERLLearner,
    DICERLLearnerConfig,
    DICEReplayBuffer,
    DICETransition,
    LinearOfflineRatioSchedule,
    MacroTransition,
    TransitionUpdateScheduler,
    sample_rlpd_batch,
)
from exrl.common.networks import MLP as ExRLMLP


class FakePrior:
    def __init__(self, horizon_steps=2, action_dim=2):
        self.horizon_steps = int(horizon_steps)
        self.ac_dim = int(action_dim)
        self.device = torch.device("cpu")

    def sample_from_state_noise(self, state, noise_action):
        state_term = state[:, :1, None]
        return torch.tanh(noise_action) * 0.25 + state_term * 0.1


def _macro(value, reward, terminated=False, truncated=False):
    return MacroTransition(
        state=np.asarray([value, value + 1], dtype=np.float32),
        action=np.asarray([value / 10.0], dtype=np.float32),
        reward=float(reward),
        next_state=np.asarray([value + 1, value + 2], dtype=np.float32),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def _dice_transition(value, action_dim=2):
    return DICETransition(
        state=np.asarray([value, value + 1], dtype=np.float32),
        action=np.full((action_dim,), value / 10.0, dtype=np.float32),
        reward=-float(value),
        discount=0.99,
        next_state=np.asarray([value + 1, value + 2], dtype=np.float32),
        terminated=False,
        truncated=False,
        mc_return=-float(value + 1),
    )


def test_resfit_actor_block_and_zero_residual_are_exact():
    from baselines.dice.dp.core import DICEResidualActor

    actor = DICEResidualActor(3, 4, hidden_dims=(16, 16))
    state = torch.randn(7, 3)
    noise = torch.randn(7, 4)
    torch.testing.assert_close(
        actor(state, noise), torch.zeros(7, 4), atol=0.0, rtol=0.0
    )
    first_linear = next(
        module for module in actor.trunk if isinstance(module, torch.nn.Linear)
    )
    gram = first_linear.weight.T @ first_linear.weight
    torch.testing.assert_close(gram, torch.eye(7), atol=2e-6, rtol=2e-6)
    assert [type(module) for module in actor.trunk] == [
        torch.nn.Linear,
        torch.nn.LayerNorm,
        torch.nn.Dropout,
        torch.nn.ReLU,
        torch.nn.Linear,
        torch.nn.LayerNorm,
        torch.nn.Dropout,
        torch.nn.ReLU,
    ]
    assert all(
        module.p == 0.0
        for module in actor.trunk
        if isinstance(module, torch.nn.Dropout)
    )


def test_critic_member_reuses_exact_exrl_mlp_without_layernorm():
    torch.manual_seed(17)
    member = DICEQMember(3, 4, hidden_dims=(8, 8), use_layernorm=False)
    torch.manual_seed(17)
    reference = ExRLMLP(7, 1, (8, 8), activation="relu", use_layernorm=False)
    assert isinstance(member.backbone, ExRLMLP)
    assert list(member.backbone.state_dict()) == list(reference.state_dict())
    for actual, expected in zip(
        member.backbone.state_dict().values(), reference.state_dict().values()
    ):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert not any(
        isinstance(module, torch.nn.LayerNorm) for module in member.modules()
    )


def test_complete_episode_accumulator_computes_mc_and_three_step_returns():
    accumulator = CompleteEpisodeAccumulator(n_step=3, gamma=0.5, n_envs=1)
    assert accumulator.append(0, _macro(0, 1.0)) == []
    assert accumulator.append(0, _macro(1, 2.0)) == []
    assert accumulator.append(0, _macro(2, 3.0)) == []
    emitted = accumulator.append(0, _macro(3, 4.0, terminated=True))
    assert len(emitted) == 4
    np.testing.assert_allclose(
        [item.mc_return for item in emitted], [3.25, 4.5, 5.0, 4.0]
    )
    assert abs(emitted[0].reward - 2.75) < 1e-8
    assert abs(emitted[0].discount - 0.125) < 1e-8
    assert abs(emitted[1].reward - 4.5) < 1e-8
    assert emitted[1].discount == 0.0
    assert all(item.terminated for item in emitted[1:])
    assert accumulator.pending_lengths() == [0]


def test_timeout_flushes_episode_but_retains_bootstrap_discount():
    accumulator = CompleteEpisodeAccumulator(n_step=3, gamma=0.5, n_envs=1)
    accumulator.append(0, _macro(0, 1.0))
    emitted = accumulator.append(0, _macro(1, 2.0, truncated=True))
    assert len(emitted) == 2
    assert abs(emitted[0].reward - 2.0) < 1e-8
    assert abs(emitted[0].discount - 0.25) < 1e-8
    assert abs(emitted[1].discount - 0.5) < 1e-8
    assert all(item.truncated and not item.terminated for item in emitted)


def test_dice_actual_utd_scheduler_and_ratio_schedule():
    scheduler = TransitionUpdateScheduler(2.5, 1.25)
    flags = scheduler.plan(4)
    assert len(flags) == 10
    assert [index for index, flag in enumerate(flags) if flag] == [1, 3, 5, 7, 9]
    flags += scheduler.plan(4)
    assert len(flags) == 20
    assert sum(flags) == 10
    assert scheduler.critic_credit == 0.0
    assert scheduler.actor_credit == 0.0

    ratio = LinearOfflineRatioSchedule(0.7, 0.1, 160_000)
    assert ratio.value(0) == 0.7
    assert abs(ratio.value(80_000) - 0.4) < 1e-12
    assert abs(ratio.value(160_000) - 0.1) < 1e-12
    assert abs(ratio.value(999_999) - 0.1) < 1e-12


def test_cpu_replay_and_rlpd_integer_split_are_separate():
    online = DICEReplayBuffer(2, 2, capacity=16, seed=1)
    offline = DICEReplayBuffer(2, 2, capacity=16, seed=2)
    for value in range(8):
        online.add(_dice_transition(value))
        offline.add(_dice_transition(value + 20))
    batch = sample_rlpd_batch(
        online,
        offline,
        batch_size=10,
        offline_ratio=0.7,
        device=torch.device("cpu"),
    )
    assert batch.states.shape == (10, 2)
    assert int((batch.data_source == 1).sum()) == 7
    assert int((batch.data_source == 0).sum()) == 3
    assert isinstance(online.states, np.ndarray)


def test_replay_capacity_independent_load_rejects_wrapped_payload():
    source = DICEReplayBuffer(2, 2, capacity=4, seed=3)
    for value in range(5):
        source.add(_dice_transition(value))
    target = DICEReplayBuffer(2, 2, capacity=8, seed=4)
    try:
        target.load_state_dict(source.state_dict(), strict_capacity=False)
    except ValueError as exc:
        assert "non-wrapped" in str(exc)
    else:
        raise AssertionError("capacity-independent load accepted a wrapped replay")


def _tiny_config():
    return DICERLLearnerConfig(
        state_dim=3,
        action_dim=2,
        horizon_steps=2,
        actor_hidden_dims=(16, 16),
        critic_hidden_dims=(16, 16),
        critic_ensemble_size=3,
        actor_num_noise_samples=2,
        target_num_noise_samples=2,
        rollout_num_noise_samples=3,
        q_filter_warmup_primitive_steps=10,
        best_of_n_warmup_primitive_steps=20,
        prior_microbatch_size=3,
    )


def _learner_batch(batch_size=6):
    from baselines.dice.dp.core import DICEReplayBatch

    return DICEReplayBatch(
        states=torch.randn(batch_size, 3),
        actions=torch.randn(batch_size, 4) * 0.1,
        rewards=torch.randn(batch_size),
        discounts=torch.full((batch_size,), 0.99**3),
        next_states=torch.randn(batch_size, 3),
        terminated=torch.zeros(batch_size, dtype=torch.bool),
        truncated=torch.zeros(batch_size, dtype=torch.bool),
        mc_returns=torch.full((batch_size,), 100.0),
        data_source=torch.tensor([0, 1, 0, 1, 0, 1]),
    )


def test_learner_initial_policy_matches_prior_and_updates_finitely():
    torch.manual_seed(3)
    learner = DICERLLearner(_tiny_config(), FakePrior(), torch.device("cpu"))
    assert isinstance(learner.actor_optimizer, torch.optim.Adam)
    assert isinstance(learner.critic_optimizer, torch.optim.Adam)
    assert not learner.critic_target.training
    states = np.zeros((2, 3), dtype=np.float32)
    selected = learner.select_action(
        states, primitive_transitions=0, candidate_count_override=1
    )
    np.testing.assert_array_equal(selected["residual"], np.zeros((2, 2, 2)))
    np.testing.assert_array_equal(selected["action"], selected["base_action"])
    assert selected["candidate_count"].tolist() == [1, 1]

    target_before = [
        parameter.detach().clone() for parameter in learner.critic_target.parameters()
    ]
    metrics = learner.update(
        _learner_batch(),
        update_actor=True,
        update_target=True,
        primitive_transitions=11,
    )
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["train/q_filter_active"] == 1.0
    assert metrics["train/bc_keep_fraction"] == 0.0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(target_before, learner.critic_target.parameters())
    )
    assert learner.critic(torch.randn(5, 3), torch.randn(5, 4)).shape == (3, 5)


def test_phase1_checkpoint_roundtrip_preserves_batched_critic_and_optimizer():
    config = _tiny_config()
    config.optimization_level = "phase1"
    torch.manual_seed(31)
    learner = DICERLLearner(config, FakePrior(), torch.device("cpu"))
    learner.update(
        _learner_batch(),
        update_actor=True,
        update_target=True,
        primitive_transitions=11,
        collect_metrics=False,
    )
    payload = learner.state_dict()
    restored = DICERLLearner(config, FakePrior(), torch.device("cpu"))
    restored.load_state_dict(payload)
    state = torch.randn(5, 3)
    action = torch.randn(5, 4)
    torch.testing.assert_close(
        restored.critic(state, action),
        learner.critic(state, action),
        rtol=0.0,
        atol=0.0,
    )
    assert restored.actor_optimizer.state_dict()["state"].keys() == (
        learner.actor_optimizer.state_dict()["state"].keys()
    )
    assert restored.critic_optimizer.state_dict()["state"].keys() == (
        learner.critic_optimizer.state_dict()["state"].keys()
    )


def test_baseline_checkpoint_migrates_to_phase1_without_losing_optimizer_state():
    baseline_config = _tiny_config()
    torch.manual_seed(41)
    baseline = DICERLLearner(
        baseline_config, FakePrior(), torch.device("cpu")
    )
    baseline.update(
        _learner_batch(),
        update_actor=True,
        update_target=True,
        primitive_transitions=11,
        collect_metrics=False,
    )
    payload = baseline.state_dict()

    optimized_config = _tiny_config()
    optimized_config.optimization_level = "phase1"
    optimized_config.prior_microbatch_size = 32
    optimized = DICERLLearner(
        optimized_config, FakePrior(), torch.device("cpu")
    )
    optimized.load_state_dict(payload, allow_runtime_migration=True)

    state = torch.randn(5, 3)
    action = torch.randn(5, 4)
    torch.testing.assert_close(
        optimized.critic(state, action),
        baseline.critic(state, action),
        rtol=2e-5,
        atol=2e-7,
    )
    assert len(optimized.critic_optimizer.state) == len(
        list(optimized.critic.parameters())
    )
    optimized.update(
        _learner_batch(),
        update_actor=True,
        update_target=True,
        primitive_transitions=12,
        collect_metrics=False,
    )


def test_rollout_best_of_n_starts_only_after_warmup_boundary():
    learner = DICERLLearner(_tiny_config(), FakePrior(), torch.device("cpu"))
    states = np.zeros((1, 3), dtype=np.float32)
    assert learner.select_action(states, 20)["candidate_count"].item() == 1
    assert learner.select_action(states, 21)["candidate_count"].item() == 3
    assert learner.select_action(states, 0, eval_mode=True)["candidate_count"].item() == 3
