import numpy as np
import torch

from exrl.common.networks import MLP as ExRLMLP
from baselines.resfit.dp.td3_core import (
    AggregatedTransition,
    ContinuousQMember,
    EnsembleContinuousCritic,
    PrimitiveNStepAccumulator,
    PrimitiveReplayBuffer,
    PrimitiveTransition,
    ReplayBatch,
    ResFiTActionScaler,
    ResFiTLearner,
    ResFiTLearnerConfig,
    ResidualActor,
    StatePreprocessor,
    TransitionUpdateScheduler,
    critic_td_loss,
    sample_training_batch,
)


def _transition(value, reward, done=False):
    vector = np.asarray([value, value + 1], dtype=np.float32)
    action = np.asarray([value / 10.0], dtype=np.float32)
    return PrimitiveTransition(
        state=vector,
        base_action=action,
        action=action,
        reward=float(reward),
        next_state=vector + 1,
        next_base_action=action + 0.1,
        terminated=bool(done),
        truncated=False,
    )


def test_action_scaler_uses_per_dimension_raw_ranges():
    action_min = np.asarray([-1.0, -1.0, -1.0, -0.44833228, -1.0, -1.0, -1.0])
    action_max = np.asarray([1.0, 1.0, 1.0, 0.20898621, 1.0, 1.0, 1.0])
    scaler = ResFiTActionScaler(
        action_min=action_min,
        action_max=action_max,
        action_scale=0.1,
        min_range_per_dim=0.1,
    )
    expected_expanded = np.asarray([1.1, 1.1, 1.1, 0.36152518, 1.1, 1.1, 1.1])
    expected_actor_raw = np.asarray([0.11, 0.11, 0.11, 0.03615252, 0.11, 0.11, 0.11])
    np.testing.assert_allclose(scaler.expanded_half_range, expected_expanded, atol=1e-7)
    np.testing.assert_allclose(
        scaler.manifest()["max_actor_mean_raw_delta"], expected_actor_raw, atol=1e-7
    )

    # This asymmetric-dimension value incurs a three-ULP error through the
    # naive float32 DP -> raw -> ResFiT -> raw -> DP round trip.
    base_dp = np.asarray(
        [[0.4, -0.2, 0.1, 0.7488130927085876, 0.3, -0.4, 0.5]], dtype=np.float32
    )
    zero = np.zeros_like(base_dp)
    combined = scaler.combine(base_dp, zero)
    np.testing.assert_array_equal(combined["action_dp"], base_dp)
    np.testing.assert_array_equal(combined["action_raw"], scaler.dp_to_raw(base_dp))
    np.testing.assert_allclose(
        scaler.raw_to_dp(scaler.dp_to_raw(base_dp)), base_dp, atol=2e-7
    )

    residual = np.full((1, 7), 0.1, dtype=np.float32)
    shifted = scaler.combine(np.zeros_like(residual), residual)
    raw_mid = scaler.dp_to_raw(np.zeros_like(residual))
    np.testing.assert_allclose(
        shifted["action_raw"] - raw_mid, expected_actor_raw[None], atol=2e-7
    )

    # The floor is applied to the raw range per dimension, while the affine
    # bridge remains exactly invertible at zero residual.
    floor_scaler = ResFiTActionScaler(
        action_min=np.asarray([0.01]),
        action_max=np.asarray([0.03]),
        action_scale=0.1,
        min_range_per_dim=0.1,
    )
    np.testing.assert_allclose(floor_scaler.half_range, [0.05])
    floor_base = np.asarray([[0.75]], dtype=np.float32)
    np.testing.assert_array_equal(
        floor_scaler.combine(floor_base, np.zeros_like(floor_base))["action_dp"],
        floor_base,
    )


def test_replay_action_mode_defaults_to_executed_clamped_action():
    scaler = ResFiTActionScaler(np.asarray([-1.0]), np.asarray([1.0]), action_scale=0.1)
    base = np.asarray([[0.95]], dtype=np.float32)
    residual = np.asarray([[0.2]], dtype=np.float32)
    fixed = scaler.combine(base, residual, replay_action_mode="executed_clamped")
    release = scaler.combine(base, residual, replay_action_mode="release_preclamp")
    np.testing.assert_allclose(fixed["replay_resfit"], fixed["full_resfit"])
    assert float(release["replay_resfit"][0, 0]) > 1.0
    np.testing.assert_allclose(fixed["action_dp"], scaler.resfit_to_dp(fixed["full_resfit"]))


def test_dsrl_raw_state_mode_is_exact_identity_for_transport_contract():
    preprocessor = StatePreprocessor(
        obs_min=np.full(59, -2.0),
        obs_max=np.full(59, 3.0),
        state_mean=np.linspace(-1.0, 1.0, 59),
        state_std=np.linspace(0.2, 2.0, 59),
        mode="dsrl_raw",
    )
    state = np.linspace(-0.9, 0.9, 2 * 59, dtype=np.float32).reshape(2, 59)
    np.testing.assert_array_equal(preprocessor.transform(state), state)
    assert preprocessor.manifest()["identity_dsrl_state"] is True


def test_residual_actor_is_exactly_zero_at_initialization():
    actor = ResidualActor(3, 2, hidden_dims=(16, 16), action_scale=0.1)
    state = torch.randn(7, 3)
    base = torch.randn(7, 2)
    torch.testing.assert_close(actor(state, base), torch.zeros(7, 2), atol=0.0, rtol=0.0)
    first_linear = next(module for module in actor.trunk if isinstance(module, torch.nn.Linear))
    gram = first_linear.weight.T @ first_linear.weight
    torch.testing.assert_close(gram, torch.eye(5), atol=2e-6, rtol=2e-6)
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
        module.p == 0.0 for module in actor.trunk if isinstance(module, torch.nn.Dropout)
    )


def test_continuous_critic_reuses_exact_exrl_mlp():
    torch.manual_seed(17)
    member = ContinuousQMember(3, 2, hidden_dims=(8, 8), use_layernorm=True)
    torch.manual_seed(17)
    reference = ExRLMLP(5, 1, (8, 8), activation="relu", use_layernorm=True)
    assert isinstance(member.backbone, ExRLMLP)
    assert list(member.backbone.state_dict()) == list(reference.state_dict())
    for actual, expected in zip(
        member.backbone.state_dict().values(), reference.state_dict().values()
    ):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_primitive_nstep_and_terminal_flush():
    accumulator = PrimitiveNStepAccumulator(n_step=3, gamma=0.5, n_envs=1)
    assert accumulator.append(0, _transition(0, 1.0)) == []
    assert accumulator.append(0, _transition(1, 2.0)) == []
    emitted = accumulator.append(0, _transition(2, 3.0))
    assert len(emitted) == 1
    assert abs(emitted[0].reward - 2.75) < 1e-8
    assert abs(emitted[0].discount - 0.125) < 1e-8

    flushed = accumulator.append(0, _transition(3, 4.0, done=True))
    assert len(flushed) == 3
    assert all(item.discount == 0.0 for item in flushed)
    assert all(item.terminated for item in flushed)


def test_primitive_nstep_timeout_flushes_but_keeps_bootstrap_discount():
    accumulator = PrimitiveNStepAccumulator(n_step=3, gamma=0.5, n_envs=1)
    accumulator.append(0, _transition(0, 1.0))
    timeout = _transition(1, 2.0)
    timeout.terminated = False
    timeout.truncated = True
    emitted = accumulator.append(0, timeout)
    assert len(emitted) == 2
    assert abs(emitted[0].reward - 2.0) < 1e-8
    assert abs(emitted[0].discount - 0.25) < 1e-8
    assert abs(emitted[1].reward - 2.0) < 1e-8
    assert abs(emitted[1].discount - 0.5) < 1e-8
    assert all(item.truncated and not item.terminated for item in emitted)


def test_update_scheduler_is_vectorization_invariant():
    scheduler = TransitionUpdateScheduler(4.0, 1.0)
    flags = scheduler.plan(4)
    assert len(flags) == 16
    assert [index for index, flag in enumerate(flags) if flag] == [3, 7, 11, 15]


def _batch(batch_size=8):
    return ReplayBatch(
        states=torch.randn(batch_size, 3),
        base_actions=torch.zeros(batch_size, 2),
        actions=torch.zeros(batch_size, 2),
        rewards=torch.randn(batch_size),
        discounts=torch.full((batch_size,), 0.99**3),
        next_states=torch.randn(batch_size, 3),
        next_base_actions=torch.zeros(batch_size, 2),
        terminated=torch.zeros(batch_size, dtype=torch.bool),
        truncated=torch.zeros(batch_size, dtype=torch.bool),
    )


def test_learner_updates_continuous_ensemble_and_actor():
    torch.manual_seed(3)
    config = ResFiTLearnerConfig(
        state_dim=3,
        action_dim=2,
        actor_hidden_dims=(16, 16),
        critic_hidden_dims=(16, 16),
        critic_ensemble_size=4,
        target_critic_subsample_size=2,
    )
    learner = ResFiTLearner(config, torch.device("cpu"))
    assert isinstance(learner.actor_optimizer, torch.optim.AdamW)
    assert isinstance(learner.critic_optimizer, torch.optim.AdamW)
    assert learner.actor_optimizer.param_groups[0]["weight_decay"] == 0.01
    assert learner.critic_optimizer.param_groups[0]["weight_decay"] == 0.01
    assert learner.actor_target.training
    assert not learner.critic_target.training
    metrics = learner.update(_batch(), update_actor=True)
    assert np.isfinite(metrics["train/critic_loss"])
    assert np.isfinite(metrics["train/actor_loss"])
    assert learner.critic(torch.randn(5, 3), torch.randn(5, 2)).shape == (4, 5)


def test_sparse_consolidated_metrics_preserve_exact_update_state():
    base_config = ResFiTLearnerConfig(
        state_dim=3,
        action_dim=2,
        actor_hidden_dims=(16, 16),
        critic_hidden_dims=(16, 16),
        critic_ensemble_size=4,
        target_critic_subsample_size=2,
        target_q_clip_min=-100.0,
        target_q_clip_max=0.0,
    )
    torch.manual_seed(91)
    legacy = ResFiTLearner(base_config, torch.device("cpu"))
    optimized = ResFiTLearner(
        ResFiTLearnerConfig(
            **{
                **base_config.__dict__,
                "consolidate_metric_syncs": True,
                "target_critic_subset_compute": True,
                "foreach_soft_update": True,
                "freeze_critic_during_actor_update": True,
            }
        ),
        torch.device("cpu"),
    )
    for name in ("actor", "actor_target", "critic", "critic_target"):
        getattr(optimized, name).load_state_dict(getattr(legacy, name).state_dict())
    torch.manual_seed(123)
    batch = _batch()
    torch.manual_seed(456)
    legacy_metrics = legacy.update(batch, update_actor=True)
    torch.manual_seed(456)
    optimized_metrics = optimized.update(batch, update_actor=True)
    assert optimized_metrics == legacy_metrics
    for name in ("actor", "actor_target", "critic", "critic_target"):
        expected = getattr(legacy, name).state_dict()
        actual = getattr(optimized, name).state_dict()
        assert expected.keys() == actual.keys()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    assert optimized.update(batch, update_actor=False, collect_metrics=False) == {}


def test_formal_critic_has_ten_independent_heads():
    critic = EnsembleContinuousCritic(3, 2, hidden_dims=(8, 8), ensemble_size=10)
    assert critic(torch.randn(6, 3), torch.randn(6, 2)).shape == (10, 6)
    parameter_ids = [id(member.output.weight) for member in critic.members]
    assert len(set(parameter_ids)) == 10


def test_corrected_td_loss_is_true_per_head_mse():
    errors = torch.tensor([[1.0], [3.0]])
    assert float(critic_td_loss(errors, "per_head_mse")) == 5.0
    assert float(critic_td_loss(errors, "release_mean_abs_square")) == 4.0


class ConstantHeadCritic(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.as_tensor(values, dtype=torch.float32))
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, states, actions):
        # Keep a gradient path through actions even when only head-reduction
        # semantics are under test.
        action_path = actions.sum(dim=-1).unsqueeze(0) * 0.0
        return self.values.view(-1, 1) + action_path + self.anchor * 0.0


class CountingSubsetCritic(ConstantHeadCritic):
    def __init__(self, values):
        super().__init__(values)
        self.full_calls = 0
        self.subset_calls = []

    def forward(self, states, actions):
        self.full_calls += 1
        return super().forward(states, actions)

    def forward_members(self, states, actions, member_indices):
        indices = [int(index) for index in member_indices]
        self.subset_calls.append(indices)
        action_path = actions.sum(dim=-1).unsqueeze(0) * 0.0
        return self.values[indices].view(-1, 1) + action_path + self.anchor * 0.0


def test_target_subset_compute_skips_unselected_critics():
    config = ResFiTLearnerConfig(
        state_dim=3,
        action_dim=2,
        actor_hidden_dims=(8, 8),
        critic_hidden_dims=(8, 8),
        critic_ensemble_size=4,
        target_critic_subsample_size=2,
        target_critic_subset_compute=True,
    )
    learner = ResFiTLearner(config, torch.device("cpu"))
    critic = CountingSubsetCritic([5.0, 2.0, 7.0, 3.0])
    learner.critic_target = critic
    batch = _batch(batch_size=5)
    original_randperm = torch.randperm

    def fixed_randperm(n, *args, **kwargs):
        return torch.tensor([3, 1, 0, 2], device=kwargs.get("device"))

    torch.randperm = fixed_randperm
    try:
        target_min, _ = learner._target_min_q(batch)
    finally:
        torch.randperm = original_randperm
    assert critic.full_calls == 0
    assert critic.subset_calls == [[3, 1]]
    torch.testing.assert_close(target_min, torch.full((5,), 2.0))


def test_target_uses_two_unique_random_heads_and_actor_uses_ensemble_mean():
    config = ResFiTLearnerConfig(
        state_dim=3,
        action_dim=2,
        actor_hidden_dims=(8, 8),
        critic_hidden_dims=(8, 8),
        critic_ensemble_size=4,
        target_critic_subsample_size=2,
    )
    learner = ResFiTLearner(config, torch.device("cpu"))
    learner.critic_target = ConstantHeadCritic([5.0, 2.0, 7.0, 3.0])
    batch = _batch(batch_size=5)

    original_randperm = torch.randperm

    def fixed_randperm(n, *args, **kwargs):
        assert n == 4
        return torch.tensor([3, 1, 0, 2], device=kwargs.get("device"))

    torch.randperm = fixed_randperm
    try:
        target_min, _ = learner._target_min_q(batch)
    finally:
        torch.randperm = original_randperm
    torch.testing.assert_close(target_min, torch.full((5,), 2.0))

    learner.critic = ConstantHeadCritic([0.0, 2.0, 4.0, 6.0])
    actor_metrics = learner.update_actor(batch)
    assert abs(actor_metrics["train/actor_loss"] + 3.0) < 1e-6


def _fill_replay(seed):
    replay = PrimitiveReplayBuffer(state_dim=2, action_dim=1, capacity=32, seed=seed)
    for value in range(16):
        replay.add(
            AggregatedTransition(
                state=np.asarray([value, value + 1], dtype=np.float32),
                base_action=np.asarray([0.0], dtype=np.float32),
                action=np.asarray([0.1], dtype=np.float32),
                reward=0.0,
                discount=0.99,
                next_state=np.asarray([value + 1, value + 2], dtype=np.float32),
                next_base_action=np.asarray([0.0], dtype=np.float32),
                terminated=False,
                truncated=False,
            )
        )
    return replay


def test_online_and_mixed_batch_contracts():
    online = _fill_replay(1)
    offline = _fill_replay(2)
    online_batch = sample_training_batch(online, None, "online", 8, torch.device("cpu"))
    assert online_batch.states.shape == (8, 2)
    mixed_batch = sample_training_batch(online, offline, "mixed", 8, torch.device("cpu"))
    assert mixed_batch.states.shape == (8, 2)
