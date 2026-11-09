"""A real image-critic update and full optimizer checkpoint round-trip."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
from flax.core import freeze
from real_robot.training import make_agent, save, restore, Replay, stack_rows


@pytest.mark.parametrize("actor", [False, True])
def test_pixel_update_and_resume(tmp_path, actor):
    import pickle

    cfg = dict(
        seed=4,
        chunk_size=3,
        learner=dict(
            hidden_dims=[16, 16],
            cnn_features=[4, 4],
            cnn_strides=[2, 1],
            encoder_type="small",
            num_qs=2,
            num_cameras=1,
            random_crop=False,
            color_jitter=False,
            aug_next=False,
            target_policy_mode="actor" if actor else "q_greedy",
            cql_alpha=0.3,
            critic_use_layer_norm=False,
            latent_dim=8,
        ),
    )
    obs = dict(
        pixels=np.zeros((16, 16, 3, 1), np.uint8),
        state=np.zeros((7, 1), np.float32),
        actions=np.zeros((3, 7), np.float32),
    )
    agent = make_agent(cfg, obs)
    batch = freeze(
        dict(
            observations=stack_rows([obs] * 2),
            next_observations=stack_rows([obs] * 2),
            actions=np.array([0, 2], np.int32),
            rewards=np.array([-1.0, 0.0], np.float32),
            discount=np.array([0.99, 0.98], np.float32),
            masks=np.array([1.0, 0.0], np.float32),
            durations=np.array([1.0, 3.0], np.float32),
        )
    )
    values = agent.update(batch, update_actor=actor, update_alpha=actor)
    assert all(np.isfinite(np.asarray(v)).all() for v in values.values())
    replay = Replay(8)
    rng = np.random.default_rng(4)
    save(tmp_path / "latest.pkl", agent, replay, dict(updates=1), rng, cfg)
    restored = make_agent(cfg, obs)
    restore(pickle.loads((tmp_path / "latest.pkl").read_bytes()), restored)
    a = agent.update(batch, update_actor=actor, update_alpha=actor)
    b = restored.update(batch, update_actor=actor, update_alpha=actor)
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-6)
