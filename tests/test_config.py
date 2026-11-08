"""Configuration validation and portable saved experiments."""

from pathlib import Path

import pytest

from exrl.config import load_config, resolve, save_config
from exrl.paths import ROOT


@pytest.mark.parametrize("path", sorted((ROOT / "configs/robomimic").rglob("*.yaml")))
def test_all_experiments_resolve(path):
    args, config = resolve(load_config(path))
    assert args.max_chunk_len == config["chunk_size"]
    assert args.utd / args.n_envs / args.train_freq == config["training"]["utd"]
    assert args.cql_alpha >= 0


def test_utd_tracks_environment_count():
    config = load_config(ROOT / "configs/robomimic/can_dp_q.yaml")
    args, _ = resolve(config, overrides=["n_envs=4"])
    assert args.utd == 1
    with pytest.raises(ValueError, match="positive integer"):
        resolve(config, overrides=["n_envs=3"])


def test_unknown_options_and_boolean_strings_fail():
    config = load_config(ROOT / "configs/robomimic/can_dp_q.yaml")
    with pytest.raises(ValueError, match="Unknown"):
        resolve(config, overrides=["n_env=4"])
    with pytest.raises(ValueError, match="true or false"):
        resolve(config, overrides=['dp_compile="false"'])


def test_saved_settings_reproduce_evaluation_arguments(tmp_path):
    config = load_config(ROOT / "configs/robomimic/can_act_demo_bc.yaml")
    args, config = resolve(
        config,
        data=tmp_path / "data",
        base_policy=tmp_path / "model",
        output=tmp_path / "run",
        seed=43,
        overrides=["hidden_dims=64,64"],
    )
    path = tmp_path / "experiment.yaml"
    save_config(config, path)
    restored, _ = resolve(load_config(path))
    assert vars(restored) == vars(args)
    assert restored.bc_offline_demo_npz == str(tmp_path / "data/train.npz")


def test_defaults_are_relative_and_cycles_fail(tmp_path):
    (tmp_path / "base.yaml").write_text("training:\n  gamma: 0.99\n")
    (tmp_path / "child.yaml").write_text(
        "defaults: [base.yaml]\ntraining:\n  n_envs: 4\n"
    )
    assert load_config(tmp_path / "child.yaml")["training"] == {
        "gamma": 0.99,
        "n_envs": 4,
    }
    (tmp_path / "base.yaml").write_text("defaults: [child.yaml]\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "child.yaml")
