"""Tiny CPU pretraining, checkpoint loading, and base-policy BC updates."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch

from exrl.paths import ROOT
from prepare_data import prepare


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    work = tmp_path_factory.mktemp("pretraining")
    source = work / "tiny.hdf5"
    rng = np.random.default_rng(0)
    with h5py.File(source, "w") as handle:
        demos = handle.create_group("data")
        demos.attrs["env_args"] = (ROOT / "configs/env/can.json").read_text()
        for index in range(2):
            demo = demos.create_group(f"demo_{index}")
            demo.create_dataset("actions", data=rng.uniform(-1, 1, (56, 7)))
            demo.create_dataset("rewards", data=np.r_[np.zeros(55), 1.0])
            demo.attrs["num_samples"] = 56
            obs = demo.create_group("obs")
            for key, dim in [
                ("robot0_eef_pos", 3),
                ("robot0_eef_quat", 4),
                ("robot0_gripper_qpos", 2),
                ("object", 14),
            ]:
                obs.create_dataset(key, data=rng.normal(size=(56, dim)))
    directory = work / "data"
    result = prepare(source, "can", directory)
    assert result["episodes"] == 2 and result["transitions"] == 112
    return directory


@pytest.mark.integration
@pytest.mark.parametrize("base", ["dp", "act"])
def test_pretrain_reload_and_imitation(base, data, tmp_path):
    output = tmp_path / base
    settings = {
        "dp": [
            "train.n_epochs=1",
            "train.batch_size=16",
            "train.lr_scheduler.first_cycle_steps=2",
            "train.lr_scheduler.warmup_steps=0",
            "model.network.mlp_dims=[32,32,32]",
        ],
        "act": [
            "max_steps=1",
            "stop_after_steps=1",
            "batch_size=2",
            "num_workers=0",
            "dim_model=32",
            "dim_feedforward=64",
            "n_encoder_layers=1",
        ],
    }
    command = [
        sys.executable,
        str(ROOT / "pretrain.py"),
        "--config",
        str(ROOT / f"configs/pretrain/{base}_can_h50.yaml"),
        "--data",
        str(data),
        "--output",
        str(output),
        "--device",
        "cpu",
    ]
    for setting in settings[base]:
        command.extend(["--set", setting])
    subprocess.run(
        command,
        check=True,
        cwd=ROOT,
        timeout=180,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "2",
            "ROBOMIMIC_ENABLE_LANG": "0",
            "HF_HUB_OFFLINE": "1",
        },
    )
    torch.set_num_threads(2)
    if base == "dp":
        from exrl.runners.dp_q.policy_loader import (
            load_dsrl_policy_wrapper,
            infer_lowdim_dims_from_stats,
        )
        from exrl.runners.dp_q.online_bc import (
            BaseBCDatasetSampler,
            canonical_module_state_dict,
            load_module_state_dict_compat,
        )

        keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
        stats = str(data / "can_lowdim_stats.npz")
        wrapper, metadata = load_dsrl_policy_wrapper(
            str(output / "checkpoint/state_1.pt"),
            torch.device("cpu"),
            stats,
            keys,
            infer_lowdim_dims_from_stats(stats, keys),
            7,
        )
        assert metadata["schedule_defaults"]["train_denoising_steps"] == 100
    else:
        from exrl.runners.act_q.policy_loader import load_act_policy_wrapper
        from exrl.runners.act_q.online_bc import BaseBCDatasetSampler

        wrapper, metadata = load_act_policy_wrapper(
            ckpt_path=str(output / "final"), device=torch.device("cpu"), task="can"
        )
        assert metadata["chunk_size"] == 50
    policy = wrapper.policy
    sampler = BaseBCDatasetSampler(
        dataset_path=str(data / "train.npz"),
        horizon_steps=50,
        cond_steps=1,
        device=torch.device("cpu"),
        padded=False,
    )
    batch = sampler.sample(2)
    model = (
        policy.nets["policy"]["noise_pred_net"] if base == "dp" else policy.policy_model
    )
    model.requires_grad_(True)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if base == "dp":
        loss, anchor = policy.diffusion_bc_loss_terms(
            batch.actions, batch.conditions, teacher_noise_pred_net=teacher
        )
    else:
        loss, anchor = policy.act_bc_loss_terms(
            batch.actions, batch.conditions, teacher_policy_model=teacher
        )
    assert torch.isfinite(loss + anchor)
    (loss + anchor).backward()
    optimizer.step()
    assert any(
        not torch.equal(initial[key], value)
        for key, value in model.state_dict().items()
    )
    assert all(
        torch.equal(initial[key], value) for key, value in teacher.state_dict().items()
    )
    assert all(param.grad is None for param in teacher.parameters())
    if base == "dp":
        restored = copy.deepcopy(policy)
        load_module_state_dict_compat(
            restored, canonical_module_state_dict(policy), strict=True
        )
    else:
        restored = copy.deepcopy(model)
        restored.load_state_dict(model.state_dict(), strict=True)
