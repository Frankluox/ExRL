"""Run in OpenPI's Python environment; no weights or robot are loaded."""

import numpy as np
import pytest

pytest.importorskip("openpi")

from exrl.config import load_config
from exrl.paths import ROOT
from exrl.pretraining.openpi import build_config
from exrl.pretraining.piper import PiperInputs, PiperOutputs


@pytest.mark.parametrize("family", ["pi0", "pi05", "pi0fast"])
def test_native_libero_model_and_data_config(family, tmp_path):
    recipe = load_config(ROOT / f"configs/pretrain/libero/{family}.yaml")
    native = build_config(recipe, tmp_path)
    assert native.model.action_horizon == 50
    assert native.data.extra_delta_transform == (family != "pi05")
    assert native.batch_size == recipe["training"]["batch_size"]
    assert not native.overwrite


def test_folding_freezes_vision_language_and_trains_action_expert(tmp_path):
    recipe = load_config(ROOT / "configs/pretrain/real_robot/clothing_folding.yaml")
    native = build_config(recipe, tmp_path)
    frozen = native.freeze_filter
    assert frozen(("PaliGemma", "img", "embedding", "kernel"), None)
    assert frozen(("PaliGemma", "llm", "layers", "mlp", "kernel"), None)
    assert not frozen(("PaliGemma", "llm", "layers", "mlp_1", "kernel"), None)
    assert not frozen(("action_in_proj", "kernel"), None)
    assert native.ema_decay is None and native.num_train_steps == 10000


def test_piper_camera_mask_and_action_output():
    rgb = np.full((224, 224, 3), 127, np.uint8)
    batch = PiperInputs()(
        dict(
            {
                "observation/state": np.zeros(7),
                "observation/image": rgb,
                "observation/wrist_image": rgb,
            },
            actions=np.ones((50, 7)),
            prompt="Fold the clothing.",
        )
    )
    assert batch["image_mask"]["base_0_rgb"]
    assert not batch["image_mask"]["right_wrist_0_rgb"]
    assert batch["actions"].shape == (50, 7)
    result = PiperOutputs()(dict(actions=np.ones((50, 32))))
    assert result["actions"].shape == (50, 7)
