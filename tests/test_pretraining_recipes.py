"""Check dataset conventions and the portable base-policy recipes."""

import numpy as np
import pytest

from exrl.config import load_config
from exrl.paths import ROOT
from exrl.pretraining.openpi import validate_recipe
from exrl.pretraining.piper_data import episode_arrays


@pytest.mark.parametrize("family", ["pi0", "pi05", "pi0fast"])
def test_libero_recipe(family):
    recipe = load_config(ROOT / f"configs/pretrain/libero/{family}.yaml")
    validate_recipe(recipe)
    assert recipe["chunk_size"] == 50


@pytest.mark.parametrize("task,epochs", [("door", 2), ("hammer", 2), ("pen", 2), ("relocate", 10)])
def test_adroit_epoch_selection(task, epochs):
    recipe = load_config(ROOT / f"configs/pretrain/adroit/{task}_dp.yaml")
    assert recipe["task"] == task
    assert recipe["training"]["num_epochs"] == epochs
    assert recipe["training"]["action_horizon"] == 32


def test_piper_crop_keeps_rgb_and_absolute_actions():
    image = np.arange(2 * 8 * 10 * 3, dtype=np.uint8).reshape(2, 8, 10, 3)
    actions = np.ones((2, 7), np.float32)
    source = dict(state=np.zeros((2, 7)), actions=actions, image=image, wrist_image=image)
    output = episode_arrays(source, 4)
    np.testing.assert_array_equal(output["image"], image[:, 2:6, 3:7])
    np.testing.assert_array_equal(output["actions"], actions)
    with pytest.raises(ValueError, match="gripper"):
        episode_arrays(dict(source, actions=actions * 2))


@pytest.mark.parametrize(
    "task", ["stack_cube", "clothing_folding", "power_insertion", "parts_organization"]
)
def test_robot_recipe_uses_finite_physical_deadline(task):
    config = load_config(ROOT / f"configs/real_robot/{task}.yaml")
    assert np.isfinite(config["episode_seconds"]) and config["episode_seconds"] > 0
    assert "episode_steps" not in config
    assert "reward" not in config
