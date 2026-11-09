"""Register the task-specific Cosmos training recipe on top of its LIBERO model."""

import copy
import json
import os
from pathlib import Path


def make_config():
    from cosmos_policy._src.imaginaire.utils import checkpoint_db

    recipe = json.loads(Path(os.environ["EXRL_COSMOS_RECIPE"]).read_text())
    original = checkpoint_db.get_checkpoint_path

    def local_checkpoint(uri, *args, **kwargs):
        if "Cosmos-Predict2-2B-Video2World" in str(uri):
            return (
                recipe["tokenizer_checkpoint"] if "tokenizer" in str(uri) else recipe["checkpoint"]
            )
        # Registration also names unused model variants. Leave those references
        # unresolved; only the selected experiment loads weights.
        return uri

    checkpoint_db.get_checkpoint_path = local_checkpoint
    try:
        from cosmos_policy.config.config import make_config as base_config

        config = base_config()
    finally:
        checkpoint_db.get_checkpoint_path = original
    from cosmos_policy.config.experiment.cosmos_policy_experiment_configs import (
        cosmos_predict2_2b_480p_libero,
    )
    from hydra.core.config_store import ConfigStore

    node = copy.deepcopy(cosmos_predict2_2b_480p_libero)
    training = recipe["training"]
    node.trainer.max_iter = training["num_train_steps"]
    node.trainer.grad_accum_iter = training["gradient_accumulation"]
    node.checkpoint.load_path = recipe["checkpoint"]
    node.checkpoint.save_iter = training["save_interval"]
    node.model.config.tokenizer.vae_pth = recipe["tokenizer_checkpoint"]
    node.dataloader_train.batch_size = training["batch_size"]
    node.dataloader_train.num_workers = training["num_workers"]
    dataset = node.dataloader_train.dataset
    dataset.data_dir = recipe["dataset"]
    dataset.t5_text_embeddings_path = str(Path(recipe["dataset"]) / "t5_embeddings.pkl")
    dataset.chunk_size = recipe["chunk_size"]
    dataset.rollout_data_dir = ""
    dataset.demonstration_sampling_prob = 1.0
    dataset.success_rollout_sampling_prob = 1.0
    node.dataloader_train.sampler.dataset = dataset
    node.job.name = "exrl_libero_h50"
    node.job.group = "base_policy"
    node.job.wandb_mode = "disabled"
    ConfigStore.instance().store(
        group="experiment", package="_global_", name="exrl_libero_h50", node=node
    )
    inference = copy.deepcopy(node)
    inference.model.config.sde.sigma_max = 80
    inference.model.config.sde.sigma_min = 4
    inference.job.name = "exrl_libero_h50_inference"
    ConfigStore.instance().store(
        group="experiment", package="_global_", name="exrl_libero_h50_inference", node=inference
    )
    return config
