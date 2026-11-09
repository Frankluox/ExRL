"""Connect the native DSRL, Policy Decorator and PA-RL learners to public assets."""

from pathlib import Path
import importlib

from exrl.paths import ROOT, VENDOR


def configuration(config):
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    path = Path(config["native_config"])
    if not path.is_absolute():
        path = ROOT / path
    cfg = OmegaConf.load(path)
    data = Path(config.get("data", "data/" + config["task"])).expanduser().resolve()
    checkpoint = Path(config.get("checkpoint", "models/base")).expanduser().resolve()
    output = Path(config["output"]).expanduser().resolve()
    cfg.update(
        base_policy_path=str(checkpoint),
        normalization_path=str(data / "normalization.npz"),
        action_stats_path=str(data / (config["task"] + "_lowdim_stats.npz")),
        offline_dataset_path=str(data / "train.npz"),
        dppo_path=str(VENDOR / "dppo"),
        act_lerobot_path=str(ROOT / "vendor/lerobot/src"),
        robomimic_env_cfg_path=str(data / (config["task"] + ".json")),
        seed=config["seed"],
        logdir=str(output),
        log_dir=str(output),
        act_steps=config["chunk_size"],
        horizon_steps=config["chunk_size"],
        diffusion_horizon_steps=config["chunk_size"],
        base_policy_horizon_steps=config["chunk_size"],
        execute_steps=config["chunk_size"],
    )
    if config["base_policy"] == "dp" and checkpoint.exists():
        metadata = checkpoint.parent.parent / ".hydra/config.yaml"
        if not metadata.exists():
            raise FileNotFoundError(
                f"Keep the pretrained policy metadata beside the checkpoint: {metadata}"
            )
        base = OmegaConf.load(metadata)
        # Use the exact architecture and diffusion schedule that produced the weights.
        target = "actor" if config["method"] == "parl" else "network"
        cfg.model[target] = base.model.network
        for key in (
            "denoising_steps",
            "predict_epsilon",
            "denoised_clip_value",
            "randn_clip_value",
        ):
            if key in base.model:
                cfg.model[key] = base.model[key]
    if config["method"] == "dsrl":
        cfg.latest_checkpoint = dict(
            interval_env_steps=500000, path=str(output / "checkpoint/latest.ckpt")
        )
    for key, value in config.get("training", {}).items():
        key = key if "." in key or key in cfg else "train." + key
        OmegaConf.update(cfg, key, value, merge=True, force_add=False)
    if config.get("resume"):
        key = {
            "dsrl": "resume.path",
            "policy_decorator": "resume_from",
            "parl": "train.resume_ckpt",
        }[config["method"]]
        OmegaConf.update(
            cfg, key, str(Path(config["resume"]).expanduser().resolve()), force_add=True
        )
    OmegaConf.resolve(cfg)
    return cfg


def run(config, cli):
    from omegaconf import OmegaConf

    cfg = configuration(config)
    method = config["method"]
    module = importlib.import_module(
        {
            "dsrl": "baselines.dsrl.train",
            "policy_decorator": "baselines.policy_decorator.train",
            "parl": "baselines.parl."
            + ("dp_train" if config["base_policy"] == "dp" else "act_train"),
        }[method]
    )
    if cli.check:
        print("Native learner imports and resolved configuration are valid.")
        return
    for path in (cfg.base_policy_path, cfg.normalization_path, cfg.robomimic_env_cfg_path):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    output = Path(cfg.logdir)
    output.mkdir(parents=True, exist_ok=True)
    (output / ".hydra").mkdir(exist_ok=True)
    OmegaConf.save(cfg, output / ".hydra/config.yaml")
    if method == "parl":
        import hydra

        hydra.utils.get_class(cfg._target_)(cfg).run()
    else:
        module.main.__wrapped__(cfg)
