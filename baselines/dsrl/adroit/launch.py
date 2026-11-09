"""Public configuration adapter for the native Adroit DSRL learner."""

from pathlib import Path
from baselines.dsrl.adroit import train


def run(config, cli):
    if config.get("resume"):
        raise ValueError(
            "This Adroit DSRL entry starts a new run; checkpoint evaluation is available separately."
        )
    cfg = train._default_cfg()
    cfg.base_policy_path = config.get("checkpoint", "")
    cfg.logdir = str(Path(config["output"]).expanduser().resolve())
    cfg.seed = config["seed"]
    cfg.env_name = f"AdroitHand{config['task'].title()}-v1"
    cfg.env.name = cfg.env_name
    cfg.obs_dim, cfg.action_dim = {
        "door": (39, 28),
        "hammer": (46, 26),
        "pen": (45, 24),
        "relocate": (39, 30),
    }[config["task"]]
    cfg.act_steps = cfg.diffusion_horizon_steps = cfg.execute_steps = config["chunk_size"]
    cfg.env.n_envs = 10
    cfg.env.terminate_on_success = True
    cfg.env.timeout_on_horizon = True
    cfg.train.utd = 30
    cfg.train.noise_critic_grad_steps = 10
    cfg.online_eval_interval_env_steps = 200000
    for key, value in config.get("training", {}).items():
        train._set_nested(cfg, key if "." in key or hasattr(cfg, key) else "train." + key, value)
    if cli.check:
        print("Adroit DSRL imports and configuration are valid.")
        return
    if not Path(cfg.base_policy_path).exists():
        raise FileNotFoundError(cfg.base_policy_path)
    train.main(cfg)
