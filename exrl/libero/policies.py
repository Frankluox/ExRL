"""OpenPI model definitions and checkpoint normalization for LIBERO."""

from pathlib import Path
import json
import numpy as np


def normalization(checkpoint, extra_delta_transform):
    path = Path(checkpoint).expanduser().resolve()
    files = list((path / "assets").glob("**/norm_stats.json"))
    if len(files) != 1:
        raise ValueError("Expected one assets/**/norm_stats.json in the base checkpoint")
    payload = json.loads(files[0].read_text())
    values = payload.get("norm_stats", payload)
    stats = {
        f"{prefix}_{field}": np.asarray(values[key][field], np.float32)[:size]
        for key, prefix, size in [("state", "state", 8), ("actions", "action", 7)]
        for field in ["mean", "std"]
    }
    if any(not np.isfinite(value).all() for value in stats.values()):
        raise ValueError("Non-finite checkpoint normalization statistics")
    if any(np.any(stats[key] < 0) for key in ("state_std", "action_std")):
        raise ValueError("Normalization standard deviations cannot be negative")
    stats.update(clip=10.0, extra_delta_transform=extra_delta_transform)
    return stats, str(files[0].parent.relative_to(path / "assets"))


def load_openpi_policy(family, checkpoint):
    from openpi.models import pi0_config, pi0_fast
    from openpi.policies import policy_config
    from openpi.training import config as training

    path = Path(checkpoint).expanduser().resolve()
    if not (path / "params").is_dir():
        raise FileNotFoundError(path / "params")
    extra_delta = family in ("pi0", "pi0fast")
    stats, asset_id = normalization(path, extra_delta)
    if family == "pi0fast":
        model = pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=50, max_token_len=320)
        sampling = dict(max_decoding_steps=320, temperature=0.0)
    else:
        model = pi0_config.Pi0Config(
            pi05=family == "pi05", action_horizon=50, discrete_state_input=False
        )
        sampling = dict(num_steps=10)
    config = training.TrainConfig(
        name=f"exrl_{family}_libero",
        model=model,
        data=training.LeRobotLiberoDataConfig(
            repo_id=asset_id,
            assets=training.AssetsConfig(asset_id=asset_id),
            base_config=training.DataConfig(prompt_from_task=True),
            extra_delta_transform=extra_delta,
        ),
    )
    return policy_config.create_trained_policy(config, path, sample_kwargs=sampling), stats


class RemotePolicy:
    """Keep policy sampling reproducible while inference runs in OpenPI's environment."""

    def __init__(self, family, host, port, seed=0):
        import jax
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(host=host, port=port)
        metadata = self.client.get_server_metadata()
        if metadata.get("family") != family or metadata.get("horizon") != 50:
            raise ValueError("Policy server does not match the requested base policy")
        self.family = family
        self._rng = jax.random.key(seed)

    def infer(self, observation):
        import jax

        self._rng, key = jax.random.split(self._rng)
        request = dict(observation)
        if self.family != "pi0fast":
            request["exrl_noise"] = np.asarray(jax.random.normal(key, (50, 32)))
        return self.client.infer(request)

    def close(self):
        socket = getattr(self.client, "_ws", None)
        if socket is not None:
            socket.close()


def connect_policy(family, checkpoint, host=None, port=None):
    import os

    stats, _ = normalization(checkpoint, family in ("pi0", "pi0fast"))
    policy = RemotePolicy(
        family,
        host or os.environ.get("EXRL_POLICY_HOST", "127.0.0.1"),
        int(port or os.environ.get("EXRL_POLICY_PORT", 8000)),
    )
    return policy, stats
