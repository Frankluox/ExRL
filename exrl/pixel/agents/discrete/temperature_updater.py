from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState


def update_temperature(
    temp: TrainState, entropy: float, target_entropy: float, max_temperature: float = -1.0
) -> Tuple[TrainState, Dict[str, float]]:

    def temperature_loss_fn(temp_params):
        temperature = temp.apply_fn({"params": temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {"temperature": temperature, "temperature_loss": temp_loss}

    grads, info = jax.grad(temperature_loss_fn, has_aux=True)(temp.params)
    new_temp = temp.apply_gradients(grads=grads)
    max_temperature = jnp.asarray(max_temperature, dtype=jnp.float32)
    clip_enabled = max_temperature > 0.0
    safe_max_temperature = jnp.maximum(max_temperature, jnp.asarray(1e-8, dtype=jnp.float32))
    max_log_temp = jnp.log(safe_max_temperature)
    new_temp = new_temp.replace(
        params=jax.tree_util.tree_map(
            lambda x: jnp.where(
                clip_enabled,
                jnp.minimum(x, max_log_temp.astype(x.dtype)),
                x,
            ),
            new_temp.params,
        )
    )
    info["temperature"] = jnp.where(
        clip_enabled,
        jnp.minimum(info["temperature"], safe_max_temperature.astype(info["temperature"].dtype)),
        info["temperature"],
    )

    return new_temp, info
