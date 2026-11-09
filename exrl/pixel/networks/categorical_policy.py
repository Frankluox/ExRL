from typing import Callable, Optional, Sequence

import numpy as np

import distrax
import flax.linen as nn
import jax.numpy as jnp

from exrl.pixel.networks import MLP
from exrl.pixel.networks.constants import default_init, xavier_init


class CategoricalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_rate: Optional[float] = None
    init_scale: Optional[float] = 1.0
    use_layer_norm: bool = False
    init_method: str = "default"
    head_init_scale: Optional[float] = 1e-2

    def _validate_min_action_id(self, min_action_id):
        if np.ndim(min_action_id) != 0:
            return
        min_action_id = int(min_action_id)
        if not (0 <= min_action_id < int(self.action_dim)):
            raise ValueError(
                f"min_action_id out of range: {min_action_id} for action_dim={self.action_dim}"
            )

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        training: bool = False,
        min_action_id: int = 0,
        return_logits: bool = False,
    ):
        outputs = MLP(
            self.hidden_dims,
            activations=self.activations,
            activate_final=True,
            dropout_rate=self.dropout_rate,
            init_scale=self.init_scale,
            use_layer_norm=self.use_layer_norm,
        )(observations, training=training)

        if self.init_method == "xavier":
            logits = nn.Dense(self.action_dim, kernel_init=xavier_init())(outputs)
        else:
            logits = nn.Dense(
                self.action_dim,
                kernel_init=default_init(self.head_init_scale),
            )(outputs)

        self._validate_min_action_id(min_action_id)
        min_action_id = jnp.asarray(min_action_id, dtype=jnp.int32)
        action_ids = jnp.arange(self.action_dim, dtype=jnp.int32)
        if min_action_id.ndim == 0:
            illegal_mask = action_ids < min_action_id
        else:
            illegal_mask = action_ids[None, :] < min_action_id[..., None]
        logits = jnp.where(illegal_mask, jnp.full_like(logits, -1e9), logits)

        if return_logits:
            return logits
        return distrax.Categorical(logits=logits)
