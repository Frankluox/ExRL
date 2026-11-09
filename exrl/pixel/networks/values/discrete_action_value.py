from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

from exrl.pixel.networks.mlp import MLP


class DiscreteActionValue(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, observations, training: bool = False) -> jnp.ndarray:
        return MLP(
            (*self.hidden_dims, self.action_dim),
            activations=self.activations,
            use_layer_norm=self.use_layer_norm,
        )(observations, training=training)


class DiscreteActionValueEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_layer_norm: bool = True
    num_qs: int = 2

    @nn.compact
    def __call__(self, observations, training: bool = False):
        VmapCritic = nn.vmap(
            DiscreteActionValue,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        return VmapCritic(
            self.hidden_dims,
            self.action_dim,
            activations=self.activations,
            use_layer_norm=self.use_layer_norm,
        )(observations, training)
