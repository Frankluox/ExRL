import flax.linen as nn
import jax.numpy as jnp


def default_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def xavier_init():
    return nn.initializers.xavier_normal()


def kaiming_init():
    return nn.initializers.kaiming_normal()
