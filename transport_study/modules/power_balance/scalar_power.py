"""Output activation shared by the scalar power submodules (p_oh, p_rad)."""

import jax
from jaxtyping import Array, ArrayLike

# Slope of the linear leak added to the softplus, the gradient never drops below it
POWER_LEAK_SLOPE = 0.01


def leaky_softplus(x: ArrayLike) -> Array:
    """softplus(x) + POWER_LEAK_SLOPE x.

    The softplus gradient is sigmoid(x) ~ exp(x) far below 0,
    so a network pushed to very negative outputs stops learning.
    The leak keeps the gradient at least POWER_LEAK_SLOPE everywhere.
    The output goes negative below x ~ -3.4, by POWER_LEAK_SLOPE per unit of x.

    P_oh signal is so terrible on DIII-D that this is necessary to prevent
    the exnihilo module from just throwing up its hands and ouputting 0 all the time
    """
    softplus_x = jax.nn.softplus(x)
    return softplus_x + POWER_LEAK_SLOPE * x
