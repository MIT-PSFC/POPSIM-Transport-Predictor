"""The leaky output of the p_oh / p_rad submodules keeps a network pushed far negative learning."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transport_study.modules.normalization import InputNormalizer, RawNormalizer
from transport_study.modules.power_balance.p_oh.module import OhmicPower
from transport_study.modules.power_balance.p_rad.module import RadiatedPower
from transport_study.modules.power_balance.scalar_power import POWER_LEAK_SLOPE

# Far enough below the softplus knee that its own gradient, sigmoid(-50) ~ 2e-22, is dead
DEAD_NN_OUT = -50.0


@pytest.mark.parametrize(
    ("module_cls", "output_name", "scale_name"),
    [
        (OhmicPower, "power_ohm_MW_pred", "ip_MA"),
        (RadiatedPower, "power_radiated_MW_pred", "n_e_line_average_1e20"),
    ],
)
def test_very_negative_network_keeps_its_gradient(module_cls, output_name, scale_name):
    """With the network output at -50 a plain softplus would leave no gradient.
    The leak keeps d(power) / d(last bias) at POWER_LEAK_SLOPE times the scale (|Ip| or the line density)."""
    module = module_cls.init(nn_width=4, nn_depth=1, prng_seed=0, normalizer=RawNormalizer())
    last_layer = module.nn.layers[-1]
    zero_weights = jnp.zeros_like(last_layer.weight)
    dead_bias = jnp.full_like(last_layer.bias, DEAD_NN_OUT)
    module = eqx.tree_at(lambda m: m.nn.layers[-1].weight, module, zero_weights)
    # Negative Ip checks the scale is its magnitude
    inputs = InputNormalizer.Inputs(
        ip_MA=-1.2,
        b_geo=5.4,
        geometric_axis_r=0.68,
        minor_radius=0.22,
        elongation=1.6,
        n_e_line_average_1e20=1.5,
        power_additional_MW=2.0,
        ds_source_idx=0,
    )

    def _power(bias):
        biased_module = eqx.tree_at(lambda m: m.nn.layers[-1].bias, module, bias)
        output = biased_module(inputs)
        return getattr(output, output_name)

    power_gradient_fn = jax.grad(_power)
    gradient = power_gradient_fn(dead_bias)
    scale_value = getattr(inputs, scale_name)
    scale_magnitude = abs(scale_value)
    np.testing.assert_allclose(gradient, POWER_LEAK_SLOPE * scale_magnitude, rtol=1e-6)
