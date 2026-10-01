import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from popsim.module_base import TimeIndepModule

from transport_study.modules.normalization import InputNormalizer


class RadiatedPower(TimeIndepModule):
    """Model that predicts radiated power from plasma parameters.

    P_rad = ne20 * softplus(network),
    i.e. the network predicts a positive per-density radiation factor

    Consumes PHYSICAL inputs and normalizes them internally with its own normalizer,
    so its NN weights and normalization statistics always travel together through checkpoints
    (standalone training, submodule restore, and transfer learning all round-trip both)
    """

    nn: eqx.Module
    normalizer: InputNormalizer

    # Physical inputs plus the device index selecting per-device normalization stats
    Inputs = InputNormalizer.Inputs

    @chex.dataclass
    class Output:
        power_radiated_MW_pred: float
        debug_info: dict

    def __call__(self, inputs: InputNormalizer.Inputs) -> Output:
        if not isinstance(inputs, InputNormalizer.Inputs):
            # Standalone training feeds a dataloader batch of raw dataset variables
            inputs = InputNormalizer.inputs_from_dict(inputs)

        features = self.normalizer(inputs)
        nn_out = self.nn(features.to_vec())
        p_rad = (jnp.abs(inputs.n_e_line_average_1e20) * jax.nn.softplus(nn_out)).squeeze()

        output = RadiatedPower.Output(
            power_radiated_MW_pred=p_rad,
            debug_info={
                "nn_out": nn_out.squeeze(),
            },
        )

        return output

    @classmethod
    def init(
        cls,
        in_size: int,
        out_size: int,
        nn_width: int,
        nn_depth: int,
        prng_seed: int,
        normalizer: InputNormalizer,
    ) -> "RadiatedPower":
        nn = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=nn_width,
            depth=nn_depth,
            key=jax.random.PRNGKey(prng_seed),
        )
        return cls(nn=nn, normalizer=normalizer)
