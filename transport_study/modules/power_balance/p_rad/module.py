import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from popsim.math_utils import soft_clip
from popsim.module_base import TimeIndepModule

from transport_study.modules.normalization import InputNormalizer


class RadiatedPower(TimeIndepModule):
    """Model that predicts radiated power from plasma parameters.
    Essentially ne20 * network, bounded within some min/max values.

    Consumes PHYSICAL inputs and normalizes them internally with its own
    normalizer, so its NN weights and normalization statistics always travel
    together through checkpoints (standalone training, submodule restore, and
    transfer learning all round-trip both).
    """

    nn: eqx.Module
    normalizer: InputNormalizer
    min_val: float = eqx.field(static=True)
    max_val: float = eqx.field(static=True)

    # Physical inputs plus the device index selecting per-device normalization stats
    Inputs = InputNormalizer.Inputs

    @chex.dataclass
    class Output:
        P_rad_MW_pred: float
        debug_info: dict

    def __call__(self, inputs: InputNormalizer.Inputs) -> Output:
        if not isinstance(inputs, InputNormalizer.Inputs):
            # Standalone training feeds a dataloader batch of raw dataset variables
            inputs = InputNormalizer.inputs_from_dict(inputs)

        features = self.normalizer(inputs)
        nn_out = self.nn(features.to_vec())
        bounded_out = soft_clip(jnp.abs(inputs.ne20_line_avg * nn_out), self.min_val, self.max_val, sharpness=8).squeeze()

        output = RadiatedPower.Output(
            P_rad_MW_pred=bounded_out,
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
        min_val: float,
        max_val: float,
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
        return cls(nn=nn, normalizer=normalizer, min_val=min_val, max_val=max_val)
