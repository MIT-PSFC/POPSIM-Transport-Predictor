import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr
from popsim.math_utils import soft_clip
from popsim.module_base import TimeIndepModule


class OhmicPower(TimeIndepModule):
    """Model that predicts ohmic heating power from plasma parameters.
    Essentially Ip * network, bounded within some min/max values.
    """

    nn: eqx.Module
    min_val: float = eqx.field(static=True)
    max_val: float = eqx.field(static=True)

    @chex.dataclass
    class Inputs:
        # Real-valued plasma current in MA
        Ip_MA_real: float
        # Neural network inputs
        Ip_MA_nn: float
        B0_nn: float
        R0_nn: float
        a_minor_nn: float
        kappa_nn: float
        ne20_nn: float
        energy_nn: float
        P_aux_nn: float

    @chex.dataclass
    class Output:
        P_oh_MW_pred: float
        debug_info: dict

    def __call__(self, inputs: Inputs) -> Output:
        if isinstance(inputs.Ip_MA_real, xr.DataArray):
            inputs = OhmicPower.Inputs(
                Ip_MA_real=inputs["Ip_MA_real"].data,
                Ip_MA_nn=inputs["Ip_MA_nn"].data,
                B0_nn=inputs["B0_nn"].data,
                R0_nn=inputs["R0_nn"].data,
                a_minor_nn=inputs["a_minor_nn"].data,
                kappa_nn=inputs["kappa_nn"].data,
                ne20_nn=inputs["ne20_nn"].data,
                energy_nn=inputs["energy_nn"].data,
                P_aux_nn=inputs["P_aux_nn"].data,
            )

        arr = jnp.array(
            [
                inputs.Ip_MA_nn,
                inputs.B0_nn,
                inputs.R0_nn,
                inputs.a_minor_nn,
                inputs.kappa_nn,
                inputs.ne20_nn,
                inputs.energy_nn,
                inputs.P_aux_nn,
            ],
        )
        nn_out = self.nn(arr)
        bounded_out = soft_clip(
            jnp.abs(inputs.Ip_MA_real * nn_out), self.min_val, self.max_val, sharpness=8
        ).squeeze()

        output = OhmicPower.Output(
            P_oh_MW_pred=bounded_out,
            debug_info={
                "nn_out": nn_out.squeeze(),  # Squeeze to match dimensions with P_oh_MW_pred
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
    ) -> "OhmicPower":
        nn = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=nn_width,
            depth=nn_depth,
            key=jax.random.PRNGKey(prng_seed),
        )
        return cls(nn=nn, min_val=min_val, max_val=max_val)
