from collections.abc import Callable

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr
from jaxtyping import ArrayLike
from popsim.math_utils import soft_clip
from popsim.module_base import TimeIndepModule


class OhmicPower(TimeIndepModule):
    """Model that predicts ohmic heating power from plasma parameters.
    Essentially Ip * network, bounded within some min/max values.
    """

    nn: eqx.Module
    min_val: float = eqx.field(static=True)
    max_val: float = eqx.field(static=True)
    input_format_fn: Callable = eqx.field(static=True)

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
        P_aux_nn: float

    @chex.dataclass
    class Output:
        P_oh_MW_pred: float
        debug_info: dict

    def __call__(self, inputs: Inputs) -> Output:
        if not isinstance(inputs, OhmicPower.Inputs):
            inputs = self.input_format_fn(inputs)

        arr = jnp.array(
            [
                inputs.Ip_MA_nn,
                inputs.B0_nn,
                inputs.R0_nn,
                inputs.a_minor_nn,
                inputs.kappa_nn,
                inputs.ne20_nn,
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
    def get_input_format_fn(cls, data_normalization: str) -> Callable[[dict], Inputs]:
        """Format the inputs according to the normalization method.
        TODO(ZanderKeith): for time-dependent models this is handled by the ModuleEvalEnv,
        but we don't have an equivalnt for time-independent models, meaning they need to do it themselves.
        Kind of a weird break in abstraction, might want to fix that at some point.
        """

        def _formalize_inputs(inputs: xr.Dataset) -> dict[str, ArrayLike]:
            return {var: inputs[var].data for var in inputs.data_vars}

        def _format_inputs_raw(inputs) -> OhmicPower.Inputs:
            inputs = _formalize_inputs(inputs)
            return OhmicPower.Inputs(
                Ip_MA_real=inputs["Ip_MA"],
                Ip_MA_nn=inputs["Ip_MA"],
                B0_nn=inputs["B0"],
                R0_nn=inputs["R0"],
                a_minor_nn=inputs["a_minor"],
                kappa_nn=inputs["kappa"],
                ne20_nn=inputs["ne20_line_avg"],
                P_aux_nn=inputs["P_aux_MW"],
            )

        def _format_inputs_physics(inputs) -> OhmicPower.Inputs:
            inputs = _formalize_inputs(inputs)
            # Yeah all the names get messed up,
            # need to fix this
            return OhmicPower.Inputs(
                Ip_MA_real=inputs["Ip_MA"],
                Ip_MA_nn=inputs["Ip_MA"],
                B0_nn=inputs["q_star"],
                R0_nn=inputs["epsilon"],
                a_minor_nn=inputs["aB0"],
                kappa_nn=inputs["kappa"],
                ne20_nn=inputs["f_G"],
                P_aux_nn=inputs["surface_power_density"],
            )

        def _format_inputs_z_score(inputs) -> OhmicPower.Inputs:
            inputs = _formalize_inputs(inputs)
            return OhmicPower.Inputs(
                Ip_MA_real=inputs["Ip_MA"],
                Ip_MA_nn=inputs["Ip_MA_z"],
                B0_nn=inputs["B0_z"],
                R0_nn=inputs["R0_z"],
                a_minor_nn=inputs["a_minor_z"],
                kappa_nn=inputs["kappa_z"],
                ne20_nn=inputs["ne20_line_avg_z"],
                P_aux_nn=inputs["P_aux_MW_z"],
            )

        def _format_inputs_coral(inputs) -> OhmicPower.Inputs:
            inputs = _formalize_inputs(inputs)
            return OhmicPower.Inputs(
                Ip_MA_real=inputs["Ip_MA"],
                Ip_MA_nn=inputs["Ip_MA_coral"],
                B0_nn=inputs["B0_coral"],
                R0_nn=inputs["R0_coral"],
                a_minor_nn=inputs["a_minor_coral"],
                kappa_nn=inputs["kappa_coral"],
                ne20_nn=inputs["ne20_line_avg_coral"],
                P_aux_nn=inputs["P_aux_MW_coral"],
            )

        if data_normalization == "raw":
            return _format_inputs_raw
        elif data_normalization == "physics":
            return _format_inputs_physics
        elif data_normalization == "z_score":
            return _format_inputs_z_score
        elif data_normalization == "coral":
            return _format_inputs_coral
        else:
            raise ValueError(f"Unknown normalization method: {data_normalization}")

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
        data_normalization: str,
    ) -> "OhmicPower":
        nn = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=nn_width,
            depth=nn_depth,
            key=jax.random.PRNGKey(prng_seed),
        )
        input_format_fn = cls.get_input_format_fn(data_normalization)
        return cls(
            nn=nn, min_val=min_val, max_val=max_val, input_format_fn=input_format_fn
        )
