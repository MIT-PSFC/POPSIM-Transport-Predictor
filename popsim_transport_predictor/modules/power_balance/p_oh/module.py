from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

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
    input_format_fn: Callable[[dict], OhmicPower.Inputs] = eqx.field(static=True)

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
    def get_input_format_fn(cls, normalization_method: str) -> Callable[[dict], Inputs]:
        """Format the inputs according to the normalization method.
        TODO(ZanderKeith): for time-dependent models this is handled by the ModuleEvalEnv,
        but we don't have an equivalnt for time-independent models, meaning they need to do it themselves.
        Kind of a weird break in abstraction, might want to fix that at some point.
        """

        def _format_inputs_raw(inputs) -> OhmicPower.Inputs:
            if isinstance(inputs, xr.Dataset):
                return OhmicPower.Inputs(
                    Ip_MA_real=inputs["Ip_MA"].data,
                    Ip_MA_nn=inputs["Ip_MA"].data,
                    B0_nn=inputs["B0"].data,
                    R0_nn=inputs["R0"].data,
                    a_minor_nn=inputs["a_minor"].data,
                    kappa_nn=inputs["kappa"].data,
                    ne20_nn=inputs["ne20_line_avg"].data,
                    P_aux_nn=inputs["P_aux_MW"].data,
                )
            else:
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
            if isinstance(inputs, xr.Dataset):
                # Yeah all the names get messed up, sloppy formatting on my part
                # it all still works though, fix if you have time TODO(ZanderKeith)
                return OhmicPower.Inputs(
                    Ip_MA_real=inputs["Ip_MA"].data,
                    Ip_MA_nn=inputs["Ip_MA"].data,
                    B0_nn=inputs["q95"].data,
                    R0_nn=inputs["epsilon"].data,
                    a_minor_nn=inputs["aB0"].data,
                    kappa_nn=inputs["kappa"].data,
                    ne20_nn=inputs["f_G"].data,
                    P_aux_nn=inputs["surface_power_density"].data,
                )
            else:
                return OhmicPower.Inputs(
                    Ip_MA_real=inputs["Ip_MA"],
                    Ip_MA_nn=inputs["Ip_MA"],
                    B0_nn=inputs["q95"],
                    R0_nn=inputs["epsilon"],
                    a_minor_nn=inputs["aB0"],
                    kappa_nn=inputs["kappa"],
                    ne20_nn=inputs["f_G"],
                    P_aux_nn=inputs["surface_power_density"],
                )

        def _format_inputs_z_score(inputs) -> OhmicPower.Inputs:
            if isinstance(inputs, xr.Dataset):
                return OhmicPower.Inputs(
                    Ip_MA_real=inputs["Ip_MA"].data,
                    Ip_MA_nn=inputs["Ip_MA_z"].data,
                    B0_nn=inputs["B0_z"].data,
                    R0_nn=inputs["R0_z"].data,
                    a_minor_nn=inputs["a_minor_z"].data,
                    kappa_nn=inputs["kappa_z"].data,
                    ne20_nn=inputs["ne20_line_avg_z"].data,
                    P_aux_nn=inputs["P_aux_MW_z"].data,
                )
            else:
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
            if isinstance(inputs, xr.Dataset):
                return OhmicPower.Inputs(
                    Ip_MA_real=inputs["Ip_MA"].data,
                    Ip_MA_nn=inputs["Ip_MA_coral"].data,
                    B0_nn=inputs["B0_coral"].data,
                    R0_nn=inputs["R0_coral"].data,
                    a_minor_nn=inputs["a_minor_coral"].data,
                    kappa_nn=inputs["kappa_coral"].data,
                    ne20_nn=inputs["ne20_line_avg_coral"].data,
                    P_aux_nn=inputs["P_aux_MW_coral"].data,
                )
            else:
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

        if normalization_method == "raw":
            return _format_inputs_raw
        elif normalization_method == "physics":
            return _format_inputs_physics
        elif normalization_method == "z_score":
            return _format_inputs_z_score
        elif normalization_method == "coral":
            return _format_inputs_coral
        else:
            raise ValueError(f"Unknown normalization method: {normalization_method}")

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
        normalization_method: str,
    ) -> OhmicPower:
        nn = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=nn_width,
            depth=nn_depth,
            key=jax.random.PRNGKey(prng_seed),
        )
        input_format_fn = cls.get_input_format_fn(normalization_method)
        return cls(
            nn=nn, min_val=min_val, max_val=max_val, input_format_fn=input_format_fn
        )
