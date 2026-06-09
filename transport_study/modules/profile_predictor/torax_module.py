import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr
from jaxtyping import Array
from popsim.ml.rtd_mlp import Activation, RtdMLP
from torax import ToraxConfig
from torax import experimental as torax_experimental
from torax._src.orchestration.step_function import SimulationStepFn

from transport_study.modules.profile_predictor.module import (
    Inputs,
    Outputs,
    ProfilePredictor,
)


class ProfilePredictorTorax(ProfilePredictor):
    nn_transport: RtdMLP
    nn_sources: RtdMLP

    step_fn: SimulationStepFn = eqx.field(static=True)

    def __init__(
        self,
        nn_width: int,
        nn_depth: int,
        psigrid: tuple,
        torax_config: ToraxConfig,
        key: jax.random.PRNGKey,
    ):
        key, subkey_transport, subkey_sources = jax.random.split(key, 3)
        self.nn_transport = RtdMLP(
            in_size=9,
            out_size=4,  # chi_i, chi_e, D_e, V_e
            width=nn_width,
            depth=nn_depth,
            activation=Activation.RELU,
            key=subkey_transport,
        )
        self.nn_sources = RtdMLP(
            in_size=9,
            out_size=1,  # S_total
            width=nn_width,
            depth=nn_depth,
            activation=Activation.RELU,
            key=subkey_sources,
        )

        self.step_fn = torax_experimental.make_step_fn(torax_config)
        self.psigrid = psigrid

    def __call__(self, inputs: Inputs | xr.Dataset, debug: bool = False) -> Outputs:
        if isinstance(inputs, xr.Dataset):
            inputs = Inputs(
                Ip=inputs["Ip_MA"].data,
                B0=inputs["B0"].data,
                betan=inputs["betan"].data,
                ne20=inputs["ne20"].data,
                R0=inputs["R0"].data,
                a_minor=inputs["a_minor"].data,
                kappa=inputs["kappa"].data,
                delta_top=inputs["delta_top"].data,
                delta_bot=inputs["delta_bot"].data,
                psi=jnp.array(self.psigrid),
            )

        # Get transport and source terms from neural networks
        nn_inputs = inputs.nn_inputs
        chi_i, chi_e, D_e, V_e = self.nn_transport(nn_inputs)
        S_total = self.nn_sources(nn_inputs)

        # Assign input and NN-predicted transport/source terms to the step function

        # Profile conditions
        ip_update = torax_experimental.TimeVaryingScalarUpdate(
            value=inputs.Ip * 1e6,
        )
        nbar_update = torax_experimental.TimeVaryingScalarUpdate(value=inputs.fGW)
        # Geometry (Might be handled differently?)
        R_major_update = torax_experimental.TimeVaryingScalarUpdate(value=inputs.R0)
        a_minor_update = torax_experimental.TimeVaryingScalarUpdate(value=inputs.a_minor)
        B0_update = torax_experimental.TimeVaryingScalarUpdate(value=inputs.B0)
        elongation_LCFS_update = torax_experimental.TimeVaryingScalarUpdate(value=inputs.kappa)
        # Transport
        chi_i_update = torax_experimental.TimeVaryingArrayUpdate(value=jnp.array([chi_i]))
        chi_e_update = torax_experimental.TimeVaryingArrayUpdate(value=jnp.array([chi_e]))
        D_e_update = torax_experimental.TimeVaryingArrayUpdate(value=jnp.array([D_e]))
        V_e_update = torax_experimental.TimeVaryingArrayUpdate(value=jnp.array([V_e]))
        # Sources
        S_total_update = torax_experimental.TimeVaryingScalarUpdate(
            value=S_total * 1e20  # Assuming S_total has shape (1,) and is in units of 1e20 particles/s
        )

        new_provider = self.step_fn.runtime_params_provider.update_provider_from_mapping(
            {
                # Profile conditions
                "profile_conditions.Ip": ip_update,
                "profile_conditions.nbar": nbar_update,
                # Geometry
                "geometry.R_major": R_major_update,
                "geometry.a_minor": a_minor_update,
                "geometry.B_0": B0_update,
                "geometry.elongation_LCFS": elongation_LCFS_update,
                # Transport
                "transport.chi_i": chi_i_update,
                "transport.chi_e": chi_e_update,
                "transport.D_e": D_e_update,
                "transport.V_e": V_e_update,
                # Sources
                "sources.gas_puff.S_total": S_total_update,
            }
        )

        sim_states, _post_processed_outputs, final_i = torax_experimental.run_loop_jit(
            step_fn=self.step_fn,
            max_steps=10,
            runtime_params_overrides=new_provider,
        )

        ne = sim_states.core_profiles.n_e.value[final_i]
        te = sim_states.core_profiles.T_e.value[final_i]

        # Interpolate onto input psi grid
        ne_interp = jnp.interp(inputs.psi, self.step_fn.geometry_provider.psi_grid, ne)
        te_interp = jnp.interp(inputs.psi, self.step_fn.geometry_provider.psi_grid, te)

        out = Outputs(
            ne=xr.DataArray(
                data=ne_interp,
                dims=("psi_n",),
                coords={"psi_n": list(self.psigrid)},
            ),
            te=xr.DataArray(
                data=te_interp,
                dims=("psi_n",),
                coords={"psi_n": list(self.psigrid)},
            ),
        )

        return out

    @classmethod
    def init(
        cls,
        psigrid: Array,
        torax_config: ToraxConfig,
        nn_width: int,
        nn_depth: int,
        in_size: int,
        prng_seed: int,
    ):
        psigrid_tuple = tuple(psigrid.tolist())
        return cls(
            nn_width=nn_width,
            nn_depth=nn_depth,
            in_size=in_size,
            psigrid=psigrid_tuple,
            torax_config=torax_config,
            key=jax.random.PRNGKey(prng_seed),
        )
