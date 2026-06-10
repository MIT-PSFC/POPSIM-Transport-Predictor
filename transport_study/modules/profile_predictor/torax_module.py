import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Array
from popsim import TimeIndepModule
from popsim.ml.rtd_mlp import Activation, RtdMLP
from torax import ToraxConfig
from torax import experimental as torax_experimental
from torax._src import jax_utils as torax_jax_utils
from torax._src.geometry import geometry as torax_geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.orchestration.step_function import SimulationStepFn
from torax._src.torax_pydantic import torax_pydantic

from transport_study.modules.profile_predictor.module import (
    Inputs,
    Outputs,
)


def _build_circular_geometry_jax(
    R_major: jax.Array,
    a_minor: jax.Array,
    B_0: jax.Array,
    elongation_LCFS: jax.Array,
    torax_mesh: torax_pydantic.Grid1D,
    rho_hires_norm_np: np.ndarray,
) -> torax_geometry.Geometry:
    """Circular geometry builder using JAX ops for differentiability.

    Mirrors _build_circular_geometry from torax but uses jnp.* so that
    R_major, a_minor, B_0, elongation_LCFS remain JAX-traced through the
    geometry construction.
    """
    rho_face_norm = jnp.array(torax_mesh.face_centers)
    rho_norm = jnp.array(torax_mesh.cell_centers)
    rho_hires_norm = jnp.array(rho_hires_norm_np)

    rho_b = a_minor
    rho_face = rho_face_norm * rho_b
    rho = rho_norm * rho_b
    rho_hires = rho_hires_norm * rho_b

    Phi = jnp.pi * B_0 * rho**2
    Phi_face = jnp.pi * B_0 * rho_face**2

    elongation = 1.0 + rho_norm * (elongation_LCFS - 1.0)
    elongation_face = 1.0 + rho_face_norm * (elongation_LCFS - 1.0)
    elongation_hires = 1.0 + rho_hires_norm * (elongation_LCFS - 1.0)

    volume = 2.0 * jnp.pi**2 * R_major * rho**2 * elongation
    volume_face = 2.0 * jnp.pi**2 * R_major * rho_face**2 * elongation_face
    area = jnp.pi * rho**2 * elongation
    area_face = jnp.pi * rho_face**2 * elongation_face

    vpr = 4.0 * jnp.pi**2 * R_major * rho * elongation * rho_b + volume / elongation * (elongation_LCFS - 1.0)
    vpr_face = 4.0 * jnp.pi**2 * R_major * rho_face * elongation_face * rho_b + volume_face / elongation_face * (elongation_LCFS - 1.0)
    spr = 2.0 * jnp.pi * rho * elongation * rho_b + area / elongation * (elongation_LCFS - 1.0)
    spr_face = 2.0 * jnp.pi * rho_face * elongation_face * rho_b + area_face / elongation_face * (elongation_LCFS - 1.0)

    delta_face = jnp.zeros_like(rho_face)

    g0 = vpr / rho_b
    g0_face = vpr_face / rho_b
    g1 = vpr**2 / rho_b**2
    g1_face = vpr_face**2 / rho_b**2
    g2 = g1 / R_major**2
    g2_face = g1_face / R_major**2
    g3 = 1.0 / (R_major**2 * (1.0 - (rho / R_major) ** 2) ** 1.5)
    g3_face = 1.0 / (R_major**2 * (1.0 - (rho_face / R_major) ** 2) ** 1.5)

    n = rho_norm.shape[0]
    n_face = rho_face_norm.shape[0]
    n_hires = rho_hires_norm.shape[0]
    F = jnp.ones(n) * R_major * B_0
    F_face = jnp.ones(n_face) * R_major * B_0
    F_hires = jnp.ones(n_hires) * B_0 * R_major
    J = jnp.ones(n)
    J_face = jnp.ones(n_face)

    g2g3_over_rhon = 4.0 * jnp.pi**2 * vpr * g3 / (J * R_major)
    g2g3_over_rhon_face = 4.0 * jnp.pi**2 * vpr_face * g3_face / (J_face * R_major)

    volume_hires = 2.0 * jnp.pi**2 * R_major * rho_hires**2 * elongation_hires
    area_hires = jnp.pi * rho_hires**2 * elongation_hires
    vpr_hires = 4.0 * jnp.pi**2 * R_major * rho_hires * elongation_hires * rho_b + volume_hires / elongation_hires * (elongation_LCFS - 1.0)
    spr_hires = 2.0 * jnp.pi * rho_hires * elongation_hires * rho_b + area_hires / elongation_hires * (elongation_LCFS - 1.0)
    g3_hires = 1.0 / (R_major**2 * (1.0 - (rho_hires / R_major) ** 2) ** 1.5)
    g2g3_over_rhon_hires = 4.0 * jnp.pi**2 * vpr_hires * g3_hires * B_0 / F_hires

    R_out = R_major + rho
    R_out_face = R_major + rho_face
    R_in = R_major - rho
    R_in_face = R_major - rho_face

    epsilon = (R_out - R_in) / (R_out + R_in)
    epsilon_face = (R_out_face - R_in_face) / (R_out_face + R_in_face)
    gm4 = B_0**-2 * (1.0 + 1.5 * epsilon**2)
    gm4_face = B_0**-2 * (1.0 + 1.5 * epsilon_face**2)
    gm5 = B_0**2 / jnp.sqrt(1.0 - epsilon**2)
    gm5_face = B_0**2 / jnp.sqrt(1.0 - epsilon_face**2)

    return torax_geometry.Geometry(
        geometry_type=torax_geometry.GeometryType.CIRCULAR,
        torax_mesh=torax_mesh,
        Phi=Phi,
        Phi_face=Phi_face,
        R_major=R_major,
        a_minor=rho_b,
        B_0=B_0,
        volume=volume,
        volume_face=volume_face,
        area=area,
        area_face=area_face,
        vpr=vpr,
        vpr_face=vpr_face,
        spr=spr,
        spr_face=spr_face,
        delta_face=delta_face,
        g0=g0,
        g0_face=g0_face,
        g1=g1,
        g1_face=g1_face,
        g2=g2,
        g2_face=g2_face,
        g3=g3,
        g3_face=g3_face,
        gm4=gm4,
        gm4_face=gm4_face,
        gm5=gm5,
        gm5_face=gm5_face,
        g2g3_over_rhon=g2g3_over_rhon,
        g2g3_over_rhon_face=g2g3_over_rhon_face,
        g2g3_over_rhon_hires=g2g3_over_rhon_hires,
        F=F,
        F_face=F_face,
        F_hires=F_hires,
        R_in=R_in,
        R_in_face=R_in_face,
        R_out=R_out,
        R_out_face=R_out_face,
        elongation=elongation,
        elongation_face=elongation_face,
        spr_hires=spr_hires,
        rho_hires_norm=rho_hires_norm,
        rho_hires=rho_hires,
        Phi_b_dot=jnp.asarray(0.0),
        _z_magnetic_axis=jnp.asarray(0.0),
    )


def _run_loop_jit_with_geo(
    step_fn: SimulationStepFn,
    input_state,
    previous_post_processed_outputs,
    runtime_params_overrides,
    geo_provider,
    max_steps: int,
    debug: bool = False,
    wrap_body_in_checkpoint: bool = False,
):
    """Local copy of torax_experimental.run_loop_jit that also accepts geo_overrides.

    Mirrors torax._src.orchestration.jit_run_loop.run_loop_jit (the recommended
    fully-JITted simulation loop pattern from the TORAX docs) but:
      - Accepts a geo_overrides argument so per-sample geometry can be passed
        in without retracing the outer step_fn.
      - Skips the per-step history buffers (we only need the final state).
      - Optionally wraps the body in jax.checkpoint so reverse-mode AD recomputes
        per-step activations rather than storing all max_steps copies.
    """

    def cond(carry):
        i, current_state, _ = carry
        is_done = step_fn.is_done(current_state.t)
        return jnp.logical_and(i < max_steps, jnp.logical_not(is_done))

    def body(carry):
        i, prev_state, prev_post = carry
        current_state, post_processed = step_fn(
            prev_state,
            prev_post,
            runtime_params_overrides=runtime_params_overrides,
            geo_overrides=geo_provider,
        )
        cp = current_state.core_profiles
        if debug:
            jax.debug.print(
                "[scan i={i}] t={t} dt={dt} err={err} Te[min,max]=[{te_lo},{te_hi}] ne[min,max]=[{ne_lo},{ne_hi}]",
                i=i,
                t=current_state.t,
                dt=current_state.dt,
                err=current_state.solver_numeric_outputs.solver_error_state,
                te_lo=cp.T_e.value.min(),
                te_hi=cp.T_e.value.max(),
                ne_lo=cp.n_e.value.min(),
                ne_hi=cp.n_e.value.max(),
            )
        return i + 1, current_state, post_processed

    if wrap_body_in_checkpoint:
        body = jax.checkpoint(body, prevent_cse=False)

    _, output_state, post_processed = torax_jax_utils.while_loop_bounded(
        cond,
        body,
        (0, input_state, previous_post_processed_outputs),
        max_steps,
    )
    return output_state, post_processed


class ProfilePredictorTorax(TimeIndepModule):
    psigrid: tuple = eqx.field(static=True)

    nn_transport: RtdMLP
    nn_sources: RtdMLP

    step_fn: SimulationStepFn = eqx.field(static=True)

    # Static mesh info for JAX-differentiable geometry construction
    _face_centers: tuple = eqx.field(static=True)
    _rho_hires_norm: tuple = eqx.field(static=True)
    # Upper bound on sub-steps in fixed_time_step; enables scan-based (differentiable) loop
    max_steps: int = eqx.field(static=True)

    def __init__(
        self,
        nn_width: int,
        nn_depth: int,
        psigrid: tuple,
        torax_config: ToraxConfig | dict,
        key: jax.random.PRNGKey,
    ):
        key, subkey_transport, subkey_sources = jax.random.split(key, 3)
        self.nn_transport = RtdMLP(
            in_size=9,
            out_size=4,  # chi_i, chi_e, D_e, V_e
            width_size=nn_width,
            depth=nn_depth,
            activation=Activation.RELU,
            key=subkey_transport,
        )
        self.nn_sources = RtdMLP(
            in_size=9,
            out_size=1,  # S_total
            width_size=nn_width,
            depth=nn_depth,
            activation=Activation.RELU,
            key=subkey_sources,
        )

        if isinstance(torax_config, dict):
            torax_config = ToraxConfig.from_dict(torax_config)

        self.step_fn = torax_experimental.make_step_fn(torax_config)
        self.psigrid = psigrid

        static_geo = self.step_fn.geometry_provider(0.0)
        self._face_centers = tuple(static_geo.torax_mesh.face_centers.tolist())
        self._rho_hires_norm = tuple(np.array(static_geo.rho_hires_norm).tolist())

        # With the fixed time-step calculator, steps to cover t_final are
        # deterministic: ceil((t_final - t_initial) / fixed_dt). Add 1 for the
        # clipped final step that lands exactly on t_final.
        numerics = self.step_fn.runtime_params_provider.numerics
        fixed_dt = float(numerics.fixed_dt.get_value(0.0))
        self.max_steps = int(np.ceil((numerics.t_final - numerics.t_initial) / fixed_dt)) + 1

    def __call__(self, inputs: Inputs | xr.Dataset, debug: bool = False) -> Outputs:
        if isinstance(inputs, xr.Dataset):
            inputs = Inputs(
                Ip=inputs["Ip_MA"].data,
                B0=inputs["B0"].data,
                betan=inputs["betan"].data,
                ne20=inputs["ne20_edge"].data,
                R0=inputs["R0"].data,
                a_minor=inputs["a_minor"].data,
                kappa=inputs["kappa"].data,
                delta_top=inputs["delta_top"].data,
                delta_bot=inputs["delta_bot"].data,
                psi=jnp.array(self.psigrid),
            )

        # Get transport and source terms from neural networks, bounded to
        # physical ranges so the TORAX solver stays stable during training
        # Approximate ranges taken from DIII-D study and TFTR
        # https://iopscience-iop-org.libproxy.mit.edu/article/10.1088/0029-5515/38/4/301/pdf
        # https://iopscience-iop-org.libproxy.mit.edu/article/10.1088/0029-5515/39/1/309/pdf
        #   chi_i: 0.1 - 10 m^2/s
        #   chi_e: 0.1 - 20 m^2/s
        #   D_e:   0.1 - 3 m^2/s   (nonzero floor prevents advection-only blowup)
        #   V_e:   -5 - 5 m/s      (signed pinch)
        #   S_total: 0 - 10 (x 1e21 below)
        nn_inputs = inputs.nn_inputs
        nn_transport_out = self.nn_transport(nn_inputs)
        chi_i = 0.1 + 4.9 * jax.nn.sigmoid(nn_transport_out[0:1])
        chi_e = 0.1 + 9.9 * jax.nn.sigmoid(nn_transport_out[1:2])
        D_e = 0.1 + 1.9 * jax.nn.sigmoid(nn_transport_out[2:3])
        V_e = 5.0 * jnp.tanh(nn_transport_out[3:4])
        S_total = jax.nn.softplus(self.nn_sources(nn_inputs))
        if debug:
            jax.debug.print(
                "[nn] chi_i={ci} chi_e={ce} D_e={d} V_e={v} S_total={s} nn_in={ni}",
                ci=chi_i,
                ce=chi_e,
                d=D_e,
                v=V_e,
                s=S_total,
                ni=nn_inputs,
            )

        # Build runtime params override
        ip_update = torax_experimental.TimeVaryingScalarUpdate(
            value=jnp.atleast_1d(inputs.Ip * 1e6),
        )
        nbar_update = torax_experimental.TimeVaryingScalarUpdate(value=jnp.atleast_1d(inputs.fGW))
        _rho = jnp.array([1.0])
        chi_i_update = torax_experimental.TimeVaryingArrayUpdate(value=jnp.broadcast_to(chi_i[:, jnp.newaxis], (1, 1)), rho_norm=_rho)
        chi_e_update = torax_experimental.TimeVaryingArrayUpdate(value=jnp.broadcast_to(chi_e[:, jnp.newaxis], (1, 1)), rho_norm=_rho)
        D_e_update = torax_experimental.TimeVaryingArrayUpdate(value=jnp.broadcast_to(D_e[:, jnp.newaxis], (1, 1)), rho_norm=_rho)
        V_e_update = torax_experimental.TimeVaryingArrayUpdate(value=jnp.broadcast_to(V_e[:, jnp.newaxis], (1, 1)), rho_norm=_rho)
        S_total_update = torax_experimental.TimeVaryingScalarUpdate(value=S_total * 1e21)

        new_provider = self.step_fn.runtime_params_provider.update_provider_from_mapping(
            {
                "profile_conditions.Ip": ip_update,
                "profile_conditions.nbar": nbar_update,
                "transport_model.chi_i": chi_i_update,
                "transport_model.chi_e": chi_e_update,
                "transport_model.D_e": D_e_update,
                "transport_model.V_e": V_e_update,
                "sources.gas_puff.S_total": S_total_update,
            }
        )

        # Build JAX-differentiable geometry from per-sample inputs
        face_centers_np = np.array(self._face_centers)
        torax_mesh = torax_pydantic.Grid1D(face_centers=face_centers_np)
        rho_hires_norm_np = np.array(self._rho_hires_norm)
        geo = _build_circular_geometry_jax(
            R_major=inputs.R0,
            a_minor=inputs.a_minor,
            B_0=inputs.B0,
            elongation_LCFS=inputs.kappa,
            torax_mesh=torax_mesh,
            rho_hires_norm_np=rho_hires_norm_np,
        )
        geo_provider = geometry_provider_lib.ConstantGeometryProvider(geo=geo)

        # Get initial state and run simulation
        initial_state, initial_post = torax_experimental.get_initial_state_and_post_processed_outputs(
            step_fn=self.step_fn,
            runtime_params_overrides=new_provider,
            geometry_overrides=geo_provider,
        )
        cp0 = initial_state.core_profiles
        if debug:
            jax.debug.print(
                "[init] Te[min,max]=[{te_lo},{te_hi}] ne[min,max]=[{ne_lo},{ne_hi}]",
                te_lo=cp0.T_e.value.min(),
                te_hi=cp0.T_e.value.max(),
                ne_lo=cp0.n_e.value.min(),
                ne_hi=cp0.n_e.value.max(),
            )

        state, _post = _run_loop_jit_with_geo(
            step_fn=self.step_fn,
            input_state=initial_state,
            previous_post_processed_outputs=initial_post,
            runtime_params_overrides=new_provider,
            geo_provider=geo_provider,
            max_steps=self.max_steps,
            debug=debug,
        )

        ne = state.core_profiles.n_e.value
        te = state.core_profiles.T_e.value

        # Interpolate onto psigrid (rho_norm for circular geometry)
        rho_norm_grid = jnp.array(torax_mesh.cell_centers)
        ne_interp = jnp.interp(inputs.psi, rho_norm_grid, ne)
        te_interp = jnp.interp(inputs.psi, rho_norm_grid, te)

        return Outputs(
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

    @classmethod
    def init(
        cls,
        psigrid: Array,
        torax_config: ToraxConfig,
        nn_width: int,
        nn_depth: int,
        prng_seed: int,
    ):
        psigrid_tuple = tuple(psigrid.tolist())
        return cls(
            nn_width=nn_width,
            nn_depth=nn_depth,
            psigrid=psigrid_tuple,
            torax_config=torax_config,
            key=jax.random.PRNGKey(prng_seed),
        )
