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
    nn_edge: RtdMLP

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
        key, subkey_transport, subkey_sources, subkey_edge = jax.random.split(key, 4)
        self.nn_transport = RtdMLP(
            in_size=9,
            out_size=5,  # chi_e_i_ratio, chi_D_ratio, VR_D_ratio, alpha, chi_stiff (CGM free parameters)
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
        self.nn_edge = RtdMLP(
            in_size=9,
            out_size=2,  # edge density fraction, edge temperature fraction
            width_size=nn_width,
            depth=nn_depth,
            activation=Activation.RELU,
            key=subkey_edge,
        )

        if isinstance(torax_config, dict):
            torax_config = ToraxConfig.from_dict(torax_config)

        self.step_fn = torax_experimental.make_step_fn(torax_config)
        # Coerce to tuple: arrays in static fields break pytree metadata
        # equality (ambiguous truth value) when two module instances coexist
        self.psigrid = tuple(np.asarray(psigrid).tolist())

        static_geo = self.step_fn.geometry_provider(0.0)
        self._face_centers = tuple(static_geo.torax_mesh.face_centers.tolist())
        self._rho_hires_norm = tuple(np.array(static_geo.rho_hires_norm).tolist())

        # With the fixed time-step calculator, steps to cover t_final are
        # deterministic: ceil((t_final - t_initial) / fixed_dt). Add 1 for the
        # clipped final step that lands exactly on t_final.
        numerics = self.step_fn.runtime_params_provider.numerics
        fixed_dt = float(numerics.fixed_dt.get_value(0.0))
        self.max_steps = int(np.ceil((numerics.t_final - numerics.t_initial) / fixed_dt)) + 1

    def _coerce_inputs(self, inputs: Inputs | xr.Dataset) -> Inputs:
        if isinstance(inputs, xr.Dataset):
            inputs = Inputs(
                Ip=inputs["Ip_MA"].data,
                B0=inputs["B0"].data,
                betan=inputs["betan"].data,
                ne20_line_avg=inputs["ne20_line_avg"].data,
                R0=inputs["R0"].data,
                a_minor=inputs["a_minor"].data,
                kappa=inputs["kappa"].data,
                delta_top=inputs["delta_top"].data,
                delta_bot=inputs["delta_bot"].data,
                psi=jnp.array(self.psigrid),
            )
        return inputs

    def _nn_coefficients(self, inputs: Inputs, debug: bool = False) -> dict:
        # Get the free parameters of the Critical Gradient Model (CGM) and the
        # particle source from neural networks, bounded to physical ranges so
        # the TORAX solver stays stable during training. The critical gradient
        # itself is computed by TORAX from the evolving state and geometry
        # (known inputs); only the dimensionless ratios are learned.
        #   chi_e_i_ratio: 0.5 - 5   (chi_e = chi_i / ratio; ITG turbulence > 1)
        #   chi_D_ratio:   1 - 20    (D_e = chi_i / ratio; must stay positive)
        #   VR_D_ratio:    -5 - 5    (R0*V_e/D_e; negative peaks the density profile)
        #   alpha:         1 - 3     (exponent of the chi power law, TORAX default 2)
        #   chi_stiff:     0.5 - 5   (stiffness parameter, TORAX default 2)
        #   S_total: 0 - 10 (x 1e21 below)
        nn_inputs = inputs.nn_inputs
        nn_transport_out = self.nn_transport(nn_inputs)
        chi_e_i_ratio = 0.5 + 4.5 * jax.nn.sigmoid(nn_transport_out[0:1])
        chi_D_ratio = 1.0 + 19.0 * jax.nn.sigmoid(nn_transport_out[1:2])
        VR_D_ratio = 5.0 * jnp.tanh(nn_transport_out[2:3])
        alpha = 1.0 + 2.0 * jax.nn.sigmoid(nn_transport_out[3:4])
        chi_stiff = 0.5 + 4.5 * jax.nn.sigmoid(nn_transport_out[4:5])
        S_total = jax.nn.softplus(self.nn_sources(nn_inputs))

        # Edge boundary conditions as NN-predicted fractions in (0, 1):
        #   n_e_right_bc = fraction * line-averaged density
        #   T_e_right_bc = fraction * te_approx (beta-derived temperature guess,
        #                  same scaling trick as the shape-init predictors)
        # A fixed edge density BC above the target profile acts as an infinite
        # particle source, so the BC must scale with the requested density.
        # The negative bias on the temperature fraction makes random-init edge
        # temperatures small (~0.007 * te_approx, a few hundred eV): te_approx
        # is a beta-derived overestimate, and a hot edge BC flattens the profile
        # relative to itself, which keeps the critical gradient model
        # subcritical (chi = 0) and kills the gradient to the transport network.
        nn_edge_out = self.nn_edge(nn_inputs)
        ne_right_bc = jax.nn.sigmoid(nn_edge_out[0:1]) * inputs.ne20_line_avg
        te_right_bc = jax.nn.sigmoid(nn_edge_out[1:2] - 5.0) * inputs.te_approx
        if debug:
            jax.debug.print(
                "[nn] chi_e_i_ratio={cei} chi_D_ratio={cd} VR_D_ratio={vrd} alpha={a} chi_stiff={cs} S_total={s} nn_in={ni}",
                cei=chi_e_i_ratio,
                cd=chi_D_ratio,
                vrd=VR_D_ratio,
                a=alpha,
                cs=chi_stiff,
                s=S_total,
                ni=nn_inputs,
            )
        return {
            "chi_e_i_ratio": chi_e_i_ratio,
            "chi_D_ratio": chi_D_ratio,
            "VR_D_ratio": VR_D_ratio,
            "alpha": alpha,
            "chi_stiff": chi_stiff,
            "S_total": S_total,
            "n_e_right_bc": ne_right_bc,  # [1e20 m^-3]
            "T_e_right_bc": te_right_bc,  # [keV]
        }

    def _build_provider_and_geo(self, inputs: Inputs, coeffs: dict):
        chi_e_i_ratio = coeffs["chi_e_i_ratio"]
        chi_D_ratio = coeffs["chi_D_ratio"]
        VR_D_ratio = coeffs["VR_D_ratio"]
        alpha = coeffs["alpha"]
        chi_stiff = coeffs["chi_stiff"]
        S_total = coeffs["S_total"]
        ne_right_bc = coeffs["n_e_right_bc"]
        te_right_bc = coeffs["T_e_right_bc"]

        # Build runtime params override
        ip_update = torax_experimental.TimeVaryingScalarUpdate(
            value=jnp.atleast_1d(inputs.Ip * 1e6),
        )
        nbar_update = torax_experimental.TimeVaryingScalarUpdate(value=jnp.atleast_1d(inputs.fGW))
        chi_e_i_ratio_update = torax_experimental.TimeVaryingScalarUpdate(value=chi_e_i_ratio)
        chi_D_ratio_update = torax_experimental.TimeVaryingScalarUpdate(value=chi_D_ratio)
        VR_D_ratio_update = torax_experimental.TimeVaryingScalarUpdate(value=VR_D_ratio)
        S_total_update = torax_experimental.TimeVaryingScalarUpdate(value=S_total * 1e21)
        ne_right_bc_update = torax_experimental.TimeVaryingScalarUpdate(value=ne_right_bc * 1e20)
        te_right_bc_update = torax_experimental.TimeVaryingScalarUpdate(value=te_right_bc)

        # Initial temperature profiles ramp linearly from core = edge BC + 0.3 keV
        # down to the NN edge BC. This keeps the initial state strictly decreasing
        # and continuous with the BC for any NN output: a discontinuity at the LCFS
        # or an exactly-flat profile both NaN the solver under the critical
        # gradient model.
        t_init_value = jnp.concatenate([te_right_bc + 0.3, te_right_bc])[jnp.newaxis, :]
        t_init_update = torax_experimental.TimeVaryingArrayUpdate(
            value=t_init_value,
            rho_norm=jnp.array([0.0, 1.0]),
        )

        new_provider = self.step_fn.runtime_params_provider.update_provider_from_mapping(
            {
                "profile_conditions.Ip": ip_update,
                "profile_conditions.nbar": nbar_update,
                "profile_conditions.n_e_right_bc": ne_right_bc_update,
                # Assume the ion edge temperature matches the electron edge temperature
                "profile_conditions.T_e_right_bc": te_right_bc_update,
                "profile_conditions.T_i_right_bc": te_right_bc_update,
                "profile_conditions.T_e": t_init_update,
                "profile_conditions.T_i": t_init_update,
                "transport_model.chi_e_i_ratio": chi_e_i_ratio_update,
                "transport_model.chi_D_ratio": chi_D_ratio_update,
                "transport_model.VR_D_ratio": VR_D_ratio_update,
                # Plain float leaves in the provider: replaced with traced scalars
                # directly rather than via TimeVaryingScalarUpdate
                "transport_model.alpha": jnp.squeeze(alpha),
                "transport_model.chi_stiff": jnp.squeeze(chi_stiff),
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
        return new_provider, geo_provider

    @property
    def rho_norm_grid(self) -> np.ndarray:
        """Cell-center grid the TORAX core profiles live on.

        This is rho_norm, the normalized toroidal flux coordinate, NOT psi_n
        (normalized poloidal flux). Use _psi_n_cells to map a TORAX state's
        cell grid to psi_n.
        """
        face_centers = np.array(self._face_centers)
        return (face_centers[:-1] + face_centers[1:]) / 2.0

    @staticmethod
    def _psi_n_cells(core_profiles) -> jax.Array:
        """Normalized poloidal flux psi_n on the cell grid, from the evolved psi profile.

        psi_n = (psi - psi_axis) / (psi_lcfs - psi_axis), monotonic 0 -> 1 from
        axis to LCFS regardless of the sign of the psi gradient.
        """
        psi_face = core_profiles.psi.face_value()
        psi_axis = psi_face[0]
        psi_lcfs = psi_face[-1]
        return (core_profiles.psi.value - psi_axis) / (psi_lcfs - psi_axis)

    def __call__(self, inputs: Inputs | xr.Dataset, debug: bool = False) -> Outputs:
        inputs = self._coerce_inputs(inputs)
        coeffs = self._nn_coefficients(inputs, debug=debug)
        new_provider, geo_provider = self._build_provider_and_geo(inputs, coeffs)

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

        # Interpolate onto psigrid in the true psi_n coordinate: TORAX evolves
        # profiles on rho_norm (toroidal flux), so map the cell grid to
        # normalized poloidal flux using the evolved psi profile.
        psi_n_cells = self._psi_n_cells(state.core_profiles)
        ne_interp = jnp.interp(inputs.psi, psi_n_cells, ne)
        te_interp = jnp.interp(inputs.psi, psi_n_cells, te)

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

    def evolve(
        self,
        inputs: Inputs | xr.Dataset,
        prescribed: dict | None = None,
    ) -> tuple[list[dict], dict]:
        """Run the TORAX relaxation step by step, recording the core profiles after every step.

        Diagnostic counterpart of __call__: same provider/geometry construction, but drives
        step_fn in a plain Python loop instead of the jitted bounded while loop, so the
        intermediate states are observable. Not differentiable; do not use for training.

        Args:
            inputs: Same as __call__ (Inputs or single-timeslice xr.Dataset).
            prescribed: Optional dict overriding NN outputs. Keys from
                {chi_e_i_ratio, chi_D_ratio, VR_D_ratio, alpha, chi_stiff, S_total,
                n_e_right_bc, T_e_right_bc}; values are floats in the same units the NN
                outputs use (the CGM parameters are dimensionless, S_total in 1e21
                particles/s, n_e_right_bc in 1e20 m^-3, T_e_right_bc in keV).
                Keys not given (or None) keep the NN prediction.

        Returns:
            (steps, coeffs):
                steps: list of dicts, one per TORAX state including the initial one, with keys
                    t [s], ne20 [1e20 m^-3], te_keV [keV] and psi_n (the cell grid mapped to
                    normalized poloidal flux for that state).
                coeffs: the transport/source coefficients actually used, as floats.
        """
        inputs = self._coerce_inputs(inputs)
        coeffs = self._nn_coefficients(inputs)
        if prescribed is not None:
            unknown = set(prescribed) - set(coeffs)
            if unknown:
                raise ValueError(f"Unknown prescribed coefficients {unknown}, valid keys: {sorted(coeffs)}")
            for name, value in prescribed.items():
                if value is not None:
                    # Match the NN output dtype exactly: a weakly-typed scalar would get
                    # demoted to float32 inside TORAX and fail its float64 checks
                    coeffs[name] = jnp.full_like(coeffs[name], float(value))
        new_provider, geo_provider = self._build_provider_and_geo(inputs, coeffs)

        state, post = torax_experimental.get_initial_state_and_post_processed_outputs(
            step_fn=self.step_fn,
            runtime_params_overrides=new_provider,
            geometry_overrides=geo_provider,
        )

        def record(s) -> dict:
            cp = s.core_profiles
            return {
                "t": float(s.t),
                "ne20": np.asarray(cp.n_e.value) / 1e20,
                "te_keV": np.asarray(cp.T_e.value),
                "psi_n": np.asarray(self._psi_n_cells(cp)),
            }

        steps = [record(state)]
        for _ in range(self.max_steps):
            if bool(self.step_fn.is_done(state.t)):
                break
            state, post = self.step_fn(
                state,
                post,
                runtime_params_overrides=new_provider,
                geo_overrides=geo_provider,
            )
            steps.append(record(state))

        coeffs_out = {name: float(np.asarray(value).squeeze()) for name, value in coeffs.items()}
        return steps, coeffs_out

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
