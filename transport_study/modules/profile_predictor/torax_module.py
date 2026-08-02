import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Array
from popsim import TimeIndepModule
from popsim.math_utils import smooth_clamp
from popsim.ml.rtd_mlp import Activation, RtdMLP
from torax import ToraxConfig
from torax import experimental as torax_experimental
from torax._src import jax_utils as torax_jax_utils
from torax._src.geometry import geometry as torax_geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.orchestration.step_function import SimulationStepFn
from torax._src.torax_pydantic import torax_pydantic

from transport_study.modules.normalization import FeatureNormalizer
from transport_study.modules.profile_predictor.module import (
    Inputs,
    Outputs,
)

# For each type of TORAX transport model:
# the coefficients predicted by the transport network, in the order of the network outputs.
TRANSPORT_COEFFICIENT_NAMES = {
    "constant": ("chi_i", "chi_e", "D_e", "V_e"),
    "cgm": ("chi_e_i_ratio", "chi_D_ratio", "VR_D_ratio", "alpha", "chi_stiff"),
    "gyrobohm": ("chi_bohm_multiplier", "chi_gyrobohm_multiplier", "D_face_c1", "D_face_c2", "V_face_coeff"),
    "qlknn": ("ITG_flux_ratio_correction", "ETG_correction_factor", "collisionality_multiplier"),
}

# Coefficients predicted by the sources network, in the order of the network outputs
SOURCE_COEFFICIENT_NAMES = (
    "S_total",
    "P_aux_total",
    "gaussian_location",
    "gaussian_width",
    "electron_heat_fraction",
)

# Reference confinement time [s] anchoring the NN aux-heating and fueling scales
# P_aux_total is a bounded fraction of w_approx / TAU_REF
# S_total a softplus multiple of particle_inventory / TAU_REF
TAU_REF_S = 0.05

# TORAX transport.model_name expected in the torax_config for each transport model
TORAX_TRANSPORT_MODEL_NAMES = {
    "constant": "constant",
    "cgm": "CGM",
    "gyrobohm": "bohm-gyrobohm",
    "qlknn": "qlknn",
}


def build_circular_geometry_jax(
    R_major: jax.Array,
    a_minor: jax.Array,
    B_0: jax.Array,
    elongation_LCFS: jax.Array,
    torax_mesh: torax_pydantic.Grid1D,
    rho_hires_norm_np: np.ndarray,
) -> torax_geometry.Geometry:
    """Circular geometry builder using JAX ops for differentiability.

    Mirrors _build_circular_geometry from torax but uses jnp.* so that
    R_major, a_minor, B_0, elongation_LCFS remain JAX-traced through the geometry construction.
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
    # Clamp 1 - (rho/R)^2 away from zero,
    # the large-aspect-ratio formulas below blow up as local epsilon -> 1
    # (MAST edge epsilon reaches 0.78 nominally, noisy per-sample a_minor/R0 can push it further)
    g3 = 1.0 / (R_major**2 * jnp.clip(1.0 - (rho / R_major) ** 2, 0.05, None) ** 1.5)
    g3_face = 1.0 / (R_major**2 * jnp.clip(1.0 - (rho_face / R_major) ** 2, 0.05, None) ** 1.5)

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
    g3_hires = 1.0 / (R_major**2 * jnp.clip(1.0 - (rho_hires / R_major) ** 2, 0.05, None) ** 1.5)
    g2g3_over_rhon_hires = 4.0 * jnp.pi**2 * vpr_hires * g3_hires * B_0 / F_hires

    R_out = R_major + rho
    R_out_face = R_major + rho_face
    R_in = R_major - rho
    R_in_face = R_major - rho_face

    epsilon = (R_out - R_in) / (R_out + R_in)
    epsilon_face = (R_out_face - R_in_face) / (R_out_face + R_in_face)
    gm4 = B_0**-2 * (1.0 + 1.5 * epsilon**2)
    gm4_face = B_0**-2 * (1.0 + 1.5 * epsilon_face**2)
    gm5 = B_0**2 / jnp.sqrt(jnp.clip(1.0 - epsilon**2, 0.05, None))
    gm5_face = B_0**2 / jnp.sqrt(jnp.clip(1.0 - epsilon_face**2, 0.05, None))

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


# Poloidal quadrature resolution for the Miller geometry builder. Periodic
# rectangle rule is spectrally accurate for smooth periodic integrands, 64
# points is overkill-safe and cheap (geometry built once per sample)
_MILLER_NTHETA = 64


def build_miller_geometry_jax(
    R_major: jax.Array,
    a_minor: jax.Array,
    B_0: jax.Array,
    elongation_LCFS: jax.Array,
    delta_top: jax.Array,
    delta_bot: jax.Array,
    torax_mesh: torax_pydantic.Grid1D,
    rho_hires_norm_np: np.ndarray,
    delta_exponent: float = 2.0,
) -> torax_geometry.Geometry:
    """Miller shaped geometry builder using JAX ops for differentiability.

    Up-down asymmetric Miller parameterization
    (R.L. Miller et al., Phys. Plasmas 5, 973 (1998), with Turnbull-style asymmetric triangularity):

        delta(rn, theta) = rn**p * (delta_mean + delta_diff*sin(theta))
        R = R_major + r*cos(theta + arcsin(delta)*sin(theta))
        Z = kappa(rn)*r*sin(theta)

    where rn is normalized rho, r = rn*a_minor, and the sin(theta) blend
    gives exactly delta_top at the top, delta_bot at the bottom, and their mean at the midplane.
    Flux-surface metrics come from poloidal quadrature with closed-form contour derivatives.
    The toroidal field model matches the circular builder
    (vacuum B = B_0*R_major/R with F = R_major*B_0), so gm4 = <R^2>/F^2 and gm5 = F^2*<1/R^2>.
    """
    rho_face_norm = jnp.array(torax_mesh.face_centers)
    rho_norm = jnp.array(torax_mesh.cell_centers)
    rho_hires_norm = jnp.array(rho_hires_norm_np)
    rho_b = a_minor

    theta = jnp.array(np.linspace(0.0, 2.0 * np.pi, _MILLER_NTHETA, endpoint=False))
    sin_t = jnp.sin(theta)
    cos_t = jnp.cos(theta)
    w_theta = 2.0 * jnp.pi / _MILLER_NTHETA

    # Clip triangularities for arcsin safety,
    # |delta| <= 0.9 everywhere keeps 1 - delta^2 >= 0.19 and avoids self-intersecting contours
    delta_top_c = jnp.clip(delta_top, -0.9, 0.9)
    delta_bot_c = jnp.clip(delta_bot, -0.9, 0.9)
    delta_mean = 0.5 * (delta_top_c + delta_bot_c)
    delta_diff = 0.5 * (delta_top_c - delta_bot_c)
    dkappa_dr = (elongation_LCFS - 1.0) / rho_b
    p = delta_exponent

    def contour(rn_col: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        # rn_col shape (n_rho, 1), broadcast against theta arrays (n_theta,)
        r = rn_col * rho_b
        rn_pow = rn_col**p
        kappa = 1.0 + rn_col * (elongation_LCFS - 1.0)
        delta_edge_t = delta_mean + delta_diff * sin_t
        delta = rn_pow * delta_edge_t
        sd = jnp.arcsin(delta)
        u = theta + sd * sin_t
        sin_u = jnp.sin(u)
        R = R_major + r * jnp.cos(u)
        inv_sqrt = 1.0 / jnp.sqrt(1.0 - delta**2)
        dsd_dtheta = rn_pow * delta_diff * cos_t * inv_sqrt
        # dsd_dr alone is singular at the axis for p < 1, but it only ever
        # appears multiplied by r, and r*dsd_dr = p*rn^p*(...) is regular
        r_dsd_dr = p * rn_pow * delta_edge_t * inv_sqrt
        du_dtheta = 1.0 + sd * cos_t + dsd_dtheta * sin_t
        dR_dr = jnp.cos(u) - sin_u * sin_t * r_dsd_dr
        dR_dt = -r * sin_u * du_dtheta
        dZ_dr = (kappa + r * dkappa_dr) * sin_t
        dZ_dt = kappa * r * cos_t
        Jp = jnp.abs(dR_dr * dZ_dt - dR_dt * dZ_dr)
        return R, Jp, dR_dt, dZ_dt

    def metrics(rn_1d: jax.Array) -> dict[str, jax.Array]:
        rn_col = rn_1d[:, None]
        # Exact contour for integral quantities,
        # all vanish at the axis without division so no floor is needed
        R, Jp, _, dZ_dt = contour(rn_col)
        vpr = rho_b * 2.0 * jnp.pi * jnp.sum(R * Jp, axis=1) * w_theta
        spr = rho_b * jnp.sum(Jp, axis=1) * w_theta
        # Green's theorem over the closed contour, exact for this shape
        volume = jnp.pi * jnp.sum(R**2 * dZ_dt, axis=1) * w_theta
        area = jnp.sum(R * dZ_dt, axis=1) * w_theta
        # Floored contour for flux-surface averages,
        # the ratios are degree-0 homogeneous in r near the axis
        # so a tiny floor evaluates the correct limit instead of 0/0
        Rf, Jpf, dR_dtf, dZ_dtf = contour(jnp.maximum(rn_col, 1e-6))
        Jpf = jnp.maximum(Jpf, 1e-12)
        Jf = Rf * Jpf
        denom = jnp.sum(Jf, axis=1)
        grad_r = jnp.sqrt(dR_dtf**2 + dZ_dtf**2) / Jpf

        def fsa(integrand: jax.Array) -> jax.Array:
            return jnp.sum(integrand * Jf, axis=1) / denom

        dv_dr = vpr / rho_b
        g0 = dv_dr * fsa(grad_r)
        g1 = dv_dr**2 * fsa(grad_r**2)
        g2 = dv_dr**2 * fsa(grad_r**2 / Rf**2)
        g3 = fsa(1.0 / Rf**2)
        R2_avg = fsa(Rf**2)
        g2g3_over_rhon = jnp.where(rn_1d > 0.0, g2 * g3 / jnp.maximum(rn_1d, 1e-12), 0.0)
        return {
            "vpr": vpr,
            "spr": spr,
            "volume": volume,
            "area": area,
            "g0": g0,
            "g1": g1,
            "g2": g2,
            "g3": g3,
            "R2_avg": R2_avg,
            "g2g3_over_rhon": g2g3_over_rhon,
        }

    cell = metrics(rho_norm)
    face = metrics(rho_face_norm)
    hires = metrics(rho_hires_norm)

    rho = rho_norm * rho_b
    rho_face = rho_face_norm * rho_b
    rho_hires = rho_hires_norm * rho_b

    # Phi = pi*B_0*rho^2 must be kept exact,
    # the Geometry.rho_b property recovers a_minor from Phi_face[-1]
    Phi = jnp.pi * B_0 * rho**2
    Phi_face = jnp.pi * B_0 * rho_face**2

    elongation = 1.0 + rho_norm * (elongation_LCFS - 1.0)
    elongation_face = 1.0 + rho_face_norm * (elongation_LCFS - 1.0)
    delta_face = rho_face_norm**p * delta_mean

    n = rho_norm.shape[0]
    n_face = rho_face_norm.shape[0]
    n_hires = rho_hires_norm.shape[0]
    F = jnp.ones(n) * R_major * B_0
    F_face = jnp.ones(n_face) * R_major * B_0
    F_hires = jnp.ones(n_hires) * R_major * B_0

    gm4 = cell["R2_avg"] / (R_major * B_0) ** 2
    gm4_face = face["R2_avg"] / (R_major * B_0) ** 2
    gm5 = (R_major * B_0) ** 2 * cell["g3"]
    gm5_face = (R_major * B_0) ** 2 * face["g3"]

    # theta = 0 and pi give exactly R_major +/- r since sd*sin(theta) = 0,
    # so rho_norm stays the normalized midplane minor radius
    R_out = R_major + rho
    R_out_face = R_major + rho_face
    R_in = R_major - rho
    R_in_face = R_major - rho_face

    return torax_geometry.Geometry(
        # Deliberately kept CIRCULAR even though the metric is shaped
        # a non-CIRCULAR type would set q_correction_factor to 1.0 instead of 1.25 (geometry.py property),
        # which shrinks q, inflates the 2|s|/q term in the CGM critical gradient,
        # and pushes samples subcritical where the NN gradient dies
        geometry_type=torax_geometry.GeometryType.CIRCULAR,
        torax_mesh=torax_mesh,
        Phi=Phi,
        Phi_face=Phi_face,
        R_major=R_major,
        a_minor=rho_b,
        B_0=B_0,
        volume=cell["volume"],
        volume_face=face["volume"],
        area=cell["area"],
        area_face=face["area"],
        vpr=cell["vpr"],
        vpr_face=face["vpr"],
        spr=cell["spr"],
        spr_face=face["spr"],
        delta_face=delta_face,
        g0=cell["g0"],
        g0_face=face["g0"],
        g1=cell["g1"],
        g1_face=face["g1"],
        g2=cell["g2"],
        g2_face=face["g2"],
        g3=cell["g3"],
        g3_face=face["g3"],
        gm4=gm4,
        gm4_face=gm4_face,
        gm5=gm5,
        gm5_face=gm5_face,
        g2g3_over_rhon=cell["g2g3_over_rhon"],
        g2g3_over_rhon_face=face["g2g3_over_rhon"],
        g2g3_over_rhon_hires=hires["g2g3_over_rhon"],
        F=F,
        F_face=F_face,
        F_hires=F_hires,
        R_in=R_in,
        R_in_face=R_in_face,
        R_out=R_out,
        R_out_face=R_out_face,
        elongation=elongation,
        elongation_face=elongation_face,
        spr_hires=hires["spr"],
        rho_hires_norm=rho_hires_norm,
        rho_hires=rho_hires,
        Phi_b_dot=jnp.asarray(0.0),
        _z_magnetic_axis=jnp.asarray(0.0),
    )


# Soft clamp bounds (lo, hi, lo_width, hi_width) for the evolving core profiles,
# in TORAX internal units: temperatures in keV, density in m^-3.
# Bounds sit far outside the physical range of C-Mod/MAST/TCV/DIII-D
# widths set how far past a bound the saturation still has usable gradient.
_TE_CLAMP_KEV = (0.005, 30.0, 0.005, 4.0)
_NE_CLAMP_M3 = (1e17, 1e21, 1e17, 5e19)


def clamp_core_profiles(state):
    """Return state with T_e, T_i, n_e cell values soft-clamped to physical range.

    Applied to the state carried between solver steps, before the next step_fn call,
    so the clamp acts before the operations that manufacture inf/NaN from an extreme state
    (resistivity ~ T^-1.5, divisions by n_e)
    Clamping after the loop would be too late: NaN propagates, and softplus(NaN) stays NaN.
    """
    cp = state.core_profiles
    cp = dataclasses.replace(
        cp,
        T_e=dataclasses.replace(cp.T_e, value=smooth_clamp(cp.T_e.value, *_TE_CLAMP_KEV)),
        T_i=dataclasses.replace(cp.T_i, value=smooth_clamp(cp.T_i.value, *_TE_CLAMP_KEV)),
        n_e=dataclasses.replace(cp.n_e, value=smooth_clamp(cp.n_e.value, *_NE_CLAMP_M3)),
    )
    return dataclasses.replace(state, core_profiles=cp)


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

    Mirrors torax._src.orchestration.jit_run_loop.run_loop_jit
    (the recommended fully-JITted simulation loop pattern from the TORAX docs)
    Modified in the following way:
      - Accepts a geo_overrides argument so per-sample geometry can be passed in
        without retracing the outer step_fn
      - Skips the per-step history buffers (we only need the final state)
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
        current_state = clamp_core_profiles(current_state)
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


def bound_transport_coefficients(transport_model: str, nn_transport_out: jax.Array) -> dict:
    """Bound the raw transport-network outputs to physical ranges for the configured model.

    The bounds keep the TORAX solver stable during training for any network output.
    Keys and order match TRANSPORT_COEFFICIENT_NAMES[transport_model].
    Shared by the profile and transport predictor TORAX modules.
    """
    if transport_model == "constant":
        # Approximate ranges taken from DIII-D study and TFTR
        # https://iopscience-iop-org.libproxy.mit.edu/article/10.1088/0029-5515/38/4/301/pdf
        # https://iopscience-iop-org.libproxy.mit.edu/article/10.1088/0029-5515/39/1/309/pdf
        #   chi_i: 0.1 - 5 m^2/s
        #   chi_e: 0.1 - 10 m^2/s
        #   D_e:   0.1 - 2 m^2/s   (nonzero floor prevents advection-only blowup)
        #   V_e:   -5 - 5 m/s      (signed pinch)
        return {
            "chi_i": 0.1 + 4.9 * jax.nn.sigmoid(nn_transport_out[0:1]),
            "chi_e": 0.1 + 9.9 * jax.nn.sigmoid(nn_transport_out[1:2]),
            "D_e": 0.1 + 1.9 * jax.nn.sigmoid(nn_transport_out[2:3]),
            "V_e": 5.0 * jnp.tanh(nn_transport_out[3:4]),
        }
    elif transport_model == "cgm":
        # Free parameters of the Critical Gradient Model.
        # The critical gradient itself is computed by TORAX
        # from the evolving state and geometry (known inputs),
        # only the dimensionless ratios are learned.
        #   chi_e_i_ratio: 0.2 - 5   (chi_e = chi_i / ratio, ITG turbulence > 1,
        #                            if electron transport dominates then < 1)
        #   chi_D_ratio:   1 - 20    (D_e = chi_i / ratio, must stay positive)
        #   VR_D_ratio:    -5 - 5    (R0*V_e/D_e, negative peaks the density profile)
        #   alpha:      1.8 - 2.2    (critical gradient exponent, TORAX default 2)
        #   chi_stiff:     0.5 - 3   (stiffness parameter, TORAX default 2)
        return {
            "chi_e_i_ratio": 0.2 + 4.8 * jax.nn.sigmoid(nn_transport_out[0:1]),
            "chi_D_ratio": 1.0 + 19.0 * jax.nn.sigmoid(nn_transport_out[1:2]),
            "VR_D_ratio": 5.0 * jnp.tanh(nn_transport_out[2:3]),
            "alpha": 1.8 + 0.4 * jax.nn.sigmoid(nn_transport_out[3:4]),
            "chi_stiff": 0.5 + 2.5 * jax.nn.sigmoid(nn_transport_out[4:5]),
        }
    elif transport_model == "gyrobohm":
        # Free parameters of the Bohm-GyroBohm model. The Bohm and GyroBohm
        # chi terms are computed by TORAX from the evolving state and geometry,
        # the NN learns one log-scale multiplier per term, applied to both
        # species, since the model already fixes the ion/electron split
        # (chi_i_B = 2 * chi_e_B, chi_i_gB = 0.5 * chi_e_gB). The coeff prefactors stay at the TORAX defaults (8e-5, 5e-6).
        #   chi_bohm_multiplier and chi_gyrobohm_multiplier:
        #     exp(-3) - exp(3), ~0.05 - 20, log-uniform around 1
        #   D_face_c1: 0.01 - 5  (diffusivity weighting at the axis, TORAX default 1.0)
        #   D_face_c2: 0.01 - 5  (diffusivity weighting at the edge, TORAX default 0.3)
        #   V_face_coeff: -4 - 2 (convectivity / diffusivity ratio, TORAX default -0.1).
        #     The sigmoid bias puts a random init at the TORAX default
        return {
            "chi_bohm_multiplier": jnp.exp(3.0 * jnp.tanh(nn_transport_out[0:1])),
            "chi_gyrobohm_multiplier": jnp.exp(3.0 * jnp.tanh(nn_transport_out[1:2])),
            "D_face_c1": 0.01 + 4.99 * jax.nn.sigmoid(nn_transport_out[2:3]),
            "D_face_c2": 0.01 + 4.99 * jax.nn.sigmoid(nn_transport_out[3:4]),
            "V_face_coeff": 2.0 - 6.0 * jax.nn.sigmoid(nn_transport_out[4:5] - 0.62),
        }
    else:  # qlknn
        # Free parameters of the QLKNN surrogate. TORAX computes ITG/TEM/ETG
        # fluxes from the evolving state via the qlknn_7_11_v1 network, the NN
        # learns three scalar correction knobs, each a log-scale multiplier
        # centered on its TORAX default so a zero-mean random init starts at
        # stock QLKNN behavior.
        # MAST sits outside the QLKNN training domain
        # (TORAX forces inputs to be clipped within its expected bounds),
        # so the knobs need more authority to compensate.
        #   ITG_flux_ratio_correction: ~0.08 - 12 around 1, multiplies the
        #     ITG electron heat flux (QLKNN10D heritage value 2.0 in range)
        #   ETG_correction_factor: ~0.03 - 4 around the default 1/3,
        #     multiplies the ETG electron heat flux
        #   collisionality_multiplier: ~0.08 - 12 around 1, scales the
        #     collisionality input (QLKNN10D heritage value 0.25 in range)
        return {
            "ITG_flux_ratio_correction": jnp.exp(2.5 * jnp.tanh(nn_transport_out[0:1])),
            "ETG_correction_factor": (1.0 / 3.0) * jnp.exp(2.5 * jnp.tanh(nn_transport_out[1:2])),
            "collisionality_multiplier": jnp.exp(2.5 * jnp.tanh(nn_transport_out[2:3])),
        }


def transport_provider_mapping(transport_model: str, coeffs: dict) -> dict:
    """Runtime-params override entries for the configured transport model.

    Shared by the profile and transport predictor TORAX modules.
    """

    def scalar(name: str) -> torax_experimental.TimeVaryingScalarUpdate:
        return torax_experimental.TimeVaryingScalarUpdate(value=coeffs[name])

    if transport_model == "constant":
        # transport coeffs are radial profiles (TimeVaryingArray) in the constant model,
        # so need to broadcast the NN scalar to a flat profile
        # TODO(ZanderKeith): What would happen if we predicted more than one?
        # Seems to me like that's just the MLP profile predictor with extra steps
        # Might be worth investigating though
        rho = jnp.array([1.0])

        def flat_profile(name: str) -> torax_experimental.TimeVaryingArrayUpdate:
            return torax_experimental.TimeVaryingArrayUpdate(
                value=jnp.broadcast_to(coeffs[name][:, jnp.newaxis], (1, 1)),
                rho_norm=rho,
            )

        return {
            "transport_model.chi_i": flat_profile("chi_i"),
            "transport_model.chi_e": flat_profile("chi_e"),
            "transport_model.D_e": flat_profile("D_e"),
            "transport_model.V_e": flat_profile("V_e"),
        }
    elif transport_model == "cgm":
        return {
            "transport_model.chi_e_i_ratio": scalar("chi_e_i_ratio"),
            "transport_model.chi_D_ratio": scalar("chi_D_ratio"),
            "transport_model.VR_D_ratio": scalar("VR_D_ratio"),
            # Plain float leaves in the provider: replaced with traced scalars
            # directly rather than via TimeVaryingScalarUpdate
            "transport_model.alpha": jnp.squeeze(coeffs["alpha"]),
            "transport_model.chi_stiff": jnp.squeeze(coeffs["chi_stiff"]),
        }
    elif transport_model == "gyrobohm":
        # Same NN multiplier applied to both species: the BGB model already
        # fixes chi_i_B = 2 * chi_e_B and chi_i_gB = 0.5 * chi_e_gB
        return {
            "transport_model.chi_e_bohm_multiplier": scalar("chi_bohm_multiplier"),
            "transport_model.chi_i_bohm_multiplier": scalar("chi_bohm_multiplier"),
            "transport_model.chi_e_gyrobohm_multiplier": scalar("chi_gyrobohm_multiplier"),
            "transport_model.chi_i_gyrobohm_multiplier": scalar("chi_gyrobohm_multiplier"),
            "transport_model.D_face_c1": scalar("D_face_c1"),
            "transport_model.D_face_c2": scalar("D_face_c2"),
            "transport_model.V_face_coeff": scalar("V_face_coeff"),
        }
    else:  # qlknn
        # Plain float leaves in the provider (like cgm alpha and chi_stiff):
        # replaced with traced scalars directly
        return {
            "transport_model.ITG_flux_ratio_correction": jnp.squeeze(coeffs["ITG_flux_ratio_correction"]),
            "transport_model.ETG_correction_factor": jnp.squeeze(coeffs["ETG_correction_factor"]),
            "transport_model.collisionality_multiplier": jnp.squeeze(coeffs["collisionality_multiplier"]),
        }


class ProfilePredictorTorax(TimeIndepModule):
    rhogrid: tuple = eqx.field(static=True)
    # Which TORAX transport model the transport network parameterizes:
    # "constant", "cgm", "gyrobohm", or "qlknn"
    transport_model: str = eqx.field(static=True)
    # Which per-sample geometry builder to use: "circular" or "miller"
    geometry_builder: str = eqx.field(static=True)
    # Radial exponent p in delta(rho_norm) = delta_edge * rho_norm**p,
    # only used by the miller builder
    delta_exponent: float = eqx.field(static=True)

    nn_transport: RtdMLP
    nn_sources: RtdMLP
    nn_edge: RtdMLP
    # Per-device CORAL stage over the 10 nn_inputs
    normalizer: FeatureNormalizer

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
        rhogrid: tuple,
        torax_config: ToraxConfig | dict,
        key: jax.random.PRNGKey,
        normalizer: FeatureNormalizer,
        transport_model: str = "cgm",
        geometry_builder: str = "circular",
        delta_exponent: float = 2.0,
    ):
        if transport_model not in TRANSPORT_COEFFICIENT_NAMES:
            raise ValueError(f"Unknown transport model '{transport_model}', valid: {sorted(TRANSPORT_COEFFICIENT_NAMES)}")
        self.transport_model = transport_model
        self.normalizer = normalizer
        if geometry_builder not in ("circular", "miller"):
            raise ValueError(f"Unknown geometry builder '{geometry_builder}', valid: ('circular', 'miller')")
        self.geometry_builder = geometry_builder
        self.delta_exponent = float(delta_exponent)

        key, subkey_transport, subkey_sources, subkey_edge = jax.random.split(key, 4)
        self.nn_transport = RtdMLP(
            in_size=10,
            out_size=len(TRANSPORT_COEFFICIENT_NAMES[transport_model]),
            width_size=nn_width,
            depth=nn_depth,
            activation=Activation.RELU,
            key=subkey_transport,
        )
        self.nn_sources = RtdMLP(
            in_size=10,
            # One output per SOURCE_COEFFICIENT_NAMES entry:
            # S_total,P_aux_total, gaussian_location, gaussian_width, electron_heat_fraction
            out_size=len(SOURCE_COEFFICIENT_NAMES),
            width_size=nn_width,
            depth=nn_depth,
            activation=Activation.RELU,
            key=subkey_sources,
        )
        self.nn_edge = RtdMLP(
            in_size=10,
            out_size=2,  # edge density fraction, edge temperature fraction
            width_size=nn_width,
            depth=nn_depth,
            activation=Activation.RELU,
            key=subkey_edge,
        )

        if isinstance(torax_config, dict):
            torax_config = ToraxConfig.from_dict(torax_config)

        expected_model_name = TORAX_TRANSPORT_MODEL_NAMES[transport_model]
        if torax_config.transport.model_name != expected_model_name:
            raise ValueError(
                f"transport_model '{transport_model}' requires torax_config transport.model_name "
                f"'{expected_model_name}', got '{torax_config.transport.model_name}'"
            )

        self.step_fn = torax_experimental.make_step_fn(torax_config)
        # Coerce to tuple: arrays in static fields break pytree metadata
        # equality (ambiguous truth value) when two module instances coexist
        self.rhogrid = tuple(np.asarray(rhogrid).tolist())

        static_geo = self.step_fn.geometry_provider(0.0)
        self._face_centers = tuple(static_geo.torax_mesh.face_centers.tolist())
        self._rho_hires_norm = tuple(np.array(static_geo.rho_hires_norm).tolist())

        # With the fixed time-step calculator, steps to cover t_final are
        # deterministic: ceil((t_final - t_initial) / fixed_dt)
        # Add 1 for the clipped final step that lands exactly on t_final
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
                ds_source_idx=inputs["ds_source_idx"].data,
                rho=jnp.array(self.rhogrid),
            )
        return inputs

    def transport_coefficients(self, nn_transport_out: jax.Array) -> dict:
        """Bound the raw transport-network outputs to physical ranges, see bound_transport_coefficients."""
        return bound_transport_coefficients(self.transport_model, nn_transport_out)

    def nn_coefficients(self, inputs: Inputs, debug: bool = False) -> dict:
        # Get the transport model free parameters and the particle / heat
        # sources from neural networks, bounded to physical ranges so the
        # TORAX solver stays stable during training.
        # Source network outputs are ordered per SOURCE_COEFFICIENT_NAMES:
        #   S_total: 0 - inf, softplus multiple of the device fueling scale
        #     particle_inventory / TAU_REF_S (x 1e21 below)
        #   P_aux_total: 0 - 4x the w_approx / TAU_REF_S power scale [MW].
        #     The heating magnitude is NN-inferred (betan encodes the stored
        #     energy the heating sustains) rather than a measured input, so
        #     every model has identical inputs.
        #     The -2 bias makes random-init heating small, starting the solver near
        #     the ohmic-only behavior (same trick as the edge-Te bias below).
        #   gaussian_location: 0 - 0.8 (deposition center in rho_norm)
        #   gaussian_width: 0.02 - 0.4 (deposition width in rho_norm)
        #   electron_heat_fraction: 0.2 - 0.95 (amount of aux heating going to electrons)
        nn_inputs = self.normalizer(inputs.nn_inputs, inputs.ds_source_idx)
        coeffs = self.transport_coefficients(self.nn_transport(nn_inputs))
        nn_sources_out = self.nn_sources(nn_inputs)
        # Particle inventory in 1e21 electrons: ne20_line_avg * volume * 0.1
        inventory = 0.1 * inputs.ne20_line_avg * inputs.volume_approx
        S_total = jax.nn.softplus(nn_sources_out[0:1]) * inventory / TAU_REF_S
        p_aux_total = 4.0 * jax.nn.sigmoid(nn_sources_out[1:2] - 2.0) * inputs.w_approx / TAU_REF_S
        gaussian_location = 0.8 * jax.nn.sigmoid(nn_sources_out[2:3])
        gaussian_width = 0.02 + 0.38 * jax.nn.sigmoid(nn_sources_out[3:4])
        electron_heat_fraction = 0.2 + 0.75 * jax.nn.sigmoid(nn_sources_out[4:5] - 0.4)

        # Edge boundary conditions as NN-predicted fractions:
        #   n_e_right_bc = fraction in (0.01, 0.95) * line-averaged density
        #   T_e_right_bc = 20 eV + fraction * clipped te_approx (beta-derived
        #                  temperature guess, same scaling trick as the shape-init predictors)
        # A fixed edge density BC above the target profile acts as an infinite
        # particle source, so the BC must scale with the requested density.
        # Both BCs are floored: a near-vacuum edge ill-conditions the density
        # equation, and te_approx (beta / ne_la) is off-scale on early-shot
        # low-density samples where betan is noisy, which can NaN training.
        # Floors are affine in the sigmoid so the NN gradient path stays intact,
        # te_approx carries no NN params so a hard clip on it costs nothing.
        # The negative bias on the temperature fraction makes random-init edge
        # temperatures small (a few tens of eV): te_approx is a beta-derived
        # overestimate, and a hot edge BC flattens the profile relative to
        # itself, which keeps the critical gradient model subcritical
        # (chi = chi_min) and kills the gradient to the transport network.
        nn_edge_out = self.nn_edge(nn_inputs)
        ne_right_bc = (0.01 + 0.94 * jax.nn.sigmoid(nn_edge_out[0:1])) * inputs.ne20_line_avg
        te_scale = jnp.clip(inputs.te_approx, 0.05, 5.0)
        te_right_bc = 0.02 + jax.nn.sigmoid(nn_edge_out[1:2] - 5.0) * te_scale

        coeffs["S_total"] = S_total
        coeffs["P_aux_total"] = p_aux_total  # [MW]
        coeffs["gaussian_location"] = gaussian_location
        coeffs["gaussian_width"] = gaussian_width
        coeffs["electron_heat_fraction"] = electron_heat_fraction
        coeffs["n_e_right_bc"] = ne_right_bc  # [1e20 m^-3]
        coeffs["T_e_right_bc"] = te_right_bc  # [keV]
        if debug:
            fmt = " ".join(f"{name}={{{name}}}" for name in coeffs)
            jax.debug.print("[nn] " + fmt + " nn_in={nn_in}", nn_in=nn_inputs, **coeffs)
        return coeffs

    def build_provider_and_geo(self, inputs: Inputs, coeffs: dict):
        S_total = coeffs["S_total"]
        ne_right_bc = coeffs["n_e_right_bc"]
        te_right_bc = coeffs["T_e_right_bc"]

        # Build runtime params override
        ip_update = torax_experimental.TimeVaryingScalarUpdate(
            value=jnp.atleast_1d(inputs.Ip * 1e6),
        )
        S_total_update = torax_experimental.TimeVaryingScalarUpdate(value=S_total * 1e21)
        p_aux_update = torax_experimental.TimeVaryingScalarUpdate(value=coeffs["P_aux_total"] * 1e6)
        gaussian_location_update = torax_experimental.TimeVaryingScalarUpdate(value=coeffs["gaussian_location"])
        gaussian_width_update = torax_experimental.TimeVaryingScalarUpdate(value=coeffs["gaussian_width"])
        electron_heat_fraction_update = torax_experimental.TimeVaryingScalarUpdate(value=coeffs["electron_heat_fraction"])
        ne_right_bc_update = torax_experimental.TimeVaryingScalarUpdate(value=ne_right_bc * 1e20)
        te_right_bc_update = torax_experimental.TimeVaryingScalarUpdate(value=te_right_bc)

        # Initial profiles are parabolic (1 - rho^2) from a core anchor down to the NN edge BC,
        # sampled on the static cell-center grid plus both endpoints
        # Static shapes, so no retracing.
        face_centers_np = np.array(self._face_centers)
        cell_centers_np = (face_centers_np[:-1] + face_centers_np[1:]) / 2.0
        rho_ic = jnp.array(np.concatenate([[0.0], cell_centers_np, [1.0]]))
        ic_shape = 1.0 - rho_ic**2

        # Temperature core anchor: edge BC + ~2x te_approx.
        # Scaling the init with te_approx starts the relaxation near the expected equilibrium
        # (a flat 0.3 keV start is several keV short on high-temperature C-Mod samples,
        # so most of the few fixed steps get burned on the transient).
        # The clip keeps a 0.3 keV floor where te_approx is small or unreliable.
        # Core = edge + positive keeps the initial state strictly decreasing and continuous with the BC for any NN output
        # A discontinuity at the LCFS or an exactly-flat profile both NaN the solver under the critical gradient model
        te_core_init = te_right_bc + jnp.clip(2.0 * inputs.te_approx, 0.3, 10.0)
        t_init_value = (te_right_bc + (te_core_init - te_right_bc) * ic_shape)[jnp.newaxis, :]
        t_init_update = torax_experimental.TimeVaryingArrayUpdate(
            value=t_init_value,
            rho_norm=rho_ic,
        )

        # Density core anchor set so the midplane chord average of the parabola matches the measured line average
        # mean of (1 - rho^2) over the chord is 2/3, so core = bc + 1.5*(line_avg - bc).
        # ne_right_bc is a fraction in (0.05, 0.95) of ne20_line_avg, so
        # core > bc always holds and the init is continuous with the BC.
        ne_core_init = ne_right_bc + 1.5 * (inputs.ne20_line_avg - ne_right_bc)
        n_init_value = 1e20 * (ne_right_bc + (ne_core_init - ne_right_bc) * ic_shape)[jnp.newaxis, :]
        n_init_update = torax_experimental.TimeVaryingArrayUpdate(
            value=n_init_value,
            rho_norm=rho_ic,
        )

        mapping = {
            "profile_conditions.Ip": ip_update,
            "profile_conditions.n_e": n_init_update,
            "profile_conditions.n_e_right_bc": ne_right_bc_update,
            # Assume the ion edge temperature matches the electron edge temperature
            "profile_conditions.T_e_right_bc": te_right_bc_update,
            "profile_conditions.T_i_right_bc": te_right_bc_update,
            "profile_conditions.T_e": t_init_update,
            "profile_conditions.T_i": t_init_update,
            "sources.gas_puff.S_total": S_total_update,
            # NN-inferred auxiliary heating, absorption_fraction stays at its
            # config value (fixed, degenerate with P_total)
            "sources.generic_heat.P_total": p_aux_update,
            "sources.generic_heat.gaussian_location": gaussian_location_update,
            "sources.generic_heat.gaussian_width": gaussian_width_update,
            "sources.generic_heat.electron_heat_fraction": electron_heat_fraction_update,
        }
        mapping.update(transport_provider_mapping(self.transport_model, coeffs))
        new_provider = self.step_fn.runtime_params_provider.update_provider_from_mapping(mapping)

        # Build JAX-differentiable geometry from per-sample inputs
        torax_mesh = torax_pydantic.Grid1D(face_centers=face_centers_np)
        rho_hires_norm_np = np.array(self._rho_hires_norm)
        if self.geometry_builder == "miller":
            geo = build_miller_geometry_jax(
                R_major=inputs.R0,
                a_minor=inputs.a_minor,
                B_0=inputs.B0,
                elongation_LCFS=inputs.kappa,
                delta_top=inputs.delta_top,
                delta_bot=inputs.delta_bot,
                torax_mesh=torax_mesh,
                rho_hires_norm_np=rho_hires_norm_np,
                delta_exponent=self.delta_exponent,
            )
        else:
            geo = build_circular_geometry_jax(
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

        This is rho_norm, the normalized toroidal flux radius. For the
        circular geometry used here, minor radius = a * rho_norm, so rho_norm
        is exactly the normalized minor radius rho the datasets use.
        """
        face_centers = np.array(self._face_centers)
        return (face_centers[:-1] + face_centers[1:]) / 2.0

    def __call__(self, inputs: Inputs | xr.Dataset, debug: bool = False) -> Outputs:
        inputs = self._coerce_inputs(inputs)
        coeffs = self.nn_coefficients(inputs, debug=debug)
        new_provider, geo_provider = self.build_provider_and_geo(inputs, coeffs)

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
            # Recompute the loop body activations in the backward pass instead
            # of storing them: reverse-mode memory otherwise scales with
            # solver steps times mesh size, and OOMs the GPU at full batch
            # The sweep now allows up to 40 steps, so every model
            # needs the flat-memory loop.
            wrap_body_in_checkpoint=True,
        )

        # n_e is in m^-3 and T_e is keV in TORAX
        ne = state.core_profiles.n_e.value / 1e20
        te = state.core_profiles.T_e.value

        # Interpolate onto rhogrid: TORAX evolves profiles on rho_norm, which
        # for the circular geometry used here equals the normalized minor
        # radius rho, so no flux-coordinate mapping is needed. Augment the
        # cell values with both endpoints before interpolating: jnp.interp
        # flat-holds outside the data range, which would ignore the exact
        # Dirichlet edge BC at rho = 1 (the flat-held edge overpredicts
        # exactly where measured profiles fall steeply). Duplicating the
        # innermost cell at rho = 0 encodes the zero-gradient axis condition.
        rho_cells = jnp.asarray(self.rho_norm_grid)
        rho_full = jnp.concatenate([jnp.zeros(1), rho_cells, jnp.ones(1)])
        ne_full = jnp.concatenate([ne[:1], ne, coeffs["n_e_right_bc"]])
        te_full = jnp.concatenate([te[:1], te, coeffs["T_e_right_bc"]])
        ne_interp = jnp.interp(inputs.rho, rho_full, ne_full)
        te_interp = jnp.interp(inputs.rho, rho_full, te_full)

        return Outputs(
            ne=xr.DataArray(
                data=ne_interp,
                dims=("rho",),
                coords={"rho": list(self.rhogrid)},
            ),
            te=xr.DataArray(
                data=te_interp,
                dims=("rho",),
                coords={"rho": list(self.rhogrid)},
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
        intermediate states are observable. Not differentiable, do not use for training.

        Args:
            inputs: Same as __call__ (Inputs or single-timeslice xr.Dataset).
            prescribed: Optional dict overriding NN outputs. Valid keys are the
                transport coefficients of the configured model
                (TRANSPORT_COEFFICIENT_NAMES[self.transport_model]) plus the
                source coefficients (SOURCE_COEFFICIENT_NAMES) and
                {n_e_right_bc, T_e_right_bc}, values are floats in the same
                units the NN outputs use (S_total in 1e21 particles/s,
                P_aux_total in MW, gaussian_location/gaussian_width in
                rho_norm, electron_heat_fraction dimensionless, n_e_right_bc
                in 1e20 m^-3, T_e_right_bc in keV, chi/D/V in m^2/s or m/s
                for the constant model, dimensionless otherwise).
                Keys not given (or None) keep the NN prediction.

        Returns:
            (steps, coeffs):
                steps: list of dicts, one per TORAX state including the initial one, with keys
                    t [s], ne20 [1e20 m^-3], te_keV [keV] and rho (the static rho_norm cell
                    grid, equal to the normalized minor radius for circular geometry).
                coeffs: the transport/source coefficients actually used, as floats.
        """
        inputs = self._coerce_inputs(inputs)
        coeffs = self.nn_coefficients(inputs)
        if prescribed is not None:
            unknown = set(prescribed) - set(coeffs)
            if unknown:
                raise ValueError(f"Unknown prescribed coefficients {unknown}, valid keys: {sorted(coeffs)}")
            for name, value in prescribed.items():
                if value is not None:
                    # Match the NN output dtype exactly: a weakly-typed scalar would get
                    # demoted to float32 inside TORAX and fail its float64 checks
                    coeffs[name] = jnp.full_like(coeffs[name], float(value))
        new_provider, geo_provider = self.build_provider_and_geo(inputs, coeffs)

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
                "rho": self.rho_norm_grid,
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
            # Same clamp as the training loop so the recorded trajectory
            # matches what __call__ actually simulates
            state = clamp_core_profiles(state)
            steps.append(record(state))

        coeffs_out = {name: float(np.asarray(value).squeeze()) for name, value in coeffs.items()}
        return steps, coeffs_out

    @classmethod
    def init(
        cls,
        rhogrid: Array,
        torax_config: ToraxConfig,
        nn_width: int,
        nn_depth: int,
        prng_seed: int,
        normalizer: FeatureNormalizer,
        transport_model: str = "cgm",
        geometry_builder: str = "circular",
        delta_exponent: float = 2.0,
    ):
        rhogrid_tuple = tuple(rhogrid.tolist())
        return cls(
            nn_width=nn_width,
            nn_depth=nn_depth,
            rhogrid=rhogrid_tuple,
            torax_config=torax_config,
            key=jax.random.PRNGKey(prng_seed),
            normalizer=normalizer,
            transport_model=transport_model,
            geometry_builder=geometry_builder,
            delta_exponent=delta_exponent,
        )
