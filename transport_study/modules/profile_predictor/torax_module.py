import dataclasses
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import ArrayLike
from popsim import TimeIndepModule
from popsim.ml.rtd_mlp import Activation, RtdMLP
from torax import ToraxConfig
from torax import experimental as torax_experimental
from torax._src import jax_utils as torax_jax_utils
from torax._src.geometry import geometry as torax_geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.geometry import trapped_fraction as torax_trapped_fraction
from torax._src.orchestration.step_function import SimulationStepFn
from torax._src.torax_pydantic import torax_pydantic

from transport_study import RADIAL_DIM
from transport_study.modules import plasma_parameters
from transport_study.modules.normalization import FeatureNormalizer
from transport_study.modules.profile_predictor.module import (
    N_NN_INPUTS,
    Inputs,
    Outputs,
    profile_outputs,
    static_rhogrid,
)

# For each type of TORAX transport model:
# the coefficients predicted by the transport network, in the order of the network outputs.
TRANSPORT_COEFFICIENT_NAMES = {
    "constant": ("chi_i", "chi_e", "D_e", "V_e"),
    "gyrobohm": ("chi_bohm_multiplier", "chi_gyrobohm_multiplier", "D_face_c1", "D_face_c2", "V_face_coeff"),
    "qlknn": ("ITG_flux_ratio_correction", "ETG_correction_factor", "collisionality_multiplier"),
}

# The torax-backed model type of every implemented transport model, torax-<transport_model>
TORAX_MODEL_TYPES = tuple(f"torax-{transport_model}" for transport_model in TRANSPORT_COEFFICIENT_NAMES)

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

# j_01^2, the lowest cylindrical diffusion eigenvalue (decay time a^2 / (j_01^2 chi))
_J01_SQUARED = 5.783

# TORAX model_name of the single core transport model each transport model's config holds,
# at torax_config.transport.core_transport_models[transport_model]
TORAX_TRANSPORT_MODEL_NAMES = {
    "constant": "prescribed",
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
    # (MAST edge epsilon reaches 0.78 nominally, noisy per-sample minor_radius / geometric_axis_r can push it further)
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
    trapped_fraction_face = torax_trapped_fraction.calculate_sauter_trapped_fraction(epsilon=epsilon_face, delta=delta_face)

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
        trapped_fraction_face=trapped_fraction_face,
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

# Surfaces on the uniform midplane-radius grid the Miller builder inverts the toroidal-flux label on.
# Linear interpolation of the inverse on 257 points is accurate to ~1e-6 in the surface radius
_MILLER_N_LABEL = 257

# Miller shape-profile closures (miller_shape_coefficients).
# The magnetic axis keeps AXIS_ELONGATION_FRACTION of the LCFS elongation excess,
# reconstructed equilibria keep roughly 0.6 - 0.9 of it.
# The axis Shafranov shift is the large-aspect-ratio (a^2 / 2 R)(beta_p + l_i / 2) at a fixed internal inductance,
# capped at MAX_SHAFRANOV_SHIFT_NORM of the minor radius, where |dR0 / dr| = 0.6 keeps the surfaces nested
AXIS_ELONGATION_FRACTION = 0.7
SHAFRANOV_INTERNAL_INDUCTANCE = 0.8
MAX_SHAFRANOV_SHIFT_NORM = 0.3


def miller_shape_coefficients(inputs, energy_mhd_MJ: ArrayLike) -> dict:
    """Dimensionless shape-profile closures of the Miller builder for one sample.

      axis_elongation_fraction: fraction of the LCFS elongation excess kept at the magnetic axis
      shafranov_shift_norm: outward shift of the magnetic axis from the LCFS center over the minor radius
    Any Inputs with the 0D shape fields, with the stored energy the beta_p of the shift comes from.
    """
    beta_poloidal = plasma_parameters.beta_poloidal(
        energy_mhd_MJ, inputs.volume_approx, inputs.ip_MA, inputs.minor_radius, inputs.elongation
    )
    shift_norm = 0.5 * inputs.epsilon * (beta_poloidal + 0.5 * SHAFRANOV_INTERNAL_INDUCTANCE)
    return {
        "axis_elongation_fraction": jnp.full((1,), AXIS_ELONGATION_FRACTION),
        "shafranov_shift_norm": jnp.atleast_1d(jnp.minimum(shift_norm, MAX_SHAFRANOV_SHIFT_NORM)),
    }


def geometry_shape_coefficients(geometry_builder: str, inputs, energy_mhd_MJ: ArrayLike) -> dict:
    """The shape coefficients build_geometry_provider reads, none for the circular builder."""
    if geometry_builder == "miller":
        return miller_shape_coefficients(inputs, energy_mhd_MJ)
    return {}


class _MillerShape(NamedTuple):
    """Per-sample Miller shape parameters and the poloidal quadrature grid, see build_miller_geometry_jax."""

    R_major: jax.Array
    a_minor: jax.Array
    F: jax.Array
    elongation_excess: jax.Array
    axis_elongation_fraction: jax.Array
    shift: jax.Array
    delta_mean: jax.Array
    delta_diff: jax.Array
    delta_exponent: ArrayLike
    theta: jax.Array


def _miller_shape_profiles(shape: _MillerShape, rn: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """kappa, R0 and rn**p of the surfaces at rn, the power safe at rn = 0 for a traced exponent."""
    kappa = 1.0 + shape.elongation_excess * (shape.axis_elongation_fraction + (1.0 - shape.axis_elongation_fraction) * rn**2)
    center_r = shape.R_major + shape.shift * (1.0 - rn**2)
    rn_safe = jnp.where(rn > 0.0, rn, 1.0)
    rn_pow = jnp.where(rn > 0.0, rn_safe**shape.delta_exponent, 0.0)
    return kappa, center_r, rn_pow


def _miller_contour(shape: _MillerShape, rn_col: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """R, the poloidal-plane Jacobian |d(R, Z) / d(r, theta)|, dR / dtheta and dZ / dtheta of each surface.

    rn_col has shape (n_rho, 1) and broadcasts against the theta grid.
    """
    sin_t = jnp.sin(shape.theta)
    cos_t = jnp.cos(shape.theta)
    r = rn_col * shape.a_minor
    kappa, center_r, rn_pow = _miller_shape_profiles(shape, rn_col)
    # r * dkappa / dr and dR0 / dr of the radial profiles
    r_dkappa_dr = 2.0 * shape.elongation_excess * (1.0 - shape.axis_elongation_fraction) * rn_col**2
    dcenter_dr = -2.0 * shape.shift * rn_col / shape.a_minor
    delta_edge_t = shape.delta_mean + shape.delta_diff * sin_t
    delta = rn_pow * delta_edge_t
    sd = jnp.arcsin(delta)
    u = shape.theta + sd * sin_t
    sin_u = jnp.sin(u)
    R = center_r + r * jnp.cos(u)
    inv_sqrt = 1.0 / jnp.sqrt(1.0 - delta**2)
    dsd_dtheta = rn_pow * shape.delta_diff * cos_t * inv_sqrt
    # dsd_dr alone is singular at the axis for p < 1, but it only ever
    # appears multiplied by r, and r*dsd_dr = p*rn^p*(...) is regular
    r_dsd_dr = shape.delta_exponent * rn_pow * delta_edge_t * inv_sqrt
    du_dtheta = 1.0 + sd * cos_t + dsd_dtheta * sin_t
    dR_dr = dcenter_dr + jnp.cos(u) - sin_u * sin_t * r_dsd_dr
    dR_dt = -r * sin_u * du_dtheta
    dZ_dr = (kappa + r_dkappa_dr) * sin_t
    dZ_dt = kappa * r * cos_t
    Jp = jnp.abs(dR_dr * dZ_dt - dR_dt * dZ_dr)
    return R, Jp, dR_dt, dZ_dt


def _miller_enclosed_flux(shape: _MillerShape, rn_1d: jax.Array) -> jax.Array:
    """Vacuum toroidal flux F * int dA / R inside each surface, by Green's theorem the loop integral of ln(R) dZ."""
    R, _, _, dZ_dt = _miller_contour(shape, rn_1d[:, None])
    w_theta = 2.0 * jnp.pi / shape.theta.shape[0]
    return shape.F * jnp.sum(jnp.log(R) * dZ_dt, axis=1) * w_theta


def _miller_metrics(shape: _MillerShape, rho_norm_1d: jax.Array, rho_norm_label: jax.Array, rn_label: jax.Array, phi_b: jax.Array) -> dict:
    """Geometry quantities of the surfaces at the toroidal-flux radii rho_norm_1d.

    rho_norm_label is the monotonic rho_norm(rn) on the rn_label grid, inverted by interpolation.
    """
    w_theta = 2.0 * jnp.pi / shape.theta.shape[0]
    rn_1d = jnp.interp(rho_norm_1d, rho_norm_label, rn_label)
    rn_col = rn_1d[:, None]
    # Exact contour for integral quantities,
    # all vanish at the axis without division so no floor is needed
    R, _, _, dZ_dt = _miller_contour(shape, rn_col)
    # Green's theorem over the closed contour, exact for this shape
    volume = jnp.pi * jnp.sum(R**2 * dZ_dt, axis=1) * w_theta
    area = jnp.sum(R * dZ_dt, axis=1) * w_theta
    # Floored contour for flux-surface averages and flux-derivative ratios,
    # all degree-0 homogeneous in r near the axis so a tiny floor evaluates the correct limit instead of 0/0
    Rf, Jpf, dR_dtf, dZ_dtf = _miller_contour(shape, jnp.maximum(rn_col, 1e-6))
    Jpf = jnp.maximum(Jpf, 1e-12)
    Jf = Rf * Jpf
    denom = jnp.sum(Jf, axis=1)
    grad_r = jnp.sqrt(dR_dtf**2 + dZ_dtf**2) / Jpf

    def fsa(integrand: jax.Array) -> jax.Array:
        return jnp.sum(integrand * Jf, axis=1) / denom

    # dV / dPhi and dA / dPhi, the shared factor a_minor drn cancels
    flux_weight = shape.F * jnp.sum(Jpf / Rf, axis=1)
    dphi_drho_norm = 2.0 * phi_b * rho_norm_1d
    # |grad V| = (dV / dr) |grad r|, so these metrics do not depend on the radial label.
    # The exact dV / dr is 0 at the axis, masked there since the floored contour gives a tiny nonzero value
    mask_off_axis = rn_1d > 0.0
    dv_dr = jnp.where(mask_off_axis, 2.0 * jnp.pi * denom * w_theta, 0.0)
    g2 = dv_dr**2 * fsa(grad_r**2 / Rf**2)
    g3 = fsa(1.0 / Rf**2)
    kappa, center_r, rn_pow = _miller_shape_profiles(shape, rn_1d)
    return {
        "rn": rn_1d,
        "vpr": 2.0 * jnp.pi * denom / flux_weight * dphi_drho_norm,
        "spr": jnp.sum(Jpf, axis=1) / flux_weight * dphi_drho_norm,
        "volume": volume,
        "area": area,
        "g0": dv_dr * fsa(grad_r),
        "g1": dv_dr**2 * fsa(grad_r**2),
        "g2": g2,
        "g3": g3,
        "R2_avg": fsa(Rf**2),
        "g2g3_over_rhon": jnp.where(rho_norm_1d > 0.0, g2 * g3 / jnp.maximum(rho_norm_1d, 1e-12), 0.0),
        "elongation": kappa,
        "center_r": center_r,
        "rn_pow": rn_pow,
    }


def build_miller_geometry_jax(
    R_major: jax.Array,
    a_minor: jax.Array,
    B_0: jax.Array,
    elongation_LCFS: jax.Array,
    triangularity_upper: jax.Array,
    triangularity_lower: jax.Array,
    axis_elongation_fraction: jax.Array,
    shafranov_shift_norm: jax.Array,
    torax_mesh: torax_pydantic.Grid1D,
    rho_hires_norm_np: np.ndarray,
    delta_exponent: ArrayLike = 2.0,
) -> torax_geometry.Geometry:
    """Miller shaped geometry builder using JAX ops for differentiability.

    Up-down asymmetric Miller parameterization
    (R.L. Miller et al., Phys. Plasmas 5, 973 (1998), with Turnbull-style asymmetric triangularity):

        R = R0(rn) + r*cos(theta + arcsin(delta)*sin(theta)),   Z = kappa(rn)*r*sin(theta)
        R0(rn) = R_major + shift*(1 - rn**2)
        kappa(rn) = 1 + (elongation_LCFS - 1)*(f + (1 - f)*rn**2)
        delta(rn, theta) = rn**p * (delta_mean + delta_diff*sin(theta))

    where rn = r / a_minor is the midplane half-width, R_major the LCFS center, shift = shafranov_shift_norm*a_minor,
    f = axis_elongation_fraction and p = delta_exponent.
    The sin(theta) blend gives exactly triangularity_upper at the top, triangularity_lower at the bottom, and their mean at the midplane.
    Flux-surface metrics come from poloidal quadrature with closed-form contour derivatives.
    The toroidal field model is vacuum (B = B_0*R_major/R with F = R_major*B_0), so gm4 = <R^2>/F^2 and gm5 = F^2*<1/R^2>.

    TORAX's radial coordinate is rho_norm = sqrt(Phi / Phi_b),
    the normalized toroidal flux radius the profile data is stored on, not rn.
    Phi(rn) = F * int dA / R comes from Green's theorem on each contour,
    the builder inverts rho_norm(rn) on a fine rn grid to find the surface of every mesh point,
    and vpr / spr are dV / dPhi and dA / dPhi times dPhi / drho_norm = 2 Phi_b rho_norm.
    """
    rho_face_norm = jnp.array(torax_mesh.face_centers)
    rho_norm = jnp.array(torax_mesh.cell_centers)
    rho_hires_norm = jnp.array(rho_hires_norm_np)

    # Clip triangularities for arcsin safety,
    # |delta| <= 0.9 everywhere keeps 1 - delta^2 >= 0.19 and avoids self-intersecting contours
    triangularity_upper_c = jnp.clip(triangularity_upper, -0.9, 0.9)
    triangularity_lower_c = jnp.clip(triangularity_lower, -0.9, 0.9)
    shape = _MillerShape(
        R_major=R_major,
        a_minor=a_minor,
        F=R_major * B_0,
        elongation_excess=elongation_LCFS - 1.0,
        axis_elongation_fraction=axis_elongation_fraction,
        shift=shafranov_shift_norm * a_minor,
        delta_mean=0.5 * (triangularity_upper_c + triangularity_lower_c),
        delta_diff=0.5 * (triangularity_upper_c - triangularity_lower_c),
        delta_exponent=delta_exponent,
        theta=jnp.array(np.linspace(0.0, 2.0 * np.pi, _MILLER_NTHETA, endpoint=False)),
    )

    # Normalized toroidal flux radius of every surface on the label grid, with a safe sqrt at the axis where Phi = 0
    rn_label = jnp.array(np.linspace(0.0, 1.0, _MILLER_N_LABEL))
    phi_label = _miller_enclosed_flux(shape, rn_label)
    phi_b = phi_label[-1]
    phi_ratio = phi_label / phi_b
    phi_ratio_safe = jnp.where(phi_ratio > 0.0, phi_ratio, 1.0)
    rho_norm_label = jnp.where(phi_ratio > 0.0, jnp.sqrt(phi_ratio_safe), 0.0)

    cell = _miller_metrics(shape, rho_norm, rho_norm_label, rn_label, phi_b)
    face = _miller_metrics(shape, rho_face_norm, rho_norm_label, rn_label, phi_b)
    hires = _miller_metrics(shape, rho_hires_norm, rho_norm_label, rn_label, phi_b)

    # theta = 0 and pi give exactly R0 +/- r since sd*sin(theta) = 0
    r_cell = cell["rn"] * a_minor
    r_face = face["rn"] * a_minor
    R_out_face = face["center_r"] + r_face
    R_in_face = face["center_r"] - r_face
    delta_face = face["rn_pow"] * shape.delta_mean
    epsilon_face = (R_out_face - R_in_face) / (R_out_face + R_in_face)
    # Sauter's effective inverse aspect ratio 0.67 (1 - 1.4 delta |delta|) epsilon turns negative above |delta| = 0.845
    # and NaNs the trapped fraction (DIII-D stores hold edge triangularities up to 0.9),
    # so the fit sees the triangularity clipped to 0.8, inside its range
    delta_trapped_face = jnp.clip(delta_face, -0.8, 0.8)

    return torax_geometry.Geometry(
        # A non-CIRCULAR type sets q_correction_factor to 1,
        # the circular model's 1.25 would inflate q on top of exact metrics (TORAX reads the type nowhere else at runtime)
        geometry_type=torax_geometry.GeometryType.CHEASE,
        torax_mesh=torax_mesh,
        # Phi = Phi_b * rho_norm^2 exactly, the Geometry.rho_b property recovers sqrt(Phi_b / (pi B_0)) from Phi_face[-1]
        Phi=phi_b * rho_norm**2,
        Phi_face=phi_b * rho_face_norm**2,
        R_major=R_major,
        a_minor=a_minor,
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
        trapped_fraction_face=torax_trapped_fraction.calculate_sauter_trapped_fraction(epsilon=epsilon_face, delta=delta_trapped_face),
        g0=cell["g0"],
        g0_face=face["g0"],
        g1=cell["g1"],
        g1_face=face["g1"],
        g2=cell["g2"],
        g2_face=face["g2"],
        g3=cell["g3"],
        g3_face=face["g3"],
        gm4=cell["R2_avg"] / shape.F**2,
        gm4_face=face["R2_avg"] / shape.F**2,
        gm5=shape.F**2 * cell["g3"],
        gm5_face=shape.F**2 * face["g3"],
        g2g3_over_rhon=cell["g2g3_over_rhon"],
        g2g3_over_rhon_face=face["g2g3_over_rhon"],
        g2g3_over_rhon_hires=hires["g2g3_over_rhon"],
        F=jnp.full(rho_norm.shape, shape.F),
        F_face=jnp.full(rho_face_norm.shape, shape.F),
        F_hires=jnp.full(rho_hires_norm.shape, shape.F),
        R_in=cell["center_r"] - r_cell,
        R_in_face=R_in_face,
        R_out=cell["center_r"] + r_cell,
        R_out_face=R_out_face,
        elongation=cell["elongation"],
        elongation_face=face["elongation"],
        spr_hires=hires["spr"],
        rho_hires_norm=rho_hires_norm,
        rho_hires=rho_hires_norm * jnp.sqrt(phi_b / (jnp.pi * B_0)),
        Phi_b_dot=jnp.asarray(0.0),
        _z_magnetic_axis=jnp.asarray(0.0),
    )


# Clamp bounds (lo, hi) for the evolving core profiles,
# in TORAX internal units: temperatures in keV, density in m^-3.
# Bounds sit far outside the physical range of C-Mod/MAST/TCV/DIII-D.
_TE_CLAMP_KEV = (0.005, 30.0)
_NE_CLAMP_M3 = (1e17, 1e21)


def clamp_core_profiles(state):
    """Return state with T_e, T_i, n_e cell values clipped to the clamp bounds.

    Applied to the state carried between solver steps, before the next step_fn call,
    so the clamp acts before the operations that manufacture inf/NaN from an extreme state
    (resistivity ~ T^-1.5, divisions by n_e).
    Clamping after the loop would be too late, NaN propagates.
    It runs after every step, so it must be the exact identity in range:
    any in-range shift accumulates per step, not per unit time.
    A softplus clamp with a 4 keV width removed 2-27 eV per call,
    an artificial heat sink of 9-27 percent of the loss power at 1 ms steps.
    The gradient is zero only past a bound.
    """
    cp = state.core_profiles
    cp = dataclasses.replace(
        cp,
        T_e=dataclasses.replace(cp.T_e, value=jnp.clip(cp.T_e.value, *_TE_CLAMP_KEV)),
        T_i=dataclasses.replace(cp.T_i, value=jnp.clip(cp.T_i.value, *_TE_CLAMP_KEV)),
        n_e=dataclasses.replace(cp.n_e, value=jnp.clip(cp.n_e.value, *_NE_CLAMP_M3)),
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
):
    """Local copy of torax._src.orchestration.jit_run_loop.run_loop_jit that accepts a geometry override.

    The upstream loop builds its own initial state and takes no geo_overrides,
    so per-sample geometry cannot reach it.
    Differences from upstream:
      - initial state and geometry provider are passed in
      - every step is clamped (clamp_core_profiles) before the next one
      - only the final state is returned
    while_loop_bounded enforces max_steps and its custom VJP reruns each step's
    vjp from the stored per-step history, so reverse mode already stores states, not activations.
    """

    def cond(carry):
        current_state, _ = carry
        return jnp.logical_not(step_fn.is_done(current_state.t))

    def body(carry):
        prev_state, prev_post = carry
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
                "[loop] t={t} dt={dt} err={err} Te[min,max]=[{te_lo},{te_hi}] ne[min,max]=[{ne_lo},{ne_hi}]",
                t=current_state.t,
                dt=current_state.dt,
                err=current_state.solver_numeric_outputs.solver_error_state,
                te_lo=cp.T_e.value.min(),
                te_hi=cp.T_e.value.max(),
                ne_lo=cp.n_e.value.min(),
                ne_hi=cp.n_e.value.max(),
            )
        return current_state, post_processed

    final_carry, _num_steps, _history = torax_jax_utils.while_loop_bounded(
        cond,
        body,
        (input_state, previous_post_processed_outputs),
        max_steps,
    )
    output_state, post_processed = final_carry
    return output_state, post_processed


def density_peaking_rate(nn_out: jax.Array) -> jax.Array:
    """Dimensionless density peaking rate c = V a / D in (-6, 2) from a raw network output.

    With no core particle source the steady-state density obeys d ln n / d rho_norm = c,
    so one range means the same peaking on every device, whatever its minor radius.
    The ln 3 bias puts a zero output at c = 0, no pinch.
    """
    return 2.0 - 8.0 * jax.nn.sigmoid(nn_out - jnp.log(3.0))


def bound_transport_coefficients(transport_model: str, nn_transport_out: jax.Array, minor_radius: ArrayLike) -> dict:
    """Bound the raw transport-network outputs to physical ranges for the configured model.

    The network predicts dimensionless values, the minor radius [m] turns them into TORAX units.
    The bounds keep the TORAX solver stable during training for any network output.
    Keys and order match TRANSPORT_COEFFICIENT_NAMES[transport_model].
    Shared by the profile and transport predictor TORAX modules.
    """
    if transport_model == "constant":
        # Multiples of the device's own diffusivity scale chi_ref = a^2 / (j_01^2 TAU_REF_S),
        # whose lowest cylindrical decay time a^2 / (j_01^2 chi) is TAU_REF_S.
        # Measured tau_E puts the effective chi at medians of 0.4 - 2 chi_ref on every device.
        #   chi_i, chi_e, D_e: 0.1 - 10 chi_ref, log-uniform around chi_ref (D ~ chi as in the TORAX defaults)
        #   V_e: density_peaking_rate x D_e / a (signed pinch)
        chi_ref = minor_radius**2 / (_J01_SQUARED * TAU_REF_S)
        D_e = chi_ref * jnp.exp(2.3 * jnp.tanh(nn_transport_out[2:3]))
        peaking_rate = density_peaking_rate(nn_transport_out[3:4])
        return {
            "chi_i": chi_ref * jnp.exp(2.3 * jnp.tanh(nn_transport_out[0:1])),
            "chi_e": chi_ref * jnp.exp(2.3 * jnp.tanh(nn_transport_out[1:2])),
            "D_e": D_e,
            "V_e": peaking_rate * D_e / minor_radius,
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
        #   V_face_coeff: density_peaking_rate / a [1/m] (BGB sets V = V_face_coeff x D)
        # The sigmoid biases put a zero output at the TORAX defaults of D_face_c1 and D_face_c2.
        # Unbiased, D_face_c1 and D_face_c2 would start at 2.5, which empties a TCV plasma in ~5 ms
        peaking_rate = density_peaking_rate(nn_transport_out[4:5])
        return {
            "chi_bohm_multiplier": jnp.exp(3.0 * jnp.tanh(nn_transport_out[0:1])),
            "chi_gyrobohm_multiplier": jnp.exp(3.0 * jnp.tanh(nn_transport_out[1:2])),
            "D_face_c1": 0.01 + 4.99 * jax.nn.sigmoid(nn_transport_out[2:3] - 1.40),
            "D_face_c2": 0.01 + 4.99 * jax.nn.sigmoid(nn_transport_out[3:4] - 2.79),
            "V_face_coeff": peaking_rate / minor_radius,
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


def validate_transport_model_name(torax_config: ToraxConfig, transport_model: str) -> None:
    """Check the TORAX transport config holds exactly the core model transport_model expects.

    The NN overrides address core_transport_models[transport_model] (see transport_provider_mapping),
    so exactly that one core model of the expected type must be configured and no pedestal model.
    Also catches the silent TORAX default,
    a transport dict without core_transport_models gets a single prescribed core model.
    """
    expected = {transport_model: TORAX_TRANSPORT_MODEL_NAMES[transport_model]}
    transport = torax_config.transport
    core_names = {key: model.model_name for key, model in transport.core_transport_models.items()}
    pedestal_keys = sorted(transport.pedestal_transport_models)
    if core_names != expected or pedestal_keys:
        raise ValueError(
            f"transport_model '{transport_model}' requires exactly the core transport models {expected} "
            f"and no pedestal transport models, got core {core_names} and pedestal {pedestal_keys}"
        )


def transport_provider_mapping(transport_model: str, coeffs: dict) -> dict:
    """Runtime-params override entries for the configured transport model.

    Shared by the profile and transport predictor TORAX modules.
    Keys are dotted provider paths into the single core transport model,
    applied with RuntimeParamsProvider.update_provider_from_mapping.
    """
    prefix = f"transport_model.core_transport_models.{transport_model}."

    def scalar(name: str) -> torax_experimental.TimeVaryingScalarUpdate:
        return torax_experimental.TimeVaryingScalarUpdate(value=coeffs[name])

    if transport_model == "constant":
        # transport coeffs are radial profiles (TimeVaryingArray) in the prescribed model,
        # so need to broadcast the NN scalar to a flat profile
        rho = jnp.array([1.0])

        def flat_profile(name: str) -> torax_experimental.TimeVaryingArrayUpdate:
            return torax_experimental.TimeVaryingArrayUpdate(
                value=jnp.broadcast_to(coeffs[name][:, jnp.newaxis], (1, 1)),
                rho_norm=rho,
            )

        updates = {
            "chi_i": flat_profile("chi_i"),
            "chi_e": flat_profile("chi_e"),
            "D_e": flat_profile("D_e"),
            "V_e": flat_profile("V_e"),
        }
    elif transport_model == "gyrobohm":
        # Same NN multiplier applied to both species: the BGB model already
        # fixes chi_i_B = 2 * chi_e_B and chi_i_gB = 0.5 * chi_e_gB
        updates = {
            "chi_e_bohm_multiplier": scalar("chi_bohm_multiplier"),
            "chi_i_bohm_multiplier": scalar("chi_bohm_multiplier"),
            "chi_e_gyrobohm_multiplier": scalar("chi_gyrobohm_multiplier"),
            "chi_i_gyrobohm_multiplier": scalar("chi_gyrobohm_multiplier"),
            "D_face_c1": scalar("D_face_c1"),
            "D_face_c2": scalar("D_face_c2"),
            "V_face_coeff": scalar("V_face_coeff"),
        }
    else:  # qlknn
        # Plain float leaves in the provider:
        # replaced with traced scalars directly rather than via TimeVaryingScalarUpdate
        updates = {
            "ITG_flux_ratio_correction": jnp.squeeze(coeffs["ITG_flux_ratio_correction"]),
            "ETG_correction_factor": jnp.squeeze(coeffs["ETG_correction_factor"]),
            "collisionality_multiplier": jnp.squeeze(coeffs["collisionality_multiplier"]),
        }
    return {prefix + field: value for field, value in updates.items()}


# Per-sample geometry builders, see build_geometry_provider
VALID_GEOMETRY_BUILDERS = ("circular", "miller")

# Bounds on the temperature scale of the edge temperature boundary condition [keV].
# The beta-derived temperature estimate is off-scale on early low-density samples with a noisy stored energy
TE_SCALE_BOUNDS_KEV = (0.05, 5.0)


def check_torax_choices(transport_model: str, geometry_builder: str) -> None:
    """Raise for a transport model or a geometry builder the torax modules do not implement."""
    if transport_model not in TRANSPORT_COEFFICIENT_NAMES:
        raise ValueError(f"Unknown transport model '{transport_model}', valid: {sorted(TRANSPORT_COEFFICIENT_NAMES)}")
    if geometry_builder not in VALID_GEOMETRY_BUILDERS:
        raise ValueError(f"Unknown geometry builder '{geometry_builder}', valid: {VALID_GEOMETRY_BUILDERS}")


def make_torax_networks(
    n_inputs: int, transport_model: str, n_source_outputs: int, nn_width: int, nn_depth: int, key: jax.Array
) -> tuple[RtdMLP, RtdMLP, RtdMLP]:
    """The transport, sources and edge networks of a torax module.

    The transport network predicts the TRANSPORT_COEFFICIENT_NAMES of transport_model,
    the edge network the edge density and temperature fractions (bound_edge_coefficients).
    """
    _, key_transport, key_sources, key_edge = jax.random.split(key, 4)

    def mlp(out_size: int, subkey: jax.Array) -> RtdMLP:
        return RtdMLP(in_size=n_inputs, out_size=out_size, width_size=nn_width, depth=nn_depth, activation=Activation.RELU, key=subkey)

    return mlp(len(TRANSPORT_COEFFICIENT_NAMES[transport_model]), key_transport), mlp(n_source_outputs, key_sources), mlp(2, key_edge)


def make_step_fn_and_grid(torax_config: ToraxConfig | dict, transport_model: str) -> tuple[SimulationStepFn, tuple, tuple]:
    """The TORAX step function of a config, with its static face-center and hires grids.

    Per-sample geometry is rebuilt from traced inputs (build_geometry_provider),
    but the mesh and hires grids fix array shapes and must be concrete there,
    so they are harvested once from the config-built placeholder geometry (a cached lookup).
    They are tuples because static eqx fields must be hashable,
    and harvesting rather than rederiving from n_rho / hires_factor stays exact for non-uniform face centers.
    """
    if isinstance(torax_config, dict):
        torax_config = ToraxConfig.from_dict(torax_config)
    validate_transport_model_name(torax_config, transport_model)
    step_fn = torax_experimental.make_step_fn(torax_config)
    static_geo = step_fn.geometry_provider(0.0)
    return step_fn, tuple(static_geo.torax_mesh.face_centers.tolist()), tuple(np.array(static_geo.rho_hires_norm).tolist())


def cell_centers(face_centers: tuple) -> np.ndarray:
    """The cell-center grid of a TORAX mesh, rho_norm, which is the rho_tor_norm the datasets use."""
    return torax_pydantic.Grid1D(face_centers=np.array(face_centers)).cell_centers


def build_geometry_provider(
    geometry_builder: str, inputs, coeffs: dict, face_centers: tuple, rho_hires_norm: tuple, delta_exponent: float
) -> geometry_provider_lib.ConstantGeometryProvider:
    """The JAX-differentiable per-sample geometry of inputs (any Inputs with the shape fields), circular or miller.

    The miller builder reads its shape-profile coefficients from coeffs (geometry_shape_coefficients).
    """
    torax_mesh = torax_pydantic.Grid1D(face_centers=np.array(face_centers))
    rho_hires_norm_np = np.array(rho_hires_norm)
    if geometry_builder == "miller":
        geo = build_miller_geometry_jax(
            R_major=inputs.geometric_axis_r,
            a_minor=inputs.minor_radius,
            B_0=inputs.b_geo,
            elongation_LCFS=inputs.elongation,
            triangularity_upper=inputs.triangularity_upper,
            triangularity_lower=inputs.triangularity_lower,
            axis_elongation_fraction=jnp.squeeze(coeffs["axis_elongation_fraction"]),
            shafranov_shift_norm=jnp.squeeze(coeffs["shafranov_shift_norm"]),
            torax_mesh=torax_mesh,
            rho_hires_norm_np=rho_hires_norm_np,
            delta_exponent=delta_exponent,
        )
    else:
        geo = build_circular_geometry_jax(
            R_major=inputs.geometric_axis_r,
            a_minor=inputs.minor_radius,
            B_0=inputs.b_geo,
            elongation_LCFS=inputs.elongation,
            torax_mesh=torax_mesh,
            rho_hires_norm_np=rho_hires_norm_np,
        )
    return geometry_provider_lib.ConstantGeometryProvider(geo=geo)


def bound_source_coefficients(
    source_names: tuple[str, ...], nn_sources_out: jax.Array, n_e_line_average_1e20: jax.Array, volume_m3: jax.Array
) -> dict:
    """The fueling and heat deposition coefficients both torax module families share, bounded to physical ranges.

    source_names orders the sources network outputs.
      S_total: 0 - inf, softplus multiple of the device fueling scale particle_inventory / TAU_REF_S [1e21 / s],
        so the magnitude transfers between devices
      gaussian_location: 0 - 0.8 (deposition center in rho_norm)
      gaussian_width: 0.02 - 0.4 (deposition width in rho_norm)
      electron_heat_fraction: 0.2 - 0.95, the bias keeps the random init balanced at 0.5
        (ST NBI heating is electron-dominated, hence the high ceiling)
    """

    def out(name: str) -> jax.Array:
        idx = source_names.index(name)
        return nn_sources_out[idx : idx + 1]

    # Particle inventory in 1e21 electrons
    inventory = 0.1 * n_e_line_average_1e20 * volume_m3
    return {
        "S_total": jax.nn.softplus(out("S_total")) * inventory / TAU_REF_S,
        "gaussian_location": 0.8 * jax.nn.sigmoid(out("gaussian_location")),
        "gaussian_width": 0.02 + 0.38 * jax.nn.sigmoid(out("gaussian_width")),
        "electron_heat_fraction": 0.2 + 0.75 * jax.nn.sigmoid(out("electron_heat_fraction") - 0.4),
    }


def bound_edge_coefficients(nn_edge_out: jax.Array, n_e_line_average_1e20: jax.Array, te_approx_keV: jax.Array) -> dict:
    """The edge Dirichlet boundary conditions as NN-predicted fractions.

      n_e_right_bc = fraction in (0.01, 0.95) of the line-averaged density [1e20 m^-3]:
        a fixed edge density above the target profile acts as an infinite particle source, so it scales with the density
      T_e_right_bc = 20 eV + fraction of the clipped temperature estimate [keV]
    Both are floored, a near-vacuum edge ill-conditions the density equation.
    The floors are affine in the sigmoids, so the NN gradient path stays intact.
    The negative bias makes random-init edge temperatures a few tens of eV:
    te_approx is an overestimate, and a hot edge flattens the profile,
    which keeps threshold models like QLKNN subcritical and kills the gradient to the transport network.
    """
    te_scale = jnp.clip(te_approx_keV, *TE_SCALE_BOUNDS_KEV)
    return {
        "n_e_right_bc": (0.01 + 0.94 * jax.nn.sigmoid(nn_edge_out[0:1])) * n_e_line_average_1e20,
        "T_e_right_bc": 0.02 + jax.nn.sigmoid(nn_edge_out[1:2] - 5.0) * te_scale,
    }


def shared_provider_mapping(ip_MA: jax.Array, transport_model: str, coeffs: dict) -> dict:
    """Runtime-params overrides both torax module families share: Ip, the edge BCs, fueling, deposition and transport.

    The ion edge temperature is assumed equal to the electron one.
    """
    te_right_bc_update = torax_experimental.TimeVaryingScalarUpdate(value=coeffs["T_e_right_bc"])
    mapping = {
        "profile_conditions.Ip": torax_experimental.TimeVaryingScalarUpdate(value=jnp.atleast_1d(ip_MA * 1e6)),
        "profile_conditions.n_e_right_bc": torax_experimental.TimeVaryingScalarUpdate(value=coeffs["n_e_right_bc"] * 1e20),
        "profile_conditions.T_e_right_bc": te_right_bc_update,
        "profile_conditions.T_i_right_bc": te_right_bc_update,
        "sources.gas_puff.S_total": torax_experimental.TimeVaryingScalarUpdate(value=coeffs["S_total"] * 1e21),
        "sources.generic_heat.gaussian_location": torax_experimental.TimeVaryingScalarUpdate(value=coeffs["gaussian_location"]),
        "sources.generic_heat.gaussian_width": torax_experimental.TimeVaryingScalarUpdate(value=coeffs["gaussian_width"]),
        "sources.generic_heat.electron_heat_fraction": torax_experimental.TimeVaryingScalarUpdate(value=coeffs["electron_heat_fraction"]),
    }
    return mapping | transport_provider_mapping(transport_model, coeffs)


def interp_cell_plus_boundaries(values: jax.Array, face_centers: tuple, rho: jax.Array) -> jax.Array:
    """Cell values with both face values (CellVariable.cell_plus_boundaries) interpolated onto rho in [0, 1].

    The face values matter: jnp.interp flat-holds outside the data range, which would ignore the Dirichlet edge value at rho = 1,
    and the zero-gradient axis condition makes the rho = 0 face equal the innermost cell.
    """
    rho_full = jnp.concatenate([jnp.asarray(face_centers[:1]), jnp.asarray(cell_centers(face_centers)), jnp.asarray(face_centers[-1:])])
    return jnp.interp(rho, rho_full, values)


def interp_core_profiles(core_profiles, face_centers: tuple, rho: jax.Array) -> tuple[jax.Array, jax.Array]:
    """n_e [1e20 m^-3] and T_e [keV] of TORAX core profiles interpolated onto rho in [0, 1], see interp_cell_plus_boundaries."""
    ne_with_faces = core_profiles.n_e.cell_plus_boundaries() / 1e20
    te_with_faces = core_profiles.T_e.cell_plus_boundaries()
    ne = interp_cell_plus_boundaries(ne_with_faces, face_centers, rho)
    te = interp_cell_plus_boundaries(te_with_faces, face_centers, rho)
    return ne, te


class ProfilePredictorTorax(TimeIndepModule):
    """Steady-state profile estimate from a differentiable TORAX relaxation with NN-predicted coefficients."""

    rhogrid: tuple = eqx.field(static=True)
    # TORAX transport model the transport network parameterizes, a TRANSPORT_COEFFICIENT_NAMES key
    transport_model: str = eqx.field(static=True)
    # Per-sample geometry builder, one of VALID_GEOMETRY_BUILDERS
    geometry_builder: str = eqx.field(static=True)
    # Radial exponent p in delta(rho_norm) = delta_edge * rho_norm**p, only used by the miller builder
    delta_exponent: float = eqx.field(static=True)

    nn_transport: RtdMLP
    nn_sources: RtdMLP
    nn_edge: RtdMLP
    # Per-device stat stage over the 10 nn_inputs
    normalizer: FeatureNormalizer

    step_fn: SimulationStepFn = eqx.field(static=True)

    # Static mesh info for the JAX-differentiable geometry construction, see make_step_fn_and_grid
    _face_centers: tuple = eqx.field(static=True)
    _rho_hires_norm: tuple = eqx.field(static=True)
    # Upper bound on sub-steps in fixed_time_step, it makes the loop a differentiable scan
    max_steps: int = eqx.field(static=True)

    def __init__(
        self,
        nn_width: int,
        nn_depth: int,
        rhogrid: tuple,
        torax_config: ToraxConfig | dict,
        key: jax.random.PRNGKey,
        normalizer: FeatureNormalizer,
        transport_model: str,
        geometry_builder: str,
        delta_exponent: float,
    ):
        check_torax_choices(transport_model, geometry_builder)
        self.transport_model = transport_model
        self.geometry_builder = geometry_builder
        self.delta_exponent = float(delta_exponent)
        self.normalizer = normalizer
        self.nn_transport, self.nn_sources, self.nn_edge = make_torax_networks(
            N_NN_INPUTS, transport_model, len(SOURCE_COEFFICIENT_NAMES), nn_width, nn_depth, key
        )
        self.step_fn, self._face_centers, self._rho_hires_norm = make_step_fn_and_grid(torax_config, transport_model)
        # Coerce to tuple: arrays in static fields break pytree metadata
        # equality (ambiguous truth value) when two module instances coexist
        self.rhogrid = static_rhogrid(rhogrid)

        # With the fixed time-step calculator the step count to cover the window is deterministic,
        # the ceiling of the window over fixed_dt, plus 1 for the clipped final step that lands exactly on t_final
        numerics = self.step_fn.runtime_params_provider.numerics
        fixed_dt = float(numerics.fixed_dt.get_value(0.0))
        self.max_steps = int(np.ceil((numerics.t_final - numerics.t_initial) / fixed_dt)) + 1

    def _coerce_inputs(self, inputs: Inputs | xr.Dataset) -> Inputs:
        if isinstance(inputs, xr.Dataset):
            inputs = Inputs.from_dataset(inputs, jnp.array(self.rhogrid))
        return inputs

    def transport_coefficients(self, nn_transport_out: jax.Array, minor_radius: ArrayLike) -> dict:
        """Bound the raw transport-network outputs to physical ranges, see bound_transport_coefficients."""
        return bound_transport_coefficients(self.transport_model, nn_transport_out, minor_radius)

    def nn_coefficients(self, inputs: Inputs, debug: bool = False) -> dict:
        """Transport, source and edge coefficients from the networks, bounded so the TORAX solver stays stable.

        The shared source coefficients and edge BCs are bound_source_coefficients and bound_edge_coefficients.
        The auxiliary heating magnitude P_aux_total is NN-inferred here (beta_tor_norm encodes the stored energy it sustains),
        so every profile model family has identical inputs:
        0 - 4x the w_approx / TAU_REF_S power scale [MW],
        the -2 bias starts the random-init solver near ohmic-only behavior.
        """
        nn_inputs = self.normalizer(inputs.nn_inputs, inputs.ds_source_idx)
        nn_sources_out = self.nn_sources(nn_inputs)
        nn_transport_out = self.nn_transport(nn_inputs)
        p_aux_idx = SOURCE_COEFFICIENT_NAMES.index("P_aux_total")
        coeffs = {
            **self.transport_coefficients(nn_transport_out, inputs.minor_radius),
            **bound_source_coefficients(SOURCE_COEFFICIENT_NAMES, nn_sources_out, inputs.n_e_line_average_1e20, inputs.volume_approx),
            "P_aux_total": 4.0 * jax.nn.sigmoid(nn_sources_out[p_aux_idx : p_aux_idx + 1] - 2.0) * inputs.w_approx / TAU_REF_S,
            **bound_edge_coefficients(self.nn_edge(nn_inputs), inputs.n_e_line_average_1e20, inputs.te_approx),
            **geometry_shape_coefficients(self.geometry_builder, inputs, inputs.w_approx),
        }
        if debug:
            fmt = " ".join(f"{name}={{{name}}}" for name in coeffs)
            jax.debug.print("[nn] " + fmt + " nn_in={nn_in}", nn_in=nn_inputs, **coeffs)
        return coeffs

    def build_provider_and_geo(self, inputs: Inputs, coeffs: dict):
        """Runtime params provider and per-sample geometry of one relaxation.

        The initial profiles are parabolic (1 - rho^2) from a core anchor down to the NN edge BC,
        sampled on the static cell-center grid plus both endpoints (static shapes, so no retracing).
        Core = edge + positive keeps them strictly decreasing and continuous with the BC for any NN output,
        a discontinuity at the LCFS or an exactly flat profile can NaN the solver.
        """
        ne_right_bc = coeffs["n_e_right_bc"]
        te_right_bc = coeffs["T_e_right_bc"]
        rho_ic = jnp.array(np.concatenate([[0.0], cell_centers(self._face_centers), [1.0]]))
        ic_shape = 1.0 - rho_ic**2

        # Temperature core anchor: edge BC + ~2x te_approx, so the relaxation starts near the expected equilibrium
        # (a flat 0.3 keV start is several keV short on hot C-Mod samples and burns the few fixed steps on the transient).
        # The clip keeps a 0.3 keV floor where te_approx is small or unreliable
        te_core_init = te_right_bc + jnp.clip(2.0 * inputs.te_approx, 0.3, 10.0)
        t_init_update = torax_experimental.TimeVaryingArrayUpdate(
            value=(te_right_bc + (te_core_init - te_right_bc) * ic_shape)[jnp.newaxis, :],
            rho_norm=rho_ic,
        )
        # Density core anchor so the midplane chord average of the parabola matches the measured line average:
        # the chord mean of (1 - rho^2) is 2/3, so core = bc + 1.5 (line_avg - bc), above the BC since the BC is a fraction below 1
        ne_core_init = ne_right_bc + 1.5 * (inputs.n_e_line_average_1e20 - ne_right_bc)
        n_init_update = torax_experimental.TimeVaryingArrayUpdate(
            value=1e20 * (ne_right_bc + (ne_core_init - ne_right_bc) * ic_shape)[jnp.newaxis, :],
            rho_norm=rho_ic,
        )

        mapping = shared_provider_mapping(inputs.ip_MA, self.transport_model, coeffs) | {
            "profile_conditions.n_e": n_init_update,
            "profile_conditions.T_e": t_init_update,
            "profile_conditions.T_i": t_init_update,
            # NN-inferred auxiliary heating, absorption_fraction stays at its config value (degenerate with P_total)
            "sources.generic_heat.P_total": torax_experimental.TimeVaryingScalarUpdate(value=coeffs["P_aux_total"] * 1e6),
        }
        new_provider = self.step_fn.runtime_params_provider.update_provider_from_mapping(mapping)
        geo_provider = build_geometry_provider(
            self.geometry_builder, inputs, coeffs, self._face_centers, self._rho_hires_norm, self.delta_exponent
        )
        return new_provider, geo_provider

    @property
    def rho_norm_grid(self) -> np.ndarray:
        """Cell-center grid the TORAX core profiles live on, see cell_centers."""
        return cell_centers(self._face_centers)

    def __call__(self, inputs: Inputs | xr.Dataset, debug: bool = False) -> Outputs:
        inputs = self._coerce_inputs(inputs)
        coeffs = self.nn_coefficients(inputs, debug=debug)
        new_provider, geo_provider = self.build_provider_and_geo(inputs, coeffs)

        initial_state, initial_post = torax_experimental.get_initial_state_and_post_processed_outputs(
            step_fn=self.step_fn,
            runtime_params_overrides=new_provider,
            geometry_overrides=geo_provider,
        )
        if debug:
            cp0 = initial_state.core_profiles
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

        # TORAX evolves the profiles on rho_norm, which is the data's rho_tor_norm, so no coordinate mapping is needed
        ne, te = interp_core_profiles(state.core_profiles, self._face_centers, inputs.rho)
        return profile_outputs(self.rhogrid, ne, te)

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
                    t [s], n_e_1e20 [1e20 m^-3], t_e_keV [keV] and rho_tor_norm
                    (the static rho_norm cell grid).
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
                "n_e_1e20": np.asarray(cp.n_e.value) / 1e20,
                "t_e_keV": np.asarray(cp.T_e.value),
                RADIAL_DIM: self.rho_norm_grid,
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
