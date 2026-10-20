"""Tests for the differentiable Miller geometry builder.

The Miller surfaces must carry the toroidal-flux label exactly (closed forms for concentric ellipses),
reduce to the circular builder's metrics at large aspect ratio, give the elliptic-cylinder edge q,
stay finite and physical in the MAST spherical-tokamak regime,
and stay differentiable in every shape parameter.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import mu_0
from torax._src.torax_pydantic import torax_pydantic

from transport_study.modules.profile_predictor.torax_module import (
    build_circular_geometry_jax,
    build_miller_geometry_jax,
)
from transport_study.tests.sample_data import requires_sample_data

N_FACES = 26

# Fields passed to the Geometry constructor, used for finiteness checks
GEOMETRY_FIELDS = [
    "Phi",
    "Phi_face",
    "volume",
    "volume_face",
    "area",
    "area_face",
    "vpr",
    "vpr_face",
    "spr",
    "spr_face",
    "delta_face",
    "trapped_fraction_face",
    "g0",
    "g0_face",
    "g1",
    "g1_face",
    "g2",
    "g2_face",
    "g3",
    "g3_face",
    "gm4",
    "gm4_face",
    "gm5",
    "gm5_face",
    "g2g3_over_rhon",
    "g2g3_over_rhon_face",
    "g2g3_over_rhon_hires",
    "F",
    "F_face",
    "F_hires",
    "R_in",
    "R_in_face",
    "R_out",
    "R_out_face",
    "elongation",
    "elongation_face",
    "spr_hires",
]

# The circular builder drops O(epsilon^2) terms in these, so they only
# agree with the Miller quadrature at small inverse aspect ratio
APPROX_FIELDS = [
    "g0",
    "g0_face",
    "g1",
    "g1_face",
    "g2",
    "g2_face",
    "g3",
    "g3_face",
    "gm4",
    "gm4_face",
    "gm5",
    "gm5_face",
    "g2g3_over_rhon",
    "g2g3_over_rhon_face",
]


def make_mesh() -> tuple[torax_pydantic.Grid1D, np.ndarray]:
    face_centers = np.linspace(0.0, 1.0, N_FACES)
    torax_mesh = torax_pydantic.Grid1D(face_centers=face_centers)
    rho_hires_norm = np.linspace(0.0, 1.0, 4 * (N_FACES - 1) + 1)
    return torax_mesh, rho_hires_norm


def build_miller(
    R_major,
    a_minor,
    B_0,
    kappa,
    triangularity_upper,
    triangularity_lower,
    axis_elongation_fraction=0.7,
    shafranov_shift_norm=0.0,
    delta_exponent=2.0,
):
    torax_mesh, rho_hires_norm = make_mesh()
    return build_miller_geometry_jax(
        R_major=jnp.asarray(R_major),
        a_minor=jnp.asarray(a_minor),
        B_0=jnp.asarray(B_0),
        elongation_LCFS=jnp.asarray(kappa),
        triangularity_upper=jnp.asarray(triangularity_upper),
        triangularity_lower=jnp.asarray(triangularity_lower),
        axis_elongation_fraction=jnp.asarray(axis_elongation_fraction),
        shafranov_shift_norm=jnp.asarray(shafranov_shift_norm),
        torax_mesh=torax_mesh,
        rho_hires_norm_np=rho_hires_norm,
        delta_exponent=delta_exponent,
    )


def build_circular(R_major, a_minor, B_0, kappa):
    torax_mesh, rho_hires_norm = make_mesh()
    return build_circular_geometry_jax(
        R_major=jnp.asarray(R_major),
        a_minor=jnp.asarray(a_minor),
        B_0=jnp.asarray(B_0),
        elongation_LCFS=jnp.asarray(kappa),
        torax_mesh=torax_mesh,
        rho_hires_norm_np=rho_hires_norm,
    )


@pytest.mark.parametrize("kappa", [1.0, 1.8])
def test_miller_flux_label_exact_for_concentric_ellipse(kappa):
    # A concentric ellipse of constant elongation encloses the vacuum toroidal flux
    # Phi(r) = 2 pi kappa B_0 R (R - sqrt(R^2 - r^2)) and the volume 2 pi^2 R r^2 kappa at any aspect ratio,
    # so at MAST's aspect ratio every mesh point must land on the surface of its toroidal-flux radius
    R_major, a_minor, B_0 = 0.81, 0.58, 0.5
    geo = build_miller(R_major, a_minor, B_0, kappa, 0.0, 0.0, axis_elongation_fraction=1.0)

    phi_b = 2.0 * np.pi * kappa * B_0 * R_major * (R_major - np.sqrt(R_major**2 - a_minor**2))
    np.testing.assert_allclose(float(geo.Phi_face[-1]), phi_b, rtol=1e-10)

    rho_face_norm = np.linspace(0.0, 1.0, N_FACES)
    phi_face = phi_b * rho_face_norm**2
    r_face = np.sqrt(R_major**2 - (R_major - phi_face / (2.0 * np.pi * kappa * B_0 * R_major)) ** 2)
    np.testing.assert_allclose(np.asarray(geo.R_out_face) - R_major, r_face, atol=1e-5 * a_minor)
    np.testing.assert_allclose(np.asarray(geo.volume_face), 2.0 * np.pi**2 * R_major * r_face**2 * kappa, rtol=1e-4, atol=1e-12)


def test_miller_vpr_spr_integrate_to_volume_area():
    # vpr and spr are dV / drho_norm and dA / drho_norm through the flux-label chain rule,
    # so integrating them over the toroidal-flux mesh must recover the contour volume and area
    # for a fully shaped, shifted MAST surface
    geo = build_miller(0.85, 0.6, 0.5, 1.7, 0.4, 0.1, shafranov_shift_norm=0.15)
    rho_face_norm = np.linspace(0.0, 1.0, N_FACES)
    volume_from_vpr = np.concatenate([[0.0], np.cumsum(0.5 * np.diff(rho_face_norm) * (geo.vpr_face[1:] + geo.vpr_face[:-1]))])
    area_from_spr = np.concatenate([[0.0], np.cumsum(0.5 * np.diff(rho_face_norm) * (geo.spr_face[1:] + geo.spr_face[:-1]))])
    np.testing.assert_allclose(volume_from_vpr[1:], np.asarray(geo.volume_face)[1:], rtol=5e-3)
    np.testing.assert_allclose(area_from_spr[1:], np.asarray(geo.area_face)[1:], rtol=5e-3)


def test_miller_edge_q_elliptic_cylinder_limit():
    # TORAX's edge q from the Ip boundary condition is g2g3_over_rhon * F / (8 pi^3 mu_0 Ip),
    # at large aspect ratio an ellipse must give the elliptic-cylinder q = 2 pi a^2 B_0 / (mu_0 R Ip) (1 + kappa^2) / 2,
    # with q_correction_factor 1 so nothing rescales it
    R_major, a_minor, B_0, kappa, ip_A = 100.0, 1.0, 2.0, 1.8, 1e6
    geo = build_miller(R_major, a_minor, B_0, kappa, 0.0, 0.0, axis_elongation_fraction=1.0)
    q_edge = float(geo.g2g3_over_rhon_face[-1] * geo.F_face[-1] / (8.0 * np.pi**3 * mu_0 * ip_A) * geo.q_correction_factor)
    q_cylinder = 2.0 * np.pi * a_minor**2 * B_0 / (mu_0 * R_major * ip_A) * (1.0 + kappa**2) / 2.0
    np.testing.assert_allclose(q_edge, q_cylinder, rtol=1e-3)


def test_miller_circular_limit_metrics_small_epsilon():
    # At edge inverse aspect ratio 0.1 and kappa = 1 the circular
    # builder's large-aspect-ratio metric approximations are accurate to
    # O(epsilon^2) ~ 1%, so the Miller quadrature must land within 2%
    circ = build_circular(3.0, 0.3, 2.5, 1.0)
    miller = build_miller(3.0, 0.3, 2.5, 1.0, 0.0, 0.0)
    for field in APPROX_FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(miller, field)),
            np.asarray(getattr(circ, field)),
            rtol=2e-2,
            atol=1e-12,
            err_msg=field,
        )


def test_miller_mast_invariants():
    geo = build_miller(0.85, 0.6, 0.5, 1.7, 0.4, 0.1, shafranov_shift_norm=0.15)
    for field in GEOMETRY_FIELDS:
        values = np.asarray(getattr(geo, field))
        assert np.all(np.isfinite(values)), field

    # Positivity on the cell grid (cell centers exclude the axis)
    for field in ["vpr", "spr", "g0", "g1", "g2", "g3", "gm4", "gm5", "volume", "area"]:
        assert np.all(np.asarray(getattr(geo, field)) > 0.0), field

    # Volume strictly increasing with rho
    assert np.all(np.diff(np.asarray(geo.volume_face)) > 0.0)

    # Cauchy-Schwarz with the flux-surface-average weights
    gm4 = np.asarray(geo.gm4)
    gm5 = np.asarray(geo.gm5)
    assert np.all(gm4 * gm5 >= 1.0 - 1e-10)
    g0 = np.asarray(geo.g0)
    g1 = np.asarray(geo.g1)
    assert np.all(g1 >= g0**2 * (1.0 - 1e-10))

    # Axis values of the rigorous g2*g3/rho_norm form are zeroed
    assert float(geo.g2g3_over_rhon_face[0]) == 0.0
    assert float(geo.g2g3_over_rhon_hires[0]) == 0.0

    # Triangularity actually threaded into the geometry
    delta_face = np.asarray(geo.delta_face)
    assert delta_face[0] == 0.0
    np.testing.assert_allclose(delta_face[-1], 0.25, rtol=1e-6)


def test_miller_extreme_triangularity_finite():
    # DIII-D stores hold edge triangularities up to 0.9 (the builder's clip),
    # where Sauter's trapped-fraction fit has a negative effective inverse aspect ratio,
    # every geometry field must stay finite there
    geo = build_miller(1.67, 0.58, 2.0, 1.93, 0.9, 0.9, shafranov_shift_norm=0.3)
    for field in GEOMETRY_FIELDS:
        assert np.all(np.isfinite(np.asarray(getattr(geo, field)))), field


def test_miller_updown_swap_invariant():
    # The parameterization is Z-symmetric, so swapping triangularity_upper and
    # triangularity_lower mirrors the contour about the midplane and all
    # flux-surface-averaged metrics must be unchanged
    geo_a = build_miller(0.85, 0.6, 0.5, 1.7, 0.4, 0.1, shafranov_shift_norm=0.15)
    geo_b = build_miller(0.85, 0.6, 0.5, 1.7, 0.1, 0.4, shafranov_shift_norm=0.15)
    for field in GEOMETRY_FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(geo_a, field)),
            np.asarray(getattr(geo_b, field)),
            rtol=1e-6,
            atol=1e-12,
            err_msg=field,
        )


def test_miller_geometry_differentiable():
    def scalar(args):
        R_major, a_minor, B_0, kappa, triangularity_upper, triangularity_lower, axis_fraction, shift_norm, exponent = args
        torax_mesh, rho_hires_norm = make_mesh()
        geo = build_miller_geometry_jax(
            R_major=R_major,
            a_minor=a_minor,
            B_0=B_0,
            elongation_LCFS=kappa,
            triangularity_upper=triangularity_upper,
            triangularity_lower=triangularity_lower,
            axis_elongation_fraction=axis_fraction,
            shafranov_shift_norm=shift_norm,
            torax_mesh=torax_mesh,
            rho_hires_norm_np=rho_hires_norm,
            delta_exponent=exponent,
        )
        return (
            jnp.sum(geo.g1_face)
            + jnp.sum(geo.gm4)
            + jnp.sum(geo.gm5)
            + jnp.sum(geo.volume)
            + jnp.sum(geo.vpr_face)
            + jnp.sum(geo.g2g3_over_rhon_face)
            + jnp.sum(geo.trapped_fraction_face)
        )

    args = tuple(jnp.asarray(v) for v in (0.85, 0.6, 0.5, 1.7, 0.4, 0.1, 0.7, 0.15, 2.0))
    grads = jax.grad(scalar)(args)
    names = (
        "R_major",
        "minor_radius",
        "B_0",
        "elongation",
        "triangularity_upper",
        "triangularity_lower",
        "axis_elongation_fraction",
        "shafranov_shift_norm",
        "delta_exponent",
    )
    for name, grad in zip(names, grads, strict=True):
        assert np.isfinite(float(grad)), name
    # The shape-profile knobs must reach the geometry, a later network can only learn them through these gradients
    for name, grad in zip(names[6:], grads[6:], strict=True):
        assert float(grad) != 0.0, name
    grad_top = float(grads[4])
    grad_bot = float(grads[5])
    assert grad_top != 0.0
    assert grad_bot != 0.0
    # At an up-down asymmetric point the two sensitivities must differ
    assert grad_top != grad_bot


@pytest.mark.slow
@requires_sample_data
@pytest.mark.parametrize("transport_model", ["gyrobohm", "qlknn"])
def test_miller_evolve_mast_no_nan(transport_model, make_torax_module, sample_timeslices):
    # Direct replay of the failure mode behind the NaN/Inf training
    # retries: run MAST samples through the full TORAX relaxation with
    # the shaped geometry and require finite profiles at every step
    timeslices = sample_timeslices("mast-high.nc", n_slices=2)
    module = make_torax_module(transport_model, geometry_builder="miller")

    for timeslice in timeslices:
        steps, _coeffs = module.evolve(timeslice)
        for step in steps:
            assert np.all(np.isfinite(step["n_e_1e20"])), f"NaN n_e_1e20 at t={step['t']}"
            assert np.all(np.isfinite(step["t_e_keV"])), f"NaN t_e_keV at t={step['t']}"
