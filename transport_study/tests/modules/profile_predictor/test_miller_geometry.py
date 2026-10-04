"""Tests for the differentiable Miller geometry builder.

The Miller quadrature must reduce to the circular builder's closed forms in the
zero-triangularity limit, stay finite and physical in the MAST spherical-tokamak
regime, and stay differentiable in every shape parameter (the NN predicts them).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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

# Exact for both builders at delta = 0: the circular builder's closed forms
# for these are exact for a concentric ellipse at any aspect ratio, and the
# Miller quadrature integrates the same trig polynomials exactly
EXACT_FIELDS = [
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


def build_miller(R_major, a_minor, B_0, kappa, triangularity_upper, triangularity_lower, delta_exponent=2.0):
    torax_mesh, rho_hires_norm = make_mesh()
    return build_miller_geometry_jax(
        R_major=jnp.asarray(R_major),
        a_minor=jnp.asarray(a_minor),
        B_0=jnp.asarray(B_0),
        elongation_LCFS=jnp.asarray(kappa),
        triangularity_upper=jnp.asarray(triangularity_upper),
        triangularity_lower=jnp.asarray(triangularity_lower),
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


@pytest.mark.parametrize(
    "R_major,a_minor,kappa",
    [
        (1, 0.3, 1.0),  # conventional aspect ratio, circular
        (6.2, 0.62, 1.7),  # conventional aspect ratio, elongated
        (0.85, 0.6, 1.7),  # MAST-like spherical tokamak
    ],
)
def test_miller_circular_limit_exact_fields(R_major, a_minor, kappa):
    circ = build_circular(R_major, a_minor, 2.5, kappa)
    miller = build_miller(R_major, a_minor, 2.5, kappa, 0.0, 0.0)
    for field in EXACT_FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(miller, field)),
            np.asarray(getattr(circ, field)),
            rtol=1e-6,
            atol=1e-12,
            err_msg=field,
        )


def test_miller_circular_limit_metrics_small_epsilon():
    # At small inverse aspect ratio (0.1) and kappa = 1 the circular
    # builder's large-aspect-ratio metric approximations are accurate to
    # O(epsilon^2) ~ 1%, so the Miller quadrature must land within 2%
    circ = build_circular(1, 0.3, 2.5, 1.0)
    miller = build_miller(1, 0.3, 2.5, 1.0, 0.0, 0.0)
    for field in APPROX_FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(miller, field)),
            np.asarray(getattr(circ, field)),
            rtol=2e-2,
            atol=1e-12,
            err_msg=field,
        )


def test_miller_mast_invariants():
    geo = build_miller(0.85, 0.6, 0.5, 1.7, 0.4, 0.1)
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


def test_miller_updown_swap_invariant():
    # The parameterization is Z-symmetric, so swapping triangularity_upper and
    # triangularity_lower mirrors the contour about the midplane and all
    # flux-surface-averaged metrics must be unchanged
    geo_a = build_miller(0.85, 0.6, 0.5, 1.7, 0.4, 0.1)
    geo_b = build_miller(0.85, 0.6, 0.5, 1.7, 0.1, 0.4)
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
        R_major, a_minor, B_0, kappa, triangularity_upper, triangularity_lower = args
        torax_mesh, rho_hires_norm = make_mesh()
        geo = build_miller_geometry_jax(
            R_major=R_major,
            a_minor=a_minor,
            B_0=B_0,
            elongation_LCFS=kappa,
            triangularity_upper=triangularity_upper,
            triangularity_lower=triangularity_lower,
            torax_mesh=torax_mesh,
            rho_hires_norm_np=rho_hires_norm,
        )
        return jnp.sum(geo.g1_face) + jnp.sum(geo.gm4) + jnp.sum(geo.gm5) + jnp.sum(geo.volume) + jnp.sum(geo.g2g3_over_rhon_face)

    args = tuple(jnp.asarray(v) for v in (0.85, 0.6, 0.5, 1.7, 0.4, 0.1))
    grads = jax.grad(scalar)(args)
    for name, grad in zip(
        ("R_major", "minor_radius", "B_0", "elongation", "triangularity_upper", "triangularity_lower"), grads, strict=True
    ):
        assert np.isfinite(float(grad)), name
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
