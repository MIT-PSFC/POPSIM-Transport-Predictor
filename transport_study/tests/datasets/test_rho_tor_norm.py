"""Tests for the shared Phi_N map (datasets/rho_tor_norm.py).

The phi_n_map cases are kept identical to transport-validation-datasets'
tests/test_rho_tor_norm.py (TestPhiNMap), so both repos are held to the same numbers.
"""

import numpy as np
import pytest
from scipy.special import xlogy

from transport_study.datasets.rho_tor_norm import SECANT_PSI_N, phi_n_map

# Flux surfaces at uniform rho_pol, so psi_N = rho_pol^2 is not uniform
PSI_N_SURFACES = np.linspace(0.0, 1.0, 41) ** 2


def log_q(psi_n):
    """q = 1.2 - 0.8 ln(1 - psi_N), the form a diverted q takes near the LCFS."""
    return 1.2 - 0.8 * np.log(1.0 - psi_n)


def log_q_integral(psi_n):
    """Closed-form integral of log_q from 0 to psi_N, finite at psi_N = 1."""
    one_minus = 1.0 - psi_n
    return 1.2 * psi_n + 0.8 * (xlogy(one_minus, one_minus) - one_minus + 1.0)


def diverted_qpsi(psi_n_surfaces):
    """log_q on the surfaces, infinite at the LCFS."""
    qpsi = np.full(psi_n_surfaces.size, np.inf)
    qpsi[:-1] = log_q(psi_n_surfaces[:-1])
    return qpsi


class TestPhiNMap:
    def test_constant_q_is_psi_n_for_either_sign(self):
        # Constant q to the LCFS (a limited plasma) makes Phi_N = psi_N, and the sign of q cancels
        psi_n = np.linspace(0.0, 1.0, 101)
        qpsi = np.full(PSI_N_SURFACES.size, 3.0)

        phi_n = phi_n_map(PSI_N_SURFACES, qpsi, "secant").phi_n(psi_n)
        phi_n_flipped_q = phi_n_map(PSI_N_SURFACES, -qpsi, "secant").phi_n(psi_n)

        np.testing.assert_allclose(phi_n, psi_n, atol=1e-12)
        np.testing.assert_allclose(phi_n_flipped_q, psi_n, atol=1e-12)

    def test_diverted_tail_matches_log_q(self):
        # q diverging at the LCFS goes through the analytic tail past the last finite surface (psi_N ~ 0.95)
        psi_n = np.concatenate([np.linspace(0.0, 0.9, 10), np.linspace(0.951, 0.9999, 50), [1.0]])
        qpsi = diverted_qpsi(PSI_N_SURFACES)

        phi_n = phi_n_map(PSI_N_SURFACES, qpsi, "secant").phi_n(psi_n)

        phi_n_expected = log_q_integral(psi_n) / log_q_integral(1.0)
        # Simpson over 41 surfaces is good to a few 1e-4 against q steepening toward the LCFS
        np.testing.assert_allclose(phi_n, phi_n_expected, rtol=1e-3)
        assert phi_n[-1] == 1.0

    def test_finite_q_at_the_lcfs_stays_close_to_the_diverted_integral(self):
        # EFIT writes a finite q(1), here q a little inside the LCFS, on a uniform 129-point grid
        psi_n_grid = np.linspace(0.0, 1.0, 129)
        qpsi = log_q(np.minimum(psi_n_grid, 1.0 - 1e-3))
        psi_n = np.linspace(0.0, 1.0, 201)

        rho_tor_norm = np.sqrt(phi_n_map(psi_n_grid, qpsi, "secant").phi_n(psi_n))

        rho_tor_norm_expected = np.sqrt(log_q_integral(psi_n) / log_q_integral(1.0))
        np.testing.assert_allclose(rho_tor_norm, rho_tor_norm_expected, atol=2e-3)

    def test_rejects_unusable_q_profiles(self):
        qpsi_good = diverted_qpsi(PSI_N_SURFACES)
        qpsi_nan = qpsi_good.copy()
        qpsi_nan[5] = np.nan
        # Diverging next to the axis leaves too few finite surfaces to fit the tail to
        qpsi_axis = qpsi_good.copy()
        qpsi_axis[2:] = np.inf
        # A tail whose q falls toward the LCFS
        qpsi_falling = qpsi_good.copy()
        qpsi_falling[-5:-1] = [4.0, 3.5, 3.0, 2.5]
        qpsi_sign_change = np.full(PSI_N_SURFACES.size, 2.0)
        qpsi_sign_change[10] = -2.0

        for qpsi in [qpsi_nan, qpsi_axis, qpsi_falling, qpsi_sign_change]:
            assert phi_n_map(PSI_N_SURFACES, qpsi, "secant") is None

    def test_rising_q_matches_closed_form_and_continues_along_the_secant(self):
        # q = 1 + 3 psi^2 gives Phi_N = (psi + psi^3) / 2 inside the LCFS
        psi_n = np.linspace(-0.01, 1.2, 242)
        psi_n_grid = np.linspace(0.0, 1.0, 129)
        qpsi = 1.0 + 3.0 * psi_n_grid**2

        phi_n = phi_n_map(psi_n_grid, qpsi, "secant").phi_n(psi_n)

        psi_n_clipped = np.maximum(psi_n, 0.0)
        mask_inside = psi_n_clipped <= 1.0
        phi_n_inside = (psi_n_clipped + psi_n_clipped**3) / 2
        np.testing.assert_allclose(phi_n[mask_inside], phi_n_inside[mask_inside], atol=1e-6)
        phi_n_at_secant_start = (SECANT_PSI_N + SECANT_PSI_N**3) / 2
        secant_slope = (1.0 - phi_n_at_secant_start) / (1.0 - SECANT_PSI_N)
        phi_n_outside = 1.0 + secant_slope * (psi_n[~mask_inside] - 1.0)
        np.testing.assert_allclose(phi_n[~mask_inside], phi_n_outside, rtol=1e-6)
        assert np.all(np.diff(phi_n[psi_n >= 0]) > 0)

    @pytest.mark.parametrize("sol_extension", ["secant", "tangent"])
    def test_inverse_round_trips_across_the_lcfs(self, sol_extension):
        psi_n = np.array([0.0, 0.03, 0.4, 0.97, 1.0, 1.05, 1.4, np.nan])
        qpsi = diverted_qpsi(PSI_N_SURFACES) if sol_extension == "secant" else log_q(PSI_N_SURFACES * 0.99)
        phi_n_mapping = phi_n_map(PSI_N_SURFACES, qpsi, sol_extension)

        phi_n = phi_n_mapping.phi_n(psi_n)
        psi_n_back = phi_n_mapping.psi_n(phi_n)

        # The inverse interpolates a dense Phi_N table inside the LCFS
        np.testing.assert_allclose(psi_n_back, psi_n, atol=1e-7)
