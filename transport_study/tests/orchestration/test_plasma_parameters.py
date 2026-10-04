"""The study's beta <-> stored energy path agrees with the store's own formula.

The stores write beta_tor_norm with transport-validation-datasets' normalized_beta,
the modules invert it with volume_approx as the volume.
With the same volume the round trip must close, and it must close to the reconstruction volume
exactly by the volume_approx / volume ratio.
"""

import numpy as np
from scipy.constants import mu_0
from transport_validation_datasets.machine.generic import (
    energy_mhd_from_normalized_beta,
    normalized_beta,
)

from transport_study.modules import plasma_parameters

IP_MA = np.array([0.8, 1.2, -0.6])
B0 = np.array([5.4, 2.5, -0.5])
MINOR_RADIUS = np.array([0.22, 0.25, 0.6])
GEOMETRIC_AXIS_R = np.array([0.68, 0.88, 0.9])
ELONGATION = np.array([1.6, 1.4, 2.0])
ENERGY_MHD_MJ = np.array([0.06, 0.3, 0.02])


def test_beta_round_trip_closes_with_the_same_volume():
    volume_m3 = plasma_parameters.volume_approx(GEOMETRIC_AXIS_R, MINOR_RADIUS, ELONGATION)

    beta_tor_norm = plasma_parameters.beta_tor_norm_from_energy_mhd_MJ(ENERGY_MHD_MJ, volume_m3, MINOR_RADIUS, B0, IP_MA)
    energy_back_MJ = plasma_parameters.energy_mhd_MJ_from_beta_tor_norm(beta_tor_norm, volume_m3, MINOR_RADIUS, B0, IP_MA)

    np.testing.assert_allclose(energy_back_MJ, ENERGY_MHD_MJ, rtol=1e-10)


def test_beta_tor_norm_matches_the_store_formula_in_si():
    volume_m3 = plasma_parameters.volume_approx(GEOMETRIC_AXIS_R, MINOR_RADIUS, ELONGATION)

    beta_tor_norm = plasma_parameters.beta_tor_norm_from_energy_mhd_MJ(ENERGY_MHD_MJ, volume_m3, MINOR_RADIUS, B0, IP_MA)
    beta_tor_norm_store = normalized_beta(ENERGY_MHD_MJ * 1e6, volume_m3, MINOR_RADIUS, B0, IP_MA * 1e6)

    np.testing.assert_allclose(beta_tor_norm, beta_tor_norm_store, rtol=1e-12)
    assert np.all(beta_tor_norm > 0), "magnitudes of ip and b0, whatever the source sign"


def test_fraction_beta_is_the_imas_percent_unrolled():
    volume_m3 = plasma_parameters.volume_approx(GEOMETRIC_AXIS_R, MINOR_RADIUS, ELONGATION)
    beta_tor_norm = plasma_parameters.beta_tor_norm_from_energy_mhd_MJ(ENERGY_MHD_MJ, volume_m3, MINOR_RADIUS, B0, IP_MA)

    beta_tor = plasma_parameters.beta_tor_from_beta_tor_norm(beta_tor_norm, IP_MA, MINOR_RADIUS, B0)
    pressure_Pa = 2.0 * ENERGY_MHD_MJ * 1e6 / (3.0 * volume_m3)
    beta_tor_direct = 2.0 * mu_0 * pressure_Pa / B0**2

    # The store's MU0 and scipy's mu_0 differ in the tenth digit
    np.testing.assert_allclose(beta_tor, beta_tor_direct, rtol=1e-8)


def test_store_volume_mismatch_scales_the_recovered_energy():
    """A store built with the reconstruction volume V gives back W volume_approx / V."""
    volume_approx_m3 = plasma_parameters.volume_approx(GEOMETRIC_AXIS_R, MINOR_RADIUS, ELONGATION)
    volume_reconstruction_m3 = 1.1 * volume_approx_m3
    beta_tor_norm_store = normalized_beta(ENERGY_MHD_MJ * 1e6, volume_reconstruction_m3, MINOR_RADIUS, B0, IP_MA * 1e6)

    energy_back_MJ = plasma_parameters.energy_mhd_MJ_from_beta_tor_norm(beta_tor_norm_store, volume_approx_m3, MINOR_RADIUS, B0, IP_MA)
    energy_back_store_J = energy_mhd_from_normalized_beta(beta_tor_norm_store, volume_approx_m3, MINOR_RADIUS, B0, IP_MA * 1e6)

    np.testing.assert_allclose(energy_back_MJ, ENERGY_MHD_MJ / 1.1, rtol=1e-10)
    np.testing.assert_allclose(energy_back_MJ * 1e6, energy_back_store_J, rtol=1e-12)
