"""Dimensionless plasma parameters shared by the profile and transport predictor Inputs.

Free functions over the study's working units (MA, T, m, 1e20 m^-3, MJ, keV),
so both Inputs classes compute every feature one way and the feature order of each stays put.
Every betan <-> stored energy conversion goes through transport-validation-datasets'
normalized_beta / energy_mhd_from_normalized_beta, the formula the stores were built with.
The stores used the reconstruction's plasma volume there, the study only has volume_approx,
so a round trip through these functions carries the volume_approx / reconstruction volume ratio
(a few percent on C-Mod, about ten percent on MAST).
"""

import jax.numpy as jnp
from popsim.cfspopcon_jax.current_drive import calc_f_shaping, calc_q_star
from popsim.cfspopcon_jax.geometry import calc_plasma_volume
from popsim.math_utils import safe_log
from scipy.constants import epsilon_0, eV, mu_0
from transport_validation_datasets.machine import generic


def inverse_aspect_ratio(minor_radius, geometric_axis_r):
    """epsilon = a / R_geo."""
    return minor_radius / geometric_axis_r


def q_star(ip_MA, b_geo, geometric_axis_r, minor_radius, elongation, triangularity_upper, triangularity_lower):
    """Cylindrical safety factor with the popcon shaping factor, b_geo the vacuum field at the geometric axis."""
    epsilon = inverse_aspect_ratio(minor_radius, geometric_axis_r)
    delta = (triangularity_upper + triangularity_lower) / 2
    f_shaping = calc_f_shaping(epsilon, elongation, delta)
    return calc_q_star(b_geo, geometric_axis_r, epsilon, ip_MA, f_shaping)


def greenwald_fraction(n_e_line_average_1e20, ip_MA, minor_radius):
    """f_GW = n_e_line_average / (Ip / (pi a^2)), transport-validation-datasets' greenwald_fraction, the one the filters use."""
    ip_A = ip_MA * 1e6
    n_e_line_average_m3 = n_e_line_average_1e20 * 1e20
    return generic.greenwald_fraction(ip_A, minor_radius, n_e_line_average_m3)


def a_b0(minor_radius, b_geo):
    """The dimensional size-field product a B_geo [m T]."""
    return minor_radius * b_geo


def volume_approx(geometric_axis_r, minor_radius, elongation):
    """Plasma volume [m^3] of the popcon elongated-torus approximation."""
    epsilon = inverse_aspect_ratio(minor_radius, geometric_axis_r)
    return calc_plasma_volume(
        major_radius=geometric_axis_r,
        inverse_aspect_ratio=epsilon,
        areal_elongation=elongation,
    )


def beta_poloidal(energy_mhd_MJ, volume_m3, ip_MA, minor_radius, elongation):
    """Poloidal beta 2 mu0 <p> / B_pa^2 of a stored energy.

    <p> = 2 W / (3 V) and B_pa = mu0 Ip / L,
    with L the perimeter of an ellipse of the LCFS minor radius and elongation, 2 pi a sqrt((1 + kappa^2) / 2).
    """
    perimeter = 2.0 * jnp.pi * minor_radius * jnp.sqrt(0.5 * (1.0 + elongation**2))
    pressure_Pa = 2.0 * energy_mhd_MJ * 1e6 / (3.0 * volume_m3)
    b_pol = mu_0 * abs(ip_MA) * 1e6 / perimeter
    return 2.0 * mu_0 * pressure_Pa / b_pol**2


def beta_tor_from_beta_tor_norm(beta_tor_norm, ip_MA, minor_radius, b0):
    """Toroidal beta as a fraction from the IMAS percent beta_tor_norm = 100 beta_tor a |b0| / |Ip|[MA]."""
    return beta_tor_norm * abs(ip_MA) / (100.0 * minor_radius * abs(b0))


def beta_tor_norm_from_energy_mhd_MJ(energy_mhd_MJ, volume_m3, minor_radius, b0, ip_MA):
    """IMAS beta_tor_norm of a stored energy, the store's own formula in J and A."""
    return generic.normalized_beta(energy_mhd_MJ * 1e6, volume_m3, minor_radius, b0, ip_MA * 1e6)


def energy_mhd_MJ_from_beta_tor_norm(beta_tor_norm, volume_m3, minor_radius, b0, ip_MA):
    """Stored energy [MJ] of an IMAS beta_tor_norm, the store's own inverse in J and A."""
    energy_mhd_J = generic.energy_mhd_from_normalized_beta(beta_tor_norm, volume_m3, minor_radius, b0, ip_MA * 1e6)
    return energy_mhd_J / 1e6


def pressure_Pa_from_beta_tor(beta_tor, b0):
    """Volume-averaged pressure <p> = beta_tor b0^2 / (2 mu0), b0 at r0 as IMAS normalizes beta_tor."""
    return beta_tor * b0**2 / (2 * mu_0)


def te_approx_keV(beta_tor, b0, n_e_line_average_1e20):
    """Single-fluid temperature estimate T_e = <p> / n_e [keV]."""
    pressure_Pa = pressure_Pa_from_beta_tor(beta_tor, b0)
    pressure_keV20 = pressure_Pa / eV / 1e3 / 1e20
    return pressure_keV20 / n_e_line_average_1e20


def nu_star(te_keV, n_e_line_average_1e20, q_star, geometric_axis_r, epsilon):
    """Characteristic collisionality, https://arxiv.org/pdf/2406.18442 eqn 2.

    SI formula with the temperature in joules, rearranged so the physical constants
    and unit conversions fold into python-float coefficients before touching the arrays.
    float32 array intermediates would otherwise overflow (ne_m3 / te_J^2 ~ 1e49)
    or underflow (eV^4 ~ 6.6e-76) and produce inf * 0 = nan.
    """
    te_eV = te_keV * 1e3
    # Coulomb logarithm of debye_length over b90,
    # log of 4 pi eps0^1.5 te_J^1.5 / (e^3 ne_m3^0.5) with te_J = te_eV * e
    lambda_coeff = 4 * jnp.pi * epsilon_0**1.5 / (eV**1.5 * 1e10)
    ln_lambda = safe_log(lambda_coeff * te_eV**1.5 / jnp.sqrt(n_e_line_average_1e20))
    # e^4 / (2 pi eps0^2) * ne_m3 / te_J^2
    collision_coeff = eV**2 / (2 * jnp.pi * epsilon_0**2) * 1e20
    collision_term = collision_coeff * n_e_line_average_1e20 / te_eV**2
    geometry_term = q_star * geometric_axis_r / (epsilon**1.5)
    return collision_term * geometry_term * ln_lambda
