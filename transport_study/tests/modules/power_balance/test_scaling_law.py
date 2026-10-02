"""Units of the tau_E scaling laws inside ScalingLawPredictor."""

import pytest

from transport_study.modules.power_balance.module import ScalingLawPredictor

# C-Mod-like operating point, DD plasma
IP_MA = 1.0
B0_T = 5.4
NE20 = 1.5
R0_M = 0.68
A_MINOR_M = 0.2176
KAPPA = 1.6
ISOTOPE_MASS = 2.0
P_AUX_MW = 1.0
P_OH_MW = 1.0


def test_confinement_laws_match_published_units():
    """The default L- and H-mode coefficients reproduce ITER89-P and IPB98(y,2) in their published units.

    Both fits take the line-averaged density in 1e19 m^-3 (ITER89-P is written in 1e20 m^-3 with the coefficient rescaled),
    while the module receives ne20, so a wrong density conversion shows up here.
    The references are written independently of the module, in the published variable set.
    """
    ne19 = 10.0 * NE20
    p_abs_mw = P_AUX_MW + P_OH_MW
    epsilon = A_MINOR_M / R0_M
    ipb98_s = (
        0.0562 * IP_MA**0.93 * B0_T**0.15 * ne19**0.41 * p_abs_mw**-0.69 * R0_M**1.97 * KAPPA**0.78 * epsilon**0.58 * ISOTOPE_MASS**0.19
    )
    iter89p_s = 0.048 * ISOTOPE_MASS**0.5 * IP_MA**0.85 * R0_M**1.2 * A_MINOR_M**0.3 * KAPPA**0.5 * NE20**0.1 * B0_T**0.2 * p_abs_mw**-0.5

    predictor = ScalingLawPredictor()
    inputs = ScalingLawPredictor.Inputs(
        ip_MA=IP_MA,
        b_geo=B0_T,
        geometric_axis_r=R0_M,
        minor_radius=A_MINOR_M,
        elongation=KAPPA,
        n_e_line_average_1e20=NE20,
        power_additional_MW=P_AUX_MW,
        power_ohm_MW=P_OH_MW,
    )
    output = predictor(inputs)

    assert ipb98_s == pytest.approx(0.0541, rel=1e-2)
    assert float(output.debug_info["taue_hmode"]) == pytest.approx(ipb98_s, rel=1e-6)
    # The module's ITER89 coefficient 0.038 is 0.048 * 10^-0.1 rounded, hence the looser tolerance
    assert float(output.debug_info["taue_lmode"]) == pytest.approx(iter89p_s, rel=5e-3)
