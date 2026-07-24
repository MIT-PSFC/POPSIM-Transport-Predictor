"""Test stubs for the 2026-07-20 torax knob-range changes.

Covers the three changes made after the direct knob-fit ceiling probe:
S_total scaled by the device particle-inventory fueling scale, the widened
V_face_coeff range, and the lowered n_e_right_bc floor. Stubs only, to be
implemented on request.
"""


def test_s_total_scales_with_device_inventory():
    """S_total should scale with ne20_line_avg * volume_approx / TAU_REF_S.

    Build two Inputs differing only in ne20_line_avg and volume-relevant
    geometry (R0, a_minor, kappa), run _nn_coefficients with a fixed
    network, and check the S_total ratio equals the inventory ratio.
    """


def test_s_total_random_init_magnitude_sane():
    """A random-init sources net should give S_total near the fueling scale.

    softplus(~0) ~ 0.7, so S_total should be order 0.7 * inventory / TAU_REF_S
    for both a C-Mod-like and a MAST-like input vector, not the fixed ~0.7e21
    per s of the old absolute-units mapping.
    """


def test_v_face_coeff_range_and_init():
    """V_face_coeff must cover (-4, 2) and sit near -0.1 at zero input.

    Sweep nn_transport_out[4] over a wide range and check the mapped
    coefficient approaches -4 and 2 at the extremes, and equals the TORAX
    default -0.1 within a few percent when the raw output is zero.
    Ceiling raised 1 -> 2 on 2026-07-24 after the dance2 rail check.
    """


def test_d_face_floor_lowered():
    """D_face_c1 and D_face_c2 must cover (0.01, 5).

    Saturate nn_transport_out[2] and [3] both ways and check the mapped
    coefficients approach 0.01 and 5, in particular that the lower limit is
    0.01 not the old 0.1 (dance2 trained models railed the 0.1 floor on
    77-100 percent of MAST test slices). Applies to both
    ProfilePredictorTorax and the transport predictor mirror.
    """


def test_electron_heat_fraction_range_and_init():
    """electron_heat_fraction must cover (0.2, 0.95) with init near 0.5.

    Saturate the sources output both ways and check the limits (ceiling
    raised 0.8 -> 0.95 on 2026-07-24), and check a zero raw output maps to
    about 0.5 so both heat channels keep gradient at random init. Applies
    to both ProfilePredictorTorax and the transport predictor mirror.
    """


def test_gaussian_width_floor_lowered():
    """gaussian_width must cover (0.02, 0.4).

    Saturate the sources output both ways and check the limits, in
    particular that the lower limit is 0.02 not the old 0.05. Applies to
    both ProfilePredictorTorax and the transport predictor mirror.
    """


def test_ne_right_bc_floor_lowered():
    """n_e_right_bc fraction of line average must cover (0.01, 0.95).

    Saturate nn_edge_out[0] both ways and check the fraction limits, in
    particular that the lower limit is 0.01 not the old 0.05.
    """


def test_evolve_prescribed_s_total_units_unchanged():
    """evolve(prescribed={"S_total": x}) still means x in 1e21 particles per s.

    The inventory scaling happens inside the NN mapping, prescribed
    overrides replace the final coefficient, so a prescribed value must
    reach the gas_puff provider multiplied only by the fixed 1e21.
    """
