"""Tests for the causality of the 0D signal helpers (datasets/zero_d_signals.py, profile_grids.held_signal_on_grid).

No stored value may draw on a later sample.
The cases are kept identical to transport-validation-datasets' tests/test_generic.py.
"""

import numpy as np
import pytest

from transport_study.datasets.profile_grids import held_signal_on_grid
from transport_study.datasets.zero_d_signals import (
    MU0,
    ohmic_power,
    trailing_boxcar_mean,
)


def test_held_signal_takes_the_last_sample_at_or_before_and_leaves_gaps_nan():
    # 5 ms source with a missing sample at 15 ms and a 20 ms gap after 25 ms
    grid = np.round(np.arange(60) * 1e-3, 3)
    source_times = np.array([0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.045, 0.050])
    values = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0])

    values_on_grid = held_signal_on_grid(source_times, values, grid)

    # 12 ms holds the 10 ms sample, never the 15 ms or 20 ms one
    assert values_on_grid[12] == 3.0
    # The missing 15 ms sample leaves 10 ms held to 17.5 ms, then NaN until 20 ms
    assert values_on_grid[17] == 3.0
    assert np.isnan(values_on_grid[18])
    assert values_on_grid[20] == 5.0
    # Past 1.5 periods after 25 ms nothing is held, the 45 ms sample is not pulled back
    assert values_on_grid[32] == 6.0
    assert np.isnan(values_on_grid[33:45]).all()
    assert values_on_grid[45] == 7.0


def test_held_signal_float64_source_on_a_float32_grid_is_held_at_its_own_grid_time():
    # A float32 grid time can sit just below a float64 sample at the same millisecond,
    # which must not leave it holding the sample before
    grid = np.round(np.arange(1001) * 1e-3, 3).astype("float32")
    source_times = np.arange(1001) * 1e-3
    values = np.arange(1001, dtype=float)

    values_on_grid = held_signal_on_grid(source_times, values, grid)

    np.testing.assert_array_equal(values_on_grid, values)


def test_held_signal_holds_on_the_clock_of_the_finite_samples():
    # A 0.1 ms clock populated only every 5 ms, as the MAST esm group stores pphix
    source_times = np.arange(500) * 1e-4
    values = np.full(500, np.nan)
    values[::50] = np.arange(10.0)
    grid = np.round(np.arange(50) * 1e-3, 3)

    values_on_grid = held_signal_on_grid(source_times, values, grid)

    # Each 5 ms sample is held up to the next one
    np.testing.assert_array_equal(values_on_grid, np.repeat(np.arange(10.0), 5))
    # A lone finite sample has no period to hold for
    values_lone = np.full(500, np.nan)
    values_lone[100] = 1.0
    values_lone_on_grid = held_signal_on_grid(source_times, values_lone, grid)
    assert np.isnan(values_lone_on_grid).all()


def test_trailing_boxcar_a_later_sample_does_not_change_earlier_ones():
    values = np.ones(30)
    values_spiked = values.copy()
    values_spiked[20] = 100.0

    smoothed = trailing_boxcar_mean(values, 5e-3, 1e-3)
    smoothed_spiked = trailing_boxcar_mean(values_spiked, 5e-3, 1e-3)

    np.testing.assert_array_equal(smoothed_spiked[:20], smoothed[:20])
    # The spike enters at its own sample and leaves 5 samples later
    assert smoothed_spiked[20] == pytest.approx((4.0 + 100.0) / 5.0)
    assert smoothed_spiked[25] == 1.0


def test_ohmic_power_current_ramp_matches_the_backward_difference_closed_form():
    # Linear Ip ramp at fixed li and R, so dW_pol/dt over one step is
    # mu0 R li / 4 * dIp/dt * (Ip_n + Ip_n-1)
    times = np.arange(10) * 1e-3
    ip_ramp_rate = 2e6
    ip = 5e5 + ip_ramp_rate * times
    v_loop = np.full(times.size, 1.5)
    li = np.full(times.size, 1.2)
    r_axis = np.full(times.size, 0.68)

    p_ohm = ohmic_power(times, ip, v_loop, li, r_axis)

    dw_pol_dt = MU0 * 0.68 * 1.2 / 4.0 * ip_ramp_rate * (ip[1:] + ip[:-1])
    expected = ip[1:] * v_loop[1:] - dw_pol_dt
    assert np.isnan(p_ohm[0])
    np.testing.assert_allclose(p_ohm[1:], expected, rtol=1e-12)
