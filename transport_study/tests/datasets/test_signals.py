"""Tests for the store to working-unit conversion every study loads its data through."""

import numpy as np
import pytest
import xarray as xr

from transport_study.signals import HEATING_POWERS_MW, convert_to_working_units

R0 = 0.66
GEOMETRIC_AXIS_R = 0.68


def _store_selection(signed: bool = True) -> xr.Dataset:
    """Two shots of the signals b_geo and power_additional_MW derive from, ip and b0 negative on the second."""
    sign = np.array([1.0, -1.0 if signed else 1.0])[:, None]
    ones = np.ones((2, 3))
    return xr.Dataset(
        {
            "ip": (("shot", "time_idx"), sign * 8e5 * ones),
            "b0": (("shot", "time_idx"), sign * 5.4 * ones),
            "geometric_axis_r": (("shot", "time_idx"), GEOMETRIC_AXIS_R * ones),
            "r0": (("shot",), np.full(2, R0)),
            "power_nbi": (("shot", "time_idx"), 1e6 * ones),
            "power_ic": (("shot", "time_idx"), 2e6 * ones),
            "power_lh": (("shot", "time_idx"), 0.5e6 * ones),
            "power_ec": (("shot", "time_idx"), 0.25e6 * ones),
        },
        coords={"shot": [1, 2]},
    )


def test_signed_ip_and_b0_become_magnitudes_and_b_geo_follows():
    ds = convert_to_working_units(_store_selection())

    np.testing.assert_allclose(ds["ip_MA"].values, 0.8)
    np.testing.assert_allclose(ds["b0"].values, 5.4)
    np.testing.assert_allclose(ds["b_geo"].values, 5.4 * R0 / GEOMETRIC_AXIS_R)
    assert ds["ip_MA"].attrs["units"] == "MA"


def test_power_additional_sums_the_four_heating_powers():
    ds = convert_to_working_units(_store_selection())

    np.testing.assert_allclose(ds["power_additional_MW"].values, 3.75)
    assert set(HEATING_POWERS_MW) <= set(ds.data_vars)


def test_b0_without_geometry_raises():
    ds_store = _store_selection().drop_vars("r0")

    with pytest.raises(ValueError, match="r0"):
        convert_to_working_units(ds_store)


def test_partial_heating_powers_raise():
    ds_store = _store_selection().drop_vars("power_ec")

    with pytest.raises(ValueError, match="power_additional_MW"):
        convert_to_working_units(ds_store)
