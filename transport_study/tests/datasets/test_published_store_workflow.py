"""Tests for the C-Mod / MAST store build from a transport-validation-datasets published store.

PublishedStoreWorkflow is what every C-Mod and MAST study dataset is built from,
so the store it writes must be exactly the on-disk schema whatever else the published store carries.
These build a tiny store in the published layout and run the real build on it.
"""

import numpy as np
import pytest
import xarray as xr

from transport_study.signals import (
    PREDICTION_STORE_NAME,
    STORE_SIGNAL_UNITS,
    STORE_SIGNALS,
)

RHO = np.linspace(0.0, 1.1, 12)
N_TIME_PADDED = 60
SHOT_LENGTHS = {1160503001: 50, 1160503002: 30, 1160503003: 40}
SIGNED_SHOT = 1160503002


def _published_store(path) -> xr.Dataset:
    """A published-layout store: trailing NaN padding, time as a data variable, signed ip / b0 on one shot,
    no power_ec, and variables the build must leave out."""
    shots = np.array(list(SHOT_LENGTHS))
    n_shots = shots.size
    mask_valid = np.zeros((n_shots, N_TIME_PADDED), dtype=bool)
    for i_shot, n_valid in enumerate(SHOT_LENGTHS.values()):
        mask_valid[i_shot, :n_valid] = True

    def padded(values_valid: np.ndarray) -> np.ndarray:
        """NaN where the shot has ended, broadcast over any trailing dims."""
        values = np.broadcast_to(values_valid, (n_shots, N_TIME_PADDED, *values_valid.shape[2:])).astype(np.float32)
        mask = mask_valid.reshape(n_shots, N_TIME_PADDED, *([1] * (values.ndim - 2)))
        return np.where(mask, values, np.nan).astype(np.float32)

    time = padded(np.arange(N_TIME_PADDED)[None, :] * 1e-3 + 0.1)
    sign = np.where(shots == SIGNED_SHOT, -1.0, 1.0)[:, None]
    scalars = {
        "ip": sign * 8e5,
        "b0": sign * 5.4,
        "energy_mhd": 6e4,
        "beta_tor_norm": 1.0,
        "n_e_line_average": 1e20,
        "minor_radius": 0.22,
        "geometric_axis_r": 0.68,
        "elongation": 1.6,
        "triangularity_upper": 0.4,
        "triangularity_lower": 0.5,
        "power_ohm": 1e6,
        "power_radiated": 4e5,
        "power_nbi": 0.0,
        "power_ic": 2e6,
        "power_lh": 0.0,
        "fresh_profile": 1.0,
        "fresh_equilibrium": 1.0,
        "t_e_fit_status": 0.0,
    }
    profile_shape = (1 - RHO**2)[None, None, :] + 0.05
    profiles = {
        "t_e": 3e3 * profile_shape,
        "t_e_error": 1e2 * profile_shape,
        "t_e_gradient": -6e3 * RHO[None, None, :],
        "t_e_gradient_error": 2e2 * profile_shape,
        "n_e": 1.5e20 * profile_shape,
        "n_e_error": 5e18 * profile_shape,
        "n_e_gradient": -3e20 * RHO[None, None, :],
        "n_e_gradient_error": 1e19 * profile_shape,
    }
    data_vars = {name: (("shot", "time_idx"), padded(np.full((n_shots, N_TIME_PADDED), 1.0) * value)) for name, value in scalars.items()}
    data_vars |= {name: (("shot", "time_idx", "rho_tor_norm"), padded(values)) for name, values in profiles.items()}
    data_vars["time"] = (("shot", "time_idx"), time)
    data_vars["cocos"] = (("shot",), np.full(n_shots, 7.0, dtype=np.float32))
    data_vars["psirz"] = (("shot", "time_idx", "r_grid", "z_grid"), padded(np.ones((n_shots, N_TIME_PADDED, 3, 3))))
    ds = xr.Dataset(data_vars, coords={"shot": shots, "rho_tor_norm": RHO, "r_grid": np.arange(3.0), "z_grid": np.arange(3.0)})
    for name, unit in STORE_SIGNAL_UNITS.items():
        if name in ds:
            ds[name].attrs["units"] = unit
    ds.to_zarr(path, mode="w")
    return ds


@pytest.fixture(scope="module")
def built_store(tmp_path_factory):
    """The source store and the prediction store the C-Mod workflow builds from it."""
    from transport_study.datasets.cmod.cmod_dataset import CModDataWorkflow

    tmp = tmp_path_factory.mktemp("published_store")
    ds_published = _published_store(tmp / "published.zarr")
    workflow = CModDataWorkflow(ds_name="cmod_test", data_assembly_dir=tmp, published_store_path=tmp / "published.zarr")
    workflow.run_processed_data_workflow()
    ds_built = xr.open_zarr(workflow.store_path(PREDICTION_STORE_NAME)).load()
    return ds_published, ds_built


def test_store_holds_exactly_the_on_disk_schema(built_store):
    """Every stored signal with its SI unit, nothing the published store carries beyond it."""
    _, ds_built = built_store

    assert set(ds_built.data_vars) == {*STORE_SIGNALS, "time"}
    assert set(ds_built.dims) == {"shot", "time_idx", "rho_tor_norm"}
    for name, unit in STORE_SIGNAL_UNITS.items():
        assert ds_built[name].attrs["units"] == unit, name


def test_every_shot_kept_and_padding_trimmed(built_store):
    """No culls here, and the time axis is cut back to the longest shot."""
    _, ds_built = built_store

    assert sorted(ds_built["shot"].values.tolist()) == sorted(SHOT_LENGTHS)
    assert ds_built.sizes["time_idx"] == max(SHOT_LENGTHS.values())
    for shot, n_valid in SHOT_LENGTHS.items():
        assert int(ds_built["time"].sel(shot=shot).notnull().sum()) == n_valid


def test_signed_ip_and_b0_become_magnitudes(built_store):
    """The C-Mod store keeps the source sign, the study uses magnitudes."""
    ds_published, ds_built = built_store

    for name in ["ip", "b0"]:
        built = ds_built[name].sel(shot=SIGNED_SHOT).dropna("time_idx").values
        published = ds_published[name].sel(shot=SIGNED_SHOT).dropna("time_idx").values
        np.testing.assert_allclose(built, np.abs(published))


def test_absent_heating_system_is_zero_power(built_store):
    """power_ec is not in the published store, the device has no such system: zero in valid time."""
    _, ds_built = built_store

    mask_valid = ds_built["time"].notnull()
    assert float(np.abs(ds_built["power_ec"].where(mask_valid)).max()) == 0.0
    assert bool(ds_built["power_ec"].where(mask_valid).notnull().sum() == mask_valid.sum())


def test_profiles_pass_through_unchanged(built_store):
    """The profiles and their companions are copied, not resampled or rescaled."""
    ds_published, ds_built = built_store

    for name in ["t_e", "n_e_gradient_error"]:
        for shot, n_valid in SHOT_LENGTHS.items():
            built = ds_built[name].sel(shot=shot).isel(time_idx=slice(0, n_valid)).values
            published = ds_published[name].sel(shot=shot).isel(time_idx=slice(0, n_valid)).values
            np.testing.assert_array_equal(built, published)
