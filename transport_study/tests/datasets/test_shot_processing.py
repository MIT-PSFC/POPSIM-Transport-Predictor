"""Tests for the per-shot processing chain (raw netCDF -> processed shot).

DataWorkflow.process_fn is what turns a raw per-shot file into what the studies
train on: device-specific culling of bad GP fits, the fresh_profiles labelling
the profile study filters on, range filtering, and the shot-level culls. It runs
on real device data in production, so these build a synthetic raw shot instead
and exercise the C-Mod and MAST implementations directly.
"""

import numpy as np
import pytest
import xarray as xr

from transport_study.datasets.workflow import PROFILE_FIT_VARS

SHOT = 1160503010
N_T = 1500  # 1.5 s at 1 kHz, comfortably over min_shot_duration
RHO = np.linspace(0.0, 1.1, 12)
# Thomson measures far slower than the 1 kHz grid, so each fit is forward-filled
# over a block of timeslices (see the workflows' assemble_shot)
TS_BLOCK = 20
N_BLOCKS = N_T // TS_BLOCK


def _raw_shot(**overrides) -> xr.Dataset:
    """A synthetic raw shot that passes every filter and cull.

    Profiles change once per TS_BLOCK slices and are constant within a block,
    matching the forward-filled layout of a real raw file.
    """
    time = np.arange(N_T) * 1e-3
    amplitude = 2.0 + 0.01 * np.repeat(np.arange(N_BLOCKS), TS_BLOCK)[:, None]
    profile = amplitude * (1 - np.tanh((RHO - 0.9) / 0.08))[None, :] / 2 + 0.05
    scalars = {
        "Wtot_MJ": np.full(N_T, 0.05),
        "P_oh_MW": np.full(N_T, 1.0),
        "P_rad_MW": np.full(N_T, 0.3),
        "P_ICRF_MW": np.zeros(N_T),
        "P_LH_MW": np.zeros(N_T),
        "P_NBI_MW": np.zeros(N_T),
        "P_ECRH_MW": np.zeros(N_T),
        "Ip_MA": np.full(N_T, 0.8),
        "B0": np.full(N_T, 5.0),
        "betan": np.full(N_T, 0.8),
        "ne20_line_avg": np.full(N_T, 1.0),
        "R0": np.full(N_T, 0.68),
        "kappa": np.full(N_T, 1.6),
        "a_minor": np.full(N_T, 0.22),
        "delta_top": np.full(N_T, 0.3),
        "delta_bot": np.full(N_T, 0.5),
        "beta_p": np.full(N_T, 0.5),
        "ne20_edge": np.full(N_T, 0.5),
    }
    profiles = {
        "Te_keV_rho": profile,
        "Te_keV_rho_error": 0.1 * profile,
        "Te_keV_rho_grad": np.gradient(profile, RHO, axis=1),
        "Te_keV_rho_grad_error": np.full_like(profile, 0.5),
        "ne20_rho": profile / 2,
        "ne20_rho_error": 0.05 * profile,
        "ne20_rho_grad": np.gradient(profile / 2, RHO, axis=1),
        "ne20_rho_grad_error": np.full_like(profile, 0.2),
    }
    data_vars = {name: (("time_idx",), vals) for name, vals in scalars.items()}
    data_vars |= {name: (("time_idx", "rho"), vals) for name, vals in profiles.items()}
    ds = xr.Dataset(data_vars, coords={"time_idx": np.arange(N_T), "time": ("time_idx", time), "rho": RHO})
    ds = ds.expand_dims(shot=[SHOT])
    for name, vals in overrides.items():
        ds[name] = vals if isinstance(vals, xr.DataArray) else (ds[name].dims, vals)
    return ds


@pytest.fixture(scope="module")
def cmod_workflow(tmp_path_factory):
    from transport_study.datasets.cmod.cmod_dataset import CModDataWorkflow

    tmp = tmp_path_factory.mktemp("cmod_processing")
    shotlist = tmp / "shotlist"
    shotlist.write_text(f"{SHOT}\n")
    return CModDataWorkflow(ds_name="cmod_test", shotlist_file=shotlist, data_assembly_dir=tmp, gp_fit_rho=RHO)


def _bad_core_block(ds: xr.Dataset, i_block: int) -> xr.Dataset:
    """Drive one TS block's core Te below the C-Mod validity threshold.

    A whole block, not a single slice: the cull masks read only the profile
    values, which are constant across a forward-filled block, so production
    culling always takes a block at a time.
    """
    te = ds["Te_keV_rho"].values.copy()
    block = slice(i_block * TS_BLOCK, (i_block + 1) * TS_BLOCK)
    te[0, block, :] = 0.5  # core Te < 1.0 keV -> culled by device_specific_processing
    return ds.assign(Te_keV_rho=(ds["Te_keV_rho"].dims, te))


def test_cmod_culls_profile_companions_together(cmod_workflow):
    """A culled block must lose its error bars and gradients too, or the profile
    study sees an error bar for a profile that is not there."""
    ds = _bad_core_block(_raw_shot(), 3)

    processed = cmod_workflow.device_specific_processing(ds)

    for var in PROFILE_FIT_VARS:
        assert processed[var].isel(shot=0, time_idx=3 * TS_BLOCK).isnull().all(), f"{var} survived the cull"
        assert processed[var].isel(shot=0, time_idx=4 * TS_BLOCK).notnull().all(), f"{var} culled beyond the bad block"


def test_cmod_process_fn_labels_one_fresh_slice_per_measurement(cmod_workflow):
    """fresh_profiles is what the profile study filters on: exactly the first
    slice of each forward-filled block, and nothing from a culled block."""
    cmod_workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
    _bad_core_block(_raw_shot(), 3).to_netcdf(cmod_workflow.raw_data_dir / f"{SHOT}.nc")

    processed = cmod_workflow.process_fn(SHOT)

    assert processed is not None, "synthetic shot should survive processing"
    fresh = processed["fresh_profiles"].isel(shot=0).values
    time = processed["time"].values
    fresh_times = np.round(time[fresh == 1] * 1000).astype(int)
    # filter_ds trims the last 50 ms, and block 3 was culled
    expected = [b * TS_BLOCK for b in range(N_BLOCKS) if b != 3 and b * TS_BLOCK <= N_T - 51]
    np.testing.assert_array_equal(fresh_times, expected)


def test_cmod_process_fn_culls_shot_with_impossible_stored_energy(cmod_workflow):
    """Stored energy the input power cannot account for means a broken power
    record (energy_sanity_cull), so the whole shot goes."""
    cmod_workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
    wtot = np.linspace(0.05, 1.5, N_T)[None, :]  # 1.45 MJ rise on 1 MW of ohmic over 1.5 s
    _raw_shot(Wtot_MJ=wtot, P_oh_MW=np.full((1, N_T), 0.1)).to_netcdf(cmod_workflow.raw_data_dir / f"{SHOT}.nc")

    assert cmod_workflow.process_fn(SHOT) is None


def test_cmod_transient_filter_cuts_from_before_the_event(cmod_workflow):
    """A UFO spike in P_oh ends the usable part of the shot 10 ms early."""
    p_oh = np.full((1, N_T), 1.0)
    p_oh[0, 1000:] = 9.0  # over the 5 MW transient threshold
    ds = _raw_shot(P_oh_MW=p_oh)

    filtered = cmod_workflow.filter_ds(ds)

    assert float(filtered["time"].max()) < 0.99
    assert float(filtered["time"].max()) > 0.97


def test_mast_culls_profile_companions_together(tmp_path):
    """MAST culls on negative fits rather than a low core, same companion rule."""
    from transport_study.datasets.mast.mast_dataset import MASTDataWorkflow

    shotlist = tmp_path / "shotlist"
    shotlist.write_text("30284\n")
    workflow = MASTDataWorkflow(ds_name="mast_test", shotlist_file=shotlist, data_assembly_dir=tmp_path)

    ds = _raw_shot()
    ne = ds["ne20_rho"].values.copy()
    block = slice(3 * TS_BLOCK, 4 * TS_BLOCK)
    ne[0, block, 3] = -0.2  # negative fit inside rho < 1.0
    ds = ds.assign(ne20_rho=(ds["ne20_rho"].dims, ne))

    processed = workflow.device_specific_processing(ds)

    for var in PROFILE_FIT_VARS:
        assert processed[var].isel(shot=0, time_idx=3 * TS_BLOCK).isnull().all(), f"{var} survived the cull"
        assert processed[var].isel(shot=0, time_idx=4 * TS_BLOCK).notnull().all(), f"{var} culled beyond the bad block"
    # ne20_edge is re-read from the culled profile, and fGW is available to filter_ds
    assert processed["ne20_edge"].isel(shot=0, time_idx=3 * TS_BLOCK).isnull()
    assert "fGW" in processed
