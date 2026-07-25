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
from transport_study.tests.datasets.synthetic_shot import (
    N_BLOCKS,
    N_T,
    RHO,
    SHOT,
    TS_BLOCK,
    raw_shot,
)


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
    ds = _bad_core_block(raw_shot(), 3)

    processed = cmod_workflow.device_specific_processing(ds)

    for var in PROFILE_FIT_VARS:
        assert processed[var].isel(shot=0, time_idx=3 * TS_BLOCK).isnull().all(), f"{var} survived the cull"
        assert processed[var].isel(shot=0, time_idx=4 * TS_BLOCK).notnull().all(), f"{var} culled beyond the bad block"


def test_cmod_process_fn_labels_one_fresh_slice_per_measurement(cmod_workflow):
    """fresh_profiles is what the profile study filters on: exactly the first
    slice of each forward-filled block, and nothing from a culled block."""
    cmod_workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
    _bad_core_block(raw_shot(), 3).to_netcdf(cmod_workflow.raw_data_dir / f"{SHOT}.nc")

    processed = cmod_workflow.process_fn(SHOT)

    assert processed is not None, "synthetic shot should survive processing"
    fresh = processed["fresh_profiles"].isel(shot=0).values
    time = processed["time"].values
    fresh_times = np.round(time[fresh == 1] * 1000).astype(int)
    # filter_ds trims end_margin_s off the end of the Ip record, and block 3 was culled
    last_idx = N_T - 1 - round(cmod_workflow.end_margin_s * 1e3)
    expected = [b * TS_BLOCK for b in range(N_BLOCKS) if b != 3 and b * TS_BLOCK <= last_idx]
    np.testing.assert_array_equal(fresh_times, expected)


def test_fresh_labels_survive_a_mid_shot_filter_gap(cmod_workflow):
    """A forward-filled slice left over after filter_ds punches a mid-shot hole
    must stay stale.

    process_fn labels fresh_profiles BEFORE filter_ds, and filter_ds drops whole
    timeslices (`where(..., drop=True)`), so the record it returns is compacted.
    The diff-based labelling only reads as stale on a compacted record because
    the labels were already attached: the first slice surviving a gap that ate a
    measurement carries a NEW profile relative to the last slice before the gap,
    so relabelling after the drop would call it fresh even though it is a
    forward-filled copy. This pins that ordering. The profile study trains on
    fresh_profiles == 1 alone, so a mislabel there feeds it a stale profile
    against current inputs.
    """
    gap_block = 10
    gap = slice(gap_block * TS_BLOCK - 5, gap_block * TS_BLOCK + 5)  # eats the block's measurement slice
    betan = np.full((1, N_T), 0.8)
    betan[0, gap] = 2.0  # over the C-Mod filter_config max, so filter_ds drops the slices
    cmod_workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
    raw_shot(betan=betan).to_netcdf(cmod_workflow.raw_data_dir / f"{SHOT}.nc")

    processed = cmod_workflow.process_fn(SHOT)

    assert processed is not None, "the gap should punch a hole, not cull the shot"
    time_ms = np.round(processed["time"].values * 1000).astype(int)
    ne = processed["ne20_rho"].isel(shot=0).values
    fresh = processed["fresh_profiles"].isel(shot=0).values

    i_gap = int(np.flatnonzero(time_ms == gap_block * TS_BLOCK + 5)[0])
    assert time_ms[i_gap - 1] == gap_block * TS_BLOCK - 6, "the gap slices should have been dropped"
    # The setup only bites if the surviving slice really does carry a new profile
    assert not np.array_equal(ne[i_gap], ne[i_gap - 1]), "gap did not span a measurement, test is not exercising anything"
    assert fresh[i_gap] == 0, "forward-filled slice after a filter gap was labelled fresh"

    # The measurement whose own slice was dropped is gone, every other block keeps its one fresh slice
    last_idx = N_T - 1 - round(cmod_workflow.end_margin_s * 1e3)
    expected = [b * TS_BLOCK for b in range(N_BLOCKS) if b != gap_block and b * TS_BLOCK <= last_idx]
    np.testing.assert_array_equal(time_ms[fresh == 1], expected)


def test_cmod_process_fn_culls_shot_with_impossible_stored_energy(cmod_workflow):
    """Stored energy the input power cannot account for means a broken power
    record (energy_sanity_cull), so the whole shot goes."""
    cmod_workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
    wtot = np.linspace(0.05, 1.5, N_T)[None, :]  # 1.45 MJ rise on 1 MW of ohmic over 1.5 s
    raw_shot(Wtot_MJ=wtot, P_oh_MW=np.full((1, N_T), 0.1)).to_netcdf(cmod_workflow.raw_data_dir / f"{SHOT}.nc")

    assert cmod_workflow.process_fn(SHOT) is None


def test_cmod_transient_filter_cuts_from_before_the_event(cmod_workflow):
    """A UFO spike in P_oh ends the usable part of the shot 10 ms early."""
    p_oh = np.full((1, N_T), 1.0)
    p_oh[0, 1000:] = 9.0  # over the 5 MW transient threshold
    ds = raw_shot(P_oh_MW=p_oh)

    filtered = cmod_workflow.filter_ds(ds)

    assert float(filtered["time"].max()) < 0.99
    assert float(filtered["time"].max()) > 0.97


def test_mast_culls_profile_companions_together(tmp_path):
    """MAST culls on negative fits rather than a low core, same companion rule."""
    from transport_study.datasets.mast.mast_dataset import MASTDataWorkflow

    shotlist = tmp_path / "shotlist"
    shotlist.write_text("30284\n")
    workflow = MASTDataWorkflow(ds_name="mast_test", shotlist_file=shotlist, data_assembly_dir=tmp_path)

    ds = raw_shot()
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
