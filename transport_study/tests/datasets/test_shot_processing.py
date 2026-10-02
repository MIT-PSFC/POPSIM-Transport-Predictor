"""Tests for the per-shot processing chain (raw netCDF -> processed shot).

RawFileWorkflow.process_fn is what turns a raw per-shot file into what the studies
train on: the fresh_profile labelling the profile study filters on, the filters,
and the shot-level culls. It runs on real device data in production, so these build
a synthetic raw shot instead and exercise the TCV implementation directly.
"""

import numpy as np
import pytest

from transport_study.signals import PREDICTION_STORE_NAME
from transport_study.tests.datasets.synthetic_shot import (
    N_BLOCKS,
    N_T,
    SHOT,
    TS_BLOCK,
    raw_shot,
)


@pytest.fixture(scope="module")
def tcv_workflow(tmp_path_factory):
    from transport_study.datasets.tcv.tcv_dataset import TCVDataWorkflow

    tmp = tmp_path_factory.mktemp("tcv_processing")
    shotlist = tmp / "shotlist"
    shotlist.write_text(f"{SHOT}\n")
    workflow = TCVDataWorkflow(ds_name="tcv_test", shotlist_file=shotlist, data_assembly_dir=tmp)
    workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
    return workflow


def _processed(workflow, ds):
    """Write ds as the shot's raw file and run it through the prediction store's processing."""
    ds.to_netcdf(workflow.raw_data_dir / f"{SHOT}.nc")
    return workflow.process_fn(SHOT, workflow.STORE_VARIABLES[PREDICTION_STORE_NAME])


def test_process_fn_labels_one_fresh_slice_per_measurement(tcv_workflow):
    """fresh_profile is what the profile study filters on: exactly the first
    slice of each forward-filled block."""
    processed = _processed(tcv_workflow, raw_shot())

    assert processed is not None, "synthetic shot should survive processing"
    fresh = processed["fresh_profile"].isel(shot=0).values
    time = processed["time"].values
    fresh_times = np.round(time[fresh == 1] * 1000).astype(int)
    # filter_ds trims end_margin_s off the end of the ip record
    last_idx = N_T - 1 - round(tcv_workflow.end_margin_s * 1e3)
    expected = [b * TS_BLOCK for b in range(N_BLOCKS) if b * TS_BLOCK <= last_idx]
    np.testing.assert_array_equal(fresh_times, expected)


def test_fresh_labels_survive_a_filter_gap(tcv_workflow):
    """The first slice kept after a filter gap that ate a measurement must stay stale.

    load_shot labels fresh_profile BEFORE filter_ds cuts the record down to its longest segment.
    The first kept slice after the gap carries a NEW profile relative to the slice before the gap,
    so labelling after the cut would call it fresh even though it is a forward-filled copy.
    This pins that ordering. The profile study trains on fresh_profile == 1 alone,
    so a mislabel there feeds it a stale profile against current inputs.
    """
    gap_block = 10
    gap = slice(gap_block * TS_BLOCK - 5, gap_block * TS_BLOCK + 5)  # eats the block's measurement slice
    energy_mhd = np.full((1, N_T), 3e4)
    energy_mhd[0, gap] = 10.0  # under the TCV energy_mhd min_filter threshold, so filter_ds cuts a gap
    ds_raw = raw_shot(energy_mhd=energy_mhd)

    processed = _processed(tcv_workflow, ds_raw)

    assert processed is not None, "the gap should cut the shot, not cull it"
    time_ms = np.round(processed["time"].values * 1000).astype(int)
    fresh = processed["fresh_profile"].isel(shot=0).values
    # The longer stretch after the gap is kept
    assert time_ms[0] == gap_block * TS_BLOCK + 5
    # The setup only bites if the first kept slice really does carry a new profile
    n_e_raw = ds_raw["n_e"].isel(shot=0).values
    assert not np.array_equal(n_e_raw[time_ms[0]], n_e_raw[gap.start - 1]), "gap did not span a measurement"
    assert fresh[0] == 0, "forward-filled slice after a filter gap was labelled fresh"

    # Every later block keeps its one fresh slice
    last_idx = N_T - 1 - round(tcv_workflow.end_margin_s * 1e3)
    expected = [b * TS_BLOCK for b in range(gap_block + 1, N_BLOCKS) if b * TS_BLOCK <= last_idx]
    np.testing.assert_array_equal(time_ms[fresh == 1], expected)


def test_process_fn_culls_shot_with_impossible_stored_energy(tcv_workflow):
    """Stored energy the input power cannot account for means a broken power
    record (energy_sanity_cull), so the whole shot goes."""
    energy_mhd = np.linspace(1e3, 4.9e5, N_T)[None, :]  # 0.49 MJ rise on 10 kW of ohmic over 1.5 s
    power_ohm = np.full((1, N_T), 1e4)

    assert _processed(tcv_workflow, raw_shot(energy_mhd=energy_mhd, power_ohm=power_ohm)) is None


def test_filter_ds_ends_before_the_plasma_current_termination(tcv_workflow):
    """A record whose ip stays finite but near 0 after the plasma (DIII-D reads it out to 8 s)
    is cut end_margin_s before the last slice with |ip| at its min_filter threshold.
    Measured from the last finite ip instead, the margin would never apply."""
    idx_plasma_end = 1200
    ip = np.full((1, N_T), 3e5)
    ip[0, idx_plasma_end + 1 :] = 1e3

    filtered = tcv_workflow.filter_ds(raw_shot(ip=ip))

    time_ms = np.round(filtered["time"].values * 1000).astype(int)
    assert time_ms[-1] == idx_plasma_end - round(tcv_workflow.end_margin_s * 1e3)


def test_filter_ds_cuts_nan_and_max_failures_as_gaps(tcv_workflow):
    """A NaN power_radiated sample and a Greenwald fraction over its maximum are gaps,
    and only the longest segment between them is kept."""
    power_radiated = np.full((1, N_T), 1e5)
    power_radiated[0, 300] = np.nan
    # n_GW of 0.3 MA in a 0.24 m minor radius is ~1.7e20 m^-3, so 5e20 is a Greenwald fraction near 3
    n_e_line_average = np.full((1, N_T), 4e19)
    n_e_line_average[0, 900:910] = 5e20

    filtered = tcv_workflow.filter_ds(raw_shot(power_radiated=power_radiated, n_e_line_average=n_e_line_average))

    time_ms = np.round(filtered["time"].values * 1000).astype(int)
    assert time_ms[0] == 301
    assert time_ms[-1] == 899
    assert np.all(np.diff(time_ms) == 1)


def test_filter_ds_cuts_a_transient_out_and_keeps_the_longest_segment(tcv_workflow, monkeypatch):
    """A sustained power_ohm excursion over its transient threshold is a gap from the first slice
    whose centered 5 ms boxcar crosses the threshold, and the longer stretch before it is kept,
    while a single-slice spike the boxcar averages under the threshold is not a gap."""
    monkeypatch.setattr(tcv_workflow, "transient_filter", {"power_ohm": 2e6})
    power_ohm = np.full((1, N_T), 3e5)
    power_ohm[0, 500] = 5e6
    power_ohm[0, 1000:1100] = 5e6

    filtered = tcv_workflow.filter_ds(raw_shot(power_ohm=power_ohm))

    time_ms = np.round(filtered["time"].values * 1000).astype(int)
    # The boxcar at 999 holds two excursion slices, (3 x 0.3 + 2 x 5) / 5 = 2.18 MW, the first crossing
    assert time_ms[-1] == 998
    assert 500 in time_ms
