"""Tests for the shot duration culling in DataWorkflow.common_culling.

The span check (time.max - time.min) cannot see mid-shot holes, so process_fn
also requires the accumulated valid time (valid slice count times the 1 kHz
dt) to reach min_shot_duration. These cover both checks.

filter_ds drops invalid timeslices outright (`where(..., drop=True)`), so a
shot arrives at the duration checks already compacted: the holes show up as
jumps in the time coordinate, not as NaN rows. The datasets here are built the
same way.
"""

import numpy as np
import pytest
import xarray as xr

from transport_study.datasets import UNIFORM_TIMEBASE_DT_S
from transport_study.tests.datasets.synthetic_shot import RHO, SHOT, raw_shot


@pytest.fixture
def cmod_workflow(tmp_path):
    """A fresh workflow per test - the duration tests move min_shot_duration."""
    from transport_study.datasets.cmod.cmod_dataset import CModDataWorkflow

    shotlist = tmp_path / "shotlist"
    shotlist.write_text(f"{SHOT}\n")
    return CModDataWorkflow(ds_name="cmod_test", shotlist_file=shotlist, data_assembly_dir=tmp_path, gp_fit_rho=RHO)


def _filtered_ds(times: np.ndarray, valid: bool = True) -> xr.Dataset:
    """What filter_ds hands to common_culling: one row per surviving timeslice.

    Carries no Wtot_MJ, so energy_sanity_cull skips it and only the duration
    checks can cull. `valid=False` makes every data var NaN, the shape a shot
    takes when filtering leaves nothing behind.
    """
    ip = np.full((1, times.size), np.nan if not valid else 0.8)
    return xr.Dataset(
        {"Ip_MA": (("shot", "time_idx"), ip)},
        coords={"shot": [SHOT], "time_idx": np.arange(times.size), "time": ("time_idx", times)},
    )


def test_dense_shot_shorter_than_min_duration_excluded(cmod_workflow, logged):
    """A shot with contiguous valid slices whose span is below
    min_shot_duration is excluded by the span check."""
    cmod_workflow.min_shot_duration = 0.5
    ds = _filtered_ds(np.arange(300) * UNIFORM_TIMEBASE_DT_S)  # 0.299 s, no holes

    assert cmod_workflow.common_culling(SHOT, ds) is True
    assert any("duration after processing is only 0.30" in msg for msg in logged), logged


def test_sparse_shot_with_wide_span_excluded(cmod_workflow, logged):
    """A shot surviving filtering as a few isolated timeslices (3 slices spread
    over 0.28 s, like MAST shot 29447) passes the span check but is excluded by
    the accumulated valid time check."""
    cmod_workflow.min_shot_duration = 0.2
    ds = _filtered_ds(np.array([0.0, 0.14, 0.28]))

    assert cmod_workflow.common_culling(SHOT, ds) is True
    assert not any("duration after processing" in msg for msg in logged), "span check should have passed"
    assert any("0.003 seconds of valid data" in msg for msg in logged), logged


def test_dense_shot_meeting_min_duration_kept(cmod_workflow):
    """A shot with accumulated valid time at or above min_shot_duration is
    returned by process_fn rather than culled."""
    cmod_workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
    raw_shot().to_netcdf(cmod_workflow.raw_data_dir / f"{SHOT}.nc")

    processed = cmod_workflow.process_fn(SHOT)

    assert processed is not None
    assert cmod_workflow.common_culling(SHOT, processed) is False
    # 1.5 s of raw data less the end margin, all of it valid
    assert processed.sizes["time_idx"] * UNIFORM_TIMEBASE_DT_S > cmod_workflow.min_shot_duration


def test_accumulated_time_exactly_at_threshold_kept(cmod_workflow):
    """A shot whose valid slice count times UNIFORM_TIMEBASE_DT_S equals
    min_shot_duration exactly is kept (the check is a strict less-than)."""
    n_valid = 500
    cmod_workflow.min_shot_duration = n_valid * UNIFORM_TIMEBASE_DT_S
    # Every other slice dropped by filtering: 0.998 s span, 0.5 s of data
    ds = _filtered_ds(np.arange(n_valid) * 2 * UNIFORM_TIMEBASE_DT_S)

    assert cmod_workflow.common_culling(SHOT, ds) is False


def test_all_nan_shot_excluded(cmod_workflow, logged):
    """A shot whose slices are all NaN (cleaned size 0) is excluded, and
    neither duration computation reduces over an empty array."""
    cmod_workflow.min_shot_duration = 0.5
    ds = _filtered_ds(np.arange(5) * UNIFORM_TIMEBASE_DT_S, valid=False)

    assert cmod_workflow.common_culling(SHOT, ds) is True
    assert any("duration after processing is only 0.00" in msg for msg in logged), logged
