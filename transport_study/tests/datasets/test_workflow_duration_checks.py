"""Tests for the shot duration culling in RawFileWorkflow.is_too_short.

The span check (time.max - time.min) cannot see mid-shot holes, so the
accumulated valid time (valid slice count times the 1 kHz dt) must also
reach min_shot_duration. These cover both checks.

filter_ds drops invalid timeslices outright (`where(..., drop=True)`), so a
shot arrives at the duration checks already compacted: the holes show up as
jumps in the time coordinate, not as NaN rows. The datasets here are built the
same way.
"""

import numpy as np
import pytest
import xarray as xr

from transport_study.datasets import UNIFORM_TIMEBASE_DT_S
from transport_study.signals import PREDICTION_STORE_NAME
from transport_study.tests.datasets.synthetic_shot import SHOT, raw_shot


@pytest.fixture
def tcv_workflow(tmp_path):
    """A fresh workflow per test - the duration tests move min_shot_duration."""
    from transport_study.datasets.tcv.tcv_dataset import TCVDataWorkflow

    shotlist = tmp_path / "shotlist"
    shotlist.write_text(f"{SHOT}\n")
    return TCVDataWorkflow(ds_name="tcv_test", shotlist_file=shotlist, data_assembly_dir=tmp_path)


def _filtered_ds(times: np.ndarray, valid: bool = True) -> xr.Dataset:
    """What filter_ds hands to the culls: one row per surviving timeslice.

    `valid=False` makes every data var NaN, the shape a shot
    takes when filtering leaves nothing behind.
    """
    ip = np.full((1, times.size), np.nan if not valid else 3e5)
    return xr.Dataset(
        {"ip": (("shot", "time_idx"), ip)},
        coords={"shot": [SHOT], "time_idx": np.arange(times.size), "time": ("time_idx", times)},
    )


def test_dense_shot_shorter_than_min_duration_excluded(tcv_workflow, logged):
    """A shot with contiguous valid slices whose span is below
    min_shot_duration is excluded by the span check."""
    tcv_workflow.min_shot_duration = 0.5
    ds = _filtered_ds(np.arange(300) * UNIFORM_TIMEBASE_DT_S)  # 0.299 s, no holes

    assert tcv_workflow.is_too_short(ds) is True
    assert any("duration after processing is only 0.30" in msg for msg in logged), logged


def test_sparse_shot_with_wide_span_excluded(tcv_workflow, logged):
    """A shot surviving filtering as a few isolated timeslices (3 slices spread
    over 0.28 s, like MAST shot 29447) passes the span check but is excluded by
    the accumulated valid time check."""
    tcv_workflow.min_shot_duration = 0.2
    ds = _filtered_ds(np.array([0.0, 0.14, 0.28]))

    assert tcv_workflow.is_too_short(ds) is True
    assert not any("duration after processing" in msg for msg in logged), "span check should have passed"
    assert any("0.003 seconds of valid data" in msg for msg in logged), logged


def test_dense_shot_meeting_min_duration_kept(tcv_workflow):
    """A shot with accumulated valid time at or above min_shot_duration is
    returned by process_fn rather than culled."""
    tcv_workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
    raw_shot().to_netcdf(tcv_workflow.raw_data_dir / f"{SHOT}.nc")

    processed = tcv_workflow.process_fn(SHOT, tcv_workflow.STORE_VARIABLES[PREDICTION_STORE_NAME])

    assert processed is not None
    assert tcv_workflow.is_too_short(processed) is False
    # 1.5 s of raw data less the end margin, all of it valid
    assert processed.sizes["time_idx"] * UNIFORM_TIMEBASE_DT_S > tcv_workflow.min_shot_duration


def test_accumulated_time_exactly_at_threshold_kept(tcv_workflow):
    """A shot whose valid slice count times UNIFORM_TIMEBASE_DT_S equals
    min_shot_duration exactly is kept (the check is a strict less-than)."""
    n_valid = 500
    tcv_workflow.min_shot_duration = n_valid * UNIFORM_TIMEBASE_DT_S
    # Every other slice dropped by filtering: 0.998 s span, 0.5 s of data
    ds = _filtered_ds(np.arange(n_valid) * 2 * UNIFORM_TIMEBASE_DT_S)

    assert tcv_workflow.is_too_short(ds) is False


def test_all_nan_shot_excluded(tcv_workflow, logged):
    """A shot whose slices are all NaN (cleaned size 0) is excluded, and
    neither duration computation reduces over an empty array."""
    tcv_workflow.min_shot_duration = 0.5
    ds = _filtered_ds(np.arange(5) * UNIFORM_TIMEBASE_DT_S, valid=False)

    assert tcv_workflow.is_too_short(ds) is True
    assert any("duration after processing is only 0.00" in msg for msg in logged), logged
