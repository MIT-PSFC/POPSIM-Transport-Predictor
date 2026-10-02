"""Tests for the shot duration culling in RawFileWorkflow.is_too_short.

filter_ds keeps one contiguous segment of a shot, so the duration is its span (kept_span),
first to last kept slice, against min_pulse_length_s.
"""

import numpy as np
import pytest
import xarray as xr

from transport_study.datasets import UNIFORM_TIMEBASE_DT_S
from transport_study.signals import PREDICTION_STORE_NAME
from transport_study.tests.datasets.synthetic_shot import SHOT, raw_shot


@pytest.fixture
def tcv_workflow(tmp_path):
    """A fresh workflow per test - the duration tests move min_pulse_length_s."""
    from transport_study.datasets.tcv.tcv_dataset import TCVDataWorkflow

    shotlist = tmp_path / "shotlist"
    shotlist.write_text(f"{SHOT}\n")
    return TCVDataWorkflow(ds_name="tcv_test", shotlist_file=shotlist, data_assembly_dir=tmp_path)


def _filtered_ds(times: np.ndarray) -> xr.Dataset:
    """What filter_ds hands to the culls: one row per kept timeslice."""
    ip = np.full((1, times.size), 3e5)
    return xr.Dataset(
        {"ip": (("shot", "time_idx"), ip)},
        coords={"shot": [SHOT], "time_idx": np.arange(times.size), "time": ("time_idx", times)},
    )


def test_segment_shorter_than_min_pulse_length_excluded(tcv_workflow, logged):
    """A kept segment spanning less than min_pulse_length_s is excluded."""
    tcv_workflow.min_pulse_length_s = 0.5
    ds = _filtered_ds(np.arange(300) * UNIFORM_TIMEBASE_DT_S)  # 0.299 s

    assert tcv_workflow.is_too_short(ds) is True
    assert any("kept segment 0.299 s" in msg for msg in logged), logged


def test_segment_exactly_at_min_pulse_length_kept(tcv_workflow):
    """A segment spanning min_pulse_length_s exactly is kept (the check is a strict less-than)."""
    n_valid = 501
    tcv_workflow.min_pulse_length_s = (n_valid - 1) * UNIFORM_TIMEBASE_DT_S
    ds = _filtered_ds(np.arange(n_valid) * UNIFORM_TIMEBASE_DT_S)

    assert tcv_workflow.is_too_short(ds) is False


def test_full_shot_kept_by_process_fn(tcv_workflow):
    """The synthetic shot, 1.5 s less the end margin, is returned by process_fn rather than culled."""
    tcv_workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
    raw_shot().to_netcdf(tcv_workflow.raw_data_dir / f"{SHOT}.nc")

    processed = tcv_workflow.process_fn(SHOT, tcv_workflow.STORE_VARIABLES[PREDICTION_STORE_NAME])

    assert processed is not None
    assert tcv_workflow.is_too_short(processed) is False
