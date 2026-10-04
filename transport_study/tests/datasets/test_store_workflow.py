"""Tests for the study store build from a transport-validation-datasets store.

StoreWorkflow is what every device's study dataset is built from,
so the store it writes must hold exactly the study signals whatever else the source store carries.
These build a tiny store in the internal layout and run the real build on it.
"""

import numpy as np
import pytest
import xarray as xr
from loguru import logger
from transport_validation_datasets.store_schema import STORE_SIGNAL_ATTRS

from transport_study.datasets import workflow as workflow_module
from transport_study.datasets.workflow import StoreWorkflow
from transport_study.signals import STUDY_STORE_SIGNALS
from transport_study.tests.datasets.synthetic_store import (
    SHOT_LENGTHS,
    write_synthetic_store,
)


@pytest.fixture(scope="module")
def built_store(tmp_path_factory):
    """The source store, the study store built from it, and the workflow that built it."""
    tmp = tmp_path_factory.mktemp("internal_store")
    ds_source = write_synthetic_store(tmp / "internal.zarr")
    workflow = StoreWorkflow(ds_name="tcv_test", data_assembly_dir=tmp, source_store_path=tmp / "internal.zarr")
    workflow.run_processed_data_workflow()
    ds_built = xr.open_zarr(workflow.study_store_path).load()
    return ds_source, ds_built, workflow


def test_store_holds_exactly_the_study_signals(built_store):
    """Every study signal with its SI unit plus time, nothing the source store carries beyond them."""
    _, ds_built, _ = built_store

    assert set(ds_built.data_vars) == {*STUDY_STORE_SIGNALS, "time"}
    assert set(ds_built.dims) == {"shot", "time_idx", "rho_tor_norm"}
    for name in STUDY_STORE_SIGNALS:
        assert ds_built[name].attrs["units"] == STORE_SIGNAL_ATTRS[name]["units"], name


def test_every_shot_kept_and_padding_trimmed(built_store):
    """No culls here, the time axis is cut back to the longest shot, and no partial store is left behind."""
    _, ds_built, workflow = built_store

    assert sorted(ds_built["shot"].values.tolist()) == sorted(SHOT_LENGTHS)
    assert ds_built.sizes["time_idx"] == max(SHOT_LENGTHS.values())
    for shot, n_valid in SHOT_LENGTHS.items():
        assert int(ds_built["time"].sel(shot=shot).notnull().sum()) == n_valid
    assert not workflow.partial_store_path.exists()


def test_signals_pass_through_unchanged(built_store):
    """Signed ip and the profiles are copied as stored, the magnitudes are taken on load."""
    ds_source, ds_built, _ = built_store

    for name in ["ip", "b0", "t_e", "n_e_gradient_error"]:
        for shot, n_valid in SHOT_LENGTHS.items():
            built = ds_built[name].sel(shot=shot).isel(time_idx=slice(0, n_valid)).values
            source = ds_source[name].sel(shot=shot).isel(time_idx=slice(0, n_valid)).values
            np.testing.assert_array_equal(built, source)


def test_windowed_store_is_refused(tmp_path):
    """Only the sample fit mode is one contiguous segment per shot."""
    write_synthetic_store(tmp_path / "windowed.zarr", fit_mode="window_average")
    workflow = StoreWorkflow(ds_name="windowed", data_assembly_dir=tmp_path, source_store_path=tmp_path / "windowed.zarr")

    with pytest.raises(ValueError, match="fit_mode"):
        workflow.run_processed_data_workflow()
    assert not workflow.study_store_path.exists()


def test_plot_errors_are_logged_with_traceback(built_store, monkeypatch):
    """A failing plot does not stop the build and its traceback reaches the log."""
    _, _, workflow = built_store

    def failing_report(*args, **kwargs):
        raise RuntimeError("no pdf today")

    monkeypatch.setattr(workflow_module, "ds_summary_report", failing_report)
    records = []
    sink_id = logger.add(lambda message: records.append(message.record), level="ERROR")
    try:
        workflow.plot_store()
    finally:
        logger.remove(sink_id)

    failures = [record for record in records if "failing_report" in record["message"]]
    assert len(failures) == 1
    assert failures[0]["exception"] is not None
    assert "no pdf today" in str(failures[0]["exception"].value)
