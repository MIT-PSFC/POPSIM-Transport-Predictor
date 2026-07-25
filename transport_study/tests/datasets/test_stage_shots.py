"""Tests for DataWorkflow._stage_shots, the source-staging phase.

Staging is the slow, network-bound half of a distributed dataset build (10.5
hours for 844 MAST shots serially), so it runs threaded.
Tests ensure the threaded selection matches what the serial loop would have picked
(same shots, same order, nothing staged beyond what was asked for)
and that one bad shot cannot take down a run that is already hours in.
"""

import threading
import time

import pytest
import xarray as xr

from transport_study.datasets.workflow import DataWorkflow


class _CountingWorkflow(DataWorkflow):
    """Workflow whose prepare_shot is instrumented instead of hitting a device.

    Records call order and peak concurrency; `bad_shots` return None (invalid
    source data), `raising_shots` raise.
    """

    def __init__(self, *args, bad_shots=(), raising_shots=(), delay=0.0, **kwargs):
        self.bad_shots = set(bad_shots)
        self.raising_shots = set(raising_shots)
        self.delay = delay
        self.calls: list[int] = []
        self.peak_concurrency = 0
        self._in_flight = 0
        self._lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def prepare_shot(self, shot: int):
        with self._lock:
            self.calls.append(shot)
            self._in_flight += 1
            self.peak_concurrency = max(self.peak_concurrency, self._in_flight)
        try:
            time.sleep(self.delay)
            if shot in self.raising_shots:
                raise RuntimeError(f"source read blew up for {shot}")
            return None if shot in self.bad_shots else f"fit_input_{shot}"
        finally:
            with self._lock:
                self._in_flight -= 1

    def _get_shotlist_from_source(self) -> list[int]:
        raise NotImplementedError

    def make_raw_data_files(self):
        raise NotImplementedError

    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset:
        raise NotImplementedError

    def device_specific_processing(self, ds: xr.Dataset) -> xr.Dataset:
        raise NotImplementedError


@pytest.fixture
def make_workflow(tmp_path):
    def _make(shots, **kwargs):
        shotlist = tmp_path / "shotlist"
        shotlist.write_text("\n".join(str(s) for s in shots) + "\n")
        workflow = _CountingWorkflow(
            ds_name="fake",
            shotlist_file=shotlist,
            data_assembly_dir=tmp_path,
            **kwargs,
        )
        workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
        return workflow

    return _make


def test_stage_shots_threaded_matches_serial_selection(make_workflow):
    shots = list(range(10))
    serial = make_workflow(shots, prepare_workers=1)
    threaded = make_workflow(shots, prepare_workers=4)

    assert serial._stage_shots(len(shots)) == threaded._stage_shots(len(shots))
    assert sorted(threaded.calls) == shots
    assert serial.peak_concurrency == 1


def test_stage_shots_runs_concurrently(make_workflow):
    """Staging is latency bound, so the threads must genuinely overlap."""
    workflow = make_workflow(list(range(8)), prepare_workers=4, delay=0.05)

    workflow._stage_shots(8)

    assert workflow.peak_concurrency > 1


def test_stage_shots_skips_shots_that_already_have_raw_files(make_workflow):
    workflow = make_workflow([1, 2, 3, 4], prepare_workers=2)
    for shot in (1, 3):
        (workflow.raw_data_dir / f"{shot}.nc").write_bytes(b"")

    n_existing, pending = workflow._stage_shots(4)

    assert n_existing == 2
    assert set(pending) == {2, 4}
    assert 1 not in workflow.calls and 3 not in workflow.calls


def test_stage_shots_stops_at_target_without_over_staging(make_workflow):
    """max_num_shots caps the run: nothing past the target may be downloaded,
    even with several threads in flight."""
    workflow = make_workflow(list(range(20)), prepare_workers=4)

    n_existing, pending = workflow._stage_shots(6)

    assert n_existing == 0
    assert list(pending) == [0, 1, 2, 3, 4, 5]
    assert len(workflow.calls) == 6


def test_stage_shots_counts_existing_files_toward_target(make_workflow):
    workflow = make_workflow(list(range(20)), prepare_workers=4)
    for shot in (0, 1):
        (workflow.raw_data_dir / f"{shot}.nc").write_bytes(b"")

    n_existing, pending = workflow._stage_shots(5)

    assert n_existing == 2
    assert list(pending) == [2, 3, 4]


def test_stage_shots_keeps_going_past_invalid_shots(make_workflow):
    """A shot with no usable source data does not count toward the target, so
    the loop has to reach further down the shotlist to fill it."""
    workflow = make_workflow(list(range(10)), prepare_workers=3, bad_shots={1, 2, 5})

    n_existing, pending = workflow._stage_shots(4)

    assert n_existing == 0
    assert list(pending) == [0, 3, 4, 6]


def test_stage_shots_survives_a_raising_shot(make_workflow):
    """One shot blowing up must not kill a staging run that may be hours in."""
    workflow = make_workflow(list(range(5)), prepare_workers=2, raising_shots={2})

    n_existing, pending = workflow._stage_shots(5)

    assert n_existing == 0
    assert list(pending) == [0, 1, 3, 4]


def test_stage_shots_exhausted_shotlist_returns_what_it_has(make_workflow):
    """Asking for more shots than the shotlist can supply terminates."""
    workflow = make_workflow([1, 2, 3], prepare_workers=2, bad_shots={2})

    n_existing, pending = workflow._stage_shots(10)

    assert n_existing == 0
    assert list(pending) == [1, 3]
