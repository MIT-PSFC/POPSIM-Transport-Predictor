"""collect_results reads the result files on a process pool, and run_study runs it in a job of its own."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from transport_study.orchestration import slurm_utils
from transport_study.orchestration import study as study_module
from transport_study.orchestration.slurm_utils import RELAUNCH_GRACE_S, JobEnd
from transport_study.orchestration.study import write_netcdf_atomic
from transport_study.tests.orchestration.test_padded_tail_masking import padded_result
from transport_study.tests.stubs import StubCase


def test_collect_results_reads_many_cases_on_processes_in_case_order(make_stub_study):
    """Two result files go to spawned processes, and each summary comes back on its own case."""
    cases = [StubCase(name="case.a"), StubCase(name="case.b")]
    study = make_stub_study(cases)
    time_real = 1e-3 * np.arange(3)
    for case, error_scale in zip(cases, (1.0, 10.0), strict=True):
        errors_real = error_scale * np.array([1.0, 2.0, 3.0])
        write_netcdf_atomic(padded_result(errors_real, time_real, shot=101), study.result_path(case))

    collected = study.collect_results()

    assert collected["err_abs_ts_mean"].values.tolist() == pytest.approx([2.0, 20.0])


@pytest.fixture
def fake_clock(monkeypatch):
    """Virtual clock that only advances on sleep, standing in for the wall clock of sacct job ends too."""
    clock = SimpleNamespace(t=0.0)
    monkeypatch.setattr(study_module.time, "monotonic", lambda: clock.t)
    monkeypatch.setattr(study_module.time, "time", lambda: clock.t)
    monkeypatch.setattr(study_module.time, "sleep", lambda s: setattr(clock, "t", clock.t + s))
    return clock


def _collection_study(make_stub_study, monkeypatch):
    """A one-case study whose in-process collection is recorded, the collection job accepted as job 7."""
    study = make_stub_study([StubCase(name="case.a")], partition="fake_partition")
    in_process_collections: list[float] = []

    def collect_here():
        in_process_collections.append(study_module.time.time())
        return xr.Dataset({"err_abs_ts_mean": ("case_idx", [1.0])})

    monkeypatch.setattr(study, "collect_results", collect_here)
    monkeypatch.setattr(study_module, "launch_collect_results_job", lambda study_arg, cpus: 7)
    monkeypatch.setattr(study_module, "get_running_job_names", lambda: set())
    return study, in_process_collections


def test_collection_job_writing_its_file_is_not_repeated_here(make_stub_study, monkeypatch, fake_clock):
    study, in_process_collections = _collection_study(make_stub_study, monkeypatch)

    def job_writes_its_file(job_ids):
        write_netcdf_atomic(xr.Dataset({"err_abs_ts_mean": ("case_idx", [1.0])}), study.collected_results_path())
        return {7: JobEnd("RUNNING", None)}

    monkeypatch.setattr(slurm_utils, "job_end_states", job_writes_its_file)

    study.write_collected_results(enable_parallelism=True)

    assert in_process_collections == []
    assert study.collected_results_path().exists()


def test_collection_job_ending_without_its_file_falls_back_after_the_grace(make_stub_study, monkeypatch, fake_clock):
    study, in_process_collections = _collection_study(make_stub_study, monkeypatch)
    monkeypatch.setattr(slurm_utils, "job_end_states", lambda job_ids: {7: JobEnd("FAILED", 0.0)})

    study.write_collected_results(enable_parallelism=True)

    assert len(in_process_collections) == 1
    assert in_process_collections[0] >= RELAUNCH_GRACE_S
    assert study.collected_results_path().exists()
