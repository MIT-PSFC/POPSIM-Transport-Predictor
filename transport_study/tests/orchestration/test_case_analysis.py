"""Tests for the parallel per-case analysis driver.

run_case_analysis_parallel must never have more than config.max_analysis_jobs
analysis jobs in the queue at once, submitting more only as earlier ones
finish, so a study with hundreds of cases does not flood the scheduler. Cases
that keep failing must fall through to the serial path rather than loop.
"""

import pytest

from transport_study.orchestration import case_analysis
from transport_study.orchestration.case_analysis import (
    ANALYSIS_MAX_ATTEMPTS,
    run_case_analysis_parallel,
)
from transport_study.tests.stubs import StubCase, mark_analysis_done

MAX_JOBS = 2
N_CASES = 5


@pytest.fixture
def study(make_stub_study):
    """A study whose every case has a result file, so all need analysis."""
    study = make_stub_study([StubCase(name=f"case.{i}") for i in range(N_CASES)], max_analysis_jobs=MAX_JOBS)
    for case in study.cases:
        result_path = study.result_path(case)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.touch()
    return study


@pytest.fixture
def empty_queue(monkeypatch):
    """squeue reports nothing running, so every pending case is launchable."""
    monkeypatch.setattr(case_analysis, "get_running_job_names", lambda partition=None: set())


def launch_batches_per_pass(monkeypatch, finish_immediately: bool = True) -> list[list[str]]:
    """Record submissions grouped by driver pass (a sleep starts a new group)."""
    batches: list[list[str]] = [[]]

    def fake_launch(study, case):
        batches[-1].append(str(case))
        if finish_immediately:
            mark_analysis_done(study, case)

    monkeypatch.setattr(case_analysis, "launch_case_analysis_parallel", fake_launch)
    monkeypatch.setattr(case_analysis.time, "sleep", lambda seconds: batches.append([]))
    return batches


def test_launches_every_case_capped_at_max_analysis_jobs(study, monkeypatch, empty_queue):
    batches = launch_batches_per_pass(monkeypatch)

    run_case_analysis_parallel(study)

    launched = [case for batch in batches for case in batch]
    assert sorted(launched) == sorted(str(case) for case in study.cases), "every case must be analyzed exactly once"
    assert max(len(batch) for batch in batches) <= MAX_JOBS, "per-pass submissions must respect the cap"


def test_job_finishing_before_the_queue_query_is_not_relaunched(study, monkeypatch):
    """A job that leaves the queue after its case was last checked must count as done, not run again (hs1_pb_primary, 2026-10-06)."""
    launched: list[str] = []
    queue = {}

    def launch_into_queue(study_arg, case):
        launched.append(str(case))
        queue[study_arg.analysis_job_name(case)] = case

    def queue_whose_jobs_just_finished(partition=None):
        """Every queued job finishes right before squeue reports, after the previous pass checked its case."""
        for case in queue.values():
            mark_analysis_done(study, case)
        queue.clear()
        return set()

    monkeypatch.setattr(case_analysis, "launch_case_analysis_parallel", launch_into_queue)
    monkeypatch.setattr(case_analysis, "get_running_job_names", queue_whose_jobs_just_finished)
    monkeypatch.setattr(case_analysis.time, "sleep", lambda seconds: None)

    run_case_analysis_parallel(study)

    assert sorted(launched) == sorted(str(case) for case in study.cases), "every case must be analyzed exactly once"


def test_no_pending_cases_submits_nothing(study, monkeypatch, empty_queue):
    batches = launch_batches_per_pass(monkeypatch)
    for case in study.cases:
        mark_analysis_done(study, case)

    run_case_analysis_parallel(study)

    assert batches == [[]]


def test_scheduler_failure_waits_instead_of_launching(make_stub_study, monkeypatch):
    """A squeue failure must not look like an empty queue and launch anyway."""
    study = make_stub_study([StubCase(name="case.0")], max_analysis_jobs=MAX_JOBS)
    result_path = study.result_path(study.cases[0])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.touch()

    snapshots = iter([None])  # first pass: squeue failed, then an empty queue
    monkeypatch.setattr(case_analysis, "get_running_job_names", lambda partition=None: next(snapshots, set()))

    launched: list[str] = []
    sleeps: list[float] = []

    def fake_launch(study_arg, case):
        launched.append(str(case))
        mark_analysis_done(study_arg, case)

    monkeypatch.setattr(case_analysis, "launch_case_analysis_parallel", fake_launch)
    monkeypatch.setattr(case_analysis.time, "sleep", sleeps.append)

    run_case_analysis_parallel(study)

    assert launched == ["case.0"]
    # One sleep for the failed pass, one for the pass that launched the job
    assert len(sleeps) == 2


def test_case_that_never_finishes_stops_after_max_attempts(make_stub_study, monkeypatch, empty_queue):
    """A case whose analysis job keeps dying falls through to the serial path
    instead of being resubmitted forever."""
    study = make_stub_study([StubCase(name="case.0")], max_analysis_jobs=MAX_JOBS)
    result_path = study.result_path(study.cases[0])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.touch()

    launched: list[str] = []
    monkeypatch.setattr(case_analysis, "launch_case_analysis_parallel", lambda study_arg, case: launched.append(str(case)))
    monkeypatch.setattr(case_analysis.time, "sleep", lambda seconds: None)

    run_case_analysis_parallel(study)

    assert launched == ["case.0"] * ANALYSIS_MAX_ATTEMPTS
