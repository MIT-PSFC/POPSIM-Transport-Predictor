"""Tests for the parallel per-case analysis driver.

run_case_analysis_parallel must never have more than config.max_analysis_jobs
analysis jobs in the queue at once, submitting more only as earlier ones
finish, so a study with hundreds of cases does not flood the scheduler. Cases
that keep failing must fall through to the serial path rather than loop.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from transport_study.orchestration import case_analysis, slurm_utils
from transport_study.orchestration.case_analysis import (
    ANALYSIS_MAX_ATTEMPTS,
    run_case_analysis_parallel,
)
from transport_study.orchestration.slurm_utils import RELAUNCH_GRACE_S, JobEnd
from transport_study.tests.stubs import StubCase, mark_analysis_done

MAX_JOBS = 2
N_CASES = 5


@pytest.fixture(autouse=True)
def sacct_ends(monkeypatch) -> dict[int, JobEnd]:
    """sacct stand-in: no analysis job of a previous orchestrator, and the ends of the jobs a test records here.

    A job a test records no end for reads as one sacct has no record of yet.
    """
    end_states: dict[int, JobEnd] = {}
    monkeypatch.setattr(case_analysis, "recent_submitted_jobs", lambda name_prefix, name_suffix: {})
    monkeypatch.setattr(
        slurm_utils, "job_end_states", lambda job_ids: {job_id: end_states[job_id] for job_id in job_ids if job_id in end_states}
    )
    return end_states


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
def single_case_study(make_stub_study):
    """A study of one case with a result file, so it needs analysis."""
    study = make_stub_study([StubCase(name="case.0")], max_analysis_jobs=MAX_JOBS)
    result_path = study.result_path(study.cases[0])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.touch()
    return study


@pytest.fixture
def empty_queue(monkeypatch):
    """squeue reports nothing running, so every pending case is launchable."""
    monkeypatch.setattr(case_analysis, "get_running_job_names", lambda: set())


@pytest.fixture
def fake_clock(monkeypatch):
    """Virtual clock that only advances on sleep, so grace windows cost no wall time.

    It stands in for the wall clock too, which sacct job ends are compared against.
    """
    clock = SimpleNamespace(t=0.0)
    monkeypatch.setattr(case_analysis.time, "time", lambda: clock.t)
    monkeypatch.setattr(case_analysis.time, "sleep", lambda seconds: setattr(clock, "t", clock.t + seconds))
    return clock


def launch_batches_per_pass(monkeypatch, finish_immediately: bool = True) -> list[list[str]]:
    """Record submissions grouped by driver pass (a sleep starts a new group)."""
    batches: list[list[str]] = [[]]

    def fake_launch(study, case):
        batches[-1].append(str(case))
        if finish_immediately:
            mark_analysis_done(study, case)
        return len(batches[-1])

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
        return len(launched)

    def queue_whose_jobs_just_finished():
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


def test_scheduler_failure_waits_instead_of_launching(single_case_study, monkeypatch):
    """A squeue failure must not look like an empty queue and launch anyway."""
    snapshots = iter([None])  # first pass: squeue failed, then an empty queue
    monkeypatch.setattr(case_analysis, "get_running_job_names", lambda: next(snapshots, set()))

    launched: list[str] = []
    sleeps: list[float] = []

    def fake_launch(study_arg, case):
        launched.append(str(case))
        mark_analysis_done(study_arg, case)
        return 7

    monkeypatch.setattr(case_analysis, "launch_case_analysis_parallel", fake_launch)
    monkeypatch.setattr(case_analysis.time, "sleep", sleeps.append)

    run_case_analysis_parallel(single_case_study)

    assert launched == ["case.0"]
    # One sleep for the failed pass, one for the pass that launched the job
    assert len(sleeps) == 2


def test_case_that_never_finishes_stops_after_max_attempts(single_case_study, monkeypatch, empty_queue, fake_clock, sacct_ends):
    """A case whose analysis job keeps dying is relaunched only RELAUNCH_GRACE_S after the job's end,
    since its outputs may not be visible yet (NFS, 31 of 77 cases relaunched in hs1_transport_primary),
    and falls through to the serial path instead of being resubmitted forever."""
    launch_times: list[float] = []

    def launch_and_die(study_arg, case):
        launch_times.append(fake_clock.t)
        job_id = len(launch_times)
        sacct_ends[job_id] = JobEnd("FAILED", fake_clock.t)
        return job_id

    monkeypatch.setattr(case_analysis, "launch_case_analysis_parallel", launch_and_die)

    run_case_analysis_parallel(single_case_study)

    assert len(launch_times) == ANALYSIS_MAX_ATTEMPTS
    relaunch_gaps = np.diff(launch_times)
    assert (relaunch_gaps >= RELAUNCH_GRACE_S).all()


def test_rejected_submission_counts_no_attempt(single_case_study, monkeypatch, empty_queue, fake_clock):
    """sbatch rejecting the job (SLURM busy) must not use up the case's attempts and push it to the serial path."""
    submitted_job_ids = iter([None] * ANALYSIS_MAX_ATTEMPTS + [7])
    launched: list[str] = []

    def flaky_launch(study_arg, case):
        job_id = next(submitted_job_ids)
        if job_id is not None:
            launched.append(str(case))
            mark_analysis_done(study_arg, case)
        return job_id

    monkeypatch.setattr(case_analysis, "launch_case_analysis_parallel", flaky_launch)

    run_case_analysis_parallel(single_case_study)

    assert launched == ["case.0"]


def test_job_ending_late_in_a_long_pass_is_held_from_its_end(single_case_study, monkeypatch, fake_clock, sacct_ends):
    """The hold runs from the job's sacct end, not from the pass that last saw the job in the queue.

    The done check loads every pending case, so a pass can outlast RELAUNCH_GRACE_S.
    Timed from the pass that saw it, a job ending late in such a pass was relaunched before its outputs were visible.
    """
    case = single_case_study.cases[0]
    # Pass 1 (t = 0) launches, pass 2 (t = 400) sees the job running, it ends at t = 750 and is gone at pass 3 (t = 800)
    snapshots = iter([set(), {single_case_study.analysis_job_name(case)}])
    monkeypatch.setattr(case_analysis, "get_running_job_names", lambda: next(snapshots, set()))
    sacct_ends[7] = JobEnd("COMPLETED", 750.0)
    launched: list[str] = []

    def launch(study_arg, case_arg):
        launched.append(str(case_arg))
        return 7

    def long_pass(seconds):
        fake_clock.t += 400.0
        # The outputs become visible on this node only after pass 3
        if fake_clock.t >= 1200.0:
            mark_analysis_done(single_case_study, case)

    monkeypatch.setattr(case_analysis, "launch_case_analysis_parallel", launch)
    monkeypatch.setattr(case_analysis.time, "sleep", long_pass)

    run_case_analysis_parallel(single_case_study)

    assert launched == ["case.0"]
