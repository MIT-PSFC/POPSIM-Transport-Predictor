"""Tests for the orchestration loop's case pickup logic.

The loop must only pick up cases that are able to run: prereq results on disk
and no training job for the case already in flight. Blocked cases wait without
being visited (which previously spammed the logs every pass).
"""

from types import SimpleNamespace

import pytest
from popsim.ml import TrainConfig

from transport_study.orchestration import study as study_module
from transport_study.orchestration.study import Study
from transport_study.tests.stubs import StubCase


@pytest.fixture
def fake_clock(monkeypatch):
    """Virtual clock that only advances on sleep, so grace windows cost no wall time."""
    clock = SimpleNamespace(t=0.0)
    monkeypatch.setattr(study_module.time, "monotonic", lambda: clock.t)
    monkeypatch.setattr(study_module.time, "sleep", lambda s: setattr(clock, "t", clock.t + s))
    return clock


@pytest.fixture
def quiet_scheduler(monkeypatch):
    """Neutral SLURM stand-ins: no elapsed jobs, a partition always available."""
    monkeypatch.setattr(study_module, "get_running_job_elapsed_s", lambda partition=None: {})
    monkeypatch.setattr(study_module, "pick_partition", lambda *args, **kwargs: "test_partition")


def set_queue_snapshots(monkeypatch, snapshots):
    """Serve one squeue snapshot per pass, then an empty queue forever."""
    remaining = iter(snapshots)
    monkeypatch.setattr(study_module, "get_running_job_names", lambda partition=None: next(remaining, set()))


def record_and_finish_cases(monkeypatch, study: Study, order: list[str]):
    """Replace run_case with a stand-in that records pickup order and finishes the case."""

    def run_case(case, skip_tuning, enable_parallelism):
        order.append(str(case))
        result_path = study.result_path(case)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.touch()

    monkeypatch.setattr(study, "run_case", run_case)


def test_blocked_case_waits_for_prereq(make_stub_study, monkeypatch):
    """A case with an unmet prereq is not picked up until its prereq finishes."""
    prereq = StubCase(name="case.prereq")
    dependent = StubCase(name="case.dependent", prereqs=[prereq])
    study = make_stub_study([dependent, prereq])
    monkeypatch.setattr(study_module.time, "sleep", lambda s: None)

    order: list[str] = []
    record_and_finish_cases(monkeypatch, study, order)

    study.run_unfinished_cases(skip_tuning=True, enable_parallelism=False)

    assert order == ["case.prereq", "case.dependent"]


def test_in_flight_case_not_picked_up(make_stub_study, monkeypatch, fake_clock, quiet_scheduler):
    """With parallelism, a case whose training job is queued or running is skipped."""
    case_a = StubCase(name="case.a")
    case_b = StubCase(name="case.b")
    study = make_stub_study([case_a, case_b])
    set_queue_snapshots(monkeypatch, [{study.train_job_name(case_a)}])

    order: list[str] = []
    record_and_finish_cases(monkeypatch, study, order)

    study.run_unfinished_cases(skip_tuning=True, enable_parallelism=True)

    assert order == ["case.b", "case.a"]


def test_result_lag_does_not_relaunch(make_stub_study, monkeypatch, fake_clock, quiet_scheduler):
    """A case whose job left the queue is not relaunched while its result file is
    still propagating across nodes: the loop holds it in the grace window and
    drops it once the result becomes visible."""
    case_a = StubCase(name="case.a")
    study = make_stub_study([case_a])

    def snapshots():
        yield {study.train_job_name(case_a)}  # pass 1: job in flight
        yield set()  # pass 2: job gone, result not yet visible (filesystem lag)
        result_path = study.result_path(case_a)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.touch()
        while True:  # pass 3 onward: the result has landed
            yield set()

    snapshot_iter = snapshots()
    monkeypatch.setattr(study_module, "get_running_job_names", lambda partition=None: next(snapshot_iter))

    order: list[str] = []
    record_and_finish_cases(monkeypatch, study, order)

    study.run_unfinished_cases(skip_tuning=True, enable_parallelism=True)

    assert order == []


def test_relaunch_after_grace_when_no_result(make_stub_study, monkeypatch, fake_clock, quiet_scheduler):
    """A case whose job exits without ever producing a result (e.g. a wall-budget
    stop) is relaunched, but only after the grace window."""
    case_a = StubCase(name="case.a")
    study = make_stub_study([case_a])
    set_queue_snapshots(monkeypatch, [{study.train_job_name(case_a)}])

    launch_times: list[float] = []
    order: list[str] = []
    record_and_finish_cases(monkeypatch, study, order)
    finish_case = study.run_case

    def run_case(case, skip_tuning, enable_parallelism):
        launch_times.append(fake_clock.t)
        finish_case(case, skip_tuning=skip_tuning, enable_parallelism=enable_parallelism)

    monkeypatch.setattr(study, "run_case", run_case)

    study.run_unfinished_cases(skip_tuning=True, enable_parallelism=True)

    assert len(launch_times) == 1
    assert launch_times[0] >= study_module.RELAUNCH_GRACE_S


def test_scheduler_failure_launches_nothing(make_stub_study, monkeypatch, quiet_scheduler):
    """When squeue fails the loop must hold off launching rather than assume no jobs."""
    case_a = StubCase(name="case.a")
    study = make_stub_study([case_a])
    set_queue_snapshots(monkeypatch, [None])  # first pass: squeue failed

    order: list[str] = []
    record_and_finish_cases(monkeypatch, study, order)
    sleeps: list[float] = []
    monkeypatch.setattr(study_module.time, "sleep", sleeps.append)

    study.run_unfinished_cases(skip_tuning=True, enable_parallelism=True)

    assert order == ["case.a"]
    # One sleep for the failed pass, one for the pass that launched the case
    assert len(sleeps) == 2


def test_case_in_flight_ignores_agent_jobs(make_stub_study):
    """Sweep agent jobs must not make a case look in flight: a tuning case with
    agents running may still need more agents launched."""
    case_a = StubCase(name="case.a")
    study = make_stub_study([case_a])

    assert not study.case_in_flight(case_a, {study.agent_job_name(case_a)})
    assert study.case_in_flight(case_a, {study.train_job_name(case_a)})


@pytest.mark.parametrize(("submitted", "expected_attempts"), [(False, 0), (True, 1)])
def test_rejected_submission_refunds_train_attempt(make_stub_study, monkeypatch, submitted, expected_attempts):
    """A training job SLURM never accepted must not count toward MAX_TRAIN_ATTEMPTS."""
    case = StubCase("case.a")
    study = make_stub_study([case])
    train_config = TrainConfig(
        project="stub_project",
        train_run_builder="stub.module.StubTRB",
        max_epochs=1,
        epochs_per_val=1,
        dataloader_config={},
        model_init_config={},
        loss_config={},
        optimizer_config={},
    )
    monkeypatch.setattr(study, "make_train_config", lambda _case: train_config)
    monkeypatch.setattr(study_module, "launch_train_parallel", lambda *args, **kwargs: submitted)

    study.launch_train(case, enable_parallelism=True, partition="test_partition")

    assert study.train_attempts.get(str(case), 0) == expected_attempts
