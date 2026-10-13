"""Tests for the orchestration loop's case pickup logic.

The loop must only pick up cases that are able to run: prereq results on disk
and no training job for the case already in flight. Blocked cases wait without
being visited (which previously spammed the logs every pass).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import requests
import xarray as xr
from popsim.ml import TrainConfig

from transport_study.config import config
from transport_study.orchestration import slurm_utils
from transport_study.orchestration import study as study_module
from transport_study.orchestration.seed_results import BEST_SEED_ATTR
from transport_study.orchestration.slurm_utils import (
    RELAUNCH_GRACE_S,
    JobEnd,
    SubmittedJob,
)
from transport_study.orchestration.study import Study, write_netcdf_atomic
from transport_study.tests.stubs import StubCase, with_seed_attrs


@pytest.fixture
def fake_clock(monkeypatch):
    """Virtual clock that only advances on sleep, so grace windows cost no wall time.

    It stands in for the wall clock too, which sacct job ends are compared against.
    """
    clock = SimpleNamespace(t=0.0)
    monkeypatch.setattr(study_module.time, "monotonic", lambda: clock.t)
    monkeypatch.setattr(study_module.time, "time", lambda: clock.t)
    monkeypatch.setattr(study_module.time, "sleep", lambda s: setattr(clock, "t", clock.t + s))
    return clock


@pytest.fixture
def quiet_scheduler(monkeypatch):
    """Neutral SLURM stand-ins: no elapsed jobs, a partition always available, no job sacct knows of."""
    monkeypatch.setattr(study_module, "get_running_job_elapsed_s", lambda partition=None: {})
    monkeypatch.setattr(study_module, "pick_partition", lambda *args, **kwargs: "test_partition")
    monkeypatch.setattr(study_module, "recent_submitted_jobs", lambda name_prefix, name_suffix: {})
    monkeypatch.setattr(slurm_utils, "job_end_states", lambda job_ids: {})


# The train config every stubbed make_train_config returns
STUB_TRAIN_CONFIG = TrainConfig(
    project="stub_project",
    train_run_builder="stub.module.StubTRB",
    max_epochs=1,
    epochs_per_val=1,
    dataloader_config={},
    model_init_config={},
    loss_config={},
    optimizer_config={},
)


def set_queue_snapshots(monkeypatch, snapshots):
    """Serve one squeue snapshot per pass, then an empty queue forever."""
    remaining = iter(snapshots)
    monkeypatch.setattr(study_module, "get_running_job_names", lambda: next(remaining, set()))


def record_and_finish_cases(monkeypatch, study: Study, order: list[str]):
    """Replace run_case with a stand-in that records pickup order and finishes the case."""

    def run_case(case, seed, skip_tuning, enable_parallelism, partition=None):
        order.append(str(case))
        result_path = study.result_path(case)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.touch()
        return True

    monkeypatch.setattr(study, "run_case", run_case)


def test_blocked_case_waits_for_prereq(make_stub_study, monkeypatch):
    """A case with an unmet prereq is not picked up until its prereq finishes.
    A serial pass that finished a case starts the next one at once, nothing runs in the background to wait for."""
    prereq = StubCase(name="case.prereq")
    dependent = StubCase(name="case.dependent", prereqs=[prereq])
    study = make_stub_study([dependent, prereq])
    sleeps: list[float] = []
    monkeypatch.setattr(study_module.time, "sleep", sleeps.append)

    order: list[str] = []
    record_and_finish_cases(monkeypatch, study, order)

    study.run_unfinished_cases(skip_tuning=True, enable_parallelism=False)

    assert order == ["case.prereq", "case.dependent"]
    assert sleeps == []


def test_in_flight_case_not_picked_up(make_stub_study, monkeypatch, fake_clock, quiet_scheduler):
    """With parallelism, a case whose training job is queued or running is skipped."""
    case_a = StubCase(name="case.a")
    case_b = StubCase(name="case.b")
    study = make_stub_study([case_a, case_b])
    set_queue_snapshots(monkeypatch, [{study.train_job_name(case_a, 0)}])

    order: list[str] = []
    record_and_finish_cases(monkeypatch, study, order)

    study.run_unfinished_cases(skip_tuning=True, enable_parallelism=True)

    assert order == ["case.b", "case.a"]


def test_result_lag_does_not_relaunch(make_stub_study, monkeypatch, fake_clock, quiet_scheduler):
    """A seed whose job left the queue is not relaunched while its result file is still propagating across nodes.

    A restarted orchestrator recalls the job its predecessor submitted,
    holds the seed until RELAUNCH_GRACE_S past the job's end, and drops the case once its result becomes visible.
    """
    case_a = StubCase(name="case.a")
    study = make_stub_study([case_a])
    job_name = study.train_job_name(case_a, 0)
    monkeypatch.setattr(study_module, "recent_submitted_jobs", lambda name_prefix, name_suffix: {job_name: SubmittedJob(7, 0.0)})
    monkeypatch.setattr(slurm_utils, "job_end_states", lambda job_ids: {7: JobEnd("COMPLETED", 0.0)})

    def snapshots():
        yield {job_name}  # pass 1: job in flight
        yield set()  # pass 2: job ended at t = 0, result not yet visible (filesystem lag)
        result_path = study.result_path(case_a)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.touch()
        while True:  # pass 3 onward: the result has landed
            yield set()

    snapshot_iter = snapshots()
    monkeypatch.setattr(study_module, "get_running_job_names", lambda: next(snapshot_iter))

    order: list[str] = []
    record_and_finish_cases(monkeypatch, study, order)

    study.run_unfinished_cases(skip_tuning=True, enable_parallelism=True)

    assert order == []


def test_relaunch_after_grace_when_no_result(make_stub_study, monkeypatch, fake_clock, quiet_scheduler):
    """A seed whose job exits without ever producing a result (e.g. a wall-budget
    stop) is relaunched, but only RELAUNCH_GRACE_S after the job's end."""
    case_a = StubCase(name="case.a")
    study = make_stub_study([case_a])
    job_name = study.train_job_name(case_a, 0)
    monkeypatch.setattr(study_module, "recent_submitted_jobs", lambda name_prefix, name_suffix: {job_name: SubmittedJob(7, 0.0)})
    monkeypatch.setattr(slurm_utils, "job_end_states", lambda job_ids: {7: JobEnd("FAILED", 0.0)})
    set_queue_snapshots(monkeypatch, [{job_name}])

    launch_times: list[float] = []
    order: list[str] = []
    record_and_finish_cases(monkeypatch, study, order)
    finish_case = study.run_case

    def run_case(case, seed, skip_tuning, enable_parallelism, partition=None):
        launch_times.append(fake_clock.t)
        return finish_case(case, seed, skip_tuning=skip_tuning, enable_parallelism=enable_parallelism)

    monkeypatch.setattr(study, "run_case", run_case)

    study.run_unfinished_cases(skip_tuning=True, enable_parallelism=True)

    assert len(launch_times) == 1
    assert launch_times[0] >= RELAUNCH_GRACE_S


def test_tuning_case_reads_its_sweep_once_per_poll_interval(make_stub_study, monkeypatch, fake_clock, quiet_scheduler):
    """A case still tuning reads its wandb project once per config.sweep_poll_interval_s, not every pass,
    and a failed read (wandb outage, exhausted rate limit) holds the case instead of ending the orchestrator."""
    case_a = StubCase(name="case.a", hyperparam=True)
    study = make_stub_study([case_a])
    set_queue_snapshots(monkeypatch, [])
    monkeypatch.setattr(study, "check_data_requirements", lambda case: True)
    monkeypatch.setattr(study, "tuned_config_path", lambda case: study.working_dir / "untuned.yaml")
    monkeypatch.setattr(study, "launch_sweep", lambda case, **kwargs: None)

    read_times: list[float] = []

    def read_sweep(project):
        read_times.append(fake_clock.t)
        if len(read_times) == 1:
            raise requests.ConnectionError("wandb unreachable")
        if len(read_times) == 3:
            result_path = study.result_path(case_a)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.touch()
        return []

    monkeypatch.setattr(study_module, "get_completed_runs", read_sweep)

    study.run_unfinished_cases(skip_tuning=False, enable_parallelism=True)

    assert len(read_times) == 3
    read_gaps = np.diff(read_times)
    assert (read_gaps >= config.sweep_poll_interval_s).all()


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


@pytest.mark.parametrize(("job_id", "expected_attempts"), [(None, 0), (7, 1)])
def test_rejected_submission_refunds_train_attempt(make_stub_study, monkeypatch, job_id, expected_attempts):
    """A training job SLURM never accepted must not count toward MAX_TRAIN_ATTEMPTS, an accepted one is recorded for the hold."""
    case = StubCase("case.a")
    study = make_stub_study([case])
    monkeypatch.setattr(study, "make_train_config", lambda _case: STUB_TRAIN_CONFIG)
    monkeypatch.setattr(study_module, "launch_train_parallel", lambda *args, **kwargs: job_id)

    study.launch_train(case, 0, enable_parallelism=True, partition="test_partition")

    train_job_name = study.train_job_name(case, 0)
    assert study.train_attempts.get(train_job_name, 0) == expected_attempts
    recorded_job = study.submitted_train_jobs.get(train_job_name)
    assert (recorded_job.job_id if recorded_job else None) == job_id


def test_seed_finished_between_two_passes_is_not_relaunched(make_stub_study, monkeypatch, fake_clock, quiet_scheduler):
    """The gap of the first apsdpp launch: a seed submitted and finished between two squeue snapshots, never seen in flight.

    Its accepted submission holds it, and its result landing during the hold finishes the case.
    The other ways a gone job stays held are tested on slurm_utils.held_job_names itself.
    """
    case_a = StubCase(name="case.a")
    study = make_stub_study([case_a])
    monkeypatch.setattr(study, "check_data_requirements", lambda case: True)
    monkeypatch.setattr(study, "make_train_config", lambda _case: STUB_TRAIN_CONFIG)
    monkeypatch.setattr(study_module, "count_running_jobs", lambda job_name: 0)
    monkeypatch.setattr(slurm_utils, "job_end_states", lambda job_ids: {7: JobEnd("COMPLETED", 15.0)})
    launches: list[str] = []

    def accept_job(train_config, job_name, *args, **kwargs):
        launches.append(job_name)
        return 7

    monkeypatch.setattr(study_module, "launch_train_parallel", accept_job)

    def snapshots():
        yield set()  # pass 1: nothing queued, the seed launches
        yield set()  # pass 2: its job came and went between the snapshots, the result not yet visible
        result_path = study.result_path(case_a)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.touch()
        while True:
            yield set()

    snapshot_iter = snapshots()
    monkeypatch.setattr(study_module, "get_running_job_names", lambda: next(snapshot_iter))

    study.run_unfinished_cases(skip_tuning=True, enable_parallelism=True)

    assert launches == [study.train_job_name(case_a, 0)]


def test_gpu_only_cases_take_the_gpu_before_cpu_capable_ones(make_stub_study, monkeypatch):
    """One GPU slot opens per pass.

    The GPU-only case.b takes it ahead of the alphabetically earlier CPU-capable case.a,
    which goes to the CPU partition instead of waiting, and GPU-only case.c holds for the next pass.
    """
    cpu_capable = StubCase(name="case.a", model_type="cpu_type")
    gpu_only = [StubCase(name="case.b", model_type="gpu_type"), StubCase(name="case.c", model_type="gpu_type")]
    study = make_stub_study([cpu_capable, *gpu_only], partition="gpu", cpu_partition="cpu", cpu_model_types=("cpu_type",))
    gpu_slots = SimpleNamespace(open=1)

    def pick_partition(cpu_capable=False):
        if gpu_slots.open > 0:
            gpu_slots.open -= 1
            return "gpu"
        return "cpu" if cpu_capable else None

    launches: list[tuple[str, str]] = []

    def run_case(case, seed, skip_tuning, enable_parallelism, partition=None):
        launches.append((str(case), partition))
        result_path = study.result_path(case)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.touch()
        return True

    monkeypatch.setattr(study_module, "pick_partition", pick_partition)
    monkeypatch.setattr(study_module, "get_running_job_names", lambda: set())
    monkeypatch.setattr(study_module, "get_running_job_elapsed_s", lambda partition=None: {})
    monkeypatch.setattr(study_module, "recent_submitted_jobs", lambda name_prefix, name_suffix: {})
    monkeypatch.setattr(study_module.time, "sleep", lambda s: setattr(gpu_slots, "open", 1))
    monkeypatch.setattr(study, "run_case", run_case)

    study.run_unfinished_cases(skip_tuning=True, enable_parallelism=True)

    assert launches == [("case.b", "gpu"), ("case.a", "cpu"), ("case.c", "gpu")]


def test_every_seed_trains_and_their_merge_finishes_the_case(make_stub_study, monkeypatch):
    """Each seed of a case is its own launch.

    Once every seed has its result the loop merges them into the case result,
    keeping only the best seed's best checkpoint, and only then does a dependent case start.
    """
    prereq = StubCase(name="case.prereq")
    dependent = StubCase(name="case.dependent", prereqs=[prereq])
    study = make_stub_study([dependent, prereq], num_seeds=2)
    monkeypatch.setattr(study_module.time, "sleep", lambda s: None)
    launches: list[tuple[str, int]] = []

    def train_seed(case, seed, skip_tuning, enable_parallelism, partition=None):
        launches.append((str(case), seed))
        checkpoint_dir = study.seed_model_dir(case, seed) / "3"
        checkpoint_dir.mkdir(parents=True)
        # Seed 1 validates best
        seed_ds = with_seed_attrs(xr.Dataset({"error_abs_ts": ("shot", [float(seed)])}), val_loss=1.0 - seed, best_epoch=3)
        write_netcdf_atomic(seed_ds, study.seed_result_path(case, seed))
        return True

    monkeypatch.setattr(study, "run_case", train_seed)
    study.run_unfinished_cases(skip_tuning=True, enable_parallelism=False)

    assert launches == [("case.prereq", 0), ("case.prereq", 1), ("case.dependent", 0), ("case.dependent", 1)]
    prereq_result = xr.load_dataset(study.result_path(prereq))
    assert prereq_result["error_abs_ts"].sel(seed=1).item() == 1.0
    assert prereq_result.attrs[BEST_SEED_ATTR] == 1
    # Only the best seed's best checkpoint is left, where every restore of the case reads
    assert [path.name for path in study.trained_model_dir(prereq).iterdir()] == ["3"]
    assert not study.seed_result_path(prereq, 0).exists()
