"""Tests for the orchestrator's two job watchdogs.

Both exist so a case can never be stranded by a job that will not finish:

- The stall watchdog (Study.kill_stuck_jobs) kills a RUNNING training job that
  deadlocks (e.g. an OpenBLAS or XLA compile-pool hang) and so sits in the
  queue forever without checkpointing.
- The pending watchdog (Study._kill_long_pending_jobs) cancels a job stuck
  PENDING on a spillover partition, handing the case back to the launch path
  so pick_partition can re-decide placement. The primary partition is exempt.
"""

import os
import time
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
from popsim.ml.checkpointing import (
    TrainState,
    create_default_checkpoint_manager,
    save_train_state,
)

from transport_study.orchestration import slurm_utils
from transport_study.orchestration import study as study_module
from transport_study.orchestration.study import (
    WATCHDOG_MIN_AGE_S,
    WATCHDOG_PENDING_S,
    WATCHDOG_STALL_S,
    Study,
)
from transport_study.tests.stubs import StubCase


@pytest.fixture
def study(make_stub_study) -> Study:
    return make_stub_study([StubCase(name="case.a")], study_name="watchdog_test")


@pytest.fixture
def case(study) -> StubCase:
    return study.cases[0]


@pytest.fixture
def killed(monkeypatch) -> list[str]:
    """Job names either watchdog cancelled."""
    cancelled: list[str] = []
    monkeypatch.setattr(study_module, "cancel_job", cancelled.append)
    return cancelled


def set_job_elapsed(monkeypatch, study, case, elapsed_s: float | None):
    """Report the case's train job as running for elapsed_s (None: not running)."""
    elapsed = {} if elapsed_s is None else {study.train_job_name(case): elapsed_s}
    monkeypatch.setattr(study_module, "get_running_job_elapsed_s", lambda partition=None: elapsed)


def write_checkpoint(study: Study, case, epoch: int, age_s: float):
    """Create a checkpoint epoch dir, backdated by age_s seconds."""
    epoch_dir = study.trained_model_dir(case) / str(epoch)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    mtime = time.time() - age_s
    os.utime(epoch_dir, (mtime, mtime))


# ----------------------------------------------------------------------
# Stall watchdog (Study.kill_stuck_jobs)
# ----------------------------------------------------------------------
def test_young_job_not_killed(study, case, killed, monkeypatch):
    """A job that has not yet run WATCHDOG_MIN_AGE_S is left alone even with no checkpoint."""
    set_job_elapsed(monkeypatch, study, case, WATCHDOG_MIN_AGE_S - 1)

    study.kill_stuck_jobs([case])

    assert killed == []


def test_old_job_with_recent_checkpoint_not_killed(study, case, killed, monkeypatch):
    """An old job still checkpointing recently is making progress, not stuck."""
    write_checkpoint(study, case, epoch=40, age_s=WATCHDOG_STALL_S - 1)
    set_job_elapsed(monkeypatch, study, case, WATCHDOG_MIN_AGE_S + 1)

    study.kill_stuck_jobs([case])

    assert killed == []


def test_old_job_with_stale_checkpoint_killed(study, case, killed, monkeypatch):
    """An old job whose newest checkpoint predates the stall window is deadlocked."""
    write_checkpoint(study, case, epoch=40, age_s=WATCHDOG_STALL_S + 1)
    set_job_elapsed(monkeypatch, study, case, WATCHDOG_MIN_AGE_S + 1)

    study.kill_stuck_jobs([case])

    assert killed == [study.train_job_name(case)]


def test_old_job_with_no_checkpoint_killed(study, case, killed, monkeypatch):
    """An old job that never wrote a first checkpoint is also deadlocked."""
    set_job_elapsed(monkeypatch, study, case, WATCHDOG_MIN_AGE_S + 1)

    study.kill_stuck_jobs([case])

    assert killed == [study.train_job_name(case)]


def test_job_absent_from_elapsed_map_not_killed(study, case, killed, monkeypatch):
    """The stall watchdog only judges RUNNING jobs.

    A job squeue reports no elapsed time for (e.g. still pending) has no
    checkpoint progress to measure, so it is the pending watchdog's business.
    """
    set_job_elapsed(monkeypatch, study, case, None)

    study.kill_stuck_jobs([case])

    assert killed == []


def test_latest_checkpoint_epoch_reads_real_checkpoint_layout(study, case):
    """Checkpoints saved by popsim itself must be visible to latest_checkpoint_epoch.

    The stall watchdog and the launch_train resume counter both read it,
    so a layout mismatch kills every healthy job and never resets the counter.
    With max_to_keep 1 and the best loss at epoch 1, the newest dir left is the unvalidated latest one.
    """
    manager = create_default_checkpoint_manager(study.trained_model_dir(case), max_to_keep=1)
    for epoch, loss in {1: 0.2, 2: 0.9, 3: None}.items():
        state = TrainState(step=epoch, epoch=epoch, model={"w": jnp.zeros(2)}, opt_state={"dummy": jnp.zeros(1)})
        save_train_state(state, manager, loss=loss)

    assert study.latest_checkpoint_epoch(case) == 3


# ----------------------------------------------------------------------
# Pending watchdog (Study._kill_long_pending_jobs, driven through run_unfinished_cases)
# ----------------------------------------------------------------------
SPILLOVER_PARTITIONS = ("spill_a", "spill_b")


@pytest.fixture
def spillover_study(make_stub_study) -> Study:
    """The primary partition also listed among the spillover ones, which the watchdog must filter out."""
    return make_stub_study(
        [StubCase(name="case.a", hyperparam=True)],
        study_name="pending_watchdog_test",
        partition="primary",
        spillover_partitions=("primary", *SPILLOVER_PARTITIONS),
    )


def run_one_pass(monkeypatch, study: Study, pending: dict[str, int] | None) -> SimpleNamespace:
    """One run_unfinished_cases pass with the given squeue pending map, the case finishing in it.

    Returns the partitions the pending squeue was asked about and every cancel_job call.
    """
    calls = SimpleNamespace(pending_queries=[], cancels=[])

    def pending_job_pending_s(partition=None):
        calls.pending_queries.append(partition)
        return pending

    def cancel_job(job_name, partition=None, state="RUNNING"):
        calls.cancels.append((job_name, partition, state))

    def finish_case(case, skip_tuning, enable_parallelism, partition=None):
        result_path = study.result_path(case)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.touch()

    monkeypatch.setattr(study_module, "get_pending_job_pending_s", pending_job_pending_s)
    monkeypatch.setattr(study_module, "cancel_job", cancel_job)
    monkeypatch.setattr(study_module, "get_running_job_names", lambda: set())
    monkeypatch.setattr(study_module, "get_running_job_elapsed_s", lambda partition=None: {})
    monkeypatch.setattr(study_module, "pick_partition", lambda *args, **kwargs: "primary")
    monkeypatch.setattr(study_module.time, "sleep", lambda s: None)
    monkeypatch.setattr(study, "run_case", finish_case)
    study.run_unfinished_cases(skip_tuning=True, enable_parallelism=True)
    return calls


@pytest.mark.parametrize(("pending_s", "cancelled"), [(WATCHDOG_PENDING_S, True), (WATCHDOG_PENDING_S - 1, False)])
def test_spillover_train_job_cancelled_past_threshold(spillover_study, monkeypatch, pending_s, cancelled):
    """Only the spillover partitions are queried and cancelled on, so a primary-partition job keeps its queue position."""
    train_job_name = spillover_study.train_job_name(spillover_study.cases[0])

    calls = run_one_pass(monkeypatch, spillover_study, {train_job_name: pending_s})

    spillover = ",".join(SPILLOVER_PARTITIONS)
    assert calls.pending_queries == [spillover]
    assert calls.cancels == ([(train_job_name, spillover, "PENDING")] if cancelled else [])


def test_no_spillover_partitions_disables_watchdog(make_stub_study, monkeypatch):
    study = make_stub_study([StubCase(name="case.a")], partition="primary", spillover_partitions=())
    train_job_name = study.train_job_name(study.cases[0])

    calls = run_one_pass(monkeypatch, study, {train_job_name: 10 * WATCHDOG_PENDING_S})

    assert calls.pending_queries == []
    assert calls.cancels == []


@pytest.mark.parametrize(("attempts_before", "attempts_after"), [(2, 1), (None, 0)])
def test_pending_cancel_refunds_train_attempt(spillover_study, monkeypatch, attempts_before, attempts_after):
    """A job cancelled while pending never ran, so its launch attempt is refunded, never below 0.

    None is an orchestrator restart, an empty counter facing a job the previous process submitted.
    """
    case = spillover_study.cases[0]
    if attempts_before is not None:
        spillover_study.train_attempts[str(case)] = attempts_before

    run_one_pass(monkeypatch, spillover_study, {spillover_study.train_job_name(case): WATCHDOG_PENDING_S})

    assert spillover_study.train_attempts[str(case)] == attempts_after


def test_agent_pending_cancel_skips_refund(spillover_study, monkeypatch):
    """launch_sweep tops agents back up on its own, so cancelling a pending agent refunds nothing."""
    case = spillover_study.cases[0]
    spillover_study.train_attempts[str(case)] = 2
    agent_job_name = spillover_study.agent_job_name(case)

    calls = run_one_pass(monkeypatch, spillover_study, {agent_job_name: WATCHDOG_PENDING_S})

    assert calls.cancels == [(agent_job_name, ",".join(SPILLOVER_PARTITIONS), "PENDING")]
    assert spillover_study.train_attempts[str(case)] == 2


def test_squeue_failure_skips_pending_watchdog(spillover_study, monkeypatch):
    """An unreachable scheduler must not read as "no pending jobs" and trigger cancels."""
    calls = run_one_pass(monkeypatch, spillover_study, None)

    assert calls.cancels == []


@pytest.fixture
def slurm_commands(monkeypatch) -> SimpleNamespace:
    """Record every SLURM command and answer it with the configured stdout."""
    record = SimpleNamespace(commands=[], stdout="")

    def run(cmd, **kwargs):
        record.commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout=record.stdout, stderr="")

    monkeypatch.setattr(slurm_utils.subprocess, "run", run)
    return record


def test_get_pending_job_pending_s_parses_output(slurm_commands):
    """PendingTime:20,Name:512 columns parse into {name: seconds}, a repeated name keeping its longest wait."""
    slurm_commands.stdout = "1234                case.a     \n\n50                  agent.b    \n900                 agent.b    \n"

    pending = slurm_utils.get_pending_job_pending_s(partition="spill_a")

    assert pending == {"case.a": 1234, "agent.b": 900}


def test_cancel_job_runs_one_scancel_per_partition_in_the_given_state(slurm_commands):
    """scancel -p takes one partition, a comma list would match nothing while still exiting 0."""
    slurm_utils.cancel_job("job.a", partition="spill_a,spill_b", state="PENDING")
    slurm_utils.cancel_job("job.a", partition="spill_a")

    partitions = [cmd[cmd.index("-p") + 1] for cmd in slurm_commands.commands]
    states = [next(arg for arg in cmd if arg.startswith("--state=")) for cmd in slurm_commands.commands]
    assert partitions == ["spill_a", "spill_b", "spill_a"]
    assert states == ["--state=PENDING", "--state=PENDING", "--state=RUNNING"]


def test_cpu_capable_job_pending_for_a_gpu_is_cancelled(make_stub_study, monkeypatch):
    """A case that can also train on CPU stops waiting for a primary GPU past the threshold, so it can be re-placed on CPU.

    A GPU-only case pending there keeps its queue position.
    """
    cpu_capable = StubCase(name="case.cpu", model_type="cpu_type")
    gpu_only = StubCase(name="case.gpu", model_type="gpu_type")
    study = make_stub_study([cpu_capable, gpu_only], partition="primary", cpu_partition="cpu_primary", cpu_model_types=("cpu_type",))
    pending = {study.train_job_name(case): WATCHDOG_PENDING_S for case in (cpu_capable, gpu_only)}

    calls = run_one_pass(monkeypatch, study, pending)

    assert calls.pending_queries == ["primary"]
    assert calls.cancels == [(study.train_job_name(cpu_capable), "primary", "PENDING")]
