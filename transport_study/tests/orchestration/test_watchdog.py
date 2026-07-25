"""Tests for the orchestrator's two job watchdogs.

Both exist so a case can never be stranded by a job that will not finish:

- The stall watchdog (Study._kill_stuck_jobs) kills a RUNNING training job that
  deadlocks (e.g. an OpenBLAS or XLA compile-pool hang) and so sits in the
  queue forever without checkpointing.
- The pending watchdog (Study._kill_long_pending_jobs) cancels a job stuck
  PENDING on a spillover partition, handing the case back to the launch path
  so pick_partition can re-decide placement. The primary partition is exempt.

The pending-watchdog bodies are blocked out as docstring stubs.
"""

import os
import time
from pathlib import Path

import pytest

from transport_study.orchestration import study as study_module
from transport_study.orchestration.study import (
    WATCHDOG_MIN_AGE_S,
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
    """Create a latest-checkpoint epoch dir, backdated by age_s seconds."""
    epoch_dir = Path(f"{study.trained_model_dir(case)}_latest") / str(epoch)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    mtime = time.time() - age_s
    os.utime(epoch_dir, (mtime, mtime))


# ----------------------------------------------------------------------
# Stall watchdog (Study._kill_stuck_jobs)
# ----------------------------------------------------------------------
def test_young_job_not_killed(study, case, killed, monkeypatch):
    """A job that has not yet run WATCHDOG_MIN_AGE_S is left alone even with no checkpoint."""
    set_job_elapsed(monkeypatch, study, case, WATCHDOG_MIN_AGE_S - 1)

    study._kill_stuck_jobs([case])

    assert killed == []


def test_old_job_with_recent_checkpoint_not_killed(study, case, killed, monkeypatch):
    """An old job still checkpointing recently is making progress, not stuck."""
    write_checkpoint(study, case, epoch=40, age_s=WATCHDOG_STALL_S - 1)
    set_job_elapsed(monkeypatch, study, case, WATCHDOG_MIN_AGE_S + 1)

    study._kill_stuck_jobs([case])

    assert killed == []


def test_old_job_with_stale_checkpoint_killed(study, case, killed, monkeypatch):
    """An old job whose newest checkpoint predates the stall window is deadlocked."""
    write_checkpoint(study, case, epoch=40, age_s=WATCHDOG_STALL_S + 1)
    set_job_elapsed(monkeypatch, study, case, WATCHDOG_MIN_AGE_S + 1)

    study._kill_stuck_jobs([case])

    assert killed == [study.train_job_name(case)]


def test_old_job_with_no_checkpoint_killed(study, case, killed, monkeypatch):
    """An old job that never wrote a first checkpoint is also deadlocked."""
    set_job_elapsed(monkeypatch, study, case, WATCHDOG_MIN_AGE_S + 1)

    study._kill_stuck_jobs([case])

    assert killed == [study.train_job_name(case)]


def test_job_absent_from_elapsed_map_not_killed(study, case, killed, monkeypatch):
    """The stall watchdog only judges RUNNING jobs.

    A job squeue reports no elapsed time for (e.g. still pending) has no
    checkpoint progress to measure, so it is the pending watchdog's business.
    """
    set_job_elapsed(monkeypatch, study, case, None)

    study._kill_stuck_jobs([case])

    assert killed == []


# ----------------------------------------------------------------------
# Pending watchdog (Study._kill_long_pending_jobs)
# ----------------------------------------------------------------------
def test_pending_job_past_threshold_is_cancelled():
    """A train job pending WATCHDOG_PENDING_S or longer on a spillover partition gets cancelled.

    Mock get_pending_job_pending_s to return the case's train job name at
    WATCHDOG_PENDING_S seconds and assert cancel_job is called once with
    state="PENDING" and the spillover partition string for that job name.
    """


def test_primary_partition_pending_jobs_are_exempt():
    """The watchdog only queries and cancels on spillover partitions.

    Both get_pending_job_pending_s and cancel_job must receive the
    comma-joined config.spillover_partitions (with config.partition filtered
    out if it appears there), never config.partition, so a job pending on the
    primary partition keeps its queue position no matter how long it pends.
    """


def test_no_spillover_partitions_disables_watchdog():
    """With config.spillover_partitions empty the watchdog is a no-op.

    Neither squeue (get_pending_job_pending_s) nor scancel (cancel_job) is
    called, every job can only be pending on the primary partition.
    """


def test_pending_job_under_threshold_is_left_alone():
    """A train job pending less than WATCHDOG_PENDING_S is not cancelled.

    Mock the pending map just under the threshold and assert cancel_job is
    never called.
    """


def test_pending_cancel_refunds_train_attempt():
    """Cancelling a pending train job decrements train_attempts for the case.

    Seed study.train_attempts[str(case)] = 2, trigger the watchdog, and assert
    the counter drops to 1. Repeated pending-cancel cycles must never reach
    MAX_TRAIN_ATTEMPTS, since the job never ran.
    """


def test_pending_refund_does_not_go_negative():
    """A cancel with no recorded launch attempt leaves the counter at 0.

    Covers the orchestrator-restart path where train_attempts is empty but a
    pending job from the previous orchestrator process is still in the queue.
    """


def test_agent_pending_cancel_skips_refund():
    """Cancelling a pending agent job leaves train_attempts untouched.

    Hyperparam case with a pending agent job past the threshold: cancel_job is
    called for the agent job name but the train attempt counter is unchanged,
    launch_sweep handles agent top-up on its own.
    """


def test_squeue_failure_skips_pending_watchdog():
    """When get_pending_job_pending_s returns None nothing is cancelled.

    Scheduler-unreachable must not look like "no pending jobs" and must not
    trigger any scancel calls.
    """


# ----------------------------------------------------------------------
# The slurm_utils helpers both watchdogs drive
# ----------------------------------------------------------------------
def test_get_pending_job_pending_s_parses_output():
    """squeue -O 'PendingTime:20,Name:512' lines parse into {name: seconds}.

    Mock subprocess.run stdout with padded columns, a blank line, and two
    entries sharing a name where the larger pending time must win.
    """


def test_cancel_job_state_parameter_reaches_scancel():
    """cancel_job(job, state="PENDING") passes --state=PENDING to scancel.

    Default call keeps --state=RUNNING so the stall watchdog behavior is
    unchanged, and one scancel runs per configured partition.
    """
