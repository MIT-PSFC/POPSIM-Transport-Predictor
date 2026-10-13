import functools
import math
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import wandb
from loguru import logger
from popsim.ml import TrainConfig
from popsim.ml.loggers import STOP_REQUESTED_SUMMARY_KEY
from requests.exceptions import HTTPError, RequestException
from wandb.errors import CommError

from transport_study.config import config

# Reading run state/summary is one graphql request per run, so projects with
# many runs can exhaust the wandb rate limit even after the client's own
# internal retries give up and raise 429
# The limit window is per-minute, so wait it out and restart the read
RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BASE_WAIT_S = 64
API_TIMEOUT_S = 64

# Metric the sweeps optimize. Also used to tell hyperband-pruned trials apart
SWEEP_METRIC = "val/loss.mean"

# What a failed wandb read raises: requests errors from the paginators,
# CommError from every call wandb wraps in normalize_exceptions (a 429 among them, in its exc)
WANDB_READ_ERRORS = (RequestException, CommError)


class SweepReadError(RuntimeError):
    """A project's sweeps could not be read, so whether it has an active sweep is unknown."""


def _is_rate_limit(error: Exception) -> bool:
    """Whether a wandb read failed on HTTP 429, raised as an HTTPError or as a CommError wrapping one."""
    http_error = error.exc if isinstance(error, CommError) else error
    return isinstance(http_error, HTTPError) and http_error.response is not None and http_error.response.status_code == 429


def retry_rate_limited(fn):
    """Retry a wandb API read that raised HTTP 429, with exponential backoff.

    Only 429 is retried, everything else propagates unchanged. The wrapped
    function must be safe to rerun from the top (all readers here are).
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for attempt in range(RATE_LIMIT_RETRIES):
            try:
                return fn(*args, **kwargs)
            except (HTTPError, CommError) as e:
                if not _is_rate_limit(e) or attempt == RATE_LIMIT_RETRIES - 1:
                    raise
                wait_s = RATE_LIMIT_BASE_WAIT_S * 1.2**attempt
                logger.warning(f"wandb rate limit (429) in {fn.__name__}, retrying in {wait_s}s")
                time.sleep(wait_s)
        raise AssertionError("unreachable")

    return wrapper


def wandb_entity() -> str:
    """The entity every study sweep, agent and API read uses."""
    if config.wandb_entity is None:
        raise ValueError("config.wandb_entity (PTPS_WANDB_ENTITY) is not set")
    return config.wandb_entity


def _project_runs(project: str) -> list:
    """Every run of a project, empty when the project does not exist yet."""
    entity = wandb_entity()
    try:
        return list(wandb.Api(timeout=API_TIMEOUT_S).runs(f"{entity}/{project}"))
    except ValueError:
        # wandb raises ValueError for a project that does not exist
        logger.warning(f"No wandb project found for {project}")
        return []


def _project_sweeps(project: str) -> list:
    """Every sweep of a project, empty when the project does not exist yet."""
    entity = wandb_entity()
    project_obj = wandb.Api(timeout=API_TIMEOUT_S).project(project, entity=entity)
    try:
        return list(project_obj.sweeps())
    except ValueError:
        logger.warning(f"No wandb project found for {project}")
        return []


@retry_rate_limited
def get_completed_runs(project: str) -> list[Any]:
    """Runs that count toward the hyperparam_sweeps trial target.

    A trial counts if it reported the sweep metric at least once: finished
    runs plus hyperband-pruned ones. Pruning is the intended fate of most
    trials, so pruned runs must count (and be kept) or the sweep would need
    many times hyperparam_sweeps trials to reach the target.
    Metric-less crashed/failed runs carry no information and are deleted.
    """
    completed_runs = []
    for run in _project_runs(project):
        run_state = run.state
        if run_state == "finished" or (run_state in ["crashed", "failed"] and SWEEP_METRIC in run.summary):
            completed_runs.append(run)
        elif run_state in ["crashed", "failed"]:
            run.delete()
    return completed_runs


def is_trained_to_completion(run) -> bool:
    """Whether a trial ran until training itself ended it.

    That is max_epochs, early stopping or divergence.
    A trial cut by the wall-clock budget ends preempted (popsim launch.finish_sweep_run),
    and counts once the agent that resumes it finishes it.
    A trial the sweep stopped (hyperband pruning) still ends "finished" on the server,
    so popsim marks it in the run summary.
    """
    return run.state == "finished" and not run.summary.get(STOP_REQUESTED_SUMMARY_KEY, False)


def _history_min(run) -> float:
    """The lowest finite sweep metric a run logged, NaN when it logged none.

    The run summary holds the last logged value instead,
    which for an early-stopped trial is patience validations past its best.
    """
    history_rows = run.scan_history(keys=[SWEEP_METRIC])
    history_values = [row[SWEEP_METRIC] for row in history_rows]
    # wandb returns a logged NaN or infinity as a float or as a string, depending on the client
    finite_values = [float(value) for value in history_values if isinstance(value, int | float) and math.isfinite(value)]
    return min(finite_values, default=math.nan)


@retry_rate_limited
def get_best_train_config(completed_runs: list) -> TrainConfig | None:
    """Train config of the best trial of completed_runs (get_completed_runs), None when no trial is eligible.

    Only trials trained to completion with a finite sweep metric are eligible.
    A crashed or hyperband-pruned trial never completed a full training run.
    Each trial ranks by the lowest val/loss.mean it logged,
    the value the bayes search fits and the checkpoint a production run is tested at.
    """
    trained_runs = [run for run in completed_runs if is_trained_to_completion(run)]
    run_minima = {run.id: _history_min(run) for run in trained_runs}
    ranked_runs = [run for run in trained_runs if math.isfinite(run_minima[run.id])]
    if len(ranked_runs) == 0:
        return None
    best_run = min(ranked_runs, key=lambda run: run_minima[run.id])
    logger.info(f"Best run is {best_run.name} with minimum val loss {run_minima[best_run.id]}")
    return TrainConfig.load(best_run.config)


@retry_rate_limited
def get_sweep_id(project: str) -> str | None:
    """Id of the project's active sweep, None when it has none.

    A failed read raises SweepReadError instead of reading as "no sweep".
    The caller would create a duplicate sweep, and two active sweeps abort the next pass.
    Rate limit errors (429) propagate to the retry decorator.
    Several active sweeps raise ValueError, no choice between them is safe.
    """
    try:
        project_sweeps = _project_sweeps(project)
    except WANDB_READ_ERRORS as e:
        if _is_rate_limit(e):
            raise
        raise SweepReadError(f"Could not read the sweeps of {project}: {e}") from e

    active_sweeps = [s for s in project_sweeps if s.state in ["RUNNING", "PENDING"]]
    if len(active_sweeps) > 1:
        raise ValueError(
            f"Multiple active sweeps found for project {project}, cannot determine which to launch. "
            f"Active sweeps: {[s.id for s in active_sweeps]}. Cancel the extras (wandb sweep --cancel) "
            "or rerun with clean_sweeps=True before continuing."
        )
    if len(active_sweeps) == 0:
        return None
    return active_sweeps[0].id


@retry_rate_limited
def has_live_agent_run(project: str, stall_s: float) -> bool:
    """Whether any run in the project is actively being trained right now.

    A run stays "running" server-side until wandb's own heartbeat timeout
    trips, so a node that died mid-trial without a clean exit would still
    read as "running" long after it stopped. Requiring a recent heartbeat on
    top of state=="running" catches that case too.
    """
    now = datetime.now(UTC)
    for run in _project_runs(project):
        heartbeat_at = run.heartbeat_at
        if run.state != "running" or heartbeat_at is None:
            continue
        heartbeat_dt = datetime.fromisoformat(heartbeat_at)
        # wandb timestamps are UTC, with or without the Z suffix
        if heartbeat_dt.tzinfo is None:
            heartbeat_dt = heartbeat_dt.replace(tzinfo=UTC)
        heartbeat_age_s = (now - heartbeat_dt).total_seconds()
        if heartbeat_age_s < stall_s:
            return True
    return False


def run_clean_sweeps(projects: Iterable[str]):
    """Cancel every sweep of each project, then delete the project's runs."""
    entity = wandb_entity()
    for project in projects:
        try:
            project_sweeps = _project_sweeps(project)
            for sweep in project_sweeps:
                logger.info(f"Cancelling sweep {sweep.id} for project {project}")
                # The interpreter's own wandb CLI, PATH may resolve to another install
                result = subprocess.run(
                    [sys.executable, "-m", "wandb", "sweep", "--cancel", f"{entity}/{project}/{sweep.id}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    logger.error(f"Cancelling sweep {sweep.id} for project {project} failed: {result.stderr}")
            if project_sweeps:
                for run in _project_runs(project):
                    run.delete()
        except WANDB_READ_ERRORS as e:
            logger.warning(f"wandb read failed for project {project}, skipping its sweep cleanup: {e}")
