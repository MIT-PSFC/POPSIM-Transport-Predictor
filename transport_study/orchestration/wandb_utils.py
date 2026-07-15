import functools
import subprocess
import time
from typing import Any

import wandb
from loguru import logger
from popsim.ml import TrainConfig
from requests.exceptions import HTTPError

from transport_study.config import config

# Reading run state/summary is one graphql request per run, so projects with
# many runs can exhaust the wandb rate limit even after the client's own
# internal retries give up and raise 429
# The limit window is per-minute, so wait it out and restart the read
RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BASE_WAIT_S = 64
API_TIMEOUT_S = 64


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
            except HTTPError as e:
                is_rate_limit = e.response is not None and e.response.status_code == 429
                if not is_rate_limit or attempt == RATE_LIMIT_RETRIES - 1:
                    raise
                wait_s = RATE_LIMIT_BASE_WAIT_S * 1.2**attempt
                logger.warning(f"wandb rate limit (429) in {fn.__name__}, retrying in {wait_s}s")
                time.sleep(wait_s)
        raise AssertionError("unreachable")

    return wrapper


def get_project(project: str):
    api = wandb.Api(timeout=API_TIMEOUT_S)

    if config.wandb_entity is None:
        raise ValueError("WANDB_ENTITY is not set, cannot check if project exists.")

    try:
        project_obj = api.project(project, entity=config.wandb_entity)
        len(project_obj.sweeps())  # Try to access sweeps to confirm project exists
        return project_obj
    except ValueError:
        return None


# Metric the sweeps optimize. Also used to tell hyperband-pruned trials apart
SWEEP_METRIC = "val/loss.mean"


@retry_rate_limited
def get_completed_runs(project: str) -> list[Any]:
    """Runs that count toward the hyperparam_sweeps trial target.

    A trial counts if it reported the sweep metric at least once: finished
    runs plus hyperband-pruned ones. Pruning is the intended fate of most
    trials, so pruned runs must count (and be kept) or the sweep would need
    many times hyperparam_sweeps trials to reach the target.
    Metric-less crashed/failed runs carry no information and are deleted.
    """
    api = wandb.Api(timeout=API_TIMEOUT_S)

    project_obj = get_project(project)
    if project_obj is None:
        logger.warning(f"No wandb project found for {project}, assuming no completed runs.")
        return []

    try:
        project_runs = api.runs(project)
        completed_runs = []
        for run in project_runs:
            run_state = run.state
            if run_state == "finished" or (run_state in ["crashed", "failed"] and SWEEP_METRIC in run.summary):
                completed_runs.append(run)
            elif run_state in ["crashed", "failed"]:
                run.delete()  # Clean up failed runs since they won't be useful and just take up space
    except ValueError as e:
        logger.warning(f"No wandb runs found for {project}, assuming no completed runs.")
        logger.debug(e)
        completed_runs = []

    return completed_runs


@retry_rate_limited
def get_best_train_config(project: str) -> TrainConfig | None:
    """Gets several pieces related to the final model for this case, if it exists.

    Only "finished" runs are eligible. A crashed or hyperband-pruned trial can
    log a low val/loss.mean in an early epoch and then diverge to NaN and die,
    its summary keeps that pre-divergence dip, so ranking those runs would pick
    a config that cannot complete a full training run. Restricting to finished
    runs guarantees the selected config trained to completion at least once.
    """
    completed_runs = get_completed_runs(project)
    finished_runs = [r for r in completed_runs if r.state == "finished"]
    if len(finished_runs) == 0:
        return None

    sorted_runs = sorted(finished_runs, key=lambda r: r.summary.get(SWEEP_METRIC, float("inf")))
    best_run = sorted_runs[0]
    logger.info(f"Best run is {best_run.name} with val loss {best_run.summary.get(SWEEP_METRIC)}")
    train_config = TrainConfig.load(best_run.config)

    return train_config


@retry_rate_limited
def get_sweep_id(project: str) -> str | None:
    """Existing active sweep id for a project, if any.

    Only the API call itself is treated as "assume no sweeps and let the
    caller create one" on failure. An ambiguous multi-active-sweep state is
    NOT caught here: it must propagate, otherwise launch_sweep sees None,
    creates yet another sweep, and the project accumulates duplicates forever
    (each extra active sweep only makes future calls more ambiguous, never
    less).
    Rate limit errors (429) also propagate for the same reason, treating a
    throttled read as "no sweeps" would create a duplicate. The decorator
    retries them with backoff instead.
    """
    project_obj = get_project(project)
    if project_obj is None:
        logger.warning(f"No wandb project found for {project}, assuming no sweeps.")
        return None

    try:
        project_sweeps = list(project_obj.sweeps())
    except HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            raise
        logger.warning(f"Error reading sweeps for {project}, assuming no sweeps.")
        logger.debug(e)
        return None
    except Exception as e:
        logger.warning(f"Error reading sweeps for {project}, assuming no sweeps.")
        logger.debug(e)
        return None

    if len(project_sweeps) == 0:
        return None
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


def run_clean_sweeps(projects: list[str]):
    def _delete_sweep(sweep):
        sweep_str = f"{sweep.entity}/{sweep.project}/{sweep.id}"
        result = subprocess.run(
            ["wandb", "sweep", "--cancel", sweep_str],
            capture_output=True,
            text=True,
            check=False,
        )
        return result

    def _delete_runs(project):
        api = wandb.Api()
        for run in api.runs(project):
            run.delete()

    for project in projects:
        try:
            project_obj = get_project(project)
            if project_obj is None:
                logger.warning(f"No wandb project found for {project}, skipping sweep cleanup.")
                continue
            project_sweeps = list(project_obj.sweeps())
            for sweep in project_sweeps:
                logger.info(f"Deleting sweep {sweep.id} for project {project}")
                _delete_sweep(sweep)
            if project_sweeps:
                _delete_runs(project)
        except Exception as e:
            logger.warning(f"Error reading sweeps for project {project}, skipping sweep cleanup.")
            logger.debug(e)
            continue
