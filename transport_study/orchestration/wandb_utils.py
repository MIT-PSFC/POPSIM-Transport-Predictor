import subprocess
from typing import Any

from loguru import logger
from popsim.ml import TrainConfig

import wandb
from transport_study.config import config


def get_project(project: str):
    api = wandb.Api()

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


def get_completed_runs(project: str) -> list[Any]:
    """Runs that count toward the hyperparam_sweeps trial target.

    A trial counts if it reported the sweep metric at least once: finished
    runs plus hyperband-pruned ones. Pruning is the intended fate of most
    trials, so pruned runs must count (and be kept) or the sweep would need
    many times hyperparam_sweeps trials to reach the target.
    Metric-less crashed/failed runs carry no information and are deleted.
    """
    api = wandb.Api()

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


def get_best_train_config(project: str) -> TrainConfig | None:
    """Gets several pieces related to the final model for this case, if it exists."""
    completed_runs = get_completed_runs(project)
    if len(completed_runs) == 0:
        return None

    sorted_runs = sorted(completed_runs, key=lambda r: r.summary.get(SWEEP_METRIC, float("inf")))
    best_run = sorted_runs[0]
    logger.info(f"Best run is {best_run.name} with val loss {best_run.summary.get(SWEEP_METRIC)}")
    train_config = TrainConfig.load(best_run.config)

    return train_config


def get_sweep_id(project: str) -> str | None:
    """Existing active sweep id for a project, if any.

    Only the API call itself is treated as "assume no sweeps and let the
    caller create one" on failure. An ambiguous multi-active-sweep state is
    NOT caught here: it must propagate, otherwise launch_sweep sees None,
    creates yet another sweep, and the project accumulates duplicates forever
    (each extra active sweep only makes future calls more ambiguous, never
    less).
    """
    project_obj = get_project(project)
    if project_obj is None:
        logger.warning(f"No wandb project found for {project}, assuming no sweeps.")
        return None

    try:
        project_sweeps = list(project_obj.sweeps())
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
            project_sweeps = project_obj.sweeps()
            for sweep in project_sweeps:
                logger.info(f"Deleting sweep {sweep.id} for project {project}")
                _delete_sweep(sweep)
                _delete_runs(project)
        except Exception as e:
            logger.warning(f"Error reading sweeps for project {project}, skipping sweep cleanup.")
            logger.debug(e)
            continue
