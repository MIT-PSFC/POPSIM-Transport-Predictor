import subprocess
from typing import Any

import wandb
from loguru import logger
from popsim.ml import TrainConfig

from transport_study.config import config


def get_project(project: str, entity: str = config.wandb_entity):
    api = wandb.Api()

    if entity is None:
        logger.critical("WANDB_ENTITY is not set, cannot check if project exists.")
        return False

    try:
        project_obj = api.project(project, entity=entity)
        len(project_obj.sweeps())  # Try to access sweeps to confirm project exists
        return project_obj
    except Exception:
        return None


def get_completed_runs(project: str, entity: str = config.wandb_entity) -> list[Any]:
    api = wandb.Api()

    project_obj = get_project(project, entity)
    if project_obj is None:
        logger.warning(
            f"No wandb project found for {project}, assuming no completed runs."
        )
        return []

    try:
        project_runs = api.runs(project)
        completed_runs = []
        for run in project_runs:
            run_state = run.state
            if run_state == "finished":
                completed_runs.append(run)
            elif run_state in ["crashed", "failed"]:
                run.delete()  # Clean up failed runs since they won't be useful and just take up space
    except Exception as e:
        logger.warning(
            f"No wandb runs found for {project}, assuming no completed runs."
        )
        logger.debug(e)
        completed_runs = []

    return completed_runs


def get_best_train_config(project: str) -> TrainConfig | None:
    """Gets several pieces related to the final model for this case, if it exists."""
    completed_runs = get_completed_runs(project)
    if len(completed_runs) == 0:
        return None

    sorted_runs = sorted(
        completed_runs, key=lambda r: r.summary.get("val/loss.mean", float("inf"))
    )
    best_run = sorted_runs[0]
    logger.info(
        f"Best run is {best_run.name} with val loss {best_run.summary.get('val/loss.mean')}"
    )
    train_config = TrainConfig.load(best_run.config)

    return train_config


def get_sweep_id(project: str, entity: str = config.wandb_entity) -> str | None:
    project_obj = get_project(project, entity)
    if project_obj is None:
        logger.warning(f"No wandb project found for {project}, assuming no sweeps.")
        return None

    try:
        project_sweeps = project_obj.sweeps()
        if len(project_sweeps) == 0:
            return None
        active_sweeps = [s for s in project_sweeps if s.state in ["RUNNING", "PENDING"]]
        if len(active_sweeps) > 1:
            raise ValueError(
                f"Multiple running sweeps found for project {project}, cannot determine which to launch. Active sweeps: {[s.id for s in active_sweeps]}"
            )
        elif len(active_sweeps) == 0:
            return None
        sweep = active_sweeps[0]
        return sweep.id
    except Exception as e:
        logger.warning(f"Error reading sweeps for {project}, assuming no sweeps.")
        logger.debug(e)
        return None


def run_clean_sweeps(projects: list[str], entity: str = config.wandb_entity):
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
        api = wandb.api()
        for run in api.runs(project):
            run.delete()

    for project in projects:
        try:
            project_obj = get_project(project, entity)
            if project_obj is None:
                logger.warning(
                    f"No wandb project found for {project}, skipping sweep cleanup."
                )
                continue
            project_sweeps = project_obj.sweeps()
            for sweep in project_sweeps:
                logger.info(f"Deleting sweep {sweep.id} for project {project}")
                _delete_sweep(sweep)
                _delete_runs(project)
        except Exception as e:
            logger.warning(
                f"Error reading sweeps for project {project}, skipping sweep cleanup."
            )
            logger.debug(e)
            return
