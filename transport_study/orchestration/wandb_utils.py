from functools import lru_cache
from typing import Any

import wandb
from loguru import logger
from popsim.ml import TrainConfig

from transport_study.config import config


@lru_cache(maxsize=1)
def wandb_api() -> wandb.Api:
    return wandb.Api()


def get_completed_runs(project: str, entity: str = config.wandb_entity) -> list[Any]:
    api = wandb_api()
    try:
        project_runs = api.runs(f"{entity}/{project}")
        completed_runs = [r for r in project_runs if r.state == "finished"]
    except ValueError:
        logger.warning(
            f"No wandb project found for {project}, assuming no completed runs."
        )
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
    api = wandb_api()
    try:
        project_sweeps = api.project(f"{entity}/{project}").sweeps()
        if len(project_sweeps) == 0:
            return None
        sweep = project_sweeps[0]
        return sweep.id
    except ValueError:
        logger.warning(f"No wandb project found for {project}, assuming no sweeps.")
        return None


def run_clean_sweeps(projects: list[str], entity: str = config.wandb_entity):
    api = wandb_api()
    for project in projects:
        try:
            project_sweeps = api.project(f"{entity}/{project}").sweeps()
            for sweep in project_sweeps:
                logger.info(f"Deleting sweep {sweep.id} for project {project}")
                sweep.delete()
        except ValueError:
            logger.warning(
                f"No wandb project found for {project}, skipping sweep cleanup."
            )
