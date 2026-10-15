"""Delete wandb projects whose name matches a regex pattern.

Project deletion is not exposed on wandb's public Python API (only run
deletion is). This uses the same `deleteModel` GraphQL mutation the wandb
web UI calls internally - confirmed via schema introspection against
api.wandb.ai, and it is a soft delete (the paired `undeleteProject`
mutation exists), but treat it as effectively permanent and confirm the
match list carefully before running with --yes.

Usage:
    python -m transport_study.scripts.delete_wandb_projects run --entity=z-team --pattern='^scratch-.*'
    python -m transport_study.scripts.delete_wandb_projects run --entity=z-team --pattern='^scratch-.*' --yes
"""

import re

import fire
import wandb
from loguru import logger
from wandb_gql import gql

DELETE_PROJECT_MUTATION = gql(
    """
    mutation DeleteProject($id: String!) {
        deleteModel(input: {id: $id}) {
            success
        }
    }
    """
)


def _matching_projects(api: wandb.Api, entity: str, pattern: str) -> list:
    regex = re.compile(pattern)
    return [p for p in api.projects(entity) if regex.search(p.name)]


def run(entity: str, pattern: str, *, yes: bool = False) -> None:
    """Delete every wandb project under `entity` whose name matches `pattern` (regex).

    Dry-run by default: lists matches and exits. Pass --yes to actually delete.
    """
    api = wandb.Api()
    matches = _matching_projects(api, entity, pattern)

    if not matches:
        logger.info(f"No projects under {entity!r} match pattern {pattern!r}.")
        return

    logger.info(f"{len(matches)} project(s) under {entity!r} match pattern {pattern!r}:")
    for p in matches:
        logger.info(f"  {p.entity}/{p.name}")

    if not yes:
        logger.info("Dry run only, rerun with --yes to delete these projects.")
        return

    for p in matches:
        try:
            result = api.client.execute(DELETE_PROJECT_MUTATION, variable_values={"id": p.id})
            success = result.get("deleteModel", {}).get("success", False)
            if success:
                logger.info(f"Deleted {p.entity}/{p.name}")
            else:
                logger.warning(f"Delete call for {p.entity}/{p.name} returned success=False")
        except Exception as e:
            logger.error(f"Failed to delete {p.entity}/{p.name}: {e}")


if __name__ == "__main__":
    fire.Fire({"run": run})
