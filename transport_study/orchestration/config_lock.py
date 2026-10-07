"""The config lock, config_lock.toml in every study's working dir.

It records the setup a study's results were produced under, so a later run can never mix them with another one.
The identity and locked fields (see transport_study/config.py) must match on every run.
The case-grid axes are kept current by the orchestrator as the record of the study's grid.
Orchestration fields are left out, they never change what a case produces.

The [lock] table, written last, holds the study type, a stamp and the parent's stamp.
The lock is deleted whenever models or results are cleaned and written fresh with a new stamp,
so a stamp identifies one consistent set of results.
A child study records its parent's stamp when its own lock is created (see lineage.py),
so a reset parent is caught by every descendant.
"""

import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path

import toml
from loguru import logger

from transport_study.config import ROLE_TABLES, FieldRole, StudyConfig

CONFIG_LOCK_FILENAME = "config_lock.toml"
LOCK_TABLE = "lock"
LOCK_ROLES = (FieldRole.IDENTITY, FieldRole.LOCKED, FieldRole.CASE_AXIS)


@dataclass(frozen=True)
class ConfigLock:
    config: StudyConfig
    study_type: str
    stamp: str
    # The parent study's stamp when this lock was created, None without a parent
    parent_stamp: str | None


def new_stamp() -> str:
    return uuid.uuid4().hex


def config_from_lock_data(config_cls: type[StudyConfig], data: dict) -> StudyConfig:
    """A config from the tables of a config lock.

    Unlike from_toml it reads the [datasets] paths alone, without PTPS_DATASET_PATHS,
    so a comparison sees exactly the devices the lock recorded.
    Fields the class no longer has are dropped with a warning,
    so removing a config field never makes every existing lock unreadable.
    """
    data = dict(data)
    datasets = dict(data.pop("datasets", {}))
    target_device = datasets.pop("target", None)
    entries = {}
    for name, value in data.items():
        if name in ROLE_TABLES.values():
            entries.update(value)
        else:
            entries[name] = value
    unknown = sorted(name for name in entries if name not in config_cls.model_fields)
    if unknown:
        logger.warning(f"Ignoring config lock fields {config_cls.__qualname__} no longer has: {unknown}")
    kwargs = {name: value for name, value in entries.items() if name in config_cls.model_fields}
    dataset_paths = {device: Path(path) for device, path in datasets.items()}
    return config_cls(**kwargs, dataset_paths=dataset_paths, target_device=target_device)


def read_config_lock(path: Path, config_cls: type[StudyConfig]) -> ConfigLock:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    lock_table = data.pop(LOCK_TABLE, None)
    if lock_table is None:
        raise RuntimeError(
            f"Config lock {path} has no [{LOCK_TABLE}] table. Clean the study (clean_models and clean_results) to recreate it."
        )
    lock_config = config_from_lock_data(config_cls, data)
    return ConfigLock(
        config=lock_config,
        study_type=lock_table["study_type"],
        stamp=lock_table["stamp"],
        parent_stamp=lock_table.get("parent_stamp"),
    )


def write_config_lock(path: Path, lock: ConfigLock) -> None:
    """Write the lock through a uniquely named temporary file, so a concurrent reader never sees a partial lock."""
    lock_data = lock.config.toml_data(LOCK_ROLES)
    lock_data[LOCK_TABLE] = {"study_type": lock.study_type, "stamp": lock.stamp}
    if lock.parent_stamp is not None:
        lock_data[LOCK_TABLE]["parent_stamp"] = lock.parent_stamp
    write_toml_atomic(path, lock_data)


def write_toml_atomic(path: Path, data: dict) -> None:
    """Write data through a uniquely named temporary file, so a concurrent reader never sees a partial file."""
    tmp_path = path.with_name(f"{path.name}.{new_stamp()}.tmp")
    with open(tmp_path, "w") as f:
        toml.dump(data, f)
    tmp_path.replace(path)
