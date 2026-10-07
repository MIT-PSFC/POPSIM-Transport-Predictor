"""
Load and parse configuration files for the project.
This is where we set global variables from env vars or config files.

Every config field has exactly one role, declared with its type,
because the role decides whether changing the field can mix results from two setups:
- Identity (Identity[T]): names the study, whose working dir is working_dir_base / study_name.
- Locked (no marker): changes what a case produces without changing the case's name.
  The config lock refuses a run that changes one.
  Unmarked fields are locked on purpose, so a new setting can never silently mix results.
- Case axis (Annotated[T, CaseAxis(<Case field>)]): the values one Case field takes across the case grid.
  It selects which cases exist, never what one case produces, so it may change between runs.
- Orchestration (Orchestration[T]): where and how jobs run.
  It never changes what a job computes, so it is neither locked nor written to the lock.

A study TOML mirrors the roles:
identity and locked fields at the top level, the axes under [cases], the orchestration settings under [orchestration],
and the device dataset paths plus the target device under [datasets].
"""

import json
import os
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Self, TypeVar

import numpy as np
import toml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

# 80/20 between train/val, by a deterministic hazard sort: the highest-hazard fifth of each source is its validation set
TRAIN_VAL_SPLIT = (0.8, 0.2)

# Uniform rho_tor_norm grid the profile and transport studies put every
# dataset on and every profile predictor family predicts on
# Changing this changes the structure of the modules, cannot restore from checkpoints trained on a different grid
N_RHO_POINTS = 51
RHO_GRID = np.linspace(0.0, 1.0, N_RHO_POINTS)

# debug runs a whole study cheaply:
# the target device keeps every shot, each source device keeps only its DEBUG_MAX_SOURCE_SHOTS most recent ones,
# training and sweep trials stop by DEBUG_MAX_EPOCHS and validate at least every DEBUG_EPOCHS_PER_VAL epochs,
# each sweep runs DEBUG_HYPERPARAM_SWEEPS trials,
# and result files average at most the top DEBUG_NUM_RESULT_CHECKPOINTS checkpoints (CaseGridConfig)
DEBUG_MAX_SOURCE_SHOTS = 100
DEBUG_MAX_EPOCHS = 12
DEBUG_EPOCHS_PER_VAL = 3
DEBUG_HYPERPARAM_SWEEPS = 1
DEBUG_NUM_RESULT_CHECKPOINTS = 3


class FieldRole(StrEnum):
    """What a config field controls, see the module docstring."""

    IDENTITY = "identity"
    LOCKED = "locked"
    CASE_AXIS = "case_axis"
    ORCHESTRATION = "orchestration"


@dataclass(frozen=True)
class CaseAxis:
    """Marks a config field as a case-grid axis, the values the Case field case_field takes across the grid."""

    case_field: str


T = TypeVar("T")
Identity = Annotated[T, FieldRole.IDENTITY]
Orchestration = Annotated[T, FieldRole.ORCHESTRATION]

# TOML table of each role kept out of the top level, where the identity and locked fields sit
ROLE_TABLES = {FieldRole.CASE_AXIS: "cases", FieldRole.ORCHESTRATION: "orchestration"}


def _toml_location(table: str | None) -> str:
    return f"in [{table}]" if table else "at the top level"


def env_dataset_paths() -> dict[str, Path]:
    """Parse PTPS_DATASET_PATHS, a JSON dict of device -> dataset path,
    e.g. PTPS_DATASET_PATHS='{"cmod": "/path/to/ds.zarr"}' -> {"cmod": Path(...)}
    """
    return {k: Path(v) for k, v in json.loads(os.environ.get("PTPS_DATASET_PATHS", "{}")).items()}


def _env_list(var_name: str, default: str = "", lower: bool = False) -> tuple[str, ...]:
    """Parse a comma-separated environment variable, e.g. PTPS_SPILLOVER_PARTITIONS="mit_preemptable,mit_normal_gpu"."""
    raw = os.environ.get(var_name, default)
    items = tuple(item.strip() for item in raw.split(",") if item.strip())
    return tuple(item.lower() for item in items) if lower else items


# SLURM memory size suffixes in MB, a bare number is MB
SLURM_MEM_UNITS_MB = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}


def slurm_mem_MB(mem: str) -> int:
    """MB of a SLURM memory size like "48G", "64000M" or "386G", raising on anything else."""
    unit = mem[-1].upper() if mem and mem[-1].isalpha() else "M"
    number = mem[:-1] if mem and mem[-1].isalpha() else mem
    if unit not in SLURM_MEM_UNITS_MB or not number.replace(".", "", 1).isdigit():
        raise ValueError(f"Not a SLURM memory size: {mem!r}")
    return int(float(number) * SLURM_MEM_UNITS_MB[unit])


# Main config for environment variables
class StudyConfig(BaseModel):
    # Shared between all studies
    study_name: Identity[str]
    dataset_paths: dict[str, Path] = Field(default_factory=env_dataset_paths)
    target_device: str

    # Cheap whole-study runs, see DEBUG_MAX_SOURCE_SHOTS
    debug: bool = False
    # Keep only the most recent shots of each device, every shot when None (get_ds logs a truncation)
    max_ds_size: int | None = None
    hyperparam_sweeps: int = 200
    max_epochs: int = 1000
    epochs_per_val: int = 20
    patience: int = 6  # epochs_per_val * patience = max epochs without improvement before stopping
    # Epoch cap for hyperparameter sweep trials and agent jobs
    # Trials are wall-clock limited to train_wall_budget_s and stop early via patience
    # Capped below max_epochs so sweep trials stay cheap.
    # The swept transition_frac is a fraction of each run's own step budget,
    # so a tuned schedule stretches to the production horizon, validated only over this shorter one
    hyperparam_max_epochs: int = 240

    # Environment-specific orchestration settings
    partition: Orchestration[str | None] = Field(default_factory=lambda: os.environ.get("PTPS_PARTITION"))
    # Partition for CPU-only analysis jobs (per-case metrics and case reports).
    # These need no GPU, so point this at a CPU partition. Falls back to `partition`
    analysis_partition: Orchestration[str | None] = Field(default_factory=lambda: os.environ.get("PTPS_ANALYSIS_PARTITION"))
    # SLURM walltime request for analysis jobs, sbatch --time format
    analysis_time_limit: Orchestration[str] = Field(default_factory=lambda: os.environ.get("PTPS_ANALYSIS_TIME_LIMIT", "07:59:59"))
    # Cap on this study's concurrent (running + pending) analysis jobs, so a study
    # with hundreds of cases doesn't flood the queue with pending jobs at once
    max_analysis_jobs: Orchestration[int] = Field(default_factory=lambda: int(os.environ.get("PTPS_MAX_ANALYSIS_JOBS", "20")))
    # SLURM walltime request for training and agent jobs, sbatch --time format
    train_time_limit: Orchestration[str] = Field(default_factory=lambda: os.environ.get("PTPS_TRAIN_TIME_LIMIT", "07:59:59"))
    # In-job wall-clock training budget in seconds. Set below the SLURM limit so the
    # trainer can save the latest checkpoint and exit cleanly, then a resubmitted job
    # resumes from that checkpoint. 27000 s = 7.5 h
    train_wall_budget_s: Orchestration[int] = Field(default_factory=lambda: int(os.environ.get("PTPS_TRAIN_WALL_BUDGET_S", "27000")))
    buffer_gpus: Orchestration[int | None] = Field(default_factory=lambda: int(os.environ.get("PTPS_BUFFER_GPUS", "12")))
    # SLURM gres GPU type names training and agent jobs may land on.
    # The default keeps float64 TORAX training off cards with slow fp64 pipelines (l40s, a40, l4, rtx_pro_6000).
    # Empty disables the exclusion.
    gpu_types: Orchestration[tuple[str, ...]] = Field(default_factory=lambda: _env_list("PTPS_GPU_TYPES", "a100,h100,h200", lower=True))
    # SLURM node names GPU jobs must never land on, e.g. a node whose GPU faults every job at startup.
    # Unlike the gpu_types exclusion it applies to every gres request, typed fallbacks included.
    # sbatch has no SBATCH_EXCLUDE environment variable, so it reaches the job as an #SBATCH --exclude line.
    # Empty disables it.
    exclude_nodes: Orchestration[tuple[str, ...]] = Field(default_factory=lambda: _env_list("PTPS_EXCLUDE_NODES"))
    # Overflow partitions for GPU jobs once `partition` has no idle GPUs beyond buffer_gpus, tried in order.
    # Jobs there can be preempted at any time, so training relies on resume-from-checkpoint.
    # Submissions per partition are capped at its per-user GPU allowance (QOS MaxTRESPU gres/gpu).
    # Empty disables spillover.
    spillover_partitions: Orchestration[tuple[str, ...]] = Field(default_factory=lambda: _env_list("PTPS_SPILLOVER_PARTITIONS"))
    # Model types whose training and sweep agent jobs may also run on the CPU partitions.
    # Prereq pseudo types count by their own name (p_oh, power_balance, profile).
    # A listed type takes a primary GPU only after the study's GPU-only cases and while
    # more than cpu_capable_buffer_gpus are idle, else the first CPU partition with room.
    # No env var, the valid names differ per study.
    cpu_model_types: Orchestration[tuple[str, ...]] = ()
    # Primary CPU partition for cpu_model_types, then the fallbacks in order
    cpu_partition: Orchestration[str | None] = Field(default_factory=lambda: os.environ.get("PTPS_CPU_PARTITION"))
    cpu_fallback_partitions: Orchestration[tuple[str, ...]] = Field(default_factory=lambda: _env_list("PTPS_CPU_FALLBACK_PARTITIONS"))
    # CPUs per CPU training or agent job.
    # XLA's CPU thread pool follows the job's cgroup, and the sequential rollouts barely speed up past 4.
    cpu_train_cpus: Orchestration[int] = Field(default_factory=lambda: int(os.environ.get("PTPS_CPU_TRAIN_CPUS", "4")), ge=1)
    # Idle primary GPUs a CPU-capable case leaves for GPU-only cases, for this running study or another running in parallel
    cpu_capable_buffer_gpus: Orchestration[int] = Field(
        default_factory=lambda: int(os.environ.get("PTPS_CPU_CAPABLE_BUFFER_GPUS", "0")), ge=0
    )
    # sbatch --mem of every training and agent job, GPU or CPU.
    # A CPU job also holds the device arrays in host RAM, so size it from the CPU peak.
    train_mem: Orchestration[str] = Field(default_factory=lambda: os.environ.get("PTPS_TRAIN_MEM", "120G"))
    # Ceiling on this user's running + pending jobs across all partitions.
    # The default is the mit_preemptable QOS MaxSubmitPU (448), the tightest limit that applies.
    max_user_jobs: Orchestration[int] = Field(default_factory=lambda: int(os.environ.get("PTPS_MAX_USER_JOBS", "448")))
    # Spillover submissions stop at max_user_jobs minus this headroom, leaving slack for analysis and interactive jobs
    spillover_job_headroom: Orchestration[int] = Field(default_factory=lambda: int(os.environ.get("PTPS_SPILLOVER_JOB_HEADROOM", "10")))
    wandb_entity: Orchestration[str | None] = Field(default_factory=lambda: os.environ.get("PTPS_WANDB_ENTITY"))
    # Scratch directory for trajectory-optimization intermediate results
    scratch_dir: Orchestration[Path | None] = None

    # make everything in the config completely immutable, including the nested dataset_paths dict
    # extra="forbid" so a mistyped field name raises instead of being silently ignored and replaced by the default
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _freeze_paths(self):
        object.__setattr__(self, "dataset_paths", MappingProxyType(self.dataset_paths))
        return self

    @model_validator(mode="after")
    def _check_cpu_partitions(self):
        """CPU training needs a CPU partition, and no partition may serve both devices, since submission picks the device by partition."""
        if (self.cpu_model_types or self.cpu_fallback_partitions) and not self.cpu_partition:
            raise ValueError("cpu_model_types and cpu_fallback_partitions need a cpu_partition")
        # Unset partitions are None, a config rebuilt from its lock carries no orchestration fields at all
        gpu_partitions = {p for p in (self.partition, *self.spillover_partitions) if p}
        shared = sorted(p for p in (self.cpu_partition, *self.cpu_fallback_partitions) if p and p in gpu_partitions)
        if shared:
            raise ValueError(f"Partitions {shared} are both CPU and GPU partitions")
        slurm_mem_MB(self.train_mem)
        return self

    @model_validator(mode="after")
    def _apply_debug_limits(self):
        """Cap the epochs, the validation interval and the sweep size of a debug run (get_ds caps the source shots)."""
        if self.debug:
            object.__setattr__(self, "max_epochs", min(self.max_epochs, DEBUG_MAX_EPOCHS))
            object.__setattr__(self, "hyperparam_max_epochs", min(self.hyperparam_max_epochs, DEBUG_MAX_EPOCHS))
            object.__setattr__(self, "epochs_per_val", min(self.epochs_per_val, DEBUG_EPOCHS_PER_VAL))
            object.__setattr__(self, "hyperparam_sweeps", min(self.hyperparam_sweeps, DEBUG_HYPERPARAM_SWEEPS))
        return self

    # dataset_paths is stored as a MappingProxyType for immutability, but the field is typed
    # dict[str, Path], so tell the serializer to emit a plain dict and avoid a Pydantic warning
    @field_serializer("dataset_paths")
    def _serialize_dataset_paths(self, v):
        return dict(v)

    @classmethod
    def field_role(cls, name: str) -> FieldRole:
        """The role a field declares with its type, LOCKED when it declares none."""
        for marker in cls.model_fields[name].metadata:
            if isinstance(marker, CaseAxis):
                return FieldRole.CASE_AXIS
            if isinstance(marker, FieldRole):
                return marker
        return FieldRole.LOCKED

    @classmethod
    def fields_with_role(cls, *roles: FieldRole) -> tuple[str, ...]:
        return tuple(name for name in cls.model_fields if cls.field_role(name) in roles)

    @classmethod
    def case_axes(cls) -> dict[str, str]:
        """Case field name -> the config field holding its values, one entry per case-grid axis."""
        axes = {}
        for name, info in cls.model_fields.items():
            for marker in info.metadata:
                if isinstance(marker, CaseAxis):
                    axes[marker.case_field] = name
        return axes

    def differing_fields(self, other: "StudyConfig", *roles: FieldRole) -> list[str]:
        """Names of the fields with one of roles whose values differ between the two configs."""
        return [name for name in self.fields_with_role(*roles) if getattr(self, name) != getattr(other, name)]

    @classmethod
    def toml_kwargs(cls, path: Path | str) -> dict:
        """Constructor kwargs of a study TOML laid out by role (see the module docstring).

        A field outside the table of its role raises, naming the right table.
        PTPS_DATASET_PATHS provides default device paths, explicit [datasets] paths win.
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)
        datasets = data.pop("datasets", {})
        target = datasets.pop("target", None)
        tables = {table: data.pop(table, {}) for table in ROLE_TABLES.values()}
        kwargs = {}
        for table, entries in [(None, data), *tables.items()]:
            for name, value in entries.items():
                if name not in cls.model_fields:
                    raise ValueError(f"Unknown config field {name} {_toml_location(table)} of {path}")
                expected_table = ROLE_TABLES.get(cls.field_role(name))
                if expected_table != table:
                    raise ValueError(
                        f"Config field {name} is {_toml_location(table)} of {path}, but it belongs {_toml_location(expected_table)}"
                    )
                kwargs[name] = value
        dataset_paths = env_dataset_paths() | {k: Path(v) for k, v in datasets.items()}
        return kwargs | {"dataset_paths": dataset_paths, "target_device": target}

    @classmethod
    def from_toml(cls, path: Path | str) -> Self:
        kwargs = cls.toml_kwargs(path)
        return cls(**kwargs)

    @property
    def ds_source_to_idx(self) -> dict[str, int]:
        """Stable integer index per device, sorted alphabetically for reproducibility."""
        return {k: i for i, k in enumerate(sorted(self.dataset_paths.keys()))}

    def toml_data(self, roles: tuple[FieldRole, ...] = tuple(FieldRole)) -> dict:
        """The TOML tables save writes and from_toml reads back, laid out by role.

        Only fields with one of roles are written, None fields are left out.
        """
        datasets = {k: str(v) for k, v in self.dataset_paths.items()}
        if self.target_device is not None:
            datasets["target"] = self.target_device
        data: dict = {}
        tables: dict = {table: {} for role, table in ROLE_TABLES.items() if role in roles}
        for name, value in self.model_dump().items():
            role = self.field_role(name)
            if name in ("dataset_paths", "target_device") or value is None or role not in roles:
                continue
            table = ROLE_TABLES.get(role)
            entries = tables[table] if table else data
            entries[name] = str(value) if isinstance(value, Path) else value
        return data | tables | {"datasets": datasets}

    def save(self, path: Path):
        data = self.toml_data()
        with open(path, "w") as f:
            toml.dump(data, f)


class _ConfigProxy:
    """Class that enables delayed instantiation of a global StudyConfig,
    so we can load it from a file at runtime instead of having it be hardcoded at import time.

    Also ensures the global config can only be created once, to prevent accidental bugs from mutable global state.
    """

    _cfg: "StudyConfig | None" = None
    initialized = False

    def __setattr__(self, name, value):
        raise AttributeError("Config is immutable after being set. Multiple calls to load_config() are not allowed.")

    def __getattr__(self, name: str):
        if _ConfigProxy._cfg is None:
            if name == "initialized":
                return False
            raise RuntimeError(f"Config not loaded. Call load_config() before accessing config.{name}")
        return getattr(_ConfigProxy._cfg, name)

    def get_subclass(self) -> type[StudyConfig]:
        """Return the actual loaded StudyConfig (or subclass) instance, bypassing the proxy.

        Needed when code must know the concrete subclass - e.g. to reconstruct it
        correctly (with its extra fields) in a subprocess, since `load_config(Path)`
        only knows how to build the base `StudyConfig`.
        """
        if _ConfigProxy._cfg is None:
            raise RuntimeError("Config not loaded. Call load_config() first.")
        return type(_ConfigProxy._cfg)

    def get_instance(self) -> StudyConfig:
        """The loaded config object itself, for code that stores or compares it (e.g. the config lock)."""
        if _ConfigProxy._cfg is None:
            raise RuntimeError("Config not loaded. Call load_config() first.")
        return _ConfigProxy._cfg


config = _ConfigProxy()


def load_config(cfg: "StudyConfig | Path | str") -> "StudyConfig":
    """Load study config from a StudyConfig object or a path to a TOML file."""
    if _ConfigProxy.initialized:
        raise RuntimeError("Config already loaded. Multiple calls to load_config() are not allowed.")
    if isinstance(cfg, (Path, str)):
        cfg = StudyConfig.from_toml(cfg)
    # Only a loaded config counts, a failed TOML parse leaves the proxy loadable
    _ConfigProxy._cfg = cfg
    _ConfigProxy.initialized = True
    return cfg


def reset_config() -> None:
    """Clear the global config so load_config can be called again.

    FOR TESTS ONLY. Production code relies on the config being immutable and
    loaded exactly once per process. The test conftest calls this around every
    test so each test starts with a clean slate and loads its own config.
    """
    _ConfigProxy._cfg = None
    _ConfigProxy.initialized = False
    logger.critical("GLOBAL CONFIG RESET!!! THIS SHOULD ONLY HAPPEN IN TESTS. DO NOT CALL THIS IN PRODUCTION CODE.")
