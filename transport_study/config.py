"""
Load and parse configuration files for the project.
This is where we set global variables from env vars or config files
"""

import json
import os
import tomllib
from pathlib import Path
from types import MappingProxyType

import jax
import toml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

# 80/20 between train/val
# 80/20 between train+val/test
TRAIN_VAL_SPLIT = (0.8, 0.2)
TRAIN_VAL_TEST_SPLIT = (0.64, 0.16, 0.2)


def env_dataset_paths() -> dict[str, Path]:
    """Parse PTPS_DATASET_PATHS, a JSON dict of device -> dataset path,
    e.g. PTPS_DATASET_PATHS='{"cmod": "/path/to/ds.zarr"}' -> {"cmod": Path(...)}
    """
    return {k: Path(v) for k, v in json.loads(os.environ.get("PTPS_DATASET_PATHS", "{}")).items()}


def _env_spillover_partitions() -> tuple[str, ...]:
    """Parse PTPS_SPILLOVER_PARTITIONS, a comma-separated ordered list of
    partitions, e.g. "mit_preemptable,mit_normal_gpu"
    """
    raw = os.environ.get("PTPS_SPILLOVER_PARTITIONS", "")
    return tuple(p.strip() for p in raw.split(",") if p.strip())


# Main config for environment variables
class StudyConfig(BaseModel):
    # Shared between all studies
    study_name: str
    dataset_paths: dict[str, Path] = Field(default_factory=env_dataset_paths)
    target_device: str

    # Debugging and dev stuff
    debug: bool = True
    dry_run: bool = False
    max_ds_size: int = 1000
    hyperparam_sweeps: int = 1000
    max_epochs: int = 1000
    epochs_per_val: int = 20
    patience: int = 4  # epochs_per_val * patience = max epochs without improvement before stopping
    # Epoch cap for hyperparameter sweep trials
    # Trials are wall-clock limited to train_wall_budget_s and stop early via patience
    # Matching max_epochs keeps tuned LR schedules consistent between sweep and final training runs
    hyperparam_max_epochs: int = 1000

    # Environment-specific orchestration settings
    partition: str | None = Field(default_factory=lambda: os.environ.get("PTPS_PARTITION"))
    # Partition for CPU-only analysis jobs (per-case metrics and case reports).
    # These need no GPU, so point this at a CPU partition. Falls back to `partition`
    analysis_partition: str | None = Field(default_factory=lambda: os.environ.get("PTPS_ANALYSIS_PARTITION"))
    # SLURM walltime request for analysis jobs, sbatch --time format
    analysis_time_limit: str = Field(default_factory=lambda: os.environ.get("PTPS_ANALYSIS_TIME_LIMIT", "07:59:59"))
    # Cap on this study's concurrent (running + pending) analysis jobs, so a study
    # with hundreds of cases doesn't flood the queue with pending jobs at once
    max_analysis_jobs: int = Field(default_factory=lambda: int(os.environ.get("PTPS_MAX_ANALYSIS_JOBS", "20")))
    # SLURM walltime request for training and agent jobs, sbatch --time format
    train_time_limit: str = Field(default_factory=lambda: os.environ.get("PTPS_TRAIN_TIME_LIMIT", "07:59:59"))
    # In-job wall-clock training budget in seconds. Set below the SLURM limit so the
    # trainer can save the latest checkpoint and exit cleanly, then a resubmitted job
    # resumes from that checkpoint. 27000 s = 7.5 h
    train_wall_budget_s: int = Field(default_factory=lambda: int(os.environ.get("PTPS_TRAIN_WALL_BUDGET_S", "27000")))
    buffer_gpus: int | None = Field(default_factory=lambda: int(os.environ.get("PTPS_BUFFER_GPUS", "12")))
    # Overflow partitions for GPU jobs once `partition` has no idle GPUs beyond
    # buffer_gpus, tried in order. Jobs submitted there can be preempted
    # (requeued) at any time, so training relies on resume-from-checkpoint.
    # Submissions per partition are capped at its per-user GPU allowance
    # (QOS MaxTRESPU gres/gpu) so jobs don't pile up pending behind a QOS cap.
    # Empty disables spillover
    spillover_partitions: tuple[str, ...] = Field(default_factory=_env_spillover_partitions)
    # Ceiling on this user's total running + pending jobs across all partitions.
    # Default matches the mit_preemptable QOS MaxSubmitPU (448), the tightest of
    # the limits that apply (association MaxSubmit is 500)
    max_user_jobs: int = Field(default_factory=lambda: int(os.environ.get("PTPS_MAX_USER_JOBS", "448")))
    # Spillover submissions stop once total jobs reach max_user_jobs - this headroom,
    # leaving slack for analysis jobs and interactive work
    spillover_job_headroom: int = Field(default_factory=lambda: int(os.environ.get("PTPS_SPILLOVER_JOB_HEADROOM", "10")))
    wandb_entity: str | None = Field(default_factory=lambda: os.environ.get("PTPS_WANDB_ENTITY"))
    # Scratch directory for trajectory-optimization intermediate results
    scratch_dir: Path | None = None

    # make everything in the config completely immutable, including the nested dataset_paths dict
    # extra="forbid" so a mistyped field name raises instead of being silently ignored and replaced by the default
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def _freeze_paths(self):
        object.__setattr__(self, "dataset_paths", MappingProxyType(self.dataset_paths))
        return self

    # dataset_paths is stored as a MappingProxyType for immutability, but the field is typed
    # dict[str, Path], so tell the serializer to emit a plain dict and avoid a Pydantic warning
    @field_serializer("dataset_paths")
    def _serialize_dataset_paths(self, v):
        return dict(v)

    @classmethod
    def from_toml(cls, path: Path | str) -> "StudyConfig":
        path = Path(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)
        datasets = data.pop("datasets", {})
        target = datasets.pop("target", None)
        # Env vars provide defaults, explicit TOML paths win
        return cls(
            **data,
            dataset_paths=env_dataset_paths() | {k: Path(v) for k, v in datasets.items()},
            target_device=target,
        )

    @property
    def ds_source_to_idx(self) -> dict[str, int]:
        """Stable integer index per device, sorted alphabetically for reproducibility."""
        return {k: i for i, k in enumerate(sorted(self.dataset_paths.keys()))}

    def is_compatible(self, cfg: "StudyConfig") -> bool:
        """Check if two configs are compatible for running the same study (e.g. dataset paths and target device must match)."""
        return self.dataset_paths == cfg.dataset_paths and self.target_device == cfg.target_device

    def save(self, path: Path):
        raw = self.model_dump()
        datasets = {k: str(v) for k, v in self.dataset_paths.items()}
        if self.target_device is not None:
            datasets["target"] = self.target_device
        data = {}
        for k, v in raw.items():
            if k in ("dataset_paths", "target_device"):
                continue
            if v is None:
                continue
            if isinstance(v, Path):
                data[k] = str(v)
            else:
                data[k] = v
        data["datasets"] = datasets
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


config = _ConfigProxy()


def load_config(cfg: "StudyConfig | Path | str") -> "StudyConfig":
    """Load study config from a StudyConfig object or a path to a TOML file."""
    if _ConfigProxy.initialized:
        raise RuntimeError("Config already loaded. Multiple calls to load_config() are not allowed.")
    _ConfigProxy.initialized = True
    if isinstance(cfg, (Path, str)):
        cfg = StudyConfig.from_toml(cfg)
    _ConfigProxy._cfg = cfg
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


if jax.devices()[0].platform not in ["gpu", "tpu", "cuda"]:
    logger.warning("JAX could not find GPU/TPU/CUDA, is your environment set correctly?")
