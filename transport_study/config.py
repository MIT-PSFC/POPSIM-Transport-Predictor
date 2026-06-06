"""
Load and parse configuration files for the project.
This is where we set global variables from env vars or config files
"""

import tomllib
from pathlib import Path
from types import MappingProxyType

import toml
from dynaconf import Dynaconf
from pydantic import BaseModel, ConfigDict, model_validator

from transport_study import PACKAGE_ROOT

# 80/20 between train/val
# 80/20 between train+val/test
TRAIN_VAL_SPLIT = (0.8, 0.2)
TRAIN_VAL_TEST_SPLIT = (0.64, 0.16, 0.2)


# Main config for environment variables
class StudyConfig(BaseModel):
    # Shared between all studies
    study_name: str
    dataset_paths: dict[str, Path] = {}
    target_device: str | None = None

    # Debugging and dev stuff
    debug: bool = True
    dry_run: bool = False
    max_ds_size: int = 1000
    hyperparam_sweeps: int = 1000
    max_epochs: int = 1000
    epochs_per_val: int = 20
    patience: int = 4  # epochs_per_val * patience = max epochs without improvement before stopping

    # Environment-specific orchestration settings
    partition: str | None = None
    buffer_gpus: int = 12
    wandb_entity: str | None = None
    scratch_dir: Path | None = (
        None  # TODO(ZanderKeith): Only used for intermediate results from trajectory optimization, can be put in that study config instead
    )

    # make everything in the config completely immutable, including the nested dataset_paths dict
    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _freeze_paths(self):
        object.__setattr__(self, "dataset_paths", MappingProxyType(self.dataset_paths))
        return self

    @classmethod
    def from_toml(cls, path: Path) -> "StudyConfig":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        datasets = data.pop("datasets", {})
        target = datasets.pop("target", None)
        return cls(
            **data,
            dataset_paths={k: Path(v) for k, v in datasets.items()},
            target_device=target,
        )

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

    # File-based device configs - always available regardless of study config
    cmod = Dynaconf(settings_files=[Path(PACKAGE_ROOT) / "datasets/cmod/config.toml"])
    d3d = Dynaconf(settings_files=[Path(PACKAGE_ROOT) / "datasets/d3d/config.toml"])
    mast = Dynaconf(settings_files=[Path(PACKAGE_ROOT) / "datasets/mast/config.toml"])
    tcv = Dynaconf(settings_files=[Path(PACKAGE_ROOT) / "datasets/tcv/config.toml"])

    def __setattr__(self, name, value):
        raise AttributeError("Config is immutable after being set. Multiple calls to load_config() are not allowed.")

    def __getattr__(self, name: str):
        if _ConfigProxy._cfg is None:
            if name == "initialized":
                return False
            raise RuntimeError(f"Config not loaded. Call load_config() before accessing config.{name}")
        return getattr(_ConfigProxy._cfg, name)


config = _ConfigProxy()


def load_config(cfg: "StudyConfig | Path") -> "StudyConfig":
    """Load study config from a StudyConfig object or a path to a TOML file."""
    if _ConfigProxy.initialized:
        raise RuntimeError("Config already loaded. Multiple calls to load_config() are not allowed.")
    _ConfigProxy.initialized = True
    if isinstance(cfg, Path):
        cfg = StudyConfig.from_toml(cfg)
    _ConfigProxy._cfg = cfg
    return cfg
