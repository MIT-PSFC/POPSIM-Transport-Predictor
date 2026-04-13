"""
Load and parse configuration files for the project.
This is where we set global variables from env vars or config files
"""

import os
from pathlib import Path

from dynaconf import Dynaconf
from popsim.data import get_path_to_ml_data_dump, get_path_to_ml_data_scratch
from pydantic_settings import BaseSettings, SettingsConfigDict

from transport_study import PACKAGE_ROOT

# 80/20 between train/val
# 80/20 between train+val/test
TRAIN_VAL_SPLIT = (0.8, 0.2)
TRAIN_VAL_TEST_SPLIT = (0.64, 0.16, 0.2)


# Main config for environment variables
class StudyConfig(BaseSettings):
    """Configuration for dataset paths."""

    study_name: str = "transport_study"

    debug: bool = True  # Debug does everything but with reduced scope (less data, fewer epochs, etc.)
    dry_run: bool = False  # Dry run skips training and evaluation and just runs the orchestration logic to make sure everything is set up correctly
    hp_test_set_size: int = 65
    max_ds_size: int = 1000

    partition: str | None = None
    buffer_gpus: int = 12
    hyperparam_sweeps: int = 1000
    max_epochs: int = 1000
    epochs_per_val: int = 20
    patience: int = 4  # 80 epochs without improvement, stop
    wandb_entity: str | None = None

    scratch_dir: Path | None = None  # Used for predict-first temp files
    ds_target: str | None = (
        None  # PTPS_DS_TARGET=DEVICE - which device is the HP target
    )

    model_config = SettingsConfigDict(
        env_prefix="PTPS_",  # Datasets: PTPS_DS_DEVICE1=/path1.nc, PTPS_DS_DEVICE2=/path2.nc, etc.
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",  # Captures all PTPS_DS_* vars dynamically
    )

    @property
    def dataset_paths(self) -> dict[str, Path]:
        """All PTPS_DS_* vars except PTPS_DS_TARGET, keyed by lowercased device name."""
        return {
            k.removeprefix("ds_"): Path(v)
            for k, v in self.model_extra.items()
            if k.startswith("ds_")
        }

    @property
    def target_device(self) -> str | None:
        if self.ds_target is None:
            return None
        return self.ds_target.lower()


config = StudyConfig()

if config.debug:
    config.max_epochs = 2
    config.epochs_per_val = 1
    config.hyperparam_sweeps = 2
    config.hp_test_set_size = 2

# Device-specific configs loaded separately to avoid namespace collisions
config.d3d = Dynaconf(
    settings_files=[os.path.join(PACKAGE_ROOT, "datasets", "d3d", "config.toml")]
)
config.cmod = Dynaconf(
    settings_files=[os.path.join(PACKAGE_ROOT, "datasets", "cmod", "config.toml")]
)
config.tcv = Dynaconf(
    settings_files=[os.path.join(PACKAGE_ROOT, "datasets", "tcv", "config.toml")]
)

DATA_DUMP_DIR = os.path.join(get_path_to_ml_data_dump(), "POPSIM/popsim_studies")
DATA_SCRATCH_DIR = os.path.join(get_path_to_ml_data_scratch(), "POPSIM/popsim_studies")
