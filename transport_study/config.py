"""
Load and parse configuration files for the project.
This is where we set global variables from env vars or config files
"""

import getpass
import os
from pathlib import Path

from dynaconf import Dynaconf
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from popsim.data import get_path_to_ml_data_dump, get_path_to_ml_data_scratch
except ImportError:
    # This is necessary for omega when numpy 1.0 .venv is in use
    def get_path_to_ml_data_dump():
        return Path("/fusion/projects/disruption_warning/data/popsim/")

    def get_path_to_ml_data_scratch():
        return Path(f"/cscratch/{getpass.getuser()}/")


# Get package root directory
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

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

    scratch_dir: Path | None = (
        None  # Datasets get pared down and copied to here before study gets run
    )
    cmod_dataset_path: Path | None = None
    tcv_dataset_path: Path | None = None
    d3d_lp_dataset_path: Path | None = None
    d3d_hp_dataset_path: Path | None = None

    d3d: dict = {}
    cmod: dict = {}
    tcv: dict = {}

    model_config = SettingsConfigDict(
        env_prefix="PTPS_",  # Put in .env like PTPS_CMOD_DATASET_PATH
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


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
