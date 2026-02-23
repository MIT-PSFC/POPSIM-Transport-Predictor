"""
Load and parse configuration files for the project.
This is where we set global variables from env vars or config files
"""

import getpass
import os
from pathlib import Path

from dynaconf import Dynaconf

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

# Main config for environment variables
config = Dynaconf(
    envvar_prefix="PTPS",
    load_dotenv=True,
)

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

DATA_DUMP_DIR = os.path.join(
    get_path_to_ml_data_dump(), "POPSIM/popsim_studies", config.study_name
)
DATA_SCRATCH_DIR = os.path.join(
    get_path_to_ml_data_scratch(), "POPSIM/popsim_studies", config.study_name
)
