"""
Load and parse configuration files for the project.
This is where we set global variables from env vars or config files
"""

import os

from dynaconf import Dynaconf
from popsim.data import get_path_to_ml_data_dump, get_path_to_ml_data_scratch

# load all configs from environment variables
config = Dynaconf(
    envvar_prefix="PTPS",
    load_dotenv=True,
)

DATA_DUMP_DIR = os.path.join(
    get_path_to_ml_data_dump(), "popsim_studies", config.study_name
)
DATA_SCRATCH_DIR = os.path.join(
    get_path_to_ml_data_scratch(), "popsim_studies", config.study_name
)
