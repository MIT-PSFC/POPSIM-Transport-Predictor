import os

import fire
from loguru import logger
from popsim import PACKAGE_ROOT
from popsim.ml.train_config import TrainConfig
from popsim.modules.transport_predictor.train_configs import (
    BASE_CONFIG,
    update_submodule_configs,
)

from popsim_transport_predictor.modules.profile_trajectory.data import get_ds
from popsim_transport_predictor.trajectory_optimization.setup import (
    get_controllable_input_ranges,
    get_trajectory_input_ranges,
)

CHECKPOINT_DIR_BASE = os.path.join(
    PACKAGE_ROOT, "checkpoints", "trajectory_optimization"
)
MAX_EPOCHS = 800
EPOCHS_PER_VAL = 20


############################################################################################################
# Scope the dataset to determine the distribution of targets / allowable ranges for optimization variables #
############################################################################################################
def characterize_dataset(ds_path: str, debug: bool = False) -> None:
    """Load the dataset and print out some basic statistics to help scope the optimization problem.

    Args:
        ds_path (str): Path to the dataset.
        debug (bool, optional): Whether to enable debug mode, reducing dataset size to a select few shots.
    """

    ds, _ = get_ds(ds_path, debug=debug)

    input_ranges = get_trajectory_input_ranges(
        ds, ["R0", "a_minor", "kappa", "delta_top", "delta_bottom"]
    )

    for input_var, stats in input_ranges.items():
        logger.info(
            f"Input variable {input_var} has the following statistics during the trajectory portion of the shots: {stats}"
        )

    controllable_input_ranges = get_controllable_input_ranges(
        ds, ["Ip_MA", "B0", "ne20_edge", "beta"]
    )

    for input_var, error in controllable_input_ranges.items():
        logger.info(
            f"Controllable input variable {input_var} has a characteristic error of {error} during the trajectory portion of the shots"
        )


#####################
# Set up the config #
#####################
def setup_optimization_config(
    ds_path: str,
    debug: bool | None = False,
) -> TrainConfig:
    """Set up the training config for trajectory optimization.

    Args:
        ds_path (str): Path to the dataset.
        debug (bool, optional): Whether to enable debug mode, reducing dataset size to at most 50 shots.

    Returns:
        TrainConfig: The training config for trajectory optimization.
    """

    # Get the dataset and determine the episode coordinate
    ds, _episode_coord = get_ds(ds_path, debug=debug)

    # Update the base config with submodule configs that are consistent with the dataset
    config = update_submodule_configs(BASE_CONFIG, ds)

    return config


if __name__ == "__main__":
    # Usage: python popsim_transport_predictor/trajectory_optimization/optimize.py <command> [--options]
    fire.Fire(
        {
            "characterize_dataset": characterize_dataset,
            "setup_optimization_config": setup_optimization_config,
        }
    )
