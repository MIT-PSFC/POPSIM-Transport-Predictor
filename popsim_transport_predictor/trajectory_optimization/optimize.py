import os

import fire
from loguru import logger
from popsim import PACKAGE_ROOT
from popsim.ml.train_config import TrainConfig

from popsim_transport_predictor.modules.profile_trajectory.config import (
    PROFILE_TRAJECTORY_OPTIMIZER_CONFIG,
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

SHAPE_TIMES = [
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
]  # 8 times to program the shape in by hand... should be fine right?


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
            f"Input variable {input_var} has the following statistics during the trajectory portion of the shots:"
        )
        for stat_name, stat_value in stats.items():
            logger.info(f"    {stat_name}: {stat_value:.5f}")

    controllable_input_ranges = get_controllable_input_ranges(
        ds, ["Ip_MA", "B0", "ne20_edge", "beta"]
    )

    for input_var, error in controllable_input_ranges.items():
        logger.info(
            f"Controllable input variable {input_var} has a characteristic error of {error:.5f} during the trajectory portion of the shots"
        )


######################
# Set up the configs #
######################
def setup_optimization_config(
    ds_path: str,
    debug: bool | None = False,
) -> TrainConfig:
    """Set up the training config for trajectory optimization.
    Based on the dataset, we determine the allowable ranges for optimization variables and update the config accordingly.

    Args:
        ds_path (str): Path to the dataset.
        debug (bool, optional): Whether to enable debug mode, reducing dataset size to at most 50 shots.

    Returns:
        TrainConfig: The training config for trajectory optimization.
    """

    max_epochs = 2 if debug else MAX_EPOCHS
    epochs_per_val = 1 if debug else EPOCHS_PER_VAL

    # Informed the dataset characterization and Jayson Barr TODO(ZanderKeith) make sure these are ok
    control_input_ranges = {
        "R0": (1.77, 1.82),  # Major radius [m]
        "a_minor": (0.58, 0.6),  # Minor radius [m]
        "kappa": (1.89, 1.97),  # Elongation
        "delta_top": (0.5, 0.91),  # Upper triangularity
        "delta_bottom": (0.7, 0.91),  # Lower triangularity
    }

    base_config = TrainConfig.load(PROFILE_TRAJECTORY_OPTIMIZER_CONFIG)

    config = base_config.model_copy(
        update={
            "max_epochs": max_epochs,
            "epochs_per_val": epochs_per_val,
            "dataloader_config": {
                **base_config.dataloader_config,
                "ds_path": ds_path,
                "debug": debug,
            },
            "model_init_config": {
                **base_config.model_init_config,
                "input_ranges": control_input_ranges,
            },
        }
    )

    return config


###############################################
# Train the model and optimize the trajectory #
###############################################
def train_profile_predictor(ds_path: str, debug: bool | None = False) -> None:
    """Train the profile predictor model for trajectory optimization.

    Args:
        ds_path (str): Path to the dataset.
        debug (bool, optional): Whether to enable debug mode, reducing dataset size to at most 10 shots.
    """
    _base_config = setup_optimization_config(ds_path, debug=debug)


def run_trajectory_optimization(
    ds_path: str,
    clean: bool | None = False,
    debug: bool | None = False,
):
    """Run trajectory optimization.

    Args:
        ds_path (str): Path to the dataset.
        clean (bool, optional): Whether to clean the checkpoint directory before training.
        debug (bool, optional): Whether to enable debug mode, reducing dataset size to only the base shots.
    """

    config = setup_optimization_config(ds_path, debug=debug)

    # Set up the checkpoint directory
    checkpoint_dir = os.path.join(
        CHECKPOINT_DIR_BASE,
        f"trajectory_optimization_{config.submodule_configs['trajectory_predictor'].model_name}",
    )
    os.makedirs(checkpoint_dir, exist_ok=True)


if __name__ == "__main__":
    # Usage: python popsim_transport_predictor/trajectory_optimization/optimize.py <command> [--options]
    fire.Fire(
        {
            "characterize_dataset": characterize_dataset,
            "setup_optimization_config": setup_optimization_config,
            "run_trajectory_optimization": run_trajectory_optimization,
        }
    )
