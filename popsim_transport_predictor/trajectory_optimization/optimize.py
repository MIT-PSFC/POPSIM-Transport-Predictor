import os
import shutil

import fire
from loguru import logger
from popsim.ml import Trainer
from popsim.ml.launch import DataLoader, launch_train
from popsim.ml.train_config import TrainConfig
from popsim.modules.transport_predictor.train_configs import update_submodule_configs

from popsim_transport_predictor import PACKAGE_ROOT
from popsim_transport_predictor.modules.profile_trajectory.data import get_ds
from popsim_transport_predictor.modules.profile_trajectory.profile_predictor.train_configs import (
    PROFILE_PREDICTOR_DIRECT_POINTS_CONFIG,
    PROFILE_PREDICTOR_SHAPE_INIT_CONFIG,
)
from popsim_transport_predictor.modules.profile_trajectory.train_configs import (
    PROFILE_TRAJECTORY_OPTIMIZER_CONFIG,
)
from popsim_transport_predictor.trajectory_optimization.setup import (
    get_controllable_input_ranges,
    get_trajectory_input_ranges,
)

CHECKPOINT_DIR_BASE = os.path.join(
    PACKAGE_ROOT, "../checkpoints", "trajectory_optimization"
)
MAX_EPOCHS = 800
EPOCHS_PER_VAL = 20

# TODO(ZanderKeith): Add a config for the number of shape times, see how sensitive the resulting optimization is.
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
        ds, ["gapin", "gapout", "rxpt1", "zxpt1", "rxpt2", "zxpt2"]
    )

    for input_var, stats in input_ranges.items():
        logger.info(
            f"Input variable {input_var} has the following statistics during the trajectory portion of the shots:"
        )
        for stat_name, stat_value in stats.items():
            logger.info(f"    {stat_name}: {stat_value:.5f}")

    controllable_input_ranges = get_controllable_input_ranges(
        ds, ["iptipp_MA", "B0", "beta", "dstdenp"]
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
    model_type: str,
    checkpoint_dir: str | None = None,
    shape_times: list[float] | None = SHAPE_TIMES,
    debug: bool | None = False,
) -> TrainConfig:
    """Set up the training config for trajectory optimization.
    Based on the dataset, we determine the allowable ranges for optimization variables and update the config accordingly.

    Args:
        ds_path (str): Path to the dataset.
        checkpoint_dir (str | None, optional): Path to the checkpoint directory. If None, a default path is used.
        shape_times (list[float] | None, optional): List of shape times to use for the trajectory optimization.
        debug (bool, optional): Whether to enable debug mode, reducing dataset size to at most 50 shots.

    Returns:
        TrainConfig: The training config for trajectory optimization.
    """

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(CHECKPOINT_DIR_BASE, "trajectory_optimizer")

    if model_type == "shape_init":
        profile_predictor_config = TrainConfig.load(PROFILE_PREDICTOR_SHAPE_INIT_CONFIG)
    elif model_type == "direct_points":
        profile_predictor_config = TrainConfig.load(
            PROFILE_PREDICTOR_DIRECT_POINTS_CONFIG
        )
    else:
        raise ValueError(
            f"Invalid model type {model_type}, must be either 'shape_init' or 'direct_points'"
        )

    max_epochs = 2 if debug else MAX_EPOCHS
    epochs_per_val = 1 if debug else EPOCHS_PER_VAL

    # Informed the dataset characterization and Jayson Barr TODO(ZanderKeith) make sure these are ok
    control_input_ranges = {
        "gapin": (0.01, 0.12),  # Inner gap [m]
        "gapout": (0.06, 0.15),  # Outer gap [m]
        "rxpt1": (1.09, 1.29),  # Lower X-point R [m]
        "zxpt1": (-1.51, -1.12),  # Lower X-point Z [m]
        "rxpt2": (1.08, 1.25),  # Upper X-point R [m]
        "zxpt2": (0.9, 1.4),  # Upper X-point Z [m]
    }

    base_config = TrainConfig.load(PROFILE_TRAJECTORY_OPTIMIZER_CONFIG)

    config = base_config.model_copy(
        update={
            "max_epochs": max_epochs,
            "epochs_per_val": epochs_per_val,
            "checkpoint_dir": checkpoint_dir,
            "dataloader_config": {
                **base_config.dataloader_config,
                "ds_path": ds_path,
                "debug": debug,
            },
            "model_init_config": {
                **base_config.model_init_config,
                "input_ranges": control_input_ranges,
                "shape_times": shape_times,
                "submodules": {
                    "profile_predictor": profile_predictor_config,
                },
            },
        }
    )

    # Update all submodule configs to use the same dataloader as the base module
    config = update_submodule_configs(
        config.model_dump(),
        [
            "profile_predictor",
        ],
    )

    return config


def setup_profile_predictor_config(
    ds_path: str,
    model_type: str,
    checkpoint_dir: str | None = None,
    debug: bool | None = False,
) -> TrainConfig:
    """Set up the training config for the profile predictor submodule.

    Args:
        model_type (str): The type of profile predictor model to set up, either "shape_init" or "direct_points".
        checkpoint_dir (str | None, optional): Path to the checkpoint directory. If None, a default path is used.
        debug (bool, optional): Whether to enable debug mode, reducing dataset size to at most 50 shots.

    Returns:
        TrainConfig: The training config for the profile predictor submodule.
    """

    base_config = setup_optimization_config(ds_path, model_type, debug=debug)

    max_epochs = 2 if debug else MAX_EPOCHS
    epochs_per_val = 1 if debug else EPOCHS_PER_VAL

    profile_predictor_config = TrainConfig.load(
        base_config.model_init_config["submodules"]["profile_predictor"]
    )
    profile_predictor_config = profile_predictor_config.model_copy(
        update={
            "max_epochs": max_epochs,
            "epochs_per_val": epochs_per_val,
            "checkpoint_dir": checkpoint_dir,
            "dataloader_config": {
                **profile_predictor_config.dataloader_config,
                "module": "profile_predictor",
                "debug": debug,
            },
        }
    )

    return profile_predictor_config


###############################################
# Train the model and optimize the trajectory #
###############################################
def train_profile_predictor(
    ds_path: str,
    model_type: str,
    checkpoint_dir: str | None = None,
    debug: bool | None = False,
    clean: bool | None = False,
) -> tuple[Trainer, DataLoader]:
    """Train the profile predictor model for trajectory optimization.

    Args:
        ds_path (str): Path to the dataset.
        model_type (str): The type of profile predictor model to train, either "shape_init" or "direct_points".
        checkpoint_dir (str | None, optional): Path to the checkpoint directory. If None, a default path is used.
        debug (bool, optional): Whether to enable debug mode, reducing dataset size to at most 10 shots.
        clean (bool, optional): Whether to clean the checkpoint directory before training.
    """

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(CHECKPOINT_DIR_BASE, "profile_predictor")

    profile_predictor_config = setup_profile_predictor_config(
        ds_path, model_type, checkpoint_dir=checkpoint_dir, debug=debug
    )

    if not os.path.exists(profile_predictor_config.checkpoint_dir) or clean:
        shutil.rmtree(profile_predictor_config.checkpoint_dir, ignore_errors=True)
        trainer, _, _, test_dl, test_results = launch_train(
            profile_predictor_config.model_dump(), use_wandb=False
        )
    else:
        logger.info(
            f"Checkpoint directory {profile_predictor_config.checkpoint_dir} already exists, skipping training of profile predictor..."
        )

    return trainer, test_dl, test_results


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

    base_config = setup_optimization_config(ds_path, debug=debug)
    config = base_config.model_copy(
        update={
            "model_init_config": {
                **base_config.model_init_config,
                "submodules": {
                    "profile_predictor": {
                        **base_config.model_init_config["submodules"][
                            "profile_predictor"
                        ],
                        "checkpoint_dir": os.path.join(
                            CHECKPOINT_DIR_BASE, "profile_predictor"
                        ),
                    },
                },
            },
        }
    )

    if not os.path.exists(config.checkpoint_dir) or clean:
        shutil.rmtree(config.checkpoint_dir, ignore_errors=True)
        launch_train(config.model_dump(), use_wandb=False)
    else:
        logger.info(
            f"Checkpoint directory {config.checkpoint_dir} already exists, skipping trajectory optimization..."
        )


if __name__ == "__main__":
    # Usage: python popsim_transport_predictor/trajectory_optimization/optimize.py <command> [--options]
    fire.Fire(
        {
            "characterize_dataset": characterize_dataset,
            "train_profile_predictor": train_profile_predictor,
            "run_trajectory_optimization": run_trajectory_optimization,
        }
    )
