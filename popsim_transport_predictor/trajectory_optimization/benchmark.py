"""
Train and benchmark the profile predictor and the trajectory optimization

Plots to make:
1. Profile predictor performance within the trajectory optimization ranges
- Histograms of errors on a per-profile basis
- For each profile, plot predicted vs true for the different implementations

2. Trajectory optimization performance
- Original trajectory loss function (only evaluated on fresh profiles) vs new trajectory loss function (evaluated on all timesteps)
- Also comparison of optimized trajectory performance vs number of shape times
"""

import os

import fire
import xarray as xr
from loguru import logger
from popsim.ml import TrainConfig, Trainer

from popsim_transport_predictor import PACKAGE_ROOT
from popsim_transport_predictor.config import config
from popsim_transport_predictor.modules.profile_trajectory.data import get_ds
from popsim_transport_predictor.modules.profile_trajectory.profile_predictor.trb import (
    ProfilePredictorTRB,
)
from popsim_transport_predictor.modules.profile_trajectory.trb import (
    ProfileTrajectoryOptimizerTRB,
)
from popsim_transport_predictor.trajectory_optimization.optimize import (
    setup_optimization_config,
    train_profile_predictor,
)
from popsim_transport_predictor.trajectory_optimization.plotting import (
    profile_comparison,
)
from popsim_transport_predictor.trajectory_optimization.setup import (
    make_optimization_dataset,
)

SAVE_DIR = os.path.join(PACKAGE_ROOT, "../scratch", "trajectory_optimization_benchmark")
NUM_SHAPE_TIMES = 8
SHAPE_TIME_MIN = 2.0
SHAPE_TIME_MAX = 5.5


################################################################
# Comparing different implementations of the profile predictor #
################################################################
def run_profile_predictor_evaluation(
    ds_path: str | None = config.d3d_dataset_path,
    save_dir: str | None = SAVE_DIR,
    clean: bool | None = False,
    debug: bool | None = False,
):
    model_types = ["shape_init", "direct_points"]
    for model_type in model_types:
        # Generate the evaluation data for each model type if it doesn't already exist
        eval_ds_path = os.path.join(save_dir, model_type, "eval_data.nc")
        if not os.path.exists(eval_ds_path) or clean:
            logger.info(f"No evaluation data found for model type {model_type}")
            checkpoint_dir = os.path.join(save_dir, model_type, "checkpoints")
            if not os.path.exists(checkpoint_dir) or clean:
                logger.info(
                    f"No checkpoints found for model type {model_type}, training model..."
                )
                trainer, test_dl, _test_results = train_profile_predictor(
                    ds_path, model_type, checkpoint_dir, debug=debug, clean=clean
                )
            else:
                logger.info(f"Restoring checkpoint for {model_type}")
                optimizer_config = setup_optimization_config(
                    ds_path, model_type, debug=debug, checkpoint_dir=checkpoint_dir
                )
                _, train_dl, _, test_dl = ProfileTrajectoryOptimizerTRB.get_dataloaders(
                    optimizer_config.dataloader_config
                )
                trainer = Trainer(
                    model=ProfilePredictorTRB.model_init(
                        train_dl,
                        optimizer_config.model_init_config["submodules"][
                            "profile_predictor"
                        ]["model_init_config"],
                    ),
                    loss_fn=ProfilePredictorTRB.get_loss_fn(
                        optimizer_config.loss_config
                    ),
                    optimizer=ProfilePredictorTRB.get_optimizer(
                        optimizer_config.optimizer_config
                    ),
                    checkpoint_dir=checkpoint_dir,
                )
            trainer.restore_best_checkpoint()
            eval_data = trainer.run_evals(test_dl)
            input_ds = eval_data.input_ds.reset_index("sample")
            output_ds = eval_data.output_ds.reset_index("sample")
            eval_ds = xr.merge([input_ds, output_ds], compat="override")
            eval_ds.to_netcdf(eval_ds_path)
        else:
            logger.info(
                f"Evaluation data already exists for model type {model_type}, skipping evaluation..."
            )
            continue

    # Plots n stuff


def eval_profile_predictor(transport_predictor_config: TrainConfig):
    """
    Evaluate how well the profile predictor works within the trajectory optimization shots
    """
    _, train_dl, _, test_dl = ProfileTrajectoryOptimizerTRB.get_dataloaders(
        transport_predictor_config.dataloader_config
    )
    trainer = Trainer(
        model=ProfilePredictorTRB.model_init(
            train_dl, transport_predictor_config.model_init_config
        ),
        loss_fn=ProfilePredictorTRB.get_loss_fn(transport_predictor_config.loss_config),
        optimizer=ProfilePredictorTRB.get_optimizer(
            transport_predictor_config.optimizer_config
        ),
        checkpoint_dir=transport_predictor_config.checkpoint_dir,
    )
    trainer.restore_best_checkpoint()

    # Run evaluation on the test set and extract the input and output datasets
    eval_data = trainer.run_evals(test_dl)

    return eval_data


def plot_profiles(
    ds_path: str,
    fig_dir: str,
):
    ds, _episode_coord = get_ds(
        ds_path,
        fresh_profiles=True,  # Only use timesteps where profile data is fresh
        debug=False,
    )
    ds_opt = make_optimization_dataset(
        ds, debug=True
    )  # Get original shots for plotting

    profile_comparison(
        profile_dir=fig_dir,
        ds_targ=ds_opt,  # Original dataset is the target since it has the true profiles
        ds_pred=None,  # No predictions for now, just plotting the original profiles
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "run_profile_predictor_evaluation": run_profile_predictor_evaluation,
        }
    )
