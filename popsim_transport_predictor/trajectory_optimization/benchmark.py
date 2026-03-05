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
import numpy as np
import xarray as xr
from loguru import logger
from popsim.ml import TrainConfig, Trainer
from popsim.ml.launch import launch_train

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
    trajectory_performance_comparison,
    trajectory_shapes_comparison,
)
from popsim_transport_predictor.trajectory_optimization.setup import (
    make_optimization_dataset,
)

SAVE_DIR = os.path.join(PACKAGE_ROOT, "../scratch", "trajectory_optimization_benchmark")
MAX_NUM_SHAPE_TIMES = 5
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
    model_types = ["direct_points"]
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
    ds_pred_list = []
    ds_pred_labels = []
    for model_type in model_types:
        eval_ds_path = os.path.join(save_dir, model_type, "eval_data.nc")
        eval_ds = xr.open_dataset(eval_ds_path)
        ds_pred_list.append(eval_ds)
        ds_pred_labels.append(model_type)

    ds_targ = ds_pred_list[
        0
    ]  # They should both have the same target dataset since they use the same test dataloader, so just take the first one.
    profile_comparison(
        profile_dir=os.path.join(save_dir, "profile_comparison"),
        ds_targ=ds_targ,
        ds_pred_list=ds_pred_list,
        ds_pred_labels=ds_pred_labels,
    )


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


##############################################
# Comparing optimized trajectory performance #
##############################################
def run_trajectory_evaluation(  # noqa: PLR0915
    ds_path: str | None = config.d3d_dataset_path,
    save_dir: str | None = SAVE_DIR,
    model_type: str | None = "direct_points",
    max_num_shape_times: int | None = MAX_NUM_SHAPE_TIMES,
    shape_time_min: float | None = SHAPE_TIME_MIN,
    shape_time_max: float | None = SHAPE_TIME_MAX,
    clean: bool | None = False,
    debug: bool | None = False,
):
    num_shape_times_list = list(
        range(1, max_num_shape_times + 1)
    )  # [1, 2, ..., max_num_shape_times]
    for num_shape_times in num_shape_times_list:
        case_dir = os.path.join(
            save_dir, "trajectory_evaluation", model_type, f"n_{num_shape_times}"
        )
        eval_ds_path = os.path.join(case_dir, "eval_data.nc")
        if not os.path.exists(eval_ds_path) or clean:
            logger.info(
                f"No evaluation data found for num_shape_times={num_shape_times}"
            )
            checkpoint_dir = os.path.join(case_dir, "checkpoints")
            shape_times = np.linspace(
                shape_time_min, shape_time_max, num_shape_times
            ).tolist()
            shape_times = [round(t, 1) for t in shape_times]  # Round to nearest 10th
            optimization_config = setup_optimization_config(
                ds_path,
                model_type,
                debug=debug,
                checkpoint_dir=checkpoint_dir,
                shape_times=shape_times,
            )
            profile_predictor_checkpoint_dir = os.path.join(
                save_dir, model_type, "checkpoints"
            )
            if not os.path.exists(profile_predictor_checkpoint_dir):
                logger.info(
                    "No checkpoints found for profile predictor, running profile predictor training..."
                )
                train_profile_predictor(
                    ds_path,
                    model_type,
                    profile_predictor_checkpoint_dir,
                    debug=False,
                    clean=True,
                )
            optimization_config.model_init_config["submodules"]["profile_predictor"][
                "checkpoint_dir"
            ] = profile_predictor_checkpoint_dir
            if not os.path.exists(checkpoint_dir) or clean:
                logger.info(
                    f"No checkpoints found for num_shape_times={num_shape_times}, running optimization..."
                )
                optimization_trainer, _, aug_dl, _, _test_results = launch_train(
                    optimization_config.model_dump(), use_wandb=False
                )
            else:
                logger.info(
                    f"Restoring checkpoint for num_shape_times={num_shape_times}"
                )
                _, train_dl, _, _test_dl = (
                    ProfileTrajectoryOptimizerTRB.get_dataloaders(
                        optimization_config.dataloader_config
                    )
                )
                optimization_trainer = Trainer(
                    model=ProfileTrajectoryOptimizerTRB.model_init(
                        train_dl, optimization_config.model_init_config
                    ),
                    loss_fn=ProfileTrajectoryOptimizerTRB.get_loss_fn(
                        optimization_config.loss_config
                    ),
                    optimizer=ProfileTrajectoryOptimizerTRB.get_optimizer(
                        optimization_config.optimizer_config
                    ),
                    checkpoint_dir=checkpoint_dir,
                )
            optimization_trainer.restore_best_checkpoint()
            eval_data = optimization_trainer.run_evals(aug_dl)
            input_ds = eval_data.input_ds.reset_index("sample")
            output_ds = eval_data.output_ds.reset_index("sample")
            eval_ds = xr.merge([input_ds, output_ds], compat="override")
            eval_ds.to_netcdf(eval_ds_path)
        else:
            logger.info(
                f"Evaluation data already exists for num_shape_times={num_shape_times}, skipping optimization..."
            )
            continue

    def _peaking_factor(ds: xr.Dataset) -> xr.DataArray:
        """Calculate the pressure profile peaking factor (max / avg) metric from the dataset.
        Args:
            ds: Dataset containing the pressure profiles to evaluate. Should have a "time_idx" dimension as well as ne and te profile variables.
        """

        ne20_psi = ds["ne20_psi"]
        Te_keV_psi = ds["Te_keV_psi"]

        P_psi = ne20_psi * Te_keV_psi
        avg = P_psi.mean(dim="psi")
        peaking = P_psi.max(dim="psi") / avg
        # Returned DataArray should have dimension (sample, time_idx) and coords (shot, time, shot_alt)
        return peaking

    ds_perf_list = []
    ds_perf_labels = []
    for i, num_shape_times in enumerate(num_shape_times_list):
        case_dir = os.path.join(
            save_dir, "trajectory_evaluation", model_type, f"n_{num_shape_times}"
        )
        eval_ds_path = os.path.join(case_dir, "eval_data.nc")
        eval_ds = xr.open_dataset(eval_ds_path)

        if i == 0:
            # For the first one, also calculate the reference peaking factor from the original dataset
            ds_ref = eval_ds[["ne20_psi", "Te_keV_psi", "fresh_profiles"]]
            ds_ref_input = ds_ref.where(
                ds_ref["fresh_profiles"] == 1, drop=True
            )  # Only use timesteps with fresh profiles for the reference
            ds_ref_input = ds_ref_input.rename({"time_idx_input": "time_idx"})
            ds_ref_input = ds_ref_input.swap_dims({"psi_input": "psi"})
            # Only get unique values of shot_alt coordinate to prevent duplicates from the augmentation
            ds_ref_input = ds_ref_input.groupby("shot_alt").first()
            ds_perf = _peaking_factor(ds_ref_input)
            ds_perf_list.append(ds_perf)
            ds_perf_labels.append("Original")

        ds_ref = eval_ds[
            ["output.profile_predictor_output.ne", "output.profile_predictor_output.te"]
        ]
        # Rename to match the target variable names expected by the peaking factor calculation
        ds_ref = ds_ref.rename(
            {
                "output.profile_predictor_output.ne": "ne20_psi",
                "output.profile_predictor_output.te": "Te_keV_psi",
                "time": "time_idx",
            }
        )
        ds_perf = _peaking_factor(ds_ref)
        ds_perf_list.append(ds_perf)
        ds_perf_labels.append(f"N={num_shape_times}")

    trajectory_performance_comparison(
        ds_perf_list=ds_perf_list,
        ds_perf_labels=ds_perf_labels,
        save_dir=os.path.join(
            save_dir, "trajectory_evaluation", model_type, "performance_comparison"
        ),
        title=f"Trajectory Performance Comparison for {model_type} Profile Predictor",
    )


def plot_trajectory_shapes(
    ds_path: str | None = config.d3d_dataset_path,
    save_dir: str | None = SAVE_DIR,
    model_type: str | None = "direct_points",
    max_num_shape_times: int | None = MAX_NUM_SHAPE_TIMES,
    shape_time_min: float | None = SHAPE_TIME_MIN,
    shape_time_max: float | None = SHAPE_TIME_MAX,
    debug: bool | None = False,
):
    """Plot trajectory shapes over time"""

    trajectory_shapes = []
    trajectory_labels = []
    num_shape_times_list = list(
        range(1, max_num_shape_times + 1)
    )  # [1, 2, ..., max_num_shape_times]
    if debug:
        num_shape_times_list = [
            max_num_shape_times
        ]  # Just plot the max one for debugging
    for num_shape_times in num_shape_times_list:
        case_dir = os.path.join(
            save_dir, "trajectory_evaluation", model_type, f"n_{num_shape_times}"
        )
        checkpoint_dir = os.path.join(case_dir, "checkpoints")
        shape_times = np.linspace(
            shape_time_min, shape_time_max, num_shape_times
        ).tolist()
        shape_times = [round(t, 1) for t in shape_times]  # Round to nearest 10th
        optimization_config = setup_optimization_config(
            ds_path,
            model_type,
            checkpoint_dir=checkpoint_dir,
            shape_times=shape_times,
        )
        profile_predictor_checkpoint_dir = os.path.join(
            save_dir, model_type, "checkpoints"
        )
        optimization_config.model_init_config["submodules"]["profile_predictor"][
            "checkpoint_dir"
        ] = profile_predictor_checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            logger.warning(
                f"No checkpoints found for num_shape_times={num_shape_times}, skipping shape plotting..."
            )
            continue

        _, train_dl, _, _test_dl = ProfileTrajectoryOptimizerTRB.get_dataloaders(
            optimization_config.dataloader_config
        )
        optimization_trainer = Trainer(
            model=ProfileTrajectoryOptimizerTRB.model_init(
                train_dl, optimization_config.model_init_config
            ),
            loss_fn=ProfileTrajectoryOptimizerTRB.get_loss_fn(
                optimization_config.loss_config
            ),
            optimizer=ProfileTrajectoryOptimizerTRB.get_optimizer(
                optimization_config.optimizer_config
            ),
            checkpoint_dir=checkpoint_dir,
        )
        optimization_trainer.restore_best_checkpoint()

        optimized_trajectory = {
            var: getattr(optimization_trainer.train_state.model.module, var)
            for var in optimization_config.model_init_config["input_ranges"].keys()
        }
        optimized_trajectory["shape_times"] = (
            optimization_trainer.train_state.model.module.config.shape_times
        )

        trajectory_shapes.append(optimized_trajectory)
        trajectory_labels.append(f"N={num_shape_times}")

    # Only need to load the evaluation data once since it contains the original trajectory shapes
    ds, _episode_coord = get_ds(
        optimization_config.dataloader_config["ds_path"],
        fresh_profiles=False,  # Use all timesteps for trajectory optimization
        debug=optimization_config.dataloader_config["debug"],
    )
    ds_aug = make_optimization_dataset(
        ds=ds,
        debug=True,
        prng_seed=optimization_config.dataloader_config["prng_seed"],
    )

    trajectory_shapes_comparison(
        trajectory_shapes,
        trajectory_labels,
        orig_traj=ds_aug,
        save_dir=os.path.join(
            save_dir, "trajectory_evaluation", model_type, "shape_comparison"
        ),
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "run_profile_predictor_evaluation": run_profile_predictor_evaluation,
            "run_trajectory_evaluation": run_trajectory_evaluation,
            "plot_trajectory_shapes": plot_trajectory_shapes,
        }
    )
