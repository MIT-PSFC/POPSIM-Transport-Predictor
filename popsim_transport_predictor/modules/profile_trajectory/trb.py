from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import optax
import xarray as xr
from popsim.ml import DataLoader, IntegralLoss, TrainRunBuilder
from popsim.ml.checkpointing import create_default_checkpoint_manager, restore_model
from popsim.ml.dataloading import make_dataloaders
from popsim.ml.split_utils import split_dataset_by_fracs

from popsim_transport_predictor.modules.profile_trajectory.data import get_ds
from popsim_transport_predictor.modules.profile_trajectory.module import (
    ProfileTrajectoryOptimizer,
    ProfileTrajectoryOptimizerEnv,
)
from popsim_transport_predictor.trajectory_optimization.setup import (
    make_optimization_dataset,
)


class ProfileTrajectoryOptimizerTRB(TrainRunBuilder):
    @staticmethod
    def get_dataloaders(dataloader_config: dict) -> tuple[xr.Dataset, list[DataLoader]]:
        """
        Get the dataset and dataloaders for training.

        Original dataset creation is shared between trajectory optimization and the profile predictor
        Profile predictor gets time-independent dataloaders (just train and val),
        while trajectory optimization gets two identical time-dependent dataloaders with a lot of augmented traces
        """

        ds, episode_coord = get_ds(
            dataloader_config["ds_path"], debug=dataloader_config["debug"]
        )

        if dataloader_config.get("module") == "profile_predictor":
            ds_train, ds_val = split_dataset_by_fracs(
                ds,
                fracs=dataloader_config["split_fracs"],
                dim=episode_coord,
                seed=dataloader_config["prng_seed"],
            )
            train_dl, val_dl = make_dataloaders(
                datasets=[ds_train, ds_val],
                time_coord="time",
                episode_coord=episode_coord,
                input_vars=dataloader_config["input_vars"],
                target_vars=dataloader_config["target_vars"],
                convert_xr_to_jnp=dataloader_config["convert_xr_to_jnp"],
                batch_size=dataloader_config["batch_size"],
                shuffle=[True, False],
            )
        elif dataloader_config.get("module") == "profile_trajectory":
            aug_config = dataloader_config["augmentation"]
            # t_min, t_max to trim the loaded dataset size
            # give it the dataset to find the typical ranges
            ds_aug = make_optimization_dataset(aug_config)
            train_dl, val_dl = make_dataloaders(
                datasets=[
                    ds_aug,
                    ds_aug,
                ],  # Using the same augmented dataset for both training and validation
                time_coord="time",
                episode_coord="shot_alt",
                input_vars=dataloader_config["input_vars"],
                target_vars=dataloader_config["target_vars"],
                convert_xr_to_jnp=dataloader_config["convert_xr_to_jnp"],
                state_init_vars=dataloader_config["state_init_vars"],
                batch_size=dataloader_config["batch_size"],
                shuffle=[True, False],
            )

        return ds, [train_dl, val_dl]

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        """Initialize the model to be trained"""
        psigrid = jnp.asarray(train_dl.ds["psin"].data)
        submodule_configs = model_init_config["submodules"]
        module = ProfileTrajectoryOptimizer.init(
            profile_predictor_config=submodule_configs["profile_predictor"],
            psigrid=psigrid,
        )

        # Wrap the module in an environment since it's time-dependent
        env = ProfileTrajectoryOptimizerEnv(module=module)

        if model_init_config.get("restore_main_module", False):
            manager = create_default_checkpoint_manager(
                model_init_config["checkpoint_dir"]
            )
            env = restore_model(manager, env)

        return env

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        def loss_fn(pred, targ):
            return jnp.mean((pred - targ) ** 2)

        return IntegralLoss(loss_fn)

    @staticmethod
    def get_optimizer(optimizer_config: dict) -> optax.GradientTransformation:
        schedule = optax.exponential_decay(
            init_value=optimizer_config["lr0"],
            transition_steps=optimizer_config["decay_steps"],
            decay_rate=optimizer_config["decay_rate"],
            end_value=optimizer_config["lrf"],
        )
        opt = optax.adamw(
            learning_rate=schedule, weight_decay=optimizer_config["weight_decay"]
        )
        return opt
