from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from popsim.ml import DataLoader, IntegralLoss, TrainRunBuilder
from popsim.ml.checkpointing import create_default_checkpoint_manager, restore_model
from popsim.ml.dataloading import make_dataloaders
from popsim.ml.train_config import load_dict

from transport_study.modules.profile_predictor.trb import (
    ProfilePredictorTRB,
)
from transport_study.modules.profile_trajectory.data import get_ds
from transport_study.modules.profile_trajectory.module import (
    ProfileTrajectoryOptimizer,
    ProfileTrajectoryOptimizerEnv,
)
from transport_study.trajectory_optimization.setup import (
    make_optimization_dataset,
)


class ProfileTrajectoryOptimizerTRB(TrainRunBuilder):
    @staticmethod
    def get_dataloaders(
        dataloader_config: dict,
    ) -> tuple[xr.Dataset, DataLoader, DataLoader, DataLoader]:
        """
        Get the dataset and dataloaders for training.

        Original dataset creation is shared between trajectory optimization and the profile predictor
        Profile predictor gets time-independent dataloaders (just train and val),
        while trajectory optimization gets two identical time-dependent dataloaders with a lot of augmented traces
        """
        if dataloader_config.get("module") == "profile_trajectory":
            ds, _ = get_ds(
                dataloader_config["ds_path"],
                fresh_profiles=False,  # Use all timesteps for trajectory optimization
                debug=dataloader_config["debug"],
            )
            ds_aug = make_optimization_dataset(
                ds=ds,
                debug=dataloader_config["debug"],
                prng_seed=dataloader_config["prng_seed"],
            )
            train_dl, val_dl = make_dataloaders(
                datasets=[
                    ds_aug,
                    ds_aug,
                ],  # Using the same augmented dataset to both train and validate the trajectory
                time_coord="time",
                episode_coord="shot_alt",
                input_vars=dataloader_config["input_vars"],
                target_vars=dataloader_config["target_vars"],
                convert_xr_to_jnp=dataloader_config["convert_xr_to_jnp"],
                state_init_vars=dataloader_config["state_vars"],
                extra_vars=dataloader_config.get("extra_vars", None),
                batch_size=dataloader_config["batch_size"],
                shuffle=[True, False],
            )
            ds = ds_aug
        else:
            raise ValueError(
                f"Called with a module that isn't a profile trajectory optimizer: {dataloader_config.get('module')}"
            )

        return ds, train_dl, val_dl, None

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        """Initialize the model to be trained"""
        psigrid = jnp.asarray(train_dl.ds["psi_n"].data)
        submodule_configs = model_init_config["submodules"]

        config = ProfileTrajectoryOptimizer.Config(
            traj_times=jnp.asarray(model_init_config["traj_times"]),
            input_ranges=model_init_config["input_ranges"],
        )

        def _restore_profile_predictor(profile_predictor_config):
            # The shape_init profile predictors need the historic data to set themselves up
            _, profile_predictor_train_dl, _, _ = ProfilePredictorTRB.get_dataloaders(
                profile_predictor_config["dataloader_config"]
            )

            profile_predictor = ProfilePredictorTRB.model_init(
                profile_predictor_train_dl,
                profile_predictor_config["model_init_config"],
            )
            profile_predictor_manager = create_default_checkpoint_manager(
                profile_predictor_config["checkpoint_dir"]
            )
            profile_predictor = restore_model(
                profile_predictor_manager, profile_predictor
            )
            return profile_predictor

        profile_predictor = _restore_profile_predictor(
            load_dict(submodule_configs["profile_predictor"])
        )

        # If we aren't optimizing density, fill in the trajectory with the programmed points
        # TODO(ZanderKeith): I suppose this only works when we're sweeping on one shot.
        if not model_init_config.get("optimize_density", False):
            # Use the first sample's time array to look up values by nearest index,
            # since the time coordinate is 2D after make_optimization_dataset and cannot be used with .sel().
            meta = train_dl.dataset.training_metadata
            sample_dim = meta.sample_dim
            time_dim = meta.time_dep_metadata.time_dim
            ds_first = train_dl.ds.isel({sample_dim: 0})
            time_arr = (
                ds_first["time"].values
                if "time" in ds_first
                else ds_first[time_dim].values
            )
            ne20_arr = ds_first["ne20_edge_prog"].values
            trajectory = {
                "ne20_edge": jnp.array(
                    [
                        ne20_arr[int(np.argmin(np.abs(time_arr - float(t))))]
                        for t in config.traj_times
                    ]
                ),
            }
        else:
            trajectory = None

        module = ProfileTrajectoryOptimizer.init(
            config=config,
            profile_predictor=profile_predictor,
            psigrid=psigrid,
            trajectory=trajectory,
        )

        # Wrap the module in an environment since it's time-dependent
        env = ProfileTrajectoryOptimizerEnv(
            module=module,
            optimize_density=model_init_config.get("optimize_density", False),
        )

        if model_init_config.get("restore_main_module", False):
            manager = create_default_checkpoint_manager(
                model_init_config["checkpoint_dir"]
            )
            env = restore_model(manager, env)

        return env

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        def loss_fn(pred, targ):
            # Loss function right now just minimizes the pressure peaking
            ne20_psi = pred.profile_predictor_output.ne.data
            Te_keV_psi = pred.profile_predictor_output.te.data

            # Ensure arrays have compatible shapes for broadcasting
            ne20_psi = jnp.asarray(ne20_psi)
            Te_keV_psi = jnp.asarray(Te_keV_psi)

            # Calculate pressure with explicit broadcasting
            P_psi = jnp.multiply(ne20_psi, Te_keV_psi)
            avg = jnp.mean(P_psi)
            # Guard against division by near-zero avg (e.g. for NaN-replaced zero inputs)
            safe_avg = jnp.maximum(avg, 1e-6)
            peaking = jnp.max(P_psi) / safe_avg
            return peaking

        return IntegralLoss(loss_fn, nan_strategy="zero")

    @staticmethod
    def get_optimizer(optimizer_config: dict) -> optax.GradientTransformation:
        schedule = optax.exponential_decay(
            init_value=optimizer_config["lr0"],
            transition_steps=optimizer_config["transition_steps"],
            decay_rate=optimizer_config["decay_rate"],
            end_value=optimizer_config["lrf"],
        )
        opt = optax.chain(
            optax.clip_by_global_norm(optimizer_config.get("max_grad_norm", 1.0)),
            optax.adamw(
                learning_rate=schedule, weight_decay=optimizer_config["weight_decay"]
            ),
        )
        return opt
