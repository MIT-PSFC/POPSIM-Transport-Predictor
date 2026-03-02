from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import optax
import xarray as xr
from loguru import logger
from popsim.ml import DataLoader, IntegralLoss, TrainRunBuilder
from popsim.ml.checkpointing import create_default_checkpoint_manager, restore_model
from popsim.ml.dataloading import make_dataloaders

from popsim_transport_predictor.modules.power_balance.module import (
    PowerBalanceEnv,
    PowerBalanceScalingLaw,
    PowerBalanceSciML,
    PowerBalanceUnstructuredNN,
)
from popsim_transport_predictor.modules.power_balance.p_oh.trb import OhmicPowerTRB
from popsim_transport_predictor.modules.power_balance.p_rad.trb import RadiatedPowerTRB
from popsim_transport_predictor.transfer_learning.orchestration.organize_data import (
    get_train_test_datasets_transfer,
    get_train_val_datasets,
)


class PowerBalanceTRB(TrainRunBuilder):
    """TrainRunBuilder for the power balance modules used in transfer learning"""

    @staticmethod
    def get_dataloaders(
        dataloader_config: dict,
    ) -> tuple[xr.Dataset, tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Get the dataset and dataloaders for training.

        For both standard learning and transfer learning, we essentially have two datasets.
        For standard learning, it's the standard train/val for hyperparameter tuning. No test is needed, so we can just return None.
        For transfer learning, we have the training dataset composed of all historic data and a small amount of new data,
        and the test dataset composed of a set amount of new data that is held out of training.
        We are not doing hyperparameter tuning for transfer learning.

        Also, slightly different from the POPSIM version, we're just returning the validation dataset.
        """

        if dataloader_config["transfer_learning"]:
            ds_train, ds_val = get_train_test_datasets_transfer(
                training_data_case=dataloader_config["training_data_case"],
                num_hp_shots=dataloader_config["num_hp_shots"],
                normalization_method=dataloader_config["normalization_method"],
            )
        else:
            ds_train, ds_val = get_train_val_datasets(
                training_data_case=dataloader_config["training_data_case"],
                normalization_method=dataloader_config["normalization_method"],
            )

        # Drop time_idx as a shared coordinate — it has duplicate values across shots and
        # causes groupby("shot") to fail when reassembling. The dataloader uses "time" instead.
        ds_train = ds_train.drop_vars("time_idx", errors="ignore")
        ds_val = ds_val.drop_vars("time_idx", errors="ignore")

        if "state_vars" in dataloader_config.keys():
            segment_lengths = [
                dataloader_config.get("segment_length_train", None),
                dataloader_config.get("segment_length_val", None),
            ]
            segment_overlaps = [
                dataloader_config.get("segment_overlap_train", None),
                dataloader_config.get("segment_overlap_val", None),
            ]
        else:
            segment_lengths = None
            segment_overlaps = None

        train_dl, val_dl = make_dataloaders(
            datasets=(ds_train, ds_val),
            time_coord="time",
            episode_coord="shot",
            input_vars=dataloader_config["input_vars"],
            target_vars=dataloader_config["target_vars"],
            extra_vars=dataloader_config.get("extra_vars", None),
            state_init_vars=dataloader_config.get("state_vars", None),
            batch_size=dataloader_config.get("batch_size", None),
            segment_lengths=segment_lengths,
            segment_overlaps=segment_overlaps,
            shuffle=[True, False],
            convert_xr_to_jnp=False,  # Needed to keep the coords for calculating loss
            nan_handling="drop_segment",
        )

        return ds_val, train_dl, val_dl, None

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        """
        Instantiate and return your model given a training DataLoader
        and a model config dict.
        """
        model_case = model_init_config["model_case"]
        if model_case in ["scaling_law", "sciml"]:
            # These cases work in real units, restore the p_oh and p_rad submodules
            p_oh_config = model_init_config["submodules"]["p_oh_predictor"]
            p_oh_predictor = OhmicPowerTRB.model_init(
                train_dl, p_oh_config["model_init_config"]
            )
            p_rad_config = model_init_config["submodules"]["p_rad_predictor"]
            p_rad_predictor = RadiatedPowerTRB.model_init(
                train_dl, p_rad_config["model_init_config"]
            )
            if model_init_config["restore_submodules"]:
                p_oh_manager = create_default_checkpoint_manager(
                    p_oh_config["checkpoint_dir"]
                )
                p_oh_predictor = restore_model(p_oh_manager, p_oh_predictor)
                p_rad_manager = create_default_checkpoint_manager(
                    p_rad_config["checkpoint_dir"]
                )
                p_rad_predictor = restore_model(p_rad_manager, p_rad_predictor)

        if model_case == "scaling_law":
            module = PowerBalanceScalingLaw.init(
                p_oh_predictor=p_oh_predictor,
                p_rad_predictor=p_rad_predictor,
                min_taue=model_init_config.get("min_taue", None),
                max_taue=model_init_config.get("max_taue", None),
            )
        elif model_case == "sciml":
            module = PowerBalanceSciML.init(
                p_oh_predictor=p_oh_predictor,
                p_rad_predictor=p_rad_predictor,
                in_size=model_init_config["in_size"],
                out_size=model_init_config["out_size"],
                nn_width=model_init_config["nn_width"],
                nn_depth=model_init_config["nn_depth"],
                min_taue=model_init_config.get("min_taue", None),
                max_taue=model_init_config.get("max_taue", None),
                prng_seed=model_init_config.get("prng_seed", 42),
            )
        elif model_case == "unstructured_nn":
            module = PowerBalanceUnstructuredNN.init(
                in_size=model_init_config["in_size"],
                out_size=model_init_config["out_size"],
                nn_width=model_init_config["nn_width"],
                nn_depth=model_init_config["nn_depth"],
                min_val=model_init_config.get("min_val", None),
                max_val=model_init_config.get("max_val", None),
                prng_seed=model_init_config.get("prng_seed", 42),
            )
        else:
            raise ValueError(f"Invalid model case: {model_init_config['model_case']}")

        env = PowerBalanceEnv(
            module=module,
            normalization_method=model_init_config["normalization_method"],
            freeze_submodules=model_init_config["freeze_submodules"],
        )

        if model_init_config.get("restore_main_module", False):
            manager = create_default_checkpoint_manager(
                model_init_config["checkpoint_dir"]
            )
            env = restore_model(manager, env)
        else:
            logger.warning("Not restoring main module from checkpoint.")

        return env

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        def loss_fn(pred, targ):
            _device_weights = loss_config["device_weight"]
            var_weights = loss_config["var_weight"]

            wtot_loss = jnp.abs(pred.Wtot_MJ_pred - targ["Wtot_MJ"].data)
            wtot_loss = optax.huber_loss(wtot_loss, delta=loss_config["huber_delta"])

            device_weight = 1  # TODO(ZanderKeith) fix device weighting device_weights[targ["ds_source"].item()]
            loss = device_weight * var_weights["Wtot_MJ"] * wtot_loss
            # TODO(ZanderKeith): might be worthwhile to put the p_oh and p_rad in here?
            return loss

        return IntegralLoss(loss_fn)

    @staticmethod
    def get_optimizer(config: dict) -> optax.GradientTransformation:
        schedule = optax.exponential_decay(
            init_value=config["lr0"],
            transition_steps=config["transition_steps"],
            decay_rate=config["decay_rate"],
            end_value=config["lrf"],
        )
        opt = optax.adamw(learning_rate=schedule, weight_decay=config["weight_decay"])
        return opt
