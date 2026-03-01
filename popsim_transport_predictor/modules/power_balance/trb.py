from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import optax
import xarray as xr
from popsim.ml import DataLoader, TrainRunBuilder
from popsim.ml.dataloading import make_dataloaders

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
        )

        return ds_val, train_dl, val_dl, None

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        """
        Instantiate and return your model given a training DataLoader
        and a model config dict.
        """

        if model_init_config["model_case"] == "scaling_law":
            pass
        elif model_init_config["model_case"] == "sciml":
            pass
        elif model_init_config["model_case"] == "unstructured_nn":
            pass
        else:
            raise ValueError(f"Invalid model case: {model_init_config['model_case']}")

    @staticmethod
    def get_loss_fn(config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        def loss_fn(pred, targ):
            raise NotImplementedError(
                "Loss function not implemented yet for PowerBalanceTRB. This is a placeholder."
            )

        return loss_fn

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
