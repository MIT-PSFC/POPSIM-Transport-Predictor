from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import optax
import xarray as xr
from popsim.ml import DataLoader, TrainRunBuilder

from transport_study.modules.power_balance.p_oh.module import OhmicPower


class OhmicPowerTRB(TrainRunBuilder):
    """TrainRunBuilder for the ohmic power predictor used in transfer learning"""

    @staticmethod
    def get_dataloaders(
        dataloader_config: dict,
    ) -> tuple[xr.Dataset, tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Get the dataset and dataloaders for training.
        """
        return None, None, None, None

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        """
        Instantiate and return your model given a training DataLoader
        and a model config dict.
        """

        # If max_val is not set, find the device with the largest median P_oh_MW in the training data
        # and set max_val to 2x that median value.
        if model_init_config["max_val"] is None:
            if train_dl.ds["ds_source"].size < 2:
                median = train_dl.ds["P_oh_MW"].median().item()
            else:
                device_medians = []
                for device in train_dl.ds["ds_source"].values:
                    device_median = (
                        train_dl.ds.where(
                            train_dl.ds["ds_source"] == device, drop=True
                        )["P_oh_MW"]
                        .median()
                        .item()
                    )
                    device_medians.append(device_median)
                median = max(device_medians)
            model_init_config["max_val"] = 2 * median

        module = OhmicPower.init(
            **model_init_config,
        )

        return module

    @staticmethod
    def get_loss_fn(config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        def loss_fn(pred, targ):
            # TODO(ZanderKeith): Fix device weights
            absolute_error = jnp.abs(pred.P_oh_MW_pred - targ["P_oh_MW"].data)
            return jnp.mean(optax.huber_loss(absolute_error))

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
