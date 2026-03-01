from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import optax
import xarray as xr
from popsim.ml import DataLoader, TrainRunBuilder

from popsim_transport_predictor.modules.power_balance.p_rad.module import RadiatedPower


class RadiatedPowerTRB(TrainRunBuilder):
    """TrainRunBuilder for the radiated power predictor used in transfer learning"""

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

        module = RadiatedPower.init(
            **model_init_config,
        )

        return module

    @staticmethod
    def get_loss_fn(config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        def loss_fn(pred, targ):
            absolute_error = jnp.abs(pred.P_rad_MW_pred - targ["P_rad_MW"].data)
            return optax.huber_loss(absolute_error)

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
