"""Shared TrainRunBuilder for the scalar power predictors (P_oh, P_rad)."""

from collections.abc import Callable
from typing import Any, ClassVar

import jax.numpy as jnp
import optax
from popsim.ml import DataLoader, TrainRunBuilder
from popsim.ml.eval import EvaluationSuite

from transport_study.config import config
from transport_study.modules.normalization import make_normalizer
from transport_study.modules.trb_utils import (
    make_exponential_adamw,
    make_loss_eval_suite,
    normalizer_fit_dataset,
    per_sample_device_values,
    restore_from_checkpoint,
    scalar_study_results,
    target_device_idx,
)


class ScalarPowerTRB(TrainRunBuilder):
    """TrainRunBuilder for a scalar power predictor used in transfer learning.

    Subclasses set SIGNAL (the target variable, e.g. "power_ohm_MW")
    and MODULE_CLS (the predictor module class).
    """

    SIGNAL: ClassVar[str]
    MODULE_CLS: ClassVar[type]

    @staticmethod
    def get_dataloaders(dataloader_config: dict):
        """The scalar power cases load their data through dataloader_config["data_train_run_builder"]."""
        raise NotImplementedError("The scalar power cases build their dataloaders with dataloader_config['data_train_run_builder']")

    @classmethod
    def model_init(cls, train_dl: DataLoader, model_init_config: dict) -> Any:
        """The predictor with its normalizer fitted on the training data, restored from the transfer checkpoint when one is set."""
        normalizer = make_normalizer(
            model_init_config["data_normalization"],
            normalizer_fit_dataset(train_dl, model_init_config),
            len(config.ds_source_to_idx),
            target_device_idx(),
        )
        module = cls.MODULE_CLS.init(
            in_size=model_init_config["in_size"],
            out_size=model_init_config["out_size"],
            nn_width=model_init_config["nn_width"],
            nn_depth=model_init_config["nn_depth"],
            prng_seed=model_init_config["prng_seed"],
            normalizer=normalizer,
        )
        if model_init_config.get("transfer_checkpoint"):
            module = restore_from_checkpoint(module, model_init_config["transfer_checkpoint"])
        return module

    @classmethod
    def _make_loss_fn(cls, loss_config: dict, use_huber: bool) -> Callable[[Any, Any], jnp.ndarray]:
        """Device-weighted error on the predicted power.

        use_huber selects the training loss (huber with the swept huber_delta)
        or the delta-free validation loss (plain absolute error),
        so the sweep metric val/loss.mean cannot be gamed by shrinking delta.
        """
        device_weights = loss_config.get("device_weights", {})
        signal = cls.SIGNAL

        def loss_fn(pred, targ):
            residual = getattr(pred, f"{signal}_pred") - targ[signal].data
            errors = optax.huber_loss(residual, delta=loss_config["huber_delta"]) if use_huber else jnp.abs(residual)
            sample_weights = per_sample_device_values(targ["ds_source_idx"].data, device_weights, 1.0)
            return jnp.mean(sample_weights * errors)

        return loss_fn

    @classmethod
    def get_loss_fn(cls, loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        return cls._make_loss_fn(loss_config, use_huber=True)

    @classmethod
    def get_val_loss_fn(cls, loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        return cls._make_loss_fn(loss_config, use_huber=False)

    @staticmethod
    def get_optimizer(optimizer_config: dict) -> optax.GradientTransformation:
        return make_exponential_adamw(optimizer_config)

    @classmethod
    def get_val_eval_suite(cls, suite_config) -> EvaluationSuite | None:
        """Validation suite computing the delta-free loss (sweep metric val/loss.mean)."""
        if suite_config is None:
            return None
        return make_loss_eval_suite(cls.get_val_loss_fn(suite_config["loss_config"]))

    @classmethod
    def get_test_eval_suite(cls, suite_config) -> EvaluationSuite | None:
        """The power study results (trb_utils.scalar_study_results), None for sweep trials."""
        if not suite_config:
            return None
        signal = cls.SIGNAL
        return {"study_results": lambda eval_data: scalar_study_results(eval_data, signal, f"{signal}_pred")}

    @staticmethod
    def get_trainable_getter(model_init_config: dict) -> Callable[[Any], Any] | None:
        """The NN, or only its last layer for a transfer case. The normalizer statistics are frozen buffers.

        Returning None here would let the trainer train every array leaf.
        """
        if model_init_config["domain_adaptation"] == "transfer":
            return lambda module: module.nn.layers[-1]
        return lambda module: module.nn
