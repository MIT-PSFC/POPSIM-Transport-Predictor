"""Shared TrainRunBuilder for the scalar power predictors (P_oh, P_rad)."""

from collections.abc import Callable
from typing import Any, ClassVar

import jax.numpy as jnp
import netCDF4  # noqa: F401
import numpy as np
import optax
import xarray as xr
from loguru import logger
from popsim.ml import DataLoader, TrainRunBuilder
from popsim.ml.checkpointing import create_default_checkpoint_manager, restore_model
from popsim.ml.eval import EvalData, EvaluationSuite

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.config import config
from transport_study.modules.normalization import make_normalizer
from transport_study.modules.trb_utils import (
    integrate_error_over_time,
    make_exponential_adamw,
    make_loss_eval_suite,
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
    def get_dataloaders(
        dataloader_config: dict,
    ) -> tuple[xr.Dataset, tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Get the dataset and dataloaders for training.
        """
        return None, None, None, None

    @classmethod
    def model_init(cls, train_dl: DataLoader, model_init_config: dict) -> Any:
        """
        Instantiate and return your model given a training DataLoader
        and a model config dict.
        """

        # Fit normalization stats from the training data only, skipping the
        # fit when a transfer checkpoint will overwrite the module anyway.
        # transfer_pretrain dataloaders carry the combined historic + target
        # fit dataset as an attribute (see PowerBalanceTRB.get_dataloaders)
        if model_init_config.get("transfer_checkpoint"):
            fit_ds = None
        else:
            fit_ds = getattr(train_dl, "normalizer_fit_ds", train_dl.ds)
        normalizer = make_normalizer(model_init_config["data_normalization"], fit_ds, len(config.ds_source_to_idx), target_device_idx())

        module = cls.MODULE_CLS.init(
            in_size=model_init_config["in_size"],
            out_size=model_init_config["out_size"],
            nn_width=model_init_config["nn_width"],
            nn_depth=model_init_config["nn_depth"],
            prng_seed=model_init_config["prng_seed"],
            normalizer=normalizer,
        )

        if model_init_config.get("transfer_checkpoint", False):
            transfer_manager = create_default_checkpoint_manager(model_init_config["transfer_checkpoint"])
            module = restore_model(transfer_manager, module)
            logger.debug(f"Restoring module from transfer learning pretrained checkpoint\n{model_init_config['transfer_checkpoint']}")

        return module

    @classmethod
    def _make_loss_fn(cls, loss_config: dict, use_huber: bool) -> Callable[[Any, Any], jnp.ndarray]:
        """Device-weighted error on the predicted power.

        use_huber selects the training loss (huber with the swept huber_delta)
        or the delta-free validation loss (plain absolute error),
        so the sweep metric val/loss.mean cannot be gamed by shrinking delta.
        """
        if "device_weights" not in loss_config:
            device_weights = dict.fromkeys(config.dataset_paths, 1.0)
        else:
            device_weights = loss_config["device_weights"]

        signal = cls.SIGNAL

        def loss_fn(pred, targ):
            residual = getattr(pred, f"{signal}_pred") - targ[signal].data
            errors = optax.huber_loss(residual, delta=loss_config["huber_delta"]) if use_huber else jnp.abs(residual)
            ds_source_idx = targ["ds_source_idx"].data
            sample_weights = jnp.ones(ds_source_idx.shape, dtype=errors.dtype)
            for device, weight in device_weights.items():
                sample_weights = jnp.where(ds_source_idx == config.ds_source_to_idx[device], weight, sample_weights)
            return jnp.mean(sample_weights * errors)

        return loss_fn

    @classmethod
    def get_loss_fn(cls, loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        return cls._make_loss_fn(loss_config, use_huber=True)

    @classmethod
    def get_val_loss_fn(cls, loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        return cls._make_loss_fn(loss_config, use_huber=False)

    @staticmethod
    def get_optimizer(config: dict) -> optax.GradientTransformation:
        return make_exponential_adamw(config)

    @classmethod
    def get_val_eval_suite(cls, suite_config) -> EvaluationSuite | None:
        """Validation suite computing the delta-free loss (sweep metric val/loss.mean)."""
        if suite_config is None:
            return None
        return make_loss_eval_suite(cls.get_val_loss_fn(suite_config["loss_config"]))

    @classmethod
    def get_test_eval_suite(cls, config) -> EvaluationSuite:
        """Evaluation suite for testing after training."""
        signal = cls.SIGNAL

        def study_results(eval_data: EvalData) -> xr.Dataset:
            """Calculate final study results
                - Target vs predicted power
                - Absolute and relative error on a per-timeslice basis
                - Integrated error over time for each shot
            This should maintain the coordinates of the original dataset, in particular `ds_source` and `shot`
            """
            # Unstack sample MultiIndex -> (shot, time_idx) so we can integrate per shot
            targ = eval_data.input_ds[signal].unstack("sample")
            pred = eval_data.output_ds[f"{signal}_pred"].unstack("sample")
            time_2d = eval_data.input_ds[TIME_COORD].unstack("sample")

            # Get relative error on a per-timeslice basis
            error_abs_ts = xr.apply_ufunc(np.abs, pred - targ)
            error_rel_ts = error_abs_ts / (xr.apply_ufunc(np.abs, targ) + 0.1)

            # Integrate absolute error over time for each shot, ignoring NaN-padded entries
            error_abs_shot = integrate_error_over_time(error_abs_ts, time_2d)
            error_rel_shot = integrate_error_over_time(error_rel_ts, time_2d)

            # ds_source is constant per shot so extract as a shot-only coordinate
            ds_source = eval_data.input_ds["ds_source"]
            if "sample" in ds_source.dims:
                # Case when there are multiple source datasets present
                ds_source_array = eval_data.input_ds["ds_source"].unstack("sample").isel({TIME_DIM: 0}).values
            else:
                # Case when there is a single source dataset present
                ds_source_array = np.array([ds_source.values.item() for _ in range(targ.sizes["shot"])])

            ds = xr.Dataset(
                data_vars={
                    f"{signal}_targ": targ,
                    f"{signal}_pred": pred,
                    "error_abs_ts": error_abs_ts,
                    "error_rel_ts": error_rel_ts,
                    "error_abs_shot": error_abs_shot,
                    "error_rel_shot": error_rel_shot,
                }
            )
            ds = ds.assign_coords(ds_source=(EPISODE_DIM, ds_source_array))
            ds = ds.drop_vars("quantile", errors="ignore")
            return ds

        if config:
            eval_suite = {
                "study_results": study_results,
            }
            return eval_suite
        else:
            # Hyperparameter tuning, do not run test evals
            return None

    @staticmethod
    def get_trainable_getter(model_init_config: dict) -> Callable[[Any], Any] | None:
        domain_adaptation = model_init_config["domain_adaptation"]

        if domain_adaptation != "transfer":
            # Only the NN trains, the normalizer stats are frozen buffers
            # (returning None here would let the trainer train every array leaf)
            def get_trainable_nn(module):
                return module.nn

            return get_trainable_nn

        def get_trainable(module):
            return module.nn.layers[-1]

        return get_trainable
