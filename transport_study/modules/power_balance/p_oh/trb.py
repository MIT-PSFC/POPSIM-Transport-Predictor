from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import netCDF4  # noqa: F401
import numpy as np
import optax
import xarray as xr
from popsim.ml import DataLoader, TrainRunBuilder
from popsim.ml.eval import EvalData, EvaluationSuite

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.modules.power_balance.p_oh.module import OhmicPower
from transport_study.orchestration.organize_data import DS_SOURCE_TO_IDX


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
                for device in np.unique(train_dl.ds["ds_source"].values):
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
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        if "device_weights" not in loss_config:
            device_weights = {
                "cmod": 1.0,
                "tcv": 1.0,
                "d3d_lp": 1.0,
                "d3d_hp": 1.0,
            }
        else:
            device_weights = loss_config["device_weights"]

        def loss_fn(pred, targ):
            absolute_error = jnp.abs(pred.P_oh_MW_pred - targ["P_oh_MW"].data)
            for device, weight in device_weights.items():
                device_mask = targ["ds_source_idx"].data == DS_SOURCE_TO_IDX[device]
                absolute_error = jnp.where(
                    device_mask, weight * absolute_error, absolute_error
                )
            huber_loss = optax.huber_loss(
                absolute_error, delta=loss_config["huber_delta"]
            )
            return jnp.mean(huber_loss)

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

    @staticmethod
    def get_test_eval_suite(config) -> EvaluationSuite:
        """Evaluation suite for testing after training."""

        def study_results(eval_data: EvalData) -> xr.Dataset:
            """Calculate final study results
                - Target vs predicted P_oh_MW
                - Absolute and relative error on a per-timeslice basis
                - Integrated error over time for each shot
            This should maintain the coordinates of the original dataset, in particular `ds_source` and `shot`
            """
            # Unstack sample MultiIndex -> (shot, time_idx) so we can integrate per shot
            targ = eval_data.input_ds.P_oh_MW.unstack("sample")
            pred = eval_data.output_ds.P_oh_MW_pred.unstack("sample")
            time_2d = eval_data.input_ds[TIME_COORD].unstack("sample")

            # Get relative error on a per-timeslice basis
            error_abs_ts = xr.apply_ufunc(np.abs, pred - targ)
            error_rel_ts = error_abs_ts / (xr.apply_ufunc(np.abs, targ) + 0.1)

            # Integrate absolute error over time for each shot, ignoring NaN-padded entries
            def _trapezoid_dropna(y, x):
                mask = ~np.isnan(x) & ~np.isnan(y)
                if mask.sum() < 2:
                    return np.nan
                y_valid, x_valid = y[mask], x[mask]
                sort_idx = np.argsort(x_valid)
                return np.trapezoid(y_valid[sort_idx], x_valid[sort_idx])

            error_abs_shot = xr.apply_ufunc(
                _trapezoid_dropna,
                error_abs_ts,
                time_2d,
                input_core_dims=[[TIME_DIM], [TIME_DIM]],
                vectorize=True,
            )

            error_rel_shot = xr.apply_ufunc(
                _trapezoid_dropna,
                error_rel_ts,
                time_2d,
                input_core_dims=[[TIME_DIM], [TIME_DIM]],
                vectorize=True,
            )

            # ds_source is constant per shot so extract as a shot-only coordinate
            ds_source = eval_data.input_ds["ds_source"]
            if "sample" in ds_source.dims:
                # Case when there are multiple source datasets present
                ds_source_array = (
                    eval_data.input_ds["ds_source"]
                    .unstack("sample")
                    .isel({TIME_DIM: 0})
                    .values
                )
            else:
                # Case when there is a single source dataset present
                ds_source_array = np.array(
                    [ds_source.values.item() for _ in range(targ.sizes["shot"])]
                )

            ds = xr.Dataset(
                data_vars={
                    "P_oh_MW_targ": targ,
                    "P_oh_MW_pred": pred,
                    "error_abs_ts": error_abs_ts,
                    "error_rel_ts": error_rel_ts,
                    "error_abs_shot": error_abs_shot,
                    "error_rel_shot": error_rel_shot,
                }
            )
            ds = ds.assign_coords(ds_source=(EPISODE_DIM, ds_source_array))
            ds = ds.drop_vars("quantile", errors="ignore")
            return ds

        eval_suite = {
            "study_results": study_results,
        }

        return eval_suite
