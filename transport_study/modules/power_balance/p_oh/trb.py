from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import netCDF4  # noqa: F401
import numpy as np
import optax
import xarray as xr
from loguru import logger
from popsim.ml import DataLoader, TrainRunBuilder
from popsim.ml.checkpointing import create_default_checkpoint_manager, restore_model
from popsim.ml.eval import EvalData, EvaluationSuite, batched_model_eval_and_loss

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.config import config
from transport_study.modules.normalization import make_normalizer
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
                for device in np.unique(train_dl.ds["ds_source"].values):
                    device_median = train_dl.ds.where(train_dl.ds["ds_source"] == device, drop=True)["P_oh_MW"].median().item()
                    device_medians.append(device_median)
                median = max(device_medians)
            model_init_config["max_val"] = 2 * median

        # Fit normalization stats from the training data only,
        # skipping the fit when a transfer checkpoint will overwrite the module anyway
        fit_ds = None if model_init_config.get("transfer_checkpoint") else train_dl.ds
        normalizer = make_normalizer(model_init_config["data_normalization"], fit_ds, len(config.ds_source_to_idx))

        module = OhmicPower.init(
            in_size=model_init_config["in_size"],
            out_size=model_init_config["out_size"],
            nn_width=model_init_config["nn_width"],
            nn_depth=model_init_config["nn_depth"],
            min_val=model_init_config["min_val"],
            max_val=model_init_config["max_val"],
            prng_seed=model_init_config["prng_seed"],
            normalizer=normalizer,
        )

        if model_init_config.get("transfer_checkpoint", False):
            transfer_manager = create_default_checkpoint_manager(model_init_config["transfer_checkpoint"])
            module = restore_model(transfer_manager, module)
            logger.debug(f"Restoring module from transfer learning pretrained checkpoint\n{model_init_config['transfer_checkpoint']}")

        return module

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        if "device_weights" not in loss_config:
            device_weights = dict.fromkeys(config.dataset_paths, 1.0)
        else:
            device_weights = loss_config["device_weights"]

        def loss_fn(pred, targ):
            absolute_error = jnp.abs(pred.P_oh_MW_pred - targ["P_oh_MW"].data)
            for device, weight in device_weights.items():
                device_mask = targ["ds_source_idx"].data == config.ds_source_to_idx[device]
                absolute_error = jnp.where(device_mask, weight * absolute_error, absolute_error)
            huber_loss = optax.huber_loss(absolute_error, delta=loss_config["huber_delta"])
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
    def get_val_eval_suite(suite_config) -> EvaluationSuite | None:
        """Validation suite computing the loss (sweep metric val/loss.mean).

        We are running with very large datasets.
        This means that the regular eval function will be uploading too much data to wandb
        This eval suite basically does the same thing but cuts the vec to be at most 100 long
        Can still see the distribution, but without all the data
        """
        if suite_config is None:
            return None
        loss_fn = OhmicPowerTRB.get_loss_fn(suite_config["loss_config"])

        def _eval_and_loss(model, loss_fn, inputs, targets):
            return batched_model_eval_and_loss(model, loss_fn, inputs, targets)

        jit_eval_and_loss = eqx.filter_jit(_eval_and_loss)

        def eval_fn(inp: EvalData) -> float:
            loss_vecs = []
            for batch in inp.dataloader:
                inputs, targets = batch.get_inputs_and_targets()
                loss_vec = jit_eval_and_loss(
                    inp.model,
                    loss_fn,
                    inputs,
                    targets,
                )
                loss_vecs.append(loss_vec)
            loss_vec = jnp.concatenate(loss_vecs)
            # Drop padded duplicate samples from the pad_last validation dataloader
            loss_vec = loss_vec[: inp.dataloader.dataset.n_samples]
            loss_vec_mean = loss_vec.mean()
            # Sort loss vec and sample at most 100 points evenly for logging
            if loss_vec.shape[0] > 100:
                sorted_indices = jnp.argsort(loss_vec)
                selected_indices = sorted_indices[jnp.linspace(0, loss_vec.shape[0] - 1, num=100, dtype=int)]
                loss_vec = loss_vec[selected_indices]
            return {
                "mean": loss_vec_mean,
                "vec": loss_vec,
            }

        return {"loss": eval_fn}

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
                ds_source_array = eval_data.input_ds["ds_source"].unstack("sample").isel({TIME_DIM: 0}).values
            else:
                # Case when there is a single source dataset present
                ds_source_array = np.array([ds_source.values.item() for _ in range(targ.sizes["shot"])])

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
            def get_trainable_nn(module: OhmicPower):
                return module.nn

            return get_trainable_nn

        def get_trainable(module: OhmicPower):
            return module.nn.layers[-1]

        return get_trainable
