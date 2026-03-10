from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from loguru import logger
from popsim.ml import DataLoader, IntegralLoss, TrainConfig, TrainRunBuilder
from popsim.ml.checkpointing import create_default_checkpoint_manager, restore_model
from popsim.ml.dataloading import make_dataloaders
from popsim.ml.eval import EvalData, EvaluationSuite

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.modules.power_balance.module import (
    PowerBalanceEnv,
    PowerBalanceScalingLaw,
    PowerBalanceSciML,
    PowerBalanceUnstructuredNN,
)
from transport_study.modules.power_balance.p_oh.trb import OhmicPowerTRB
from transport_study.modules.power_balance.p_rad.trb import RadiatedPowerTRB
from transport_study.orchestration.organize_data import (
    DS_SOURCE_TO_IDX,
    get_train_test_datasets,
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

        if dataloader_config["domain_adaptation"] is None:
            logger.info("Using standard learning dataloader")
            ds_train, ds_val = get_train_val_datasets(
                training_data=dataloader_config["training_data"],
                data_normalization=dataloader_config["data_normalization"],
            )
        else:
            logger.info(
                f"Using transfer learning dataloader with domain adaptation {dataloader_config['domain_adaptation']}"
            )
            ds_train, ds_val = get_train_test_datasets(
                training_data=dataloader_config["training_data"],
                data_normalization=dataloader_config["data_normalization"],
                domain_adaptation=dataloader_config["domain_adaptation"],
                num_hp_shots=dataloader_config["num_hp_shots"],
                hp_test_set_size=dataloader_config.get("hp_test_set_size", None),
            )

        # Drop time_idx as a shared coordinate — it has duplicate values across shots and
        # causes groupby("shot") to fail when reassembling. The dataloader uses "time" instead.
        ds_train = ds_train.drop_vars(TIME_DIM, errors="ignore")
        ds_val = ds_val.drop_vars(TIME_DIM, errors="ignore")

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
            time_coord=TIME_COORD,
            episode_coord=EPISODE_DIM,
            input_vars=dataloader_config["input_vars"],
            target_vars=dataloader_config["target_vars"],
            extra_vars=dataloader_config.get("extra_vars", None),
            state_init_vars=dataloader_config.get("state_vars", None),
            batch_size=dataloader_config.get("batch_size", None),
            segment_lengths=segment_lengths,
            segment_overlaps=segment_overlaps,
            shuffle=[True, False],
            convert_xr_to_jnp=False,  # Needed to keep the coords for calculating loss
            # TODO(ZanderKeith): Switch to 'drop_segment' after you fix the dataset setup
            nan_handling="drop_slice_any",
        )
        # Running test evaluation on the validation set, since we don't need a dedicated test set
        # In the no domain adaptation case, we are hyperparameter tuning on all historic data, pick the best one and test on it
        # In the domain adaptation case, we are training on all historic data + some new data, and testing on the rest of the new data
        # No hyperparameter tuning is happening, so we treat the validation set as the test set and just return it for evaluation after training
        return ds_val, train_dl, val_dl, val_dl

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        """
        Instantiate and return your model given a training DataLoader
        and a model config dict.
        """
        model_type = model_init_config["model_type"]
        if model_type in ["scaling_law", "sciml"]:
            p_oh_config = model_init_config["submodules"]["p_oh_predictor"]
            if isinstance(p_oh_config, TrainConfig):
                p_oh_config = p_oh_config.model_dump()
            p_oh_predictor = OhmicPowerTRB.model_init(
                train_dl, p_oh_config["model_init_config"]
            )
            p_rad_config = model_init_config["submodules"]["p_rad_predictor"]
            if isinstance(p_rad_config, TrainConfig):
                p_rad_config = p_rad_config.model_dump()
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

        if model_type == "scaling_law":
            module = PowerBalanceScalingLaw.init(
                p_oh_predictor=p_oh_predictor,
                p_rad_predictor=p_rad_predictor,
                min_taue=model_init_config.get("min_taue", None),
                max_taue=model_init_config.get("max_taue", None),
            )
        elif model_type == "sciml":
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
        elif model_type == "unstructured_nn":
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

        if model_init_config.get("freeze_submodules", False):
            freeze_submodules = ["p_oh_predictor", "p_rad_predictor"]
        else:
            freeze_submodules = []

        env = PowerBalanceEnv(
            module=module,
            data_normalization=model_init_config["data_normalization"],
            freeze_submodules=freeze_submodules,
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
            absolute_error = jnp.abs(pred.Wtot_MJ_pred - targ["Wtot_MJ"].data)
            for device, weight in device_weights.items():
                device_mask = targ["ds_source_idx"].data == DS_SOURCE_TO_IDX[device]
                absolute_error = jnp.where(
                    device_mask, weight * absolute_error, absolute_error
                )
            huber_loss = optax.huber_loss(
                absolute_error, delta=loss_config["huber_delta"]
            )
            return jnp.mean(huber_loss)

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

    @staticmethod
    def get_test_eval_suite(config) -> EvaluationSuite:
        """Evaluation suite for testing after training."""

        def study_results(eval_data: EvalData) -> xr.Dataset:
            """Calculate final study results
                - Target vs predicted Wtot_MJ
                - Absolute and relative error on a per-timeslice basis
                - Integrated error over time for each shot
            This should maintain the coordinates of the original dataset, in particular `ds_source` and `shot`
            """

            # Unstack sample MultiIndex -> (shot, time_idx) and sqeeze out batch dimension so we can integrate per shot
            targ = eval_data.input_ds.Wtot_MJ.unstack("sample").squeeze()
            pred = (
                eval_data.output_ds["output.Wtot_MJ_pred"].unstack("sample").squeeze()
            )
            time_2d = eval_data.input_ds[TIME_COORD].unstack("sample").squeeze()

            # time-dependent modules modify the time dimension name, change it back to avoid confusion
            targ = targ.rename({TIME_DIM + "_input": TIME_DIM})
            pred = pred.rename({TIME_DIM + "_input": TIME_DIM})
            time_2d = time_2d.rename({TIME_DIM + "_input": TIME_DIM})

            # Get relative error on a per-timeslice basis
            error_abs_ts = xr.apply_ufunc(np.abs, pred - targ)
            error_rel_ts = error_abs_ts / (xr.apply_ufunc(np.abs, targ) + 0.1)

            # Integrate absolute error over time for each shot, ignoring NaN-padded entries
            def _trapezoid_dropna(y, x):
                mask = ~np.isnan(x)
                return np.trapezoid(y[mask], x[mask])

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

            ds_source = eval_data.input_ds["ds_source"]
            if "sample" in ds_source.dims:
                # Case when there are multiple source datasets present
                ds_source_array = (
                    eval_data.input_ds["ds_source"].unstack("sample").squeeze().values
                )
            else:
                # Case when there is a single source dataset present
                ds_source_array = np.array(
                    [ds_source.values.item() for _ in range(targ.sizes["shot"])]
                )

            ds = xr.Dataset(
                data_vars={
                    "Wtot_MJ_targ": targ,
                    "Wtot_MJ_pred": pred,
                    "error_abs_ts": error_abs_ts,
                    "error_rel_ts": error_rel_ts,
                    "error_abs_shot": error_abs_shot,
                    "error_rel_shot": error_rel_shot,
                }
            )
            ds = ds.assign_coords(ds_source=(EPISODE_DIM, ds_source_array))
            ds = ds.drop_vars("quantile", errors="ignore")
            ds = ds.drop_vars("input_batch", errors="ignore")
            return ds

        eval_suite = {
            "study_results": study_results,
        }

        return eval_suite
