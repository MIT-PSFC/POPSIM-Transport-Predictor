from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from loguru import logger
from popsim.ml import DataLoader, IntegralLoss, TrainConfig, TrainRunBuilder
from popsim.ml.checkpointing import create_default_checkpoint_manager, restore_model
from popsim.ml.dataloading import make_dataloaders
from popsim.ml.eval import EvalData, EvaluationSuite
from popsim.ml.preprocess_utils import mask_to_largest_group_mask

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.config import config
from transport_study.modules.normalization import make_normalizer
from transport_study.modules.power_balance.module import (
    PowerBalanceEnv,
    PowerBalanceScalingLaw,
    PowerBalanceSciML,
    PowerBalanceTransformer,
    PowerBalanceUnstructuredNN,
)
from transport_study.modules.power_balance.p_oh.trb import OhmicPowerTRB
from transport_study.modules.power_balance.p_rad.trb import RadiatedPowerTRB
from transport_study.modules.trb_utils import (
    integrate_error_over_time,
    make_exponential_adamw,
    make_loss_eval_suite,
)
from transport_study.orchestration.organize_data import (
    TrainingData,
    get_train_test_datasets,
    get_train_val_datasets,
    get_transfer_pretrain_datasets,
)

STUDY_TYPE = "power_balance_transfer"


def mask_to_largest_contiguous_segment(ds: xr.Dataset, training_vars: list[str]) -> xr.Dataset:
    """Keep only each episode's longest contiguous run of non-NaN training vars.

    Everything outside that run (including the time coordinate) is set to NaN,
    which the dataloader treats as leading/trailing padding. Uses POPSIM's
    mask_to_largest_group_mask rather than force_drop_nans because the latter's
    ds.where() would broadcast per-shot vars (performance etc.) against time.
    """

    def _var_nan(da: xr.DataArray) -> xr.DataArray:
        extra_dims = [d for d in da.dims if d not in (EPISODE_DIM, TIME_DIM)]
        return da.isnull().any(dim=extra_dims) if extra_dims else da.isnull()

    nan_mask = _var_nan(ds[training_vars[0]])
    for var in training_vars[1:]:
        nan_mask = nan_mask | _var_nan(ds[var])
    keep = mask_to_largest_group_mask(~nan_mask, EPISODE_DIM, TIME_DIM)

    out = ds.copy()
    for name, da in ds.data_vars.items():
        if TIME_DIM in da.dims:
            out[name] = da.where(keep)
    if TIME_DIM in ds[TIME_COORD].dims:
        out[TIME_COORD] = ds[TIME_COORD].where(keep)
    return out


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

        training_data = dataloader_config["training_data"]

        if isinstance(training_data, dict):  # when the config is passed from WandB, it's a dict
            training_data = TrainingData(**training_data)

        normalizer_fit_ds = None
        if dataloader_config.get("domain_adaptation") == "transfer_pretrain":
            logger.info("Using transfer pretrain dataloader (trains on historic data, normalizer fit on historic + target shots)")
            ds_train, normalizer_fit_ds, ds_val = get_transfer_pretrain_datasets(
                training_data=training_data,
                num_target_shots=dataloader_config["num_target_shots"],
                target_test_set_size=dataloader_config.get("target_test_set_size", None),
                study_type=STUDY_TYPE,
            )
        elif dataloader_config.get("domain_adaptation") is None:
            logger.info("Using standard learning dataloader")
            if not training_data.exnihilo:
                ds_train, ds_val = get_train_val_datasets(
                    training_data=training_data,
                    study_type=STUDY_TYPE,
                )
            else:
                ds_train, ds_val = get_train_test_datasets(
                    training_data=training_data,
                    domain_adaptation=None,
                    num_target_shots=dataloader_config["num_target_shots"],
                    target_test_set_size=dataloader_config.get("target_test_set_size", None),
                    study_type=STUDY_TYPE,
                )
                # Double check there's no source (non-target) data anywhere in here
                non_target = set(config.dataset_paths.keys()) - {config.target_device}
                if any((ds_train["ds_source"] == src).any() for src in non_target):
                    raise ValueError(
                        "Historic data found in training set for exnihilo training_data option. Please check the dataset construction logic."
                    )
        else:
            logger.info(f"Using transfer learning dataloader with domain adaptation {dataloader_config['domain_adaptation']}")
            ds_train, ds_val = get_train_test_datasets(
                training_data=training_data,
                domain_adaptation=dataloader_config["domain_adaptation"],
                num_target_shots=dataloader_config["num_target_shots"],
                target_test_set_size=dataloader_config.get("target_test_set_size", None),
                study_type=STUDY_TYPE,
            )

        # Drop time_idx as a shared coordinate — it has duplicate values across shots and
        # causes groupby("shot") to fail when reassembling. The dataloader uses "time" instead.
        ds_train = ds_train.drop_vars(TIME_DIM, errors="ignore")
        ds_val = ds_val.drop_vars(TIME_DIM, errors="ignore")

        # The modules take ds_source_idx as an input (it selects per-device
        # normalization stats), but it is stored per shot. Broadcast it against
        # time so the dataloader can slice and segment it like the other inputs
        input_vars = list(dataloader_config["input_vars"])
        if "ds_source_idx" not in input_vars:
            input_vars.append("ds_source_idx")
        for ds in (ds_train, ds_val):
            # Float dtype so the dataloader can NaN-pad it like the other inputs
            ds["ds_source_idx"] = ds["ds_source_idx"].broadcast_like(ds["Ip_MA"]).astype(ds["Ip_MA"].dtype)

        if "state_vars" in dataloader_config.keys():
            # Validation samples are whole episodes, so a mid-shot time gap
            # (NaN slices after the uniform-timebase reindex) would either be
            # stitched over, handing the Euler stepper a huge dt, or drop the
            # whole episode under drop_segment. Keep only each episode's
            # longest contiguous non-NaN run so val simulates a single
            # gap-free window
            val_vars = sorted(
                v
                for v in {
                    *input_vars,
                    *dataloader_config["target_vars"],
                    *dataloader_config["state_vars"],
                    *(dataloader_config.get("extra_vars") or []),
                }
                if v in ds_val
            )
            ds_val = mask_to_largest_contiguous_segment(ds_val, val_vars)

            segment_length = [
                dataloader_config.get("segment_length_train", None),
                dataloader_config.get("segment_length_val", None),
            ]
            segment_overlap = [
                dataloader_config.get("segment_overlap_train", 0) or 0,
                dataloader_config.get("segment_overlap_val", 0) or 0,
            ]
        else:
            # Segments only apply to time-dependent (state-carrying) dataloaders
            segment_length = None
            segment_overlap = 0

        train_dl, val_dl = make_dataloaders(
            datasets=(ds_train, ds_val),
            time_coord=TIME_COORD,
            episode_coord=EPISODE_DIM,
            input_vars=input_vars,
            target_vars=dataloader_config["target_vars"],
            extra_vars=dataloader_config.get("extra_vars", None),
            state_init_vars=dataloader_config.get("state_vars", None),
            batch_size=dataloader_config.get("batch_size", None),
            segment_length=segment_length,
            segment_overlap=segment_overlap,
            shuffle=[True, False],
            convert_xr_to_jnp=False,  # Needed to keep the coords for calculating loss
            # The datasets are reindexed to a uniform 1 kHz grid with NaN at
            # missing times (organize_data.reindex_to_uniform_timebase), so
            # drop_segment discards train segments spanning a time gap and the
            # Euler stepper never sees dt larger than the nominal timebase.
            # Val episodes were already masked to their longest contiguous
            # run above, drop_slice_any there only clears the leading and
            # trailing padding.
            nan_handling=["drop_segment", "drop_slice_any"],
            # Keep every batch the same shape so the jitted train step never
            # retraces on a ragged final batch (whose static xr metadata is not
            # comparable across calls). Train drops the ragged tail (reshuffled
            # every epoch, so no data is permanently lost), val pads it and
            # consumers trim the duplicates.
            drop_last=[True, False],
            pad_last=[False, True],
        )
        # Transfer pretrain fits the normalizer on more data than it trains on
        # (historic + target shots). model_init reads this attribute off the
        # train dataloader, every other case fits on train_dl.ds itself
        # Yes I know this looks stupid but it's a fairly simlple way to pass the extra dataset
        # to the model_init without changing everything else
        if normalizer_fit_ds is not None:
            train_dl.normalizer_fit_ds = normalizer_fit_ds
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

        def _build_module(train_dl: DataLoader, model_init_config: dict) -> Any:
            model_type = model_init_config["model_type"]
            # Fit normalization stats from the training data only. When a
            # transfer checkpoint will overwrite the module anyway, skip the
            # fit (a CORAL fit on a handful of target shots is ill-conditioned
            # and the restored stats, fitted on historic + target shots by the
            # transfer_pretrain prereq case, are the correct ones).
            # transfer_pretrain dataloaders carry that combined fit dataset as
            # an attribute (see get_dataloaders)
            n_devices = len(config.ds_source_to_idx)
            if model_init_config.get("transfer_checkpoint"):
                fit_ds = None
            else:
                fit_ds = getattr(train_dl, "normalizer_fit_ds", train_dl.ds)
            normalizer = make_normalizer(model_init_config["data_normalization"], fit_ds, n_devices)
            if model_type in ["scaling_law", "sciml"]:
                p_oh_config = model_init_config["submodules"]["p_oh_predictor"]
                if isinstance(p_oh_config, TrainConfig):
                    p_oh_config = p_oh_config.model_dump()
                p_oh_predictor = OhmicPowerTRB.model_init(train_dl, p_oh_config["model_init_config"])
                p_rad_config = model_init_config["submodules"]["p_rad_predictor"]
                if isinstance(p_rad_config, TrainConfig):
                    p_rad_config = p_rad_config.model_dump()
                p_rad_predictor = RadiatedPowerTRB.model_init(train_dl, p_rad_config["model_init_config"])
                if model_init_config["restore_submodules"]:
                    p_oh_manager = create_default_checkpoint_manager(p_oh_config["checkpoint_dir"])
                    p_oh_predictor = restore_model(p_oh_manager, p_oh_predictor)
                    p_rad_manager = create_default_checkpoint_manager(p_rad_config["checkpoint_dir"])
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
                    normalizer=normalizer,
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
                    normalizer=normalizer,
                    min_val=model_init_config.get("min_val", None),
                    max_val=model_init_config.get("max_val", None),
                    prng_seed=model_init_config.get("prng_seed", 42),
                )
            elif model_type == "transformer":
                module = PowerBalanceTransformer.init(
                    d_model=model_init_config["d_model"],
                    num_heads=model_init_config["num_heads"],
                    history_len=model_init_config["history_len"],
                    nn_width=model_init_config["nn_width"],
                    nn_depth=model_init_config["nn_depth"],
                    normalizer=normalizer,
                    min_val=model_init_config.get("min_val", None),
                    max_val=model_init_config.get("max_val", None),
                    prng_seed=model_init_config.get("prng_seed", 42),
                )
            else:
                raise ValueError(f"Invalid model type: {model_type}")

            return module

        module = _build_module(train_dl, model_init_config)

        if model_init_config.get("freeze_submodules", False):
            freeze_submodules = ["p_oh_predictor", "p_rad_predictor"]
        else:
            freeze_submodules = []

        env = PowerBalanceEnv(
            module=module,
            domain_adaptation=model_init_config["domain_adaptation"],
            freeze_submodules=freeze_submodules,
        )

        if model_init_config.get("transfer_checkpoint", False):
            transfer_manager = create_default_checkpoint_manager(model_init_config["transfer_checkpoint"])
            env = restore_model(transfer_manager, env)
            # Restoring the whole env overwrote the freshly restored submodule weights, restore them again from their own checkpoints
            if model_init_config["model_type"] in ["scaling_law", "sciml"]:
                p_oh_config = model_init_config["submodules"]["p_oh_predictor"]
                if isinstance(p_oh_config, TrainConfig):
                    p_oh_config = p_oh_config.model_dump()
                p_rad_config = model_init_config["submodules"]["p_rad_predictor"]
                if isinstance(p_rad_config, TrainConfig):
                    p_rad_config = p_rad_config.model_dump()

                p_oh_manager = create_default_checkpoint_manager(p_oh_config["checkpoint_dir"])
                p_oh_restored = restore_model(p_oh_manager, env.module.p_oh_predictor)
                p_rad_manager = create_default_checkpoint_manager(p_rad_config["checkpoint_dir"])
                p_rad_restored = restore_model(p_rad_manager, env.module.p_rad_predictor)
                env = eqx.tree_at(
                    lambda e: (e.module.p_oh_predictor, e.module.p_rad_predictor),
                    env,
                    (p_oh_restored, p_rad_restored),
                )

            logger.debug(f"Restoring module from transfer learning pretrained checkpoint\n{model_init_config['transfer_checkpoint']}")

        # This restoration of the main module is separate from the transfer learning restoration
        # This would get the post-trained model, AFTER transfer learning has already been done
        if model_init_config.get("restore_main_module", False):
            manager = create_default_checkpoint_manager(model_init_config["checkpoint_dir"])
            env = restore_model(manager, env)
            logger.debug(f"Restoring module from post-training checkpoint\n{model_init_config['checkpoint_dir']}")
        else:
            logger.debug("Not restoring main module from post-training checkpoint.")

        return env

    @staticmethod
    def _make_wtot_loss_fn(loss_config: dict, use_huber: bool) -> IntegralLoss:
        """Device-weighted loss on Wtot_MJ_pred, wrapped for time integration.

        use_huber selects the training loss (huber, with the swept
        huber_delta) or the delta-free validation loss (plain absolute error),
        so the sweep metric val/loss.mean cannot be gamed by shrinking delta.
        """
        if "device_weights" not in loss_config:
            device_weights = dict.fromkeys(config.dataset_paths, 1.0)
        else:
            device_weights = loss_config["device_weights"]

        def loss_fn(pred, targ):
            if use_huber:
                errors = optax.huber_loss(
                    pred.Wtot_MJ_pred,
                    targ["Wtot_MJ"].data,
                    delta=loss_config["huber_delta"],
                )
            else:
                errors = jnp.abs(pred.Wtot_MJ_pred - targ["Wtot_MJ"].data)

            # Build per-sample weights from device labels
            ds_source_idx = targ["ds_source_idx"].data
            sample_weights = jnp.ones(ds_source_idx.shape, dtype=errors.dtype)
            for device, weight in device_weights.items():
                sample_weights = jnp.where(
                    ds_source_idx == config.ds_source_to_idx[device],
                    weight,
                    sample_weights,
                )

            # Broadcast sample weights to match the error shape if needed
            while sample_weights.ndim < errors.ndim:
                sample_weights = sample_weights[..., None]

            return jnp.mean(sample_weights * errors)

        return IntegralLoss(loss_fn)

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        return PowerBalanceTRB._make_wtot_loss_fn(loss_config, use_huber=True)

    @staticmethod
    def get_val_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        return PowerBalanceTRB._make_wtot_loss_fn(loss_config, use_huber=False)

    @staticmethod
    def get_val_eval_suite(suite_config) -> EvaluationSuite | None:
        """Validation suite computing the delta-free loss (sweep metric val/loss.mean)."""
        if suite_config is None:
            return None
        return make_loss_eval_suite(PowerBalanceTRB.get_val_loss_fn(suite_config["loss_config"]))

    @staticmethod
    def get_optimizer(config: dict) -> optax.GradientTransformation:
        return make_exponential_adamw(config)

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
            pred = eval_data.output_ds["output.Wtot_MJ_pred"].unstack("sample").squeeze()
            time_2d = eval_data.input_ds[TIME_COORD].unstack("sample").squeeze()

            # time-dependent modules modify the time dimension name, change it back to avoid confusion
            targ = targ.rename({TIME_DIM + "_input": TIME_DIM})
            pred = pred.rename({TIME_DIM + "_input": TIME_DIM})
            time_2d = time_2d.rename({TIME_DIM + "_input": TIME_DIM})

            # Get relative error on a per-timeslice basis
            error_abs_ts = xr.apply_ufunc(np.abs, pred - targ)
            error_rel_ts = error_abs_ts / (xr.apply_ufunc(np.abs, targ) + 0.1)

            # Integrate absolute error over time for each shot, ignoring NaN-padded entries
            error_abs_shot = integrate_error_over_time(error_abs_ts, time_2d)
            error_rel_shot = integrate_error_over_time(error_rel_ts, time_2d)

            ds_source = eval_data.input_ds["ds_source"]
            if "sample" in ds_source.dims:
                # Case when there are multiple source datasets present
                ds_source_array = eval_data.input_ds["ds_source"].unstack("sample").squeeze().values
            else:
                # Case when there is a single source dataset present
                ds_source_array = np.array([ds_source.values.item() for _ in range(targ.sizes["shot"])])

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

        if config:
            eval_suite = {
                "study_results": study_results,
            }
            return eval_suite
        else:
            # Hyperparameter tuning, do not run test evals
            return None
