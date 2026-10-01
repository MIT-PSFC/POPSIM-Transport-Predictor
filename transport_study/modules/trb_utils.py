"""Shared helpers for the study TrainRunBuilders."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from loguru import logger
from popsim.ml import DataLoader
from popsim.ml.dataloading import make_dataloaders
from popsim.ml.eval import EvalData, EvaluationSuite, batched_model_eval_and_loss
from popsim.ml.preprocess_utils import mask_to_largest_group_mask

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.config import config
from transport_study.orchestration.organize_data import (
    TrainingData,
    get_train_test_datasets,
    get_train_val_datasets,
    get_transfer_pretrain_datasets,
)


def trapezoid_dropna(y, x):
    """Trapezoid-integrate y over x ignoring NaN pairs, NaN when fewer than 2 valid points."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    y_valid, x_valid = y[mask], x[mask]
    sort_idx = np.argsort(x_valid)
    return np.trapezoid(y_valid[sort_idx], x_valid[sort_idx])


def integrate_error_over_time(error_ts: xr.DataArray, time_2d: xr.DataArray) -> xr.DataArray:
    """Per-shot time integral of a per-timeslice error, ignoring NaN-padded entries."""
    return xr.apply_ufunc(
        trapezoid_dropna,
        error_ts,
        time_2d,
        input_core_dims=[[TIME_DIM], [TIME_DIM]],
        vectorize=True,
    )


def make_exponential_adamw(optimizer_config: dict) -> optax.GradientTransformation:
    """AdamW on an exponentially decaying learning rate schedule."""
    schedule = optax.exponential_decay(
        init_value=optimizer_config["lr0"],
        transition_steps=optimizer_config["transition_steps"],
        decay_rate=optimizer_config["decay_rate"],
        end_value=optimizer_config["lrf"],
    )
    return optax.adamw(learning_rate=schedule, weight_decay=optimizer_config["weight_decay"])


def make_grouped_exponential_adamw(optimizer_config: dict) -> optax.GradientTransformation:
    """AdamW where selected submodules run a scaled copy of the exponential schedule.

    optimizer_config["submodule_lr_factors"] maps a module attribute name
    (e.g. "p_oh_predictor") to a multiplier on lr0/lrf. Any trainable leaf
    whose pytree path contains that attribute follows the scaled schedule,
    everything else the base one. Labeling is by pytree path, so it works both
    for the full-module partition and for the transfer-mode last-layer
    partition (frozen leaves are None in the trainable pytree and are never
    labeled). Without the key (or with all factors 1.0) this is exactly
    make_exponential_adamw.
    """
    factors = optimizer_config.get("submodule_lr_factors") or {}
    factors = {name: factor for name, factor in factors.items() if factor != 1.0}
    if not factors:
        return make_exponential_adamw(optimizer_config)

    def scaled_config(factor: float) -> dict:
        cfg = dict(optimizer_config)
        cfg["lr0"] = cfg["lr0"] * factor
        cfg["lrf"] = cfg["lrf"] * factor
        return cfg

    transforms = {"base": make_exponential_adamw(optimizer_config)}
    for name, factor in factors.items():
        transforms[name] = make_exponential_adamw(scaled_config(factor))

    def label_params(params):
        def label(path, _leaf):
            for name in factors:
                # GetAttrKey carries .name, DictKey carries .key
                if any(name in (getattr(key, "name", None), getattr(key, "key", None)) for key in path):
                    return name
            return "base"

        return jax.tree_util.tree_map_with_path(label, params)

    return optax.multi_transform(transforms, label_params)


def make_loss_eval_suite(loss_fn) -> EvaluationSuite:
    """Validation suite computing the loss mean plus a <=100 point sampled loss vector.

    The datasets are large and uploading the full per-sample loss vector to
    wandb every validation is too much data, so the vector is sorted and
    subsampled evenly - the distribution stays visible without all the data.

    A fresh closure is created and jitted per suite so the compiled forward
    persists across every validation of a training run while its cache stays
    isolated from other cases in the same process. Equinox keys
    eqx.filter_jit's cache off the wrapped function's identity, so jitting a
    shared module-level function would let unrelated cases collide in the
    same cache entry, which can raise instead of just retracing.
    """

    def _eval_and_loss(model, loss_fn, inputs, targets):
        return batched_model_eval_and_loss(model, loss_fn, inputs, targets)

    jit_eval_and_loss = eqx.filter_jit(_eval_and_loss)

    def eval_fn(inp: EvalData) -> dict:
        loss_vecs = []
        for batch in inp.dataloader:
            inputs, targets = batch.get_inputs_and_targets()
            loss_vecs.append(jit_eval_and_loss(inp.model, loss_fn, inputs, targets))
        loss_vec = jnp.concatenate(loss_vecs)
        # Drop padded duplicate samples from the pad_last validation dataloader
        loss_vec = loss_vec[: inp.dataloader.dataset.n_samples]
        loss_vec_mean = loss_vec.mean()
        # Sort and sample at most 100 points evenly for logging
        if loss_vec.shape[0] > 100:
            sorted_indices = jnp.argsort(loss_vec)
            selected_indices = sorted_indices[jnp.linspace(0, loss_vec.shape[0] - 1, num=100, dtype=int)]
            loss_vec = loss_vec[selected_indices]
        return {
            "mean": loss_vec_mean,
            "vec": loss_vec,
        }

    return {"loss": eval_fn}


def mask_to_largest_contiguous_segment(ds: xr.Dataset, training_vars: list[str]) -> xr.Dataset:
    """Keep only each episode's longest contiguous run of non-NaN training vars.

    Everything outside that run (including the time coordinate) is set to NaN,
    which the dataloader treats as leading/trailing padding. Uses POPSIM's
    mask_to_largest_group_mask rather than force_drop_nans because the latter's
    ds.where() would broadcast per-shot vars (hazard etc.) against time.
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


def resolve_case_datasets(
    dataloader_config: dict,
    study_type: str,
) -> tuple[xr.Dataset, xr.Dataset, list[str], xr.Dataset | None]:
    """Select and prepare the train/val datasets for a case, shared by every study TRB.

    Selection by domain_adaptation:
    - transfer_pretrain: train on historic data only, plus a combined
      historic + target dataset for the normalizer fit
    - None with historic data: standard train/val split for hyperparameter tuning
    - None with exnihilo: target-only training set (verified to hold no source data)
    - weighted / addition / transfer: historic + target training set, target test set as val

    Every branch returns the validation set as the test set: checkpoint
    selection and final evaluation share it (no separate test split, slightly
    optimistic but consistent across cases).

    Preparation applied to both datasets:
    - drop the time_idx coordinate (duplicate values across shots break
      groupby("shot") on reassembly, the dataloader uses "time" instead)
    - broadcast the per-shot ds_source_idx against time as a float so the
      dataloader can slice, segment, and NaN-pad it like the other inputs,
      and append it to input_vars (the modules consume it to select
      per-device normalization stats)

    Returns (ds_train, ds_val, input_vars, normalizer_fit_ds), the last one
    None except for transfer_pretrain.
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
            study_type=study_type,
        )
    elif dataloader_config.get("domain_adaptation") is None:
        logger.info("Using standard learning dataloader")
        if not training_data.exnihilo:
            ds_train, ds_val = get_train_val_datasets(
                training_data=training_data,
                study_type=study_type,
            )
        else:
            ds_train, ds_val = get_train_test_datasets(
                training_data=training_data,
                domain_adaptation=None,
                num_target_shots=dataloader_config["num_target_shots"],
                target_test_set_size=dataloader_config.get("target_test_set_size", None),
                study_type=study_type,
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
            study_type=study_type,
        )

    ds_train = ds_train.drop_vars(TIME_DIM, errors="ignore")
    ds_val = ds_val.drop_vars(TIME_DIM, errors="ignore")

    input_vars = list(dataloader_config["input_vars"])
    if "ds_source_idx" not in input_vars:
        input_vars.append("ds_source_idx")
    for ds in (ds_train, ds_val):
        ds["ds_source_idx"] = ds["ds_source_idx"].broadcast_like(ds["ip_MA"]).astype(ds["ip_MA"].dtype)

    return ds_train, ds_val, input_vars, normalizer_fit_ds


def get_time_dep_dataloaders(
    dataloader_config: dict,
    study_type: str,
) -> tuple[xr.Dataset, DataLoader, DataLoader, DataLoader]:
    """Dataset and dataloaders shared by the time-dependent (state-carrying) TRBs.

    Used by the power balance and transport predictor TrainRunBuilders, which
    differ only in the study_type their datasets are prepared with.
    """
    ds_train, ds_val, input_vars, normalizer_fit_ds = resolve_case_datasets(dataloader_config, study_type)

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
    if normalizer_fit_ds is not None:
        train_dl.normalizer_fit_ds = normalizer_fit_ds
    # The validation set doubles as the test set (see resolve_case_datasets)
    return ds_val, train_dl, val_dl, val_dl
