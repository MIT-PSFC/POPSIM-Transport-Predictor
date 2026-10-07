"""Shared helpers for the study TrainRunBuilders."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from loguru import logger
from popsim.ml import DataLoader, TrainConfig
from popsim.ml.checkpointing import create_default_checkpoint_manager, restore_model
from popsim.ml.dataloading import XarrayPreppedDataset, make_dataloaders
from popsim.ml.eval import EvalData, EvaluationSuite, batched_model_eval_and_loss
from popsim.ml.preprocess_utils import mask_to_largest_group_mask
from transport_validation_datasets.machine.generic import UNIFORM_TIMEBASE_DT

from transport_study import EPISODE_DIM, RADIAL_DIM, TIME_COORD, TIME_DIM
from transport_study.config import config
from transport_study.orchestration.organize_data import (
    TrainingData,
    concat_with_nan_padding,
    get_ds,
    get_train_test_datasets,
    get_train_val_datasets,
    get_transfer_pretrain_datasets,
    merge_parts,
)
from transport_study.orchestration.target_shots import TargetSplit

# Floor on the per-timeslice profile peak used for peak normalization.
# In channel units (1e20 m^-3 for ne, keV for Te) any real profile peaks far above it,
# it only guards degenerate targets from blowing up the 1/scale division
PROFILE_SCALE_FLOOR = 1e-2

# Chi is the profile residual in units of the GP-fit error bar.
# Each error bar is floored at this percentile of its own peak-normalized distribution per device,
# so a single overconfident fit point cannot carry unbounded weight
CHI_SIGMA_FLOOR_PERCENTILE = 5.0
# Profile gradients count below this rho, in the training losses, the validation chi and the analysis chi alike.
# Beyond it the GP fits extrapolate into the pedestal and scrape-off layer, where the measured gradients are unreliable
GRAD_RHO_MAX = 0.9
# Floor added to |target| in the relative error of the scalar study results, in the signal's units.
# Each sits below the signal's typical size so the error stays relative,
# Wtot medians are 0.03-0.05 MJ on C-Mod and MAST
SCALAR_REL_ERROR_FLOORS = {
    "energy_mhd_MJ": 0.01,
    "power_ohm_MW": 0.1,
    "power_radiated_MW": 0.1,
}

# Profile channel -> (value error bar, gradient error bar) chi divides by
CHI_ERROR_VARS = {
    "n_e_1e20": ("n_e_1e20_error", "n_e_1e20_gradient_error"),
    "t_e_keV": ("t_e_keV_error", "t_e_keV_gradient_error"),
}

# Shot number of the room-keeper episode segmented_train_dataloader adds to each training part, below every real shot
ROOM_KEEPER_SHOT = -1


def peak_scale(targ: jnp.ndarray) -> jnp.ndarray:
    """Per-timeslice peak of a target profile (..., rho), floored, the scale every profile loss normalizes by."""
    return jnp.maximum(jnp.max(jnp.abs(targ), axis=-1, keepdims=True), PROFILE_SCALE_FLOOR)


def to_mid(arr: jnp.ndarray) -> jnp.ndarray:
    """Grid-point signal (..., rho) averaged to the rho midpoints, where finite-difference gradients live."""
    return 0.5 * (arr[..., :-1] + arr[..., 1:])


def per_sample_device_values(ds_source_idx: jnp.ndarray, values_by_device: dict[str, float], fill: float) -> jnp.ndarray:
    """values_by_device[device] at every sample of that device and fill elsewhere, shaped like ds_source_idx."""
    values = jnp.full(jnp.shape(ds_source_idx), fill)
    for device, value in values_by_device.items():
        values = jnp.where(ds_source_idx == config.ds_source_to_idx[device], value, values)
    return values


def per_sample_sigma_floor(ds_source_idx: jnp.ndarray, sigma_floors: dict[str, dict[str, float]], error_var: str) -> jnp.ndarray:
    """The chi_sigma_floors entry of error_var for each sample's device, shaped (..., 1) to broadcast over rho.

    NaN for a device without floors, so a missing entry fails loudly.
    """
    floors = {device: device_floors[error_var] for device, device_floors in sigma_floors.items()}
    return per_sample_device_values(ds_source_idx, floors, jnp.nan)[..., None]


def chi_value(pred: jnp.ndarray, targ: jnp.ndarray, sigma: jnp.ndarray, floor: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
    """int_0^1 |pred - targ| / max(sigma, floor) drho on peak-normalized profiles.

    Profiles and sigma are (..., rho), floor is the peak-normalized sigma floor broadcast to (..., 1).
    """
    scale = peak_scale(targ)
    chi = jnp.abs(pred - targ) / scale / jnp.maximum(sigma / scale, floor)
    return jnp.trapezoid(chi, x=rho, axis=-1)


def chi_gradient(
    pred: jnp.ndarray, targ: jnp.ndarray, grad_targ: jnp.ndarray, grad_sigma: jnp.ndarray, floor: jnp.ndarray, rho: jnp.ndarray
) -> jnp.ndarray:
    """int_0^GRAD_RHO_MAX |pred' - targ'| / max(sigma', floor) drho on peak-normalized profiles.

    pred' is the finite difference of the prediction at the rho midpoints,
    targ' and sigma' are the GP-fit gradient and its error bar averaged to the same midpoints.
    """
    scale = peak_scale(targ)
    rho_mid = to_mid(rho)
    grad_pred = jnp.diff(pred / scale, axis=-1) / jnp.diff(rho, axis=-1)
    chi = jnp.abs(grad_pred - to_mid(grad_targ) / scale) / jnp.maximum(to_mid(grad_sigma) / scale, floor)
    return jnp.trapezoid((rho_mid < GRAD_RHO_MAX) * chi, x=rho_mid, axis=-1)


def huber_profile_loss(
    pred: jnp.ndarray,
    targ: jnp.ndarray,
    grad_targ: jnp.ndarray,
    rho: jnp.ndarray,
    huber_delta: float,
    huber_delta_grad: float,
    gradient_weight: float,
) -> jnp.ndarray:
    """Training loss of one profile channel per timeslice, the profile and transport studies' shared form.

    int_0^1 huber(pred - targ) drho on peak-normalized profiles,
    plus gradient_weight times int huber(pred' - targ') drho below GRAD_RHO_MAX,
    pred' the finite difference of the prediction at the rho midpoints and targ' the GP-fit gradient averaged there.
    Profiles are (..., rho), the gradient huber transition has its own delta
    since normalized gradients are larger than normalized values (a pedestal can reach d/drho of order 10).
    """
    scale = peak_scale(targ)
    rho_mid = to_mid(rho)
    value_err = optax.huber_loss(pred / scale - targ / scale, delta=huber_delta)
    grad_pred = jnp.diff(pred / scale, axis=-1) / jnp.diff(rho, axis=-1)
    grad_err = optax.huber_loss(grad_pred - to_mid(grad_targ) / scale, delta=huber_delta_grad)
    value_term = jnp.trapezoid(value_err, x=rho, axis=-1)
    grad_term = jnp.trapezoid((rho_mid < GRAD_RHO_MAX) * grad_err, x=rho_mid, axis=-1)
    return value_term + gradient_weight * grad_term


def chi_profile_loss(
    pred: jnp.ndarray,
    targ: jnp.ndarray,
    sigma: jnp.ndarray,
    grad_targ: jnp.ndarray,
    grad_sigma: jnp.ndarray,
    value_floor: jnp.ndarray,
    grad_floor: jnp.ndarray,
    rho: jnp.ndarray,
    gradient_weight: float,
) -> jnp.ndarray:
    """Validation loss of one profile channel per timeslice: value chi plus gradient_weight times gradient chi."""
    value_chi = chi_value(pred, targ, sigma, value_floor, rho)
    grad_chi = chi_gradient(pred, targ, grad_targ, grad_sigma, grad_floor, rho)
    return value_chi + gradient_weight * grad_chi


def chi_sigma_floors(device: str) -> dict[str, float]:
    """The peak-normalized error-bar floor of every CHI_ERROR_VARS variable for one device.

    The CHI_SIGMA_FLOOR_PERCENTILE-th percentile of the positive normalized error bars over the device's fresh profiles,
    a property of the fits rather than of any case.
    """
    ds = get_ds(device, "profile_transfer")
    floors = {}
    for channel, error_vars in CHI_ERROR_VARS.items():
        scale = np.asarray(peak_scale(ds[channel].transpose(..., RADIAL_DIM).values))
        for var in error_vars:
            sigma = (ds[var].transpose(..., RADIAL_DIM).values / scale).ravel()
            sigma_positive = sigma[np.isfinite(sigma) & (sigma > 0)]
            if not sigma_positive.size:
                raise ValueError(f"{device} has no positive {var} to floor the chi error bars with")
            floors[var] = float(np.percentile(sigma_positive, CHI_SIGMA_FLOOR_PERCENTILE))
    logger.info(f"Chi sigma floors for {device}: " + ", ".join(f"{var}={floor:.5f}" for var, floor in floors.items()))
    return floors


def restore_from_checkpoint(module, checkpoint_dir: str):
    """module with every leaf restored from the best checkpoint in checkpoint_dir, module supplies the pytree structure.

    A missing dir raises instead of being created by the checkpoint manager,
    since it may sit in another study's working dir, which a restore must never write (see orchestration/lineage.py).
    """
    if not Path(checkpoint_dir).is_dir():
        raise FileNotFoundError(f"No checkpoint dir to restore from at {checkpoint_dir}")
    return restore_model(create_default_checkpoint_manager(checkpoint_dir), module)


def submodule_config_dict(submodule_config: TrainConfig | dict) -> dict:
    """A submodule prereq case's train config as a dict.

    make_train_config nests TrainConfig objects, a config reloaded from yaml or wandb holds plain dicts.
    """
    return submodule_config.model_dump() if isinstance(submodule_config, TrainConfig) else submodule_config


def attach_normalizer_fit_ds(train_dl: DataLoader, normalizer_fit_ds: xr.Dataset | None) -> None:
    """Carry a transfer_pretrain case's normalizer fit dataset on its train dataloader, see normalizer_fit_dataset."""
    if normalizer_fit_ds is not None:
        train_dl.normalizer_fit_ds = normalizer_fit_ds


def normalizer_fit_dataset(train_dl: DataLoader, model_init_config: dict) -> xr.Dataset | None:
    """The dataset a model_init fits its normalizer statistics on, None when a transfer checkpoint will overwrite them.

    A transfer_pretrain case trains on historic data only, but fits its normalizer on historic + target shots,
    so the transfer case it pretrains for inherits target-aware statistics through the checkpoint restore.
    model_init only receives the train dataloader, so that fit dataset rides on it as an attribute (attach_normalizer_fit_ds).
    Every other case fits on its own training data.
    A transfer case skips the fit: one on its few target shots would be ill-conditioned,
    and the statistics restored from its transfer_pretrain prereq are the right ones.
    """
    if model_init_config.get("transfer_checkpoint"):
        return None
    return getattr(train_dl, "normalizer_fit_ds", train_dl.ds)


def unstack_samples(da: xr.DataArray) -> xr.DataArray:
    """An evaluation variable back on its (shot, time_idx[, rho]) dims.

    Unstacks the sample MultiIndex and drops the length-1 batch dims, keeping the shot and time dims even at length 1.
    Time-dependent dataloaders suffix every input-side dim with _input,
    the suffix is removed so targets and predictions share one grid.
    """
    da = da.unstack("sample")
    protected_dims = {EPISODE_DIM, TIME_DIM, TIME_DIM + "_input"}
    squeeze_dims = [dim for dim, size in da.sizes.items() if size == 1 and dim not in protected_dims]
    if squeeze_dims:
        da = da.squeeze(dim=squeeze_dims, drop=True)
    renames = {dim: dim.removesuffix("_input") for dim in da.dims if isinstance(dim, str) and dim.endswith("_input")}
    return da.rename(renames) if renames else da


def ds_source_per_shot(ds_source: xr.DataArray, n_shots: int) -> np.ndarray:
    """The device of every shot, from the ds_source coordinate of an evaluation input dataset.

    A single-device evaluation set carries ds_source as a scalar coordinate.
    Otherwise it lives on the sample MultiIndex, and unstacking fills the absent (shot, time) pairs with NaN.
    """
    if "sample" not in ds_source.dims:
        return np.full(n_shots, ds_source.values.item(), dtype=object)
    unstacked = ds_source.unstack("sample").transpose(EPISODE_DIM, ...)
    values = unstacked.values.reshape(n_shots, -1)
    mask_valid = unstacked.notnull().values.reshape(n_shots, -1)
    return np.array([row[np.flatnonzero(mask_row)[0]] for row, mask_row in zip(values, mask_valid, strict=True)], dtype=object)


def scalar_study_results(eval_data: EvalData, signal: str, pred_var: str) -> xr.Dataset:
    """Final study results of a scalar signal, keeping the shot and ds_source coordinates.

    Target vs predicted signal, absolute and relative error per timeslice,
    and their per-shot time integrals (NaN-padded entries ignored).
    A non-finite prediction diverged: its errors are NaN, never inf,
    and error_diverged_ts (1 diverged, 0 finite, NaN without a target) counts it instead.
    """
    targ = unstack_samples(eval_data.input_ds[signal])
    pred = unstack_samples(eval_data.output_ds[pred_var])
    time_2d = unstack_samples(eval_data.input_ds[TIME_COORD])

    mask_pred_finite = xr.apply_ufunc(np.isfinite, pred)
    error_diverged_ts = (~mask_pred_finite).astype(float).where(targ.notnull())
    error_abs_ts = xr.apply_ufunc(np.abs, pred - targ).where(mask_pred_finite)
    error_rel_ts = error_abs_ts / (xr.apply_ufunc(np.abs, targ) + SCALAR_REL_ERROR_FLOORS[signal])
    ds = xr.Dataset(
        data_vars={
            f"{signal}_targ": targ,
            f"{signal}_pred": pred,
            "error_abs_ts": error_abs_ts,
            "error_rel_ts": error_rel_ts,
            "error_diverged_ts": error_diverged_ts,
            "error_abs_shot": integrate_error_over_time(error_abs_ts, time_2d),
            "error_rel_shot": integrate_error_over_time(error_rel_ts, time_2d),
        }
    )
    ds_source = ds_source_per_shot(eval_data.input_ds["ds_source"], ds.sizes[EPISODE_DIM])
    return ds.assign_coords(ds_source=(EPISODE_DIM, ds_source)).drop_vars(["quantile", "input_batch"], errors="ignore")


def target_device_idx() -> int:
    """Global device index of the target device, the reference the CORAL methods align every device to."""
    return config.ds_source_to_idx[config.target_device]


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


def path_names(path: tuple) -> set:
    """The attribute names and dict keys along a pytree path."""
    # GetAttrKey carries .name, DictKey carries .key
    return {getattr(key, "name", None) for key in path} | {getattr(key, "key", None) for key in path}


def make_exponential_adamw(optimizer_config: dict, no_decay_names: tuple[str, ...] = ()) -> optax.GradientTransformation:
    """AdamW on an exponentially decaying learning rate schedule.

    The rate falls from lr0 by decay_rate every transition_steps and holds at lr0 * lrf_frac.
    lrf_frac <= 1 keeps the final rate at or below lr0
    (optax treats end_value as a floor, an end_value above lr0 would run the whole schedule there).
    Leaves whose pytree path holds one of no_decay_names skip the weight decay.
    """
    lr0 = optimizer_config["lr0"]
    lrf = lr0 * optimizer_config["lrf_frac"]
    schedule = optax.exponential_decay(
        init_value=lr0,
        transition_steps=optimizer_config["transition_steps"],
        decay_rate=optimizer_config["decay_rate"],
        end_value=lrf,
    )

    def decay_mask(params):
        return jax.tree_util.tree_map_with_path(lambda path, _leaf: not (path_names(path) & set(no_decay_names)), params)

    return optax.adamw(
        learning_rate=schedule,
        weight_decay=optimizer_config["weight_decay"],
        mask=decay_mask if no_decay_names else None,
    )


def make_grouped_exponential_adamw(optimizer_config: dict, no_decay_names: tuple[str, ...] = ()) -> optax.GradientTransformation:
    """AdamW where selected submodules run a scaled copy of the exponential schedule.

    optimizer_config["submodule_lr_factors"] maps a module attribute name
    (e.g. "p_oh_predictor") to a multiplier on the whole schedule. Any trainable leaf
    whose pytree path contains that attribute follows the scaled schedule,
    everything else the base one. Labeling is by pytree path, so it works both
    for the full-module partition and for the transfer-mode last-layer
    partition (frozen leaves are None in the trainable pytree and are never
    labeled). Without the key (or with all factors 1.0) this is exactly
    make_exponential_adamw.
    no_decay_names passes through to every group.
    """
    factors = optimizer_config.get("submodule_lr_factors") or {}
    factors = {name: factor for name, factor in factors.items() if factor != 1.0}
    if not factors:
        return make_exponential_adamw(optimizer_config, no_decay_names)

    transforms = {"base": make_exponential_adamw(optimizer_config, no_decay_names)}
    for name, factor in factors.items():
        # lrf is a fraction of lr0, so scaling lr0 scales the whole schedule
        scaled_config = {**optimizer_config, "lr0": optimizer_config["lr0"] * factor}
        transforms[name] = make_exponential_adamw(scaled_config, no_decay_names)

    def label_params(params):
        def label(path, _leaf):
            names_on_path = path_names(path)
            return next((name for name in factors if name in names_on_path), "base")

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
) -> tuple[list[xr.Dataset], xr.Dataset, list[str], xr.Dataset | None]:
    """Select and prepare the training parts and the val dataset for a case, shared by every study TRB.

    Selection by domain_adaptation:
    - transfer_pretrain: train on historic data only, plus a combined
      historic + target dataset for the normalizer fit
    - None with historic data: standard train/val split for hyperparameter tuning
    - None with exnihilo: target-only training set (verified to hold no source data)
    - weighted / addition / transfer: historic + target training set, target test set as val

    Every branch returns the validation set as the test set: checkpoint
    selection and final evaluation share it (no separate test split, slightly
    optimistic but consistent across cases).

    The training data stays one part per source split plus the target training shots (see organize_data.merge_parts).

    Preparation applied to every training part and the val dataset:
    - drop the time_idx coordinate (duplicate values across shots break
      groupby("shot") on reassembly, the dataloader uses "time" instead)
    - broadcast the per-shot ds_source_idx against time as a float so the
      dataloader can slice, segment, and NaN-pad it like the other inputs,
      and append it to input_vars (the modules consume it to select
      per-device normalization stats)

    Returns (train_parts, ds_val, input_vars, normalizer_fit_ds), the last one
    None except for transfer_pretrain.
    """
    training_data = dataloader_config["training_data"]

    if isinstance(training_data, dict):  # when the config is passed from WandB, it's a dict
        training_data = TrainingData(**training_data)

    normalizer_fit_ds = None
    domain_adaptation = dataloader_config["domain_adaptation"]
    target_split = TargetSplit.from_dataloader_config(dataloader_config)
    if domain_adaptation == "transfer_pretrain":
        logger.info("Using transfer pretrain dataloader (trains on historic data, normalizer fit on historic + target shots)")
        train_parts, normalizer_fit_ds, ds_val = get_transfer_pretrain_datasets(
            training_data=training_data,
            target_split=target_split,
            study_type=study_type,
        )
    elif domain_adaptation is None:
        logger.info("Using standard learning dataloader")
        if not training_data.exnihilo:
            train_parts, ds_val = get_train_val_datasets(
                training_data=training_data,
                study_type=study_type,
            )
        else:
            train_parts, ds_val = get_train_test_datasets(
                training_data=training_data,
                domain_adaptation=None,
                target_split=target_split,
                study_type=study_type,
            )
            # Double check there's no source (non-target) data anywhere in here
            non_target = set(config.dataset_paths.keys()) - {config.target_device}
            if any((part["ds_source"] == src).any() for part in train_parts for src in non_target):
                raise ValueError(
                    "Historic data found in training set for exnihilo training_data option. Please check the dataset construction logic."
                )
    else:
        logger.info(f"Using transfer learning dataloader with domain adaptation {domain_adaptation}")
        train_parts, ds_val = get_train_test_datasets(
            training_data=training_data,
            domain_adaptation=domain_adaptation,
            target_split=target_split,
            study_type=study_type,
        )

    input_vars = list(dataloader_config["input_vars"])
    if "ds_source_idx" not in input_vars:
        input_vars.append("ds_source_idx")
    train_parts = [prepared_for_dataloader(part) for part in train_parts]
    ds_val = prepared_for_dataloader(ds_val)
    return train_parts, ds_val, input_vars, normalizer_fit_ds


def prepared_for_dataloader(ds: xr.Dataset) -> xr.Dataset:
    """ds without the time_idx coordinate and with ds_source_idx broadcast against time, see resolve_case_datasets."""
    ds = ds.drop_vars(TIME_DIM, errors="ignore")
    ds["ds_source_idx"] = ds["ds_source_idx"].broadcast_like(ds["ip_MA"]).astype(ds["ip_MA"].dtype)
    return ds


class FixedStepsDataLoader(DataLoader):
    """A training DataLoader whose every epoch is exactly steps_per_epoch batches of the natural loader's batch size.

    Every case of a study trains the same steps per epoch (Study.steps_per_epoch),
    so the LR schedule, the validation cadence and the patience count the same optimizer steps
    whatever the size of the case's training set.
    An epoch chains reshuffled passes over the samples until steps_per_epoch batches are out,
    and the rest of its last pass is dropped.
    """

    def __init__(self, train_dl: DataLoader, steps_per_epoch: int):
        n_batches_natural = len(train_dl)
        if n_batches_natural > steps_per_epoch:
            raise ValueError(
                f"One pass over this training set is {n_batches_natural} batches, more than the study's {steps_per_epoch} steps per epoch, "
                "which come from the largest training set its locked fields allow"
            )
        super().__init__(
            train_dl.dataset,
            batch_size=train_dl.batch_size,
            shuffle=train_dl.shuffle,
            drop_last=train_dl.drop_last,
            pad_last=train_dl.pad_last,
            key=train_dl.key,
            generate_prng=train_dl.generate_prng,
        )
        self.steps_per_epoch = steps_per_epoch

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        n_batches = 0
        while True:
            for batch in super().__iter__():
                yield batch
                n_batches += 1
                if n_batches == self.steps_per_epoch:
                    return


def fixed_steps_train_dataloader(train_dl: DataLoader, steps_per_epoch: int | None) -> DataLoader:
    """train_dl at the study's steps per epoch, as built when None (only Study.steps_per_epoch measures it so)."""
    if steps_per_epoch is None:
        return train_dl
    return FixedStepsDataLoader(train_dl, steps_per_epoch)


def get_time_dep_dataloaders(
    dataloader_config: dict,
    study_type: str,
) -> tuple[xr.Dataset, DataLoader, DataLoader, DataLoader]:
    """Dataset and dataloaders shared by the time-dependent (state-carrying) TRBs.

    Used by the power balance and transport predictor TrainRunBuilders, which
    differ only in the study_type their datasets are prepared with,
    and by the p_oh / p_rad TRBs, whose configs carry no state_vars and get time-independent dataloaders.
    """
    train_parts, ds_val, input_vars, normalizer_fit_ds = resolve_case_datasets(dataloader_config, study_type)
    loader_vars = {
        "input_vars": input_vars,
        "target_vars": dataloader_config["target_vars"],
        "extra_vars": dataloader_config.get("extra_vars"),
        "state_init_vars": dataloader_config.get("state_vars"),
    }
    common = {"time_coord": TIME_COORD, "episode_coord": EPISODE_DIM, "convert_xr_to_jnp": False, **loader_vars}

    if "state_vars" in dataloader_config.keys():
        train_dl = segmented_train_dataloader(train_parts, time_dep_train_loader_kwargs(dataloader_config, input_vars))
        # Validation samples are whole episodes.
        # The stores are contiguous in time, but a profile NaN beyond the store's
        # forward-fill hold would either be stitched over or drop the whole episode under drop_segment.
        # Keep only each episode's longest contiguous non-NaN run so val simulates a single window
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
        (val_dl,) = make_dataloaders(
            datasets=(ds_val,),
            **common,
            batch_size=dataloader_config["batch_size"],
            segment_length=dataloader_config["segment_length_val"],
            segment_overlap=dataloader_config["segment_overlap_val"],
            shuffle=False,
            # drop_slice_any only clears the leading and trailing padding of the masked episodes
            nan_handling="drop_slice_any",
            # Pad the ragged final batch so the jitted eval step keeps one shape, consumers trim the duplicates
            drop_last=False,
            pad_last=True,
        )
    else:
        # Time-independent samples: each part is stacked into timeslices, so merging the 0D parts costs no padding of note
        train_dl, val_dl = make_dataloaders(
            datasets=(merge_parts(train_parts), ds_val),
            **common,
            batch_size=dataloader_config["batch_size"],
            shuffle=[True, False],
            drop_last=[True, False],
            pad_last=[False, True],
        )
    train_dl = fixed_steps_train_dataloader(train_dl, dataloader_config["steps_per_epoch"])
    attach_normalizer_fit_ds(train_dl, normalizer_fit_ds)
    # The validation set doubles as the test set (see resolve_case_datasets)
    return ds_val, train_dl, val_dl, val_dl


def time_dep_train_loader_kwargs(dataloader_config: dict, input_vars: list[str]) -> dict:
    """popsim make_dataloaders kwargs of a time-dependent case's training dataloader."""
    return {
        "time_coord": TIME_COORD,
        "episode_coord": EPISODE_DIM,
        "input_vars": input_vars,
        "target_vars": dataloader_config["target_vars"],
        "extra_vars": dataloader_config.get("extra_vars"),
        "state_init_vars": dataloader_config["state_vars"],
        "batch_size": dataloader_config["batch_size"],
        "segment_length": dataloader_config["segment_length_train"],
        "segment_overlap": dataloader_config["segment_overlap_train"],
        "shuffle": True,
        "convert_xr_to_jnp": False,  # Needed to keep the coords for calculating loss
        # Every shot is one contiguous 1 kHz segment (organize_data.check_uniform_timebase),
        # so NaN only marks the trailing padding and the profile slices beyond the store's forward-fill hold.
        # drop_segment discards the segments touching an interior NaN,
        # while the partial segments at a shot's end are forward-filled with epsilon-padded times and kept
        "nan_handling": "drop_segment",
        # Keep every batch the same shape so the jitted train step never
        # retraces on a ragged final batch (whose static xr metadata is not
        # comparable across calls). The ragged tail is dropped,
        # reshuffled every epoch, so no data is permanently lost
        "drop_last": True,
        "pad_last": False,
    }


def segmented_train_dataloader(train_parts: list[xr.Dataset | None], loader_kwargs: dict) -> DataLoader:
    """The training dataloader popsim builds on the parts merged into one dataset, without ever merging them.

    Merged, every shot is NaN-padded to the longest shot of any part (a DIII-D shot is 20x a MAST one).
    Instead each part is segmented on its own, and the segments are merged and put back in popsim's order.
    The only coupling between episodes in popsim's segmenting is the time axis length T:
    windows start every segment_length - segment_overlap steps from an episode's first all-finite slice,
    up to T - segment_length, and a window reaching past the episode's last all-finite slice is forward-filled and kept.
    So a shot of trimmed length L gets window starts up to min(T - segment_length, L - 1), and T depends on the other parts.
    Merged, ds_source_idx (broadcast over the whole time axis) keeps every column up to the merged length less the smallest shift.
    A room-keeper episode, finite over min(that T, the part's longest L + segment_length - 1) steps,
    gives every shot of its part the same window starts, and its own segments are dropped.
    The samples are then concatenated in popsim's window-major order (window, then episode in part order),
    so the shuffled batches are the merged ones too (see tests/modules/test_segmented_dataloader.py).
    The list is consumed: each part is released (set to None) once segmented,
    so the raw parts and the segments never all sit in memory at once.
    """
    segment_length = loader_kwargs["segment_length"]
    training_vars = sorted(
        {
            *loader_kwargs["input_vars"],
            *loader_kwargs["target_vars"],
            *loader_kwargs["state_init_vars"],
            *(loader_kwargs["extra_vars"] or []),
        }
    )
    if "ds_source_idx" not in training_vars:
        raise ValueError("segmented_train_dataloader relies on the time-broadcast ds_source_idx being a training variable")
    shot_lengths = [_trimmed_shot_lengths(part, training_vars) for part in train_parts]
    merged_length = max(part.sizes[TIME_DIM] for part in train_parts)
    smallest_shift = min(int(shifts.min()) for shifts, _ in shot_lengths if shifts.size)
    merged_time_length = merged_length - smallest_shift

    part_samples = []
    for part_idx, (_, lengths) in enumerate(shot_lengths):
        part = train_parts[part_idx]
        train_parts[part_idx] = None
        if ROOM_KEEPER_SHOT in part[EPISODE_DIM].values:
            raise ValueError(f"Shot {ROOM_KEEPER_SHOT} is reserved for the room-keeper episode")
        if not lengths.size:
            continue
        keeper_length = min(merged_time_length, int(lengths.max()) + segment_length - 1)
        part_kept = concat_with_nan_padding([part[training_vars], _room_keeper(part, training_vars, keeper_length)], concat_dim=EPISODE_DIM)
        del part
        (part_dl,) = make_dataloaders(datasets=(part_kept.drop_vars(TIME_DIM),), **loader_kwargs)
        del part_kept
        part_samples.append(part_dl.ds)
        training_metadata = part_dl.metadata

    # Each part's samples are window-major already, its shots in part order with the room-keeper last in every window,
    # so popsim's merged order is one slice per (window, part), each cut short of the room-keeper
    windows = np.unique(np.concatenate([samples["input_batch"].values for samples in part_samples]))
    blocks = []
    for window in windows:
        for samples in part_samples:
            window_idxs = samples["input_batch"].values
            start, stop = np.searchsorted(window_idxs, window, side="left"), np.searchsorted(window_idxs, window, side="right")
            if stop > start and samples[EPISODE_DIM].values[stop - 1] == ROOM_KEEPER_SHOT:
                stop -= 1
            if stop > start:
                blocks.append(samples.isel(sample=slice(start, stop)))
    if not blocks:
        raise ValueError("All samples were dropped due to NaN values. Cannot create DataLoader.")
    merged_samples = xr.concat(blocks, dim="sample", coords="different", compat="equals")
    del blocks, part_samples
    return DataLoader(
        XarrayPreppedDataset(merged_samples, training_metadata),
        batch_size=loader_kwargs["batch_size"],
        shuffle=loader_kwargs["shuffle"],
        drop_last=loader_kwargs["drop_last"],
        pad_last=loader_kwargs["pad_last"],
    )


def _trimmed_shot_lengths(part: xr.Dataset, training_vars: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per shot with an all-finite slice: popsim's leading shift, and the trimmed length from it to the last all-finite slice."""
    mask_all_finite = np.ones((part.sizes[EPISODE_DIM], part.sizes[TIME_DIM]), dtype=bool)
    for name in training_vars:
        da = part[name]
        if TIME_DIM not in da.dims:
            da = da.broadcast_like(part[TIME_COORD])
        finite = np.isfinite(da.transpose(EPISODE_DIM, TIME_DIM, ...).values)
        mask_all_finite &= finite.reshape(finite.shape[0], finite.shape[1], -1).all(axis=-1)
    mask_valid_shot = mask_all_finite.any(axis=1)
    mask_all_finite = mask_all_finite[mask_valid_shot]
    first = np.argmax(mask_all_finite, axis=1)
    last = mask_all_finite.shape[1] - 1 - np.argmax(mask_all_finite[:, ::-1], axis=1)
    return first, last + 1 - first


def _room_keeper(part: xr.Dataset, training_vars: list[str], length: int) -> xr.Dataset:
    """One episode of zeros in every training variable over length steps, on the part's other dims and shot-free coords."""
    data_vars = {}
    for name in training_vars:
        other_dims = [dim for dim in part[name].dims if dim not in (EPISODE_DIM, TIME_DIM)]
        shape = (1, length, *(part.sizes[dim] for dim in other_dims))
        data_vars[name] = ((EPISODE_DIM, TIME_DIM, *other_dims), np.zeros(shape, dtype=part[name].dtype))
    time = np.nanmin(part[TIME_COORD].values) + UNIFORM_TIMEBASE_DT * np.arange(length)
    coords = {EPISODE_DIM: [ROOM_KEEPER_SHOT], TIME_COORD: ((EPISODE_DIM, TIME_DIM), time[np.newaxis, :])}
    for coord_name, coord in part.coords.items():
        if coord_name not in coords and EPISODE_DIM not in coord.dims and TIME_DIM not in coord.dims:
            coords[coord_name] = coord
    return xr.Dataset(data_vars, coords=coords)
