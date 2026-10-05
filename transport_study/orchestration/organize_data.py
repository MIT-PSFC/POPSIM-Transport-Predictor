from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from loguru import logger
from popsim.ml.split_utils import split_dataset_by_fracs
from transport_validation_datasets.machine.generic import UNIFORM_TIMEBASE_DT

from transport_study import EPISODE_DIM, RADIAL_DIM, TIME_COORD, TIME_DIM
from transport_study.config import (
    DEBUG_MAX_SOURCE_SHOTS,
    RHO_GRID,
    TRAIN_VAL_SPLIT,
    config,
)
from transport_study.modules import plasma_parameters
from transport_study.modules.normalization import (
    NORM_INPUT_VARS,
    PHYSICS_FEATURE_NAMES,
    apply_coral,
    apply_z_score,
    fit_coral_stats,
    fit_z_score_stats,
    flat_columns,
    identity_coral_stats,
    physics_feature_vec,
)
from transport_study.modules.profile_predictor.module import (
    NN_INPUT_NAMES,
    NN_INPUT_SOURCE_VARS,
    nn_input_matrix,
)
from transport_study.signals import convert_to_working_units, store_signals_for

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class TrainingData:
    """The source devices a case trains on.

    sources: device keys, stored sorted for stable concatenation and indexing,
             so equal device sets compare and hash equal
    exnihilo: train from scratch on the target device only,
              every source device is loaded and then stripped from the training set again
    """

    # A list rather than a tuple, train configs are YAML-dumped and safe_load rejects python tuples
    sources: list[str]
    exnihilo: bool = False

    def __post_init__(self):
        object.__setattr__(self, "sources", sorted(self.sources))

    def __hash__(self):
        return hash((tuple(self.sources), self.exnihilo))

    def __str__(self) -> str:
        if self.exnihilo:
            return "exnihilo"
        return "_".join(self.sources)

    @property
    def source_idxs(self) -> list:
        """Integer indices for all sources in this TrainingData, looked up from global config."""
        return [config.ds_source_to_idx[s] for s in self.sources]


def parse_training_data(s: str, dataset_paths: dict, target_device: str | None) -> TrainingData:
    """Convert a string like 'cmod_tcv' or 'exnihilo' to a TrainingData object."""
    if s == "exnihilo":
        non_target = set(dataset_paths.keys()) - ({target_device} if target_device else set())
        return TrainingData(sources=sorted(non_target), exnihilo=True)
    sources = s.split("_")
    if target_device in sources:
        raise ValueError(
            f"training_data {s!r} names the target device {target_device!r}, whose held-out test shots would land in the training set"
        )
    return TrainingData(sources=sources)


REQUIRED_SIGNALS_POWER_BALANCE = [
    # Targets, the powers those of the p_oh / p_rad submodule cases and anchors of the joint training loss
    "energy_mhd_MJ",
    "power_ohm_MW",
    "power_radiated_MW",
    # Inputs
    "ip_MA",
    "b_geo",
    "geometric_axis_r",
    "elongation",
    "minor_radius",  # For inverse aspect ratio
    "n_e_line_average_1e20",
    "power_additional_MW",
    # Extra
    "time",  # Data variable holding per-shot time values, promoted to the time coordinate downstream
    "b0",  # Visualization-only beta_tor of normalize_domain, normalized with b0 at r0 as IMAS does
]


# Profile channels and the gradient and error-bar companions every device store carries.
# For each base signal <v> the companions are <v>_gradient, <v>_error and <v>_gradient_error,
# the GP fit's (C-Mod, MAST, TCV) or IDA's (DIII-D) 1-sigma uncertainties.
PROFILE_BASE_SIGNALS = ["t_e_keV", "n_e_1e20"]
PROFILE_GRAD_SIGNALS = [f"{v}_gradient" for v in PROFILE_BASE_SIGNALS]
PROFILE_ERROR_SIGNALS = [f"{v}_error" for v in PROFILE_BASE_SIGNALS] + [f"{v}_gradient_error" for v in PROFILE_BASE_SIGNALS]

# Everything the profile-predictor loss reads from the target side:
# the profiles themselves plus their gradients and error-bars
PROFILE_TARGET_VARS = [*PROFILE_BASE_SIGNALS, *PROFILE_GRAD_SIGNALS, *PROFILE_ERROR_SIGNALS]


REQUIRED_SIGNALS_PROFILE_TRANSFER = [
    # Target-related
    *PROFILE_BASE_SIGNALS,
    *PROFILE_GRAD_SIGNALS,
    *PROFILE_ERROR_SIGNALS,
    "fresh_profile",  # Needed so we only train on time points where the profile data is fresh, avoiding forward-filled.
    # Inputs
    "ip_MA",
    "b0",
    "b_geo",
    "beta_tor_norm",
    "n_e_line_average_1e20",
    "geometric_axis_r",
    "minor_radius",
    "elongation",
    "triangularity_upper",
    "triangularity_lower",
    # Extra
    "time",  # Data variable holding per-shot time values, promoted to the time coordinate downstream
    "energy_mhd_MJ",  # The hazard metric needs it
]

# Union of the profile and power balance needs, minus beta_tor_norm:
# the transport modules derive every beta quantity from the evolving stored-energy state
# instead of a measured beta_tor_norm (see modules/transport_predictor/module.py).
REQUIRED_SIGNALS_TRANSPORT_TRANSFER = [
    # Targets and their companions
    *PROFILE_BASE_SIGNALS,
    *PROFILE_GRAD_SIGNALS,
    *PROFILE_ERROR_SIGNALS,
    "fresh_profile",  # Kept as data (not a filter) so downstream losses or metrics can mask stale forward-filled profiles
    # Inputs
    "ip_MA",
    "b0",
    "b_geo",
    "n_e_line_average_1e20",
    "geometric_axis_r",
    "minor_radius",
    "elongation",
    "triangularity_upper",
    "triangularity_lower",
    "power_additional_MW",
    # Extra
    "time",  # Data variable holding per-shot time values, promoted to the time coordinate downstream
    "energy_mhd_MJ",  # Seeds the sciml stored-energy state and the normalizer fit, the hazard metric, an anchor target of the sciml training loss
    "power_ohm_MW",  # Anchor target for the sciml training loss
    "power_radiated_MW",  # Anchor target for the sciml training loss
]

REQUIRED_SIGNALS = {
    "profile_transfer": REQUIRED_SIGNALS_PROFILE_TRANSFER,
    "power_balance_transfer": REQUIRED_SIGNALS_POWER_BALANCE,
    "transport_transfer": REQUIRED_SIGNALS_TRANSPORT_TRANSFER,
}

# Per-shot variables add_hazard writes, on (shot,)
HAZARD_VARS = ("hazard", "ip_MA_p95", "energy_mhd_MJ_p95")

# A time step within this of UNIFORM_TIMEBASE_DT is the 1 kHz step, the slack absorbs float32 round-off
TIMEBASE_STEP_TOLERANCE_S = 1e-5


def concat_with_nan_padding(
    datasets: list[xr.Dataset],
    concat_dim: str,
    pad_dim: str = TIME_DIM,
) -> xr.Dataset:
    """Concatenate datasets while padding shorter `pad_dim` with NaNs.

    pad_dim gets an ordinal index when it has none,
    so the outer join of the concat pads the shorter datasets.
    """
    prepared_datasets = []
    for dataset in datasets:
        if pad_dim in dataset.dims and pad_dim not in dataset.coords:
            ds = dataset.assign_coords({pad_dim: np.arange(dataset.sizes[pad_dim])})
        else:
            ds = dataset
        prepared_datasets.append(ds)

    ds_padded = xr.concat(
        prepared_datasets,
        dim=concat_dim,
        join="outer",
        coords="different",
        compat="equals",
        fill_value=np.nan,
    )

    return ds_padded


def check_uniform_timebase(ds: xr.Dataset) -> None:
    """Raise unless every shot is one contiguous run of 1 kHz samples followed by NaN padding.

    The stores hold one contiguous segment per shot,
    which the simple-Euler steppers rely on (a step of dt >~ tau_e is unstable),
    so a gap or an off-grid sample is a broken store, not something to repair here.
    """
    time2d = ds[TIME_COORD].transpose(EPISODE_DIM, TIME_DIM).values
    shots = ds[EPISODE_DIM].values
    mask_finite = np.isfinite(time2d)
    # A finite time after a NaN one means the padding is not trailing
    mask_resumes = ~mask_finite[:, :-1] & mask_finite[:, 1:]
    shots_with_gaps = shots[mask_resumes.any(axis=1)]
    if shots_with_gaps.size:
        raise ValueError(
            f"Shots {shots_with_gaps.tolist()} have NaN times inside the shot, the store is not one contiguous segment per shot"
        )
    dt = np.diff(time2d, axis=1)
    mask_both_finite = mask_finite[:, :-1] & mask_finite[:, 1:]
    mask_off_grid = mask_both_finite & (np.abs(dt - UNIFORM_TIMEBASE_DT) > TIMEBASE_STEP_TOLERANCE_S)
    shots_off_grid = shots[mask_off_grid.any(axis=1)]
    if shots_off_grid.size:
        raise ValueError(f"Shots {shots_off_grid.tolist()} have time steps other than the {UNIFORM_TIMEBASE_DT} s step")


def to_rho_grid(ds: xr.Dataset) -> xr.Dataset:
    """Profiles interpolated onto the shared RHO_GRID, with the t_e / n_e shape variables.

    The store's fit grid covers RHO_GRID on every device, so nothing extrapolates.
    Attrs are stripped: they become static jit metadata in the xarray pytree registration,
    and array-valued ones break the treedef equality check.
    """
    ds_rho = ds.interp({RADIAL_DIM: RHO_GRID})
    ds_rho["t_e_shape"] = ds_rho["t_e_keV"] / ds_rho["t_e_keV"].integrate(RADIAL_DIM)
    ds_rho["n_e_shape"] = ds_rho["n_e_1e20"] / ds_rho["n_e_1e20"].integrate(RADIAL_DIM)
    return ds_rho.drop_attrs()


def keep_fresh_timeslices(ds: xr.Dataset) -> xr.Dataset:
    """NaN the forward-filled profile timeslices and drop the shots without a fresh one.

    Only the time-dependent variables are masked:
    a whole-dataset where would broadcast the per-shot variables (hazard) against time.
    """
    mask_fresh = ds["fresh_profile"] == 1
    names_per_shot = [name for name in ds.data_vars if TIME_DIM not in ds[name].dims]
    ds_time_dep = ds.drop_vars(names_per_shot).where(mask_fresh, drop=True)
    return ds_time_dep.merge(ds[names_per_shot], join="left")


def _profile_transfer(ds: xr.Dataset) -> xr.Dataset:
    """Fresh profile timeslices only, on RHO_GRID."""
    ds_fresh = keep_fresh_timeslices(ds)
    return to_rho_grid(ds_fresh)


def _power_balance(ds: xr.Dataset) -> xr.Dataset:
    """Scalar time series as stored, attrs stripped (see to_rho_grid)."""
    return ds.drop_attrs()


def _transport_transfer(ds: xr.Dataset) -> xr.Dataset:
    """Every timeslice on RHO_GRID.

    No fresh-profile filter: the time-dependent rollouts need contiguous segments,
    so the forward-filled profile timeslices stay in as targets
    and fresh_profile rides along as data for masking downstream.
    """
    return to_rho_grid(ds)


STUDY_PREPS = {
    "profile_transfer": _profile_transfer,
    "power_balance_transfer": _power_balance,
    "transport_transfer": _transport_transfer,
}


def get_ds(
    source_ds: str,
    study_type: str,
) -> xr.Dataset:
    """Open a device store and prepare it for one study.

    Reads only the store signals the study needs (the power balance study never loads the profiles),
    converts to working units, keeps the max_ds_size most recent shots
    (and in debug at most DEBUG_MAX_SOURCE_SHOTS of a source device, the target keeps every shot),
    checks every shot is one contiguous 1 kHz segment,
    adds the hazard metric on the full 0D series so every study holds out the same target shots,
    drops the shots without one, then runs the study's prep.

    Args:
        source_ds (str): Identifier for the source dataset.
        study_type (str): Type of study for which to prepare the dataset.

    Returns:
        xr.Dataset: The prepared dataset on (shot, time_idx[, rho_tor_norm])
    """
    if source_ds not in config.dataset_paths:
        raise ValueError(f"Unknown source dataset: {source_ds!r}. Available: {set(config.dataset_paths)}")
    if study_type not in STUDY_PREPS:
        raise ValueError(f"Unknown study type: {study_type}")
    ds_path = Path(config.dataset_paths[source_ds])
    required_signals = REQUIRED_SIGNALS[study_type]
    store_signals = store_signals_for(required_signals)

    with xr.open_dataset(ds_path) as ds_store:
        ds_selected = ds_store[store_signals].load()

    # Stores are IMAS names in SI, the study works in the suffixed working units
    ds_working = convert_to_working_units(ds_selected)
    ds = ds_working[required_signals]
    ds = ds.astype(jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32)

    if EPISODE_DIM not in ds.dims:
        raise ValueError(f"Expected dataset to have {EPISODE_DIM} dimension, but it was not found. Found dimensions: {ds.dims}")

    ds = ds.sortby(EPISODE_DIM, ascending=False)  # Most recent shots first
    n_shots_stored = ds.sizes[EPISODE_DIM]
    max_shots = config.max_ds_size
    if config.debug and source_ds != config.target_device:
        max_shots = DEBUG_MAX_SOURCE_SHOTS if max_shots is None else min(max_shots, DEBUG_MAX_SOURCE_SHOTS)
    if max_shots is not None and n_shots_stored > max_shots:
        logger.warning(f"Keeping the {max_shots} most recent of the {n_shots_stored} {source_ds} shots")
        ds = ds.isel({EPISODE_DIM: slice(0, max_shots)})

    check_uniform_timebase(ds)

    ds = add_hazard(ds)
    mask_hazard_valid = ds["hazard"].notnull().values
    if not mask_hazard_valid.all():
        shots_without_hazard = ds[EPISODE_DIM].values[~mask_hazard_valid]
        logger.warning(f"Dropping {shots_without_hazard.size} {source_ds} shots without a hazard metric: {shots_without_hazard.tolist()}")
        ds = ds.isel({EPISODE_DIM: mask_hazard_valid})

    ds = STUDY_PREPS[study_type](ds)

    # From a zarr store the time variable is a data variable, the dataloaders consume it as the time coordinate
    if TIME_COORD not in ds.coords:
        ds = ds.set_coords(TIME_COORD)

    return ds


def add_hazard(ds: xr.Dataset) -> xr.Dataset:
    """Add the per-shot hazard metric, the p95 along the shot of (energy_mhd_MJ^2 + ip_MA^2)^0.5,
    each normalized by its dataset maximum, so shots with more stored energy and plasma current rank higher.
    NaNs are ignored.

    Also stores ip_MA and energy_mhd_MJ at the timeslice whose hazard is closest to the p95
    for plotting in parameter space.
    """
    max_energy_mhd = float(ds["energy_mhd_MJ"].max().values)
    max_ip = float(ds["ip_MA"].max().values)
    energy_mhd_scale = 1.0 / max_energy_mhd if max_energy_mhd != 0 else 1.0
    ip_scale = 1.0 / max_ip if max_ip != 0 else 1.0

    energy_mhd_normalized = energy_mhd_scale * ds["energy_mhd_MJ"]
    ip_normalized = ip_scale * ds["ip_MA"]
    hazard_timeseries = (energy_mhd_normalized**2 + ip_normalized**2) ** 0.5
    hazard_p95 = hazard_timeseries.quantile(0.95, dim=TIME_DIM, skipna=True).drop_vars("quantile")
    ds["hazard"] = hazard_p95

    # The timeslice closest to the p95, NaN hazards sit at +inf so a shot with any valid slice picks a valid one
    distance_to_p95 = abs(hazard_timeseries - hazard_p95).fillna(np.inf)
    idx_p95 = distance_to_p95.argmin(dim=TIME_DIM)
    mask_shot_valid = hazard_p95.notnull()
    ip_MA_p95 = ds["ip_MA"].isel({TIME_DIM: idx_p95}).reset_coords(drop=True)
    energy_mhd_MJ_p95 = ds["energy_mhd_MJ"].isel({TIME_DIM: idx_p95}).reset_coords(drop=True)
    ds["ip_MA_p95"] = ip_MA_p95.where(mask_shot_valid)
    ds["energy_mhd_MJ_p95"] = energy_mhd_MJ_p95.where(mask_shot_valid)

    return ds


def normalize_domain(ds: xr.Dataset, method: str = "raw", feature_space: str = "power_balance") -> xr.Dataset:
    """Apply a domain normalization method to a dataset holding every device, for data visualization only.

    Thin wrapper around transport_study.modules.normalization:
    the per-method math (physics features, per-device z-score, CORAL) is exactly the module implementation the models consume,
    fitted on ds and applied to it.
    The dataset is modified in place and returned.

    feature_space selects which model family's features the physics* methods run on,
    so each study visualizes what its own modules consume:
        - "power_balance": the 7 physics features of normalization.physics_feature_vec, fitted over the 7 physical inputs
        - "profile": the 10 dimensionless nn_inputs of the profile predictor
        - "transport": those 10 slots with the beta-derived ones from the measured energy_mhd_MJ, plus paux_norm
    The raw-variable methods ("zscore", "coral") are power-balance inputs by definition.

    Methods:
        - "raw": no normalization, variables stay in their working units
        - "physics": the module's dimensionless features.
          The power-balance space adds beta as a visualization-only extra, the other spaces carry their own beta slot.
        - "zscore": per-device zero mean and unit variance of each variable, `_z` suffix.
          energy_mhd_MJ is a visualization-only extra column.
        - "coral": CORAL alignment of every device's covariance to the target device's over the 7 input vars, `_coral` suffix
        - "physics-coral": the same CORAL alignment over the physics features of the feature space, `_pcoral` suffix
        - "physics-zscore": per-device z-score over the physics features of the feature space, `_pz` suffix
    """
    physics_names: tuple[str, ...]
    physics_source_vars: tuple[str, ...]
    physics_matrix_fn: Callable[[xr.Dataset], np.ndarray] | None
    if feature_space == "transport":
        # Function-level import: the transport predictor module pulls in TORAX,
        # and organize_data is imported by everything
        from transport_study.modules.transport_predictor.module import (
            TRANSPORT_NN_INPUT_NAMES,
            TRANSPORT_NN_INPUT_SOURCE_VARS,
            transport_nn_input_matrix,
        )

        physics_names, physics_source_vars, physics_matrix_fn = (
            TRANSPORT_NN_INPUT_NAMES,
            TRANSPORT_NN_INPUT_SOURCE_VARS,
            transport_nn_input_matrix,
        )
    elif feature_space == "profile":
        physics_names, physics_source_vars, physics_matrix_fn = NN_INPUT_NAMES, NN_INPUT_SOURCE_VARS, nn_input_matrix
    elif feature_space == "power_balance":
        physics_names, physics_source_vars, physics_matrix_fn = PHYSICS_FEATURE_NAMES, NORM_INPUT_VARS, None
    else:
        raise ValueError(f"Unknown feature space: {feature_space}")
    # Broadcast template carrying the full per-sample dims
    reference = ds[NORM_INPUT_VARS[0]]

    def _feature_matrix_for(variables: tuple[str, ...]) -> np.ndarray:
        missing = [var for var in variables if var not in ds]
        if missing:
            raise ValueError(f"Feature matrix needs {missing}, the dataset lacks them")
        return flat_columns(ds, variables)

    def _physics_matrix() -> np.ndarray:
        if physics_matrix_fn is not None:
            return physics_matrix_fn(ds)
        raw = jnp.asarray(_feature_matrix_for(NORM_INPUT_VARS))
        return np.asarray(jax.vmap(physics_feature_vec)(raw))

    def _write_features(matrix: np.ndarray, names: tuple[str, ...], suffix: str) -> None:
        for j, name in enumerate(names):
            ds[f"{name}{suffix}"] = (reference.dims, np.asarray(matrix[:, j], dtype=float).reshape(reference.shape))

    if method == "raw":
        return ds
    if method == "physics":
        physics_matrix = _physics_matrix()
        # Physics slots that are identity mappings of a raw input var already exist as raw vars
        identity_slots = set(physics_source_vars)
        for j, name in enumerate(physics_names):
            if name not in identity_slots:
                ds[name] = (reference.dims, np.asarray(physics_matrix[:, j], dtype=float).reshape(reference.shape))
        if feature_space == "power_balance":
            # beta needs the stored energy, the predicted state rather than a model input, so it is a visualization-only extra.
            # The same fraction the modules call beta, through the store's betan formula with volume_approx.
            volume_m3 = plasma_parameters.volume_approx(ds["geometric_axis_r"], ds["minor_radius"], ds["elongation"])
            beta_tor_norm = plasma_parameters.beta_tor_norm_from_energy_mhd_MJ(
                ds["energy_mhd_MJ"], volume_m3, ds["minor_radius"], ds["b0"], ds["ip_MA"]
            )
            ds["beta"] = plasma_parameters.beta_tor_from_beta_tor_norm(beta_tor_norm, ds["ip_MA"], ds["minor_radius"], ds["b0"])
        return ds

    # Stat-bearing methods key per-device statistics on an integer index built locally from the ds_source coordinate.
    # The modules use the global config.ds_source_to_idx, any consistent indexing gives the same stats.
    registry = {str(device): idx for idx, device in enumerate(dict.fromkeys(np.atleast_1d(ds.coords["ds_source"].values)))}
    source_idx_da = xr.apply_ufunc(np.vectorize(lambda d: registry[str(d)]), ds.coords["ds_source"])
    source_idx = np.asarray(source_idx_da.broadcast_like(reference).values).ravel().astype(int)

    if method == "zscore":
        zscore_vars = (*NORM_INPUT_VARS, "energy_mhd_MJ")
        zscore_inputs = _feature_matrix_for(zscore_vars)
        means, stds = fit_z_score_stats(zscore_inputs, source_idx, len(registry))
        zscore_matrix = apply_z_score(jnp.asarray(zscore_inputs), source_idx, means, stds)
        _write_features(np.asarray(zscore_matrix), zscore_vars, "_z")
        return ds
    if method == "physics-zscore":
        physics_matrix = _physics_matrix()
        means, stds = fit_z_score_stats(physics_matrix, source_idx, len(registry))
        zscore_matrix = apply_z_score(jnp.asarray(physics_matrix), source_idx, means, stds)
        _write_features(np.asarray(zscore_matrix), physics_names, "_pz")
        return ds
    if method not in ("coral", "physics-coral"):
        raise ValueError(f"Unknown normalization method: {method}")

    # Every device aligns to the target device's covariance, all keep the identity transform without the target.
    # Source devices below MIN_CORAL_SHOTS keep the identity transform, so their features pass through raw.
    # Rows with any NaN feature come out all-NaN (the joint transform needs complete rows).
    coral_names: tuple[str, ...]
    if method == "coral":
        coral_names, coral_suffix, coral_inputs = NORM_INPUT_VARS, "_coral", _feature_matrix_for(NORM_INPUT_VARS)
    else:
        coral_names, coral_suffix, coral_inputs = physics_names, "_pcoral", _physics_matrix()
    stats = None
    if config.target_device in registry:
        shot_idx = np.asarray(ds.coords["shot"].broadcast_like(reference).values).ravel()
        stats = fit_coral_stats(coral_inputs, source_idx, len(registry), shot_idx, registry[config.target_device])
    means, transforms = identity_coral_stats(len(registry), len(coral_names)) if stats is None else stats
    batched_apply = jax.vmap(apply_coral, in_axes=(0, 0, None, None))
    coral_matrix = batched_apply(jnp.asarray(coral_inputs), jnp.asarray(source_idx), means, transforms)
    _write_features(np.asarray(coral_matrix), coral_names, coral_suffix)
    return ds


def get_train_val_datasets(
    training_data: TrainingData,
    study_type: str = "profile_transfer",
):
    """
    Split each source device into training and validation sets by a deterministic hazard sort:
    the TRAIN_VAL_SPLIT highest-hazard shots of every source are its validation set.

    There is no test set here because the true test set is the
    high-hazard target device shots, handled separately.
    That means all historic source data can be used for training and validation.
    """
    ds_sources: dict[str, tuple] = {}

    for source in training_data.sources:
        ds = get_ds(source, study_type)
        # popsim requires a seed, the hazard sort makes the split deterministic and leaves it unused
        train_src, val_src = split_dataset_by_fracs(
            ds,
            fracs=TRAIN_VAL_SPLIT,
            dim=EPISODE_DIM,
            seed=0,
            sortby="hazard",
        )
        src_idx = config.ds_source_to_idx[source]
        train_src["ds_source_idx"] = (
            EPISODE_DIM,
            np.full(train_src.sizes[EPISODE_DIM], src_idx),
        )
        val_src["ds_source_idx"] = (
            EPISODE_DIM,
            np.full(val_src.sizes[EPISODE_DIM], src_idx),
        )
        train_src = train_src.assign_coords(ds_source=source)
        val_src = val_src.assign_coords(ds_source=source)
        ds_sources[source] = (train_src, val_src)

    if not ds_sources:
        raise ValueError("training_data.sources is empty - cannot build train/val datasets")

    if len(ds_sources) == 1:
        source = next(iter(ds_sources))
        train_ds, val_ds = ds_sources[source]
    else:
        train_ds = concat_with_nan_padding([pair[0] for pair in ds_sources.values()], concat_dim=EPISODE_DIM)
        val_ds = concat_with_nan_padding([pair[1] for pair in ds_sources.values()], concat_dim=EPISODE_DIM)

    logger.debug("Historic Training dataset size: {}", train_ds.sizes[EPISODE_DIM])
    logger.debug("Historic Validation dataset size: {}", val_ds.sizes[EPISODE_DIM])

    return train_ds, val_ds


def get_loaded_shot_count(source_ds: str, study_type: str = "profile_transfer") -> int:
    """Episode count get_ds actually yields for this device.

    This is the shot count the training set really sees, after max_ds_size
    truncation and study-type filtering (e.g. dropping shots without fresh
    profiles), unlike the raw on-disk shot count. Requires loading the dataset
    from disk, so it is not free, but it is only needed once per device when
    building a train config.
    """
    ds = get_ds(source_ds, study_type)
    return int(ds.sizes[EPISODE_DIM])


def _split_target_shots(
    num_target_shots: int,
    target_test_set_size: int,
    study_type: str,
):
    """Load the target device and split it into training shots and the held-out test set.

    A deterministic hazard sort: the test set is the target_test_set_size highest-hazard shots,
    the training shots are the first num_target_shots of the remaining pool
    (or every shot for -1, the cheating upper-bound reference).
    Returns (train_ds_target, test_ds).
    """
    target = config.target_device
    if target is None:
        raise ValueError("config.target_device must be set before transfer learning")

    ds_target = get_ds(target, study_type=study_type)
    ds_target["ds_source_idx"] = (
        EPISODE_DIM,
        np.full(ds_target.sizes[EPISODE_DIM], config.ds_source_to_idx[target]),
    )
    ds_target = ds_target.assign_coords(ds_source=target)
    sorted_shots = np.argsort(ds_target["hazard"].values)

    test_shot_pool = sorted_shots[-target_test_set_size:] if target_test_set_size else sorted_shots[:0]
    test_ds = ds_target.isel({EPISODE_DIM: test_shot_pool})

    if num_target_shots == -1:
        # All available target shots in training and testing (upper-bound reference, CHEATING!)
        train_ds_target = ds_target.isel({EPISODE_DIM: sorted_shots})
    else:
        # Exclude the held-out test shots before selecting training shots so the two pools
        # never overlap (otherwise a large num_target_shots would leak high-hazard test
        # shots into training)
        train_candidate_pool = sorted_shots[:-target_test_set_size] if target_test_set_size else sorted_shots
        if num_target_shots > len(train_candidate_pool):
            raise ValueError(
                f"num_target_shots={num_target_shots} requested but only {len(train_candidate_pool)} target "
                f"shots remain after holding out target_test_set_size={target_test_set_size} of "
                f"{len(sorted_shots)} loaded shots. Is the dataset smaller than expected "
                f"(max_ds_size truncation)?"
            )
        train_shot_pool = train_candidate_pool[:num_target_shots]
        assert not (set(train_shot_pool.tolist()) & set(test_shot_pool.tolist())), "Target train and test shot pools overlap - data leakage"
        train_ds_target = ds_target.isel({EPISODE_DIM: train_shot_pool})

    return train_ds_target, test_ds


def get_train_test_datasets(
    training_data: "TrainingData",
    domain_adaptation: str | None,
    num_target_shots: int,
    target_test_set_size: int,
    study_type: str = "profile_transfer",
):
    """
    Split dataset into training and test sets for the target learning case.
    If domain adaptation is 'weighted' or 'addition', makes a combined training set of historic
    data and target device shots (the two methods share this dataset, 'weighted' additionally
    gets per-device loss weights injected in make_train_config). If domain adaptation is
    'transfer' or training_data.exnihilo is True, removes all historic data from the training
    set, leaving only the target device shots.
    The number of target shots included in training is specified by `num_target_shots`.

    The test set is always the same set of target device shots.
    The training set is the historic data from training_data.sources plus num_target_shots target shots.

    There is no validation set here since hyperparameters are not tuned on transfer learning data.
    We treat the test set as a validation set for checkpoint selection, which is slightly optimistic
    but consistent across all models so comparisons are fair.
    """
    train_ds_target, test_ds = _split_target_shots(num_target_shots, target_test_set_size, study_type)

    # Load historic source data for training (for exnihilo it is stripped again below)
    # exnihilo.sources contains all non-target devices, so we can pass training_data directly
    train_ds_hist, val_ds_hist = get_train_val_datasets(training_data, study_type=study_type)
    train_ds = concat_with_nan_padding(
        [train_ds_hist, val_ds_hist, train_ds_target],
        concat_dim=EPISODE_DIM,
    )

    # For 'transfer' and exnihilo: strip historic data, train only on target device shots
    if domain_adaptation == "transfer" or training_data.exnihilo:
        train_ds = train_ds.where(
            train_ds["ds_source_idx"] == config.ds_source_to_idx[config.target_device],
            drop=True,
        )

    logger.debug("HP Training dataset size: {}", train_ds.sizes[EPISODE_DIM])
    logger.debug("HP Test dataset size: {}", test_ds.sizes[EPISODE_DIM])

    return train_ds, test_ds


def get_transfer_pretrain_datasets(
    training_data: "TrainingData",
    num_target_shots: int,
    target_test_set_size: int,
    study_type: str = "profile_transfer",
):
    """Datasets for a transfer_pretrain case (the pretrain half of a stat-normalized transfer).

    The case trains on ALL historic data (train and val splits together, the
    source val split is not needed because checkpoint selection uses the
    target test set, like every other domain-adaptation case). The combined
    dataset (historic + the transfer case's num_target_shots target shots)
    exists only to fit the normalization statistics, which the transfer case
    then inherits through the checkpoint restore.

    Returns (train_ds_hist, train_ds_combined, test_ds).
    """
    train_ds_target, test_ds = _split_target_shots(num_target_shots, target_test_set_size, study_type)

    train_ds_hist, val_ds_hist = get_train_val_datasets(training_data, study_type=study_type)
    train_ds_hist = concat_with_nan_padding([train_ds_hist, val_ds_hist], concat_dim=EPISODE_DIM)
    train_ds_combined = concat_with_nan_padding([train_ds_hist, train_ds_target], concat_dim=EPISODE_DIM)

    logger.debug("Transfer pretrain historic dataset size: {}", train_ds_hist.sizes[EPISODE_DIM])
    logger.debug("Transfer pretrain normalizer-fit dataset size: {}", train_ds_combined.sizes[EPISODE_DIM])
    logger.debug("Transfer pretrain test dataset size: {}", test_ds.sizes[EPISODE_DIM])

    return train_ds_hist, train_ds_combined, test_ds
