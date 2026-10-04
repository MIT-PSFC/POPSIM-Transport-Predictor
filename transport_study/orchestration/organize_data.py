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
from transport_study.config import RHO_GRID, TRAIN_VAL_SPLIT, config
from transport_study.modules import plasma_parameters
from transport_study.modules.normalization import (
    NORM_INPUT_VARS,
    PHYSICS_FEATURE_NAMES,
    apply_coral,
    apply_z_score,
    fit_coral_stats,
    fit_z_score_stats,
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

    sources_unsorted: device keys, read sorted through sources
    exnihilo: train from scratch on the target device only,
              every source device is loaded and then stripped from the training set again
    """

    sources_unsorted: list[str]
    exnihilo: bool = False

    def __hash__(self):
        return hash((tuple(self.sources), self.exnihilo))

    def __str__(self) -> str:
        if self.exnihilo:
            return "exnihilo"
        return "_".join(sorted(self.sources))

    @property
    def sources(self) -> list[str]:
        """Sources in deterministic order for stable concatenation and indexing."""
        return sorted(self.sources_unsorted)

    @property
    def source_idxs(self) -> list:
        """Integer indices for all sources in this TrainingData, looked up from global config."""
        return [config.ds_source_to_idx[s] for s in self.sources]


def parse_training_data(s: str, dataset_paths: dict, target_device: str | None) -> TrainingData:
    """Convert a string like 'cmod_tcv' or 'exnihilo' to a TrainingData object."""
    if s == "exnihilo":
        non_target = set(dataset_paths.keys()) - ({target_device} if target_device else set())
        return TrainingData(sources_unsorted=sorted(non_target), exnihilo=True)
    sources = s.split("_")
    if target_device in sources:
        raise ValueError(
            f"training_data {s!r} names the target device {target_device!r}, whose held-out test shots would land in the training set"
        )
    return TrainingData(sources_unsorted=sources)


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
    The shape variables let the sciml profile submodule skeleton run its
    PCA / k-means initial guess on this dataset (ProfilePredictorTRB.model_init).
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
) -> tuple[xr.Dataset, str]:
    """Open a device store and prepare it for one study.

    Reads only the store signals the study needs (the power balance study never loads the profiles),
    converts to working units, keeps the max_ds_size most recent shots,
    checks every shot is one contiguous 1 kHz segment,
    adds the hazard metric on the full 0D series so every study holds out the same target shots,
    drops the shots without one, then runs the study's prep.

    Args:
        source_ds (str): Identifier for the source dataset.
        study_type (str): Type of study for which to prepare the dataset.

    Returns:
        tuple[xr.Dataset, str]: The processed dataset and the dimension along which to group the data
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
    if config.max_ds_size is not None and n_shots_stored > config.max_ds_size:
        logger.warning(f"max_ds_size keeps the {config.max_ds_size} most recent of the {n_shots_stored} {source_ds} shots")
        ds = ds.isel({EPISODE_DIM: slice(0, config.max_ds_size)})

    check_uniform_timebase(ds)

    ds = add_hazard(ds, EPISODE_DIM)
    mask_hazard_valid = ds["hazard"].notnull().values
    if not mask_hazard_valid.all():
        shots_without_hazard = ds[EPISODE_DIM].values[~mask_hazard_valid]
        logger.warning(f"Dropping {shots_without_hazard.size} {source_ds} shots without a hazard metric: {shots_without_hazard.tolist()}")
        ds = ds.isel({EPISODE_DIM: mask_hazard_valid})

    ds = STUDY_PREPS[study_type](ds)

    # From a zarr store the time variable is a data variable, the dataloaders consume it as the time coordinate
    if TIME_COORD not in ds.coords:
        ds = ds.set_coords(TIME_COORD)

    return ds, EPISODE_DIM


def add_hazard(
    ds: xr.Dataset,
    episode_coord: str,
) -> xr.Dataset:
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


def normalize_domain(
    ds_source: xr.Dataset,
    ds_target: xr.Dataset | None = None,
    method: str = "raw",
    feature_space: str = "power_balance",
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """Apply the specified domain normalization method to the dataset.

    Thin wrapper around transport_study.modules.normalization: the per-method
    math (physics features, per-device z-score, CORAL) is exactly the module
    implementation the models consume, fitted here from ds_source and applied
    to both datasets. Used for data visualization only. Datasets are modified
    in place and returned.

    ds_source is used to inform the normalization parameters (e.g. mean and std for z-score, covariance for coral),
    but the normalization is applied to both source and target datasets.

    feature_space selects which model family's features the stat stage runs on,
    so each study visualizes what its own modules consume:
        - "power_balance": the 7 physics features of normalization.physics_feature_vec
          (q_star, epsilon, aB0, f_G, surface_power_density, ...), fitted over the
          7 physical inputs
        - "profile": the 10 dimensionless nn_inputs of the profile predictor
          (beta, q_star, epsilon, f_G, aB0, beta_tor_norm, elongation,
          triangularity_upper, triangularity_lower, log_nu_star)
        - "transport": those 10 slots with the beta-derived ones computed from
          the measured energy_mhd_MJ, plus the normalized aux power (paux_norm)
    Only the physics* methods honor it, the raw-variable methods ("zscore",
    "coral") are power-balance inputs by definition.

    Methods:
        - "raw": No normalization, ip_MA, energy_mhd_MJ, etc. are in their working units
        - "physics": The module's dimensionless features. In the power-balance space that is
          (q_star, epsilon, aB0, f_G, surface_power_density) plus beta as a visualization-only extra,
          in the profile space the nn_inputs themselves (which carry their own beta)
        - "zscore": Within each device, normalize each variable to zero mean and unit variance. Variable gets a `_z` suffix after normalization. energy_mhd_MJ is a visualization-only extra column (harmless, z-scoring is per-variable)
        - "coral": Use the CORAL method to align covariances of various devices over exactly the model's 7 input vars. Variable gets a `_coral` suffix after normalization.
        - "physics-coral": CORAL alignment over the physics features of the selected feature space. Variable gets a `_pcoral` suffix (a `_coral` suffix would collide with the raw coral vars).
        - "physics-zscore": Per-device z-score over the physics features of the selected feature space. Variable gets a `_pz` suffix.

    Args:
        ds_source: The source dataset (e.g. historic data)
        ds_target: The target dataset (e.g. DIII-D high-hazard shots)
        method: The normalization method to apply
        feature_space: Which model family's feature vector the physics* methods use

    Returns:
        The normalized source and target datasets.
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
    datasets = [ds_source] if ds_target is None else [ds_source, ds_target]

    def _reference(ds: xr.Dataset) -> xr.DataArray:
        # Broadcast template carrying the full per-sample dims
        return ds[NORM_INPUT_VARS[0]]

    def _feature_matrix_for(ds: xr.Dataset, variables: tuple[str, ...]) -> np.ndarray:
        reference = _reference(ds)
        missing = [var for var in variables if var not in ds]
        if missing:
            raise ValueError(f"Feature matrix needs {missing}, the dataset lacks them")
        columns = [np.asarray(ds[var].broadcast_like(reference).values, dtype=float).ravel() for var in variables]
        return np.column_stack(columns)

    def _physics_matrix(ds: xr.Dataset) -> np.ndarray:
        if physics_matrix_fn is not None:
            return physics_matrix_fn(ds)
        raw = jnp.asarray(_feature_matrix_for(ds, NORM_INPUT_VARS))
        return np.asarray(jax.vmap(physics_feature_vec)(raw))

    # Physics slots that are identity mappings of a raw input var of the same name
    identity_slots = set(physics_source_vars)

    def _write_features(ds: xr.Dataset, matrix: np.ndarray, names: tuple[str, ...], suffix: str) -> None:
        reference = _reference(ds)
        for j, name in enumerate(names):
            ds[f"{name}{suffix}"] = (reference.dims, np.asarray(matrix[:, j], dtype=float).reshape(reference.shape))

    def _add_physics_vars(ds: xr.Dataset) -> None:
        phys = _physics_matrix(ds)
        reference = _reference(ds)
        for j, name in enumerate(physics_names):
            # Identity slots (ip_MA / elongation, plus beta_tor_norm / triangularity_* in the profile
            # feature space) already exist as raw vars
            if name in identity_slots:
                continue
            ds[name] = (reference.dims, np.asarray(phys[:, j], dtype=float).reshape(reference.shape))
        if feature_space != "power_balance":
            # The profile and transport feature vectors carry their own beta slot
            return
        # beta needs the stored energy, which is the predicted state rather than
        # a model input, so it is a visualization-only extra.
        # The same fraction the modules call beta, through the store's betan formula with volume_approx
        volume_m3 = plasma_parameters.volume_approx(ds["geometric_axis_r"], ds["minor_radius"], ds["elongation"])
        beta_tor_norm = plasma_parameters.beta_tor_norm_from_energy_mhd_MJ(
            ds["energy_mhd_MJ"], volume_m3, ds["minor_radius"], ds["b0"], ds["ip_MA"]
        )
        ds["beta"] = plasma_parameters.beta_tor_from_beta_tor_norm(beta_tor_norm, ds["ip_MA"], ds["minor_radius"], ds["b0"])

    if method == "raw":
        return ds_source, ds_target
    if method == "physics":
        for ds in datasets:
            _add_physics_vars(ds)
        return ds_source, ds_target

    # Stat-bearing methods key per-device statistics on an integer index built
    # locally from the ds_source coordinate (the modules use the global
    # config.ds_source_to_idx, but any consistent indexing gives the same stats)
    registry: dict[str, int] = {}
    for ds in datasets:
        for device in np.atleast_1d(ds.coords["ds_source"].values):
            registry.setdefault(str(device), len(registry))

    def _source_idx_for(ds: xr.Dataset) -> np.ndarray:
        idx_da = xr.apply_ufunc(np.vectorize(lambda d: registry[str(d)]), ds.coords["ds_source"])
        return np.asarray(idx_da.broadcast_like(_reference(ds)).values).ravel().astype(int)

    if method == "zscore":
        zscore_vars = (*NORM_INPUT_VARS, "energy_mhd_MJ")
        means, stds = fit_z_score_stats(_feature_matrix_for(ds_source, zscore_vars), _source_idx_for(ds_source), len(registry))
        for ds in datasets:
            matrix = apply_z_score(jnp.asarray(_feature_matrix_for(ds, zscore_vars)), _source_idx_for(ds), means, stds)
            _write_features(ds, np.asarray(matrix), zscore_vars, "_z")
        return ds_source, ds_target

    if method == "physics-zscore":
        means, stds = fit_z_score_stats(_physics_matrix(ds_source), _source_idx_for(ds_source), len(registry))
        for ds in datasets:
            matrix = apply_z_score(jnp.asarray(_physics_matrix(ds)), _source_idx_for(ds), means, stds)
            _write_features(ds, np.asarray(matrix), physics_names, "_pz")
        return ds_source, ds_target

    def _shot_idx_for(ds: xr.Dataset) -> np.ndarray:
        return np.asarray(ds.coords["shot"].broadcast_like(_reference(ds)).values).ravel()

    def _coral_normalization(variables: tuple[str, ...], suffix: str, matrix_fn) -> None:
        # Devices below MIN_CORAL_SHOTS (or absent from ds_source) keep the
        # identity transform, so their features pass through raw. Rows with any
        # NaN feature come out all-NaN (the joint transform needs complete rows).
        stats = fit_coral_stats(matrix_fn(ds_source), _source_idx_for(ds_source), len(registry), _shot_idx_for(ds_source))
        means, transforms = identity_coral_stats(len(registry), len(variables)) if stats is None else stats
        batched_apply = jax.vmap(apply_coral, in_axes=(0, 0, None, None))
        for ds in datasets:
            matrix = batched_apply(jnp.asarray(matrix_fn(ds)), jnp.asarray(_source_idx_for(ds)), means, transforms)
            _write_features(ds, np.asarray(matrix), variables, suffix)

    if method == "coral":
        _coral_normalization(NORM_INPUT_VARS, "_coral", lambda ds: _feature_matrix_for(ds, NORM_INPUT_VARS))
        return ds_source, ds_target
    if method == "physics-coral":
        _coral_normalization(physics_names, "_pcoral", _physics_matrix)
        return ds_source, ds_target

    raise ValueError(f"Unknown normalization method: {method}")


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
    episode_coord = None

    for source in training_data.sources:
        ds, episode_coord = get_ds(source, study_type)
        train_src, val_src = split_dataset_by_fracs(
            ds,
            fracs=TRAIN_VAL_SPLIT,
            dim=episode_coord,
            sortby="hazard",
        )
        src_idx = config.ds_source_to_idx[source]
        train_src["ds_source_idx"] = (
            episode_coord,
            np.full(train_src.sizes[episode_coord], src_idx),
        )
        val_src["ds_source_idx"] = (
            episode_coord,
            np.full(val_src.sizes[episode_coord], src_idx),
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
        train_ds = concat_with_nan_padding([pair[0] for pair in ds_sources.values()], concat_dim=episode_coord)
        val_ds = concat_with_nan_padding([pair[1] for pair in ds_sources.values()], concat_dim=episode_coord)

    logger.debug("Historic Training dataset size: {}", train_ds.sizes[episode_coord])
    logger.debug("Historic Validation dataset size: {}", val_ds.sizes[episode_coord])

    return train_ds, val_ds


def get_loaded_shot_count(source_ds: str, study_type: str = "profile_transfer") -> int:
    """Episode count get_ds actually yields for this device.

    This is the shot count the training set really sees, after max_ds_size
    truncation and study-type filtering (e.g. dropping shots without fresh
    profiles), unlike the raw on-disk shot count. Requires loading the dataset
    from disk, so it is not free, but it is only needed once per device when
    building a train config.
    """
    ds, episode_coord = get_ds(source_ds, study_type)
    return int(ds.sizes[episode_coord])


def _split_target_shots(
    num_target_shots: int,
    target_test_set_size: int,
    study_type: str,
):
    """Load the target device and split it into training shots and the held-out test set.

    A deterministic hazard sort: the test set is the target_test_set_size highest-hazard shots,
    the training shots are the first num_target_shots of the remaining pool
    (or every shot for -1, the cheating upper-bound reference).
    Returns (train_ds_target, test_ds, episode_coord).
    """
    target = config.target_device
    if target is None:
        raise ValueError("config.target_device must be set before transfer learning")

    ds_target, episode_coord = get_ds(target, study_type=study_type)
    ds_target["ds_source_idx"] = (
        episode_coord,
        np.full(ds_target.sizes[episode_coord], config.ds_source_to_idx[target]),
    )
    ds_target = ds_target.assign_coords(ds_source=target)
    sorted_shots = np.argsort(ds_target["hazard"].values)

    test_shot_pool = sorted_shots[-target_test_set_size:] if target_test_set_size else sorted_shots[:0]
    test_ds = ds_target.isel({episode_coord: test_shot_pool})

    if num_target_shots == -1:
        # All available target shots in training and testing (upper-bound reference, CHEATING!)
        train_ds_target = ds_target.isel({episode_coord: sorted_shots})
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
        train_ds_target = ds_target.isel({episode_coord: train_shot_pool})

    return train_ds_target, test_ds, episode_coord


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
    train_ds_target, test_ds, episode_coord = _split_target_shots(num_target_shots, target_test_set_size, study_type)

    # Load historic source data for training (for exnihilo it is stripped again below)
    # exnihilo.sources contains all non-target devices, so we can pass training_data directly
    train_ds_hist, val_ds_hist = get_train_val_datasets(training_data, study_type=study_type)
    train_ds = concat_with_nan_padding(
        [train_ds_hist, val_ds_hist, train_ds_target],
        concat_dim=episode_coord,
    )

    # For 'transfer' and exnihilo: strip historic data, train only on target device shots
    if domain_adaptation == "transfer" or training_data.exnihilo:
        train_ds = train_ds.where(
            train_ds["ds_source_idx"] == config.ds_source_to_idx[config.target_device],
            drop=True,
        )

    logger.debug("HP Training dataset size: {}", train_ds.sizes[episode_coord])
    logger.debug("HP Test dataset size: {}", test_ds.sizes[episode_coord])

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
    train_ds_target, test_ds, episode_coord = _split_target_shots(num_target_shots, target_test_set_size, study_type)

    train_ds_hist, val_ds_hist = get_train_val_datasets(training_data, study_type=study_type)
    train_ds_hist = concat_with_nan_padding([train_ds_hist, val_ds_hist], concat_dim=episode_coord)
    train_ds_combined = concat_with_nan_padding([train_ds_hist, train_ds_target], concat_dim=episode_coord)

    logger.debug("Transfer pretrain historic dataset size: {}", train_ds_hist.sizes[episode_coord])
    logger.debug("Transfer pretrain normalizer-fit dataset size: {}", train_ds_combined.sizes[episode_coord])
    logger.debug("Transfer pretrain test dataset size: {}", test_ds.sizes[episode_coord])

    return train_ds_hist, train_ds_combined, test_ds
