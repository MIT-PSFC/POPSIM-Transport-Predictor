from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from loguru import logger
from popsim.cfspopcon_jax.geometry import calc_plasma_volume
from popsim.ml.split_utils import split_dataset_by_fracs
from scipy.constants import mu_0
from transport_validation_datasets.machine.generic import (
    UNIFORM_TIMEBASE_DT,
    make_uniform_1kHz_timebase,
)

from transport_study import EPISODE_DIM, RADIAL_DIM, TIME_COORD, TIME_DIM
from transport_study.config import RHO_GRID, TRAIN_VAL_SPLIT, config
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
from transport_study.signals import convert_to_working_units

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class TrainingData:
    """Specifies which source devices to use for training.

    sources: tuple of device keys
    exnihilo: if True, load sources for normalization only - strip them from the
              actual training set, leaving only target device shots.
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
    return TrainingData(sources_unsorted=s.split("_"))


REQUIRED_SIGNALS_POWER_BALANCE = [
    # Target
    "energy_mhd_MJ",
    # Inputs
    "ip_MA",
    "b0",
    "geometric_axis_r",
    "elongation",
    "minor_radius",  # For inverse aspect ratio
    "n_e_line_average_1e20",
    "power_additional_MW",
]


# Profile channels and the gradient and error-bar companions every device store carries.
# For each base signal <v> the companions are <v>_gradient, <v>_error and <v>_gradient_error.
# An error of 0 is the sentinel for "no rigorous error quantification" (TCV),
# the loss treats it as a zero-width error bar
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
    "beta_tor_norm",
    "n_e_line_average_1e20",
    "geometric_axis_r",
    "minor_radius",
    "elongation",
    "triangularity_upper",
    "triangularity_lower",
    # Extra
    "time",  # Data variable holding per-shot time values, the var selection below would drop it and the dataloader consumes it as the time coordinate
    "energy_mhd_MJ",  # Not strictly necessary but used for hazard extrapolation
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
    "n_e_line_average_1e20",
    "geometric_axis_r",
    "minor_radius",
    "elongation",
    "triangularity_upper",
    "triangularity_lower",
    "power_additional_MW",
    # Extra
    "time",  # Data variable holding per-shot time values, promoted to the time coordinate downstream
    "energy_mhd_MJ",  # Seeds the sciml stored-energy state and the normalizer fit, also used for hazard extrapolation and as an anchor target in the sciml training loss
    "power_ohm_MW",  # Anchor target for the sciml training loss
    "power_radiated_MW",  # Anchor target for the sciml training loss
]


def concat_with_nan_padding(
    datasets: list[xr.Dataset],
    concat_dim: str,
    pad_dim: str = TIME_DIM,
) -> xr.Dataset:
    """Concatenate datasets while padding shorter `pad_dim` with NaNs.

    This is needed when datasets have different lengths along `pad_dim` and
    do not all have an explicit coordinate index for that dimension.
    """
    prepared_datasets = []
    for dataset in datasets:
        if pad_dim in dataset.dims and pad_dim not in dataset.coords:
            ds = dataset.assign_coords({pad_dim: np.arange(dataset.sizes[pad_dim])})
        else:
            ds = dataset
        prepared_datasets.append(ds)

    # Align only along pad_dim to avoid creating duplicates along concat_dim
    if len(prepared_datasets) > 1 and pad_dim in prepared_datasets[0].dims:
        aligned_datasets = xr.align(*prepared_datasets, join="outer", fill_value=np.nan, exclude=concat_dim)
    else:
        aligned_datasets = prepared_datasets

    ds_padded = xr.concat(
        aligned_datasets,
        dim=concat_dim,
        join="outer",
        coords="different",
        compat="equals",
        fill_value=np.nan,
    )

    return ds_padded


def reindex_to_uniform_timebase(ds: xr.Dataset) -> xr.Dataset:
    """Place every shot back on the canonical 1 kHz grid, NaN at missing times.

    The device workflows build their timebases with make_uniform_1kHz_timebase
    (an absolute grid anchored at t=0), but the stored shots are compacted and
    contain mid-shot gaps where time slices were dropped during acquisition or
    dataset generation. Downstream, the simple-Euler stepper integrates with
    real dt, and steps with dt >~ tau_e are numerically unstable. Mapping each
    sample to its slot on the canonical grid turns those gaps into NaN slices,
    which the dataloader's nan_handling can then drop as whole segments, so
    every surviving training segment is contiguous.

    Columns are absolute grid slots, so equal columns mean equal times across
    shots. The time coordinate is NaN outside each shot's first..last sample
    window (leading and trailing padding convention).
    """
    time2d = ds[TIME_COORD].transpose(EPISODE_DIM, TIME_DIM).values
    n_shots = time2d.shape[0]
    finite = np.isfinite(time2d)
    if not finite.any():
        return ds

    slots = time2d / UNIFORM_TIMEBASE_DT
    k = np.rint(np.where(finite, slots, 0)).astype(np.int64)
    k_finite = k[finite]
    if k_finite.min() < 0:
        raise ValueError("Negative time values found while reindexing to the uniform timebase.")

    residual = np.abs(np.where(finite, slots - k, 0.0))
    max_residual = float(residual.max())
    if max_residual > 0.25:
        logger.warning(
            f"Time values deviate from the nominal {UNIFORM_TIMEBASE_DT} s grid by up to {max_residual:.2f} steps, "
            "the stored timebase may not actually be uniform at this rate."
        )

    n_new = int(k_finite.max()) + 1
    shot_i, time_i = np.nonzero(finite)
    grid_i = k[shot_i, time_i]
    keys = shot_i * n_new + grid_i
    if np.unique(keys).size != keys.size:
        dup_shots = np.unique(shot_i[np.isin(keys, keys[np.diff(np.sort(keys), prepend=-1) == 0])])
        raise ValueError(
            f"Multiple samples map to the same uniform-grid slot for shots {ds[EPISODE_DIM].values[dup_shots]}, "
            f"the data is sampled faster than the {UNIFORM_TIMEBASE_DT} s grid."
        )

    k_first = np.full(n_shots, n_new, dtype=np.int64)
    k_last = np.full(n_shots, -1, dtype=np.int64)
    np.minimum.at(k_first, shot_i, grid_i)
    np.maximum.at(k_last, shot_i, grid_i)

    grid = make_uniform_1kHz_timebase(float(np.nanmax(time2d)))
    if grid.size < n_new:
        raise ValueError(f"Canonical timebase has {grid.size} slots but the data spans {n_new}.")
    col = np.arange(n_new)[None, :]
    in_window = (col >= k_first[:, None]) & (col <= k_last[:, None])
    new_time = np.where(in_window, grid[None, :n_new], np.nan).astype(time2d.dtype)

    data_vars = {}
    for name, da in ds.data_vars.items():
        if name == TIME_COORD:
            continue
        if TIME_DIM not in da.dims:
            data_vars[name] = da
            continue
        da_t = da.transpose(EPISODE_DIM, TIME_DIM, ...)
        arr = da_t.values
        new_arr = np.full((n_shots, n_new, *arr.shape[2:]), np.nan, dtype=arr.dtype)
        new_arr[shot_i, grid_i, ...] = arr[shot_i, time_i, ...]
        data_vars[name] = xr.DataArray(new_arr, dims=da_t.dims)

    coords = {name: coord for name, coord in ds.coords.items() if TIME_DIM not in coord.dims}
    time_da = xr.DataArray(new_time, dims=(EPISODE_DIM, TIME_DIM))
    if TIME_COORD in ds.coords:
        coords[TIME_COORD] = time_da
    else:
        data_vars[TIME_COORD] = time_da

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds.attrs)


def get_ds(
    source_ds: str,
    study_type: str,
) -> tuple[xr.Dataset, str]:
    """Open the dataset, and do some light processing to get it ready for training.

    Args:
        source_ds (str): Identifier for the source dataset.
        study_type (str): Type of study for which to prepare the dataset.

    Returns:
        tuple[xr.Dataset, str]: The processed dataset and the dimension along which to group the data
    """
    if source_ds not in config.dataset_paths:
        raise ValueError(f"Unknown source dataset: {source_ds!r}. Available: {set(config.dataset_paths)}")
    ds_path = Path(config.dataset_paths[source_ds])

    ds_store = xr.open_dataset(ds_path)

    if study_type == "power_balance_transfer" and RADIAL_DIM in ds_store.dims:
        # Power balance only uses scalar time series. Drop profile variables
        # before the eager astype below so their (shot, time, rho) arrays are
        # never read from disk. Keeping them OOMs training jobs: concatenating
        # devices with mismatched rho grids NaN-pads every profile variable to
        # the union grid across all shots, blowing past the SLURM memory request
        ds_store = ds_store.drop_dims(RADIAL_DIM)

    # Stores are IMAS names in SI, the study works in the suffixed working units
    ds = convert_to_working_units(ds_store)
    ds = ds.astype(jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32)

    if EPISODE_DIM not in ds.dims:
        raise ValueError(f"Expected dataset to have {EPISODE_DIM} dimension, but it was not found. Found dimensions: {ds.dims}")

    ds = ds.sortby(EPISODE_DIM, ascending=False)  # Most recent shots first
    ds = ds.isel({EPISODE_DIM: slice(0, config.max_ds_size)})

    def _profile_transfer(ds: xr.Dataset) -> xr.Dataset:
        ds = ds[REQUIRED_SIGNALS_PROFILE_TRANSFER]

        # Only keep fresh profiles for training
        ds = ds.where(ds["fresh_profile"] == 1, drop=True)
        # TCV only has profile data out to rho=1
        # Put all the datasets on the shared uniform rho grid for consistency
        ds = ds.interp({RADIAL_DIM: RHO_GRID}, kwargs={"fill_value": "extrapolate"})

        # Linear extrapolation at the grid edges can push error bars slightly
        # negative, error bars are widths so clamp them
        for err_sig in PROFILE_ERROR_SIGNALS:
            ds[err_sig] = ds[err_sig].clip(min=0.0)

        # Compute means and shapes
        ds["t_e_shape"] = ds["t_e_keV"] / ds["t_e_keV"].integrate(RADIAL_DIM)
        ds["n_e_shape"] = ds["n_e_1e20"] / ds["n_e_1e20"].integrate(RADIAL_DIM)
        return ds

    def _power_balance(ds: xr.Dataset) -> xr.Dataset:
        # Ensure all required signals are present
        for signal in REQUIRED_SIGNALS_POWER_BALANCE:
            if signal not in ds:
                raise ValueError(f"Required signal for training {signal} not found in dataset.")

        # Some device datasets carry multi-element numpy arrays in variable
        # attrs (e.g. a 'validity' time range). Attrs become static jit
        # metadata in the xarray pytree registration and arrays there break
        # the treedef equality check, so strip them. (The profile branch loses
        # attrs implicitly through interp, this branch must do it explicitly.)
        for var in ds.variables:
            ds[var].attrs = {}

        # The stored timebases have mid-shot gaps, put every shot on a strict
        # 1 kHz grid with NaN at the missing times so the train dataloader's
        # drop_segment nan_handling only keeps contiguous segments
        ds = reindex_to_uniform_timebase(ds)

        return ds

    def _transport_transfer(ds: xr.Dataset) -> xr.Dataset:
        ds = ds[REQUIRED_SIGNALS_TRANSPORT_TRANSFER]

        # Unlike the profile branch there is NO fresh-profile filter here: the
        # time-dependent rollouts need contiguous segments, so the stale
        # (forward-filled) profile timeslices stay in as targets and
        # fresh_profile rides along as data for masking downstream
        ds = ds.interp({RADIAL_DIM: RHO_GRID}, kwargs={"fill_value": "extrapolate"})

        # Linear extrapolation at the grid edges can push error bars slightly
        # negative, error bars are widths so clamp them
        for err_sig in PROFILE_ERROR_SIGNALS:
            ds[err_sig] = ds[err_sig].clip(min=0.0)

        # Shape variables so the sciml profile submodule skeleton can run its
        # PCA / k-means initial guess on this dataset (ProfilePredictorTRB.model_init)
        ds["t_e_shape"] = ds["t_e_keV"] / ds["t_e_keV"].integrate(RADIAL_DIM)
        ds["n_e_shape"] = ds["n_e_1e20"] / ds["n_e_1e20"].integrate(RADIAL_DIM)

        # Same attr-stripping rationale as the power balance branch: array-valued
        # attrs become static jit metadata and break the treedef equality check
        for var in ds.variables:
            ds[var].attrs = {}

        # Strict 1 kHz grid with NaN at missing times so the train dataloader's
        # drop_segment nan_handling only keeps contiguous segments
        ds = reindex_to_uniform_timebase(ds)

        return ds

    if study_type == "profile_transfer":
        ds = _profile_transfer(ds)
    elif study_type == "power_balance_transfer":
        ds = _power_balance(ds)
    elif study_type == "transport_transfer":
        ds = _transport_transfer(ds)
    else:
        raise ValueError(f"Unknown study type: {study_type}")

    # If dataset was from a zarr store, must promote the 'time' data var to a coordinate
    if TIME_COORD not in ds.coords:
        ds = ds.set_coords(TIME_COORD)

    # Dataset retains all signals, the dataloader will filter out the ones that are not needed.
    return ds, EPISODE_DIM


def add_hazard(
    ds: xr.Dataset,
    episode_coord: str,
) -> xr.Dataset:
    """
    Add hazard metric to dataset
    We are saying hazard is 95th percentile of (energy_mhd_MJ^2 + ip_MA^2)**0.5 along a shot,
    indicating shots are more dangerous with more stored energy and plasma current
    Ignoring nans in the calculation

    Also stores the specific ip_MA and energy_mhd_MJ values at the time point where the
    hazard metric reaches its 95th percentile for plotting in parameter space
    """

    max_energy_mhd = float(ds["energy_mhd_MJ"].max().values)
    max_ip = float(ds["ip_MA"].max().values)
    energy_mhd_scale = 1.0 / max_energy_mhd if max_energy_mhd != 0 else 1.0
    ip_scale = 1.0 / max_ip if max_ip != 0 else 1.0

    # Calculate hazard at each time step (once for all shots)
    energy_mhd_normalized = energy_mhd_scale * ds["energy_mhd_MJ"]
    ip_normalized = ip_scale * ds["ip_MA"]
    hazard_timeseries = (energy_mhd_normalized**2 + ip_normalized**2) ** 0.5

    # Get the 95th percentile value per shot
    ds["hazard"] = hazard_timeseries.quantile(0.95, dim=TIME_DIM, skipna=True)

    n_shots = ds.sizes[episode_coord]

    ip_MA_p95 = np.full(n_shots, np.nan)
    energy_mhd_MJ_p95 = np.full(n_shots, np.nan)

    # Per shot, take ip_MA and energy_mhd_MJ at the timeslice whose hazard is closest to the p95 value
    hazard_ts_data = hazard_timeseries.values  # shape: (n_shots, n_time)
    p95_vals = ds["hazard"].values  # shape: (n_shots,)
    ip_MA_data = ds["ip_MA"].values
    energy_mhd_MJ_data = ds["energy_mhd_MJ"].values

    for i in range(n_shots):
        hazard_shot = hazard_ts_data[i]
        p95_val = p95_vals[i]
        valid_mask = ~np.isnan(hazard_shot)

        if valid_mask.sum() > 0 and not np.isnan(p95_val):
            # Find index where hazard is closest to p95
            abs_diff = np.abs(hazard_shot - p95_val)
            abs_diff[~valid_mask] = np.inf
            idx_p95 = np.argmin(abs_diff)
            ip_MA_p95[i] = ip_MA_data[i, idx_p95]
            energy_mhd_MJ_p95[i] = energy_mhd_MJ_data[i, idx_p95]

    ds["ip_MA_p95"] = (episode_coord, ip_MA_p95)
    ds["energy_mhd_MJ_p95"] = (episode_coord, energy_mhd_MJ_p95)

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
        columns = []
        for var in variables:
            if var in ds:
                col = np.asarray(ds[var].broadcast_like(reference).values, dtype=float).ravel()
            else:
                # Missing signals become zeros (profile-transfer sets lack power_additional_MW unless zero-filled upstream)
                col = np.zeros(reference.size)
            columns.append(col)
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
        # a model input, so it is a visualization-only extra
        epsilon = ds["minor_radius"] / ds["geometric_axis_r"]
        avg_pressure = (2.0 / 3.0) * (ds["energy_mhd_MJ"] * 1e6) / calc_plasma_volume(ds["geometric_axis_r"], epsilon, ds["elongation"])
        ds["beta"] = 100 * avg_pressure / ((ds["b0"] ** 2) / (2 * mu_0))

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
    Split dataset into training and validation sets based on the specified training data case.

    The reason why we only have train and val sets here is because our true test set is the
    high-hazard target device shots, handled separately.
    That means all historic source data can be used for training and validation.
    """
    ds_sources: dict[str, tuple] = {}
    episode_coord = None

    for source in training_data.sources:
        ds, episode_coord = get_ds(source, study_type)
        ds = add_hazard(ds, episode_coord)
        train_src, val_src = split_dataset_by_fracs(
            ds,
            fracs=TRAIN_VAL_SPLIT,
            dim=episode_coord,
            seed=42,
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

    The test set is the target_test_set_size highest-hazard shots
    The training shots are the first num_target_shots of the remaining pool
    (or every shot for -1, the cheating upper-bound reference).
    Returns (train_ds_target, test_ds, episode_coord).
    """
    target = config.target_device
    if target is None:
        raise ValueError("config.target_device must be set before transfer learning")

    ds_target, episode_coord = get_ds(target, study_type=study_type)
    ds_target = add_hazard(ds_target, episode_coord)
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
                f"(debug mode / max_ds_size truncation)?"
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
