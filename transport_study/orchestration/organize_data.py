import jax
import numpy as np
import xarray as xr
from loguru import logger
from popsim.cfspopcon_jax.current_drive import calc_f_shaping, calc_q_star
from popsim.cfspopcon_jax.geometry import calc_plasma_surface_area, calc_plasma_volume
from popsim.ml.split_utils import split_dataset_by_fracs
from scipy.constants import mu_0
from scipy.linalg import fractional_matrix_power

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.config import TRAIN_VAL_SPLIT, config

# Canonical idx for each device, used to pass device identity through the dataloader as a float variable.
DS_SOURCE_TO_IDX: dict[str, int] = {"cmod": 0, "tcv": 1, "d3d_lp": 2, "d3d_hp": 3}
IDX_TO_DS_SOURCE: dict[int, str] = {v: k for k, v in DS_SOURCE_TO_IDX.items()}


def _add_ds_source_idx(ds: xr.Dataset) -> xr.Dataset:
    """Broadcast the per-shot ds_source coordinate to a (shot, time_idx) data variable.

    `ds_source` is a string coordinate that lives only on the shot dimension and is
    dropped when the dataloader flattens (shot, time_idx) -> sample.  Converting it to
    an integer data variable makes it survive that reshape so loss functions can look up
    per-sample device weights via targ["ds_source_idx"].
    """
    ds_source = ds.coords["ds_source"]
    float_type = ds[next(iter(ds.data_vars))].dtype  # match dataset float precision
    if ds_source.dims == ():  # scalar coordinate for single device
        int_val = DS_SOURCE_TO_IDX[ds_source.item()]
        arr = np.full(
            (ds.sizes[EPISODE_DIM], ds.sizes[TIME_DIM]), int_val, dtype=float_type
        )
    else:  # per-shot coordinate (shot,)
        int_vals = np.array(
            [DS_SOURCE_TO_IDX[s] for s in ds_source.values], dtype=float_type
        )
        arr = np.broadcast_to(
            int_vals[:, None], (len(int_vals), ds.sizes[TIME_DIM])
        ).copy()
    return ds.assign({"ds_source_idx": xr.DataArray(arr, dims=[EPISODE_DIM, TIME_DIM])})


REQUIRED_SIGNALS_POWER_BALANCE = [
    "Wtot_MJ",
    "Ip_MA",
    "B0",
    "R0",
    "kappa",
    "a_minor",  # For inverse aspect ratio
]
INPUT_POWER_SIGNALS = ["P_ECRH_MW", "P_NBI_MW", "P_ICRF_MW", "P_LH_MW"]

REQUIRED_SIGNALS_PROFILE_TRANSFER = [
    "time",
    "Te_keV_psi",
    "ne20_psi",
    "fresh_profiles",
    "Ip_MA",
    "B0",
    "betan",
    "ne20_edge",
    "R0",
    "a_minor",
    "kappa",
    "delta_top",
    "delta_bot",
    "Wtot_MJ",  # Not strictly necessary but used for performance extrapolation
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
        aligned_datasets = xr.align(
            *prepared_datasets, join="outer", fill_value=np.nan, exclude=concat_dim
        )
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


def get_ds(
    source_ds: str,
    study_type: str,
    debug: bool | None = config.debug,
) -> tuple[xr.Dataset, str]:
    """Open the dataset, and do some light processing to get it ready for training.

    Args:
        source_ds (str): Identifier for the source dataset.
        study_type (str): Type of study for which to prepare the dataset.
        debug (bool, optional): Whether to enable debug mode, reducing dataset size to at most 50 shots.

    Returns:
        tuple[xr.Dataset, str]: The processed dataset and the dimension along which to group the data
    """
    if source_ds == "cmod":
        ds_path = config.cmod_dataset_path
    elif source_ds == "tcv":
        ds_path = config.tcv_dataset_path
    elif source_ds == "d3d_lp":
        ds_path = config.d3d_lp_dataset_path
    elif source_ds == "d3d_hp":
        ds_path = config.d3d_hp_dataset_path
    else:
        raise ValueError(f"Unknown source dataset: {source_ds}")

    ds = xr.open_dataset(ds_path).astype(
        jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32
    )

    if debug:
        ds = ds.isel(shot=slice(0, 10))  # Limit to 10 shots
    else:
        # Sort dataset by shot count, get the X most recent as set by config
        ds = ds.sortby("shot", ascending=False).isel(shot=slice(0, config.max_ds_size))

    def _profile_transfer(ds: xr.Dataset) -> xr.Dataset:
        ds = ds[REQUIRED_SIGNALS_PROFILE_TRANSFER]

        # Only keep fresh profiles for training
        ds = ds.where(ds["fresh_profiles"] == 1, drop=True)
        # TCV only has profile data out to rho=1 / psi_n=1
        # Put all the datasets on a uniform 51 point psi_n grid for consistency
        psi_n_grid = np.linspace(0, 1, 51)
        ds = ds.interp(psi_n=psi_n_grid, kwargs={"fill_value": "extrapolate"})

        # Compute means and shapes.
        ds["Te_keV_line_avg"] = ds["Te_keV_psi"].integrate("psi_n")
        ds["ne20_line_avg"] = ds["ne20_psi"].integrate("psi_n")
        ds["Te_shape"] = ds["Te_keV_psi"] / ds["Te_keV_line_avg"]
        ds["ne_shape"] = ds["ne20_psi"] / ds["ne20_line_avg"]
        return ds

    def _power_balance(ds: xr.Dataset) -> xr.Dataset:
        # Ensure all required signals are present
        for signal in REQUIRED_SIGNALS_POWER_BALANCE:
            if signal not in ds:
                raise ValueError(
                    f"Required signal for training {signal} not found in dataset."
                )

        # Additional signals and duplicates for slight renames between submodules
        # This is for the individual submodule training to work, since when they're running on their own they expect these names.
        for signal in INPUT_POWER_SIGNALS:
            if signal not in ds:
                ds[signal] = xr.zeros_like(ds["Ip_MA"])

        # Calculate aux power and absorbed power
        ds["P_aux_MW"] = (
            ds["P_NBI_MW"] + ds["P_ECRH_MW"] + ds["P_ICRF_MW"] + ds["P_LH_MW"]
        )

        return ds

    if study_type == "profile_transfer":
        ds = _profile_transfer(ds)
    elif study_type == "power_balance_transfer":
        ds = _power_balance(ds)
    else:
        raise ValueError(f"Unknown study type: {study_type}")

    # If dataset was from a zarr store, must promote the 'time' data var to a coordinate
    if TIME_COORD not in ds.coords:
        ds = ds.set_coords(TIME_COORD)

    # Dataset retains all signals, the dataloader will filter out the ones that are not needed.
    return ds, EPISODE_DIM


def add_performance(
    ds: xr.Dataset,
    episode_coord: str,
) -> xr.Dataset:
    """
    Add performance metric to dataset
    We are saying performance is 95th percentile of (Wtot_MJ^2 + Ip_MA^2)**0.5 along a shot
    Ignoring nans in the calculation

    Also stores the specific Ip_MA and Wtot_MJ values at the time point where the
    performance metric reaches its 95th percentile for plotting in parameter space
    """

    max_Wtot = ds["Wtot_MJ"].max().item()
    max_Ip = ds["Ip_MA"].max().item()
    Wtot_scale = 1.0 / max_Wtot if max_Wtot != 0 else 1.0
    Ip_scale = 1.0 / max_Ip if max_Ip != 0 else 1.0

    # Calculate performance at each time step (once for all shots)
    perf_timeseries = ds.eval(
        f"(({Wtot_scale} * Wtot_MJ)**2 + ({Ip_scale} * Ip_MA)**2)**0.5"
    )

    # Get the 95th percentile value per shot
    if TIME_DIM in ds.dims:
        ds["performance"] = perf_timeseries.quantile(0.95, dim=TIME_DIM, skipna=True)
    else:
        ds["performance"] = perf_timeseries.quantile(0.95, dim=TIME_COORD, skipna=True)

    n_shots = ds.sizes[episode_coord]

    # Initialize arrays for Ip_MA and Wtot_MJ at p95
    ip_ma_p95 = np.full(n_shots, np.nan)
    wtot_mj_p95 = np.full(n_shots, np.nan)

    # For each shot, find the time index closest to 95th percentile
    perf_ts_data = perf_timeseries.values  # shape: (n_shots, n_time)
    p95_vals = ds["performance"].values  # shape: (n_shots,)
    ip_ma_data = ds["Ip_MA"].values
    wtot_mj_data = ds["Wtot_MJ"].values

    for i in range(n_shots):
        # Get performance timeseries for this shot
        perf_shot = perf_ts_data[i]
        p95_val = p95_vals[i]

        # Find valid (non-NaN) indices
        valid_mask = ~np.isnan(perf_shot)

        if valid_mask.sum() > 0 and not np.isnan(p95_val):
            # Find index where performance is closest to p95
            abs_diff = np.abs(perf_shot - p95_val)
            abs_diff[~valid_mask] = np.inf  # Ignore NaN positions
            idx_p95 = np.argmin(abs_diff)

            # Extract Ip_MA and Wtot_MJ at that time
            ip_ma_p95[i] = ip_ma_data[i, idx_p95]
            wtot_mj_p95[i] = wtot_mj_data[i, idx_p95]

    # Add to dataset
    ds["Ip_MA_p95"] = (episode_coord, ip_ma_p95)
    ds["Wtot_MJ_p95"] = (episode_coord, wtot_mj_p95)

    return ds


def normalize_domain(  # noqa: PLR0915
    ds_source: xr.Dataset,
    ds_target: xr.Dataset | None = None,
    method: str | None = "raw",
) -> tuple[xr.Dataset, xr.Dataset]:
    """Apply the specified domain normalization method to the dataset.

    ds_source is used to inform the normalization parameters (e.g. mean and std for z-score, covariance for coral),
    but the normalization is applied to both source and target datasets.

    Methods:
        - "raw": No normalization, Ip, Wtot, etc. are in their original units
        - "physics": Convert to typical dimensionless parameters like beta, q95, f_G, etc.
        - "z_score": Within each device, normalize each variable to zero mean and unit variance. Variable gets a `_zscore` suffix after normalization.
        - "coral": Use the CORAL method to align covariances of various devices. Variable gets a `_coral` suffix after normalization.

    Args:
        ds_source: The source dataset (e.g. historic data)
        ds_target: The target dataset (e.g. DIII-D high-performance shots)
        method: The normalization method to apply

    Returns:
        The normalized source and target datasets.
    """
    normalize_vars = [
        "Ip_MA",
        "B0",
        "ne20_line_avg",
        "R0",
        "kappa",
        "a_minor",
        "Wtot_MJ",
        "P_aux_MW",
    ]

    def _separate_devices(ds: xr.Dataset) -> dict[str, xr.Dataset]:
        devices = np.unique(ds.coords["ds_source"].values)
        return {
            device: ds.where(ds.coords["ds_source"] == device, drop=True)
            for device in devices
        }

    def _physics_normalization(ds_source: xr.Dataset, ds_target: xr.Dataset | None):
        def _epsilon(ds: xr.Dataset) -> xr.DataArray:
            return ds["a_minor"] / ds["R0"]

        def _beta(ds: xr.Dataset) -> xr.DataArray:
            avg_pressure = (
                (2.0 / 3.0)
                * (ds["Wtot_MJ"] * 1e6)
                / calc_plasma_volume(ds["R0"], ds["epsilon"], ds["kappa"])
            )
            magnetic_pressure = (ds["B0"] ** 2) / (2 * mu_0)
            beta = 100 * avg_pressure / magnetic_pressure
            return beta

        def _q_star(ds: xr.Dataset) -> xr.DataArray:
            # TODO(ZanderKeith) using 0 triangularity because it isn't part of H89/H98.
            # Do we care about doing that comparison? If not, could easily add delta_top and delta_bottom to the dataset and use them here.
            f_shaping = calc_f_shaping(
                ds["epsilon"], ds["kappa"], xr.zeros_like(ds["epsilon"])
            )
            q_star = calc_q_star(
                ds["B0"], ds["R0"], ds["epsilon"], ds["Ip_MA"], f_shaping
            )
            return q_star

        def _greenwald_fraction(ds: xr.Dataset) -> xr.DataArray:
            greenwald_limit = ds["Ip_MA"] / (np.pi * ds["a_minor"] ** 2)
            f_G = ds["ne20_line_avg"] / greenwald_limit
            return f_G

        def _aB0(ds: xr.Dataset) -> xr.DataArray:
            aB0 = ds["a_minor"] * ds["B0"]
            return aB0

        def _surface_power_density(ds: xr.Dataset) -> xr.DataArray:
            surface_area = calc_plasma_surface_area(
                ds["R0"], ds["epsilon"], ds["kappa"]
            )
            if "P_aux_MW" not in ds:
                power_density = xr.zeros_like(surface_area)
            else:
                power_density = ds["P_aux_MW"] / surface_area
            return power_density

        for ds in [ds_source, ds_target] if ds_target is not None else [ds_source]:
            ds["epsilon"] = _epsilon(ds)
            ds["beta"] = _beta(ds)
            ds["q_star"] = _q_star(ds)
            ds["f_G"] = _greenwald_fraction(ds)
            ds["aB0"] = _aB0(ds)
            ds["surface_power_density"] = _surface_power_density(ds)

        return ds_source, ds_target

    def _z_score_normalization(ds_source: xr.Dataset, ds_target: xr.Dataset | None):
        # Calculate mean and std from source dataset, then apply to both source and target
        # This is done per device to avoid washing out differences in variable distributions across devices
        # Calculate normalization parameters from source dataset per device
        norm_params = {}
        source_devices = _separate_devices(ds_source)

        for device, ds_device in source_devices.items():
            device_params = {}
            for var in normalize_vars:
                if var in ds_device:
                    # Calculate mean and std across all dimensions except coordinates
                    data_var = ds_device[var]
                    mean_val = data_var.mean(skipna=True)
                    std_val = data_var.std(skipna=True)

                    # Avoid division by zero
                    std_val = std_val.where(std_val != 0, 1.0)

                    device_params[var] = {"mean": mean_val, "std": std_val}
            norm_params[device] = device_params

        # Apply normalization to source dataset
        ds_source_norm = ds_source.copy()
        for var in normalize_vars:
            # Start with raw values, then overwrite per-device
            z_var = ds_source_norm[var].copy()
            for device, device_params in norm_params.items():
                if var in device_params:
                    mean_val = device_params[var]["mean"]
                    std_val = device_params[var]["std"]
                    mask = ds_source_norm.coords["ds_source"] == device
                    z_var = z_var.where(
                        ~mask, (ds_source_norm[var] - mean_val) / std_val
                    )
            ds_source_norm[f"{var}_z"] = z_var

        # Apply same normalization to target dataset if provided
        if ds_target is not None:
            ds_target_norm = ds_target.copy()
            for var in normalize_vars:
                z_var = ds_target_norm[var].copy()
                for device, device_params in norm_params.items():
                    if var in device_params:
                        mean_val = device_params[var]["mean"]
                        std_val = device_params[var]["std"]
                        mask = ds_target_norm.coords["ds_source"] == device
                        z_var = z_var.where(
                            ~mask, (ds_target_norm[var] - mean_val) / std_val
                        )
                ds_target_norm[f"{var}_z"] = z_var
        else:
            ds_target_norm = None

        return ds_source_norm, ds_target_norm

    def _coral_normalization(ds_source: xr.Dataset, ds_target: xr.Dataset | None):  # noqa: PLR0915
        # CORAL aligns second-order statistics (covariance) across domains.
        # We use the pooled source data as the reference domain and transform
        # each device's features so their covariance matches the reference.
        # The transform for device d is: center, whiten with C_d^{-1/2}, re-color with C_ref^{1/2}, then re-add mean.
        # A small regularization term is added to covariance diagonals for numerical stability.
        # TODO(ZanderKeith) vet this thoroughly!!!

        reg = 1e-6  # Regularization for covariance matrix inversion

        def _build_feature_matrix(ds: xr.Dataset, variables: list[str]) -> np.ndarray:
            """Build (N, D) feature matrix from dataset, flattening all dims except variables."""
            arrays = []
            for var in variables:
                arr = ds[var].values.flatten()
                arrays.append(arr)
            return np.column_stack(arrays)

        def _coral_transform(
            X: np.ndarray,
            mu_source: np.ndarray,
            cov_source: np.ndarray,
            cov_ref: np.ndarray,
        ) -> np.ndarray:
            """Apply CORAL transformation: whiten with source covariance, re-color with reference."""
            d = cov_source.shape[0]
            cov_source_reg = cov_source + reg * np.eye(d)
            cov_ref_reg = cov_ref + reg * np.eye(d)

            cs_neg_half = np.real(fractional_matrix_power(cov_source_reg, -0.5))
            cr_pos_half = np.real(fractional_matrix_power(cov_ref_reg, 0.5))

            X_centered = X - mu_source
            X_transformed = X_centered @ cs_neg_half @ cr_pos_half + mu_source
            return X_transformed

        # Compute reference statistics from pooled source data
        all_source_features = _build_feature_matrix(ds_source, normalize_vars)
        valid_rows = ~np.any(np.isnan(all_source_features), axis=1)
        all_source_valid = all_source_features[valid_rows]
        cov_ref = np.cov(all_source_valid, rowvar=False)

        # Compute per-device statistics from source
        source_devices = _separate_devices(ds_source)
        device_stats = {}
        for device, ds_device in source_devices.items():
            X_device = _build_feature_matrix(ds_device, normalize_vars)
            valid = ~np.any(np.isnan(X_device), axis=1)
            X_valid = X_device[valid]
            if len(X_valid) > 1:
                device_stats[device] = {
                    "mean": np.mean(X_valid, axis=0),
                    "cov": np.cov(X_valid, rowvar=False),
                }

        def _apply_coral_to_ds(
            ds: xr.Dataset,
            device_stats: dict,
            cov_ref: np.ndarray,
        ) -> xr.Dataset:
            """Apply CORAL transformation to a dataset, writing results with _coral suffix."""
            ds_norm = ds.copy()
            # Initialize coral variables with raw values
            for var in normalize_vars:
                ds_norm[f"{var}_coral"] = ds_norm[var].copy()

            source_vals = ds.coords["ds_source"].values
            # Handle scalar ds_source (single-device dataset)
            if np.ndim(source_vals) == 0:
                devices_in_ds = [str(source_vals)]
            else:
                devices_in_ds = np.unique(source_vals)

            for device in devices_in_ds:
                if device not in device_stats:
                    logger.warning(
                        "Device {} not in source stats, skipping CORAL for it",
                        device,
                    )
                    continue

                # Determine which shots belong to this device
                if np.ndim(source_vals) == 0:
                    # Scalar ds_source: all shots belong to this single device
                    ds_device = ds
                    is_single_device = True
                else:
                    mask_xr = ds.coords["ds_source"] == device
                    ds_device = ds.where(mask_xr, drop=True)
                    device_indices = np.where(np.atleast_1d(mask_xr.values))[0]
                    is_single_device = False

                # Build feature matrix for this device in the dataset
                X_device = _build_feature_matrix(ds_device, normalize_vars)

                # Handle NaNs: transform valid rows, leave NaNs in place
                valid = ~np.any(np.isnan(X_device), axis=1)
                X_transformed = X_device.copy()
                if valid.sum() > 0:
                    X_transformed[valid] = _coral_transform(
                        X_device[valid],
                        device_stats[device]["mean"],
                        device_stats[device]["cov"],
                        cov_ref,
                    )

                # Write back transformed values per variable
                device_shape = ds_device[normalize_vars[0]].shape
                for j, var in enumerate(normalize_vars):
                    col = X_transformed[:, j].reshape(device_shape)
                    if is_single_device:
                        # All shots are this device, just assign directly
                        ds_norm[f"{var}_coral"].values = col
                    else:
                        full_vals = ds_norm[f"{var}_coral"].values.copy()
                        for idx_out, idx_in in enumerate(device_indices):
                            full_vals[idx_in] = col[idx_out]
                        ds_norm[f"{var}_coral"].values = full_vals

            return ds_norm

        ds_source_norm = _apply_coral_to_ds(ds_source, device_stats, cov_ref)

        if ds_target is not None:
            ds_target_norm = _apply_coral_to_ds(ds_target, device_stats, cov_ref)
        else:
            ds_target_norm = None

        return ds_source_norm, ds_target_norm

    if method == "raw":
        return ds_source, ds_target
    elif method == "physics":
        return _physics_normalization(ds_source, ds_target)
    elif method == "z_score":
        return _z_score_normalization(ds_source, ds_target)
    elif method == "coral":
        return _coral_normalization(ds_source, ds_target)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_train_val_datasets(
    training_data: str,
    data_normalization: str,
    study_type: str = "profile_transfer",
):
    """
    Split dataset into training and validation sets based on the specified training data case.

    The reason why we only have train and val sets here is because our true test set is the high-performance D3D shots, handled separately.
    That means all our historic data can be used for training (with the model) and validation (picking the best checkpoint / hyperparameters).
    """

    if training_data in ["cmod", "tcv", "d3d_lp"]:
        # Single device historic training data
        ds, episode_coord = get_ds(training_data, study_type)
        ds = add_performance(ds, episode_coord)
        train_ds, val_ds = split_dataset_by_fracs(
            ds,
            fracs=TRAIN_VAL_SPLIT,
            dim=episode_coord,
            seed=42,
            sortby="performance",
        )
        train_ds = train_ds.assign_coords(ds_source=training_data)
        val_ds = val_ds.assign_coords(ds_source=training_data)

    else:
        # Multi-device historic training data
        ds_cmod, episode_coord = get_ds("cmod", study_type=study_type)
        ds_cmod = add_performance(ds_cmod, episode_coord)
        train_ds_cmod, val_ds_cmod = split_dataset_by_fracs(
            ds_cmod,
            fracs=TRAIN_VAL_SPLIT,
            dim=episode_coord,
            seed=42,
            sortby="performance",
        )
        train_ds_cmod = train_ds_cmod.assign_coords(ds_source="cmod")
        val_ds_cmod = val_ds_cmod.assign_coords(ds_source="cmod")

        ds_tcv, episode_coord = get_ds("tcv", study_type=study_type)
        ds_tcv = add_performance(ds_tcv, episode_coord)
        train_ds_tcv, val_ds_tcv = split_dataset_by_fracs(
            ds_tcv,
            fracs=TRAIN_VAL_SPLIT,
            dim=episode_coord,
            seed=42,
            sortby="performance",
        )
        train_ds_tcv = train_ds_tcv.assign_coords(ds_source="tcv")
        val_ds_tcv = val_ds_tcv.assign_coords(ds_source="tcv")

        if training_data in ["cmod_tcv"]:
            train_ds = concat_with_nan_padding(
                [train_ds_cmod, train_ds_tcv], concat_dim=episode_coord
            )
            val_ds = concat_with_nan_padding(
                [val_ds_cmod, val_ds_tcv], concat_dim=episode_coord
            )

        elif training_data == "cmod_tcv_d3d_lp":
            ds_d3d_lp, episode_coord = get_ds("d3d_lp", study_type=study_type)
            ds_d3d_lp = add_performance(ds_d3d_lp, episode_coord)
            train_ds_d3d_lp, val_ds_d3d_lp = split_dataset_by_fracs(
                ds_d3d_lp,
                fracs=TRAIN_VAL_SPLIT,
                dim=episode_coord,
                seed=42,
                sortby="performance",
            )
            train_ds_d3d_lp = train_ds_d3d_lp.assign_coords(ds_source="d3d_lp")
            val_ds_d3d_lp = val_ds_d3d_lp.assign_coords(ds_source="d3d_lp")

            train_ds = concat_with_nan_padding(
                [train_ds_cmod, train_ds_tcv, train_ds_d3d_lp],
                concat_dim=episode_coord,
            )
            val_ds = concat_with_nan_padding(
                [val_ds_cmod, val_ds_tcv, val_ds_d3d_lp],
                concat_dim=episode_coord,
            )
        else:
            raise ValueError(f"Unknown training data case: {training_data}")

    logger.debug("Historic Training dataset size: {}", train_ds.sizes[episode_coord])
    logger.debug("Historic Validation dataset size: {}", val_ds.sizes[episode_coord])

    train_ds, val_ds = normalize_domain(train_ds, val_ds, method=data_normalization)

    train_ds = _add_ds_source_idx(train_ds)
    val_ds = _add_ds_source_idx(val_ds)

    return train_ds, val_ds


def get_train_test_datasets(
    training_data: str,
    data_normalization: str,
    domain_adaptation: str,
    num_hp_shots: int,
    hp_test_set_size: int,
    study_type: str = "profile_transfer",
):
    """
    Split dataset into training and test sets for the target learning case.
    If domain adaptation is 'mixing', makes a combined training set of historic data and high-performance DIII-D shots,
    while if domain adaptation is 'transfer' or the training data case is 'exnihilo', removes all historic data from the training set, leaving only the high-performance DIII-D shots.
    The number of high-performance shots included in training is specified by `num_hp_shots`.

    The test set is always the same set of high-performance DIII-D shots
    The training set is made up of the historic data specified by `training_data` plus the `num_hp_shots` highest-performing shots from DIII-D.

    There is no validation set in this case, since we are not tuning hyperparameters in this case.
    We are treating the test set as a validation set in a sense, since we are using it to pick the best checkpoint for evaluation,
    which I understand is a bit cheaty but given the extremely limited amount of high-performance data in some cases
    it would be better to do this than train and validate on the same 3-4 high-performance shots.

    Since we're doing this for all the models it should be a fair comparison.
    """

    # Load the high-performance dataset and split into train/test
    # No validation needed because we are not tuning hyperparameters on transfer learning data
    ds_hp, episode_coord = get_ds("d3d_hp", study_type=study_type)
    ds_hp = add_performance(ds_hp, episode_coord)
    ds_hp = ds_hp.assign_coords(ds_source="d3d_hp")
    sorted_shots = np.argsort(ds_hp["performance"].values)

    test_shot_pool = sorted_shots[-hp_test_set_size:]
    test_ds = ds_hp.isel({episode_coord: test_shot_pool})

    if num_hp_shots == -1:
        # If num_hp_shots is -1, put all available high-performance shots in training and testing set (this is cheating, but allows us to see the maximum theoretical performance)
        train_ds_hp = ds_hp.isel({episode_coord: sorted_shots})
    else:
        train_shot_pool = sorted_shots[:num_hp_shots]
        train_ds_hp = ds_hp.isel({episode_coord: train_shot_pool})

    # Load historic data and put it all in the training set
    if training_data == "exnihilo":
        # Exnihilo still needs historic data for normalization,
        # we will strip out all the data from historic devices later
        train_ds_hist, val_ds_hist = get_train_val_datasets(
            "cmod_tcv", data_normalization, study_type=study_type
        )
    else:
        train_ds_hist, val_ds_hist = get_train_val_datasets(
            training_data, data_normalization, study_type=study_type
        )
    train_ds = concat_with_nan_padding(
        [train_ds_hist, val_ds_hist, train_ds_hp],
        concat_dim=episode_coord,
    )
    # Normalize the combined dataset (using only the historic data to calculate normalization parameters to avoid data leakage from the test set)
    train_ds, test_ds = normalize_domain(train_ds, test_ds, method=data_normalization)

    # Now depending on the domain adaptation method / training data case, remove stuff from the training set
    # Everything should already be set up for the 'mixing' case, but the 'transfer' and 'exnihilo' cases require removing historic data from the training set
    if domain_adaptation == "transfer" or training_data == "exnihilo":
        # Remove all the historic data from the training set, leaving only the high-performance DIII-D shots
        train_ds = train_ds.where(train_ds.coords["ds_source"] == "d3d_hp", drop=True)

    logger.debug("HP Training dataset size: {}", train_ds.sizes[episode_coord])
    logger.debug("HP Test dataset size: {}", test_ds.sizes[episode_coord])

    train_ds = _add_ds_source_idx(train_ds)
    test_ds = _add_ds_source_idx(test_ds)

    return train_ds, test_ds
