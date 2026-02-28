import numpy as np
import xarray as xr
from loguru import logger
from popsim.ml.split_utils import split_dataset_by_fracs

from popsim_transport_predictor.transfer_learning.config import config
from popsim_transport_predictor.transfer_learning.orchestration import (
    HP_SHOTS_INCLUDED,
    TRAIN_VAL_SPLIT,
)

MAX_DS_SIZE_GB = 100  # If the dataset is larger than this, do not load into memory

REQUIRED_SIGNALS = [
    # Signals for power balance predictor
    "Wtot_MJ",
    # To start (because it's easy) we will be comparing against the H98 and H89 scaling laws https://wiki.fusion.ciemat.es/wiki/Scaling_law
    # and threshold powers https://iopscience.iop.org/article/10.1088/1742-6596/123/1/012033/pdf#:~:text=The%20estimated%20power%20law%20scalings,the%20energy%20confinement%20time%20increases.
    "Ip_MA",
    "B0",
    "ne20_line_avg",
    "R0",
    "kappa",
    "a_minor",  # For inverse aspect ratio
    # TODO(ZanderKeith), missing the Hydrogen isotope mass info. Can we get that in these devices?
]

INPUT_POWER_SIGNALS = ["P_ECRH_MW", "P_NBI_MW", "P_ICRF_MW", "P_LH_MW"]


def concat_with_nan_padding(
    datasets: list[xr.Dataset],
    concat_dim: str,
    pad_dim: str = "time_idx",
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
        fill_value=np.nan,
    )

    return ds_padded


def get_ds(
    source_ds: str,
    debug: bool | None = config.debug,
) -> tuple[xr.Dataset, str]:
    """Open the dataset, and do some light processing to get it ready for training.

    Args:
        source_ds (str): Identifier for the source dataset.
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

    ds = xr.open_dataset(ds_path)

    if debug:
        ds = ds.isel(shot=slice(0, 10))  # Limit to 10 shots

    if ds.nbytes < MAX_DS_SIZE_GB * 1e9:
        ds = ds.load()  # Load into memory if not too large

    # Ensure all required signals are present
    for signal in REQUIRED_SIGNALS:
        if signal not in ds:
            raise ValueError(
                f"Required signal for transport predictor training {signal} not found in dataset."
            )

    # Additional signals and duplicates for slight renames between submodules
    # This is for the individual submodule training to work, since when they're running on their own they expect these names.
    for signal in INPUT_POWER_SIGNALS:
        if signal not in ds:
            ds[signal] = xr.zeros_like(ds["Ip_MA"])

    # Calculate aux power and absorbed power
    ds["P_aux_MW"] = ds["P_NBI_MW"] + ds["P_ECRH_MW"] + ds["P_ICRF_MW"] + ds["P_LH_MW"]
    ds["P_abs_MW"] = ds["P_oh_MW"] + ds["P_aux_MW"]

    # Set up signals for the confinement time predictors
    ds["ne19_line_avg"] = ds["ne20_line_avg"] * 10
    ds["epsilon"] = ds["a_minor"] / ds["R0"]
    ds["surface_area_m2"] = 4 * np.pi**2 * ds["R0"] * ds["a_minor"] * ds["kappa"]

    # If dataset was from a zarr store, must promote the 'time' data var to a coordinate
    if "time" not in ds.coords:
        ds = ds.set_coords("time")

    # Dataset retains all signals, the dataloader will filter out the ones that are not needed.
    return ds, "shot"


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
    if "time_idx" in ds.dims:
        ds["performance"] = perf_timeseries.quantile(0.95, dim="time_idx", skipna=True)
    else:
        ds["performance"] = perf_timeseries.quantile(0.95, dim="time", skipna=True)

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


def normalize_domain(  # noqa: PLR0915, PLR0912
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

    def _separate_devices(ds: xr.Dataset) -> dict[str, xr.Dataset]:
        devices = np.unique(ds.coords["ds_source"].values)
        return {
            device: ds.where(ds.coords["ds_source"] == device, drop=True)
            for device in devices
        }

    if method == "raw":
        return ds_source, ds_target
    elif method == "physics":
        ds_source["beta"] = xr.ones_like(ds_source["Ip_MA"])
        ds_source["q95"] = xr.ones_like(ds_source["Ip_MA"])
        ds_source["f_G"] = xr.ones_like(ds_source["Ip_MA"])
        ds_source["aB0"] = xr.ones_like(ds_source["Ip_MA"])
        ds_source["surface_power_density"] = xr.ones_like(ds_source["Ip_MA"])
        if ds_target is not None:
            ds_target["beta"] = xr.ones_like(ds_target["Ip_MA"])
            ds_target["q95"] = xr.ones_like(ds_target["Ip_MA"])
            ds_target["f_G"] = xr.ones_like(ds_target["Ip_MA"])
            ds_target["aB0"] = xr.ones_like(ds_target["Ip_MA"])
            ds_target["surface_power_density"] = xr.ones_like(ds_target["Ip_MA"])
        return ds_source, ds_target
    elif method == "z_score":
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
        for device, ds_device in _separate_devices(ds_source_norm).items():
            if device in norm_params:
                for var in normalize_vars:
                    if var in ds_device and var in norm_params[device]:
                        mean_val = norm_params[device][var]["mean"]
                        std_val = norm_params[device][var]["std"]

                        # Apply z-score normalization to new variable with _z suffix
                        mask = ds_source_norm.coords["ds_source"] == device
                        normalized_values = ds_source_norm[var].where(
                            ~mask, (ds_source_norm[var] - mean_val) / std_val
                        )
                        ds_source_norm[f"{var}_z"] = normalized_values.where(
                            mask, ds_source_norm[var]
                        )

        # Apply same normalization to target dataset if provided
        if ds_target is not None:
            ds_target_norm = ds_target.copy()
            for device, ds_device in _separate_devices(ds_target_norm).items():
                if device in norm_params:
                    for var in normalize_vars:
                        if var in ds_device and var in norm_params[device]:
                            mean_val = norm_params[device][var]["mean"]
                            std_val = norm_params[device][var]["std"]

                            # Apply z-score normalization to new variable with _z suffix
                            mask = ds_target_norm.coords["ds_source"] == device
                            normalized_values = ds_target_norm[var].where(
                                ~mask, (ds_target_norm[var] - mean_val) / std_val
                            )
                            ds_target_norm[f"{var}_z"] = normalized_values.where(
                                mask, ds_target_norm[var]
                            )
        else:
            ds_target_norm = None

        return ds_source_norm, ds_target_norm
    elif method == "coral":
        raise NotImplementedError("CORAL normalization not implemented yet")
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_train_val_datasets(
    training_data_case: str,
    normalization_method: str | None = "raw",
):
    """
    Split dataset into training and validation sets based on the specified training data case.

    The reason why we only have train and val sets here is because our true test set is the high-performance D3D shots, handled separately.
    That means all our historic data can be used for training (with the model) and validation (picking the best checkpoint / hyperparameters).
    """

    if training_data_case in ["cmod", "tcv", "d3d_lp"]:
        # Single device historic training data
        ds, episode_coord = get_ds(training_data_case)
        ds = add_performance(ds, episode_coord)
        train_ds, val_ds = split_dataset_by_fracs(
            ds,
            fracs=TRAIN_VAL_SPLIT,
            dim=episode_coord,
            seed=42,
            sortby="performance",
        )
        train_ds = train_ds.assign_coords(ds_source=training_data_case)
        val_ds = val_ds.assign_coords(ds_source=training_data_case)

    else:
        # Multi-device historic training data
        ds_cmod, episode_coord = get_ds("cmod")
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

        ds_tcv, episode_coord = get_ds("tcv")
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

        if training_data_case == "cmod_tcv":
            train_ds = concat_with_nan_padding(
                [train_ds_cmod, train_ds_tcv], concat_dim=episode_coord
            )
            val_ds = concat_with_nan_padding(
                [val_ds_cmod, val_ds_tcv], concat_dim=episode_coord
            )

        elif training_data_case == "cmod_tcv_d3d_lp":
            ds_d3d_lp, episode_coord = get_ds("d3d_lp")
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
            raise ValueError(f"Unknown training data case: {training_data_case}")

    logger.debug("Training dataset size: {}", train_ds.sizes[episode_coord])
    logger.debug("Validation dataset size: {}", val_ds.sizes[episode_coord])

    train_ds, val_ds = normalize_domain(train_ds, val_ds, method=normalization_method)

    return train_ds, val_ds


def get_train_test_datasets_transfer(
    training_data_case: str,
    num_hp_shots: int,
    normalization_method: str | None = "raw",
):
    """
    Split dataset into training and test sets for the transfer learning case.
    The number of high-performance shots included in training is specified by `num_hp_shots`.

    The test set is always the same set of high-performance DIII-D shots
    The training set is made up of the historic data specified by `training_data_case` plus the `num_hp_shots` highest-performing shots from DIII-D.

    There is no validation set in this case, since we are not tuning hyperparameters in this case.
    We are treating the test set as a validation set in a sense, since we are using it to pick the best checkpoint for evaluation,
    which I understand is a bit cheaty but given the extremely limited amount of high-performance data in some cases
    it would be better to do this than train and validate on the same 3-4 high-performance shots.

    In any case, since we're doing this for all the models it should be a fair comparison.
    """

    # Load the high-performance dataset and split into train/test
    # No validation needed because we are not tuning hyperparameters on transfer learning data
    ds_hp, episode_coord = get_ds("d3d_hp")
    ds_hp = add_performance(ds_hp, episode_coord)
    ds_hp = ds_hp.assign_coords(ds_source="d3d_hp")
    sorted_shots = np.argsort(ds_hp[episode_coord].values)

    max_train_size = len(HP_SHOTS_INCLUDED)
    if num_hp_shots > max_train_size:
        raise ValueError(
            f"num_hp_shots {num_hp_shots} exceeds maximum available {max_train_size}"
        )

    train_shot_pool = sorted_shots[:max_train_size]
    test_shot_pool = sorted_shots[max_train_size:]

    test_ds = ds_hp.isel({episode_coord: test_shot_pool})
    train_ds_hp = ds_hp.isel({episode_coord: train_shot_pool[:num_hp_shots]})

    if training_data_case == "exnihilo":
        train_ds = train_ds_hp
    else:
        # Load historic data and put it all in the training set
        train_ds_hist, val_ds_hist = get_train_val_datasets(training_data_case)
        train_ds = concat_with_nan_padding(
            [train_ds_hist, val_ds_hist, train_ds_hp],
            concat_dim=episode_coord,
        )

    train_ds, test_ds = normalize_domain(train_ds, test_ds, method=normalization_method)

    return train_ds, test_ds
