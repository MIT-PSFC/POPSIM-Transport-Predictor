import numpy as np
import xarray as xr
from loguru import logger
from popsim.ml.split_utils import split_dataset_by_fracs

from popsim_transport_predictor.transfer_learning.config import config
from popsim_transport_predictor.transfer_learning.orchestration import (
    HP_SHOTS_INCLUDED,
    TRAIN_VAL_TEST_SPLIT,
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


def get_ds(
    source_ds: str,
    debug: bool | None = False,
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
    We are saying performance is 75th percentile of (Wtot_MJ^2 + Ip_MA^2)**0.5 along a shot
    Ignoring nans in the calculation

    Also stores the specific Ip_MA and Wtot_MJ values at the time point where the
    performance metric reaches its 75th percentile for plotting in parameter space
    """
    # TODO(ZanderKeith): I want the performance metric to be 'square-ish' in both Ip and Wtot
    # so that when we do performance extrapolation plots the contours are not too skewed by one being bigger
    # But that means we're gonna need to look at the entire dataset from all devices
    # For now we can do them individually

    max_Wtot = ds["Wtot_MJ"].max().item()
    max_Ip = ds["Ip_MA"].max().item()
    Wtot_scale = 1.0 / max_Wtot if max_Wtot != 0 else 1.0
    Ip_scale = 1.0 / max_Ip if max_Ip != 0 else 1.0

    # Calculate performance at each time step (once for all shots)
    perf_timeseries = ds.eval(
        f"(({Wtot_scale} * Wtot_MJ)**2 + ({Ip_scale} * Ip_MA)**2)**0.5"
    )

    # Get the 75th percentile value per shot
    if "time_idx" in ds.dims:
        ds["performance"] = perf_timeseries.quantile(0.75, dim="time_idx", skipna=True)
    else:
        ds["performance"] = perf_timeseries.quantile(0.75, dim="time", skipna=True)

    n_shots = ds.sizes[episode_coord]

    # Initialize arrays for Ip_MA and Wtot_MJ at p75
    ip_ma_p75 = np.full(n_shots, np.nan)
    wtot_mj_p75 = np.full(n_shots, np.nan)

    # For each shot, find the time index closest to 75th percentile
    perf_ts_data = perf_timeseries.values  # shape: (n_shots, n_time)
    p75_vals = ds["performance"].values  # shape: (n_shots,)
    ip_ma_data = ds["Ip_MA"].values
    wtot_mj_data = ds["Wtot_MJ"].values

    for i in range(n_shots):
        # Get performance timeseries for this shot
        perf_shot = perf_ts_data[i]
        p75_val = p75_vals[i]

        # Find valid (non-NaN) indices
        valid_mask = ~np.isnan(perf_shot)

        if valid_mask.sum() > 0 and not np.isnan(p75_val):
            # Find index where performance is closest to p75
            abs_diff = np.abs(perf_shot - p75_val)
            abs_diff[~valid_mask] = np.inf  # Ignore NaN positions
            idx_p75 = np.argmin(abs_diff)

            # Extract Ip_MA and Wtot_MJ at that time
            ip_ma_p75[i] = ip_ma_data[i, idx_p75]
            wtot_mj_p75[i] = wtot_mj_data[i, idx_p75]

    # Add to dataset
    ds["Ip_MA_p75"] = (episode_coord, ip_ma_p75)
    ds["Wtot_MJ_p75"] = (episode_coord, wtot_mj_p75)

    return ds


def get_train_val_test_datasets(
    training_data_case: str,
):
    """
    Split dataset into training, validation, and test sets based on the specified case.
    """

    if training_data_case in ["cmod", "tcv", "d3d_lp"]:
        ds, episode_coord = get_ds(training_data_case)
        ds = add_performance(ds, episode_coord)
        train_ds, val_ds, test_ds = split_dataset_by_fracs(
            ds,
            fracs=TRAIN_VAL_TEST_SPLIT,
            dim=episode_coord,
            seed=42,
            sortby="performance",
        )
        train_ds = train_ds.assign_coords(ds_source=training_data_case)
        val_ds = val_ds.assign_coords(ds_source=training_data_case)
        test_ds = test_ds.assign_coords(ds_source=training_data_case)

    else:
        ds_cmod, episode_coord = get_ds("cmod")
        ds_cmod = add_performance(ds_cmod, episode_coord)
        train_ds_cmod, val_ds_cmod, test_ds_cmod = split_dataset_by_fracs(
            ds_cmod,
            fracs=TRAIN_VAL_TEST_SPLIT,
            dim=episode_coord,
            seed=42,
            sortby="performance",
        )
        train_ds_cmod = train_ds_cmod.assign_coords(ds_source="cmod")
        val_ds_cmod = val_ds_cmod.assign_coords(ds_source="cmod")
        test_ds_cmod = test_ds_cmod.assign_coords(ds_source="cmod")

        ds_tcv, episode_coord = get_ds("tcv")
        ds_tcv = add_performance(ds_tcv, episode_coord)
        train_ds_tcv, val_ds_tcv, test_ds_tcv = split_dataset_by_fracs(
            ds_tcv,
            fracs=TRAIN_VAL_TEST_SPLIT,
            dim=episode_coord,
            seed=42,
            sortby="performance",
        )
        train_ds_tcv = train_ds_tcv.assign_coords(ds_source="tcv")
        val_ds_tcv = val_ds_tcv.assign_coords(ds_source="tcv")
        test_ds_tcv = test_ds_tcv.assign_coords(ds_source="tcv")

        if training_data_case == "cmod_tcv":
            train_ds = xr.concat([train_ds_cmod, train_ds_tcv], dim=episode_coord)
            val_ds = xr.concat([val_ds_cmod, val_ds_tcv], dim=episode_coord)
            test_ds = xr.concat([test_ds_cmod, test_ds_tcv], dim=episode_coord)

        elif training_data_case == "cmod_tcv_d3d_lp":
            ds_d3d_lp, episode_coord = get_ds("d3d_lp")
            ds_d3d_lp = add_performance(ds_d3d_lp, episode_coord)
            train_ds_d3d_lp, val_ds_d3d_lp, test_ds_d3d_lp = split_dataset_by_fracs(
                ds_d3d_lp,
                fracs=TRAIN_VAL_TEST_SPLIT,
                dim=episode_coord,
                seed=42,
                sortby="performance",
            )
            train_ds_d3d_lp = train_ds_d3d_lp.assign_coords(ds_source="d3d_lp")
            val_ds_d3d_lp = val_ds_d3d_lp.assign_coords(ds_source="d3d_lp")
            test_ds_d3d_lp = test_ds_d3d_lp.assign_coords(ds_source="d3d_lp")

            train_ds = xr.concat(
                [train_ds_cmod, train_ds_tcv, train_ds_d3d_lp], dim=episode_coord
            )
            val_ds = xr.concat(
                [val_ds_cmod, val_ds_tcv, val_ds_d3d_lp], dim=episode_coord
            )
            test_ds = xr.concat(
                [test_ds_cmod, test_ds_tcv, test_ds_d3d_lp], dim=episode_coord
            )

        else:
            raise ValueError(f"Unknown training data case: {training_data_case}")

    logger.debug("Training dataset size: {}", train_ds.sizes[episode_coord])
    logger.debug("Validation dataset size: {}", val_ds.sizes[episode_coord])
    logger.debug("Test dataset size: {}", test_ds.sizes[episode_coord])

    return train_ds, val_ds, test_ds


def get_train_test_datasets_transfer(
    training_data_case: str,
    num_hp_shots: int,
):
    """
    Split dataset into training, validation, and test sets for transfer learning case.
    The number of high-performance shots included in training is specified by `num_hp_shots`.
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
        train_ds_hist, val_ds_hist, test_ds_hist = get_train_val_test_datasets(
            training_data_case
        )
        train_ds = xr.concat(
            [train_ds_hist, val_ds_hist, test_ds_hist, train_ds_hp], dim=episode_coord
        )

    return train_ds, test_ds
