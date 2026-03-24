import jax
import numpy as np
import xarray as xr

from transport_study.trajectory_optimization.setup_data import (
    FEEDBACK_CONTROL_SHOTS,
)

REQUIRED_SIGNALS = [
    "time",
    # Programmed into DIII-D PCS, unchanged
    "Ip_MA_prog",
    "B0_prog",
    "betan_prog",
    # Things our controller may replace
    "ne20_edge_prog",
    "R0_prog",
    "rxbot_prog",
    "zxbot_prog",
    "rxtop_prog",
    "zxtop_prog",
    # Measured equivalents for comparison
    "Ip_MA",
    "B0",
    "betan",
    "ne20_edge",
    "R0",
    "rxbot",
    "zxbot",
    "rxtop",
    "zxtop",
    "gapin",  # <- special handling
    "a_minor",
    "kappa",
    "delta_top",
    "delta_bot",
    # Target profiles for the loss function
    "ne20_psi",
    "Te_keV_psi",
    "fresh_profiles",
]


def correct_B0_prog(ds: xr.Dataset) -> xr.Dataset:
    """
    Programmed B0 is extremly wonk on DIII-D

    Either the signal goes to 0 and it follows an L/R curve,
    or another part of the PCS completely ignores the programmed signal and does something else.

    I'm gonna say that we can know what the B0 will be in advance if we want to, minus noise.
    As such, overwrite the B0_prog with a smoothed version of the actual B0
    This means we avoid having to deal with weird edge cases in the dataset where the programmed B0 is completely wrong,
    and we can still capture the typical noise in the B0 signal that the predictor will have to deal with
    """
    smoothed_B0 = ds["B0"].rolling(time_idx=100, center=False, min_periods=1).mean()
    ds = ds.assign(B0_prog=smoothed_B0)
    return ds


def add_gapin_prog(ds: xr.Dataset) -> xr.Dataset:
    """
    Similar to B0_prog, there isn't really a 'programmed inner gap' signal

    I know it should probably be changing slowly, so do a similar thing to B0_prog where
    we make the programmed gapin a smoothed version of the actual gapin,
    so that we can still capture typical noise in the gapin signal that the predictor will have to deal with
    """

    smoothed_gapin = (
        ds["gapin"].rolling(time_idx=100, center=False, min_periods=1).mean(skipna=True)
    )
    # Fill leading/trailing NaNs (e.g. before EFIT is valid) by propagating nearest valid value
    smoothed_gapin = smoothed_gapin.bfill("time_idx").ffill("time_idx")
    ds = ds.assign(gapin_prog=smoothed_gapin)
    return ds


def get_ds(
    ds_path: str,
    selected_shots: list[dict[str, float]] = FEEDBACK_CONTROL_SHOTS,
    fresh_profiles: bool = False,
    debug: bool | None = False,
) -> tuple[xr.Dataset, str]:
    """Load the dataset, and do some light processing to get it ready for training.

    Args:
        ds_path (str): Path to the dataset.
        selected_shots (list[dict[str, float]], optional): A list of dictionaries specifying the shots and time windows to include. Defaults to FEEDBACK_CONTROL_SHOTS.
        fresh_profiles (bool, optional): Whether to filter the dataset to only include time steps where we have fresh profile measurements. Defaults to False.
        debug (bool, optional): Whether to enable debug mode, reducing dataset size to at most 50 shots.

    Returns:
        tuple[xr.Dataset, str]: The processed dataset and the dimension along which to group the data
    """

    # Load dataset according to JAX setting
    if jax.config.jax_enable_x64:
        ds = xr.open_dataset(ds_path).astype(jax.numpy.float64)
    else:
        ds = xr.open_dataset(ds_path).astype(jax.numpy.float32)

    ds = ds.sel(
        shot=list(selected_shots.keys())
    )  # Limit to specifically these feedback control shots
    # Also limit to within the time window of interest
    max_time = max(times["end"] for times in selected_shots.values())
    min_time = min(times["start"] for times in selected_shots.values())
    ds = ds.where((ds["time"] >= min_time) & (ds["time"] <= max_time), drop=True)

    if fresh_profiles:
        ds = ds.where(ds["fresh_profiles"] == 1, drop=True)
    elif debug:
        # Resample at lower time resolution to speed up training (only every 100 ms)
        ds = ds.sel(time_idx=ds["time_idx"].values[::100])

    # Limit to required signals
    ds = ds[REQUIRED_SIGNALS]

    # Put dataset on an even psi grid [0, 1]
    psi_n_grid = np.linspace(0, 1.0, 51)
    ds = ds.interp(psi_n=psi_n_grid, kwargs={"fill_value": "extrapolate"})

    # Compute means and shapes.
    ds["Te_keV_line_avg"] = ds["Te_keV_psi"].integrate("psi_n")
    ds["ne20_line_avg"] = ds["ne20_psi"].integrate("psi_n")
    ds["Te_shape"] = ds["Te_keV_psi"] / ds["Te_keV_line_avg"]
    ds["ne_shape"] = ds["ne20_psi"] / ds["ne20_line_avg"]

    ds = correct_B0_prog(ds)
    ds = add_gapin_prog(ds)

    # Add a data variable for the trajectory time
    ds["traj_time"] = ds["time"].copy()

    # If dataset was from a zarr store, must promote the 'time' data var to a coordinate
    if "time" not in ds.coords:
        ds = ds.set_coords("time")

    # Dataset retains all signals, the dataloader will filter out the ones that are not needed.
    return ds, "shot"
