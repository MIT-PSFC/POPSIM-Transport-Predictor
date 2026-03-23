import jax
import numpy as np
import xarray as xr

from transport_study.trajectory_optimization import FEEDBACK_CONTROL_SHOTS

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

    # Bridge low values in B0_prog with L/R interpolation.
    # B0_prog can be small during field program switches even though the actual field (with L/R time constant)
    # remains ~1.5 T. Treat low values as missing and interpolate between surrounding valid values.
    B0_prog = ds["B0_prog"].where(ds["B0_prog"] > 1)
    B0_prog = B0_prog.interpolate_na(dim="time_idx")
    B0_prog = B0_prog.ffill(dim="time_idx").bfill(dim="time_idx")
    ds["B0_prog"] = B0_prog

    # Add a data variable for the trajectory time
    ds["traj_time"] = ds["time"].copy()

    # If dataset was from a zarr store, must promote the 'time' data var to a coordinate
    if "time" not in ds.coords:
        ds = ds.set_coords("time")

    # Dataset retains all signals, the dataloader will filter out the ones that are not needed.
    return ds, "shot"
