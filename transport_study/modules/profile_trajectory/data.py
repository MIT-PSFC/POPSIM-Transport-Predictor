import jax
import numpy as np
import xarray as xr

from transport_study.trajectory_optimization import IP_RAMP_SHOTS

REQUIRED_SIGNALS = [
    "time",
    # DIII-D PCS handles these
    "iptipp_MA",
    "B0",
    "beta",
    "dstdenp",
    # Our trajectory optimization is over these variables
    "gapin",
    "gapout",
    "rxpt1",
    "zxpt1",
    "rxpt2",
    "zxpt2",
    # Target profiles for the loss function
    "ne20_psi",
    "Te_keV_psi",
    "fresh_profiles",
]


def get_ds(
    ds_path: str,
    fresh_profiles: bool = False,
    debug: bool | None = False,
) -> tuple[xr.Dataset, str]:
    """Load the dataset, and do some light processing to get it ready for training.

    Args:
        ds_path (str): Path to the dataset.
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

    if debug:
        ds = ds.sel(
            shot=list(IP_RAMP_SHOTS.keys())
        )  # Limit to specifically these Ip ramp shots

    # If signals are in terms of rho replace them with psi
    # TODO(ZanderKeith) this is sloppy dataset creation on my end, should really standardize this naming scheme
    # Double check if the TCV dataset is in terms of rho or psi and handle accordingly
    if "rho" in ds.coords and "psi" not in ds.coords:
        ds = ds.rename({"rho": "psi"})
        for signal in ["ne20_rho", "Te_keV_rho"]:
            if signal in ds:
                ds = ds.rename({signal: signal.replace("rho", "psi")})

    if fresh_profiles:
        ds = ds.where(ds["fresh_profiles"] == 1, drop=True)

    # Limit to required signals
    ds = ds[REQUIRED_SIGNALS]

    # Put dataset on an even psi grid [0, 1.2]
    psigrid = np.linspace(0, 1.2, 61)
    ds = ds.interp(psi=psigrid, kwargs={"fill_value": "extrapolate"})

    # Calculate shape variables
    ds["ne_shape"] = ds["ne20_psi"] / ds["ne20_psi"].integrate("psi")
    ds["Te_shape"] = ds["Te_keV_psi"] / ds["Te_keV_psi"].integrate("psi")

    # Add a data variable for the trajectory time
    ds["traj_time"] = ds["time"].copy()

    # If dataset was from a zarr store, must promote the 'time' data var to a coordinate
    if "time" not in ds.coords:
        ds = ds.set_coords("time")

    # Dataset retains all signals, the dataloader will filter out the ones that are not needed.
    return ds, "shot"
