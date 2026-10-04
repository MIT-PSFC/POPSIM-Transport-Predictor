"""A tiny store in the transport-validation-datasets schema, for the tests that build or load one.

Every shot is one contiguous 1 kHz segment padded with NaN to N_TIME_PADDED,
ip and b0 carry a source sign on SIGNED_SHOT, and the per-shot amplitudes grow with the shot index
so the hazard metric orders the shots the way they are listed.
"""

from pathlib import Path

import numpy as np
import xarray as xr
from transport_validation_datasets.store_schema import STORE_SIGNAL_ATTRS

RHO = np.linspace(0.0, 1.1, 12)
N_TIME_PADDED = 60
SHOT_LENGTHS = {1160503001: 50, 1160503002: 30, 1160503003: 40}
SIGNED_SHOT = 1160503002
T_START = 0.1
DT = 1e-3
# Store amplitudes of the signals every shot shares
IP_A = 8e5
B0_T = 5.4
ENERGY_MHD_J = 6e4
R0_M = 0.66
MINOR_RADIUS_M = 0.22
GEOMETRIC_AXIS_R_M = 0.68
ELONGATION = 1.6
HEATING_POWERS_W = {"power_nbi": 0.0, "power_ic": 2e6, "power_lh": 0.0, "power_ec": 0.0}


def write_synthetic_store(path: Path, fit_mode: str = "sample", internal: bool = True) -> xr.Dataset:
    """Write the store to path and return it.

    internal adds what an internal store carries beyond the shared schema
    (raw Thomson channels, the equilibrium, fit statuses), which a build must leave out.
    """
    shots = np.array(list(SHOT_LENGTHS))
    n_shots = shots.size
    mask_valid = np.zeros((n_shots, N_TIME_PADDED), dtype=bool)
    for i_shot, n_valid in enumerate(SHOT_LENGTHS.values()):
        mask_valid[i_shot, :n_valid] = True

    def padded(values_valid: np.ndarray) -> np.ndarray:
        """NaN where the shot has ended, broadcast over any trailing dims."""
        values = np.broadcast_to(values_valid, (n_shots, N_TIME_PADDED, *values_valid.shape[2:])).astype(np.float32)
        mask = mask_valid.reshape(n_shots, N_TIME_PADDED, *([1] * (values.ndim - 2)))
        return np.where(mask, values, np.nan).astype(np.float32)

    time = padded(np.arange(N_TIME_PADDED)[None, :] * DT + T_START)
    sign = np.where(shots == SIGNED_SHOT, -1.0, 1.0)[:, None]
    # Amplitudes growing with the shot index, so the hazard sort is known
    growth = (1.0 + 0.2 * np.arange(n_shots))[:, None]
    scalars = {
        "ip": sign * IP_A * growth,
        "b0": sign * B0_T,
        "energy_mhd": ENERGY_MHD_J * growth,
        "beta_tor_norm": 1.0,
        "n_e_line_average": 1e20,
        "minor_radius": MINOR_RADIUS_M,
        "geometric_axis_r": GEOMETRIC_AXIS_R_M,
        "elongation": ELONGATION,
        "triangularity_upper": 0.4,
        "triangularity_lower": 0.5,
        "power_ohm": 1e6,
        "power_radiated": 4e5,
        **HEATING_POWERS_W,
        "fresh_profile": 1.0,
        "fresh_equilibrium": 1.0,
    }
    if internal:
        scalars["t_e_fit_status"] = 0.0
    profile_shape = (1 - RHO**2)[None, None, :] + 0.05
    profiles = {
        "t_e": 3e3 * profile_shape,
        "t_e_error": 1e2 * profile_shape,
        "t_e_gradient": -6e3 * RHO[None, None, :],
        "t_e_gradient_error": 2e2 * profile_shape,
        "n_e": 1.5e20 * profile_shape,
        "n_e_error": 5e18 * profile_shape,
        "n_e_gradient": -3e20 * RHO[None, None, :],
        "n_e_gradient_error": 1e19 * profile_shape,
    }
    data_vars = {name: (("shot", "time_idx"), padded(np.ones((n_shots, N_TIME_PADDED)) * value)) for name, value in scalars.items()}
    data_vars |= {name: (("shot", "time_idx", "rho_tor_norm"), padded(values)) for name, values in profiles.items()}
    data_vars["cocos"] = (("shot",), np.full(n_shots, 7.0, dtype=np.float32))
    data_vars["r0"] = (("shot",), np.full(n_shots, R0_M, dtype=np.float32))
    coords = {"shot": shots, "rho_tor_norm": RHO, "time": (("shot", "time_idx"), time)}
    if internal:
        data_vars["psirz"] = (("shot", "time_idx", "r_grid", "z_grid"), padded(np.ones((n_shots, N_TIME_PADDED, 3, 3))))
        data_vars["ts_channel_t_e"] = (("shot", "time_idx", "ts_channel"), padded(np.full((n_shots, N_TIME_PADDED, 4), 1e3)))
        coords |= {"r_grid": np.arange(3.0), "z_grid": np.arange(3.0), "ts_channel": np.arange(4)}
    ds = xr.Dataset(data_vars, coords=coords, attrs={"fit_mode": fit_mode})
    for name, attrs in STORE_SIGNAL_ATTRS.items():
        if name in ds:
            ds[name].attrs["units"] = attrs["units"]
    ds.to_zarr(path, mode="w")
    return ds
