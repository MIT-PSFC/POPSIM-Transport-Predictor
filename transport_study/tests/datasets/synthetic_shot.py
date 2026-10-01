"""A synthetic raw per-shot dataset the dataset tests build their cases from.

The processing chain (RawFileWorkflow.load_shot and cull_shot) runs on real device data in
production, so the tests feed it a hand-built raw file instead. This one is a TCV raw
file in the on-disk schema (IMAS names, SI units, no profile companions), it passes
every TCV filter and cull, and each test breaks exactly the one thing it is about.
"""

import numpy as np
import xarray as xr

from transport_study.datasets import UNIFORM_TIMEBASE_DT_S

SHOT = 70000
N_T = 1500  # 1.5 s at 1 kHz, comfortably over min_shot_duration
RHO = np.linspace(0.0, 1.1, 12)
# Thomson measures far slower than the 1 kHz grid, so each profile is forward-filled
# over a block of timeslices
TS_BLOCK = 20
N_BLOCKS = N_T // TS_BLOCK


def raw_shot(n_t: int = N_T, shot: int = SHOT, **overrides) -> xr.Dataset:
    """A synthetic TCV raw shot that passes every filter and cull.

    Profiles change once per TS_BLOCK slices and are constant within a block,
    matching the forward-filled layout of a real raw file. Overrides replace a
    variable's values in place, so they carry the (shot, time_idx[, rho_tor_norm]) shape.
    """
    time = np.arange(n_t) * UNIFORM_TIMEBASE_DT_S
    n_blocks = int(np.ceil(n_t / TS_BLOCK))
    amplitude = 2.0 + 0.01 * np.repeat(np.arange(n_blocks), TS_BLOCK)[:n_t, None]
    profile_shape = amplitude * (1 - np.tanh((RHO - 0.9) / 0.08))[None, :] / 2 + 0.05
    scalars = {
        "ip": np.full(n_t, 3e5),
        "b0": np.full(n_t, 1.4),
        "energy_mhd": np.full(n_t, 3e4),
        "beta_tor_norm": np.full(n_t, 1.5),
        "n_e_line_average": np.full(n_t, 3e19),
        "minor_radius": np.full(n_t, 0.24),
        "geometric_axis_r": np.full(n_t, 0.88),
        "elongation": np.full(n_t, 1.5),
        "triangularity_upper": np.full(n_t, 0.3),
        "triangularity_lower": np.full(n_t, 0.2),
        "power_ohm": np.full(n_t, 3e5),
        "power_radiated": np.full(n_t, 1e5),
        "power_nbi": np.zeros(n_t),
        "power_ic": np.zeros(n_t),
        "power_lh": np.zeros(n_t),
        "power_ec": np.zeros(n_t),
        "n_e_edge_line_average": np.full(n_t, 2e19),
    }
    profiles = {
        "t_e": 1e3 * profile_shape,
        "n_e": 2e19 * profile_shape,
    }
    data_vars = {name: (("time_idx",), vals) for name, vals in scalars.items()}
    data_vars |= {name: (("time_idx", "rho_tor_norm"), vals) for name, vals in profiles.items()}
    ds = xr.Dataset(data_vars, coords={"time_idx": np.arange(n_t), "time": ("time_idx", time), "rho_tor_norm": RHO})
    ds = ds.expand_dims(shot=[shot])
    for name, vals in overrides.items():
        ds[name] = vals if isinstance(vals, xr.DataArray) else (ds[name].dims, vals)
    return ds
