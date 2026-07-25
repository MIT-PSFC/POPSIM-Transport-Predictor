"""A synthetic raw per-shot dataset the dataset tests build their cases from.

The processing chain (DataWorkflow.process_fn) runs on real device data in
production, so the tests feed it a hand-built raw file instead. This one passes
every C-Mod and MAST filter and cull, and each test breaks exactly the one thing
it is about.
"""

import numpy as np
import xarray as xr

from transport_study.datasets import UNIFORM_TIMEBASE_DT_S

SHOT = 1160503010
N_T = 1500  # 1.5 s at 1 kHz, comfortably over min_shot_duration
RHO = np.linspace(0.0, 1.1, 12)
# Thomson measures far slower than the 1 kHz grid, so each fit is forward-filled
# over a block of timeslices (see the workflows' assemble_shot)
TS_BLOCK = 20
N_BLOCKS = N_T // TS_BLOCK


def raw_shot(n_t: int = N_T, shot: int = SHOT, **overrides) -> xr.Dataset:
    """A synthetic raw shot that passes every filter and cull.

    Profiles change once per TS_BLOCK slices and are constant within a block,
    matching the forward-filled layout of a real raw file. Overrides replace a
    variable's values in place, so they carry the (shot, time_idx[, rho]) shape.
    """
    time = np.arange(n_t) * UNIFORM_TIMEBASE_DT_S
    n_blocks = int(np.ceil(n_t / TS_BLOCK))
    amplitude = 2.0 + 0.01 * np.repeat(np.arange(n_blocks), TS_BLOCK)[:n_t, None]
    profile = amplitude * (1 - np.tanh((RHO - 0.9) / 0.08))[None, :] / 2 + 0.05
    scalars = {
        "Wtot_MJ": np.full(n_t, 0.05),
        "P_oh_MW": np.full(n_t, 1.0),
        "P_rad_MW": np.full(n_t, 0.3),
        "P_ICRF_MW": np.zeros(n_t),
        "P_LH_MW": np.zeros(n_t),
        "P_NBI_MW": np.zeros(n_t),
        "P_ECRH_MW": np.zeros(n_t),
        "Ip_MA": np.full(n_t, 0.8),
        "B0": np.full(n_t, 5.0),
        "betan": np.full(n_t, 0.8),
        "ne20_line_avg": np.full(n_t, 1.0),
        "R0": np.full(n_t, 0.68),
        "kappa": np.full(n_t, 1.6),
        "a_minor": np.full(n_t, 0.22),
        "delta_top": np.full(n_t, 0.3),
        "delta_bot": np.full(n_t, 0.5),
        "beta_p": np.full(n_t, 0.5),
        "ne20_edge": np.full(n_t, 0.5),
    }
    profiles = {
        "Te_keV_rho": profile,
        "Te_keV_rho_error": 0.1 * profile,
        "Te_keV_rho_grad": np.gradient(profile, RHO, axis=1),
        "Te_keV_rho_grad_error": np.full_like(profile, 0.5),
        "ne20_rho": profile / 2,
        "ne20_rho_error": 0.05 * profile,
        "ne20_rho_grad": np.gradient(profile / 2, RHO, axis=1),
        "ne20_rho_grad_error": np.full_like(profile, 0.2),
    }
    data_vars = {name: (("time_idx",), vals) for name, vals in scalars.items()}
    data_vars |= {name: (("time_idx", "rho"), vals) for name, vals in profiles.items()}
    ds = xr.Dataset(data_vars, coords={"time_idx": np.arange(n_t), "time": ("time_idx", time), "rho": RHO})
    ds = ds.expand_dims(shot=[shot])
    for name, vals in overrides.items():
        ds[name] = vals if isinstance(vals, xr.DataArray) else (ds[name].dims, vals)
    return ds
