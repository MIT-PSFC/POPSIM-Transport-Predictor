"""Makes the 'raw' CMOD dataset on mfews, to be processed later by POPSIM"""

import os
import xarray as xr
import loguru

from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings
from disruption_py.workflow import get_shots_data
from popsim_transport_predictor.sql import summary

CMOD_RAW_DS_DIR = "/usr/local/mfe/ml_data_dump/studies/transport_predictor/cmod"

CMOD_DATASET_SIGNALS = [
    # Profiles being predicted
    "te_rho",  # Electron temperature profile [eV]
    "ne_rho",  # Electron density profile [m^-3]
    # Global quantities
    "ip",  # Plasma current
    "btor",  # On-axis magnetic field
    "wmhd",  # Total stored energy (TODO(ZanderKeith): I don't think C-Mod has a consistent fast particle measurement, so this is all we've got)
    "beta_p",  # Plasma beta
    "n_e",  # Line average electron density [m^-3]
    "a_minor",  # Plasma minor radius
    "kappa",  # Plasma elongation
    "tritop",  # Top triangularity
    "tribot",  # Bottom triangularity
    "rmagx",  # Major radius [m]
    # Power sources and sinks
    "p_oh",  # Ohmic heating power
    "p_rad",  # Bulk radiated heating power
    "p_icrf",  # ICRF heating power
    "p_lh",  # Lower hybrid heating power (yes this is actually lower hybrid on C-Mod, NOT the LH transition threshold like on TCV)
    # Other
    # TODO(ZanderKeith): Add gas valves when we get to that point
]

SUMMARY_TABLE = "summary"
IPMAX = 100e3  # [A]
PULSE_LENGTH = 0.1  # [s]
MIN_SHOT = 1050204013
MAX_SHOT = 1160930043


def make_raw_dataset(shotlist: list[int]) -> xr.Dataset:
    retrieval_settings = RetrievalSettings(
        run_columns=CMOD_DATASET_SIGNALS,
        time_setting="tmdb",
        only_requested_columns=True,
    )

    result = get_shots_data(
        tokamak=Tokamak.CMOD,
        shotlist_setting=shotlist,
        retrieval_settings=retrieval_settings,
        output_setting="dataset",
        num_processes=6,
    )

    return result


def get_shotlist_from_sql(num_shots: int = None) -> list[int]:
    data = summary(
        summary_table=SUMMARY_TABLE,
        ipmax=IPMAX,
        pulse_length=PULSE_LENGTH,
        min_shot=MIN_SHOT,
        max_shot=MAX_SHOT,
        shots=False,
    )
    shotlist = data[:, 0].astype(int).tolist()
    if num_shots is not None:
        shotlist = shotlist[:num_shots]
    return shotlist


if __name__ == "__main__":
    num_shots = None
    shotlist = get_shotlist_from_sql(num_shots)
    loguru.logger.info(f"Selected {len(shotlist)} shots out of {num_shots} requested")
    save_file = f"cmod_{len(shotlist)}_raw.nc"
    save_path = os.path.join(CMOD_RAW_DS_DIR, save_file)
    os.makedirs(CMOD_RAW_DS_DIR, exist_ok=True)
    if not os.path.exists(save_path):
        loguru.logger.info(f"Creating new dataset at {save_path}")
        ds = make_raw_dataset(shotlist)
        ds.to_netcdf(save_path)
    else:
        loguru.logger.info(f"Dataset already exists at {save_path}, loading")
        ds = xr.load_dataset(save_path)
    print(ds)
