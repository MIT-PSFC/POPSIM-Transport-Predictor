"""Makes the 'raw' CMOD dataset on mfews, to be processed later by POPSIM"""
import os
import xarray as xr

from disruption_py.machine.tokamak import Tokamak, resolve_tokamak_from_environment
from disruption_py.settings import RetrievalSettings
from disruption_py.workflow import get_shots_data


CMOD_DATASET_SIGNALS = [
    # Profiles being predicted
    # "Te_rho",  # Electron temperature profile [eV]
    # "Ne_rho",  # Electron density profile [m^-3]
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
    # "p_oh",  # Ohmic heating power
    # "p_rad",  # Bulk radiated heating power
    # "p_icrf", # ICRF heating power
    # "p_lh",  # Lower hybrid heating power
    # Other
]

def make_raw_dataset(shotlist: list[int]) -> xr.Dataset:
    run_columns = CMOD_DATASET_SIGNALS
    retrieval_settings = RetrievalSettings(
        run_columns=run_columns,
        efit_nickname_setting="default",
        time_setting="tmdb",
        only_requested_columns=True,
    )

    result = get_shots_data(
        tokamak=Tokamak.CMOD,
        shotlist_setting=shotlist,
        retrieval_settings=retrieval_settings,
        output_setting="dataset",
    )

    return result

if __name__ == "__main__":
    # Example usage
    shot_number = 1110316031
    save_file = f"{shot_number}.nc"
    ds = make_raw_dataset([shot_number])
    ds.to_netcdf(save_file)