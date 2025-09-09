"""Makes the 'raw' CMOD dataset on mfews, to be processed later by POPSIM"""
import os
import xarray as xr

from disruption_py.machine.tokamak import Tokamak, resolve_tokamak_from_environment
from disruption_py.settings import RetrievalSettings
from disruption_py.workflow import get_shots_data


def make_raw_dataset(shotlist: list[int]) -> xr.Dataset:
    run_methods = ["efit"]
    retrieval_settings = RetrievalSettings(
        run_methods=run_methods,
        efit_nickname_setting="default",
        time_setting="tmdb"
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
    save_file = f"{shot_number}_sxx.nc"
    if not os.path.exists(save_file):
        ds = make_raw_dataset([shot_number])
        ds.to_netcdf(save_file)
    else:
        ds = xr.open_dataset(save_file)

    print(f"SXX data for shot {shot_number} has been saved.")