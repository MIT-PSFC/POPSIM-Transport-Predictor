"""Makes the 'raw' CMOD dataset on mfews, to be processed later by POPSIM"""

import os
import xarray as xr
import loguru
import fire

from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings
from disruption_py.workflow import get_shots_data

from popsim_transport_predictor.datasets_dispy.dispy_utils import summary
from popsim_transport_predictor.datasets_dispy.cmod import (
    SUMMARY_TABLE,
    IPMAX,
    PULSE_LENGTH,
    MIN_SHOT,
    MAX_SHOT,
    BLESSED_THOMSON_DAYS,
)


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

    # Filter to blessed Thomson days
    shotlist = [shot for shot in shotlist if int(shot / 1000) in BLESSED_THOMSON_DAYS]

    if num_shots is not None:
        shotlist = shotlist[:num_shots]
    return shotlist


def get_thomson_dataset(shotlist: list[int]) -> xr.Dataset:
    retrieval_settings = RetrievalSettings(
        run_methods=["get_thomson_channels"],
        only_requested_columns=False,
    )
    result = get_shots_data(
        tokamak=Tokamak.CMOD,
        shotlist_setting=shotlist,
        retrieval_settings=retrieval_settings,
        output_setting="dataset",
        num_processes=6,
    )
    return result


def make_profile_dataset(ds_thomson: xr.Dataset) -> xr.Dataset:
    """
    Perform fitting with GPtools
    """
    return xr.Dataset()  # TODO(ZanderKeith)


def get_efit_dataset(shotlist: list[int]) -> xr.Dataset:
    retrieval_settings = RetrievalSettings(
        run_columns=["ip"],
        time_setting="efit",
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


def make_final_dataset(
    ds_profiles: xr.Dataset,
    ds_efit: xr.Dataset,
) -> xr.Dataset:
    """
    Combine profile and EFIT datasets onto a common timebase
    """
    ds_final = ds_profiles.set_index(idx=["shot", "time"]).unstack("idx")
    return ds_final  # TODO(ZanderKeith)


def make_cmod_dataset(save_path: str, num_shots: int, clean: bool = False):
    """
    Makee the source CMOD dataset for POPSIM transport predictor study.

    Workflow is as follows:
    1) Get shotlist from SQL summary table
    2) Filter to shots that have blessed Thomson scattering data
    3) Retrieve the raw TS data on its native timebase
    4) Filter to shots that have both core and edge TS data
    5) Retrieve EFIT and other 1D signals on the EFIT timebase
    6) Put TS data and EFIT data together on a uniform 1kHz timebase
    """

    os.makedirs(save_path, exist_ok=True)
    ds_final_path = os.path.join(save_path, "cmod_source.nc")
    ds_thomson_path = os.path.join(save_path, "cmod_thomson_raw.nc")
    ds_profile_path = os.path.join(save_path, "cmod_profiles_raw.nc")
    ds_efit_path = os.path.join(save_path, "cmod_efit_raw.nc")

    if clean:
        for path in [
            ds_final_path,
            ds_thomson_path,
            ds_profile_path,
            ds_efit_path,
        ]:
            if os.path.exists(path):
                os.remove(path)
                loguru.logger.info(f"Removed existing file {path}")

    if os.path.exists(ds_final_path):
        loguru.logger.info(
            f"Final dataset already exists at {ds_final_path}, skipping creation"
        )
        return

    shotlist = get_shotlist_from_sql(num_shots=num_shots)
    loguru.logger.info(f"Retrieved shotlist of {len(shotlist)} shots from SQL")

    if not os.path.exists(ds_thomson_path):
        ds_thomson = get_thomson_dataset(shotlist)
        ds_thomson.to_netcdf(ds_thomson_path)
        loguru.logger.info(f"Saved raw Thomson dataset to {ds_thomson_path}")
    else:
        ds_thomson = xr.load_dataset(ds_thomson_path)
        loguru.logger.info(f"Loaded existing Thomson dataset from {ds_thomson_path}")

    if not os.path.exists(ds_profile_path):
        ds_profiles = make_profile_dataset(ds_thomson)
        ds_profiles.to_netcdf(ds_profile_path)
        loguru.logger.info(f"Saved raw profile dataset to {ds_profile_path}")
    else:
        ds_profiles = xr.load_dataset(ds_profile_path)
        loguru.logger.info(f"Loaded existing profile dataset from {ds_profile_path}")

    if not os.path.exists(ds_efit_path):
        ds_efit = get_efit_dataset(shotlist)
        ds_efit.to_netcdf(ds_efit_path)
        loguru.logger.info(f"Saved raw EFIT dataset to {ds_efit_path}")
    else:
        ds_efit = xr.load_dataset(ds_efit_path)
        loguru.logger.info(f"Loaded existing EFIT dataset from {ds_efit_path}")


if __name__ == "__main__":
    fire.Fire(make_cmod_dataset)
