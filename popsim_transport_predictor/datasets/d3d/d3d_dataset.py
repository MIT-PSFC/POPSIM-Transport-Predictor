"""Makes the 'raw' DIII-D dataset on omega, to be processed later by POPSIM"""

import os

import fire
from loguru import logger
import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings
from disruption_py.workflow import get_shots_data

from popsim_transport_predictor.datasets import make_uniform_1khz_timebase
from popsim_transport_predictor.datasets.d3d import (
    D3D_DATASET_SIGNALS,
    HP_SHOTLIST,
    IPMAX,
    MAX_SHOT,
    MIN_SHOT,
    PULSE_LENGTH,
    SUMMARY_TABLE,
)
from popsim_transport_predictor.datasets.dispy_utils import summary

def get_shotlist_from_sql(num_shots: int | None, hp_shots: bool) -> list[int]:
    data = summary(
        summary_table=SUMMARY_TABLE,
        ipmax=IPMAX,
        pulse_length=PULSE_LENGTH,
        min_shot=MIN_SHOT,
        max_shot=MAX_SHOT,
        shots=False,
    )
    shotlist = data[:, 0].astype(int).tolist()

    if hp_shots:
        shotlist = sorted(set(shotlist).intersection(set(HP_SHOTLIST)))
    else:
        shotlist = sorted(set(shotlist).difference(set(HP_SHOTLIST)))

    if num_shots is not None:
        shotlist = shotlist[-num_shots:]

    return shotlist

def get_efit_dataset(shot: int) -> xr.Dataset:
    retrieval_settings = RetrievalSettings(
        run_columns=D3D_DATASET_SIGNALS,
        time_setting="efit",
        only_requested_columns=True,
    )
    result = get_shots_data(
        tokamak=Tokamak.D3D,
        shotlist_setting=shot,
        retrieval_settings=retrieval_settings,
        num_processes=1,
    )
    result = result.set_index(idx=["shot", "time"]).unstack("idx")
    return result

def make_final_dataset(
    shotlist,
    ds_thomson_dir,
    ds_profile_dir,
    ds_efit_dir,
    ds_assembly_dir,
) -> xr.Dataset:
    """
    Combine datasets together and align to uniform 1 kHz timebase
    """

    for shot in shotlist:
        ds_thomson_path = os.path.join(ds_thomson_dir, f"{shot}.nc")
        ds_profiles_path = os.path.join(ds_profile_dir, f"{shot}.nc")
        ds_efit_path = os.path.join(ds_efit_dir, f"{shot}.nc")
        ds_assembly_path = os.path.join(ds_assembly_dir, f"{shot}.nc")

        ds_thomson = xr.load_dataset(ds_thomson_path)
        ds_profiles = xr.load_dataset(ds_profiles_path)
        ds_efit = xr.load_dataset(ds_efit_path)

        # Put each dataset on a 1 kHz timebase, using previous value fill
        max_time = max(
            ds_thomson["time"].max().item(),
            ds_profiles["time"].max().item(),
            ds_efit["time"].max().item(),
        )
        timebase = make_uniform_1khz_timebase(max_time)

        ds_thomson = ds_thomson.reindex(time=timebase, method="ffill")
        ds_profiles = ds_profiles.reindex(time=timebase, method="ffill")
        ds_efit = ds_efit.interp(
            time=timebase, method="nearest"
        )  # This should be okay since EFIT is already at high time resolution
        ds_assembly = xr.merge([ds_thomson, ds_profiles, ds_efit], compat="override")

        ds_assembly.to_netcdf(ds_assembly_path)
        logger.info(f"Saved assembled dataset to {ds_assembly_path}")

    # Now put all shots together
    ds_final = xr.concat(
        [
            xr.load_dataset(os.path.join(ds_assembly_dir, f"{shot}.nc"))
            for shot in shotlist
        ],
        dim=xr.IndexVariable("shot", shotlist),
    )
    return ds_final

def make_d3d_dataset(
    save_dir: str,
    num_shots: int,
    clean: bool = False,
    debug: bool = False,
    hp_shots: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)
    ds_final_path = os.path.join(save_dir, "d3d_source.nc")
    ds_thomson_dir = os.path.join(save_dir, "d3d_thomson_raw")
    ds_profile_dir = os.path.join(save_dir, "d3d_profiles_raw")
    ds_efit_dir = os.path.join(save_dir, "d3d_efit_raw")
    ds_assembly_dir = os.path.join(save_dir, "d3d_assembly")

    for directory in [
        ds_thomson_dir,
        ds_profile_dir,
        ds_efit_dir,
        ds_assembly_dir,
    ]:
        os.makedirs(directory, exist_ok=True)

    if clean:
        for path in [
            ds_final_path,
        ]:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removed existing file {path}")
        for directory in [
            ds_thomson_dir,
            ds_profile_dir,
            ds_efit_dir,
            ds_assembly_dir,
        ]:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                os.remove(file_path)
            logger.info(f"Removed existing files in directory {directory}")

    if os.path.exists(ds_final_path):
        logger.info(
            f"Final dataset already exists at {ds_final_path}, skipping creation"
        )
        return

    shotlist = get_shotlist_from_sql(num_shots=num_shots, hp_shots=hp_shots)
    logger.info(f"Retrieved shotlist of {len(shotlist)} shots from SQL")

    valid_shotlist = shotlist.copy()

    for shot in shotlist:
        ds_efit_path = os.path.join(ds_efit_dir, f"{shot}.nc")
        if not os.path.exists(ds_efit_path):
            ds_efit = get_efit_dataset(shot)
            ds_efit.to_netcdf(ds_efit_path)
            logger.info(f"Saved raw EFIT dataset to {ds_efit_path}")

    ds_final = make_final_dataset(
        valid_shotlist,
        ds_thomson_dir,
        ds_profile_dir,
        ds_efit_dir,
        ds_assembly_dir,
    )
    ds_final.to_netcdf(ds_final_path)
    logger.info(f"Saved final D3D dataset to {ds_final_path}")


if __name__ == "__main__":
    fire.Fire(make_d3d_dataset)