"""Makes the 'raw' CMOD dataset on mfews, to be processed later by POPSIM"""

import os

import fire
import loguru
import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings
from disruption_py.workflow import get_shots_data

from popsim_transport_predictor.datasets_dispy.cmod import (
    BLESSED_THOMSON_DAYS,
    IPMAX,
    MAX_SHOT,
    MIN_SHOT,
    PULSE_LENGTH,
    SUMMARY_TABLE,
)
from popsim_transport_predictor.datasets_dispy.cmod.gp_fit import gp_profile
from popsim_transport_predictor.datasets_dispy.dispy_utils import summary


def get_shotlist_from_sql(num_shots: int | None) -> list[int]:
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


def get_thomson_dataset(shot) -> xr.Dataset:
    retrieval_settings = RetrievalSettings(
        run_methods=["get_thomson_channels"],
        only_requested_columns=False,
    )
    result = get_shots_data(
        tokamak=Tokamak.CMOD,
        shotlist_setting=[shot],
        retrieval_settings=retrieval_settings,
        output_setting="dataset",
        num_processes=1,
    )
    result = result.set_index(idx=["shot", "time"]).unstack("idx")
    return result


def make_profile_dataset(ds_thomson: xr.Dataset, gp_fit_rho: np.ndarray) -> xr.Dataset:
    """
    Perform fitting with GPtools

    This assumes the input data is raw Thomson scattering data from get_thomson_dataset()
    Where Te is in keV and ne is in m^-3
    This returns Te in keV and ne in 1e20 m^-3

    Parameters
    ----------
    ds_thomson : xr.Dataset
        Raw Thomson scattering dataset
    gp_fit_rho : np.ndarray
        Radial locations to predict at (normalized minor radius)
    """

    shot_prediction = {}

    for shot in ds_thomson["shot"].values:
        ds_shot = ds_thomson.where(ds_thomson["shot"] == shot, drop=True)
        times = ds_shot["time"].values
        data_x = ds_shot["ts_channel_rho"].values

        te_data = np.full((len(times), len(gp_fit_rho)), np.nan)
        te_err = np.full((len(times), len(gp_fit_rho)), np.nan)
        ne_data = np.full((len(times), len(gp_fit_rho)), np.nan)
        ne_err = np.full((len(times), len(gp_fit_rho)), np.nan)

        for variable in ["te", "ne"]:
            data_y = ds_shot[f"ts_channel_{variable}"].values
            err_y = ds_shot[f"ts_channel_{variable}_error"].values

            if variable == "ne":
                data_y = data_y * 1e-20  # Convert to [1e20 m^-3]
                err_y = err_y * 1e-20

            # If data or error bar is incredibly small, set to NaN since it's probably bad data
            data_y = np.where(data_y < 0.001, np.nan, data_y)
            err_y = np.where(err_y < 0.001, np.nan, err_y)

            # I do not trust you can measure within 10 eV or within 1e18 m^-3
            err_y = np.where(err_y < 0.01, 0.01, err_y)

            for i_time, _ in enumerate(times):
                y_star, std_y_star, _, _ = gp_profile(
                    data_X=data_x[i_time, :],
                    data_y=data_y[i_time, :],
                    err_y=err_y[i_time, :],
                    X_star=gp_fit_rho,
                    calc_gradient=False,
                )

                if variable == "te":
                    te_data[i_time, :] = y_star
                    te_err[i_time, :] = std_y_star
                elif variable == "ne":
                    ne_data[i_time, :] = y_star
                    ne_err[i_time, :] = std_y_star

        shot_prediction[shot] = xr.Dataset(
            data_vars={
                "gp_fit_te": (("time", "gp_fit_rho"), te_data),
                "gp_fit_te_error": (("time", "gp_fit_rho"), te_err),
                "gp_fit_ne": (("time", "gp_fit_rho"), ne_data),
                "gp_fit_ne_error": (("time", "gp_fit_rho"), ne_err),
            },
            coords={
                "time": times,
                "gp_fit_rho": gp_fit_rho,
            },
            attrs={
                "description": f"GP fitted Thomson scattering profiles for shot {shot}",
            },
        )

    # Put the shots together into the original dataset with 'idx' as the dimension
    ds_profiles = xr.concat(
        [shot_prediction[shot] for shot in shot_prediction],
        dim=xr.IndexVariable("shot", list(shot_prediction.keys())),
    )

    return ds_profiles


def get_efit_dataset(shot: int) -> xr.Dataset:
    retrieval_settings = RetrievalSettings(
        run_columns=["ip"],
        time_setting="efit",
        only_requested_columns=True,
    )
    result = get_shots_data(
        tokamak=Tokamak.CMOD,
        shotlist_setting=shot,
        retrieval_settings=retrieval_settings,
        output_setting="dataset",
        num_processes=1,
    )
    return result


def make_final_dataset() -> xr.Dataset:
    """
    Combine profile and EFIT datasets onto a common timebase
    """


def make_cmod_dataset(  # noqa: PLR0912
    save_dir: str, num_shots: int, clean: bool = False, debug: bool = False
):
    """
    Makee the source CMOD dataset for POPSIM transport predictor study.

    This is handling one shot at a time to save on disk space since we aren't locking to a common timebase until the very end.
    Each file on its own is ~10 kB so we aren't too worried about wasted 4kB blocks.

    Workflow is as follows:
    1) Get shotlist from SQL summary table
    2) Filter to shots that have blessed Thomson scattering data
    3) Retrieve the raw TS data on its native timebase
    4) Filter to shots that have both core and edge TS data
    5) Retrieve EFIT and other 1D signals on the EFIT timebase
    6) Put TS data and EFIT data together on a uniform 1kHz timebase
    """

    os.makedirs(save_dir, exist_ok=True)
    ds_final_path = os.path.join(save_dir, "cmod_source.nc")
    ds_thomson_dir = os.path.join(save_dir, "cmod_thomson_raw")
    ds_profile_dir = os.path.join(save_dir, "cmod_profiles_raw")
    ds_efit_dir = os.path.join(save_dir, "cmod_efit_raw")
    for directory in [
        ds_thomson_dir,
        ds_profile_dir,
        ds_efit_dir,
    ]:
        os.makedirs(directory, exist_ok=True)

    if clean:
        for path in [
            ds_final_path,
        ]:
            if os.path.exists(path):
                os.remove(path)
                loguru.logger.info(f"Removed existing file {path}")
        for directory in [
            ds_thomson_dir,
            ds_profile_dir,
            ds_efit_dir,
        ]:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                os.remove(file_path)
            loguru.logger.info(f"Removed existing files in directory {directory}")

    if os.path.exists(ds_final_path):
        loguru.logger.info(
            f"Final dataset already exists at {ds_final_path}, skipping creation"
        )
        return

    shotlist = get_shotlist_from_sql(num_shots=num_shots)
    loguru.logger.info(f"Retrieved shotlist of {len(shotlist)} shots from SQL")

    for shot in shotlist:
        ds_thomson_path = os.path.join(ds_thomson_dir, f"{shot}.nc")
        if not os.path.exists(ds_thomson_path):
            ds_thomson = get_thomson_dataset(shot)
            ds_thomson.to_netcdf(ds_thomson_path)
            loguru.logger.info(f"Saved raw Thomson dataset to {ds_thomson_path}")
        else:
            ds_thomson = xr.load_dataset(ds_thomson_path)
            loguru.logger.info(
                f"Loaded existing Thomson dataset from {ds_thomson_path}"
            )

        ds_profile_path = os.path.join(ds_profile_dir, f"{shot}.nc")
        if not os.path.exists(ds_profile_path):
            if debug:
                # Only pick time within the range (0.2, 0.24) seconds for faster testing
                ds_thomson = ds_thomson.where(
                    (ds_thomson["time"] >= 0.2) & (ds_thomson["time"] <= 0.24),
                    drop=True,
                )
            gp_fit_rho = np.linspace(0, 1.1, 56)
            ds_profiles = make_profile_dataset(ds_thomson, gp_fit_rho)
            ds_profiles.to_netcdf(ds_profile_path)
            loguru.logger.info(f"Saved raw profile dataset to {ds_profile_path}")
        else:
            ds_profiles = xr.load_dataset(ds_profile_path)
            loguru.logger.info(
                f"Loaded existing profile dataset from {ds_profile_path}"
            )

        ds_efit_path = os.path.join(ds_efit_dir, f"{shot}.nc")
        if not os.path.exists(ds_efit_path):
            ds_efit = get_efit_dataset(shot)
            ds_efit.to_netcdf(ds_efit_path)
            loguru.logger.info(f"Saved raw EFIT dataset to {ds_efit_path}")
        else:
            ds_efit = xr.load_dataset(ds_efit_path)
            loguru.logger.info(f"Loaded existing EFIT dataset from {ds_efit_path}")

    ds_final = make_final_dataset()
    ds_final.to_netcdf(ds_final_path)
    loguru.logger.info(f"Saved final CMOD dataset to {ds_final_path}")


if __name__ == "__main__":
    fire.Fire(make_cmod_dataset)
