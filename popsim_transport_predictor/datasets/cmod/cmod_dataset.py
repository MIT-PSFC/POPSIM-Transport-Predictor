"""Makes the 'raw' CMOD dataset on mfews, to be processed later by POPSIM"""

import os

import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings
from disruption_py.workflow import get_shots_data
from loguru import logger

from popsim_transport_predictor import EPISODE_DIM, PACKAGE_ROOT, TIME_COORD, TIME_DIM
from popsim_transport_predictor.config import config
from popsim_transport_predictor.datasets import make_uniform_1khz_timebase
from popsim_transport_predictor.datasets.cmod import (
    CMOD_DATASET_SIGNALS,
)
from popsim_transport_predictor.datasets.cmod.gp_fit import gp_profile
from popsim_transport_predictor.datasets.dispy_utils import summary
from popsim_transport_predictor.datasets.workflow import DataWorkflow

DEFAULT_SHOTLIST_FILE = os.path.join(PACKAGE_ROOT, "datasets", "cmod", "cmod_shotlist")


class CModDataWorkflow(DataWorkflow):
    """C-Mod specific data workflow for creating and processing datasets.

    This workflow retrieves data from C-Mod's MDSPlus server, fits Thomson scattering
    profiles using Gaussian processes, standardizes signal names, and creates a uniform
    1 kHz timebase dataset suitable for POPSIM transport prediction studies.

    Note: This workflow can execute on the present cluster with C-Mod data access.
    """

    def __init__(
        self,
        ds_name: str,
        shotlist_file: str | None,
        data_assembly_dir: str,
        max_num_shots: int | None = None,
        gp_fit_rho: np.ndarray | None = None,
        skip_profiles: bool = False,
    ):
        """Initialize the C-Mod data workflow.

        Parameters
        ----------
        ds_name : str
            Name of the dataset/study, used for directory naming
        shotlist_file : str | None
            Path to file containing list of shots to process. If None, retrieves
            shotlist from SQL database using parameters in config.toml.
        data_assembly_dir : str
            Directory where data files are stored and final dataset will be saved
        max_num_shots : int | None
            Maximum number of shots to process (for testing). If None, process all shots.
        gp_fit_rho : np.ndarray | None
            Radial locations for GP profile fitting. If None, uses default from config.
        skip_profiles : bool
            If True, skip profile fitting and use zero arrays instead. Useful for testing.
        """

        # Use centralized config
        self.config = config.cmod

        # Set up GP fitting rho grid
        if gp_fit_rho is not None:
            self.gp_fit_rho = gp_fit_rho
        else:
            # Use config values
            prof_config = self.config["profile_fitting"]
            self.gp_fit_rho = np.linspace(
                prof_config["rho_min"],
                prof_config["rho_max"],
                prof_config["num_rho_points"],
            )

        # Call parent init (which will call _get_shotlist_from_source if needed)
        super().__init__(
            ds_name,
            shotlist_file,
            data_assembly_dir,
            max_num_shots=max_num_shots,
            skip_profiles=skip_profiles,
        )

    def _get_shotlist_from_source(self) -> list[int]:
        """Retrieve shotlist from C-Mod SQL database.

        Uses the summary() function to query the C-Mod database for shots
        matching the criteria in config.toml, then filters to blessed Thomson days.

        Returns
        -------
        list[int]
            List of shot numbers to process
        """
        query_config = self.config["shotlist_query"]

        # Query the SQL database
        data = summary(
            summary_table=query_config["summary_table"],
            ipmax=query_config["ipmax"],
            pulse_length=query_config["pulse_length"],
            min_shot=query_config["min_shot"],
            max_shot=query_config["max_shot"],
            shots=False,
        )
        shotlist = data[:, 0].astype(int).tolist()

        # Build list of blessed Thomson days
        thomson_config = self.config["thomson_filtering"]
        blessed_days = list(thomson_config["blessed_days"])
        for day_range in thomson_config["blessed_day_ranges"]:
            blessed_days.extend(range(day_range[0], day_range[1]))

        # Filter to blessed Thomson days
        shotlist = [shot for shot in shotlist if int(shot / 1000) in blessed_days]

        return shotlist

    def _get_thomson_dataset(self, shot: int) -> xr.Dataset | None:
        """Retrieve Thomson scattering data for a shot.

        Parameters
        ----------
        shot : int
            Shot number to retrieve

        Returns
        -------
        xr.Dataset | None
            Raw Thomson scattering data, or None if retrieval fails
        """
        retrieval_settings = RetrievalSettings(
            run_methods=["get_thomson_channels"],
            only_requested_columns=False,
        )
        result = get_shots_data(
            tokamak=Tokamak.CMOD,
            shotlist_setting=[shot],
            retrieval_settings=retrieval_settings,
            num_processes=1,
        )
        if len(result) == 0:
            return None
        result = result.set_index(idx=["shot", "time"]).unstack("idx")
        return result

    def _make_profile_dataset(self, ds_thomson: xr.Dataset) -> xr.Dataset:
        """Perform GP fitting on Thomson scattering data.

        This assumes the input data is raw Thomson scattering data from _get_thomson_dataset()
        where Te is in keV and ne is in m^-3. Returns Te in keV and ne in 1e20 m^-3.

        Parameters
        ----------
        ds_thomson : xr.Dataset
            Raw Thomson scattering dataset

        Returns
        -------
        xr.Dataset
            GP-fitted profiles on rho grid
        """

        shot_prediction = {}

        for shot in ds_thomson["shot"].values:
            ds_shot = ds_thomson.where(ds_thomson["shot"] == shot, drop=True)
            ds_shot = ds_shot.squeeze(dim="shot", drop=True)
            times = ds_shot["time"].values
            data_x = ds_shot["ts_channel_rho"].values.T  # shape (time, channel)

            te_data = np.full((len(times), len(self.gp_fit_rho)), np.nan)
            te_err = np.full((len(times), len(self.gp_fit_rho)), np.nan)
            ne_data = np.full((len(times), len(self.gp_fit_rho)), np.nan)
            ne_err = np.full((len(times), len(self.gp_fit_rho)), np.nan)

            for variable in ["te", "ne"]:
                data_y = ds_shot[
                    f"ts_channel_{variable}"
                ].values.T  # shape (time, channel)
                err_y = ds_shot[
                    f"ts_channel_{variable}_error"
                ].values.T  # shape (time, channel)

                if variable == "ne":
                    data_y = data_y * 1e-20  # Convert to [1e20 m^-3]
                    err_y = err_y * 1e-20

                # If data or error bar is incredibly small, set to NaN since it's probably bad data
                data_y = np.where(data_y < 0.001, np.nan, data_y)
                err_y = np.where(err_y < 0.001, np.nan, err_y)

                # I do not trust you can measure within 20 eV or within 2e18 m^-3
                err_y = np.where(err_y < 0.02, 0.02, err_y)

                for i_time, _ in enumerate(times):
                    y_star, std_y_star, _, _ = gp_profile(
                        data_X=data_x[i_time, :],
                        data_y=data_y[i_time, :],
                        err_y=err_y[i_time, :],
                        X_star=self.gp_fit_rho,
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
                    "Te_keV_rho": (("time", "rho"), te_data),
                    "Te_keV_rho_error": (("time", "rho"), te_err),
                    "ne20_rho": (("time", "rho"), ne_data),
                    "ne20_rho_error": (("time", "rho"), ne_err),
                },
                coords={
                    "time": times,
                    "rho": self.gp_fit_rho,
                },
            )

        # Put the shots together into the original dataset with shot dimension
        ds_profiles = xr.concat(
            [shot_prediction[shot] for shot in shot_prediction],
            dim=xr.IndexVariable("shot", list(shot_prediction.keys())),
        )

        return ds_profiles

    def _get_efit_dataset(self, shot: int) -> xr.Dataset:
        """Retrieve EFIT and 0D signals for a shot.

        Parameters
        ----------
        shot : int
            Shot number to retrieve

        Returns
        -------
        xr.Dataset
            Dataset with EFIT and 0D signals
        """
        retrieval_settings = RetrievalSettings(
            run_columns=CMOD_DATASET_SIGNALS,
            time_setting="efit",
            only_requested_columns=True,
        )
        result = get_shots_data(
            tokamak=Tokamak.CMOD,
            shotlist_setting=shot,
            retrieval_settings=retrieval_settings,
            num_processes=1,
        )
        result = result.set_index(idx=["shot", "time"]).unstack("idx")
        return result

    def make_raw_data_files(self):
        """Create raw data files from source for C-Mod dataset.

        This method retrieves Thomson scattering data, performs GP fitting for profiles,
        retrieves EFIT and 0D signals, combines them on a uniform 1 kHz timebase,
        standardizes signal names, and saves one netCDF file per shot.
        """

        processed_shots = 0
        for shot in self.shotlist:
            if self.max_num_shots is not None and processed_shots >= self.max_num_shots:
                logger.info(
                    f"Reached maximum number of shots to process: {self.max_num_shots}"
                )
                break

            ds_path = os.path.join(self.raw_data_dir, f"{shot}.nc")
            if os.path.exists(ds_path):
                logger.info(f"Raw dataset for shot {shot} already exists at {ds_path}")
                processed_shots += 1
                continue

            # Get EFIT and 0D data
            try:
                ds_efit = self._get_efit_dataset(shot)
            except Exception as e:
                logger.warning(f"Failed to retrieve EFIT data for shot {shot}: {e}")
                continue

            if self.skip_profiles:
                # Skip profile fitting, use zeros instead
                logger.info(
                    f"Skipping profile fitting for shot {shot} (skip_profiles=True)"
                )
                max_time = ds_efit["time"].max().item()
                timebase = make_uniform_1khz_timebase(max_time)

                # Create dummy profile dataset with zeros
                ds_profiles = xr.Dataset(
                    data_vars={
                        "Te_keV_rho": (
                            ("time", "rho"),
                            np.zeros((len(timebase), len(self.gp_fit_rho))),
                        ),
                        "Te_keV_rho_error": (
                            ("time", "rho"),
                            np.zeros((len(timebase), len(self.gp_fit_rho))),
                        ),
                        "ne20_rho": (
                            ("time", "rho"),
                            np.zeros((len(timebase), len(self.gp_fit_rho))),
                        ),
                        "ne20_rho_error": (
                            ("time", "rho"),
                            np.zeros((len(timebase), len(self.gp_fit_rho))),
                        ),
                    },
                    coords={
                        "time": timebase,
                        "rho": self.gp_fit_rho,
                        "shot": shot,
                    },
                )
                ds_profiles = ds_profiles.expand_dims("shot")

                ds_efit = ds_efit.interp(time=timebase, method="nearest")
                ds_assembly = xr.merge([ds_profiles, ds_efit], compat="override")
            else:
                # Get Thomson data
                ds_thomson = self._get_thomson_dataset(shot)
                if ds_thomson is None:
                    logger.warning(
                        f"Skipping shot {shot} since no Thomson data was retrieved"
                    )
                    continue

                # Fit Thomson profiles
                ds_profiles = self._make_profile_dataset(ds_thomson)

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
                )  # EFIT is already at high time resolution

                ds_assembly = xr.merge(
                    [ds_thomson, ds_profiles, ds_efit], compat="override"
                )

            ds_standardized = self.standardize_signal_names(ds_assembly)
            if ds_standardized is None:
                logger.warning(f"Standardization failed for shot {shot}, skipping")
                continue

            ds_standardized.to_netcdf(ds_path)
            logger.info(f"Saved raw dataset for shot {shot} to {ds_path}")
            processed_shots += 1

        logger.info("Finished making raw data files.")

    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset | None:
        """Rename signals in the dataset to match the POPSIM convention.

        This includes unit conversions and creating derived quantities.
        Also validates that critical signals are present.

        Parameters
        ----------
        ds : xr.Dataset
            Raw dataset with device-specific signal names

        Returns
        -------
        xr.Dataset | None
            Standardized dataset, or None if critical signals are missing
        """

        # Simple renames
        ds = ds.rename(
            {
                "tritop": "delta_top",
                "tribot": "delta_bottom",
                "rmagx": "R0",
            }
        )

        # Conversions (Te_keV_rho and ne20_rho already in correct units from GP fitting)
        ds["Wtot_MJ"] = ds["wmhd"] / 1e6  # Convert J to MJ
        ds["B0"] = np.abs(ds["btor"])
        ds["Ip_MA"] = np.abs(ds["ip"]) / 1e6  # Convert A to MA
        ds["ne20_line_avg"] = ds["n_e"] / 1e20  # Convert m^-3 to 10^20 m^-3

        # Convert all powers to MW
        ds["P_oh_MW"] = ds["p_oh"] / 1e6
        ds["P_rad_MW"] = ds["p_rad"] / 1e6
        ds["P_ICRF_MW"] = ds["p_icrf"] / 1e6
        ds["P_LH_MW"] = ds["p_lh"] / 1e6

        # C-Mod doesn't have NBI or ECRH, set to zero where Ip_MA is valid
        ds["P_NBI_MW"] = xr.zeros_like(ds["Ip_MA"])
        ds["P_ECRH_MW"] = xr.zeros_like(ds["Ip_MA"])

        # C-Mod doesn't have tau_conf from standard diagnostics
        ds["tau_conf"] = xr.zeros_like(ds["Ip_MA"])

        # Only keep variables of interest
        ds = ds[
            [
                "Te_keV_rho",
                "ne20_rho",
                "Wtot_MJ",
                "R0",
                "B0",
                "Ip_MA",
                "a_minor",
                "kappa",
                "delta_top",
                "delta_bottom",
                "ne20_line_avg",
                "P_ECRH_MW",
                "P_NBI_MW",
                "P_oh_MW",
                "P_rad_MW",
                "P_ICRF_MW",
                "P_LH_MW",
                "tau_conf",
            ]
        ]

        # If any *important* signal is all NaN, return None to skip this shot
        for signal in ["Te_keV_rho", "ne20_rho", "Ip_MA"]:
            if ds[signal].isnull().all():
                logger.warning(
                    f"Signal {signal} is all NaN for shot {ds['shot'].item()}, skipping shot."
                )
                return None

        # Make episode dimension, time dimension, and time coordinate names consistent
        if TIME_DIM not in ds.dims:
            ds = ds.rename_dims({"time": TIME_DIM})
        if EPISODE_DIM not in ds.dims:
            ds = ds.rename_dims({"shot": EPISODE_DIM})
        if TIME_COORD not in ds.coords:
            ds = ds.rename_vars({"time": TIME_COORD})

        return ds

    def device_specific_processing(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply C-Mod specific processing steps.

        Currently no special processing is needed for C-Mod beyond the base workflow.

        Parameters
        ----------
        ds : xr.Dataset
            Standardized dataset

        Returns
        -------
        xr.Dataset
            Processed dataset ready for general workflow
        """
        # No special processing needed for C-Mod at this time
        return ds
