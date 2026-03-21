"""Makes the 'raw' CMOD dataset on mfews, to be processed later by POPSIM"""

import os

import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import (
    LogSettings,
    RetrievalSettings,
)
from disruption_py.settings.output_setting import DatasetOutputSetting
from disruption_py.workflow import get_shots_data
from loguru import logger

from transport_study import EPISODE_DIM, PACKAGE_ROOT, TIME_COORD, TIME_DIM
from transport_study.config import config
from transport_study.datasets import make_uniform_1khz_timebase
from transport_study.datasets.cmod import (
    CMOD_DATASET_SIGNALS,
)
from transport_study.datasets.cmod.gp_fit import fit_gp_hyperparameters, gp_profile
from transport_study.datasets.dispy_utils import summary
from transport_study.datasets.workflow import DataWorkflow

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
        gp_fit_psi: np.ndarray | None = None,
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
        gp_fit_psi : np.ndarray | None
            Radial locations for GP profile fitting. If None, uses default from config.
        skip_profiles : bool
            If True, skip profile fitting and use zero arrays instead. Useful for testing.
        """

        # Use centralized config
        self.config = config.cmod

        # Set up GP fitting psi grid
        if gp_fit_psi is not None:
            self.gp_fit_psi = gp_fit_psi
        else:
            # Use config values
            prof_config = self.config["profile_fitting"]
            self.gp_fit_psi = np.linspace(
                prof_config["psi_min"],
                prof_config["psi_max"],
                prof_config["num_psi_points"],
            )

        # Call parent init (which will call _get_shotlist_from_source if needed)
        super().__init__(
            ds_name,
            shotlist_file,
            data_assembly_dir,
            max_num_shots=max_num_shots,
        )

        self.filter_config = {
            "Wtot_MJ": {"min": 0.001, "max": 2},
            "ne20_line_avg": {"min": 0.01, "max": 4},
            "betan": {"min": 0, "max": 5},
        }
        self.individual_filter_config = None
        self.skip_profiles = skip_profiles

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
            output_setting=DatasetOutputSetting(path=False),
            log_settings=LogSettings(file_path=None),
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
            GP-fitted profiles on psi grid
        """

        shot_prediction = {}
        cached_hyperparams: dict[str, np.ndarray | None] = {"te": None, "ne": None}

        for shot in ds_thomson["shot"].values:
            ds_shot = ds_thomson.where(ds_thomson["shot"] == shot, drop=True)
            ds_shot = ds_shot.squeeze(dim="shot", drop=True)
            times = ds_shot["time"].values
            data_x = ds_shot["ts_channel_rho"].values.T  # shape (time, channel)

            te_data = np.full((len(times), len(self.gp_fit_psi)), np.nan)
            te_err = np.full((len(times), len(self.gp_fit_psi)), np.nan)
            ne_data = np.full((len(times), len(self.gp_fit_psi)), np.nan)
            ne_err = np.full((len(times), len(self.gp_fit_psi)), np.nan)

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

                # Historic data, we're mostly going off vibes anyway
                err_y = np.where(err_y < 0.1, 0.1, err_y)

                if cached_hyperparams[variable] is None:
                    # Fit once and reuse for the remainder of the dataset build.
                    for i_seed, _ in enumerate(times):
                        cached_hyperparams[variable] = fit_gp_hyperparameters(
                            data_X=data_x[i_seed, :],
                            data_y=data_y[i_seed, :],
                            err_y=err_y[i_seed, :],
                        )
                        if cached_hyperparams[variable] is not None:
                            break

                for i_time, _ in enumerate(times):
                    y_star, std_y_star, _, _ = gp_profile(
                        data_X=data_x[i_time, :],
                        data_y=data_y[i_time, :],
                        err_y=err_y[i_time, :],
                        X_star=self.gp_fit_psi,
                        calc_gradient=False,
                        hyperparams=cached_hyperparams[variable],
                        optimize_hyperparams=cached_hyperparams[variable] is None,
                    )
                    if y_star is None:
                        continue

                    if variable == "te":
                        te_data[i_time, :] = y_star
                        te_err[i_time, :] = std_y_star
                    elif variable == "ne":
                        ne_data[i_time, :] = y_star
                        ne_err[i_time, :] = std_y_star

                    if i_time % 10 == 0:
                        logger.verbose(
                            f"Completed {i_time}/{len(times)} fits for {variable}"
                        )

                    if config.debug and i_time > 20:
                        break

            shot_prediction[shot] = xr.Dataset(
                data_vars={
                    "Te_keV_psi": (("time", "psi"), te_data),
                    "Te_keV_psi_error": (("time", "psi"), te_err),
                    "ne20_psi": (("time", "psi"), ne_data),
                    "ne20_psi_error": (("time", "psi"), ne_err),
                },
                coords={
                    "time": times,
                    "psi": self.gp_fit_psi,
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
            output_setting=DatasetOutputSetting(path=False),
            log_settings=LogSettings(file_path=None),
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
                        "Te_keV_psi": (
                            ("time", "psi"),
                            np.zeros((len(timebase), len(self.gp_fit_psi))),
                        ),
                        "Te_keV_psi_error": (
                            ("time", "psi"),
                            np.zeros((len(timebase), len(self.gp_fit_psi))),
                        ),
                        "ne20_psi": (
                            ("time", "psi"),
                            np.zeros((len(timebase), len(self.gp_fit_psi))),
                        ),
                        "ne20_psi_error": (
                            ("time", "psi"),
                            np.zeros((len(timebase), len(self.gp_fit_psi))),
                        ),
                    },
                    coords={
                        "time": timebase,
                        "psi": self.gp_fit_psi,
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
        # POWER BALANCE
        ds["Wtot_MJ"] = ds["wmhd"] / 1e6  # Convert J to MJ
        ds["B0"] = np.abs(ds["btor"])
        ds["Ip_MA"] = np.abs(ds["ip"]) / 1e6  # Convert A to MA
        ds["R0"] = ds["rout"]
        ds["ne20_line_avg"] = ds["n_e"] / 1e20  # Convert m^-3 to 10^20 m^-3
        ds["P_oh_MW"] = ds["p_oh"] / 1e6
        ds["P_rad_MW"] = ds["p_rad"] / 1e6
        ds["P_ICRF_MW"] = ds["p_icrf"] / 1e6
        ds["P_LH_MW"] = ds["p_lh"] / 1e6
        # C-Mod doesn't have NBI or ECRH, set to zero where Ip_MA is valid
        ds["P_NBI_MW"] = xr.zeros_like(ds["Ip_MA"])
        ds["P_ECRH_MW"] = xr.zeros_like(ds["Ip_MA"])

        # PROFILE PREDICTOR TRAINING
        # Te_kev_psi
        # ne20_psi
        # Ip_MA
        # B0
        ds["betan"] = ds["beta_n"]
        ds["ne20_edge"] = ds["ne20_psi"].sel(
            psi=0.9
        )  # C-Mod doesn't have edge interferometry, get the density from psi=0.9
        # R0
        # kappa
        # a_minor
        ds["delta_top"] = ds["tritop"]
        ds["delta_bot"] = ds["tribot"]

        # C-Mod doesn't have tau_conf from standard diagnostics
        ds["tau_conf"] = xr.zeros_like(ds["Ip_MA"])

        # Only keep variables of interest
        kept_vars = {
            # POWER BALANCE
            "Wtot_MJ",
            # PROFILE PREDICTOR TRAINING
            "Te_keV_psi",
            "ne20_psi",
            "Ip_MA",
            "B0",
            "betan",
            "ne20_edge",
            "R0",
            "kappa",
            "a_minor",
            "delta_top",
            "delta_bot",
            # OTHER
            "beta_p",  # EFIT
        }

        ds = ds[list(kept_vars)]

        # If any *important* signal is all NaN, return None to skip this shot
        for signal in ["Te_keV_psi", "ne20_psi", "Ip_MA"]:
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

        Parameters
        ----------
        ds : xr.Dataset
            Standardized dataset

        Returns
        -------
        xr.Dataset
            Processed dataset ready for general workflow
        """

        # Cull obviously bad fits, such as when the point at psi = 0 is super low (1160503009 0.83)
        # Or when any profile value at psi < 1.0 is negative
        negative_profile_mask = (ds["ne20_psi"].where(ds["psi"] < 1.0) < 0).any(
            dim="psi"
        ) | (ds["Te_keV_psi"].where(ds["psi"] < 1.0) < 0).any(dim="psi")
        low_value_mask = (ds["Te_keV_psi"].sel(psi=0) < 1.0) | (
            ds["ne20_psi"].sel(psi=0) < 0.3
        )
        valid_profile_mask = ~(negative_profile_mask | low_value_mask)
        if valid_profile_mask.sum() == 0:
            logger.warning(f"All profiles are invalid for shot {ds['shot'].item()}")
        else:
            logger.debug(
                f"Culled {(~valid_profile_mask).sum().item() / (valid_profile_mask.sum().item()) * 100:.2f}% invalid profiles for shot {ds['shot'].item()}"
            )
        ds["ne20_psi"] = ds["ne20_psi"].where(valid_profile_mask)
        ds["Te_keV_psi"] = ds["Te_keV_psi"].where(valid_profile_mask)

        # Rename 'psi' dimension to 'psi_n'
        ds = ds.rename_dims({"psi": "psi_n"})
        ds = ds.rename_vars({"psi": "psi_n"})

        return ds
