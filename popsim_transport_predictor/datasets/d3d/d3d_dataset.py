"""Makes the 'raw' DIII-D dataset on omega, to be processed later by POPSIM"""

import os

import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings
from disruption_py.workflow import get_shots_data
from loguru import logger

from popsim_transport_predictor import EPISODE_DIM, PACKAGE_ROOT, TIME_COORD, TIME_DIM
from popsim_transport_predictor.datasets import make_uniform_1khz_timebase
from popsim_transport_predictor.datasets.workflow import DataWorkflow

DEFAULT_SHOTLIST_FILE = os.path.join(
    PACKAGE_ROOT, "datasets", "d3d", "HBP_shotlist_2024"
)


class D3DDataWorkflow(DataWorkflow):
    def __init__(
        self,
        ds_name: str,
        shotlist_file: str,
        raw_data_dir: str,
        final_ds_dir: str,
        max_num_shots: int | None = None,
        use_ida: bool | None = True,
    ):
        """
        Parameters
        ----------
        ds_name : str
            Name of the dataset (e.g., 'd3d', 'tcv', 'cmod')
        shotlist_file: str
            Path to file containing list of shots to process
        raw_data_dir : str
            Directory where raw data files are stored
        final_ds_dir : str
            Directory to save the final combined dataset
        max_num_shots : int | None
            Maximum number of shots to process (for testing). If None, process all shots.
        use_ida : bool | None
            Whether to use IDA for profile data (True) or Zipfit (False)
        """

        super().__init__(
            ds_name,
            shotlist_file,
            raw_data_dir,
            final_ds_dir,
            max_num_shots=max_num_shots,
        )
        self.use_ida = use_ida

        self.valid_signal_bounds = {
            "Wtot_MJ": (1e-3, None),
        }

    def _get_0D_dataset(self, shot: int) -> xr.Dataset:
        retrieval_settings = RetrievalSettings(
            run_methods=["get_efit_parameters"],
            time_setting="efit",
            only_requested_columns=False,
        )
        efit_result = get_shots_data(
            tokamak=Tokamak.D3D,
            shotlist_setting=shot,
            retrieval_settings=retrieval_settings,
            num_processes=1,
        )
        efit_result = efit_result.set_index(idx=["shot", "time"]).unstack("idx")

        retrieval_settings = RetrievalSettings(
            run_columns=[
                "ip",
                "bt",
                "wmhdf",
                "betapf",
                "n_e",
                "p_ohm",
                "p_rad",
                "p_nbi",
                "p_ech",
                "p_ich",
                "p_lhcd",
            ],
            time_setting="efit",
            only_requested_columns=True,
        )
        global_result = get_shots_data(
            tokamak=Tokamak.D3D,
            shotlist_setting=shot,
            retrieval_settings=retrieval_settings,
            num_processes=1,
        )
        global_result = global_result.set_index(idx=["shot", "time"]).unstack("idx")

        result = xr.merge([efit_result, global_result], compat="override")

        return result

    def _get_profile_dataset_zipfit(self, shot: int) -> xr.Dataset:
        retrieval_settings = RetrievalSettings(
            run_columns=["ne_rho", "te_rho"],
            only_requested_columns=False,
        )
        profile_result = get_shots_data(
            tokamak=Tokamak.D3D,
            shotlist_setting=shot,
            retrieval_settings=retrieval_settings,
            num_processes=1,
        )
        # Make the "time" coordinate the dimension instead of "idx"
        profile_result = profile_result.swap_dims({"idx": "time"})

        return profile_result

    def _get_profile_dataset_ida(self, shot: int) -> xr.Dataset | None:
        ida_path = f"/fusion/projects/results/ida-results/HBP_database/IDA_{shot}_.cdf"
        if not os.path.exists(ida_path):
            logger.warning(
                f"IDA profile file for shot {shot} not found at {ida_path}, skipping shot."
            )
            return None
        ds = xr.open_dataset(ida_path)
        # Rename profile varaibles to avoid conflict with 0D signals
        ds["Te_rho"] = ds["T_e"]
        ds["ne_rho"] = ds["n_e"]
        ds = ds[["Te_rho", "ne_rho"]]

        ds["time"] = ds["time"] / 1e3  # Convert ms to s
        ds = ds.expand_dims("shot")
        return ds

    def make_raw_data_files(self):
        """Create raw data files from source for DIII-D dataset

        We are getting 0D signals from MDSPlus and using profiles from IDA or Zipfit.
        """

        if not int(np.version.version.split(".")[0]) < 2:
            raise RuntimeError(
                "disruption_py on DIII-D currently requires numpy < 2 please use the make_d3d_venv.sh script to create the correct environment."
            )

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

            if self.use_ida:
                ds_profile = self._get_profile_dataset_ida(shot)
                if ds_profile is None:
                    continue
            else:
                ds_profile = self._get_profile_dataset_zipfit(shot)

            ds_0d = self._get_0D_dataset(shot)

            # Put each dataset on a 1 kHz timebase, using previous value fill
            max_time = max(
                ds_profile["time"].max().item(),
                ds_0d["time"].max().item(),
            )
            timebase = make_uniform_1khz_timebase(max_time)

            ds_profile = ds_profile.reindex(time=timebase, method="ffill")
            ds_0d = ds_0d.interp(
                time=timebase, method="nearest"
            )  # This should be okay since 0D signal is already at high time resolution
            ds_assembly = xr.merge([ds_profile, ds_0d], compat="no_conflicts")

            ds_standardized = self.standardize_signal_names(ds_assembly)
            if ds_standardized is None:
                logger.warning(f"Standardization failed for shot {shot}, skipping")
                continue

            ds_standardized.to_netcdf(ds_path)
            logger.info(f"Saved raw dataset for shot {shot} to {ds_path}")
            processed_shots += 1

        logger.info("Finished making raw data files.")

    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename signals in the dataset to match the POPSIM convention"""

        # Simple renames
        ds = ds.rename(
            {
                "aminor": "a_minor",
                "tritop": "delta_top",
                "tribot": "delta_bottom",
                "psi_n": "rho",
            }
        )

        # Conversions
        ds["Te_keV_rho"] = ds["Te_rho"] / 1e3  # Convert eV to keV
        ds["ne20_rho"] = ds["ne_rho"] / 1e20  # Convert m^-3 to 10^20 m^-3

        ds["R0"] = ds["rmaxis"]
        ds["B0"] = np.abs(ds["bt"])
        ds["Ip_MA"] = np.abs(ds["ip"]) / 1e6  # Convert A to MA
        ds["Wmhd_MJ"] = ds["wmhdf"] / 1e6  # Convert J to MJ
        ds["ne20_line_avg"] = ds["n_e"] / 1e20  # Convert m^-3 to 10^20 m^-3

        ds["P_ECRH_MW"] = ds["p_ech"] / 1e6  # Convert W to MW
        ds["P_NBI_MW"] = ds["p_nbi"] / 1e6  # Convert W to MW
        ds["P_oh_MW"] = ds["p_ohm"] / 1e6  # Convert W to MW
        ds["P_rad_MW"] = ds["p_rad"] / 1e6  # Convert W to MW
        ds["P_ICRF_MW"] = ds["p_ich"] / 1e6  # Convert W to MW
        ds["P_LH_MW"] = ds["p_lhcd"] / 1e6  # Convert W to MW

        # Drop unnecessary variables
        ds = ds[
            [
                "Te_keV_rho",
                "ne20_rho",
                "Wmhd_MJ",
                "R0",
                "B0",
                "Ip_MA",
                "a_minor",
                "kappa",
                "delta_top",
                "delta_bottom",
                "P_ECRH_MW",
                "P_NBI_MW",
                "P_oh_MW",
                "P_rad_MW",
                "P_ICRF_MW",
                "P_LH_MW",
            ]
        ]

        # If any signal is all NaN, return None to skip this shot
        for signal in ds.data_vars:
            if ds[signal].isnull().all():
                logger.warning(
                    f"Signal {signal} is all NaN for shot {ds['shot'].item()}, skipping shot."
                )
                return None

        # Make episode dimension, time dimension, and time coordinate names consistent
        # Rename time dimension to follow POPSIM convention: time_dim -> time_idx
        if TIME_DIM not in ds.dims:
            ds = ds.rename_dims({"time": TIME_DIM})
        if EPISODE_DIM not in ds.dims:
            ds = ds.rename_dims({"shot": EPISODE_DIM})
        if TIME_COORD not in ds.coords:
            ds = ds.rename_vars({"time": TIME_COORD})

        return ds
