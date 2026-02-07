"""Makes the 'raw' TCV dataset, to be processed later by POPSIM"""

import os

import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from loguru import logger

from popsim_transport_predictor import EPISODE_DIM, PACKAGE_ROOT, TIME_COORD, TIME_DIM
from popsim_transport_predictor.datasets import make_uniform_1khz_timebase
from popsim_transport_predictor.datasets.workflow import DataWorkflow

DEFAULT_SHOTLIST_FILE = os.path.join(PACKAGE_ROOT, "datasets", "tcv", "tcv_shotlist")


class TCVDataWorkflow(DataWorkflow):
    """TCV specific data workflow for creating and processing datasets.

    This workflow reads from a pre-existing TCV dataset (e.g., from DEFUSE),
    standardizes signal names, applies filtering and cleaning, and creates
    individual shot files on a uniform 1 kHz timebase suitable for POPSIM
    transport prediction studies.

    Note: This workflow can execute on the present cluster with TCV data access.
    """

    def __init__(
        self,
        shotlist_file: str,
        raw_data_dir: str,
        final_ds_dir: str,
        max_num_shots: int | None = None,
    ):
        """Initialize the TCV data workflow.

        Parameters
        ----------
        shotlist_file : str
            Path to file containing list of shots to process
        raw_data_dir : str
            Directory where raw data files are stored
        final_ds_dir : str
            Directory to save the final combined dataset
        max_num_shots : int | None
            Maximum number of shots to process (for testing). If None, process all shots.
        """

        super().__init__(
            "tcv",
            shotlist_file,
            raw_data_dir,
            final_ds_dir,
            max_num_shots=max_num_shots,
        )

    def _load_source_dataset(self) -> xr.Dataset:
        """Load the source TCV dataset.

        Returns
        -------
        xr.Dataset
            The source TCV dataset
        """
        logger.info(f"Loading source TCV dataset from {self.source_dataset_path}")
        ds = xr.open_dataset(self.source_dataset_path)
        return ds

    def make_raw_data_files(self):
        """Create raw data files from source TCV dataset.

        This method loads the source TCV dataset, standardizes signal names and units,
        applies basic filtering, and saves one netCDF file per shot on a uniform 1 kHz timebase.
        """

        # Load the source dataset
        ds_source = self._load_source_dataset()

        # Filter to requested shots
        available_shots = ds_source["shot"].values.tolist()
        shots_to_process = [shot for shot in self.shotlist if shot in available_shots]

        if len(shots_to_process) < len(self.shotlist):
            missing_shots = set(self.shotlist) - set(shots_to_process)
            logger.warning(
                f"The following shots are not available in source dataset: {missing_shots}"
            )

        processed_shots = 0
        for shot in shots_to_process:
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

            # Extract this shot from the source dataset
            ds_shot = ds_source.sel(shot=shot)

            # Standardize signal names
            ds_standardized = self.standardize_signal_names(ds_shot)
            if ds_standardized is None:
                logger.warning(f"Standardization failed for shot {shot}, skipping")
                continue

            # Ensure on uniform 1 kHz timebase
            max_time = ds_standardized["time"].max().item()
            timebase = make_uniform_1khz_timebase(max_time)
            ds_standardized = ds_standardized.reindex(time=timebase, method="ffill")

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

        # Make a copy to avoid modifying the original
        ds = ds.copy()

        # Simple renames
        rename_map = {
            "RMAG": "R0",
            "KAPPA": "kappa",
            "DELTA_TOP": "delta_top",
            "DELTA_BOTTOM": "delta_bottom",
        }
        # Only rename variables that exist
        rename_map = {k: v for k, v in rename_map.items() if k in ds}
        ds = ds.rename(rename_map)

        # Conversions
        ds["B0"] = np.abs(ds["BZERO"])
        ds["Ip_MA"] = np.abs(ds["I_P"]) * 1e-6  # Convert A to MA
        ds["P_oh_MW"] = ds["POHM"] * 1e-6  # Convert W to MW
        ds["P_rad_MW"] = ds["PradBulk"] * 1e-6  # Convert W to MW
        ds["ne20_line_avg"] = ds["NEavg"] * 1e-20  # Convert m^-3 to 10^20 m^-3
        ds["Wtot_MJ"] = ds["Wtot"] * 1e-6  # Convert J to MJ
        ds["ne20_rho"] = ds["Ne_rho"] * 1e-20  # Convert m^-3 to 10^20 m^-3
        ds["Te_keV_rho"] = ds["Te_rho"] * 1e-3  # Convert eV to keV

        # Handle heating power signals (set to zero if missing, or convert if present)
        if "NBI" in ds:
            ds["P_NBI_MW"] = ds["NBI"].fillna(0.0) * 1e-6
        else:
            ds["P_NBI_MW"] = xr.zeros_like(ds["Ip_MA"])

        if "ECRH" in ds:
            ds["P_ECRH_MW"] = ds["ECRH"].fillna(0.0) * 1e-6
        else:
            ds["P_ECRH_MW"] = xr.zeros_like(ds["Ip_MA"])

        # TCV doesn't have LH or ICRF in this dataset
        ds["P_LH_MW"] = xr.zeros_like(ds["Ip_MA"])
        ds["P_ICRF_MW"] = xr.zeros_like(ds["Ip_MA"])

        # TCV doesn't have tau_conf from standard diagnostics
        ds["tau_conf"] = xr.zeros_like(ds["Ip_MA"])

        # Drop old variable names
        old_vars = [
            "BZERO",
            "I_P",
            "POHM",
            "PradBulk",
            "NEavg",
            "Wtot",
            "Te_rho",
            "Ne_rho",
            "NBI",
            "ECRH",
        ]
        for var in old_vars:
            if var in ds:
                ds = ds.drop_vars(var)

        # If any *important* signal is all NaN, return None to skip this shot
        for signal in ["Te_keV_rho", "ne20_rho", "Ip_MA"]:
            if signal in ds and ds[signal].isnull().all():
                logger.warning(
                    f"Signal {signal} is all NaN for shot {ds['shot'].item() if 'shot' in ds.coords else 'unknown'}, skipping shot."
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
        """Apply TCV specific processing steps.

        This includes:
        - Simple fringe-jump correction for ne20_line_avg
        - Setting data 20ms before disruption to NaN
        - Filtering unrealistic data points

        Parameters
        ----------
        ds : xr.Dataset
            Standardized dataset

        Returns
        -------
        xr.Dataset
            Processed dataset ready for general workflow
        """

        # Simple fringe-jump correction for ne20_line_avg
        # Detect large step changes and remove the offset for the remainder of the trace
        if "ne20_line_avg" in ds:
            ne_values = ds["ne20_line_avg"].values
            if not np.all(np.isnan(ne_values)):
                trace = ne_values.copy()
                diff = np.diff(trace)
                median_abs_diff = np.nanmedian(np.abs(diff))
                jump_threshold = max(0.1, 5.0 * median_abs_diff)

                # Cumulative offset after each detected jump
                offset = 0.0
                corrected = trace.copy()
                for i in range(1, trace.size):
                    if np.isnan(trace[i - 1]) or np.isnan(trace[i]):
                        corrected[i] = trace[i] - offset
                        continue
                    step = trace[i] - trace[i - 1]
                    if np.abs(step) >= jump_threshold:
                        offset += step
                    corrected[i] = trace[i] - offset

                ds["ne20_line_avg"] = (ds["ne20_line_avg"].dims, corrected)

        # Only keep data where ne20_line_avg is above 0.1e20
        if "ne20_line_avg" in ds:
            ds["ne20_line_avg"] = xr.where(
                ds["ne20_line_avg"] > 0.1, ds["ne20_line_avg"], np.nan
            )

        # Only keep data where Wtot is above 1e-3 MJ
        if "Wtot_MJ" in ds:
            ds["Wtot_MJ"] = xr.where(ds["Wtot_MJ"] > 1e-3, ds["Wtot_MJ"], np.nan)

        # Only keep data where Te_keV_rho at rho=1.0 is less than 0.25 keV and above 0.0
        if "Te_keV_rho" in ds and "rho" in ds.coords:
            ds["Te_keV_rho"] = xr.where(
                (ds["Te_keV_rho"].sel(rho=1.0, method="nearest") < 0.25)
                & (ds["Te_keV_rho"].sel(rho=1.0, method="nearest") > 0.0),
                ds["Te_keV_rho"],
                np.nan,
            )

        # Only keep data where ne20_rho at rho=1.0 is less than 0.4e20 and above 0.0
        if "ne20_rho" in ds and "rho" in ds.coords:
            ds["ne20_rho"] = xr.where(
                (ds["ne20_rho"].sel(rho=1.0, method="nearest") < 0.4)
                & (ds["ne20_rho"].sel(rho=1.0, method="nearest") > 0.0),
                ds["ne20_rho"],
                np.nan,
            )

        return ds
