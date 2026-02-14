"""Makes the 'raw' TCV dataset, to be processed later by POPSIM"""

import gc
import glob
import os

import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from loguru import logger

from popsim_transport_predictor import PACKAGE_ROOT, TIME_COORD, TIME_DIM
from popsim_transport_predictor.config import config
from popsim_transport_predictor.datasets import make_uniform_1khz_timebase
from popsim_transport_predictor.datasets.workflow import DataWorkflow

DEFAULT_SHOTLIST_FILE = os.path.join(PACKAGE_ROOT, "datasets", "tcv", "tcv_shotlist")

TCV_0D_SIGNALS = [
    # Required by the transport predictor module
    "I_P",
    "BZERO",
    "DELTA",
    "DELTA_TOP",
    "DELTA_BOTTOM",
    "ECRH",
    "KAPPA",
    "NBI",
    "NEavg",
    "Ne_edge_avg",
    "POHM",
    "P_LH",
    "PradBulk",
    "RMAG",
    "Wtot",
    "a_minor",
    # Extra things for comparison
    "BETAP",
    "BETAN",
    "TAU_conf_calc",
]


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
        ds_name: str,
        shotlist_file: str | None,
        data_assembly_dir: str,
        source_dataset_path: str | None = None,
        max_num_shots: int | None = None,
        skip_profiles: bool | None = False,
    ):
        """Initialize the TCV data workflow.

        Parameters
        ----------
        ds_name : str
            Name of the dataset/study, used for directory naming
        shotlist_file : str | None
            Path to file containing list of shots to process. If None, retrieves
            all shots from the source directory.
        data_assembly_dir : str
            Directory where data files are stored and final dataset will be saved
        source_dataset_path : str | None
            Path to the directory containing source TCV dataset files. If None, uses default from config.toml.
            Each file should be named like TCVno{shot_number}.nc
        max_num_shots : int | None
            Maximum number of shots to process (for testing). If None, process all shots.
        skip_profiles: bool | None = False,
            If True, skip profile fitting and use zero arrays instead. Useful for testing.
        """

        # Use centralized config
        self.config = config.tcv

        # Dictionary for valid signal ranges for filtering
        self.filter_config = {
            "Wtot_MJ": {"min": 0.001, "max": 0.5},
            "ne20_line_avg": {"min": 0.01, "max": 1.4},
            "ne20_edge": {"min": 0.01, "max": 2},
        }

        # Set source directory path
        if source_dataset_path is not None:
            self.source_dir = source_dataset_path
        else:
            self.source_dir = self.config["data_sources"]["default_source_dir"]

        # Call parent init (which will call _get_shotlist_from_source if needed)
        super().__init__(
            ds_name,
            shotlist_file,
            data_assembly_dir,
            max_num_shots=max_num_shots,
            skip_profiles=skip_profiles,
        )

    def _get_shotlist_from_source(self) -> list[int]:
        """Retrieve shotlist from TCV source directory.

        Lists all .nc files in the source directory and extracts shot numbers
        from filenames with pattern TCVno{shot_number}.nc

        Returns
        -------
        list[int]
            List of shot numbers to process
        """
        logger.info(f"Loading shotlist from TCV source directory {self.source_dir}")

        # List all .nc files in the directory
        files = glob.glob(os.path.join(self.source_dir, "TCVno*.nc"))

        # Extract shot numbers from filenames (TCVno{shot}.nc)
        shotlist = []
        for file in files:
            basename = os.path.basename(file)
            if basename.startswith("TCVno") and basename.endswith(".nc"):
                shot_str = basename[
                    5:-3
                ]  # Extract the number between "TCVno" and ".nc"
                try:
                    shot = int(shot_str)
                    shotlist.append(shot)
                except ValueError:
                    logger.warning(
                        f"Could not extract shot number from filename: {basename}"
                    )

        shotlist.sort()
        logger.info(f"Found {len(shotlist)} shots in source directory")
        return shotlist

    def _load_shot_file(self, shot: int) -> xr.Dataset | None:
        """Load a single TCV shot file from the source directory.

        Parameters
        ----------
        shot : int
            Shot number to load

        Returns
        -------
        xr.Dataset | None
            Dataset for the shot, or None if file not found or cannot be loaded
        """
        shot_file = os.path.join(self.source_dir, f"TCVno{shot}.nc")
        if not os.path.exists(shot_file):
            logger.warning(f"Shot file not found: {shot_file}")
            return None

        try:
            ds = xr.open_dataset(shot_file)
            return ds
        except Exception as e:
            logger.warning(f"Failed to load shot file {shot_file}: {e}")
            return None

    def make_raw_data_files(self):
        """Create raw data files from source TCV dataset files.

        This method loads individual shot files from the source directory, standardizes
        signal names and units, applies basic filtering, and saves one netCDF file per shot
        on a uniform 1 kHz timebase.
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

            # Load this shot's dataset file
            ds_shot = self._load_shot_file(shot)
            if ds_shot is None:
                logger.warning(f"Could not load shot file for shot {shot}, skipping")
                continue

            # If profile data is missing and we're not skipping profiles, skip this shot
            def _profiles_exist(ds_shot):
                if "Ne_rho" in ds_shot:
                    if ds_shot["Ne_rho"].size == 1:
                        return False
                else:
                    return False
                if "Te_rho" in ds_shot:
                    if ds_shot["Te_rho"].size == 1:
                        return False
                else:
                    return False
                return True

            if not self.skip_profiles and not _profiles_exist(ds_shot):
                logger.warning(
                    f"Shot {shot} is missing profile data and skip_profiles is False, skipping"
                )
                continue

            # Create uniform 1 kHz timebase (max time where I_P is greater than 50 kA)
            valid_ip_mask = np.abs(ds_shot["I_P"]) > 50e3
            max_time = ds_shot["time_I_P"].where(valid_ip_mask, drop=True).max().item()
            timebase = make_uniform_1khz_timebase(max_time)

            # Put signals on uniform timebase with standardized names
            ds_standardized = self._create_uniform_timebase_dataset(ds_shot, timebase)
            ds_standardized = self.standardize_signal_names(ds_standardized)

            # Add shot as a dimension (not just coordinate) - required for processing pipeline
            # The processing pipeline expects all data files to have shape (1, time_idx, ...)
            # where the first dimension is 'shot' with size 1
            ds_standardized = ds_standardized.expand_dims(shot=[shot])

            # Make sure data is f32 or int64
            for var in ds_standardized.data_vars:
                if np.issubdtype(ds_standardized[var].dtype, np.floating):
                    ds_standardized[var] = ds_standardized[var].astype(np.float32)
                elif np.issubdtype(ds_standardized[var].dtype, np.integer):
                    ds_standardized[var] = ds_standardized[var].astype(np.int64)
            ds_standardized.to_netcdf(ds_path)
            logger.info(f"Saved raw dataset for shot {shot} to {ds_path}")
            processed_shots += 1
            if processed_shots > 1 and processed_shots % 100 == 0:
                logger.info(f"Processed {processed_shots} shots so far...")
                gc.collect()  # Clean up memory after every 100 shots

        logger.info("Finished making raw data files.")

    def _create_uniform_timebase_dataset(
        self, ds: xr.Dataset, timebase: np.ndarray
    ) -> xr.Dataset:
        """Interpolate all signals from a raw TCV dataset onto a uniform 1 kHz timebase.

        Each signal in the raw TCV dataset has its own time coordinate (e.g., time_I_P for I_P).
        This method interpolates all signals onto a common timebase.

        Parameters
        ----------
        ds : xr.Dataset
            Raw TCV dataset with signal-specific time coordinates
        timebase : np.ndarray
            Target uniform timebase (in seconds)

        Returns
        -------
        xr.Dataset
            Dataset with all signals interpolated onto the common timebase
        """
        # Dictionary to store interpolated variables
        interp_vars = {}

        # Interpolate scalar signals
        for signal in TCV_0D_SIGNALS:
            if signal in ds and f"time_{signal}" in ds.coords:
                try:
                    # Get the signal and its time coordinate
                    signal_data = ds[signal].values
                    signal_time = ds[f"time_{signal}"].values

                    interp_data = np.interp(
                        timebase, signal_time, signal_data, left=np.nan, right=np.nan
                    )

                    interp_da = xr.DataArray(
                        interp_data,
                        dims=[TIME_DIM],
                        coords={
                            TIME_DIM: np.arange(len(timebase)),
                            TIME_COORD: (TIME_DIM, timebase),
                        },
                    )
                    interp_vars[signal] = interp_da
                except Exception as e:
                    logger.warning(f"Failed to interpolate {signal}: {e}")

        # Handle profile data (Ne_rho, Te_rho)
        # Use rectilinear interpolation (forward-fill) for slow diagnostic signals
        for signal in ["Ne_rho", "Te_rho"]:
            if (
                signal in ds
                and f"t_{signal}" in ds.coords
                and f"x_{signal}" in ds.coords
            ):
                try:
                    profile_data = ds[
                        signal
                    ].values  # Shape: (time_profile, rho_profile)
                    profile_time = ds[f"t_{signal}"].values
                    profile_rho = ds[f"x_{signal}"].values

                    # Find indices in profile_time for each timebase point
                    # searchsorted with side='right' gives us the index after each timebase point
                    indices = np.searchsorted(profile_time, timebase, side="right") - 1
                    interp_profile = np.full((len(timebase), len(profile_rho)), np.nan)
                    valid_mask = (indices >= 0) & (indices < len(profile_time))
                    interp_profile[valid_mask, :] = profile_data[indices[valid_mask], :]

                    # Store with proper dimensions
                    interp_vars[signal] = xr.DataArray(
                        interp_profile,
                        dims=[TIME_DIM, "rho"],
                        coords={
                            TIME_DIM: np.arange(len(timebase)),
                            TIME_COORD: (TIME_DIM, timebase),
                            "rho": profile_rho,
                        },
                    )

                except Exception as e:
                    logger.warning(f"Failed to interpolate {signal}: {e}")

        # Create the dataset
        # Coords are already set in the DataArrays, so we just need to extract rho if present
        coords = {TIME_DIM: np.arange(len(timebase)), TIME_COORD: (TIME_DIM, timebase)}
        if "rho" in interp_vars:
            coords["rho"] = interp_vars.pop("rho")[1]
        elif "Ne_rho" in interp_vars or "Te_rho" in interp_vars:
            # Get rho from one of the profile variables
            if "Ne_rho" in interp_vars:
                coords["rho"] = interp_vars["Ne_rho"].coords["rho"]
            elif "Te_rho" in interp_vars:
                coords["rho"] = interp_vars["Te_rho"].coords["rho"]

        ds_uniform = xr.Dataset(data_vars=interp_vars, coords=coords)

        return ds_uniform

    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset | None:
        """Process raw TCV dataset and prepare it for uniform timebase.

        The raw TCV dataset has each signal on its own timebase. This method
        validates that critical signals are present but does NOT yet interpolate
        onto a uniform timebase (that will happen in make_raw_data_files).

        Parameters
        ----------
        ds : xr.Dataset
            Raw dataset with device-specific signal names and per-signal time coordinates

        Returns
        -------
        xr.Dataset | None
            Dataset ready for interpolation, or None if critical signals are missing
        """

        # Simple renames
        ds = ds.rename(
            {
                "RMAG": "R0",
                "KAPPA": "kappa",
                "DELTA_TOP": "delta_top",
                "DELTA_BOTTOM": "delta_bottom",
            }
        )

        # Conversions
        ds["B0"] = np.abs(ds["BZERO"])
        ds["Ip_MA"] = np.abs(ds["I_P"]) * 1e-6
        ds["P_oh_MW"] = ds["POHM"] * 1e-6
        ds["P_rad_MW"] = ds["PradBulk"] * 1e-6
        ds["ne20_line_avg"] = ds["NEavg"] * 1e-20
        ds["ne20_edge"] = ds["Ne_edge_avg"] * 1e-20
        ds["Wtot_MJ"] = ds["Wtot"] * 1e-6
        ds["LH_transition_threshold_MW"] = ds["P_LH"] * 1e-6

        # Profile data may not be available for all shots
        if "Ne_rho" in ds:
            ds["ne20_rho"] = ds["Ne_rho"] * 1e-20
        if "Te_rho" in ds:
            ds["Te_keV_rho"] = ds["Te_rho"] * 1e-3

        # If the signal is not present, create it as zeros up to shape of Ip_MA
        # But where Ip_MA is NaN, keep it NaN
        for new_name, original_name in zip(
            ["P_NBI_MW", "P_ECRH_MW", "P_LH_MW", "P_ICRF_MW"],
            ["NBI", "ECRH", "P_LH_MW", "P_ICRF_MW"],
            strict=True,
        ):
            if original_name not in ds:
                ds[new_name] = xr.where(ds["Ip_MA"].notnull(), 0.0, np.nan)
            else:
                ds[new_name] = xr.where(
                    ds["Ip_MA"].notnull(), ds[original_name].fillna(0.0), np.nan
                )

        # Drop old names
        ds = ds.drop(
            [
                "I_P",
                "BZERO",
                "DELTA",
                "DELTA_TOP",
                "DELTA_BOTTOM",
                "ECRH",
                "KAPPA",
                "NBI",
                "NEavg",
                "Ne_edge_avg",
                "POHM",
                "PradBulk",
                "RMAG",
                "Wtot",
                "Ne_rho",
                "Te_rho",
                "P_LH",
            ],
            errors="ignore",
        )

        return ds

    def device_specific_processing(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply TCV specific processing steps.

        This includes:
        - Simple fringe-jump correction for ne20_line_avg

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
            ne_values = ds["ne20_line_avg"].values[0, :]
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

                ds["ne20_line_avg"] = (
                    ds["ne20_line_avg"].dims,
                    corrected[np.newaxis, :],
                )

        return ds

    def device_specific_culling(self, ds: xr.Dataset) -> bool:
        """Apply TCV-specific culling criteria to the dataset

        Returns True if the dataset should be culled, False otherwise
        """

        sus_shots = [
            85117,  # P_rad consistently higher than P_oh and no other power sources
            83412,  # P_rad consistently higher than P_oh and no other power sources
        ]

        if ds.shot.values[0] in sus_shots:
            logger.info(f"Culling shot {ds.shot.values[0]} due to known data issues")
            return True

        p_rad_avg = ds["P_rad_MW"].mean().item()
        if p_rad_avg < 0.02 or ds["P_rad_MW"].isnull().all():
            logger.info(
                f"Culling shot {ds.shot.values[0]} due to consistently low or missing radiated power measurement (P_rad_MW.mean() < 0.02 MW)"
            )
            return True

        return False
