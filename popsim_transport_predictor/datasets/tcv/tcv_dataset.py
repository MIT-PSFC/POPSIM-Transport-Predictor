"""Makes the 'raw' TCV dataset, to be processed later by POPSIM"""

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

            # Validate shot has required signals
            ds_validated = self.standardize_signal_names(ds_shot)
            if ds_validated is None:
                logger.warning(f"Validation failed for shot {shot}, skipping")
                continue

            # Determine maximum time from all time coordinates in the dataset
            max_time = 0.0
            for coord_name in ds_validated.coords:
                if coord_name.startswith(("time_", "t_")):
                    coord_max = float(ds_validated[coord_name].max().values)
                    max_time = max(max_time, coord_max)

            if max_time <= 0:
                logger.warning(
                    f"Could not determine valid max time for shot {shot}, skipping"
                )
                continue

            # Create uniform 1 kHz timebase
            timebase = make_uniform_1khz_timebase(max_time)

            # Interpolate all signals onto uniform timebase
            ds_standardized = self._create_uniform_timebase_dataset(
                ds_validated, timebase
            )

            # Rename time_idx coordinate to time for consistency
            if (
                TIME_DIM in ds_standardized.coords
                and TIME_COORD not in ds_standardized.coords
            ):
                ds_standardized = ds_standardized.rename({TIME_DIM: TIME_COORD})

            # Add shot as a dimension (not just coordinate) - required for processing pipeline
            # The processing pipeline expects all data files to have shape (1, time_idx, ...)
            # where the first dimension is 'shot' with size 1
            ds_standardized = ds_standardized.expand_dims(shot=[shot])

            ds_standardized.to_netcdf(ds_path)
            logger.info(f"Saved raw dataset for shot {shot} to {ds_path}")
            processed_shots += 1

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

        # Define signal mappings: {target_name: (source_name, time_coord, unit_conversion_factor)}
        scalar_signals = {
            "B0": ("BZERO", "time_BZERO", lambda x: np.abs(x)),
            "Ip_MA": ("I_P", "time_I_P", lambda x: np.abs(x) * 1e-6),
            "P_oh_MW": ("POHM", "time_POHM", lambda x: x * 1e-6),
            "P_rad_MW": ("PradBulk", "time_PradBulk", lambda x: x * 1e-6),
            "ne20_line_avg": ("NEavg", "time_NEavg", lambda x: x * 1e-20),
            "Wtot_MJ": ("Wtot", "time_Wtot", lambda x: x * 1e-6),
            "R0": ("RMAG", "time_RMAG", lambda x: x),
            "kappa": ("KAPPA", "time_KAPPA", lambda x: x),
            "delta_top": ("DELTA_TOP", "time_DELTA_TOP", lambda x: x),
            "delta_bottom": ("DELTA_BOTTOM", "time_DELTA_BOTTOM", lambda x: x),
        }

        # Interpolate scalar signals
        for target_name, (source_name, time_coord, converter) in scalar_signals.items():
            if source_name in ds and time_coord in ds.coords:
                try:
                    # Get the signal and its time coordinate
                    signal = ds[source_name]

                    # Convert and interpolate
                    signal_converted = converter(signal)
                    signal_interp = signal_converted.interp(
                        {time_coord: timebase},
                        method="linear",
                        kwargs={"fill_value": np.nan},
                    )

                    # Store with new time dimension
                    interp_vars[target_name] = (TIME_DIM, signal_interp.values)
                except Exception as e:
                    logger.warning(f"Failed to interpolate {source_name}: {e}")

        # Handle heating power signals (may not exist)
        for target, source, time_coord in [
            ("P_ECRH_MW", "ECRH", "time_ECRH"),
            ("P_NBI_MW", "NBI", "time_NBI"),
        ]:
            if source in ds and time_coord in ds.coords:
                signal_interp = (ds[source] * 1e-6).interp(
                    {time_coord: timebase},
                    method="linear",
                    kwargs={"fill_value": np.nan},
                )
                interp_vars[target] = (TIME_DIM, signal_interp.values)
            else:
                interp_vars[target] = (TIME_DIM, np.zeros_like(timebase))

        # TCV doesn't have LH or ICRF
        interp_vars["P_LH_MW"] = (TIME_DIM, np.zeros_like(timebase))
        interp_vars["P_ICRF_MW"] = (TIME_DIM, np.zeros_like(timebase))

        # TCV doesn't typically have tau_conf
        interp_vars["tau_conf"] = (TIME_DIM, np.zeros_like(timebase))

        # Handle profile data (Ne_rho, Te_rho)
        # Use rectilinear (nearest-neighbor with forward fill) interpolation for slow diagnostic signals
        if "Ne_rho" in ds and "t_Ne_rho" in ds.coords and "x_Ne_rho" in ds.coords:
            try:
                ne_prof = ds["Ne_rho"]
                ne_rho = ds["x_Ne_rho"]

                # Use nearest-neighbor interpolation (rectilinear) for profile data
                # This is appropriate for slow diagnostics where we want piecewise constant values
                ne_interp = ne_prof.interp(
                    t_Ne_rho=timebase, method="nearest", kwargs={"fill_value": np.nan}
                )
                # Convert units
                ne_interp = ne_interp * 1e-20

                interp_vars["ne20_rho"] = ((TIME_DIM, "rho"), ne_interp.values)
                interp_vars["rho"] = ("rho", ne_rho.values)
            except Exception as e:
                logger.warning(f"Failed to interpolate Ne_rho: {e}")

        if "Te_rho" in ds and "t_Te_rho" in ds.coords and "x_Te_rho" in ds.coords:
            try:
                te_prof = ds["Te_rho"]
                te_rho = ds["x_Te_rho"]

                # Use nearest-neighbor interpolation (rectilinear) for profile data
                # This is appropriate for slow diagnostics where we want piecewise constant values
                te_interp = te_prof.interp(
                    t_Te_rho=timebase, method="nearest", kwargs={"fill_value": np.nan}
                )
                # Convert units (eV to keV)
                te_interp = te_interp * 1e-3

                interp_vars["Te_keV_rho"] = ((TIME_DIM, "rho"), te_interp.values)
                if "rho" not in interp_vars:
                    interp_vars["rho"] = ("rho", te_rho.values)
            except Exception as e:
                logger.warning(f"Failed to interpolate Te_rho: {e}")

        # Create the dataset
        coords = {TIME_DIM: timebase}
        if "rho" in interp_vars:
            coords["rho"] = interp_vars.pop("rho")[1]

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
        # Check for critical signals
        required_signals = ["I_P", "BZERO"]
        for signal in required_signals:
            if signal not in ds:
                logger.warning(f"Critical signal {signal} missing, skipping shot")
                return None

        # Check if profile data exists
        has_ne_profile = "Ne_rho" in ds and "t_Ne_rho" in ds.coords
        has_te_profile = "Te_rho" in ds and "t_Te_rho" in ds.coords

        if not has_ne_profile or not has_te_profile:
            logger.warning(
                f"Missing profile data (Ne_rho: {has_ne_profile}, Te_rho: {has_te_profile}), skipping shot"
            )
            return None

        # Return the dataset as-is - interpolation will happen in make_raw_data_files
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
