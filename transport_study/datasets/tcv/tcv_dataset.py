"""Makes the 'raw' TCV dataset, to be processed later by POPSIM"""

import gc
from pathlib import Path
from typing import ClassVar

import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from dynaconf import Dynaconf
from loguru import logger

from transport_study import PACKAGE_ROOT, RADIAL_DIM, TIME_COORD, TIME_DIM
from transport_study.datasets import make_uniform_1khz_timebase
from transport_study.datasets.workflow import RawFileWorkflow
from transport_study.signals import (
    PREDICTION_STORE_NAME,
    STORE_PROFILE_COMPANIONS,
    STORE_SIGNALS,
)

config = Dynaconf(settings_files=[Path(PACKAGE_ROOT) / "datasets/tcv/config.toml"])

TCV_0D_SIGNALS = [
    "I_P",
    "BZERO",
    "DELTA_TOP",
    "DELTA_BOTTOM",
    "ECRH",
    "KAPPA",
    "NBI",
    "NBI2",
    "NEavg",
    "Ne_edge_avg",
    "POHM",
    "PradBulk",
    "R_geom",
    "Wtot",
    "a_minor",
    "BETAN",
]

# Raw DEFUSE signals that standardize_signal_names cannot do without
# A source file missing any of these (partial DEFUSE export) is skipped, not crashed on.
TCV_REQUIRED_RAW_SIGNALS = {
    "I_P",
    "BZERO",
    "POHM",
    "PradBulk",
    "NEavg",
    "Ne_edge_avg",
    "Wtot",
    "R_geom",
    "KAPPA",
    "DELTA_TOP",
    "DELTA_BOTTOM",
    "BETAN",
    "a_minor",
    "Ne_rho",
    "Te_rho",
}

# Edge line-averaged density, kept in the raw files for the edge-to-line-average check only
EDGE_LINE_AVERAGE = "n_e_edge_line_average"

# Smallest density step [m^-3] read as an interferometer fringe jump
FRINGE_JUMP_MIN_M3 = 1e19


class TCVDataWorkflow(RawFileWorkflow):
    """TCV specific data workflow for creating and processing datasets.

    This workflow reads from a pre-existing TCV dataset (e.g., from DEFUSE),
    standardizes signal names, applies filtering and cleaning, and creates
    individual shot files on a uniform 1 kHz timebase suitable for POPSIM
    transport prediction studies.

    DEFUSE has no profile error bars or gradients, so the store leaves the companions out
    and organize_data.add_missing_profile_companions fills them on load.
    DEFUSE profiles are on rho_tor_norm.

    Note: This workflow can execute on the present cluster with TCV data access.
    """

    STORE_VARIABLES: ClassVar[dict[str, tuple[str, ...]]] = {
        PREDICTION_STORE_NAME: tuple(name for name in STORE_SIGNALS if name not in STORE_PROFILE_COMPANIONS)
    }

    def __init__(
        self,
        ds_name: str,
        shotlist_file: Path | str | None,
        data_assembly_dir: Path | str,
        source_dataset_path: Path | str | None = None,
        max_num_shots: int | None = None,
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
        """

        # Use the TCV dataset config from datasets/tcv/config.toml
        self.config = config

        # Dictionary for valid signal ranges for filtering, SI units
        self.filter_config = {
            "energy_mhd": {"min": 1e3, "max": 5e5},
            "n_e_line_average": {"min": 1e18, "max": 4e20},
            EDGE_LINE_AVERAGE: {"min": 1e18, "max": 4e20},
            "power_ec": {"min": 0, "max": 1e7},
            # Bad interferometer data can satisfy the absolute density cap at low ip
            # so stack another check based on the Greenwald fraction
            "greenwald_fraction": {"min": 0.0, "max": 2.0},
            "ip": {"min": 5e4, "max": 5e5},
            "beta_tor_norm": {"min": 0.01, "max": 10},
            # LIUQE geometry moments go nonphysical during the current ramp
            # (minor_radius down to 0.04 m, elongation below 1), which drives derived
            # features like q_star far outside the physical range
            "minor_radius": {"min": 0.15, "max": 0.30},
            "geometric_axis_r": {"min": 0.80, "max": 1.0},
            "elongation": {"min": 0.9, "max": 3.0},
        }
        self.individual_filter_config = None

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
            min_shot_duration=self.config["shot_filters"]["min_duration"],
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
        files = Path(self.source_dir).glob("TCVno*.nc")

        # Extract shot numbers from filenames (TCVno{shot}.nc)
        shotlist = []
        for file in files:
            basename = file.name
            if basename.startswith("TCVno") and basename.endswith(".nc"):
                shot_str = basename[5:-3]  # Extract the number between "TCVno" and ".nc"
                try:
                    shot = int(shot_str)
                    shotlist.append(shot)
                except ValueError:
                    logger.warning(f"Could not extract shot number from filename: {basename}")

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
        shot_file = Path(self.source_dir) / f"TCVno{shot}.nc"
        if not shot_file.exists():
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
                logger.info(f"Reached maximum number of shots to process: {self.max_num_shots}")
                break

            ds_path = Path(self.raw_data_dir) / f"{shot}.nc"
            if ds_path.exists():
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

            if not _profiles_exist(ds_shot):
                logger.warning(f"Shot {shot} is missing profile data, skipping")
                continue

            # Create uniform 1 kHz timebase (max time where I_P is greater than 50 kA)
            valid_ip_mask = np.abs(ds_shot["I_P"]) > 50e3
            max_time = ds_shot["time_I_P"].where(valid_ip_mask, drop=True).max().item()
            timebase = make_uniform_1khz_timebase(max_time)

            # Put signals on uniform timebase with standardized names
            ds_standardized = self._create_uniform_timebase_dataset(ds_shot, timebase)
            ds_standardized = self.standardize_signal_names(ds_standardized)
            if ds_standardized is None:
                logger.warning(f"Shot {shot} failed signal standardization, skipping")
                continue

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

    def _create_uniform_timebase_dataset(self, ds: xr.Dataset, timebase: np.ndarray) -> xr.Dataset:
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

                    interp_data = np.interp(timebase, signal_time, signal_data, left=np.nan, right=np.nan)

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
            if signal in ds and f"t_{signal}" in ds.coords and f"x_{signal}" in ds.coords:
                try:
                    profile_data = ds[signal].values  # Shape: (time_profile, rho_profile)
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
                        dims=[TIME_DIM, RADIAL_DIM],
                        coords={
                            TIME_DIM: np.arange(len(timebase)),
                            TIME_COORD: (TIME_DIM, timebase),
                            RADIAL_DIM: profile_rho,
                        },
                    )

                except Exception as e:
                    logger.warning(f"Failed to interpolate {signal}: {e}")

        # Create the dataset
        # Coords are already set in the DataArrays, so we just need to extract the radial grid if present
        coords = {TIME_DIM: np.arange(len(timebase)), TIME_COORD: (TIME_DIM, timebase)}
        if "Ne_rho" in interp_vars:
            coords[RADIAL_DIM] = interp_vars["Ne_rho"].coords[RADIAL_DIM]
        elif "Te_rho" in interp_vars:
            coords[RADIAL_DIM] = interp_vars["Te_rho"].coords[RADIAL_DIM]

        ds_uniform = xr.Dataset(data_vars=interp_vars, coords=coords)

        return ds_uniform

    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset | None:
        """Convert the uniform-timebase DEFUSE signals into the on-disk schema (IMAS names, SI units).

        Parameters
        ----------
        ds : xr.Dataset
            DEFUSE signals interpolated onto the uniform timebase

        Returns
        -------
        xr.Dataset | None
            Dataset in the on-disk schema plus the processing-only edge density,
            or None if critical signals are missing
        """

        # A source file missing required signals (partial DEFUSE export) is
        # skipped instead of crashing the whole run on a KeyError below
        missing = TCV_REQUIRED_RAW_SIGNALS - set(ds.data_vars)
        if missing:
            logger.warning(f"Missing required raw signals {sorted(missing)}, skipping shot (partial DEFUSE export?)")
            return None

        # DEFUSE is SI except NBI/NBI2/ECRH, which it stores in MW
        ip = abs(ds["I_P"])
        ds_standardized = xr.Dataset(
            {
                "ip": ip,
                "b0": abs(ds["BZERO"]),
                "energy_mhd": ds["Wtot"],
                "beta_tor_norm": ds["BETAN"],
                "n_e_line_average": ds["NEavg"],
                "minor_radius": ds["a_minor"],
                "geometric_axis_r": ds["R_geom"],
                "elongation": ds["KAPPA"],
                "triangularity_upper": ds["DELTA_TOP"],
                "triangularity_lower": ds["DELTA_BOTTOM"],
                "power_ohm": ds["POHM"],
                "power_radiated": ds["PradBulk"],
                "t_e": ds["Te_rho"],
                "n_e": ds["Ne_rho"],
                EDGE_LINE_AVERAGE: ds["Ne_edge_avg"],
            }
        )

        # Auxiliary heating
        # Either beam and ECRH can be absent or (1,) in some shots
        # Missing heating means zero power where the plasma exists, NaN elsewhere.
        mask_valid_ip = ip.notnull()
        power_nbi_MW = xr.zeros_like(ip)
        for beam in ["NBI", "NBI2"]:
            if beam in ds:
                power_nbi_MW = power_nbi_MW + ds[beam].fillna(0.0)
        power_ec_MW = ds["ECRH"].fillna(0.0) if "ECRH" in ds else xr.zeros_like(ip)
        ds_standardized["power_nbi"] = xr.where(mask_valid_ip, power_nbi_MW * 1e6, np.nan)
        ds_standardized["power_ec"] = xr.where(mask_valid_ip, power_ec_MW * 1e6, np.nan)
        # TCV has no ICRF or LH heating systems
        ds_standardized["power_ic"] = xr.where(mask_valid_ip, 0.0, np.nan)
        ds_standardized["power_lh"] = xr.where(mask_valid_ip, 0.0, np.nan)

        # If any *important* signal is all NaN, return None to skip this shot
        if self.has_all_nan_signal(ds_standardized, ["t_e", "n_e", "ip"]):
            return None

        return ds_standardized

    def device_specific_processing(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply TCV specific processing steps.

        This includes:
        - Greenwald fraction for the greenwald_fraction range filter
        - Simple fringe-jump correction for the line-averaged densities

        Parameters
        ----------
        ds : xr.Dataset
            Standardized dataset

        Returns
        -------
        xr.Dataset
            Processed dataset ready for general workflow
        """

        # Greenwald fraction for the range filter
        # n_GW [1e20 m^-3] = ip [MA] / (pi a^2)
        n_e_line_average_1e20 = ds["n_e_line_average"] * 1e-20
        ip_MA = ds["ip"] * 1e-6
        n_greenwald_1e20 = ip_MA / (np.pi * ds["minor_radius"] ** 2)
        ds["greenwald_fraction"] = n_e_line_average_1e20 / n_greenwald_1e20

        # Simple fringe-jump correction for the line-averaged densities
        # Detect large step changes and remove the offset for the remainder of the trace
        for density_var in ["n_e_line_average", EDGE_LINE_AVERAGE]:
            ne_values = ds[density_var].values[0, :]
            if not np.all(np.isnan(ne_values)):
                trace = ne_values.copy()
                diff = np.diff(trace)
                median_abs_diff = np.nanmedian(np.abs(diff))
                jump_threshold = max(FRINGE_JUMP_MIN_M3, 5.0 * median_abs_diff)

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

                ds[density_var] = (
                    ds[density_var].dims,
                    corrected[np.newaxis, :],
                )

        return ds

    def device_specific_culling(self, ds: xr.Dataset) -> bool:
        """Apply TCV-specific culling criteria to the dataset

        Returns True if the dataset should be culled, False otherwise
        """

        sus_shots = [
            85117,  # P_rad consistently higher than P_oh with no aux power (checked raw: no NBI/NBI2/ECRH)
        ]

        if ds.shot.values[0] in sus_shots:
            logger.info(f"Culling shot {ds.shot.values[0]} due to known data issues")
            return True

        power_radiated_mean = ds["power_radiated"].mean().item()
        if power_radiated_mean < 2e4 or ds["power_radiated"].isnull().all():
            logger.info(
                f"Culling shot {ds.shot.values[0]} due to consistently low or missing radiated power measurement (mean power_radiated < 20 kW)"
            )
            return True

        n_e_line_average_mean = ds["n_e_line_average"].mean().item()
        n_e_edge_mean = ds[EDGE_LINE_AVERAGE].mean().item()
        mask_edge_too_high = n_e_edge_mean > (2 * n_e_line_average_mean)
        if mask_edge_too_high or ds["n_e_line_average"].isnull().all() or ds[EDGE_LINE_AVERAGE].isnull().all():
            logger.info(f"Culling shot {ds.shot.values[0]} due to edge density being significantly higher than line-avg density")
            return True

        return False
