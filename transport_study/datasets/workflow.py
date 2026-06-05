from abc import abstractmethod
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from transport_study import EPISODE_DIM, TIME_DIM
from transport_study.datasets.plotting import ds_profile_plot, ds_profile_time_plot


class DataWorkflow:
    """Class that handles organization of data processing steps

    For this study, the general workflow is:
    1. Create raw data files from source.
    - One file per shot
    - On a common timebase (1 kHz)
    - Standardized signal names
    2. Process and filter data as needed to remove bad shots / fix signals where possible
    - Logging of issues encountered, with plots where relevant to see what went wrong
    3. Combine all shots together into a single xarray Dataset and save to disk
    """

    def __init__(
        self,
        ds_name: str,
        shotlist_file: Path | str | None,
        data_assembly_dir: Path | str,
        max_num_shots: int | None = None,
    ):
        """
        Parameters
        ----------
        ds_name : str
            Name of the dataset ('d3d', 'tcv', 'cmod')
        shotlist_file : Path | str | None
            Path to file containing list of shots to process. If None, will call
            _get_shotlist_from_source() to retrieve shotlist from device-specific source.
        data_assembly_dir : Path | str
            Directory where data files are stored and final dataset will be saved
        max_num_shots : int | None
            Maximum number of shots to process (for testing). If None, process all shots.
        """

        self.ds_name = ds_name
        self.data_assembly_dir = Path(data_assembly_dir)
        self.raw_data_dir = self.data_assembly_dir / ds_name / "raw_data"
        if max_num_shots is None:
            self.final_ds_dir = self.data_assembly_dir / ds_name / "dataset_full"
        else:
            self.final_ds_dir = self.data_assembly_dir / ds_name / f"dataset_{max_num_shots}"

        self.max_num_shots = max_num_shots

        if shotlist_file is None:
            logger.info("No shotlist file provided, retrieving shotlist from device-specific source")
            self.shotlist = self._get_shotlist_from_source()
            logger.info(f"Retrieved {len(self.shotlist)} shots from source")
        else:
            with open(shotlist_file) as f:
                lines = f.readlines()
                self.shotlist = [int(line.strip()) for line in lines if line.strip().isdigit()]
            logger.info(f"Loaded {len(self.shotlist)} shots from {shotlist_file}")

    @abstractmethod
    def _get_shotlist_from_source(self) -> list[int]:
        """Retrieve shotlist from device-specific source.

        This method is called when no shotlist file is provided. Subclasses should
        implement their own logic (e.g., SQL database query, reading from existing dataset).

        Returns
        -------
        list[int]
            List of shot numbers to process
        """

    @abstractmethod
    def make_raw_data_files(self):
        """Create the raw data files by pulling from source

        The resulting files should be one per shot, on a common timebase,
        and have standardized signal names
        """

    @abstractmethod
    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename signals in the dataset to match the POPSIM convention"""

    @abstractmethod
    def device_specific_processing(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply any device-specific processing steps before the general workflow"""

    @abstractmethod
    def device_specific_culling(self, ds: xr.Dataset) -> bool:
        """Apply any device-specific culling logic to determine if this shot should be excluded from the dataset"""

    def log_ds_details(self, ds: xr.Dataset):
        logger.info(f"Final dataset dimensions: {ds.dims}")
        logger.info(f"Final dataset variables: {list(ds.data_vars)}")
        # For each variable, log the maximum value and the shot in which it occurs, to check for any outliers that might indicate issues with the processing
        for var in ds.data_vars:
            # Compute statistics (needed for dask arrays)
            max_per_shot = ds[var].max(dim="time_idx", skipna=True)
            if len(max_per_shot.sizes) > 1:  # Handle multidimensional case
                collapse_dims = [dim for dim in max_per_shot.dims if dim != "shot"]
                max_per_shot = max_per_shot.max(dim=collapse_dims, skipna=True)
            try:
                max_shot_idx = np.nanargmax(max_per_shot.values)
            except ValueError:
                logger.warning(f"Variable {var} has no valid values, skipping stats logging")
                continue
            max_shot = ds["shot"].values[max_shot_idx]
            max_val = max_per_shot.values[max_shot_idx]

            min_per_shot = ds[var].min(dim="time_idx", skipna=True)
            if len(min_per_shot.sizes) > 1:  # Handle multidimensional case
                collapse_dims = [dim for dim in min_per_shot.dims if dim != "shot"]
                min_per_shot = min_per_shot.min(dim=collapse_dims, skipna=True)
            min_shot_idx = np.nanargmin(min_per_shot.values)
            min_shot = ds["shot"].values[min_shot_idx]
            min_val = min_per_shot.values[min_shot_idx]

            mean = ds[var].mean(skipna=True).compute()
            std = ds[var].std(skipna=True).compute()

            logger.info(f"Stats for {var}")
            logger.info(f"  Max is {max_val:.6g} at shot {max_shot}")
            logger.info(f"  Min is {min_val:.6g} at shot {min_shot}")
            logger.info(f"  Mean is {mean:.6g}")
            logger.info(f"  Std is {std:.6g}")

    def run_processed_data_workflow(self):
        """Run the data processing workflow"""
        from popsim.data.dataset_utils import build_tensorized_dataset

        if not int(np.version.version.split(".")[0]) >= 2:
            raise RuntimeError("Numpy version must be greater than 2 to run data processing workflow on all devices.")

        zarr_path = self.final_ds_dir / "ds.zarr"
        if zarr_path.exists():
            print(f"Dataset already exists at {zarr_path}, skipping processing.")
            return

        identifiers = [int(p.stem) for p in self.raw_data_dir.glob("*.nc")]
        if self.max_num_shots:
            identifiers = identifiers[: self.max_num_shots]

        logger.info(f"Processing {len(identifiers)} shots to build dataset")

        # Run the data processing workflow and save to a POPSIM tensorized dataset
        ds = build_tensorized_dataset(
            process_fn=self.process_fn,
            identifiers=identifiers,
            zarr_path=zarr_path,
            time_dim=TIME_DIM,
            episode_dim=EPISODE_DIM,
            extend_existing=False,
            mb_per_chunk=100,
        )

        logger.info(f"Saved processed dataset to {zarr_path}")

        # Log some stats about the resulting dataset
        self.log_ds_details(ds)

        # Make some diagnostic plots of the resulting dataset to check that it looks reasonable. These can be used to spot any remaining issues with the data, and to get a sense of the overall characteristics of the dataset (e.g., typical signal ranges, how many shots have valid profiles, etc.)
        try:
            ds_profile_time_plot(
                zarr_path,
                self.final_ds_dir / "time_traces",
                title=f"{self.ds_name.upper()} Dataset Time Traces",
            )
        except Exception as e:
            logger.error(f"Error generating time trace plots: {e}")
        try:
            ds_profile_plot(
                zarr_path,
                self.final_ds_dir / "profile_traces",
                title=f"{self.ds_name.upper()} Dataset Profile Traces",
            )
        except Exception as e:
            logger.error(f"Error generating profile plots: {e}")

    def filter_ds(self, shot_ds: xr.Dataset) -> xr.Dataset:
        """Apply filtering steps based on device config"""

        # Cut all data 50ms before Ip_MA is NAN to avoid including obviously disruptive data
        last_valid_idx = np.where(shot_ds["Ip_MA"].notnull())[1][-1]
        valid_mask = shot_ds.time <= shot_ds.time[last_valid_idx] - 0.05

        # Apply full-timeslice filters
        for var, valid_range in self.filter_config.items():
            if var in shot_ds:
                var_mask = shot_ds[var].notnull() & (shot_ds[var] >= valid_range["min"]) & (shot_ds[var] <= valid_range["max"])
                valid_mask = valid_mask & var_mask
            else:
                logger.debug(f"Variable {var} specified in filter_config not found in dataset for shot {shot_ds.shot.values[0]}")

        if valid_mask.sum() == 0:
            logger.warning(f"Excluding shot {shot_ds.shot.values[0]} because all data points are invalid after filtering")
            return None

        # Apply individual filters that set out-of-range values to NaN, but don't drop the entire timeslice.
        if self.individual_filter_config is not None:
            for var, valid_range in self.individual_filter_config.items():
                if var in shot_ds:
                    shot_ds[var] = shot_ds[var].where(
                        (shot_ds[var].notnull()) & (shot_ds[var] >= valid_range["min"]) & (shot_ds[var] <= valid_range["max"]),
                        other=np.nan,
                    )
                else:
                    logger.debug(
                        f"Variable {var} specified in individual_filter_config not found in dataset for shot {shot_ds.shot.values[0]}"
                    )

        shot_ds = shot_ds.where(valid_mask, drop=True)
        return shot_ds

    def _debug_plots(self, shot_ds: xr.Dataset):
        debug_fig_dir = self.final_ds_dir / "debug_plots"
        debug_fig_dir.mkdir(parents=True, exist_ok=True)
        ds_profile_time_plot(shot_ds, debug_fig_dir, title=f"Debug: {shot_ds.shot.values[0]}")
        ds_profile_plot(shot_ds, debug_fig_dir, title=f"Debug: {shot_ds.shot.values[0]}")

    def process_fn(self, shot_id: int) -> xr.Dataset:
        raw_ds_path = self.raw_data_dir / f"{shot_id}.nc"
        shot_ds = xr.open_dataset(raw_ds_path)

        # Processing that is specific to the device, implemented in the subclass
        shot_ds = self.device_specific_processing(shot_ds)
        if shot_ds is None:
            logger.warning(f"Skipping shot {shot_id} due to device-specific processing failure")
            return None

        # Processing that is common across devices
        # Ensure powers are non-negative
        power_signals = [sig for sig in shot_ds.data_vars if "P_" in sig and sig.endswith("_MW")]
        for sig in power_signals:
            shot_ds[sig] = shot_ds[sig].clip(min=0)

        # Label where the profiles are fresh (not made by ffill)
        if "fresh_profiles" not in shot_ds:
            ne20 = shot_ds["ne20_psi"]
            ne20_filled = ne20.fillna(0)
            diff_result = ne20_filled != ne20_filled.shift(time_idx=1, fill_value=0)
            first_valid_is_fresh = ne20.notnull().cumsum("time_idx") == 1
            fresh_profiles_1D = diff_result | first_valid_is_fresh
            fresh_profiles = fresh_profiles_1D.any("psi_n")
            shot_ds["fresh_profiles"] = fresh_profiles.astype(np.float32)

        debug_ds = shot_ds.copy()  # Copy for plotting later if need be
        # Filtering based on config thresholds defined in the subclass
        # Making sure data is within valid ranges, and cutting data 50ms before Ip_MA goes to NaN to avoid including disruptive data
        shot_ds = self.filter_ds(shot_ds)
        if shot_ds is None:
            self._debug_plots(debug_ds)
            return None

        # Culling that is specific to the device, implemented in the subclass.
        # This is applied after the device-specific processing and the general processing steps, so that it can take into account any corrections or fixes that were made to the data in those steps.
        if self.device_specific_culling(shot_ds):
            logger.warning(f"Excluding shot {shot_id} based on device-specific culling criteria")
            self._debug_plots(debug_ds)
            return None

        # Culling that is common across devices
        # If shot is too short (less than 500 ms) after processing, exclude it
        cleaned_ds = shot_ds.dropna("time_idx", how="all")
        valid_time_duration = 0 if cleaned_ds.time.size == 0 else float(cleaned_ds.time.max() - cleaned_ds.time.min())
        if valid_time_duration < 0.5:
            logger.warning(f"Excluding shot {shot_id} because duration after processing is only {valid_time_duration:.2f} seconds")
            self._debug_plots(debug_ds)
            return None

        return shot_ds
