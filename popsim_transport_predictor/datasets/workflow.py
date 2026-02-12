import os
from abc import abstractmethod

import numpy as np
import xarray as xr
from loguru import logger

from popsim_transport_predictor import EPISODE_DIM, TIME_DIM


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
        shotlist_file: str | None,
        data_assembly_dir: str,
        max_num_shots: int | None = None,
        skip_profiles: bool | None = False,
    ):
        """
        Parameters
        ----------
        ds_name : str
            Name of the dataset (e.g., 'd3d', 'tcv', 'cmod')
        shotlist_file : str | None
            Path to file containing list of shots to process. If None, will call
            _get_shotlist_from_source() to retrieve shotlist from device-specific source.
        data_assembly_dir : str
            Directory where data files are stored and final dataset will be saved
        max_num_shots : int | None
            Maximum number of shots to process (for testing). If None, process all shots.
        skip_profiles : bool
            If True, skip profile fitting and use zero arrays instead. Useful for testing.
        """

        self.ds_name = ds_name
        self.data_assembly_dir = data_assembly_dir
        self.raw_data_dir = os.path.join(data_assembly_dir, ds_name, "raw_data")
        if max_num_shots is None:
            self.final_ds_dir = os.path.join(
                data_assembly_dir, ds_name, "final_dataset_full"
            )
        else:
            self.final_ds_dir = os.path.join(
                data_assembly_dir, ds_name, f"final_dataset_{max_num_shots}"
            )

        self.max_num_shots = max_num_shots
        self.skip_profiles = skip_profiles

        if shotlist_file is None:
            logger.info(
                "No shotlist file provided, retrieving shotlist from device-specific source"
            )
            self.shotlist = self._get_shotlist_from_source()
            logger.info(f"Retrieved {len(self.shotlist)} shots from source")
        else:
            with open(shotlist_file) as f:
                lines = f.readlines()
                self.shotlist = [
                    int(line.strip()) for line in lines if line.strip().isdigit()
                ]
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

    def log_ds_details(self, ds: xr.Dataset):  # noqa: PLR0912
        logger.info(f"Final dataset dimensions: {ds.dims}")
        logger.info(f"Final dataset variables: {list(ds.data_vars)}")
        # For each variable, log the maximum value and the shot in which it occurs, to check for any outliers that might indicate issues with the processing
        for var in ds.data_vars:
            # Compute statistics (needed for dask arrays)
            max_value = float(ds[var].max().compute())
            min_value = float(ds[var].min().compute())
            mean_value = float(ds[var].mean().compute())
            std_value = float(ds[var].std().compute())

            # Handle shot identification for max/min, accounting for potential NaN values
            max_shot = None
            min_shot = None

            if not np.isnan(max_value):
                try:
                    max_shot_result = ds[var].idxmax(dim=EPISODE_DIM).compute()
                    # Handle different possible return types
                    if hasattr(max_shot_result, "values"):
                        max_shot_val = max_shot_result.values
                    else:
                        max_shot_val = max_shot_result

                    # Extract scalar value safely
                    if np.isscalar(max_shot_val):
                        max_shot = int(max_shot_val)
                    else:
                        max_shot = int(np.asarray(max_shot_val).flat[0])
                except Exception as e:
                    logger.debug(
                        f"Failed to get max shot for {var}: {e}, max_shot_result type: {type(max_shot_result)}"
                    )

            if not np.isnan(min_value):
                try:
                    min_shot_result = ds[var].idxmin(dim=EPISODE_DIM).compute()
                    # Handle different possible return types
                    if hasattr(min_shot_result, "values"):
                        min_shot_val = min_shot_result.values
                    else:
                        min_shot_val = min_shot_result

                    # Extract scalar value safely
                    if np.isscalar(min_shot_val):
                        min_shot = int(min_shot_val)
                    else:
                        min_shot = int(np.asarray(min_shot_val).flat[0])
                except Exception as e:
                    logger.debug(
                        f"Failed to get min shot for {var}: {e}, min_shot_result type: {type(min_shot_result)}"
                    )

            logger.info(f"Variable {var} stats:")
            if np.isnan(max_value):
                logger.info("  Max: NaN (all values are NaN)")
            else:
                logger.info(
                    f"  Max: {max_value:.6g}"
                    + (
                        f" (shot {max_shot})"
                        if max_shot is not None
                        else " (shot unknown)"
                    )
                )

            if np.isnan(min_value):
                logger.info("  Min: NaN (all values are NaN)")
            else:
                logger.info(
                    f"  Min: {min_value:.6g}"
                    + (
                        f" (shot {min_shot})"
                        if min_shot is not None
                        else " (shot unknown)"
                    )
                )

            logger.info(f"  Mean: {mean_value:.6g}")
            logger.info(f"  Std: {std_value:.6g}")

    def run_processed_data_workflow(self):
        """Run the data processing workflow"""
        from popsim.data.dataset_utils import build_tensorized_dataset

        if not int(np.version.version.split(".")[0]) >= 2:
            raise RuntimeError(
                "Numpy version must be greater than 2 to run data processing workflow on all devices."
            )

        zarr_path = os.path.join(self.final_ds_dir, f"{self.ds_name}.zarr")
        if os.path.exists(zarr_path):
            print(f"Dataset already exists at {zarr_path}, skipping processing.")
            return

        identifiers = [
            int(fname.split(".")[0])
            for fname in os.listdir(self.raw_data_dir)
            if fname.endswith(".nc")
        ]
        if self.max_num_shots:
            identifiers = identifiers[: self.max_num_shots]

        # Run the data processing workflow and save to a POPSIM tensorized dataset
        ds = build_tensorized_dataset(
            process_fn=self.process_fn,
            identifiers=identifiers,
            zarr_path=zarr_path,
            time_dim=TIME_DIM,
            episode_dim=EPISODE_DIM,
            extend_existing=False,
            mb_per_chunk=None,
        )

        logger.info(f"Saved processed dataset to {zarr_path}")

        # Log some stats about the resulting dataset
        self.log_ds_details(ds)

    def process_fn(self, shot_id: int) -> xr.Dataset:
        raw_ds_path = os.path.join(self.raw_data_dir, f"{shot_id}.nc")
        shot_ds = xr.open_dataset(raw_ds_path)

        # Apply any processing steps needed. If something breaks, return None to skip this shot.
        shot_ds = self.device_specific_processing(shot_ds)
        if shot_ds is None:
            logger.warning(f"Skipping shot {shot_id} due to processing issues")
            return None

        # Ensure powers are non-negative
        power_signals = [
            sig for sig in shot_ds.data_vars if "P_" in sig and sig.endswith("_MW")
        ]
        for sig in power_signals:
            shot_ds[sig] = shot_ds[sig].clip(min=0)

        ne_valid_mask = (
            shot_ds["ne20_line_avg"].notnull()
            & (shot_ds["ne20_line_avg"] > 0)
            & (shot_ds["ne20_line_avg"] < 4e20)
        )

        wtot_valid_mask = (
            shot_ds["Wtot_MJ"].notnull()
            & (shot_ds["Wtot_MJ"] > 0)
            & (shot_ds["Wtot_MJ"] < 2)
        )

        # Cut all data 50ms before Ip_MA is NAN to avoid including disruptive data
        last_valid_index = np.where(~np.isnan(shot_ds["Ip_MA"]))[1][-1]
        end_of_shot_mask = shot_ds.time <= shot_ds.time[last_valid_index] - 0.05

        valid_mask = ne_valid_mask & end_of_shot_mask & wtot_valid_mask

        shot_ds = shot_ds.where(valid_mask, drop=True)

        # Label where the profiles are fresh (not carried forward by ffill)
        if "fresh_profiles" not in shot_ds:
            shot_ds["fresh_profiles"] = (
                shot_ds["ne20_rho"].diff("time", label="upper").fillna(1.0) > 0
            )

        return shot_ds
