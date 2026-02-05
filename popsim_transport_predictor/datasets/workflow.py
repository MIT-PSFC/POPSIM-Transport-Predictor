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
        shotlist_file: str,
        raw_data_dir: str,
        final_ds_dir: str,
        max_num_shots: int | None = None,
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
        """

        self.ds_name = ds_name
        self.raw_data_dir = raw_data_dir
        self.final_ds_dir = final_ds_dir
        self.max_num_shots = max_num_shots

        with open(shotlist_file) as f:
            lines = f.readlines()
            self.shotlist = [
                int(line.strip()) for line in lines if line.strip().isdigit()
            ]

        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.final_ds_dir, exist_ok=True)

    @abstractmethod
    def make_raw_data_files(self):
        """Create the raw data files by pulling from source

        The resulting files should be one per shot, on a common timebase,
        and have standardized signal names
        """

    @abstractmethod
    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename signals in the dataset to match the POPSIM convention"""

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
        build_tensorized_dataset(
            process_fn=self.process_fn,
            identifiers=identifiers,
            zarr_path=zarr_path,
            time_dim=TIME_DIM,
            episode_dim=EPISODE_DIM,
            extend_existing=False,
            mb_per_chunk=None,
        )

        logger.info(f"Saved processed dataset to {zarr_path}")

    def process_fn(self, shot_id: int) -> xr.Dataset:
        raw_ds_path = os.path.join(self.raw_data_dir, f"{shot_id}.nc")
        shot_ds = xr.open_dataset(raw_ds_path)

        # Sometimes wmhdf was missing, ensure that it exists / is in a valid range

        # Apply any processing steps needed. If something breaks, return None to skip this shot.

        return shot_ds
