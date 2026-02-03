import os
import shutil
from abc import abstractmethod

import xarray as xr
from loguru import logger

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
        processed_data_dir : str
            Directory to save intermediate processed data
        final_ds_dir : str
            Directory to save the final combined dataset
        """

        self.ds_name = ds_name
        self.raw_data_dir = raw_data_dir
        self.final_ds_dir = final_ds_dir

        with open(shotlist_file, "r") as f:
            lines = f.readlines()
            self.shotlist = [int(line.strip()) for line in lines if line.strip().isdigit()]

        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.final_ds_dir, exist_ok=True)

        @abstractmethod
        def make_raw_data_files(self):
            """Create the raw data files by pulling from source"""

        @abstractmethod
        def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset:
            """Rename signals in the dataset to match the POPSIM convention"""

        def run_workflow(self):
            """Run the full data processing workflow"""
            logger.info(f"Starting data workflow for dataset: {self.ds_name}")

            # Step 1: Read raw data
            raw_ds = self.make_raw_data_files()

            # Step 2: Process data to common timebase
            processed_ds = self.process_to_common_timebase(raw_ds)

            # Step 3: Standardize signal names
            standardized_ds = self.standardize_signal_names(processed_ds)

            # Step 4: Filter and clean data
            cleaned_ds = self.clean_data(standardized_ds)

            # Step 5: Save final dataset
            final_ds_path = os.path.join(self.final_ds_dir, f"{self.ds_name}_final.nc")
            cleaned_ds.to_netcdf(final_ds_path)
            logger.info(f"Final dataset saved to {final_ds_path}")