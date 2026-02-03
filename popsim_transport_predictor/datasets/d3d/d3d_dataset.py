"""Makes the 'raw' DIII-D dataset on omega, to be processed later by POPSIM"""

import os

import fire
from loguru import logger
import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings
from disruption_py.workflow import get_shots_data

from popsim_transport_predictor.datasets import make_uniform_1khz_timebase
from popsim_transport_predictor.datasets.d3d import (
    D3D_DATASET_SIGNALS,
    HP_SHOTLIST,
    IPMAX,
    MAX_SHOT,
    MIN_SHOT,
    PULSE_LENGTH,
    SUMMARY_TABLE,
)
from popsim_transport_predictor.datasets.dispy_utils import summary

from popsim_transport_predictor.datasets.workflow import DataWorkflow

class D3DDataWorkflow(DataWorkflow):

    def __init__(
        self,
        ds_name: str,
        shotlist_file: str,
        raw_data_dir: str,
        final_ds_dir: str,
        use_ida: bool = True,
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
        use_ida : bool
            Whether to use IDA for profile data (True) or Zipfit (False)
        """

        super().__init__(
            ds_name,
            shotlist_file,
            raw_data_dir,
            final_ds_dir,
        )
        self.use_ida = use_ida
            
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
            run_columns=["ip", "bt", "wmhdf", "betapf", "n_e", "p_ohm", "p_rad", "p_nbi", "p_ech"],
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
    
    def _get_profile_dataset_ida(self, shot: int) -> xr.Dataset:
        ida_path = f"/fusion/projects/results/ida-results/HBP_database/IDA_{shot}_.cdf"
        ds = xr.open_dataset(ida_path)
        return ds

    def make_raw_data_files(self):
        """Create raw data files from source for DIII-D dataset
        
        We are getting 0D signals from MDSPlus and using profiles from IDA or Zipfit.
        """

        for shot in self.shotlist:
            ds_0d = self._get_0D_dataset(shot)
            if self.use_ida:
                ds_profile = self._get_profile_dataset_ida(shot)
            else:
                ds_profile = self._get_profile_dataset_zipfit(shot)

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

            ds_path = os.path.join(self.raw_data_dir, f"{shot}.nc")
            ds_standardized.to_netcdf(ds_path)
            logger.info(f"Saved raw dataset for shot {shot} to {ds_path}")

    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename signals in the dataset to match the POPSIM convention"""

        rename_dict = {
            "TE_RHO": "te_rho",
            "NE_RHO": "ne_rho",
            "IP": "ip",
            "BT0": "bt",
            "WMHDF": "wmhdf",
            "BETAPF": "betapf",
            "N_E": "n_e",
            "A_MINOR": "aminor",
            "KAPPA": "kappa",
            "TRITOP": "tritop",
            "TRIBOT": "tribot",
            "R0": "R0",
            "P_OHM": "p_ohm",
            "P_RAD": "p_rad",
            "P_NBI": "p_nbi",
            "P_ECH": "p_ech",
        }
        ds_renamed = ds.rename(rename_dict)
        return ds_renamed