"""Makes the 'raw' DIII-D dataset on omega, to be processed later by POPSIM"""

import os

import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings, TimeSetting, TimeSettingParams
from disruption_py.settings.time_setting import _postprocess
from disruption_py.workflow import get_shots_data
from loguru import logger

from popsim_transport_predictor import EPISODE_DIM, PACKAGE_ROOT, TIME_COORD, TIME_DIM
from popsim_transport_predictor.datasets import make_uniform_1khz_timebase
from popsim_transport_predictor.datasets.workflow import DataWorkflow

DEFAULT_SHOTLIST_FILE = os.path.join(
    PACKAGE_ROOT, "datasets", "d3d", "HBP_shotlist_2024"
)


class Uniform1kHzTimeSetting(TimeSetting):
    """
    Time setting for creating a uniform timebase at 1 kHz, based on the maximum EFIT time.
    """

    def _get_times(self, params: TimeSettingParams) -> np.ndarray:
        """
        Parameters
        ----------
        params : TimeSettingParams
            Parameters needed to retrieve the timebase.

        Returns
        -------
        np.ndarray
            Array of times in the timebase.
        """
        (efit_time,) = params.mds_conn.get_dims(
            r"\efit_aeqdsk:ali", tree_name="_efit_tree"
        )

        max_time = np.max(efit_time)
        if params.tokamak == Tokamak.CMOD:
            times = np.round(np.arange(0, max_time + 1e-3, 1e-3), 3)
            efit_time_unit = "s"
        if params.tokamak == Tokamak.D3D:
            times = np.round(np.arange(0, max_time + 1, 1), 0)
            efit_time_unit = "ms"
        return _postprocess(times=times, units=efit_time_unit)


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

    def _toksearch_signals(self, shot: int, max_time_ms: int) -> xr.Dataset:
        # Originally assembled by Oak Nelson here:
        # https://github.com/cfs-energy-internal/POPSIM/blob/datasets_d3d_mast/popsim/data/d3d/d3d_fetch_toksearch_ex.py
        # MANY THANKS TO HIM
        from toksearch import MdsSignal, Pipeline

        p = Pipeline([shot])

        POHM = MdsSignal(
            r"\pohm", "aot", location="remote://atlas.gat.com"
        )  # Ohmic heating power
        PradBulk = MdsSignal(
            r"\prad_tot", "bolom", location="remote://atlas.gat.com"
        )  # Bulk radiated heating power (total)
        TAU_conf = MdsSignal(
            r"\taue", "transport", location="remote://atlas.gat.com"
        )  # Confinement time [s]

        sigs_dict = {
            "p_oh_toksearch": POHM,
            "p_rad_toksearch": PradBulk,
            "tau_conf": TAU_conf,
        }

        p.fetch_dataset("toksearch", sigs_dict)
        timeline = np.round(np.arange(0, max_time_ms + 1, 1), 0)
        p.align("toksearch", timeline)
        results = p.compute_serial()
        ds = results[0]["toksearch"]

        # Match disruption-py output
        ds = ds.rename({"times": "time"})
        ds["time"] = ds["time"] / 1e3

        # Make everything f32 unless it's an int
        for key in list(ds.data_vars) + list(ds.coords) + list(ds.dims):
            if ds[key].dtype not in [np.float32, np.int64]:
                ds[key] = ds[key].astype(np.float32)

        return ds

    def _get_0D_dataset(self, shot: int) -> xr.Dataset:
        retrieval_settings = RetrievalSettings(
            run_methods=["get_efit_parameters"],
            time_setting=Uniform1kHzTimeSetting(),
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
            time_setting=Uniform1kHzTimeSetting(),
            only_requested_columns=True,
        )
        global_result = get_shots_data(
            tokamak=Tokamak.D3D,
            shotlist_setting=shot,
            retrieval_settings=retrieval_settings,
            num_processes=1,
        )
        global_result = global_result.set_index(idx=["shot", "time"]).unstack("idx")

        toksearch_result = self._toksearch_signals(
            shot, max_time_ms=int(efit_result["time"].max().item() * 1e3)
        )

        # Put toksearch result on the same timebase as efit/global
        toksearch_result = toksearch_result.reindex(
            time=efit_result["time"], method="ffill"
        )

        result = xr.merge(
            [efit_result, global_result, toksearch_result],
            compat="override",
            join="exact",
        )

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

            ds_0d = self._get_0D_dataset(shot)

            if self.use_ida:
                ds_profile = self._get_profile_dataset_ida(shot)
                if ds_profile is None:
                    continue
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
            )  # This should be okay since 0D signal is already on 1 kHz timebase
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

        # The module should be using Wtot, but Wmhd should be close enough if Wtot is missing
        ds["Wtot_MJ"] = ds["wmhdf"] / 1e6  # Convert J to MJ
        ds["Wmhd_MJ"] = ds["wmhd"] / 1e6  # Convert J to MJ

        ds["R0"] = ds["rmaxis"]
        ds["B0"] = np.abs(ds["bt"])
        ds["Ip_MA"] = np.abs(ds["ip"]) / 1e6  # Convert A to MA
        ds["ne20_line_avg"] = ds["n_e"] / 1e20  # Convert m^-3 to 10^20 m^-3

        ds["P_ECRH_MW"] = ds["p_ech"] / 1e6  # Convert W to MW
        ds["P_NBI_MW"] = ds["p_nbi"] / 1e6  # Convert W to MW
        ds["P_oh_MW"] = ds["p_ohm"] / 1e6  # Convert W to MW
        ds["P_oh_MW_alt"] = ds["p_oh_toksearch"] / 1e6  # Convert W to MW
        ds["P_rad_MW"] = ds["p_rad"] / 1e6  # Convert W to MW
        ds["P_rad_MW_alt"] = ds["p_rad_toksearch"] / 1e6  # Convert W to MW
        ds["P_ICRF_MW"] = ds["p_ich"] / 1e6  # Convert W to MW
        ds["P_LH_MW"] = ds["p_lhcd"] / 1e6  # Convert W to MW

        # Only keep variables of interest
        ds = ds[
            [
                "Te_keV_rho",
                "ne20_rho",
                "Wtot_MJ",
                "Wmhd_MJ",
                "R0",
                "B0",
                "Ip_MA",
                "a_minor",
                "kappa",
                "delta_top",
                "delta_bottom",
                "ne20_line_avg",
                "P_ECRH_MW",
                "P_NBI_MW",
                "P_oh_MW",
                "P_oh_MW_alt",
                "P_rad_MW",
                "P_rad_MW_alt",
                "P_ICRF_MW",
                "P_LH_MW",
                "tau_conf",
            ]
        ]

        # If any *important* signal is all NaN, return None to skip this shot
        for signal in ["Te_keV_rho", "ne20_rho", "Ip_MA"]:
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
