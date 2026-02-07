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
from popsim_transport_predictor.config import config
from popsim_transport_predictor.datasets import make_uniform_1khz_timebase
from popsim_transport_predictor.datasets.dispy_utils import summary
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


class Uniform1MHzTimeSetting(TimeSetting):
    """
    Time setting for creating a uniform timebase at 1 MHz, based on the maximum EFIT time.
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
            times = np.round(np.arange(0, max_time + 1e-6, 1e-6), 6)
            efit_time_unit = "s"
        if params.tokamak == Tokamak.D3D:
            times = np.round(np.arange(0, max_time + 1e-3, 1e-3), 3)
            efit_time_unit = "ms"
        return _postprocess(times=times, units=efit_time_unit)


class D3DDataWorkflow(DataWorkflow):
    """DIII-D specific data workflow for creating and processing datasets.

    This workflow retrieves data from DIII-D's MDSPlus server and IDA/Zipfit
    profile databases, standardizes the signal names, and creates a uniform
    1 kHz timebase dataset suitable for POPSIM transport prediction studies.

    Note: This workflow requires numpy < 2 and access to the DIII-D data servers.
    It cannot be executed on clusters without DIII-D data access.
    """

    def __init__(
        self,
        ds_name: str,
        shotlist_file: str | None,
        data_assembly_dir: str,
        max_num_shots: int | None = None,
        use_ida: bool = True,
    ):
        """Initialize the DIII-D data workflow.

        Parameters
        ----------
        ds_name : str
            Name of the dataset/study, used for directory naming
        shotlist_file : str | None
            Path to file containing list of shots to process. If None, retrieves
            shotlist from SQL database using parameters in config.toml.
        data_assembly_dir : str
            Directory where data files are stored and final dataset will be saved
        max_num_shots : int | None
            Maximum number of shots to process (for testing). If None, process all shots.
        use_ida : bool
            Whether to use IDA for profile data (True) or Zipfit (False). Default is True.
        """

        # Use centralized config
        self.config = config.d3d
        self.use_ida = use_ida

        # Call parent init (which will call _get_shotlist_from_source if needed)
        super().__init__(
            ds_name,
            shotlist_file,
            data_assembly_dir,
            max_num_shots=max_num_shots,
        )

        self.valid_signal_bounds = {
            "Wtot_MJ": (1e-3, None),
        }

    def _get_shotlist_from_source(self) -> list[int]:
        """Retrieve shotlist from DIII-D SQL database.

        Uses the summary() function to query the DIII-D database for shots
        matching the criteria in config.toml.

        Returns
        -------
        list[int]
            List of shot numbers to process
        """
        query_config = self.config["shotlist_query"]

        # Query the SQL database
        data = summary(
            summary_table=query_config["summary_table"],
            ipmax=query_config["ipmax"],
            pulse_length=query_config["pulse_length"],
            min_shot=query_config["min_shot"],
            max_shot=query_config["max_shot"],
            shots=False,
        )
        shotlist = data[:, 0].astype(int).tolist()

        return shotlist

    def _toksearch_signals(self, shot: int, max_time_ms: int) -> xr.Dataset:
        """Retrieve signals from TokSearch that require special handling.

        Originally assembled by Oak Nelson.
        Reference: https://github.com/cfs-energy-internal/POPSIM/blob/datasets_d3d_mast/popsim/data/d3d/d3d_fetch_toksearch_ex.py
        """
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

    def _get_fast_dataset_toksearch(self, shot: int, max_time_ms: int) -> xr.Dataset:
        """Retrieve fast signals from TokSearch with averaging.

        Certain signals on DIII-D (NBI, ECRH) are either too noisy or PWM-modulated,
        so interpolating on a 1ms grid doesn't make sense. This method acquires data
        on the fast timebase and takes the average over the previous 1ms window.

        Parameters
        ----------
        shot : int
            Shot number to retrieve
        max_time_ms : int
            Maximum time in milliseconds

        Returns
        -------
        xr.Dataset
            Dataset with resampled fast signals
        """
        from toksearch import MdsSignal, Pipeline

        p = Pipeline([shot])
        NBI = MdsSignal(r"\pabs", "nb", location="remote://atlas.gat.com")
        ECRH = MdsSignal(r"\pech", "transport", location="remote://atlas.gat.com")

        sigs_dict = {
            "p_nbi_alt": NBI,
            "p_ecrh_alt": ECRH,
        }
        p.fetch_dataset("ds", sigs_dict)
        results = p.compute_serial()
        ds = results[0]["ds"]

        # Resample to 1 kHz by taking the mean over previous 1ms window
        timeline = make_uniform_1khz_timebase(max_time_ms / 1e3)
        n_times = len(timeline)
        times_array = (
            ds["times"].values / 1e3
        )  # Convert to seconds to match timeline units

        # Pre-allocate result arrays for better performance
        resampled_data = {}
        for var in ds.data_vars:
            resampled_data[var] = np.full(n_times, np.nan, dtype=np.float32)

        for i, t in enumerate(timeline):
            mask = (times_array > t - 1e-3) & (times_array <= t)
            if np.any(mask):
                for var in ds.data_vars:
                    var_data = ds[var].values[mask]
                    resampled_data[var][i] = np.nanmean(var_data)

        # Create Dataset with proper dimensions and coordinates
        data_vars = {}
        for var in ds.data_vars:
            data_vars[var] = (["time"], resampled_data[var])

        coords = {"time": timeline, "shot": shot}

        ds_resampled = xr.Dataset(data_vars, coords=coords)
        ds_resampled = ds_resampled.expand_dims("shot")
        return ds_resampled

    def _get_fast_dataset_dispy(self, shot: int) -> xr.Dataset | None:
        """Retrieve fast signals using disruption_py and coarsen to 1 kHz.

        Parameters
        ----------
        shot : int
            Shot number to retrieve

        Returns
        -------
        xr.Dataset | None
            Dataset with coarsened fast signals, or None if retrieval fails
        """
        try:
            retrieval_settings = RetrievalSettings(
                run_columns=["p_nbi", "p_ech", "p_ohm"],
                time_setting=Uniform1MHzTimeSetting(),
                only_requested_columns=True,
            )
            fast_result = get_shots_data(
                tokamak=Tokamak.D3D,
                shotlist_setting=shot,
                retrieval_settings=retrieval_settings,
                num_processes=1,
            )
            fast_result = fast_result.set_index(idx=["shot", "time"]).unstack("idx")

            # Coarsen to 1 kHz by taking the mean over previous 1ms window
            ds_coarse = fast_result.coarsen(time=1000, boundary="trim").mean()

            # Rename signals to _fast
            ds_coarse = ds_coarse.rename(
                {
                    "p_nbi": "p_nbi_fast",
                    "p_ech": "p_ech_fast",
                    "p_ohm": "p_ohm_fast",
                }
            )
        except Exception as e:
            logger.warning(
                f"Failed to get fast dataset for shot {shot} using disruption_py: {e}. Falling back to toksearch."
            )
            return None

        return ds_coarse

    def _get_0D_dataset(self, shot: int) -> xr.Dataset:
        """Retrieve 0D (time-varying scalar) signals for a shot.

        This includes EFIT parameters and global quantities like Ip, Bt, stored energy,
        and heating powers.

        Parameters
        ----------
        shot : int
            Shot number to retrieve

        Returns
        -------
        xr.Dataset
            Dataset with 0D signals on 1 kHz timebase
        """
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
                "p_rad",
                "p_ohm",
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
        """Retrieve profile data from Zipfit.

        Parameters
        ----------
        shot : int
            Shot number to retrieve

        Returns
        -------
        xr.Dataset
            Dataset with electron density and temperature profiles
        """
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
        """Retrieve profile data from IDA (Integrated Data Analysis).

        Parameters
        ----------
        shot : int
            Shot number to retrieve

        Returns
        -------
        xr.Dataset | None
            Dataset with electron density and temperature profiles, or None if file not found
        """
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
        """Create raw data files from source for DIII-D dataset.

        This method retrieves 0D signals from MDSPlus and profiles from IDA or Zipfit,
        combines them on a uniform 1 kHz timebase, standardizes signal names, and saves
        one netCDF file per shot.

        Note: Requires numpy < 2 and access to DIII-D data servers.
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

            ds_fast = self._get_fast_dataset_dispy(shot)
            if ds_fast is None:
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
                ds_fast["time"].max().item(),
            )
            timebase = make_uniform_1khz_timebase(max_time)

            ds_profile = ds_profile.reindex(time=timebase, method="ffill")
            ds_0d = ds_0d.interp(
                time=timebase, method="nearest"
            )  # This should be okay since 0D signal is already on 1 kHz timebase
            ds_fast = ds_fast.interp(
                time=timebase, method="nearest"
            )  # Fast dataset is already on 1 kHz timebase
            ds_assembly = xr.merge([ds_profile, ds_0d, ds_fast], compat="no_conflicts")

            ds_standardized = self.standardize_signal_names(ds_assembly)
            if ds_standardized is None:
                logger.warning(f"Standardization failed for shot {shot}, skipping")
                continue

            ds_standardized.to_netcdf(ds_path)
            logger.info(f"Saved raw dataset for shot {shot} to {ds_path}")
            processed_shots += 1

        logger.info("Finished making raw data files.")

    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset | None:
        """Rename signals in the dataset to match the POPSIM convention.

        This includes unit conversions (e.g., eV to keV, J to MJ) and creating
        derived quantities. Also validates that critical signals are present.

        Parameters
        ----------
        ds : xr.Dataset
            Raw dataset with device-specific signal names

        Returns
        -------
        xr.Dataset | None
            Standardized dataset, or None if critical signals are missing
        """

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

        # Convert all powers to MW
        ds["P_ECRH_MW"] = ds["p_ech"] / 1e6
        ds["P_NBI_MW"] = ds["p_nbi"] / 1e6
        ds["P_NBI_MW_alt"] = ds["p_nbi_fast"] / 1e6
        ds["P_oh_MW"] = ds["p_ohm"] / 1e6
        ds["P_oh_MW_alt"] = ds["p_ohm_fast"] / 1e6
        ds["P_rad_MW"] = ds["p_rad"] / 1e6
        ds["P_rad_MW_alt"] = ds["p_rad_toksearch"] / 1e6
        ds["P_ICRF_MW"] = ds["p_ich"] / 1e6
        ds["P_LH_MW"] = ds["p_lhcd"] / 1e6

        # If tau_conf doesn't exist, replace with 0's like Ip_MA
        if "tau_conf" not in ds:
            ds["tau_conf"] = xr.zeros_like(ds["Ip_MA"])

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
                "P_NBI_MW_alt",
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

    def device_specific_processing(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply DIII-D specific processing steps.

        This includes:
        - Using alternative (less noisy) signals where available
        - Substituting Wmhd for Wtot when Wtot is missing or zero
        - Using smoothed NBI power signal
        - Dropping unnecessary alternative signals

        Parameters
        ----------
        ds : xr.Dataset
            Standardized dataset

        Returns
        -------
        xr.Dataset
            Processed dataset ready for general workflow
        """

        # If alternative radiated power exists, use that (significantly less noisy)
        if (
            "P_rad_MW_alt" in ds
            and not ds["P_rad_MW_alt"].isnull().all()
            and not (ds["P_rad_MW_alt"] == 0).all()
        ):
            ds["P_rad_MW"] = ds["P_rad_MW_alt"]

        # Wtot_MJ is close enough to Wmhd_MJ while being less available
        # Have the Wtot_MJ signal take the Wmhd_MJ values when Wtot_MJ is missing or zero
        if "Wtot_MJ" in ds and "Wmhd_MJ" in ds:
            wtot_missing_or_zero = ds["Wtot_MJ"].isnull() | (ds["Wtot_MJ"] == 0)
            ds["Wtot_MJ"] = ds["Wtot_MJ"].where(
                ~wtot_missing_or_zero, other=ds["Wmhd_MJ"]
            )

        # Use the smoothed version of P_NBI
        ds["P_NBI_MW"] = np.abs(ds["P_NBI_MW_alt"])

        # Drop unnecessary alternative signals
        alt_signals = [
            "P_NBI_MW_alt",
            "P_oh_MW_alt",
            "P_rad_MW_alt",
        ]
        for sig in alt_signals:
            if sig in ds:
                ds = ds.drop_vars(sig)

        return ds
