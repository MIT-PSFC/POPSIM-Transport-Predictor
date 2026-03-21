"""Makes the 'raw' DIII-D dataset on omega, to be processed later by POPSIM"""

import os

import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import (
    LogSettings,
    RetrievalSettings,
)
from disruption_py.settings.output_setting import DatasetOutputSetting
from disruption_py.workflow import get_shots_data
from loguru import logger

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.config import config
from transport_study.datasets import make_uniform_1khz_timebase
from transport_study.datasets.d3d.utils import (
    Uniform1kHzTimeSetting,
    compare_densities,
    compare_efits,
    disruption_efit,
)
from transport_study.datasets.dispy_utils import summary
from transport_study.datasets.workflow import DataWorkflow


class D3DDataWorkflow(DataWorkflow):
    """DIII-D specific data workflow for creating and processing datasets.

    This workflow retrieves data from DIII-D's MDSPlus server and IDA/Zipfit
    profile databases, standardizes the signal names, and creates a uniform
    1 kHz timebase dataset suitable for POPSIM transport prediction studies.

    Note: This workflow requires numpy < 2 and access to the DIII-D data servers.
    It cannot be executed on clusters without DIII-D data access.

    POWER BALANCE SIGNALS:
    Wtot_MJ
    Ip_MA
    - measured:
    B0
    R0
    a_minor
    kappa
    ne20_line_avg
    P_aux_MW

    PROFILE PREDICTOR TRAINING SIGNALS:
    Ip_MA
    B0
    betan
    ne20_edge
    R0
    a_minor
    kappa
    delta_top
    delta_bot

    PROFILE PREDICTOR PREDICT-FIRST SIGNALS:
    ne20_psi
    - measured: n_e (ida)
    Te_keV_psi
    - measured: T_e (ida)
    Ip_MA
    - measured: ip (toksearch)
    - programmed: iptipp (toksearch)
    B0
    - measured: bt (toksearch)
    - programmed: bttbt (toksearch)
    betan
    - measured: betanf (toksearch) -> beta_n (dispy)
    - programmed: bmtpwrtar (toksearch)
    ne20_edge
    - measured: dssneped (toksearch) -> denv3
    - programmed: dstdenp (toksearch)
    R0
    - measured: rsurf (dispy)
    - programmed: idtrp (toksearch)
    (kappa, a_minor, delta_top, delta_bot)
    gapin
    - measured: gapin (dispy slow)
    - programmed: ieeseg07 (toksearch)
    rxbot
    - measured: rxpt1 (dispy)
    - programmed: idtrxbot (toksearch)
    zxtop
    - measured: zxpt1 (dispy)
    - programmed: idtzxtop (toksearch)
    rxtop
    - measured: rxpt2 (dispy)
    - programmed: idtrxtop (toksearch)
    rxbot
    - measured: zxpt2 (dispy)
    - programmed: idtzxbot (toksearch)

    OTHER COMPARISON THINGS
    betap
    - measured: betapf
    beta_p from slow EFIT
    beta_n from slow EFIT
    wmhd_MJ
    - wmhd from slow EFIT (just to fill in missing power balance)

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

        # If any of these signals are out of range, drop the entire timeslice
        self.filter_config = {
            "Wtot_MJ": {"min": 0.01, "max": 2},
            "ne20_line_avg": {"min": 0.01, "max": 4},
            "ne20_edge": {"min": 0.01, "max": 4},
        }

        # Set signals outside this range to nan, but don't drop the entire timeslice
        self.individual_filter_config = {
            "P_ECRH_MW": {"min": 0, "max": 10},
            "P_oh_MW": {
                "min": 0,
                "max": 7,
            },  # 201849 P_oh signal goes crazy, remove some of those spikes
            "P_rad_MW": {
                "min": 0,
                "max": 10,
            },  # 201855 also has a crazy P_rad far out of distribution, get rid of it
            # Drop shape points where this is clearly wrong
            "rxpt1": {"min": 0, "max": 3},
            "rxpt2": {
                "min": 0,
                "max": 3,
            },
            "zxpt1": {
                "min": -4,
                "max": 4,
            },
            "zxpt2": {
                "min": -4,
                "max": 4,
            },
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
        import MDSplus as mds
        from toksearch import MdsSignal, Pipeline
        from toksearch.signal.signal import Signal

        class PtDataSignal(Signal):
            """Fetches a PTDATA pointname from atlas.gat.com via MDSplus connection."""

            def __init__(self, pointname: str, server: str = "atlas.gat.com"):
                super().__init__()
                self.pointname = pointname
                self.server = server
                self._connection = None

            def _get_connection(self):
                if self._connection is None:
                    self._connection = mds.Connection(self.server)
                return self._connection

            def gather(self, shot: int) -> dict:
                connection = self._get_connection()
                expr = f'ptdata("{self.pointname}", {shot})'
                data = connection.get(expr).value
                times = connection.get(f"dim_of({expr})").value
                return {"data": data, "times": times}

            def cleanup_shot(self, shot: int):
                pass

            def cleanup(self):
                if self._connection is not None:
                    try:
                        self._connection.disconnect()
                    except Exception:
                        pass
                    self._connection = None

        def _cm3_to_m3(result_dict):
            result_dict["data"] *= 1e6
            result_dict["units"]["data"] = "m^-3"
            return result_dict

        p = Pipeline([shot])

        # POWER BALANCE SIGNALS
        wmhdf = MdsSignal(r"\wmhdf", "pedestal", location="remote://atlas.gat.com")

        # MEASURED SIGNALS
        ip = PtDataSignal("ip")
        bt = PtDataSignal("bt")
        betanf = MdsSignal(r"\betanf", "pedestal", location="remote://atlas.gat.com")
        dssneped = PtDataSignal("dssneped")
        # rsurf, kappa, a_minor, delta_top, delta_bot, gapin, rxpt1, zxpt1, rxpt2, zxpt2 handled by dispy

        # PROGRAMMED INPUTS
        iptipp = PtDataSignal("iptipp")
        bttbt = PtDataSignal("bttbt")
        bmtpwrtar = PtDataSignal("bmtpwrtar")
        dstdenp = PtDataSignal("dstdenp")
        idtrp = PtDataSignal("idtrp")
        ieeseg07 = PtDataSignal("ieeseg07")
        idtrxbot = PtDataSignal("idtrxbot")
        idtzxbot = PtDataSignal("idtzxbot")
        idtrxtop = PtDataSignal("idtrxtop")
        idtzxtop = PtDataSignal("idtzxtop")

        # OTHER
        betapf = MdsSignal(r"\betapf", "pedestal", location="remote://atlas.gat.com")
        ne_line_avg = MdsSignal(
            r"\denv3", "bci", location="remote://atlas.gat.com"
        ).set_callback(_cm3_to_m3)  # Line average electron density at the edge [m^-3]

        sigs_dict = {
            "ip": ip,
            "bt": bt,
            "betanf": betanf,
            "dssneped": dssneped,
            "iptipp": iptipp,
            "bttbt": bttbt,
            "bmtpwrtar": bmtpwrtar,
            "dstdenp": dstdenp,
            "idtrp": idtrp,
            "ieeseg07": ieeseg07,
            "idtrxbot": idtrxbot,
            "idtzxbot": idtzxbot,
            "idtrxtop": idtrxtop,
            "idtzxtop": idtzxtop,
            "betapf": betapf,
            "wmhdf": wmhdf,
            "ne_line_avg": ne_line_avg,
        }

        p.fetch_dataset("toksearch", sigs_dict)
        timeline = np.round(np.arange(0, max_time_ms + 1, 1), 0)
        p.align("toksearch", timeline)
        results = p.compute_serial()
        ds_tok = results[0]["toksearch"].squeeze()

        # Toksearch puts things on a forward fill, we can't have that for training.
        # Replace these signals with nan until their value changes for the first time (e.g. from 0.1 to 0.2)
        for sig in ["betanf", "wmhdf", "dssneped", "ip", "ne_line_avg"]:
            if sig in ds_tok.keys():
                sig_data = ds_tok[sig].values
                first_valid_idx = np.where(sig_data != sig_data[0])[0]
                if len(first_valid_idx) > 0:
                    first_valid_idx = first_valid_idx[0]
                    sig_data[:first_valid_idx] = np.nan
                    ds_tok[sig] = xr.DataArray(
                        sig_data, coords={"time": ds_tok["times"].values}, dims=["time"]
                    )

        # Match disruption-py output
        ds = xr.Dataset(
            data_vars={
                var: (["shot", "time"], ds_tok[var].expand_dims("shot").values)
                for var in ds_tok.data_vars
            },
            coords={
                "shot": np.atleast_1d(
                    shot
                ),  # Problem with xarray https://github.com/pydata/xarray/issues/1709
                "time": ds_tok["times"].values / 1e3,
            },
        )

        # Make everything f32 unless it's an int
        for key in list(ds.data_vars) + list(ds.coords) + list(ds.dims):
            if ds[key].dtype not in [np.float32, np.int64]:
                ds[key] = ds[key].astype(np.float32)

        for sig in sigs_dict.keys():
            if sig not in ds.data_vars:
                logger.warning(
                    f"Signal {sig} not found in TokSearch results for shot {shot}. Filling with NaNs."
                )
                ds[sig] = (
                    ["shot", "time"],
                    np.full_like(ds["time"].values, np.nan, dtype=np.float32).reshape(
                        1, -1
                    ),
                )

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

    def _get_efit_dataset(self, shot: int) -> xr.Dataset:
        """Retrieve EFIT parameters for a shot.

        Parameters
        ----------
        shot : int
            Shot number to retrieve

        Returns
        -------
        xr.Dataset
            Dataset with EFIT parameters on 1 kHz timebase
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
            output_setting=DatasetOutputSetting(path=False),
            log_settings=LogSettings(file_path=None),
            num_processes=1,
        )
        efit_result = efit_result.set_index(idx=["shot", "time"]).unstack("idx")

        disruption_efit_path = f"/fusion/projects/disruption_warning/data/popsim/popsim_studies/profopt/archive/{shot}.tgz"
        if os.path.exists(disruption_efit_path):
            fast_efit_result = disruption_efit(disruption_efit_path, shot)
            fig_dir = os.path.join(
                self.data_assembly_dir, self.ds_name, "raw_data", "debug_figs", "efits"
            )
            os.makedirs(fig_dir, exist_ok=True)
            compare_efits(
                fast_efit_result,
                efit_result,
                shot,
                fig_path=os.path.join(fig_dir, f"{shot}.png"),
            )

            slow_efit_result = efit_result
            efit_result = fast_efit_result
            # recomputation mangles this, fallback to EFIT01
            efit_result["gapin"] = slow_efit_result["gapin"]
            efit_result["beta_n"] = slow_efit_result["beta_n"]
            efit_result["wmhd"] = slow_efit_result["wmhd"]

        valid_xpoint_mask = (efit_result["rxpt1"] > 0) & (efit_result["rxpt2"] > 0)
        for var in ["rxpt1", "zxpt1", "rxpt2", "zxpt2"]:
            efit_result[var] = efit_result[var].where(valid_xpoint_mask, np.nan)

        return efit_result

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

        efit_result = self._get_efit_dataset(shot)

        toksearch_result = self._toksearch_signals(
            shot, max_time_ms=int(efit_result["time"].max().item() * 1e3)
        )

        # Put toksearch result on the same timebase as efit
        toksearch_result = toksearch_result.reindex(
            time=efit_result["time"], method="ffill"
        )

        result = xr.merge(
            [efit_result, toksearch_result],
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
            run_columns=[
                "ne_rho",
                "te_rho",
            ],  # TODO(ZanderKeith): Are these rho or psi?
            only_requested_columns=False,
        )
        profile_result = get_shots_data(
            tokamak=Tokamak.D3D,
            shotlist_setting=shot,
            retrieval_settings=retrieval_settings,
            output_setting=DatasetOutputSetting(path=False),
            log_settings=LogSettings(file_path=None),
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
            logger.warning(f"IDA profile file for shot {shot} not found at {ida_path}")
            return None
        ds = xr.open_dataset(ida_path)
        # Rename profile varaibles to avoid conflict with 0D signals
        ds["Te_psi"] = ds["T_e"]
        ds["ne_psi"] = ds["n_e"]
        ds = ds[["Te_psi", "ne_psi"]]

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
            try:
                if (
                    self.max_num_shots is not None
                    and processed_shots >= self.max_num_shots
                ):
                    logger.info(
                        f"Reached maximum number of shots to process: {self.max_num_shots}"
                    )
                    break

                ds_path = os.path.join(self.raw_data_dir, f"{shot}.nc")
                if os.path.exists(ds_path):
                    logger.info(
                        f"Raw dataset for shot {shot} already exists at {ds_path}"
                    )
                    processed_shots += 1
                    continue

                if self.use_ida:
                    ds_profile = self._get_profile_dataset_ida(shot)
                    if ds_profile is None:
                        logger.info(
                            f"Skipping shot {shot} since IDA profiles are not available and skip_profiles is False"
                        )
                        continue

                ds_0d = self._get_0D_dataset(shot)

                # Put each dataset on a 1 kHz timebase, using previous value fill
                max_time = max(
                    ds_profile["time"].max().item(),
                    ds_0d["time"].max().item(),
                )
                timebase = make_uniform_1khz_timebase(max_time)

                ds_profile = ds_profile.reindex(time=timebase, method="ffill")
                ds_0d = ds_0d.reindex(
                    time=timebase, method="ffill"
                )  # Hold last value to avoid nearest-neighbor jumps at startup
                ds_assembly = xr.merge([ds_profile, ds_0d], compat="no_conflicts")

                ds_standardized = self.standardize_signal_names(ds_assembly)
                if ds_standardized is None:
                    logger.warning(f"Standardization failed for shot {shot}, skipping")
                    continue

                ds_standardized.to_netcdf(ds_path)
                logger.success(f"Saved raw dataset for shot {shot} to {ds_path}")
                processed_shots += 1
            except Exception as e:
                logger.error(f"Error processing shot {shot}: {e}", exc_info=True)
                continue

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

        fig_dir = os.path.join(
            self.data_assembly_dir, self.ds_name, "raw_data", "debug_figs", "densities"
        )
        os.makedirs(fig_dir, exist_ok=True)
        compare_densities(
            ds,
            ds.shot.item(),
            fig_path=os.path.join(fig_dir, f"{ds.shot.item()}.png"),
        )

        # POWER BALANCE TRAINING
        ds["Wtot_MJ"] = ds["wmhdf"] / 1e6
        ds["Wmhd_MJ"] = ds["wmhd"] / 1e6  # Convert J to MJ
        ds["Ip_MA"] = np.abs(ds["ip"]) / 1e6  # Convert A to MA
        ds["B0"] = np.abs(ds["bt"])
        ds["R0"] = ds["rsurf"]
        ds["a_minor"] = ds["aminor"]
        ds["ne20_line_avg"] = ds["ne_line_avg"] / 1e20  # Convert to 10^20 m^-3
        # kappa

        # PROFILE PREDICTOR TRAINING
        ds["Te_keV_psi"] = ds["Te_psi"] / 1e3  # Convert eV to keV
        ds["ne20_psi"] = ds["ne_psi"] / 1e20  # Convert m^-3 to 10^20 m^-3
        # Ip_MA
        # B0
        ds["betan"] = ds["betanf"]  # Already unitless
        ds["ne20_edge"] = ds["dssneped"] / 10  # Convert 10^19 m^-3 to 10^20 m^-3
        # R0
        # a_minor
        # kappa
        ds["delta_top"] = ds["tritop"]
        ds["delta_bot"] = ds["tribot"]

        # PROFILE PREDICTOR PREDICT-FIRST
        # Te_keV_psi
        # Ip_MA
        ds["Ip_MA_prog"] = np.abs(ds["iptipp"]) / 1e6  # Convert A to MA
        # B0
        ds["B0_prog"] = np.abs(ds["bttbt"])
        # betan
        ds["betan_prog"] = ds["bmtpwrtar"]
        # ne20_edge
        ds["ne20_edge_prog"] = ds["dstdenp"] / 10  # Convert 10^19 m^-3 to 10^20 m^-3
        # R0
        ds["R0_prog"] = ds["idtrp"]
        # kappa, a_minor, delta_top, delta_bot
        # gapin
        ds["gapin_prog"] = ds["ieeseg07"]
        ds["rxbot"] = ds["rxpt1"]
        ds["rxbot_prog"] = ds["idtrxbot"]
        ds["zxbot"] = ds["zxpt1"]
        ds["zxbot_prog"] = ds["idtzxbot"]
        ds["rxtop"] = ds["rxpt2"]
        ds["rxtop_prog"] = ds["idtrxtop"]
        ds["zxtop"] = ds["zxpt2"]
        ds["zxtop_prog"] = ds["idtzxtop"]

        # Only keep variables of interest
        kept_vars = {
            # POWER BALANCE TRAINING
            "Wtot_MJ",
            "Ip_MA",
            "ne20_line_avg",
            # PROFILE PREDICTOR PREDICT-FIRST SIGNALS
            "Te_keV_psi",
            "ne20_psi",
            "Ip_MA_prog",
            "B0",
            "B0_prog",
            "betan",
            "betan_prog",
            "ne20_edge",
            "ne20_edge_prog",
            "R0",
            "R0_prog",
            "kappa",
            "a_minor",
            "delta_top",
            "delta_bot",
            "gapin",
            "rxbot",
            "rxbot_prog",
            "zxbot",
            "zxbot_prog",
            "rxtop",
            "rxtop_prog",
            "zxtop",
            "zxtop_prog",
            # OTHER
            "betapf",  # pedestal, obtained from toksearch
            "beta_n",  # EFIT, only from MDSPlus tree (local recomputation mangles)
            "beta_p",  # EFIT
            "betat",  # EFIT
            "Wmhd_MJ",  # From EFIT, to fill in missing Wtot_MJ if need be
        }

        ds = ds[list(kept_vars)]

        # If any *important* signal is all NaN, return None to skip this shot
        for signal in ["Te_keV_psi", "ne20_psi", "Ip_MA", "ne20_edge"]:
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

        # Similarly, we sometimes need to fill in betan (from pedestal) with betan from some other source
        # Our order of preference is as follows:
        # 1: betan from pedestal (betanf from toksearch)
        betan = ds["betan"]
        # 2: beta_n from fast efit (recomputed from betat)
        betan_fast = ds["betat"] * ds["a_minor"] * ds["B0"] / ds["Ip_MA"]
        betan = betan.where(betan.notnull() & (betan > 0), betan_fast)
        # 3: beta_n from slow efit
        betan = betan.where(betan.notnull() & (betan > 0), ds["beta_n"])
        ds["betan"] = betan

        # dssneped is often missing, so fill it in with density from the profile where need be
        ds["ne20_edge"] = ds["ne20_edge"].where(
            ds["ne20_edge"].notnull() & (ds["ne20_edge"] > 0.001),
            ds["ne20_psi"].sel(psi_n=0.9, method="nearest"),
        )

        # Use the smoothed version of P_NBI
        if "P_NBI_MW" in ds and "P_NBI_MW_alt" in ds:
            ds["P_NBI_MW"] = np.abs(ds["P_NBI_MW_alt"])

        return ds
