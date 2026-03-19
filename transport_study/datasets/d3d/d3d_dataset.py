"""Makes the 'raw' DIII-D dataset on omega, to be processed later by POPSIM"""

import glob
import io
import os
import tarfile
import tempfile

import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import (
    LogSettings,
    RetrievalSettings,
    TimeSetting,
    TimeSettingParams,
)
from disruption_py.settings.output_setting import DatasetOutputSetting
from disruption_py.settings.time_setting import _postprocess
from disruption_py.workflow import get_shots_data
from freeqdsk import aeqdsk, geqdsk
from loguru import logger

from transport_study import EPISODE_DIM, PACKAGE_ROOT, TIME_COORD, TIME_DIM
from transport_study.config import config
from transport_study.datasets import make_uniform_1khz_timebase
from transport_study.datasets.dispy_utils import summary
from transport_study.datasets.workflow import DataWorkflow


def _clean_aeqdsk_file(f):
    """Clean AEQDSK file by removing invalid text lines that don't fit Fortran format.

    Some AEQDSK files have stray text labels (like "MAG") at the end that are not
    valid floating-point values for Fortran format descriptors. This function filters
    those out by reading all lines, checking if they're valid numeric data, and
    returning a cleaned file-like object.

    Parameters
    ----------
    f : file-like object
        File handle to read from

    Returns
    -------
    io.StringIO
        A file-like object with cleaned content
    """
    content = f.read()
    lines = content.split("\n")
    cleaned_lines = []

    for line in lines:
        # Check if this line contains only whitespace and text (no numbers)
        # Valid FORTRAN formatted lines will have numbers in scientific notation
        stripped = line.strip()
        if stripped:
            # If the line contains valid numeric indicators, keep it
            # Valid indicators: E+, E-, D+, D-, digits, -, +, ., or leading spaces
            if any(
                c in stripped
                for c in [
                    "E",
                    "D",
                    "e",
                    "d",
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "+",
                    "-",
                    ".",
                ]
            ):
                cleaned_lines.append(line)
        else:
            # Keep blank lines as they're part of the format
            cleaned_lines.append(line)

    # Reconstruct the file content and return as StringIO
    cleaned_content = "\n".join(cleaned_lines)
    return io.StringIO(cleaned_content)


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
        # If timebase is much slower than 1 kHz, log a warning
        typical_delta = np.median(np.diff(efit_time))
        if typical_delta > 2:
            logger.critical(
                f"EFIT timebase is much slower than 1 kHz (typical delta: {typical_delta:.3f} ms). This may cause issues with interpolation and data quality."
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
        skip_profiles: bool = False,
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
        skip_profiles : bool
            Whether to skip profile retrieval entirely and only get 0D signals. Default is False.
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
            skip_profiles=skip_profiles,
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
            "rxpt1": {
                "min": 0.5,
                "max": 2,
            },  # Drop shape points where this is clearly wrong (-10)
            "rxpt2": {
                "min": 0.5,
                "max": 2,
            },  # Drop shape points where this is clearly wrong (-10)
            "zxpt1": {
                "min": -4,
                "max": 4,
            },  # Drop shape points where this is clearly wrong (-10)
            "zxpt2": {
                "min": -4,
                "max": 4,
            },  # Drop shape points where this is clearly wrong (-10)
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

        betapf = MdsSignal(r"\betapf", "pedestal", location="remote://atlas.gat.com")
        POHM = MdsSignal(
            r"\pohm", "aot", location="remote://atlas.gat.com"
        )  # Ohmic heating power
        PradBulk = MdsSignal(
            r"\prad_tot", "bolom", location="remote://atlas.gat.com"
        )  # Bulk radiated heating power (total)
        TAU_conf = MdsSignal(
            r"\taue", "transport", location="remote://atlas.gat.com"
        )  # Confinement time [s]
        Ne_edge_avg = MdsSignal(
            r"\denv3", "bci", location="remote://atlas.gat.com"
        ).set_callback(_cm3_to_m3)  # Line average electron density at the edge [m^-3]
        Wtot = MdsSignal(
            r"\wmhdf", "pedestal", location="remote://atlas.gat.com"
        )  # Total stored energy
        BETAP = MdsSignal(
            r"\betapf", "pedestal", location="remote://atlas.gat.com"
        )  # plasma poloidal beta
        BETAN = MdsSignal(
            r"\betan", "pedestal", location="remote://atlas.gat.com"
        )  # Normalized plasma beta

        # Programmed inputs
        iptipp = PtDataSignal("iptipp")
        bttbt = PtDataSignal("bttbt")
        dstdenp = PtDataSignal("dstdenp")
        bmtpwrtar = PtDataSignal("bmtpwrtar")
        ieeseg07 = PtDataSignal("ieeseg07")
        idtrp = PtDataSignal("idtrp")
        idtrxbot = PtDataSignal("idtrxbot")
        idtzxbot = PtDataSignal("idtzxbot")
        idtrxtop = PtDataSignal("idtrxtop")
        idtzxtop = PtDataSignal("idtzxtop")

        # Measured inputs
        # Ip_MA handled
        # B0 handled
        dssneped = PtDataSignal("dssneped")
        # betapf above
        gapin = MdsSignal(r"\gapin", "efit01", location="remote://atlas.gat.com")
        rsurf = MdsSignal(r"\rsurf", "efit01", location="remote://atlas.gat.com")
        rxpt1 = MdsSignal(r"\rxpt1", "efit01", location="remote://atlas.gat.com")
        zxpt1 = MdsSignal(r"\zxpt1", "efit01", location="remote://atlas.gat.com")
        rxpt2 = MdsSignal(r"\rxpt2", "efit01", location="remote://atlas.gat.com")
        zxpt2 = MdsSignal(r"\zxpt2", "efit01", location="remote://atlas.gat.com")

        sigs_dict = {
            "betapf": betapf,
            "p_oh_toksearch": POHM,
            "p_rad_toksearch": PradBulk,
            "tau_conf": TAU_conf,
            "ne_edge_avg": Ne_edge_avg,
            "wmhdf_toksearch": Wtot,
            "betan_toksearch": BETAN,
            # Programmed inputs
            "iptipp": iptipp,
            "bttbt": bttbt,
            "dstdenp": dstdenp,
            "bmtpwrtar": bmtpwrtar,
            "ieeseg07": ieeseg07,
            "idtrp": idtrp,
            "idtrxbot": idtrxbot,
            "idtzxbot": idtzxbot,
            "idtrxtop": idtrxtop,
            "idtzxtop": idtzxtop,
            # Measured inputs
            "betap_toksearch": BETAP,
            "dssneped": dssneped,
            "gapin": gapin,
            "rsurf": rsurf,
            "rxpt1": rxpt1,
            "zxpt1": zxpt1,
            "rxpt2": rxpt2,
            "zxpt2": zxpt2,
        }

        p.fetch_dataset("toksearch", sigs_dict)
        timeline = np.round(np.arange(0, max_time_ms + 1, 1), 0)
        p.align("toksearch", timeline)
        results = p.compute_serial()
        ds_tok = results[0]["toksearch"].squeeze()

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
                output_setting=DatasetOutputSetting(path=False),
                log_settings=LogSettings(file_path=None),
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
            output_setting=DatasetOutputSetting(path=False),
            log_settings=LogSettings(file_path=None),
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
            output_setting=DatasetOutputSetting(path=False),
            log_settings=LogSettings(file_path=None),
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

                if not self.skip_profiles:
                    if self.use_ida:
                        ds_profile = self._get_profile_dataset_ida(shot)
                        if ds_profile is None:
                            logger.info(
                                f"Skipping shot {shot} since IDA profiles are not available and skip_profiles is False"
                            )
                            continue
                    else:
                        ds_profile = self._get_profile_dataset_zipfit(shot)

                ds_fast = self._get_fast_dataset_dispy(shot)
                if ds_fast is None:
                    continue
                ds_0d = self._get_0D_dataset(shot)

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
                ds_assembly = xr.merge(
                    [ds_profile, ds_0d, ds_fast], compat="no_conflicts"
                )

                ds_standardized = self.standardize_signal_names(ds_assembly)
                if ds_standardized is None:
                    logger.warning(f"Standardization failed for shot {shot}, skipping")
                    continue

                ds_standardized.to_netcdf(ds_path)
                logger.info(f"Saved raw dataset for shot {shot} to {ds_path}")
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

        # Simple renames
        ds = ds.rename(
            {
                "aminor": "a_minor",
                "tritop": "delta_top",
                "tribot": "delta_bottom",
                "psi_n": "rho",  # Yeah I know that this mapping isn't exact, I just need *something*
                "betapf": "beta",  # Only for DIII-D profile prediction, using beta feedback TODO(ZanderKeith) ensure this is the actual signal
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
        ds["iptipp_MA"] = np.abs(ds["iptipp"]) / 1e6  # Convert A to MA
        ds["ne20_line_avg"] = ds["n_e"] / 1e20  # Convert m^-3 to 10^20 m^-3
        ds["ne20_edge"] = ds["ne_edge_avg"] / 1e20  # Convert m^-3 to 10^20 m^-3

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
                "beta",  # Poloidal beta from pedestal
                "beta_p",  # Poloidal beta from EFIT
                "R0",
                "B0",
                "Ip_MA",
                "a_minor",
                "kappa",
                "delta_top",
                "delta_bottom",
                "ne20_line_avg",
                "ne20_edge",
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
                # Trajectory optimization on DIII-D
                "iptipp_MA",
                "dstdenp",
                "gapin",
                "gapout",
                "rxpt1",
                "zxpt1",
                "rxpt2",
                "zxpt2",
            ]
        ]

        # If any *important* signal is all NaN, return None to skip this shot
        for signal in ["Te_keV_rho", "ne20_rho", "Ip_MA", "ne20_edge"]:
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

        # Similarly, beta_p (EFIT) is close enough to beta (pedestal)
        if "beta" in ds and "beta_p" in ds:
            beta_missing_or_zero = ds["beta"].isnull() | (ds["beta"] == 0)
            ds["beta"] = ds["beta"].where(~beta_missing_or_zero, other=ds["beta_p"])

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

    def _1kHz_efit(self, shot: int) -> xr.Dataset:
        """Directly parse the saved 1 kHz EFIT results into an xarray dataset for a shot

        We re-computed EFIT01 at 1 kHz, but the DIII-D data curators did not allow us to have a dedicated tree in MDSPlus,
        so we had to put the results under scratch paths like EFIT02-EFIT06 or something
        Of course some shots got overwritten by other researchers using the same scratch paths, so we can't rely on it being consistent.
        So here I'm just going to where we have the results saved and parse the data directly into Xarray
        """

        efit_tgz_path = f"/cscratch/keithz/disruption-efit/2026-03-19/manager/13-16-00.525590/archive/{shot}.tgz"
        if not os.path.exists(efit_tgz_path):
            logger.warning(
                f"EFIT tgz file for shot {shot} not found at {efit_tgz_path}"
            )
            return None

        start_dir = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(efit_tgz_path, "r:gz") as tar:
                tar.extractall(path=tmpdir)
            os.chdir(os.path.join(tmpdir, "efit"))
            a_files = glob.glob("a*")  # Mainly 0D scalars
            g_files = glob.glob("g*")  # Grids and boundaries

            # Ensure they're sorted from low to high
            for files in [g_files, a_files]:
                files.sort()
            if len(a_files) != len(g_files):
                raise ValueError("Inconstent number of EFIT files!")

            datasets = []
            for i, a_file in enumerate(a_files):
                with open(a_file) as f:
                    cleaned_f = _clean_aeqdsk_file(f)
                    a_data = aeqdsk.read(cleaned_f)
                with open(g_files[i]) as f:
                    g_data = geqdsk.read(f)

                # Create dataset for this time slice
                # The only things we're using in this study are as follows:
                # TODO(ZanderKeith): Complete this after you verify they line up
                # major radius
                # minor radius
                # X point R and Z
                data_vars = {
                    "gapin": a_data["oleft"] / 100,  # Inner gap
                    "rxpt1": a_data["rseps1"] / 100,  # Lower X-point R
                    "zxpt1": a_data["zseps1"] / 100,  # Lower X-point Z
                    "rxpt2": a_data["rseps2"] / 100,  # Upper X-point R
                    "zxpt2": a_data["zseps2"] / 100,  # Upper X-point Z
                    "rsurf": a_data["rout"] / 100,  # Geometric major radius
                    "aminor": a_data["rout"] - a_data["rinn"],  # Geometric minor radius
                    "beans": g_data["beans"],  # Elongation
                }

                ds = xr.Dataset(
                    data_vars=data_vars,
                    coords={
                        "time": a_data["time"] / 1e3,  # Convert ms to s
                    },
                )

                # If the x-point coordinates are broken (e.g. -9.9), set them to NaN
                # TODO(ZanderKeith): This should be done in a common location for things obtained from EFIT, too
                valid_xpoint_mask = (ds["rxpt1"] > 0) & (ds["rxpt2"] > 0)
                for var in ["rxpt1", "zxpt1", "rxpt2", "zxpt2"]:
                    ds[var] = ds[var].where(valid_xpoint_mask, other=np.nan)

                datasets.append(ds)

            final_dataset = xr.concat(datasets, dim="time")

        os.chdir(start_dir)
        return final_dataset
