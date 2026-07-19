"""Makes the 'raw' DIII-D dataset on omega, to be processed later by POPSIM"""

import re
from pathlib import Path

import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings
from disruption_py.settings.output_setting import DatasetOutputSetting
from disruption_py.workflow import get_shots_data
from dynaconf import Dynaconf
from loguru import logger

from transport_study import PACKAGE_ROOT
from transport_study.datasets.d3d.physics_methods import (
    D3DDatasetMethods,
    Uniform1kHzTimeSetting,
    find_ida_path,
)
from transport_study.datasets.dispy_utils import passive_log_settings
from transport_study.datasets.workflow import DataWorkflow

INNER_WALL = 1.05  # Location of the inner wall, used to calculate minor radius from gapin and R0

config = Dynaconf(settings_files=[Path(PACKAGE_ROOT) / "datasets/d3d/config.toml"])

# Everything is fetched through one disruption-py call per shot.
# Built-in methods live in the disruption-py submodule, custom ones in physics_methods.py.
RUN_METHODS = [
    # built-in
    "get_efit_parameters",  # beta_n, beta_p, kappa, wmhd, q95, chisq, ...
    "get_power_parameters",  # p_rad, p_nbi, p_ech
    "get_ohmic_parameters",  # p_ohm
    "get_density_parameters",  # n_e, greenwald_fraction
    "get_ip_parameters",  # ip, ip_prog
    "get_btor",  # btor
    # custom methods from physics_methods.py
    "get_pedestal_parameters",  # wmhdf, betanf, betapf, dssneped
    "get_programmed_parameters",  # PCS waveforms for predict-first
    "get_extended_efit_parameters",  # betat, aminor, tritop, tribot, rsurf, X points, gapin
    "get_ida_profiles",  # Te/ne on psi_n and rho
]


class D3DDataWorkflow(DataWorkflow):
    """DIII-D specific data workflow for creating and processing datasets.

    All signals come from disruption-py (branch zk/ptps) in a single
    get_shots_data call per shot: 1 kHz EFIT from the DISPY runtag trees,
    PTDATA and pedestal-tree signals via the custom physics methods in
    physics_methods.py, and Te/ne profiles from the IDA database mapped onto
    a uniform rho grid using the EFIT equilibrium.

    Note: raw data fetching requires numpy < 2 (MDSplus backend on OMEGA) and
    access to the DIII-D data servers, see make_d3d_venv.sh.
    Processing (--mode process) runs in the main uv venv.

    SIGNAL MAP (standardized name: source):
    Wtot_MJ: wmhdf (pedestal tree), filled from EFIT wmhd where missing
    Ip_MA / Ip_MA_prog: ip / ip_prog (get_ip_parameters, PTDATA ip / iptipp)
    B0 / B0_prog: btor (PTDATA bt) / PTDATA bttbt
    betan / betan_prog: betanf (pedestal, fallbacks efsbetan, betat scaling,
        EFIT beta_n) / PTDATA bmtpwrtar
    ne20_line_avg, fGW: n_e, greenwald_fraction (get_density_parameters)
    ne20_edge / ne20_edge_prog: PTDATA dssneped / dstdenp
    R0 / R0_prog: EFIT rsurf / PTDATA idtrp
    a_minor, kappa, delta_top, delta_bot: EFIT aminor, kappa, tritop, tribot
    gapin / gapin_prog: EFIT gapin / PTDATA ieeseg07
    rx,zx bot,top (+_prog): EFIT rxpt1, zxpt1, rxpt2, zxpt2 / PTDATA idtr,zx...
    P_oh_MW, P_rad_MW, P_NBI_MW, P_ECRH_MW: p_ohm, p_rad, p_nbi, p_ech
    Te_keV_psi, ne20_psi: IDA T_e, n_e on the native psi_n grid
    Te_keV_rho, ne20_rho (+_error, _grad, _grad_error): IDA profiles mapped
        to rho (normalized midplane minor radius, same definition as C-Mod/MAST)
    """

    def __init__(
        self,
        ds_name: str,
        shotlist_file: Path | str | None,
        data_assembly_dir: Path | str,
        max_num_shots: int | None = None,
    ):
        """Initialize the DIII-D data workflow.

        Parameters
        ----------
        ds_name : str
            Name of the dataset/study, used for directory naming
        shotlist_file : str | None
            Path to file containing list of shots to process. If None, uses the
            shots with IDA profile files available.
        data_assembly_dir : str
            Directory where data files are stored and final dataset will be saved
        max_num_shots : int | None
            Maximum number of shots to process (for testing). If None, process all shots.
        """

        # Use the DIII-D dataset config from datasets/d3d/config.toml
        self.config = config

        # Call parent init (which will call _get_shotlist_from_source if needed)
        super().__init__(
            ds_name,
            shotlist_file,
            data_assembly_dir,
            max_num_shots=max_num_shots,
        )

        # If any of these signals are out of range, drop the entire timeslice.
        # Provisional bounds, tuned against the percentile scan of the HBP shots.
        self.filter_config = {
            "Wtot_MJ": {"min": 0.01, "max": 4},
            "Ip_MA": {"min": 0.2, "max": 2.5},
            "ne20_line_avg": {"min": 0.01, "max": 2},
            "Te_keV_core": {"min": 0.1, "max": 15},
            "fGW": {"min": 0.0, "max": 2.0},
            "betan": {"min": 0.01, "max": 6},
        }

        # Set signals outside this range to nan, but don't drop the entire timeslice
        self.individual_filter_config = {
            "P_ECRH_MW": {"min": 0, "max": 10},
            "P_oh_MW": {"min": 0, "max": 7},  # 201849 P_oh spikes
            "P_rad_MW": {"min": 0, "max": 10},  # 201855 P_rad far out of distribution
            "P_NBI_MW": {"min": 0, "max": 25},
            "ne20_edge": {"min": 0, "max": 2},
            "Ip_MA_prog": {"min": 0, "max": 2.5},
            "B0_prog": {"min": 0, "max": 3},
            "betan_prog": {"min": 0, "max": 6},
            "ne20_edge_prog": {"min": 0, "max": 2},
            "R0_prog": {"min": 1.4, "max": 2.0},
            "gapin": {"min": 0, "max": 0.5},
            "gapin_prog": {"min": 0, "max": 0.5},
            # X points, measured and programmed
            "rxbot": {"min": 1.0, "max": 2.0},
            "rxbot_prog": {"min": 1.0, "max": 2.0},
            "rxtop": {"min": 1.0, "max": 2.0},
            "rxtop_prog": {"min": 1.0, "max": 2.0},
            "zxbot": {"min": -1.5, "max": 0.0},
            "zxbot_prog": {"min": -1.5, "max": 0.0},
            "zxtop": {"min": 0.0, "max": 1.5},
            "zxtop_prog": {"min": 0.0, "max": 1.5},
        }

    def _get_shotlist_from_source(self) -> list[int]:
        """Union of shots with an IDA profile file in any configured database."""
        patterns = self.config["data_sources"]["ida_path_patterns"]
        shots: set[int] = set()
        for pattern in patterns:
            pattern_path = Path(str(pattern))
            name_re = re.compile(re.escape(pattern_path.name).replace(re.escape("{shot}"), r"(\d+)"))
            for p in pattern_path.parent.glob(pattern_path.name.format(shot="*")):
                match = name_re.fullmatch(p.name)
                if match:
                    shots.add(int(match.group(1)))
        if not shots:
            raise FileNotFoundError(f"No IDA files found for any pattern in {patterns}")
        return sorted(shots)

    def _get_shot_dataset(self, shot: int) -> xr.Dataset:
        """Fetch every signal for one shot through disruption-py."""
        retrieval_settings = RetrievalSettings(
            run_methods=RUN_METHODS,
            custom_physics_methods=[D3DDatasetMethods],
            time_setting=Uniform1kHzTimeSetting(),
            only_requested_columns=False,
        )
        result = get_shots_data(
            tokamak=Tokamak.D3D,
            shotlist_setting=shot,
            retrieval_settings=retrieval_settings,
            output_setting=DatasetOutputSetting(path=False),
            log_settings=passive_log_settings(),
            num_processes=1,
        )
        result = result.set_index(idx=["shot", "time"]).unstack("idx")
        return result.transpose("shot", "time", "psi_n", "rho", missing_dims="ignore")

    def make_raw_data_files(self):
        """Create raw data files from source for the DIII-D dataset.

        One netCDF file per shot on a uniform 1 kHz timebase with standardized
        signal names. Requires numpy < 2 and DIII-D data server access.
        """

        if not int(np.version.version.split(".")[0]) < 2:
            raise RuntimeError(
                "The MDSplus backend on DIII-D requires numpy < 2, "
                "please use the make_d3d_venv.sh script to create the correct environment."
            )

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

            if find_ida_path(shot) is None:
                logger.info(f"Skipping shot {shot} since IDA profiles are not available")
                continue

            try:
                ds = self._get_shot_dataset(shot)
            except Exception as e:
                logger.error(f"Error retrieving shot {shot}: {e}", exc_info=True)
                continue

            ds_standardized = self.standardize_signal_names(ds)
            if ds_standardized is None:
                logger.warning(f"Standardization failed for shot {shot}, skipping")
                continue

            ds_standardized.to_netcdf(ds_path)
            logger.success(f"Saved raw dataset for shot {shot} to {ds_path}")
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

        raw_required = [
            "wmhdf",
            "wmhd",
            "ip",
            "ip_prog",
            "btor",
            "rsurf",
            "aminor",
            "kappa",
            "n_e",
            "greenwald_fraction",
            "p_ohm",
            "p_rad",
            "p_nbi",
            "p_ech",
            "te_psi",
            "ne_psi",
            "te_rho",
            "te_rho_error",
            "te_rho_grad",
            "te_rho_grad_error",
            "ne_rho",
            "ne_rho_error",
            "ne_rho_grad",
            "ne_rho_grad_error",
            "betanf",
            "betapf",
            "dssneped",
            "tritop",
            "tribot",
            "bttbt",
            "bmtpwrtar",
            "dstdenp",
            "idtrp",
            "ieeseg07",
            "idtrxbot",
            "idtzxbot",
            "idtrxtop",
            "idtzxtop",
            "rxpt1",
            "zxpt1",
            "rxpt2",
            "zxpt2",
            "gapin",
            "beta_n",
            "beta_p",
            "betat",
        ]
        missing = [var for var in raw_required if var not in ds]
        if missing:
            logger.warning(f"Shot {ds['shot'].item()}: missing raw signals {missing}, skipping shot.")
            return None

        # POWER BALANCE TRAINING
        ds["Wtot_MJ"] = ds["wmhdf"] / 1e6
        ds["Wmhd_MJ"] = ds["wmhd"] / 1e6  # Convert J to MJ
        ds["Ip_MA"] = np.abs(ds["ip"]) / 1e6  # Convert A to MA
        ds["B0"] = np.abs(ds["btor"])
        ds["R0"] = ds["rsurf"]
        ds["a_minor"] = ds["aminor"]
        ds["ne20_line_avg"] = ds["n_e"] / 1e20  # Convert to 10^20 m^-3
        ds["fGW"] = ds["greenwald_fraction"]
        ds["P_oh_MW"] = ds["p_ohm"] / 1e6
        ds["P_rad_MW"] = ds["p_rad"] / 1e6
        ds["P_NBI_MW"] = ds["p_nbi"] / 1e6
        ds["P_ECRH_MW"] = ds["p_ech"] / 1e6
        ds["P_ICRF_MW"] = xr.zeros_like(ds["Ip_MA"])  # fast wave unused in these campaigns
        ds["P_LH_MW"] = xr.zeros_like(ds["Ip_MA"])  # DIII-D has no LHCD
        # kappa passes through

        # PROFILES (IDA: T_e in eV, n_e in m^-3)
        ds["Te_keV_psi"] = ds["te_psi"] / 1e3  # Convert eV to keV
        ds["ne20_psi"] = ds["ne_psi"] / 1e20  # Convert m^-3 to 10^20 m^-3
        for src, dst, scale in [("te_rho", "Te_keV_rho", 1e3), ("ne_rho", "ne20_rho", 1e20)]:
            for suffix in ["", "_error", "_grad", "_grad_error"]:
                ds[f"{dst}{suffix}"] = ds[f"{src}{suffix}"] / scale

        # PROFILE PREDICTOR TRAINING
        ds["betan"] = ds["betanf"]  # Already unitless
        ds["ne20_edge"] = ds["dssneped"] / 10  # assumes 10^19 m^-3, verified vs ne20_psi(0.9)
        ds["delta_top"] = ds["tritop"]
        ds["delta_bot"] = ds["tribot"]

        # PROFILE PREDICTOR PREDICT-FIRST
        ds["Ip_MA_prog"] = np.abs(ds["ip_prog"]) / 1e6  # Convert A to MA
        ds["B0_prog"] = np.abs(ds["bttbt"])
        ds["betan_prog"] = ds["bmtpwrtar"]
        ds["ne20_edge_prog"] = ds["dstdenp"] / 10  # same unit assumption as dssneped
        ds["R0_prog"] = ds["idtrp"]
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
            "B0",
            "R0",
            "a_minor",
            "kappa",
            "ne20_line_avg",
            "P_oh_MW",
            "P_rad_MW",
            "P_NBI_MW",
            "P_ECRH_MW",
            "P_ICRF_MW",
            "P_LH_MW",
            # PROFILE PREDICTOR TRAINING
            "Te_keV_rho",
            "Te_keV_rho_error",
            "Te_keV_rho_grad",
            "Te_keV_rho_grad_error",
            "ne20_rho",
            "ne20_rho_error",
            "ne20_rho_grad",
            "ne20_rho_grad_error",
            "betan",
            "ne20_edge",
            "delta_top",
            "delta_bot",
            "fGW",
            # PROFILE PREDICTOR PREDICT-FIRST
            "Te_keV_psi",
            "ne20_psi",
            "Ip_MA_prog",
            "B0_prog",
            "betan_prog",
            "ne20_edge_prog",
            "R0_prog",
            "gapin",
            "gapin_prog",
            "rxbot",
            "rxbot_prog",
            "zxbot",
            "zxbot_prog",
            "rxtop",
            "rxtop_prog",
            "zxtop",
            "zxtop_prog",
            # FALLBACK SOURCES
            "betapf",  # pedestal
            "beta_n",  # EFIT
            "beta_p",  # EFIT
            "betat",  # EFIT
            "Wmhd_MJ",  # EFIT, fills in missing Wtot_MJ
        }

        ds = ds[sorted(kept_vars)]

        # If any *important* signal is all NaN, return None to skip this shot
        if self.has_all_nan_signal(ds, ["Te_keV_rho", "ne20_rho", "Te_keV_psi", "ne20_psi", "Ip_MA"]):
            return None

        return self.standardize_dim_names(ds)

    def device_specific_processing(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply DIII-D specific processing steps.

        This includes:
        - Substituting Wmhd for Wtot when Wtot is missing or zero
        - Filling betan from fallback sources
        - Filling ne20_edge from the profile where the measurement is missing
        - Deriving Te_keV_core so filter_config can range-check the profiles

        Parameters
        ----------
        ds : xr.Dataset
            Standardized dataset

        Returns
        -------
        xr.Dataset
            Processed dataset ready for general workflow
        """

        # Wtot_MJ is close enough to Wmhd_MJ while being less available
        # Have the Wtot_MJ signal take the Wmhd_MJ values when Wtot_MJ is missing or zero
        wtot_missing_or_zero = ds["Wtot_MJ"].isnull() | (ds["Wtot_MJ"] == 0)
        ds["Wtot_MJ"] = ds["Wtot_MJ"].where(~wtot_missing_or_zero, other=ds["Wmhd_MJ"])

        # Similarly, we sometimes need to fill in betan (from pedestal) with betan from some other source
        # Our order of preference is as follows:
        # 1: betan from the pedestal tree (betanf, with efsbetan fallback in the physics method)
        betan = ds["betan"]
        # 2: beta_n recomputed from betat
        betan_fast = ds["betat"] * ds["a_minor"] * ds["B0"] / ds["Ip_MA"]
        betan = betan.where(betan.notnull() & (betan > 0), betan_fast)
        # 3: beta_n from EFIT
        betan = betan.where(betan.notnull() & (betan > 0), ds["beta_n"])
        ds["betan"] = betan

        # dssneped is often missing, so fill it in with density from the profile where need be
        ds["ne20_edge"] = ds["ne20_edge"].where(
            ds["ne20_edge"].notnull() & (ds["ne20_edge"] > 0.001),
            ds["ne20_psi"].sel(psi_n=0.9, method="nearest"),
        )

        # Scalar core temperature so the shared filter_ds can range-check the profile
        ds["Te_keV_core"] = ds["Te_keV_rho"].sel(rho=0, method="nearest")

        return ds
