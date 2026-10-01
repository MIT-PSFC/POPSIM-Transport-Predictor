"""Makes the 'raw' DIII-D dataset on omega, to be processed later by POPSIM"""

import re
from pathlib import Path
from typing import ClassVar

import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings
from disruption_py.settings.output_setting import DatasetOutputSetting
from disruption_py.workflow import get_shots_data
from dynaconf import Dynaconf
from loguru import logger

from transport_study import PACKAGE_ROOT, RADIAL_DIM
from transport_study.datasets.d3d.physics_methods import (
    D3DDatasetMethods,
    Uniform1kHzTimeSetting,
    find_ida_path,
)
from transport_study.datasets.dispy_utils import passive_log_settings
from transport_study.datasets.workflow import RawFileWorkflow
from transport_study.signals import PREDICTION_STORE_NAME, STORE_SIGNALS

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
    "get_pedestal_parameters",  # wmhdf, betanf, dssneped
    "get_programmed_parameters",  # PCS waveforms for predict-first
    "get_extended_efit_parameters",  # betat, aminor, tritop, tribot, rsurf, X points, gapin
    "get_ida_profiles",  # Te/ne on psi_n and rho_tor_norm
]

# Signals only the trajectory optimization reads. TODO(ZanderKeith): IMAS-equivalent names and units?
# The PCS programmed targets (feedforward control) and the measured signals they are compared against.
# They go in their own store, on the same shot / time_idx layout as the prediction store.
TRAJOPT_STORE_NAME = "trajopt"
D3D_TRAJOPT_STORE_SIGNALS = (
    "Ip_MA_prog",
    "B0_prog",
    "betan_prog",
    "ne20_edge_prog",
    "R0_prog",
    "gapin_prog",
    "rxbot_prog",
    "zxbot_prog",
    "rxtop_prog",
    "zxtop_prog",
    "gapin",
    "rxbot",
    "zxbot",
    "rxtop",
    "zxtop",
    "ne20_edge",
    "Te_keV_psi",
    "ne20_psi",
)


class D3DDataWorkflow(RawFileWorkflow):
    """DIII-D specific data workflow for creating and processing datasets.

    All signals come from disruption-py (branch zk/ptps) in a single
    get_shots_data call per shot: 1 kHz EFIT from the DISPY runtag trees,
    PTDATA and pedestal-tree signals via the custom physics methods in
    physics_methods.py, and Te/ne profiles from the IDA database on its own rho_tor_norm.

    Two stores: the prediction store (the shared IMAS schema, SI units)
    and the trajopt store (D3D_TRAJOPT_STORE_SIGNALS),
    which only the trajectory optimization reads.

    Note: raw data fetching requires access to the DIII-D data servers.

    SIGNAL MAP, prediction store (IMAS name: source):
    energy_mhd: wmhdf (pedestal tree), filled from EFIT wmhd where missing
    ip: ip (get_ip_parameters, PTDATA ip)
    b0: btor (PTDATA bt)
    beta_tor_norm: betanf (pedestal, fallbacks efsbetan, betat scaling, EFIT beta_n)
    n_e_line_average: n_e (get_density_parameters)
    geometric_axis_r, minor_radius, elongation, triangularity_upper / _lower:
        EFIT rsurf, aminor, kappa, tritop, tribot
    power_ohm, power_radiated, power_nbi, power_ec: p_ohm, p_rad, p_nbi, p_ech
    t_e, n_e (+_error, _gradient, _gradient_error): IDA profiles on rho_tor_norm

    SIGNAL MAP, trajopt store (standardized name: source):
    Ip_MA_prog: ip_prog (PTDATA iptipp)
    B0_prog, betan_prog, ne20_edge_prog, R0_prog, gapin_prog: PTDATA bttbt, bmtpwrtar, dstdenp, idtrp, ieeseg07
    rx,zx bot,top (+_prog): EFIT rxpt1, zxpt1, rxpt2, zxpt2 / PTDATA idtr,zx...
    gapin: EFIT gapin
    ne20_edge: PTDATA dssneped, filled from ne20_psi at psi_n 0.9 where missing
    Te_keV_psi, ne20_psi: IDA T_e, n_e on the native psi_n grid
    """

    STORE_VARIABLES: ClassVar[dict[str, tuple[str, ...]]] = {
        PREDICTION_STORE_NAME: STORE_SIGNALS,
        TRAJOPT_STORE_NAME: D3D_TRAJOPT_STORE_SIGNALS,
    }

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

        # If any of these signals are out of range, drop the entire timeslice. SI units.
        # Provisional bounds, tuned against the percentile scan of the HBP shots.
        self.filter_config = {
            "energy_mhd": {"min": 1e4, "max": 4e6},
            "ip": {"min": 2e5, "max": 2.5e6},
            "n_e_line_average": {"min": 1e18, "max": 2e20},
            "t_e_axis": {"min": 100, "max": 1.5e4},
            "greenwald_fraction": {"min": 0.0, "max": 2.0},
            "beta_tor_norm": {"min": 0.01, "max": 6},
        }

        # Set signals outside this range to nan, but don't drop the entire timeslice.
        # SI for the prediction store signals, pre-IMAS units for the trajopt ones.
        self.individual_filter_config = {
            "power_ec": {"min": 0, "max": 1e7},
            "power_ohm": {"min": 0, "max": 7e6},  # 201849 P_oh spikes
            "power_radiated": {"min": 0, "max": 1e7},  # 201855 P_rad far out of distribution
            "power_nbi": {"min": 0, "max": 2.5e7},
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
        return result.transpose("shot", "time", "psi_n", RADIAL_DIM, missing_dims="ignore")

    def make_raw_data_files(self):
        """Create raw data files from source for the DIII-D dataset.

        One netCDF file per shot on a uniform 1 kHz timebase with standardized
        signal names. Requires DIII-D data server access.
        """

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
        """Build the on-disk schema (IMAS names, SI units) and the trajopt signals from the raw disruption-py columns.

        Builds a fresh Dataset rather than renaming in place:
        disruption-py's n_e column is the line-averaged density, not the IDA profile.

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
            "betat",
        ]
        missing = [var for var in raw_required if var not in ds]
        if missing:
            logger.warning(f"Shot {ds['shot'].item()}: missing raw signals {missing}, skipping shot.")
            return None

        # PREDICTION STORE, IMAS names in SI units (IDA: T_e in eV, n_e in m^-3)
        signals = {
            "ip": np.abs(ds["ip"]),
            "b0": np.abs(ds["btor"]),
            "energy_mhd": ds["wmhdf"],
            "beta_tor_norm": ds["betanf"],
            "n_e_line_average": ds["n_e"],
            "minor_radius": ds["aminor"],
            "geometric_axis_r": ds["rsurf"],
            "elongation": ds["kappa"],
            "triangularity_upper": ds["tritop"],
            "triangularity_lower": ds["tribot"],
            "power_ohm": ds["p_ohm"],
            "power_radiated": ds["p_rad"],
            "power_nbi": ds["p_nbi"],
            "power_ec": ds["p_ech"],
            "power_ic": xr.zeros_like(ds["p_ohm"]),  # fast wave unused in these campaigns
            "power_lh": xr.zeros_like(ds["p_ohm"]),  # DIII-D has no LHCD
        }
        for src, dst in [("te_rho", "t_e"), ("ne_rho", "n_e")]:
            for src_suffix, dst_suffix in [("", ""), ("_error", "_error"), ("_grad", "_gradient"), ("_grad_error", "_gradient_error")]:
                signals[f"{dst}{dst_suffix}"] = ds[f"{src}{src_suffix}"]

        # PROCESSING ONLY: fallbacks and filter inputs
        signals["energy_mhd_efit"] = ds["wmhd"]
        signals["beta_tor_norm_efit"] = ds["beta_n"]
        signals["beta_tor"] = ds["betat"]  # [%]
        signals["greenwald_fraction"] = ds["greenwald_fraction"]

        # TRAJOPT STORE, pre-IMAS names and units
        signals["Te_keV_psi"] = ds["te_psi"] / 1e3  # Convert eV to keV
        signals["ne20_psi"] = ds["ne_psi"] / 1e20  # Convert m^-3 to 10^20 m^-3
        signals["ne20_edge"] = ds["dssneped"] / 10  # assumes 10^19 m^-3, verified vs ne20_psi(0.9)
        signals["Ip_MA_prog"] = np.abs(ds["ip_prog"]) / 1e6  # Convert A to MA
        signals["B0_prog"] = np.abs(ds["bttbt"])
        signals["betan_prog"] = ds["bmtpwrtar"]
        signals["ne20_edge_prog"] = ds["dstdenp"] / 10  # same unit assumption as dssneped
        signals["R0_prog"] = ds["idtrp"]
        signals["gapin"] = ds["gapin"]
        signals["gapin_prog"] = ds["ieeseg07"]
        signals["rxbot"] = ds["rxpt1"]
        signals["rxbot_prog"] = ds["idtrxbot"]
        signals["zxbot"] = ds["zxpt1"]
        signals["zxbot_prog"] = ds["idtzxbot"]
        signals["rxtop"] = ds["rxpt2"]
        signals["rxtop_prog"] = ds["idtrxtop"]
        signals["zxtop"] = ds["zxpt2"]
        signals["zxtop_prog"] = ds["idtzxtop"]

        ds_standardized = xr.Dataset(signals)

        # If any *important* signal is all NaN, return None to skip this shot
        if self.has_all_nan_signal(ds_standardized, ["t_e", "n_e", "Te_keV_psi", "ne20_psi", "ip"]):
            return None

        return self.standardize_dim_names(ds_standardized)

    def device_specific_processing(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply DIII-D specific processing steps.

        This includes:
        - Substituting the EFIT stored energy where the pedestal-tree one is missing or zero
        - Filling beta_tor_norm from fallback sources
        - Filling ne20_edge from the psi_n profile where the measurement is missing
        - Deriving t_e_axis so filter_config can range-check the profiles

        Parameters
        ----------
        ds : xr.Dataset
            Standardized dataset

        Returns
        -------
        xr.Dataset
            Processed dataset ready for general workflow
        """

        # The pedestal-tree stored energy is close enough to EFIT's while being less available
        # Take the EFIT values where it is missing or zero
        mask_energy_missing = ds["energy_mhd"].isnull() | (ds["energy_mhd"] == 0)
        ds["energy_mhd"] = ds["energy_mhd"].where(~mask_energy_missing, other=ds["energy_mhd_efit"])

        # Similarly, we sometimes need to fill in beta_tor_norm (from pedestal) from some other source
        # Our order of preference is as follows:
        # 1: beta_tor_norm from the pedestal tree (betanf, with efsbetan fallback in the physics method)
        beta_tor_norm = ds["beta_tor_norm"]
        # 2: recomputed from beta_tor [%] with the Troyon normalization, ip in MA
        ip_MA = ds["ip"] * 1e-6
        beta_tor_norm_from_beta_tor = ds["beta_tor"] * ds["minor_radius"] * ds["b0"] / ip_MA
        beta_tor_norm = beta_tor_norm.where(beta_tor_norm.notnull() & (beta_tor_norm > 0), beta_tor_norm_from_beta_tor)
        # 3: beta_tor_norm from EFIT
        beta_tor_norm = beta_tor_norm.where(beta_tor_norm.notnull() & (beta_tor_norm > 0), ds["beta_tor_norm_efit"])
        ds["beta_tor_norm"] = beta_tor_norm

        # dssneped is often missing, so fill it in with density from the profile where need be
        ds["ne20_edge"] = ds["ne20_edge"].where(
            ds["ne20_edge"].notnull() & (ds["ne20_edge"] > 0.001),
            ds["ne20_psi"].sel(psi_n=0.9, method="nearest"),
        )

        # Scalar axis temperature so the shared filter_ds can range-check the profile
        ds["t_e_axis"] = ds["t_e"].sel({RADIAL_DIM: 0}, method="nearest")

        return ds
