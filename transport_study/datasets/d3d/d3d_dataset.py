"""Makes the 'raw' DIII-D dataset on omega, to be processed later by POPSIM"""

import json
import warnings
from collections import Counter
from pathlib import Path
from typing import ClassVar

import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
import zarr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings
from disruption_py.settings.output_setting import DatasetOutputSetting
from disruption_py.workflow import get_shots_data
from loguru import logger
from transport_validation_datasets.dispy_utils import passive_log_settings
from transport_validation_datasets.machine.generic import MU0
from transport_validation_datasets.store_schema import STORE_SIGNALS
from zarr.errors import ZarrUserWarning

from transport_study import EPISODE_DIM, RADIAL_DIM, TIME_COORD
from transport_study.datasets.d3d import config
from transport_study.datasets.d3d.physics_methods import (
    D3DDatasetMethods,
    DispyEfitNicknameSetting,
    Uniform1kHzTimeSetting,
)
from transport_study.datasets.d3d.profiles import (
    PSI_NORM_DIM,
    RHO_TOR_NORM_DEFINITION,
    find_ida_path,
    find_ida_shots,
)
from transport_study.datasets.workflow import RawFileWorkflow
from transport_study.signals import PREDICTION_STORE_NAME

INNER_WALL = 1.05  # Location of the inner wall, used to calculate minor radius from gapin and R0
# Major radius EFIT gives the vacuum toroidal field at (rcentr), the store's r0 [m]
R0 = 1.6955
# b0 = mu0 N I / (2 pi R0) of the N = 144 turn TF coil from its current bcoil [A], the EFIT bcentr formula.
# PTDATA bt is not used, it reads 2.4-2.8 percent above this at a reference radius nowhere documented.
TF_COIL_TURNS = 144
BCOIL_TO_B0 = MU0 * TF_COIL_TURNS / (2 * np.pi * R0)

# Everything is fetched through one disruption-py call per shot, all EFIT signals from the shot's DISPY run
RUN_METHODS = [
    # custom methods from physics_methods.py, every one placed on the timebase causally (signal_on_grid) rather than interpolated
    "get_plasma_current",  # ip, ip_prog
    "get_efit_scalars",  # wmhd, beta_n
    "get_ptdata_parameters",  # bcoil, dssneped, PCS programmed waveforms
    "get_line_average_density",  # n_e_line_average
    "get_ohmic_power",  # p_ohm
    "get_radiated_power",  # p_rad
    "get_heating_powers",  # p_nbi, p_ech
    "get_boundary_parameters",  # aminor, rsurf, kappa, tritop, tribot
    "get_xpoint_gap_parameters",  # gapin, X points
    "get_ida_profiles",  # Te/ne on rho_tor_norm and psi_norm
]

# Store signal -> (raw disruption-py column, factor to SI units)
PREDICTION_SOURCES = {
    "ip": ("ip", 1.0),
    "b0": ("bcoil", BCOIL_TO_B0),
    "energy_mhd": ("wmhd", 1.0),
    "beta_tor_norm": ("beta_n", 1.0),
    "n_e_line_average": ("n_e_line_average", 1.0),
    "minor_radius": ("aminor", 1.0),
    "geometric_axis_r": ("rsurf", 1.0),
    "elongation": ("kappa", 1.0),
    "triangularity_upper": ("tritop", 1.0),
    "triangularity_lower": ("tribot", 1.0),
    "power_ohm": ("p_ohm", 1.0),
    "power_radiated": ("p_rad", 1.0),
    "power_nbi": ("p_nbi", 1.0),
    "power_ec": ("p_ech", 1.0),
    **{
        f"{store_profile}{store_suffix}": (f"{raw_profile}{raw_suffix}", 1.0)
        for raw_profile, store_profile in [("te_rho", "t_e"), ("ne_rho", "n_e")]
        for raw_suffix, store_suffix in [("", ""), ("_error", "_error"), ("_grad", "_gradient"), ("_grad_error", "_gradient_error")]
    },
    "fresh_profile": ("fresh_profile", 1.0),
    "fresh_equilibrium": ("fresh_equilibrium", 1.0),
}
TRAJOPT_SOURCES = {
    "ip_reference": ("ip_prog", 1.0),
    "beta_tor_norm_reference": ("bmtpwrtar", 1.0),
    "geometric_axis_r_reference": ("idtrp", 1.0),
    "x_point_lower_r_reference": ("idtrxbot", 1.0),
    "x_point_lower_z_reference": ("idtzxbot", 1.0),
    "x_point_upper_r_reference": ("idtrxtop", 1.0),
    "x_point_upper_z_reference": ("idtzxtop", 1.0),
    "gap_inner": ("gapin", 1.0),
    "x_point_lower_r": ("rxpt1", 1.0),
    "x_point_lower_z": ("zxpt1", 1.0),
    "x_point_upper_r": ("rxpt2", 1.0),
    "x_point_upper_z": ("zxpt2", 1.0),
    "n_e_pedestal": ("dssneped", 1e19),
    "t_e_psi_norm": ("te_psi", 1.0),
    "n_e_psi_norm": ("ne_psi", 1.0),
    # TODO(ZanderKeith): pin down these three PCS pointnames, kept raw (name and units) until then.
    # bttbt is labeled the programmed toroidal field but does not track the measured one
    # (1.62 T vs 2.08 T on 206743), maybe another pointname is the active target.
    # dstdenp is the density request, line average or pedestal depending on the PCS control scheme.
    # ieeseg07 is labeled the programmed inner gap but behaves like an isoflux segment control error.
    "bttbt": ("bttbt", 1.0),
    "dstdenp": ("dstdenp", 1.0),
    "ieeseg07": ("ieeseg07", 1.0),
}
# Signed in the source, stored as magnitudes
MAGNITUDE_SIGNALS = ("ip", "b0", "ip_reference")
# (min, max) of the trajopt signals, a value outside is NaN, see D3DDataWorkflow.device_specific_processing
TRAJOPT_VALID_RANGES = {
    # dssneped reads 0 when the PCS does not estimate the pedestal
    "n_e_pedestal": (1e18, 2e20),
    "ip_reference": (0, 2.5e6),
    "beta_tor_norm_reference": (0, 6),
    "geometric_axis_r_reference": (1.4, 2.0),
    "gap_inner": (0, 0.5),
    # X points, measured and programmed
    "x_point_lower_r": (1.0, 2.0),
    "x_point_lower_r_reference": (1.0, 2.0),
    "x_point_upper_r": (1.0, 2.0),
    "x_point_upper_r_reference": (1.0, 2.0),
    "x_point_lower_z": (-1.5, 0.0),
    "x_point_lower_z_reference": (-1.5, 0.0),
    "x_point_upper_z": (0.0, 1.5),
    "x_point_upper_z_reference": (0.0, 1.5),
}
# The PCS programmed targets (feedforward control) and the measured signals they are compared against,
# read only by the trajectory optimization.
# They go in their own store, on the same shot / time_idx layout as the prediction store.
TRAJOPT_STORE_NAME = "trajopt"
D3D_TRAJOPT_STORE_SIGNALS = tuple(TRAJOPT_SOURCES)

# Raw-file attribute holding the IDA file a shot's profiles came from
RAW_IDA_PATH_ATTR = "ida_path"
# Store attribute mapping every stored shot to its IDA folder, a JSON object keyed by shot
IDA_SOURCE_ATTR = "ida_source"

PCS_UNVERIFIED = "Raw PCS pointname whose meaning is unverified, see the TODO on D3D_TRAJOPT_STORE_SIGNALS. "

# description of every store variable and coordinate, with units and ref (IMAS path) where the shared schema has none.
# The prediction store signals take their units and refs from transport-validation-datasets' STORE_SIGNAL_ATTRS.
D3D_SIGNAL_ATTRS = {
    # Coordinates
    TIME_COORD: {"units": "s", "description": "Time on the uniform 1 kHz timebase"},
    RADIAL_DIM: {
        "units": "dimensionless",
        "description": RHO_TOR_NORM_DEFINITION,
        "ref": "/core_profiles/profiles_1d(itime)/grid/rho_tor_norm",
    },
    PSI_NORM_DIM: {
        "units": "dimensionless",
        "description": "Normalized poloidal flux of the IDA reconstruction",
        "ref": "/equilibrium/time_slice(itime)/profiles_1d/psi_norm",
    },
    # Prediction store
    "ip": {"description": "Measured plasma current magnitude (PTDATA ip)"},
    "b0": {
        "description": "Vacuum toroidal field magnitude at r0, mu0 144 bcoil / (2 pi r0) from the TF coil current (PTDATA bcoil)",
    },
    "r0": {"description": "Reference major radius b0 is given at, the EFIT rcentr"},
    "energy_mhd": {
        "description": "Stored energy from the 1 kHz DISPY EFIT (wmhd)",
    },
    "beta_tor_norm": {
        "description": "Normalized toroidal beta from the 1 kHz DISPY EFIT (betan)",
    },
    "n_e_line_average": {
        "description": "Line-averaged electron density from the DISPY EFIT tree (density), else the PCS estimate (PTDATA dssdenest)",
    },
    "minor_radius": {
        "description": "Minor radius of the plasma boundary, DISPY EFIT aminor",
    },
    "geometric_axis_r": {
        "description": "Major radius of the geometric center of the boundary, DISPY EFIT rsurf",
    },
    "elongation": {
        "description": "Elongation of the plasma boundary, DISPY EFIT kappa",
    },
    "triangularity_upper": {
        "description": "Upper triangularity of the plasma boundary, DISPY EFIT tritop",
    },
    "triangularity_lower": {
        "description": "Lower triangularity of the plasma boundary, DISPY EFIT tribot",
    },
    "power_ohm": {
        "description": (
            "Ohmic heating power from the 1 kHz DISPY EFIT (poh = Ip V_surf - dW_pol/dt), "
            "its derivatives centered least-squares slopes over +-100 ms (non-causal), clipped at 0"
        ),
    },
    "power_radiated": {
        "description": (
            "Total radiated power including the divertor, bolometer analysis prad_tot (4 ms), "
            "smoothed by a centered 50 ms boxcar applied twice to the raw channels (non-causal), clipped at 0"
        ),
    },
    "power_nbi": {
        "description": "Neutral beam power injected into the vessel (pinj)",
    },
    "power_ic": {
        "description": "Ion cyclotron heating power, zero (fast wave unused in these campaigns)",
    },
    "power_lh": {
        "description": "Lower hybrid heating power, zero (DIII-D has no LHCD)",
    },
    "power_ec": {
        "description": "Electron cyclotron power injected into the vessel (echpwrc)",
    },
    "fresh_profile": {"description": "1 where the profiles are a new IDA slice, 0 where an earlier slice is held"},
    "fresh_equilibrium": {
        "description": (
            "1 where a usable DISPY EFIT slice (chisq and q profile) lands, "
            "0 where the equilibrium signals hold an earlier one (for at least 10 ms)"
        )
    },
    **{
        f"{store_profile}{suffix}": attrs
        for store_profile, quantity in [("t_e", "electron temperature"), ("n_e", "electron density")]
        for suffix, attrs in [
            ("", {"description": f"IDA {quantity} profile"}),
            ("_error", {"description": f"1-sigma uncertainty of the IDA {quantity} profile"}),
            ("_gradient", {"description": f"d/drho_tor_norm gradient of the IDA {quantity} profile"}),
            (
                "_gradient_error",
                {"description": f"1-sigma uncertainty of the d/drho_tor_norm gradient of the IDA {quantity}, assuming independent points"},
            ),
        ]
    },
    # Trajopt store
    "ip_reference": {
        "units": "A",
        "description": "PCS programmed plasma current magnitude (PTDATA iptipp)",
        "ref": "/pulse_schedule/flux_control/ip/reference",
    },
    "beta_tor_norm_reference": {
        "units": "dimensionless",
        "description": "PCS programmed normalized beta (PTDATA bmtpwrtar), not reached without active beta control",
        "ref": "/pulse_schedule/flux_control/beta_tor_norm/reference",
    },
    "geometric_axis_r_reference": {
        "units": "m",
        "description": "PCS programmed major radius (PTDATA idtrp)",
        "ref": "/pulse_schedule/position_control/geometric_axis/r/reference",
    },
    **{
        f"x_point_{position}_{component}_reference": {
            "units": "m",
            "description": f"PCS programmed {position} X point {component.upper()} (PTDATA idt{component}x{pcs_position})",
            "ref": f"/pulse_schedule/position_control/x_point(i1)/{component}/reference",
        }
        for position, pcs_position in [("lower", "bot"), ("upper", "top")]
        for component in ["r", "z"]
    },
    "gap_inner": {
        "units": "m",
        "description": "Gap between the separatrix and the inner wall, DISPY EFIT gapin",
        "ref": "/equilibrium/time_slice(itime)/boundary/gap(i1)/value",
    },
    **{
        f"x_point_{position}_{component}": {
            "units": "m",
            "description": f"{position.capitalize()} X point {component.upper()}, DISPY EFIT {component}xpt{efit_index}",
            "ref": f"/equilibrium/time_slice(itime)/contour_tree/node(i1)/{component}",
        }
        for position, efit_index in [("lower", 1), ("upper", 2)]
        for component in ["r", "z"]
    },
    "n_e_pedestal": {
        "units": "m^-3",
        "description": "PCS real-time pedestal density estimate (PTDATA dssneped)",
        "ref": "/summary/local/pedestal/n_e/value",
    },
    "t_e_psi_norm": {
        "units": "eV",
        "description": "IDA electron temperature profile on psi_norm",
        "ref": "/core_profiles/profiles_1d(itime)/electrons/temperature",
    },
    "n_e_psi_norm": {
        "units": "m^-3",
        "description": "IDA electron density profile on psi_norm",
        "ref": "/core_profiles/profiles_1d(itime)/electrons/density",
    },
    "bttbt": {"units": "T", "description": f"{PCS_UNVERIFIED}Labeled the programmed toroidal field (PTDATA bttbt)"},
    "dstdenp": {"units": "1e19 m^-3", "description": f"{PCS_UNVERIFIED}PCS density request (PTDATA dstdenp)"},
    "ieeseg07": {"units": "m", "description": f"{PCS_UNVERIFIED}Labeled the programmed inner gap (PTDATA ieeseg07)"},
}


class D3DDataWorkflow(RawFileWorkflow):
    """DIII-D specific data workflow for creating and processing datasets.

    All signals come from disruption-py 0.14 in a single get_shots_data call per shot:
    the 1 kHz EFIT of the shot's DISPY run (DispyEfitNicknameSetting skips a shot without one),
    PTDATA via the custom physics methods in physics_methods.py,
    and Te/ne from the IDA database, mapped from IDA's psi_n onto rho_tor_norm through the DISPY q profile.
    Raw data fetching requires access to the DIII-D data servers.

    Two stores: the prediction store (the shared IMAS schema, SI units)
    and the trajopt store (D3D_TRAJOPT_STORE_SIGNALS), which only the trajectory optimization reads.
    Every variable carries description, units and ref (its IMAS path) attributes, see D3D_SIGNAL_ATTRS.
    The signal sources are PREDICTION_SOURCES and TRAJOPT_SOURCES.
    """

    STORE_VARIABLES: ClassVar[dict[str, tuple[str, ...]]] = {
        PREDICTION_STORE_NAME: STORE_SIGNALS,
        TRAJOPT_STORE_NAME: D3D_TRAJOPT_STORE_SIGNALS,
    }
    SIGNAL_ATTRS: ClassVar[dict[str, dict[str, str]]] = D3D_SIGNAL_ATTRS
    STORE_ATTRS: ClassVar[dict[str, str]] = {
        "efit_runtag": config["efit"]["runtag"],
        "profile_source": "IDA",
        "sol_extension": config["profile_grid"]["sol_extension"],
        "rho_tor_norm_definition": RHO_TOR_NORM_DEFINITION,
    }

    # The filter spec of every device store (RawFileWorkflow.filter_ds), SI units
    min_filter: ClassVar[dict[str, float]] = {
        "ip": 2e5,
        "energy_mhd": 1e4,
        # Ramp-ups reach 1e18 (199121), interferometer dropouts read below 3e17 (203836, 204188),
        # and a lost fringe count drives it negative (199126, down to -1.7e20)
        "n_e_line_average": 5e17,
    }
    max_filter: ClassVar[dict[str, float]] = {"greenwald_fraction": 2.0}
    # The EFIT P_oh stays below 0.6 MW through the ramp-up and above 2 MW only at disruptive terminations
    transient_filter: ClassVar[dict[str, float]] = {"power_ohm": 2e6}
    # ip reads near 0 out to the 8 s end of the timebase, so the end is the last ip above its threshold
    end_margin_s: ClassVar[float] = 0.05
    min_pulse_length_s: ClassVar[float] = 0.5
    min_radiated_fraction: ClassVar[float] = 0.025
    # More radiated than put in, the same physical ceiling as the other devices (not checked on DIII-D data)
    max_radiated_fraction: ClassVar[float] = 1.0
    density_ratio_bounds: ClassVar[tuple[float, float]] = (0.7, 1.3)

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

    def _get_shotlist_from_source(self) -> list[int]:
        """Union of the shots every configured IDA database serves."""
        shots = find_ida_shots()
        if not shots:
            raise FileNotFoundError("No IDA files found in any configured IDA database")
        return shots

    def _get_shot_dataset(self, shot: int) -> xr.Dataset:
        """Fetch every signal for one shot through disruption-py.

        Raises:
            RuntimeError: If disruption-py returns no data, e.g. the shot has no DISPY EFIT run (see the log).
        """
        retrieval_settings = RetrievalSettings(
            run_methods=RUN_METHODS,
            custom_physics_methods=[D3DDatasetMethods],
            efit_nickname_setting=DispyEfitNicknameSetting(),
            time_setting=Uniform1kHzTimeSetting(),
            only_requested_columns=False,
        )
        result = get_shots_data(
            tokamak=Tokamak.D3D,
            shotlist_setting=[shot],
            retrieval_settings=retrieval_settings,
            output_setting=DatasetOutputSetting(path=False),
            log_settings=passive_log_settings(),
            num_processes=1,
        )
        if result.sizes.get("idx", 0) == 0:
            raise RuntimeError(f"disruption-py returned no data for shot {shot}")
        result = result.set_index(idx=["shot", "time"]).unstack("idx")
        return result.transpose("shot", "time", RADIAL_DIM, PSI_NORM_DIM, missing_dims="ignore")

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

            ida_path = find_ida_path(shot)
            if ida_path is None:
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

            # get_ida_profiles resolves the same file, record_ida_sources reads it back into the stores
            ds_standardized.attrs[RAW_IDA_PATH_ATTR] = str(ida_path)
            ds_standardized.to_netcdf(ds_path)
            logger.success(f"Saved raw dataset for shot {shot} to {ds_path}")
            processed_shots += 1

        logger.info("Finished making raw data files.")

    def run_processed_data_workflow(self):
        """Build the stores, then record each stored shot's IDA folder on them (record_ida_sources)."""
        super().run_processed_data_workflow()
        self.record_ida_sources()

    def record_ida_sources(self):
        """Set IDA_SOURCE_ATTR on every store: a JSON object from each stored shot to the folder of its IDA file.

        Written after the build since the stores keep the dataset attributes of their first shot only.
        """
        ds_prediction = xr.open_zarr(self.store_path(PREDICTION_STORE_NAME))
        ida_sources = {}
        for shot in ds_prediction[EPISODE_DIM].values:
            with xr.open_dataset(self.raw_data_dir / f"{int(shot)}.nc") as raw_file:
                ida_path = Path(raw_file.attrs[RAW_IDA_PATH_ATTR])
            ida_sources[str(int(shot))] = str(ida_path.parent)
        ida_sources_json = json.dumps(ida_sources)
        for store_name in self.STORE_VARIABLES:
            store_path = self.store_path(store_name)
            store_group = zarr.open_group(store_path, mode="r+")
            store_group.attrs[IDA_SOURCE_ATTR] = ida_sources_json
            # Consolidated metadata is a zarr v3 extension, the stores use it anyway
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ZarrUserWarning)
                zarr.consolidate_metadata(store_path)
        ida_folder_counts = Counter(ida_sources.values())
        logger.info(f"Recorded the IDA source of {len(ida_sources)} shots: {dict(ida_folder_counts)}")

    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset | None:
        """Build both stores' signals (IMAS names, SI units) from the raw disruption-py columns.

        Parameters
        ----------
        ds : xr.Dataset
            Raw dataset with disruption-py column names

        Returns
        -------
        xr.Dataset | None
            Standardized dataset, or None if a source column is missing or a critical signal is all NaN
        """
        sources = {**PREDICTION_SOURCES, **TRAJOPT_SOURCES}
        raw_required = {raw_name for raw_name, _ in sources.values()}
        missing = sorted(raw_name for raw_name in raw_required if raw_name not in ds)
        if missing:
            logger.warning(f"Shot {ds['shot'].item()}: missing raw signals {missing}, skipping shot.")
            return None

        signals = {}
        for store_name, (raw_name, factor) in sources.items():
            signal = ds[raw_name] * factor
            if store_name in MAGNITUDE_SIGNALS:
                signal = abs(signal)
            signals[store_name] = signal
        signals["r0"] = xr.full_like(ds["shot"], R0, dtype=float)
        signals["power_ic"] = xr.zeros_like(ds["p_ohm"])
        signals["power_lh"] = xr.zeros_like(ds["p_ohm"])
        ds_standardized = xr.Dataset(signals)

        # p_nbi is 0 without beams, so it is only NaN when get_heating_powers itself failed
        critical_signals = ["ip", "n_e_line_average", "power_ohm", "power_radiated", "power_nbi", "t_e", "n_e", "t_e_psi_norm"]
        if self.has_all_nan_signal(ds_standardized, critical_signals):
            return None

        return self.standardize_dim_names(ds_standardized)

    def device_specific_processing(self, ds: xr.Dataset) -> xr.Dataset:
        """NaN the trajopt signals outside TRAJOPT_VALID_RANGES.

        The trajopt store is not part of the shared filter spec, so an out-of-range value is NaN
        rather than a gap in the shot.

        Parameters
        ----------
        ds : xr.Dataset
            Standardized dataset

        Returns
        -------
        xr.Dataset
            Processed dataset ready for general workflow
        """
        for name, (valid_min, valid_max) in TRAJOPT_VALID_RANGES.items():
            mask_in_range = (ds[name] >= valid_min) & (ds[name] <= valid_max)
            ds[name] = ds[name].where(mask_in_range)

        # Per-shot, so it must not become a store attribute through the first stored shot (see record_ida_sources)
        del ds.attrs[RAW_IDA_PATH_ATTR]
        return ds

    def device_specific_culling(self, ds: xr.Dataset) -> bool:
        """The config's excluded shots and the default profile cull."""
        shot = int(ds[EPISODE_DIM].values[0])
        if shot in config["culling"]["excluded_shots"]:
            logger.warning(f"Culling shot {shot}: listed in excluded_shots of d3d/config.toml")
            return True
        return super().device_specific_culling(ds)
