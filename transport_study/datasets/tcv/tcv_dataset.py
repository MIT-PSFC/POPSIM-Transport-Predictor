"""Makes the 'raw' TCV dataset from DEFUSE exports and LIUQE reconstructions, to be processed later by POPSIM"""

from pathlib import Path
from typing import ClassVar

import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from loguru import logger

from transport_study import RADIAL_DIM, TIME_COORD
from transport_study.datasets import make_uniform_1khz_timebase
from transport_study.datasets.tcv import config
from transport_study.datasets.tcv.profiles import (
    RHO_TOR_NORM_DEFINITION,
    RHO_TOR_NORM_GRID,
    defuse_profile_on_grid,
)
from transport_study.datasets.tcv.sources import (
    defuse_path,
    find_tcv_shots,
    meqdb_path,
    read_defuse,
    read_liuqe,
)
from transport_study.datasets.workflow import RawFileWorkflow
from transport_study.signals import STORE_PROFILES

# Store signal -> (DEFUSE signal, factor to SI units). DEFUSE is SI apart from the heating powers.
PREDICTION_SOURCES = {
    "ip": ("I_P", 1.0),
    "b0": ("BZERO", 1.0),
    "energy_mhd": ("Wtot", 1.0),
    "beta_tor_norm": ("BETAN", 1.0),
    "n_e_line_average": ("NEavg", 1.0),
    "minor_radius": ("a_minor", 1.0),
    "geometric_axis_r": ("R_geom", 1.0),
    "elongation": ("KAPPA", 1.0),
    "triangularity_upper": ("DELTA_TOP", 1.0),
    "triangularity_lower": ("DELTA_BOTTOM", 1.0),
    "power_ohm": ("POHM", 1.0),
    "power_radiated": ("PradBulk", 1.0),
    "t_e": ("Te_rho", 1.0),
    "n_e": ("Ne_rho", 1.0),
    "t_e_gradient": ("Te_rho_grad", 1.0),
    "n_e_gradient": ("Ne_rho_grad", 1.0),
}
# Store heating power -> DEFUSE signals summed into it, in MW.
# A system a shot does not have is absent from its export, or an empty placeholder, and counts as zero.
HEATING_SOURCES_MW = {
    "power_nbi": ("NBI", "NBI2"),
    "power_ec": ("ECRH",),
}
# Signed in the source (negative in the usual TCV configuration), stored as magnitudes
MAGNITUDE_SIGNALS = ("ip", "b0")

# Edge line-averaged density, kept in the raw files for the edge-to-line-average check only
EDGE_LINE_AVERAGE = "n_e_edge_line_average"
EDGE_LINE_AVERAGE_SOURCE = "Ne_edge_avg"

DEFUSE_PROFILE_SIGNALS = ("Te_rho", "Ne_rho")
# The profile gradients are taken on the DEFUSE fit points, under the profile's name with this suffix
DEFUSE_GRADIENT_SUFFIX = "_grad"
DEFUSE_PROFILE_COLUMNS = (*DEFUSE_PROFILE_SIGNALS, *(f"{name}{DEFUSE_GRADIENT_SUFFIX}" for name in DEFUSE_PROFILE_SIGNALS))
# Every DEFUSE 0D signal read from an export
DEFUSE_SIGNALS = (
    *(raw_name for raw_name, _ in PREDICTION_SOURCES.values() if raw_name not in DEFUSE_PROFILE_COLUMNS),
    *(raw_name for raw_names in HEATING_SOURCES_MW.values() for raw_name in raw_names),
    EDGE_LINE_AVERAGE_SOURCE,
)

# The raw timebase ends at the last time the plasma current magnitude exceeds this [A]
IP_TIMEBASE_MIN_A = 50e3

# Smallest density step [m^-3] read as an interferometer fringe jump
FRINGE_JUMP_MIN_M3 = 1e19

ZERO_ERROR = "Zero, the no-uncertainty sentinel, since DEFUSE gives no uncertainty for its profile fits"

# description and ref (IMAS path) of every store variable and coordinate. ref is absent where IMAS has no leaf.
# The units come from signals.STORE_SIGNAL_UNITS.
TCV_SIGNAL_ATTRS = {
    # Coordinates
    TIME_COORD: {"units": "s", "description": "Time on the uniform 1 kHz timebase"},
    RADIAL_DIM: {
        "units": "dimensionless",
        "description": RHO_TOR_NORM_DEFINITION,
        "ref": "/core_profiles/profiles_1d(itime)/grid/rho_tor_norm",
    },
    # Prediction store
    "ip": {"description": "Measured plasma current magnitude (DEFUSE I_P)", "ref": "/summary/global_quantities/ip/value"},
    "b0": {
        "description": "Vacuum toroidal field magnitude at R0 = 0.88 m (DEFUSE BZERO)",
        "ref": "/summary/global_quantities/b0/value",
    },
    "energy_mhd": {
        "description": "Stored energy on the LIUQE timebase (DEFUSE Wtot)",
        "ref": "/equilibrium/time_slice(itime)/global_quantities/energy_mhd",
    },
    "beta_tor_norm": {
        "description": "Normalized toroidal beta on the LIUQE timebase (DEFUSE BETAN)",
        "ref": "/equilibrium/time_slice(itime)/global_quantities/beta_tor_norm",
    },
    "n_e_line_average": {
        "description": "Line-averaged electron density from the FIR interferometer (DEFUSE NEavg), fringe jumps removed",
        "ref": "/summary/line_average/n_e/value",
    },
    "minor_radius": {
        "description": "Minor radius of the plasma boundary, LIUQE (DEFUSE a_minor)",
        "ref": "/equilibrium/time_slice(itime)/boundary/minor_radius",
    },
    "geometric_axis_r": {
        "description": "Major radius of the geometric center of the boundary, LIUQE (DEFUSE R_geom)",
        "ref": "/equilibrium/time_slice(itime)/boundary/geometric_axis/r",
    },
    "elongation": {
        "description": "Elongation of the plasma boundary, LIUQE (DEFUSE KAPPA)",
        "ref": "/equilibrium/time_slice(itime)/boundary/elongation",
    },
    "triangularity_upper": {
        "description": "Upper triangularity of the plasma boundary, LIUQE (DEFUSE DELTA_TOP)",
        "ref": "/equilibrium/time_slice(itime)/boundary/triangularity_upper",
    },
    "triangularity_lower": {
        "description": "Lower triangularity of the plasma boundary, LIUQE (DEFUSE DELTA_BOTTOM)",
        "ref": "/equilibrium/time_slice(itime)/boundary/triangularity_lower",
    },
    "power_ohm": {
        "description": "Ohmic heating power (DEFUSE POHM), clipped at 0",
        "ref": "/summary/global_quantities/power_ohm/value",
    },
    "power_radiated": {
        "description": "Radiated power from the confined plasma, bolometry (DEFUSE PradBulk), clipped at 0",
        "ref": "/summary/global_quantities/power_radiated/value",
    },
    "power_nbi": {
        "description": "Neutral beam power, summed over both beamlines (DEFUSE NBI + NBI2), zero where a beam is absent",
        "ref": "/summary/heating_current_drive/power_launched_nbi/value",
    },
    "power_ic": {
        "description": "Ion cyclotron heating power, zero (TCV has no ICRH)",
        "ref": "/summary/heating_current_drive/power_ic/value",
    },
    "power_lh": {
        "description": "Lower hybrid heating power, zero (TCV has no LHCD, DEFUSE P_LH is the L-H threshold power)",
        "ref": "/summary/heating_current_drive/power_lh/value",
    },
    "power_ec": {
        "description": "Electron cyclotron power, summed over gyrotrons (DEFUSE ECRH), zero where absent",
        "ref": "/summary/heating_current_drive/power_launched_ec/value",
    },
    "fresh_profile": {"description": "1 where the profiles are a new DEFUSE slice, 0 where an earlier slice is held"},
    **{
        f"{store_profile}{suffix}": attrs
        for store_profile, quantity, ref_leaf in [
            ("t_e", "electron temperature", "temperature"),
            ("n_e", "electron density", "density"),
        ]
        for suffix, attrs in [
            (
                "",
                {
                    "description": f"DEFUSE {quantity} profile fit, mapped from rho_pol onto rho_tor_norm through LIUQE",
                    "ref": f"/core_profiles/profiles_1d(itime)/electrons/{ref_leaf}",
                },
            ),
            ("_error", {"description": ZERO_ERROR, "ref": f"/core_profiles/profiles_1d(itime)/electrons/{ref_leaf}_error_upper"}),
            ("_gradient", {"description": f"d/drho_tor_norm gradient of the DEFUSE {quantity}, taken on the DEFUSE fit points"}),
            ("_gradient_error", {"description": ZERO_ERROR}),
        ]
    },
}


class TCVDataWorkflow(RawFileWorkflow):
    """TCV specific data workflow for creating and processing datasets.

    The 0D signals and the Te/ne profile fits come from the DEFUSE export of each shot.
    DEFUSE fits the profiles on rho_pol = sqrt(psi_N),
    so each slice is mapped onto rho_tor_norm through the q profile of the nearest LIUQE reconstruction
    of the shot's MEQ database, and only shots with one are built.
    DEFUSE gives no uncertainty, so the error companions are the 0 sentinel.
    Every variable carries description, units and ref (its IMAS path) attributes, see TCV_SIGNAL_ATTRS.
    """

    SIGNAL_ATTRS: ClassVar[dict[str, dict[str, str]]] = TCV_SIGNAL_ATTRS
    STORE_ATTRS: ClassVar[dict[str, str]] = {
        "profile_source": "DEFUSE",
        "equilibrium_source": "LIUQE, MEQ databases",
        "rho_tor_norm_definition": RHO_TOR_NORM_DEFINITION,
    }
    # TCV bolometry reads a few percent of the input power or more, a dead bolometer far less
    min_radiated_fraction = 0.025
    density_ratio_bounds = (0.7, 1.3)

    def __init__(
        self,
        ds_name: str,
        shotlist_file: Path | str | None,
        data_assembly_dir: Path | str,
        max_num_shots: int | None = None,
    ):
        """Initialize the TCV data workflow.

        Parameters
        ----------
        ds_name : str
            Name of the dataset/study, used for directory naming
        shotlist_file : str | None
            Path to file containing list of shots to process. If None, uses the
            shots with both a DEFUSE export and a LIUQE MEQ database.
        data_assembly_dir : str
            Directory where data files are stored and final dataset will be saved
        max_num_shots : int | None
            Maximum number of shots to process (for testing). If None, process all shots.
        """

        # Use the TCV dataset config from datasets/tcv/config.toml
        self.config = config

        # Call parent init (which will call _get_shotlist_from_source if needed)
        super().__init__(
            ds_name,
            shotlist_file,
            data_assembly_dir,
            max_num_shots=max_num_shots,
            min_shot_duration=self.config["shot_filters"]["min_duration"],
        )

        # If any of these signals are out of range, drop the entire timeslice. SI units.
        self.filter_config = {
            "energy_mhd": {"min": 1e3, "max": 5e5},
            "n_e_line_average": {"min": 1e18, "max": 4e20},
            EDGE_LINE_AVERAGE: {"min": 1e18, "max": 4e20},
            # Bad interferometer data can satisfy the absolute density cap at low ip
            # so stack another check based on the Greenwald fraction
            "greenwald_fraction": {"min": 0.0, "max": 2.0},
            "ip": {"min": 5e4, "max": 5e5},
            "beta_tor_norm": {"min": 0.01, "max": 10},
            "t_e_axis": {"min": 100, "max": 2e4},
            # LIUQE geometry moments go nonphysical during the current ramp
            # (minor_radius down to 0.04 m, elongation below 1), which drives derived
            # features like q_star far outside the physical range
            "minor_radius": {"min": 0.15, "max": 0.30},
            "geometric_axis_r": {"min": 0.80, "max": 1.0},
            "elongation": {"min": 0.9, "max": 3.0},
        }

        # Set signals outside this range to nan, but don't drop the entire timeslice. SI units.
        self.individual_filter_config = {
            "power_ec": {"min": 0, "max": 1e7},
            "power_nbi": {"min": 0, "max": 1e7},
        }

    def _get_shotlist_from_source(self) -> list[int]:
        """Shots with both a DEFUSE export and a LIUQE MEQ database."""
        shots = find_tcv_shots()
        if not shots:
            raise FileNotFoundError("No shot has both a DEFUSE export and a LIUQE MEQ database, see datasets/tcv/config.toml")
        return shots

    def _get_shot_dataset(self, shot: int) -> xr.Dataset | None:
        """Every DEFUSE signal of one shot on the uniform 1 kHz timebase, under the DEFUSE names.

        The profiles and their gradients (DEFUSE name + DEFUSE_GRADIENT_SUFFIX) are on RHO_TOR_NORM_GRID.
        Returns None when the shot has no plasma current or a profile fit with fewer than two slices.
        """
        signals, profiles = read_defuse(defuse_path(shot), DEFUSE_SIGNALS, DEFUSE_PROFILE_SIGNALS)
        if "I_P" not in signals:
            logger.warning(f"Shot {shot}: DEFUSE has no I_P, skipping shot.")
            return None
        for name in DEFUSE_PROFILE_SIGNALS:
            if name not in profiles or profiles[name].time.size < 2:
                logger.warning(f"Shot {shot}: DEFUSE has no {name} fit with at least two slices, skipping shot.")
                return None

        ip = signals["I_P"]
        mask_ip_valid = np.abs(ip.values) > IP_TIMEBASE_MIN_A
        if not mask_ip_valid.any():
            logger.warning(f"Shot {shot}: |I_P| never exceeds {IP_TIMEBASE_MIN_A:.0f} A, skipping shot.")
            return None
        timebase = make_uniform_1khz_timebase(ip.time[mask_ip_valid].max())

        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        for name, signal in signals.items():
            values_on_timebase = np.interp(timebase, signal.time, signal.values, left=np.nan, right=np.nan)
            data_vars[name] = (("time",), values_on_timebase)
        equilibria = read_liuqe(meqdb_path(shot))
        profile_columns = {}
        for name, profile in profiles.items():
            values_on_grid, gradient_on_grid = defuse_profile_on_grid(profile, equilibria, timebase)
            profile_columns[name] = values_on_grid
            profile_columns[f"{name}{DEFUSE_GRADIENT_SUFFIX}"] = gradient_on_grid
        # A time keeps its profiles only where both fits exist,
        # so fresh_profile (labelled from n_e) never marks a time without a Te
        mask_both_fits = np.ones(timebase.size, dtype=bool)
        for name in DEFUSE_PROFILE_SIGNALS:
            mask_both_fits &= np.isfinite(profile_columns[name][:, 0])
        for name, values in profile_columns.items():
            values[~mask_both_fits] = np.nan
            data_vars[name] = (("time", RADIAL_DIM), values)
        coords = {"time": timebase, RADIAL_DIM: RHO_TOR_NORM_GRID.astype(np.float32)}
        ds = xr.Dataset(data_vars, coords=coords)
        return ds.expand_dims(shot=[shot])

    def make_raw_data_files(self):
        """Create raw data files from the DEFUSE exports and LIUQE reconstructions.

        One netCDF file per shot on a uniform 1 kHz timebase with the on-disk signal names.
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

            if not (defuse_path(shot).exists() and meqdb_path(shot).exists()):
                logger.info(f"Skipping shot {shot} since it lacks a DEFUSE export or a LIUQE MEQ database")
                continue

            try:
                ds = self._get_shot_dataset(shot)
            except Exception as e:
                logger.error(f"Error retrieving shot {shot}: {e}", exc_info=True)
                continue
            if ds is None:
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
        """Build the store signals (IMAS names, SI units) from the DEFUSE signals on the uniform timebase.

        Parameters
        ----------
        ds : xr.Dataset
            DEFUSE signals on the uniform timebase, profiles on the store grid

        Returns
        -------
        xr.Dataset | None
            Store signals plus the processing-only edge density,
            or None if a required DEFUSE signal is missing or a critical signal is all NaN
        """
        raw_required = [raw_name for raw_name, _ in PREDICTION_SOURCES.values()] + [EDGE_LINE_AVERAGE_SOURCE]
        missing = sorted(raw_name for raw_name in raw_required if raw_name not in ds)
        if missing:
            logger.warning(f"Shot {ds['shot'].item()}: missing DEFUSE signals {missing}, skipping shot.")
            return None

        signals = {}
        for store_name, (raw_name, factor) in PREDICTION_SOURCES.items():
            signal = ds[raw_name] * factor
            if store_name in MAGNITUDE_SIGNALS:
                signal = abs(signal)
            signals[store_name] = signal
        power_zero = xr.zeros_like(ds["POHM"])
        for store_name, raw_names in HEATING_SOURCES_MW.items():
            power_MW = power_zero
            for raw_name in raw_names:
                if raw_name in ds:
                    power_MW = power_MW + ds[raw_name].fillna(0.0)
            signals[store_name] = power_MW * 1e6
        signals["power_ic"] = power_zero
        signals["power_lh"] = power_zero
        # The no-uncertainty sentinel wherever the profile exists
        for profile in STORE_PROFILES:
            error_zero = xr.zeros_like(signals[profile]).where(signals[profile].notnull())
            signals[f"{profile}_error"] = error_zero
            signals[f"{profile}_gradient_error"] = error_zero
        signals[EDGE_LINE_AVERAGE] = ds[EDGE_LINE_AVERAGE_SOURCE]
        ds_standardized = xr.Dataset(signals)

        critical_signals = ["ip", "n_e_line_average", "t_e", "n_e"]
        if self.has_all_nan_signal(ds_standardized, critical_signals):
            return None

        return self.standardize_dim_names(ds_standardized)

    def device_specific_processing(self, ds: xr.Dataset) -> xr.Dataset:
        """Derive the processing-only filter inputs and remove interferometer fringe jumps.

        greenwald_fraction and t_e_axis are range-checked by filter_config and never stored.

        Parameters
        ----------
        ds : xr.Dataset
            Standardized dataset

        Returns
        -------
        xr.Dataset
            Processed dataset ready for general workflow
        """
        # n_Greenwald [1e20 m^-3] = Ip [MA] / (pi a^2)
        ip_MA = ds["ip"] * 1e-6
        n_greenwald_1e20 = ip_MA / (np.pi * ds["minor_radius"] ** 2)
        ds["greenwald_fraction"] = ds["n_e_line_average"] * 1e-20 / n_greenwald_1e20

        ds["t_e_axis"] = ds["t_e"].sel({RADIAL_DIM: 0}, method="nearest")

        for density_var in ["n_e_line_average", EDGE_LINE_AVERAGE]:
            density_trace = ds[density_var].values[0, :]
            density_corrected = _remove_fringe_jumps(density_trace)
            ds[density_var] = (ds[density_var].dims, density_corrected[np.newaxis, :])
        return ds

    def device_specific_culling(self, ds: xr.Dataset) -> bool:
        """The default profile cull, plus shots with a missing bolometer or interferometer record,
        or an edge density far above the line average."""
        if super().device_specific_culling(ds):
            return True
        if self.has_all_nan_signal(ds, ["power_radiated", "n_e_line_average", EDGE_LINE_AVERAGE]):
            return True

        n_e_line_average_mean = ds["n_e_line_average"].mean().item()
        n_e_edge_mean = ds[EDGE_LINE_AVERAGE].mean().item()
        if n_e_edge_mean > 2 * n_e_line_average_mean:
            logger.info(f"Culling shot {ds.shot.values[0]} due to edge density being significantly higher than line-avg density")
            return True
        return False


def _remove_fringe_jumps(density_trace: np.ndarray) -> np.ndarray:
    """A line-averaged density trace with interferometer fringe jumps removed.

    A step larger than max(FRINGE_JUMP_MIN_M3, 5 x the median step) is read as a fringe jump,
    and its offset is removed from the rest of the trace.
    """
    if np.isnan(density_trace).all():
        return density_trace
    density_steps = np.diff(density_trace)
    median_abs_step = np.nanmedian(np.abs(density_steps))
    jump_threshold = max(FRINGE_JUMP_MIN_M3, 5.0 * median_abs_step)

    offset = 0.0
    density_corrected = density_trace.copy()
    for i in range(1, density_trace.size):
        if np.isnan(density_trace[i - 1]) or np.isnan(density_trace[i]):
            density_corrected[i] = density_trace[i] - offset
            continue
        step = density_trace[i] - density_trace[i - 1]
        if np.abs(step) >= jump_threshold:
            offset += step
        density_corrected[i] = density_trace[i] - offset
    return density_corrected
