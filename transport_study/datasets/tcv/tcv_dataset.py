"""Makes the 'raw' TCV dataset from DEFUSE exports and LIUQE reconstructions, to be processed later by POPSIM"""

from pathlib import Path
from typing import ClassVar

import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view
from transport_validation_datasets.machine.generic import (
    EQUILIBRIUM_HOLD_FLOOR,
    MU0,
    hold_onto_grid,
    make_uniform_1kHz_timebase,
    ohmic_power,
    signal_on_grid,
    smoothed_power,
)

from transport_study import RADIAL_DIM, TIME_COORD
from transport_study.datasets.tcv import config
from transport_study.datasets.tcv.profiles import (
    RHO_TOR_NORM_DEFINITION,
    RHO_TOR_NORM_GRID,
    defuse_profile_on_grid,
    liuqe_usable,
)
from transport_study.datasets.tcv.sources import (
    DefuseSignal,
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
    "n_e_line_average": ("NEavg", 1.0),
    "minor_radius": ("a_minor", 1.0),
    "geometric_axis_r": ("R_geom", 1.0),
    "elongation": ("KAPPA", 1.0),
    "triangularity_upper": ("DELTA_TOP", 1.0),
    "triangularity_lower": ("DELTA_BOTTOM", 1.0),
    "power_radiated": ("PradTot", 1.0),
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
# Major radius DEFUSE BZERO is the vacuum toroidal field at, LIUQE's r0 and the store's r0 [m]
R0 = 0.88

# DEFUSE signals of the LIUQE reconstruction, BZERO (LIUQE rBt / r0) among them.
# Each is held for at least EQUILIBRIUM_HOLD_FLOOR, so a few missing reconstructions are bridged.
LIUQE_SOURCES = ("Wtot", "Vol", "a_minor", "R_geom", "KAPPA", "DELTA_TOP", "DELTA_BOTTOM", "LI", "BZERO")
# PradTot follows the Thomson cadence (~17 ms) but often skips one or two samples (33-50 ms steps),
# or comes in bursts 50 ms apart (61056).
# The 1.5-step hold left gaps that cut ~10 percent of the kept time, and 60 ms bridges both.
# A causal zero-order hold, not the 50 ms triangle of smoothed_power, since PradTot already reads smooth.
PRAD_TOT_HOLD_FLOOR_S = 60e-3
# DEFUSE signal -> shortest hold [s], 0 for the rest
HOLD_FLOORS_S = {**dict.fromkeys(LIUQE_SOURCES, EQUILIBRIUM_HOLD_FLOOR), "PradTot": PRAD_TOT_HOLD_FLOOR_S}

# DEFUSE signals power_ohm is computed from (ohmic_power), DEFUSE POHM has no documented definition
OHMIC_POWER_SOURCES = ("I_P", "Vloop", "LI", "R_geom")
# DEFUSE signals beta_tor_norm is computed from (_normalized_beta) in the B_geo convention of every store.
# DEFUSE BETAN normalizes beta_tor by the volume-averaged vacuum field and multiplies by |BZERO| at r0,
# which reads a median 5.6 percent below it.
NORMALIZED_BETA_SOURCES = ("Wtot", "Vol", "a_minor", "R_geom", "BZERO", "I_P")
# DEFUSE Vloop has the opposite sign convention to I_P:
# Ip * Vloop is negative at flat-top on all 39 shots checked, of both current polarities
DEFUSE_VLOOP_SIGN = -1.0
DEFUSE_PROFILE_SIGNALS = ("Te_rho", "Ne_rho")
# The profile gradients are taken on the DEFUSE fit points, under the profile's name with this suffix
DEFUSE_GRADIENT_SUFFIX = "_grad"
DEFUSE_PROFILE_COLUMNS = (*DEFUSE_PROFILE_SIGNALS, *(f"{name}{DEFUSE_GRADIENT_SUFFIX}" for name in DEFUSE_PROFILE_SIGNALS))
# Every DEFUSE signal a store signal is read from
PREDICTION_RAW_NAMES = tuple(raw_name for raw_name, _ in PREDICTION_SOURCES.values())
# Every DEFUSE 0D signal read from an export
DEFUSE_SIGNALS = (
    *(raw_name for raw_name in PREDICTION_RAW_NAMES if raw_name not in DEFUSE_PROFILE_COLUMNS),
    *(raw_name for raw_names in HEATING_SOURCES_MW.values() for raw_name in raw_names),
    # The store signals among them are read already
    *(raw_name for raw_name in (*OHMIC_POWER_SOURCES, *NORMALIZED_BETA_SOURCES) if raw_name not in PREDICTION_RAW_NAMES),
)

# The raw timebase ends at the last time the plasma current magnitude exceeds this [A]
IP_TIMEBASE_MIN_A = 50e3

# Fringe jumps of the FIR interferometer, removed from the raw NEavg samples (remove_fringe_jumps).
# Smallest level shift read as a fringe jump [m^-3]. The clean jumps in 185 shots are 1.1-2.5e19.
FRINGE_JUMP_MIN_M3 = 1e19
# A fringe jump completes within a few raw samples, and no real density change is that fast.
# So a jump is looked for between the medians of this long on either side of each sample [s].
FRINGE_SHARP_WINDOW_S = 0.25e-3
# Sharp shifts closer together than this are one episode, such as a dropout and its recovery [s]
FRINGE_EPISODE_GAP_S = 5e-3
# The levels on either side of an episode are the medians from FRINGE_SETTLE_S to FRINGE_LEVEL_WINDOW_S away from it,
# so a spike decaying back to the level it left is not read as a jump [s]
FRINGE_SETTLE_S = 2e-3
FRINGE_LEVEL_WINDOW_S = 10e-3
# An episode longer than FRINGE_BURST_WINDOW_S, or this many corrected episodes within it,
# means the interferometer has lost count, and the rest of the record is cut
FRINGE_BURST_EPISODES = 3
FRINGE_BURST_WINDOW_S = 0.05

ZERO_ERROR = "Zero, the no-uncertainty sentinel, since DEFUSE gives no uncertainty for its profile fits"

# description of every store variable and coordinate, with units and ref (IMAS path) where the shared schema has none.
# The store signals take their units and refs from transport-validation-datasets' STORE_SIGNAL_ATTRS.
TCV_SIGNAL_ATTRS = {
    # Coordinates
    TIME_COORD: {"units": "s", "description": "Time on the uniform 1 kHz timebase"},
    RADIAL_DIM: {
        "units": "dimensionless",
        "description": RHO_TOR_NORM_DEFINITION,
        "ref": "/core_profiles/profiles_1d(itime)/grid/rho_tor_norm",
    },
    # Prediction store
    "ip": {"description": "Measured plasma current magnitude (DEFUSE I_P)"},
    "b0": {
        "description": "Vacuum toroidal field magnitude at r0, DEFUSE BZERO (LIUQE rBt / r0)",
    },
    "r0": {"description": "Reference major radius b0 is given at, LIUQE's r0"},
    "energy_mhd": {
        "description": "Stored energy on the LIUQE timebase (DEFUSE Wtot)",
    },
    "beta_tor_norm": {
        "description": (
            "Normalized toroidal beta with B_geo, 100 beta_tor a B_geo / Ip[MA] with beta_tor = 2 mu0 <p> / B_geo^2, "
            "<p> = 2 Wtot / (3 Vol) and B_geo = |BZERO| r0 / R_geom (DEFUSE Wtot, Vol, a_minor, R_geom, BZERO, I_P), "
            "not DEFUSE BETAN"
        ),
    },
    "n_e_line_average": {
        "description": (
            "Line-averaged electron density from the FIR interferometer (DEFUSE NEavg), "
            "fringe jumps removed from the raw samples (non-causal), NaN from where the interferometer lost count"
        ),
    },
    "minor_radius": {
        "description": "Minor radius of the plasma boundary, LIUQE (DEFUSE a_minor)",
    },
    "geometric_axis_r": {
        "description": "Major radius of the geometric center of the boundary, LIUQE (DEFUSE R_geom)",
    },
    "elongation": {
        "description": "Elongation of the plasma boundary, LIUQE (DEFUSE KAPPA)",
    },
    "triangularity_upper": {
        "description": "Upper triangularity of the plasma boundary, LIUQE (DEFUSE DELTA_TOP)",
    },
    "triangularity_lower": {
        "description": "Lower triangularity of the plasma boundary, LIUQE (DEFUSE DELTA_BOTTOM)",
    },
    "power_ohm": {
        "description": (
            "Ohmic heating power, Ip * V_loop minus the rate of change of the internal poloidal magnetic energy "
            "mu0 R_geo li Ip^2 / 4 (DEFUSE I_P, Vloop, LI, R_geom, backward difference), "
            "smoothed by a centered 50 ms boxcar applied twice (non-causal), clipped at 0"
        ),
    },
    "power_radiated": {
        "description": (
            "Total radiated power including the divertor, bolometry (DEFUSE PradTot on its ~17 ms cadence, "
            "not smoothed further, held causally for at least 60 ms over skipped samples), clipped at 0"
        ),
    },
    "power_nbi": {
        "description": "Neutral beam power, summed over both beamlines (DEFUSE NBI + NBI2), zero where a beam is absent",
    },
    "power_ic": {
        "description": "Ion cyclotron heating power, zero (TCV has no ICRH)",
    },
    "power_lh": {
        "description": "Lower hybrid heating power, zero (TCV has no LHCD, DEFUSE P_LH is the L-H threshold power)",
    },
    "power_ec": {
        "description": "Electron cyclotron power, summed over gyrotrons (DEFUSE ECRH), zero where absent",
    },
    "fresh_profile": {"description": "1 where the profiles are a new DEFUSE slice, 0 where an earlier slice is held"},
    "fresh_equilibrium": {
        "description": (
            "1 where a LIUQE reconstruction usable for the profile mapping (a mappable q, MEQ database) lands, 0 otherwise. "
            "The LIUQE 0D signals come from the DEFUSE export on its own times and hold for at least 10 ms, "
            "so they can update where it is 0"
        )
    },
    **{
        f"{store_profile}{suffix}": attrs
        for store_profile, quantity in [("t_e", "electron temperature"), ("n_e", "electron density")]
        for suffix, attrs in [
            ("", {"description": f"DEFUSE {quantity} profile fit, mapped from rho_pol onto rho_tor_norm through LIUQE"}),
            ("_error", {"description": ZERO_ERROR}),
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

    # The filter spec of every device store (RawFileWorkflow.filter_ds), SI units
    min_filter: ClassVar[dict[str, float]] = {
        "ip": 5e4,
        "energy_mhd": 1e3,
        # A broken FIR record reads ~0 or negative, and Thomson calibrated to it reads ~0 too (70353, 70356).
        # The lowest real plasma in 280 shots is 3.4e18 (74082, where Thomson agrees).
        "n_e_line_average": 2e18,
    }
    max_filter: ClassVar[dict[str, float]] = {
        # Bad interferometer data can pass an absolute density cap at low ip
        "greenwald_fraction": 2.0,
    }
    # P_oh as on DIII-D. In 246 shots only 75026 has a 5 ms P_rad peak above 3 MW (12 MW).
    transient_filter: ClassVar[dict[str, float]] = {"power_ohm": 2e6, "power_radiated": 5e6}
    end_margin_s: ClassVar[float] = 0.05
    min_pulse_length_s: ClassVar[float] = 0.5
    # TCV bolometry reads a few percent of the input power or more, a dead bolometer far less
    min_radiated_fraction: ClassVar[float] = 0.025
    # 5 of 246 shots radiate more than is put in, two by far (75026 4.1x with a 12 MW PradTot spike, 78926 1.7x),
    # while the 99th percentile is 1.17.
    max_radiated_fraction: ClassVar[float] = 1.0
    density_ratio_bounds: ClassVar[tuple[float, float]] = (0.7, 1.3)

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
        )

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
        timebase = make_uniform_1kHz_timebase(ip.time[mask_ip_valid].max())
        density_raw_name, _ = PREDICTION_SOURCES["n_e_line_average"]
        if density_raw_name in signals:
            density = signals[density_raw_name]
            density_corrected, cut_time = remove_fringe_jumps(density.time, density.values)
            signals[density_raw_name] = DefuseSignal(time=density.time, values=density_corrected)
            if cut_time is not None:
                logger.info(f"Shot {shot}: the FIR interferometer lost count, {density_raw_name} cut from {cut_time:.3f} s")

        # Every signal is placed causally (signal_on_grid), so no time draws on a later one
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        for name, signal in signals.items():
            hold_floor = HOLD_FLOORS_S.get(name, 0.0)
            values_on_timebase = signal_on_grid(signal.time, signal.values, timebase, hold_floor)
            data_vars[name] = (("time",), values_on_timebase)
        equilibria = read_liuqe(meqdb_path(shot))
        # Each reconstruction builds a Phi_N map to be judged, so only once per shot
        mask_eq_usable = liuqe_usable(equilibria)
        # The grid times a usable reconstruction lands on, the ones the profiles map through
        _, fresh_equilibrium = hold_onto_grid(timebase, equilibria.time[mask_eq_usable], False)
        data_vars["fresh_equilibrium"] = (("time",), fresh_equilibrium.astype(np.float32))
        profile_columns = {}
        fresh_by_profile = {}
        for name, profile in profiles.items():
            values_on_grid, gradient_on_grid, fresh = defuse_profile_on_grid(profile, equilibria, mask_eq_usable, timebase)
            profile_columns[name] = values_on_grid
            profile_columns[f"{name}{DEFUSE_GRADIENT_SUFFIX}"] = gradient_on_grid
            fresh_by_profile[name] = fresh
        # A time keeps its profiles only where both fits exist,
        # so fresh_profile (from the n_e slices) never marks a time without a Te
        mask_both_fits = np.ones(timebase.size, dtype=bool)
        for name in DEFUSE_PROFILE_SIGNALS:
            mask_both_fits &= np.isfinite(profile_columns[name][:, 0])
        for name, values in profile_columns.items():
            values[~mask_both_fits] = np.nan
            data_vars[name] = (("time", RADIAL_DIM), values)
        ne_raw_name, _ = PREDICTION_SOURCES["n_e"]
        fresh_profile = fresh_by_profile[ne_raw_name] & mask_both_fits
        data_vars["fresh_profile"] = (("time",), fresh_profile.astype(np.float32))
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
            Store signals, or None if a required DEFUSE signal is missing or a critical signal is all NaN
        """
        raw_required = {*PREDICTION_RAW_NAMES, *OHMIC_POWER_SOURCES, *NORMALIZED_BETA_SOURCES, "fresh_profile", "fresh_equilibrium"}
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
        signals["r0"] = xr.full_like(ds["shot"], R0, dtype=float)
        signals["power_ohm"] = _ohmic_power(ds)
        signals["beta_tor_norm"] = _normalized_beta(ds)
        signals["fresh_profile"] = ds["fresh_profile"]
        signals["fresh_equilibrium"] = ds["fresh_equilibrium"]
        power_zero = xr.zeros_like(ds["I_P"])
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
        ds_standardized = xr.Dataset(signals)

        critical_signals = ["ip", "n_e_line_average", "t_e", "n_e"]
        if self.has_all_nan_signal(ds_standardized, critical_signals):
            return None

        return self.standardize_dim_names(ds_standardized)

    def device_specific_processing(self, ds: xr.Dataset) -> xr.Dataset:
        """Nothing TCV-specific, the fringe jumps are removed from the raw NEavg samples in the raw stage."""
        return ds


def _ohmic_power(ds: xr.Dataset) -> xr.DataArray:
    """Ohmic power Ip V_loop - dW_pol/dt (ohmic_power) from the DEFUSE signals on the timebase.

    Vloop is flipped onto the sign convention of I_P (DEFUSE_VLOOP_SIGN),
    and the result is smoothed non-causally (smoothed_power), as on C-Mod and MAST.
    """
    times = ds["time"].values
    ip = ds["I_P"].isel(shot=0).values
    v_loop_defuse = ds["Vloop"].isel(shot=0).values
    v_loop = DEFUSE_VLOOP_SIGN * v_loop_defuse
    li = ds["LI"].isel(shot=0).values
    major_radius = ds["R_geom"].isel(shot=0).values
    p_ohm_raw = ohmic_power(times, ip, v_loop, li, major_radius)
    time_steps = np.diff(times)
    dt = float(np.median(time_steps))
    p_ohm = smoothed_power(p_ohm_raw, dt)
    return xr.DataArray(p_ohm[np.newaxis, :], dims=ds["I_P"].dims, coords=ds["I_P"].coords)


def _normalized_beta(ds: xr.Dataset) -> xr.DataArray:
    """Normalized toroidal beta with B_geo, from the LIUQE signals and I_P on the timebase.

    beta_tor = 2 mu0 <p> / B_geo^2 with the volume-averaged pressure <p> = 2 Wtot / (3 Vol),
    and beta_N = 100 beta_tor a B_geo / Ip[MA], the convention every store holds.
    B_geo = |BZERO| R0 / R_geom carries LIUQE's vacuum field at R0 out to the geometric axis.
    """
    pressure_mean = 2.0 * ds["Wtot"] / (3.0 * ds["Vol"])
    b_center_magnitude = abs(ds["BZERO"])
    b_geo = b_center_magnitude * R0 / ds["R_geom"]
    beta_tor = 2.0 * MU0 * pressure_mean / b_geo**2
    ip_magnitude = abs(ds["I_P"])
    ip_magnitude_ma = ip_magnitude / 1e6
    return 100.0 * beta_tor * ds["a_minor"] * b_geo / ip_magnitude_ma


def _sharp_shift_samples(density: np.ndarray, n_sharp: int) -> np.ndarray:
    """Samples after which the median of the next n_sharp samples differs from the median of the n_sharp up to it
    by FRINGE_JUMP_MIN_M3 or more, in order."""
    sample_windows = sliding_window_view(density, n_sharp)
    window_medians = np.median(sample_windows, axis=1)
    # Shift across the boundary after sample k, for k from n_sharp - 1 to n - n_sharp - 1
    sharp_shift = window_medians[n_sharp:] - window_medians[:-n_sharp]
    sharp_shift_magnitude = np.abs(sharp_shift)
    idx_window_pair = np.flatnonzero(sharp_shift_magnitude >= FRINGE_JUMP_MIN_M3)
    return idx_window_pair + n_sharp - 1


def _fringe_episode_spans(sample_time: np.ndarray, idx_sharp: np.ndarray, n_sharp: int) -> tuple[np.ndarray, np.ndarray]:
    """First and last sample of each episode: sharp shifts within FRINGE_EPISODE_GAP_S, with their sharp windows."""
    sharp_times = sample_time[idx_sharp]
    sharp_gaps = np.diff(sharp_times)
    mask_episode_start = np.r_[True, sharp_gaps > FRINGE_EPISODE_GAP_S]
    episode_start = np.flatnonzero(mask_episode_start)
    episode_stop = np.r_[episode_start[1:], idx_sharp.size]
    idx_first_shift = idx_sharp[episode_start]
    idx_last_shift = idx_sharp[episode_stop - 1]
    span_first = np.maximum(idx_first_shift - n_sharp + 1, 0)
    span_last = np.minimum(idx_last_shift + n_sharp, sample_time.size - 1)
    return span_first, span_last


def _level_samples(idx_settled: np.ndarray, idx_adjacent: np.ndarray, n_sharp: int) -> np.ndarray:
    """The settled samples on one side of an episode, or the ones next to it when a neighbor leaves too few."""
    if idx_settled.size >= n_sharp:
        return idx_settled
    return idx_adjacent


def remove_fringe_jumps(sample_time: np.ndarray, density: np.ndarray) -> tuple[np.ndarray, float | None]:
    """The raw NEavg samples with the interferometer fringe jumps removed, non-causally.

    A sharp shift is a sample after which the median of the next FRINGE_SHARP_WINDOW_S
    differs from the median of the FRINGE_SHARP_WINDOW_S up to it by FRINGE_JUMP_MIN_M3 or more.
    Sharp shifts within FRINGE_EPISODE_GAP_S of each other form one episode.
    When the settled levels on either side of an episode differ by FRINGE_JUMP_MIN_M3 or more,
    the difference is removed from everything after it.
    The samples inside every episode are replaced by a straight line between its edges.
    So a spike that decays back, or a dropout that recovers, only loses its inside.
    An episode longer than FRINGE_BURST_WINDOW_S, or FRINGE_BURST_EPISODES corrections within it,
    means the interferometer has lost count, and every sample from the start of the first such episode on is NaN.

    Args:
        sample_time: (n,) sorted, unique sample times [s].
        density: (n,) NEavg [m^-3].

    Returns:
        (n,) the corrected samples, and the time the record is cut from, None when it is not.
    """
    density_corrected = density.copy()
    sample_steps = np.diff(sample_time)
    sample_step = float(np.median(sample_steps))
    n_sharp_window = round(FRINGE_SHARP_WINDOW_S / sample_step)
    n_sharp = max(3, n_sharp_window)
    if density.size < 2 * n_sharp + 1:
        return density_corrected, None
    idx_sharp = _sharp_shift_samples(density, n_sharp)
    if idx_sharp.size == 0:
        return density_corrected, None
    span_first, span_last = _fringe_episode_spans(sample_time, idx_sharp, n_sharp)

    idx_samples = np.arange(density.size)
    corrected_episode_times = []
    lost_count_time = np.inf
    for i_episode in range(span_first.size):
        first = span_first[i_episode]
        last = span_last[i_episode]
        # The levels stop short of the neighboring episodes
        previous_last = span_last[i_episode - 1] if i_episode > 0 else -1
        next_first = span_first[i_episode + 1] if i_episode + 1 < span_first.size else density.size
        time_first = sample_time[first]
        time_last = sample_time[last]
        if time_last - time_first > FRINGE_BURST_WINDOW_S:
            lost_count_time = time_first
            break
        mask_before = (sample_time >= time_first - FRINGE_LEVEL_WINDOW_S) & (sample_time < time_first - FRINGE_SETTLE_S)
        mask_after = (sample_time > time_last + FRINGE_SETTLE_S) & (sample_time <= time_last + FRINGE_LEVEL_WINDOW_S)
        mask_before &= idx_samples > previous_last
        mask_after &= idx_samples < next_first
        adjacent_before_first = max(previous_last + 1, first - n_sharp)
        adjacent_after_stop = min(next_first, last + 1 + n_sharp)
        idx_adjacent_before = np.arange(adjacent_before_first, first)
        idx_adjacent_after = np.arange(last + 1, adjacent_after_stop)
        if idx_adjacent_before.size == 0 or idx_adjacent_after.size == 0:
            continue
        idx_settled_before = np.flatnonzero(mask_before)
        idx_settled_after = np.flatnonzero(mask_after)
        idx_before = _level_samples(idx_settled_before, idx_adjacent_before, n_sharp)
        idx_after = _level_samples(idx_settled_after, idx_adjacent_after, n_sharp)
        level_before = np.nanmedian(density_corrected[idx_before])
        level_after = np.nanmedian(density_corrected[idx_after])
        level_shift = level_after - level_before
        if np.abs(level_shift) >= FRINGE_JUMP_MIN_M3:
            density_corrected[last + 1 :] -= level_shift
            corrected_episode_times.append(time_first)

        # A straight line across the episode, between the samples next to it
        edge_before = np.nanmedian(density_corrected[idx_adjacent_before])
        edge_after = np.nanmedian(density_corrected[idx_adjacent_after])
        edge_times = [sample_time[first - 1], sample_time[last + 1]]
        span_times = sample_time[first : last + 1]
        density_corrected[first : last + 1] = np.interp(span_times, edge_times, [edge_before, edge_after])

    episode_times = np.asarray(corrected_episode_times)
    for episode_time in episode_times:
        mask_burst = (episode_times >= episode_time) & (episode_times <= episode_time + FRINGE_BURST_WINDOW_S)
        if mask_burst.sum() >= FRINGE_BURST_EPISODES:
            lost_count_time = min(lost_count_time, episode_time)
            break
    if not np.isfinite(lost_count_time):
        return density_corrected, None
    density_corrected[sample_time >= lost_count_time] = np.nan
    return density_corrected, float(lost_count_time)
