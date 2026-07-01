"""Makes the raw MAST dataset from the STFC ECHO S3 open-access Zarr store."""

import gc
from pathlib import Path

# disruption_py physics methods log at their custom VERBOSE level; importing
# log_settings registers it (and logger.verbose) on the loguru logger class
import disruption_py.settings.log_settings  # noqa: F401
import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from disruption_py.core.physics_method.params import PhysicsMethodParams
from disruption_py.inout.xr import XarrayDataConnection
from disruption_py.machine.mast.physics import MastPhysicsMethods
from disruption_py.machine.mast.util import MastUtilMethods
from disruption_py.machine.tokamak import Tokamak
from dynaconf import Dynaconf
from loguru import logger
from scipy.interpolate import interp1d
from threadpoolctl import threadpool_limits

from transport_study import EPISODE_DIM, PACKAGE_ROOT, TIME_COORD, TIME_DIM
from transport_study.datasets import make_uniform_1khz_timebase
from transport_study.datasets.gp_fitting.fit_worker import (
    ShotFitInput,
    ShotFitOutput,
    fit_batch,
)
from transport_study.datasets.workflow import GC_INTERVAL, DataWorkflow

DEFAULT_SHOTLIST_FILE = Path(PACKAGE_ROOT) / "datasets" / "mast" / "mast_shotlist"

config = Dynaconf(settings_files=[Path(PACKAGE_ROOT) / "datasets/mast/config.toml"])


def _make_fs(endpoint_url: str) -> "s3fs.S3FileSystem":  # noqa: F821
    # Lazy import so machines without s3fs can still import this module (e.g. via the CLI)
    import s3fs

    return s3fs.S3FileSystem(anon=True, endpoint_url=endpoint_url)


# Zarr variable paths (relative to shot store root) that must be present.
_REQUIRED_ZARR_VARS = [
    "equilibrium/vloop_dynamic",
    "summary/line_average_n_e",
    "summary/power_radiated",
]


def _check_required_signals(shot: int, cfg) -> bool:
    """Return True if all required zarr variables exist in the level2 store.

    Uses fs.ls() on each variable prefix rather than checking for a specific
    metadata filename, so it works for both zarr v2 (.zarray) and v3 (zarr.json).
    """
    fs = _make_fs(cfg["level2_endpoint"])
    base = f"{cfg['level2_path']}/{shot}.{cfg['level2_ext']}"
    for var_path in _REQUIRED_ZARR_VARS:
        try:
            entries = fs.ls(f"{base}/{var_path}", detail=False)
            if not entries:
                raise FileNotFoundError
        except Exception:
            logger.warning(f"Shot {shot}: {var_path} not found in store, skipping")
            return False
    return True


def _open_level2(shot: int, cfg) -> xr.DataTree | None:
    try:
        import s3fs

        fs = _make_fs(cfg["level2_endpoint"])
        path = f"{cfg['level2_path']}/{shot}.{cfg['level2_ext']}"
        store = s3fs.S3Map(path, s3=fs)
        return xr.open_datatree(store, engine="zarr", chunks=None, consolidated=True)
    except Exception as e:
        logger.warning(f"Failed to open level2 zarr for shot {shot}: {e}")
        return None


def _open_level1_efm(shot: int, cfg) -> xr.Dataset | None:
    try:
        import s3fs

        fs = _make_fs(cfg["level1_endpoint"])
        path = f"{cfg['level1_path']}/{shot}.{cfg['level2_ext']}"
        store = s3fs.S3Map(path, s3=fs)
        return xr.open_zarr(store, group=cfg["level1_efm_group"], chunks=None, consolidated=True)
    except Exception as e:
        logger.warning(f"Failed to open level1 EFM zarr for shot {shot}: {e}")
        return None


def _lcfs_crossing_radius(r_from_axis: np.ndarray, psi_n_from_axis: np.ndarray) -> float:
    """Midplane radius where psi_n first crosses 1, walking away from the axis.

    Both arrays must be ordered starting at the axis and moving outward
    (monotonic R, increasing distance). Returns NaN if psi_n never reaches 1.
    """
    above = psi_n_from_axis >= 1.0
    if not above.any():
        return np.nan
    idx = int(np.argmax(above))
    if idx == 0:
        return float(r_from_axis[0])
    r0, r1 = float(r_from_axis[idx - 1]), float(r_from_axis[idx])
    p0, p1 = float(psi_n_from_axis[idx - 1]), float(psi_n_from_axis[idx])
    if p1 == p0:
        return r1
    return r0 + (1.0 - p0) * (r1 - r0) / (p1 - p0)


def _map_thomson_midplane(
    ts_r: np.ndarray,
    psi_2d: np.ndarray,
    z_grid: np.ndarray,
    r_grid: np.ndarray,
    psi_axis: float,
    psi_bry: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map Thomson R positions to psi_n and rho at the midplane.

    Parameters
    ----------
    ts_r : (n_channels,) R positions of Thomson channels [m]
    psi_2d : (n_z, n_r) poloidal flux at this time step [Wb/rad]
    z_grid : (n_z,) vertical grid [m]
    r_grid : (n_r,) radial grid [m]
    psi_axis : scalar - poloidal flux at the magnetic axis
    psi_bry : scalar - poloidal flux at the plasma boundary

    Returns
    -------
    psi_n : (n_channels,) normalised poloidal flux (0 = axis, 1 = boundary)
    rho : (n_channels,) normalised minor radius - midplane distance from the
        magnetic axis divided by the axis-to-LCFS distance on the same side
        (inboard/outboard), so 0 = axis and 1 = LCFS on both sides
    """
    nan_out = np.full(len(ts_r), np.nan, dtype=np.float32)
    z_mid_idx = int(np.argmin(np.abs(z_grid)))
    psi_midplane = psi_2d[z_mid_idx, :]  # (n_r,)

    denom = psi_bry - psi_axis
    if np.abs(denom) < 1e-10:
        return nan_out, nan_out.copy()

    # The equilibrium psi map can be NaN outside the converged region, so do
    # all midplane work on the finite subset of grid points
    finite = np.isfinite(psi_midplane)
    if finite.sum() < 4:
        return nan_out, nan_out.copy()
    r_f = r_grid[finite]
    psi_f = psi_midplane[finite]

    psi_ts = np.interp(ts_r, r_f, psi_f)
    psi_n = ((psi_ts - psi_axis) / denom).astype(np.float32)

    # Magnetic axis radius from the midplane psi_n minimum, parabola-refined
    # since the equilibrium grid is coarse (a few cm)
    psi_n_mid = (psi_f - psi_axis) / denom
    i_axis = int(np.argmin(psi_n_mid))
    r_axis = float(r_f[i_axis])
    if 0 < i_axis < len(r_f) - 1:
        p_m, p_0, p_p = psi_n_mid[i_axis - 1], psi_n_mid[i_axis], psi_n_mid[i_axis + 1]
        curv = p_m - 2 * p_0 + p_p
        if curv > 0:
            r_axis += 0.5 * (p_m - p_p) / curv * float(r_f[i_axis + 1] - r_f[i_axis - 1]) / 2.0

    # LCFS midplane radii on each side of the axis
    r_lcfs_out = _lcfs_crossing_radius(r_f[i_axis:], psi_n_mid[i_axis:])
    r_lcfs_in = _lcfs_crossing_radius(r_f[i_axis::-1], psi_n_mid[i_axis::-1])

    rho = nan_out.copy()
    outboard = ts_r >= r_axis
    if np.isfinite(r_lcfs_out) and r_lcfs_out > r_axis:
        rho[outboard] = (ts_r[outboard] - r_axis) / (r_lcfs_out - r_axis)
    if np.isfinite(r_lcfs_in) and r_lcfs_in < r_axis:
        rho[~outboard] = (r_axis - ts_r[~outboard]) / (r_axis - r_lcfs_in)

    return psi_n, rho


def _make_params(shot: int, dt: xr.DataTree, timebase: np.ndarray) -> PhysicsMethodParams:
    """Wrap an already-opened level2 DataTree in a PhysicsMethodParams."""
    conn = XarrayDataConnection(shot, dt)
    return PhysicsMethodParams(
        shot_id=shot,
        tokamak=Tokamak.MAST,
        disruption_time=None,
        data_conn=conn,
        times=timebase,
    )


class MASTDataWorkflow(DataWorkflow):
    """MAST data workflow for creating and processing datasets.

    Reads 0D equilibrium and global signals from the MAST open-access level2
    Zarr store (https://s3.echo.stfc.ac.uk/mast/level2/shots/{shot}.zarr).
    For Thomson scattering profiles, the workflow reads the raw channel data
    via disruption_py and converts to (rho, time), where rho is the
    normalized minor radius from the midplane equilibrium (see
    _map_thomson_midplane), using the 2D psi grid combined with
    psi_axis/psi_boundary from the level1 EFM Zarr. Profiles are fit and
    stored in rho, not psi_n, because psi_n squishes the core in real space.

    Power balance signals produced:
        Wtot_MJ, Ip_MA, B0, R0, kappa, a_minor, ne20_line_avg,
        P_oh_MW, P_NBI_MW, P_ECRH_MW (=0), P_ICRF_MW (=0), P_LH_MW (=0),
        P_rad_MW

    Profile transfer signals produced:
        Te_keV_rho, ne20_rho, Ip_MA, B0, betan, ne20_edge, R0,
        a_minor, kappa, delta_top, delta_bot, Wtot_MJ
    """

    # Normalize each slice to O(1) before fitting (see _extract_fit_input)
    fit_scale_per_slice = True

    def __init__(
        self,
        ds_name: str,
        shotlist_file: Path | str | None,
        data_assembly_dir: Path | str,
        max_num_shots: int | None = None,
        cluster_config=None,
        fit_workers: int = 1,
    ):
        self.config = config
        ds_cfg = self.config["data_sources"]
        self.level2_cfg = ds_cfg
        self.level1_cfg = ds_cfg

        prof_cfg = self.config["profile_fitting"]
        self.gp_fit_rho = np.linspace(
            prof_cfg["rho_min"],
            prof_cfg["rho_max"],
            prof_cfg["num_rho_points"],
        )
        self.min_ts_points = int(prof_cfg["min_ts_points"])
        self.fit_min_points = self.min_ts_points
        self.debug_plot_dir = None

        super().__init__(
            ds_name,
            shotlist_file,
            data_assembly_dir,
            max_num_shots=max_num_shots,
            min_shot_duration=self.config["shot_filters"]["min_duration"],
            cluster_config=cluster_config,
            fit_workers=fit_workers,
        )

        self.filter_config = {
            "Wtot_MJ": {"min": 0.0005, "max": 2.0},
            "ne20_line_avg": {"min": 0.01, "max": 6.0},
            "betan": {"min": 0, "max": 10},
        }
        self.individual_filter_config = None

    # ------------------------------------------------------------------
    def _get_shotlist_from_source(self) -> list[int]:
        """Return shotlist from the default shotlist file.

        MAST doesn't have a SQL database accessible outside Culham, so
        the shotlist must be supplied explicitly.
        """
        if not DEFAULT_SHOTLIST_FILE.exists():
            raise FileNotFoundError(
                f"No MAST shotlist file found at {DEFAULT_SHOTLIST_FILE}. Create a text file with one shot number per line."
            )
        with open(DEFAULT_SHOTLIST_FILE) as f:
            return [int(line.strip()) for line in f if line.strip().isdigit()]

    # ------------------------------------------------------------------
    def _get_0d_dataset(self, shot: int, params: PhysicsMethodParams) -> dict | None:
        """Extract all required 0D signals.

        Two-phase:
          1. Existence check via return_xarray=True - no S3 data reads, fast fail.
          2. Fetch (.values) and interpolate only for shots that pass phase 1.
        Returns a dict of arrays on params.times.
        """
        conn = params.data_conn
        n_t = len(params.times)

        # --- Phase 1: check existence only (no .values / no S3 reads) ---
        # vloop_dynamic included because get_ohmic_parameters requires it.
        required_paths = [
            "equilibrium/time",
            "equilibrium/wmhd",
            "equilibrium/bvac_rmag",
            "equilibrium/elongation",
            "equilibrium/minor_radius",
            "equilibrium/magnetic_axis_r",
            "equilibrium/beta_tor_normal",
            "equilibrium/beta_pol",
            "equilibrium/triangularity_upper",
            "equilibrium/triangularity_lower",
            "equilibrium/vloop_dynamic",
            "summary/time",
            "summary/ip",
            "summary/line_average_n_e",
            "summary/power_radiated",
        ]
        for path in required_paths:
            try:
                conn.get_data(path, return_xarray=True)
            except Exception as e:
                logger.warning(f"Shot {shot}: {path} missing ({e}), skipping")
                return None

        # --- Phase 2: fetch data (.values triggers S3 reads) and interpolate ---
        times = params.times
        eq_time = conn.get_data("equilibrium/time")
        ip_time = conn.get_data("summary/time")

        eq_map = {
            "wmhd": "wmhd",
            "bvac_rmag": "bvac_rmag",
            "kappa": "elongation",
            "a_minor": "minor_radius",
            "rmagx": "magnetic_axis_r",
            "beta_n": "beta_tor_normal",
            "beta_p": "beta_pol",
            "tritop": "triangularity_upper",
            "tribot": "triangularity_lower",
        }
        data = {}
        for key, prop in eq_map.items():
            data[key] = MastUtilMethods.interpolate_1d(eq_time, conn.get_data(f"equilibrium/{prop}"), times)

        data["ip"] = MastUtilMethods.interpolate_1d(ip_time, conn.get_data("summary/ip"), times)
        data["n_e"] = MastUtilMethods.interpolate_1d(ip_time, conn.get_data("summary/line_average_n_e"), times)
        data["p_rad"] = MastUtilMethods.interpolate_1d(ip_time, conn.get_data("summary/power_radiated"), times)

        try:
            data["p_nbi"] = MastUtilMethods.interpolate_1d(ip_time, conn.get_data("summary/power_nbi"), times)
        except Exception as e:
            logger.warning(f"Shot {shot}: summary/power_nbi missing ({e}), filling zeros")
            data["p_nbi"] = np.zeros(n_t, dtype=np.float32)

        try:
            ohm = MastPhysicsMethods.get_ohmic_parameters(params)
            data["p_oh"] = ohm["p_oh"]
        except Exception as e:
            logger.warning(f"Shot {shot}: ohmic power calculation failed ({e}), skipping")
            return None

        return data

    # ------------------------------------------------------------------
    def _get_thomson_raw(
        self,
        shot: int,
        params: PhysicsMethodParams,
        timebase: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Get raw Thomson channel data at the TS measurement times, mapped to psi_n and rho.

        Returns (ts_time, te_eV, ne_m3, psi_n_ts, rho_ts) where the arrays are
        shaped (n_ts_time, n_channels), or None on failure. rho is the
        normalised minor radius (see _map_thomson_midplane) and is the
        coordinate the GP fits are done in; psi_n is kept for diagnostics.
        Invalid channels at a given measurement time are NaN. Only TS times
        within the shot timebase are kept.
        """
        try:
            ts_data = MastPhysicsMethods.get_ts_channels(params)
            te_da = ts_data["ts_te_eV"]  # (time, major_radius)
            ne_da = ts_data["ts_ne"]
            ts_time = te_da.coords["time"].values
            r_ts = te_da.coords["major_radius"].values
            te_raw = te_da.values  # (n_ts_time, n_ch)
            ne_raw = ne_da.values
        except Exception as e:
            logger.warning(f"Shot {shot}: failed to read Thomson data: {e}")
            return None

        # Keep only measurement times within the shot timebase
        time_mask = (ts_time >= timebase[0]) & (ts_time <= timebase[-1])
        ts_time = ts_time[time_mask]
        te_raw = te_raw[time_mask, :]
        ne_raw = ne_raw[time_mask, :]
        if len(ts_time) == 0:
            logger.warning(f"Shot {shot}: no Thomson measurement times within shot timebase")
            return None

        efm = _open_level1_efm(shot, self.level1_cfg)
        if efm is None:
            return None
        try:
            psi_axis_arr = efm["psi_axis"].values
            psi_bry_arr = efm["psi_boundary"].values
            efm_time = efm.coords["time"].values
            psirz = efm["psirz"].values  # (T_eq, n_z, n_r)
            z_grid = efm.coords["profile_z"].values
            r_grid_eq = efm.coords["profile_r"].values
        except Exception as e:
            logger.warning(f"Shot {shot}: failed to read EFM psi data: {e}")
            return None

        n_ch = len(r_ts)
        n_t = len(ts_time)
        te_out = np.full((n_t, n_ch), np.nan, dtype=np.float32)
        ne_out = np.full((n_t, n_ch), np.nan, dtype=np.float32)
        psi_n_out = np.full((n_t, n_ch), np.nan, dtype=np.float32)
        rho_out = np.full((n_t, n_ch), np.nan, dtype=np.float32)

        psi_ax_all = interp1d(efm_time, psi_axis_arr, bounds_error=False, fill_value=np.nan)(ts_time)
        psi_br_all = interp1d(efm_time, psi_bry_arr, bounds_error=False, fill_value=np.nan)(ts_time)
        t_eq_indices = np.argmin(np.abs(efm_time[:, None] - ts_time[None, :]), axis=0)

        for i in range(n_t):
            psi_ax = float(psi_ax_all[i])
            psi_br = float(psi_br_all[i])
            if not (np.isfinite(psi_ax) and np.isfinite(psi_br)):
                continue

            psi_2d = psirz[t_eq_indices[i], :, :]  # (n_z, n_r)
            psi_n_ts, rho_ts = _map_thomson_midplane(r_ts, psi_2d, z_grid, r_grid_eq, psi_ax, psi_br)

            te_at_t = te_raw[i, :]
            ne_at_t = ne_raw[i, :]

            valid = (
                np.isfinite(te_at_t)
                & np.isfinite(ne_at_t)
                & np.isfinite(rho_ts)
                & (rho_ts >= 0)
                & (rho_ts <= 1.05)
                & (te_at_t > 0)
                & (ne_at_t > 0)
            )
            psi_n_out[i, valid] = psi_n_ts[valid]
            rho_out[i, valid] = rho_ts[valid]
            te_out[i, valid] = te_at_t[valid]
            ne_out[i, valid] = ne_at_t[valid]

        return ts_time, te_out, ne_out, psi_n_out, rho_out

    # ------------------------------------------------------------------
    def _extract_fit_input(
        self,
        te_eV: np.ndarray,
        ne_m3: np.ndarray,
        rho_ts: np.ndarray,
    ) -> ShotFitInput:
        """Build GP fit input arrays from raw Thomson channel data.

        Converts to standard units (Te [keV], ne [1e20 m^-3]) and attaches
        synthetic 10% fractional errors with a floor of 0.01 [keV or 1e20 m^-3].
        The per-slice max normalization and the min_ts_points validity check
        are applied at fit time (fit_scale_per_slice / fit_min_points).

        Parameters
        ----------
        te_eV, ne_m3, rho_ts : (n_t, n_ch) - NaN where channel invalid.
        """
        arrays = {}
        for variable, data_y_all in [("te", te_eV / 1e3), ("ne", ne_m3 / 1e20)]:
            err_y_all = np.where(np.isfinite(data_y_all), 0.1 * np.abs(data_y_all), np.nan)
            err_y_all = np.where(err_y_all < 0.01, 0.01, err_y_all)
            arrays[f"{variable}_y"] = data_y_all
            arrays[f"{variable}_err"] = err_y_all

        return ShotFitInput(x=rho_ts, **arrays)

    def _checked_fit_input(
        self,
        shot: int,
        te_eV: np.ndarray,
        ne_m3: np.ndarray,
        rho_ts: np.ndarray,
    ) -> ShotFitInput | None:
        """Extract fit inputs, skipping shots the GP fit could only return all NaN for
        (e.g. when the rho mapping failed and ts_rho is all NaN)."""
        fit_input = self._extract_fit_input(te_eV, ne_m3, rho_ts)
        if not fit_input.has_fittable_points():
            logger.warning(f"Shot {shot}: no finite (rho, te, ne) channel data to fit, skipping")
            return None
        return fit_input

    # ------------------------------------------------------------------
    def _staging_path(self, shot: int) -> Path:
        return self.fit_staging_dir / f"{shot}_staging.nc"

    def _prepare_shot(self, shot: int) -> ShotFitInput | None:  # noqa: PLR0911 - one early return per validation failure
        """Retrieve and stage source data for one shot, returning GP fit inputs.

        Reads the 0D signals and raw Thomson channel data from the MAST S3
        Zarr stores and caches them as netCDF in fit_staging_dir, so restarts
        (and the later assembly step) don't hit S3 again. Returns None if the
        shot has no valid data.
        """
        staging_path = self._staging_path(shot)
        self.fit_staging_dir.mkdir(parents=True, exist_ok=True)

        if staging_path.exists():
            logger.info(f"Using staged source data for shot {shot}")
            ds_staging = xr.load_dataset(staging_path)
            return self._checked_fit_input(
                shot,
                ds_staging["ts_te_eV"].values,
                ds_staging["ts_ne_m3"].values,
                ds_staging["ts_rho"].values,
            )

        if not _check_required_signals(shot, self.level2_cfg):
            return None

        dt = _open_level2(shot, self.level2_cfg)
        if dt is None:
            return None

        try:
            summ = dt["summary"].ds
            summ_time = summ.coords["time"].values
            ip_vals = summ["ip"].values
            min_ip = self.config["shot_filters"]["min_ip"]
            ip_mask = np.abs(ip_vals) > min_ip
            if ip_mask.sum() < 2:
                logger.warning(f"Shot {shot}: no valid IP above {min_ip} A, skipping")
                return None
            t_start = float(summ_time[ip_mask][0])
            t_end = float(summ_time[ip_mask][-1])
            if (t_end - t_start) < self.config["shot_filters"]["min_duration"]:
                logger.warning(f"Shot {shot}: plasma duration too short, skipping")
                return None
        except Exception as e:
            logger.warning(f"Shot {shot}: failed to determine timebase: {e}")
            return None

        timebase = make_uniform_1khz_timebase(t_end)
        params = _make_params(shot, dt, timebase)

        raw_0d = self._get_0d_dataset(shot, params)
        if raw_0d is None:
            return None

        raw_ts = self._get_thomson_raw(shot, params, timebase)
        if raw_ts is None:
            logger.warning(f"Shot {shot}: failed to get raw Thomson data, skipping")
            return None
        ts_time, te_eV, ne_m3, psi_n_ts, rho_ts = raw_ts

        ds_staging = xr.Dataset(
            data_vars={
                **{name: (("time",), np.asarray(vals, dtype=np.float32)) for name, vals in raw_0d.items()},
                "ts_te_eV": (("ts_time", "channel"), te_eV),
                "ts_ne_m3": (("ts_time", "channel"), ne_m3),
                "ts_psi_n": (("ts_time", "channel"), psi_n_ts),
                "ts_rho": (("ts_time", "channel"), rho_ts),
            },
            coords={
                "time": timebase,
                "ts_time": ts_time,
            },
        )
        ds_staging.to_netcdf(staging_path)

        return self._checked_fit_input(shot, te_eV, ne_m3, rho_ts)

    # ------------------------------------------------------------------
    def _assemble_shot(self, shot: int, fit_output: ShotFitOutput) -> bool:
        """Combine staged source data and GP fit results into the raw data file."""
        staging_path = self._staging_path(shot)
        ds_path = Path(self.raw_data_dir) / f"{shot}.nc"

        ds_staging = xr.load_dataset(staging_path)
        timebase = ds_staging["time"].values
        ts_time = ds_staging["ts_time"].values
        te_eV = ds_staging["ts_te_eV"].values
        ne_m3 = ds_staging["ts_ne_m3"].values
        rho_ts = ds_staging["ts_rho"].values
        raw_0d = {name: ds_staging[name].values for name in ds_staging.data_vars if ds_staging[name].dims == ("time",)}

        Te_keV_rho = fit_output.te_fit.astype(np.float32)
        ne20_rho = fit_output.ne_fit.astype(np.float32)

        # Put the fitted profiles on the 1 kHz timebase using previous value fill,
        # consistent with the C-Mod workflow (no interpolation in time)
        ds_profiles = xr.Dataset(
            data_vars={
                "Te_keV_rho": (("time", "rho"), Te_keV_rho),
                "ne20_rho": (("time", "rho"), ne20_rho),
            },
            coords={
                "time": ts_time,
                "rho": self.gp_fit_rho.astype(np.float32),
            },
        )
        ds_profiles = ds_profiles.reindex(time=timebase, method="ffill")

        data_vars = {}
        for name, vals in raw_0d.items():
            data_vars[name] = ([TIME_DIM], vals.astype(np.float32))

        data_vars["Te_keV_rho"] = ([TIME_DIM, "rho"], ds_profiles["Te_keV_rho"].values)
        data_vars["ne20_rho"] = ([TIME_DIM, "rho"], ds_profiles["ne20_rho"].values)

        coords = {
            TIME_DIM: np.arange(len(timebase)),
            TIME_COORD: (TIME_DIM, timebase.astype(np.float32)),
            "rho": self.gp_fit_rho.astype(np.float32),
        }

        ds = xr.Dataset(data_vars, coords=coords)
        ds = ds.expand_dims(shot=[shot])

        ds_standardized = self.standardize_signal_names(ds)
        if ds_standardized is None:
            logger.warning(f"Standardization failed for shot {shot}, skipping")
            return False

        ds_standardized.to_netcdf(ds_path)
        logger.info(f"Saved raw dataset for shot {shot} to {ds_path}")

        # Only make fit diagnostic plots for shots that are kept
        try:
            self._debug_plot_profiles(
                shot,
                ts_time,
                te_eV / 1e3,
                ne_m3 / 1e20,
                rho_ts,
                Te_keV_rho,
                ne20_rho,
                self.debug_plot_dir,
            )
        except Exception as e:
            logger.error(f"Failed to make TS fit diagnostic plot for shot {shot}: {e}")

        # Staged source data is no longer needed once the raw file exists
        staging_path.unlink(missing_ok=True)
        return True

    # ------------------------------------------------------------------
    def _debug_plot_profiles(
        self,
        shot: int,
        ts_time: np.ndarray,
        te_keV: np.ndarray,
        ne_20: np.ndarray,
        rho_ts: np.ndarray,
        Te_out: np.ndarray,
        ne_out: np.ndarray,
        debug_plot_dir: Path | str | None = None,
    ) -> None:
        """Save a PDF of raw TS points (with error bars) vs GP fit for sampled measurement times.

        MAST has a single Thomson system, so there is no core/edge channel split.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        if debug_plot_dir is None:
            debug_plot_dir = self.data_assembly_dir / self.ds_name / "ts_fit_plots"
        debug_plot_dir = Path(debug_plot_dir)
        debug_plot_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = debug_plot_dir / f"{shot}_ts_gp_fit.pdf"

        n_t = te_keV.shape[0]
        step = max(1, n_t // 20)
        t_indices = range(0, n_t, step)

        with PdfPages(pdf_path) as pdf:
            for i_time in t_indices:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                for ax, data_y, gp_y, label, unit in [
                    (axes[0], te_keV[i_time, :], Te_out[i_time, :], "Te", "[keV]"),
                    (axes[1], ne_20[i_time, :], ne_out[i_time, :], "ne", "[1e20 m^-3]"),
                ]:
                    rho_raw = rho_ts[i_time, :]
                    valid = np.isfinite(rho_raw) & np.isfinite(data_y)
                    # Synthetic errors, matching what is used in _make_profile_dataset
                    err_y = np.where(0.1 * np.abs(data_y) < 0.01, 0.01, 0.1 * np.abs(data_y))
                    if valid.any():
                        ax.errorbar(
                            rho_raw[valid],
                            data_y[valid],
                            yerr=err_y[valid],
                            fmt="o",
                            ms=4,
                            color="tab:blue",
                            label="raw TS",
                            zorder=3,
                        )
                    gp_valid = np.isfinite(gp_y)
                    if gp_valid.any():
                        ax.plot(self.gp_fit_rho[gp_valid], gp_y[gp_valid], color="black", label="GP fit")
                    ax.set_xlabel("rho")
                    ax.set_ylabel(f"{label} {unit}")
                    ax.set_ylim(bottom=0)
                    ax.set_title(f"shot {shot}  t={ts_time[i_time]:.3f} s  n_valid={valid.sum()}")
                    ax.grid(alpha=0.3)
                    ax.legend(fontsize=8)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        logger.info(f"Saved TS fit diagnostic plot to {pdf_path}")

    # ------------------------------------------------------------------
    def make_raw_data_files(self, debug_plot_dir: Path | str | None = None):
        """Create one netCDF per shot in the raw_data directory.

        GP fitting runs in-process. If a cluster_config was provided, fitting
        is dispatched to a SLURM cluster via make_raw_data_files_distributed()
        instead. MAST data is public S3, so the "cluster" can also be the one
        this process runs on (cluster profile "local").
        """
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.debug_plot_dir = debug_plot_dir

        if self.cluster_config is not None:
            self.make_raw_data_files_distributed()
            return

        processed_shots = 0
        for i, shot in enumerate(self.shotlist):
            if i > 0 and i % GC_INTERVAL == 0:
                logger.debug(f"Forcing garbage collection after {i} processed shots")
                gc.collect()

            if self.max_num_shots is not None and processed_shots >= self.max_num_shots:
                logger.info(f"Reached maximum shots: {self.max_num_shots}")
                break

            ds_path = Path(self.raw_data_dir) / f"{shot}.nc"
            if ds_path.exists():
                logger.info(f"Raw dataset for shot {shot} already exists at {ds_path}")
                processed_shots += 1
                continue

            # Retrieve and stage source data
            fit_input = self._prepare_shot(shot)
            if fit_input is None:
                continue

            # Fit profiles at the TS measurement times only. numpy is already
            # imported by this point (this module imports it directly above
            # fit_worker), so fit_worker's own OPENBLAS_NUM_THREADS=1 setdefault
            # came too late to take effect and OpenBLAS defaults to one thread
            # per core. Cap it here instead: GP fit matrices are tiny (tens of
            # points), so multi-threaded BLAS is pure overhead, not speedup.
            with threadpool_limits(1):
                outputs = fit_batch(
                    {shot: fit_input},
                    x_star=self.gp_fit_rho,
                    min_points=self.fit_min_points,
                    scale_per_slice=self.fit_scale_per_slice,
                    num_workers=self.fit_workers,
                )

            if self._assemble_shot(shot, outputs[shot]):
                processed_shots += 1

        logger.info("Finished making MAST raw data files.")

    # ------------------------------------------------------------------
    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset | None:
        """Rename/convert signals to the POPSIM convention."""
        ds["Wtot_MJ"] = ds["wmhd"] / 1e6
        ds["Ip_MA"] = np.abs(ds["ip"]) / 1e6
        ds["B0"] = np.abs(ds["bvac_rmag"])
        ds["R0"] = ds["rmagx"]
        ds["ne20_line_avg"] = ds["n_e"] / 1e20
        ds["P_oh_MW"] = ds["p_oh"] / 1e6
        ds["P_NBI_MW"] = ds["p_nbi"] / 1e6
        ds["P_rad_MW"] = ds["p_rad"] / 1e6
        # MAST has no ECRH, ICRF, or LH
        ds["P_ECRH_MW"] = xr.zeros_like(ds["Ip_MA"])
        ds["P_ICRF_MW"] = xr.zeros_like(ds["Ip_MA"])
        ds["P_LH_MW"] = xr.zeros_like(ds["Ip_MA"])

        # Profile predictor signals - map disruption_py names to POPSIM names
        ds["betan"] = ds["beta_n"]
        ds["delta_top"] = ds["tritop"]
        ds["delta_bot"] = ds["tribot"]

        if "ne20_rho" in ds and "rho" in ds.coords:
            ds["ne20_edge"] = ds["ne20_rho"].sel(rho=0.9, method="nearest")
        else:
            ds["ne20_edge"] = xr.zeros_like(ds["Ip_MA"])

        # Expose time as a data variable (needed by organize_data profile pipeline)
        if TIME_COORD in ds.coords and TIME_COORD not in ds.data_vars:
            ds = ds.reset_coords([TIME_COORD])

        # Same set of variables as the C-Mod workflow, plus 'time' which the
        # organize_data profile pipeline needs as a data variable
        kept_vars = {
            # POWER BALANCE
            "Wtot_MJ",
            "P_oh_MW",
            "P_rad_MW",
            "P_ICRF_MW",
            "P_LH_MW",
            "P_NBI_MW",
            "P_ECRH_MW",
            # PROFILE PREDICTOR TRAINING
            "Te_keV_rho",
            "ne20_rho",
            "Ip_MA",
            "B0",
            "betan",
            "ne20_line_avg",
            "R0",
            "kappa",
            "a_minor",
            "delta_top",
            "delta_bot",
            # OTHER
            "beta_p",
            "ne20_edge",
            "time",
        }

        present = kept_vars & set(ds.data_vars)
        missing_critical = {"Te_keV_rho", "ne20_rho", "Ip_MA", "Wtot_MJ"} - present
        if missing_critical:
            shot_id = ds["shot"].item() if "shot" in ds else "unknown"
            logger.warning(f"Shot {shot_id}: missing critical signals {missing_critical}")
            return None

        for signal in ["Te_keV_rho", "ne20_rho", "Ip_MA"]:
            if signal in ds and ds[signal].isnull().all():
                shot_id = ds["shot"].item() if "shot" in ds else "unknown"
                logger.warning(f"Shot {shot_id}: {signal} is all NaN, skipping")
                return None

        ds = ds[list(present)]

        if TIME_DIM not in ds.dims:
            ds = ds.rename_dims({"time": TIME_DIM})
        if EPISODE_DIM not in ds.dims:
            ds = ds.rename_dims({"shot": EPISODE_DIM})
        if TIME_COORD not in ds.coords:
            ds = ds.rename_vars({"time": TIME_COORD})

        return ds

    # ------------------------------------------------------------------
    def device_specific_processing(self, ds: xr.Dataset) -> xr.Dataset:
        """Clip unphysical GP-fitted profile values and fill ne20_edge from profile."""
        # Clamp to non-negative, consistent with the C-Mod workflow (raw files made
        # before the fit-time clamp was added can still contain negative values)
        if "ne20_rho" in ds:
            ds["ne20_rho"] = ds["ne20_rho"].clip(min=0)
        if "Te_keV_rho" in ds:
            ds["Te_keV_rho"] = ds["Te_keV_rho"].clip(min=0)

        if "ne20_edge" in ds and "ne20_rho" in ds and "rho" in ds["ne20_rho"].dims:
            ne_edge_from_profile = ds["ne20_rho"].sel(rho=0.9, method="nearest")
            ds["ne20_edge"] = ds["ne20_edge"].where(
                ds["ne20_edge"].notnull() & (ds["ne20_edge"] > 0.001),
                ne_edge_from_profile,
            )

        return ds

    # ------------------------------------------------------------------
    def device_specific_culling(self, ds: xr.Dataset) -> bool:
        """Return True if shot should be excluded."""
        if "P_rad_MW" in ds:
            if ds["P_rad_MW"].isnull().all() or ds["P_rad_MW"].mean().item() < 0.005:
                shot_id = ds.shot.values[0] if "shot" in ds else "unknown"
                logger.info(f"Culling shot {shot_id}: P_rad missing or near zero")
                return True
        return False
