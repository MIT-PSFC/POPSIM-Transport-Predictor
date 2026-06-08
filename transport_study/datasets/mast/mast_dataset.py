"""Makes the raw MAST dataset from the STFC ECHO S3 open-access Zarr store."""

from pathlib import Path

import netCDF4  # noqa: F401
import numpy as np
import s3fs
import xarray as xr
from disruption_py.core.physics_method.params import PhysicsMethodParams
from disruption_py.inout.xr import XarrayDataConnection
from disruption_py.machine.mast.physics import MastPhysicsMethods
from disruption_py.machine.mast.util import MastUtilMethods
from disruption_py.machine.tokamak import Tokamak
from dynaconf import Dynaconf
from loguru import logger
from scipy.interpolate import interp1d

from transport_study import EPISODE_DIM, PACKAGE_ROOT, TIME_COORD, TIME_DIM
from transport_study.datasets import make_uniform_1khz_timebase
from transport_study.datasets.workflow import DataWorkflow

DEFAULT_SHOTLIST_FILE = Path(PACKAGE_ROOT) / "datasets" / "mast" / "mast_shotlist"

config = Dynaconf(settings_files=[Path(PACKAGE_ROOT) / "datasets/mast/config.toml"])


def _ensure_verbose_level():
    """Register loguru VERBOSE level (used by disruption_py physics_method timing)."""
    try:
        logger.level("VERBOSE")
    except ValueError:
        logger.level("VERBOSE", no=5, color="<cyan>", icon="V")


def _make_fs(endpoint_url: str) -> s3fs.S3FileSystem:
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
        fs = _make_fs(cfg["level2_endpoint"])
        path = f"{cfg['level2_path']}/{shot}.{cfg['level2_ext']}"
        store = s3fs.S3Map(path, s3=fs)
        return xr.open_datatree(store, engine="zarr", chunks=None, consolidated=True)
    except Exception as e:
        logger.warning(f"Failed to open level2 zarr for shot {shot}: {e}")
        return None


def _open_level1_efm(shot: int, cfg) -> xr.Dataset | None:
    try:
        fs = _make_fs(cfg["level1_endpoint"])
        path = f"{cfg['level1_path']}/{shot}.{cfg['level2_ext']}"
        store = s3fs.S3Map(path, s3=fs)
        return xr.open_zarr(store, group=cfg["level1_efm_group"], chunks=None, consolidated=True)
    except Exception as e:
        logger.warning(f"Failed to open level1 EFM zarr for shot {shot}: {e}")
        return None


def _map_thomson_to_psi_n(
    ts_r: np.ndarray,
    psi_2d: np.ndarray,
    z_grid: np.ndarray,
    r_grid: np.ndarray,
    psi_axis: float,
    psi_bry: float,
) -> np.ndarray:
    """Map Thomson R positions to normalised poloidal flux at the midplane.

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
    """
    z_mid_idx = int(np.argmin(np.abs(z_grid)))
    psi_midplane = psi_2d[z_mid_idx, :]  # (n_r,)

    psi_ts = np.interp(ts_r, r_grid, psi_midplane)

    denom = psi_bry - psi_axis
    if np.abs(denom) < 1e-10:
        return np.full(len(ts_r), np.nan, dtype=np.float32)
    psi_n = (psi_ts - psi_axis) / denom
    return psi_n.astype(np.float32)


def _make_params(shot: int, dt: xr.DataTree, timebase: np.ndarray) -> PhysicsMethodParams:
    """Wrap an already-opened level2 DataTree in a PhysicsMethodParams."""
    _ensure_verbose_level()
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
    via disruption_py and converts to (psi_n, time) using the 2D psi grid
    combined with psi_axis/psi_boundary from the level1 EFM Zarr.

    Power balance signals produced:
        Wtot_MJ, Ip_MA, B0, R0, kappa, a_minor, ne20_line_avg,
        P_oh_MW, P_NBI_MW, P_ECRH_MW (=0), P_ICRF_MW (=0), P_LH_MW (=0),
        P_rad_MW

    Profile transfer signals produced:
        Te_keV_psi, ne20_psi, Ip_MA, B0, betan, ne20_edge, R0,
        a_minor, kappa, delta_top, delta_bot, Wtot_MJ
    """

    def __init__(
        self,
        ds_name: str,
        shotlist_file: Path | str | None,
        data_assembly_dir: Path | str,
        max_num_shots: int | None = None,
    ):
        self.config = config
        ds_cfg = self.config["data_sources"]
        self.level2_cfg = ds_cfg
        self.level1_cfg = ds_cfg

        prof_cfg = self.config["profile_fitting"]
        self.psi_n_grid = np.linspace(
            prof_cfg["psi_n_min"],
            prof_cfg["psi_n_max"],
            prof_cfg["num_psi_points"],
        )

        super().__init__(
            ds_name,
            shotlist_file,
            data_assembly_dir,
            max_num_shots=max_num_shots,
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
            "equilibrium/li",
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
            "li": "li",
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
    def _get_profile_dataset(
        self,
        shot: int,
        params: PhysicsMethodParams,
        timebase: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute Te and ne profiles on the uniform psi_n grid.

        Returns (te_psi, ne_psi) each shaped (n_t, n_psi), or None on failure.
        te_psi in keV, ne_psi in 1e20 m^-3.
        """
        try:
            # Thomson scattering raw channel data via disruption_py
            ts_data = MastPhysicsMethods.get_ts_channels(params)
            te_da = ts_data["ts_te_eV"]  # (time, major_radius)
            ne_da = ts_data["ts_ne"]
            ts_time = te_da.coords["time"].values
            r_ts = te_da.coords["major_radius"].values
            # Transpose to (major_radius, time) for per-timestep indexing
            te_raw = te_da.values.T
            ne_raw = ne_da.values.T
        except Exception as e:
            logger.warning(f"Shot {shot}: failed to read Thomson data: {e}")
            return None

        # 2D psi flux map and psi_axis/psi_boundary from level1 EFM zarr.
        # The level2 zarr equilibrium group has only 0D/1D scalars; psirz is
        # only available in the level1 EFM store.
        efm = _open_level1_efm(shot, self.level1_cfg)
        if efm is None:
            return None
        try:
            psi_axis_arr = efm["psi_axis"].values
            psi_bry_arr = efm["psi_boundary"].values
            efm_time = efm.coords["time"].values
            # psirz shape from level1 EFM: (time, profile_z, profile_r)
            psirz = efm["psirz"].values  # (T_eq, n_z, n_r)
            z_grid = efm.coords["profile_z"].values
            r_grid_eq = efm.coords["profile_r"].values
        except Exception as e:
            logger.warning(f"Shot {shot}: failed to read EFM psi data: {e}")
            return None

        n_psi = len(self.psi_n_grid)
        n_t = len(timebase)
        te_out = np.full((n_t, n_psi), np.nan, dtype=np.float32)
        ne_out = np.full((n_t, n_psi), np.nan, dtype=np.float32)

        # Pre-compute psi_axis/psi_boundary at all timebase points
        psi_ax_all = interp1d(efm_time, psi_axis_arr, bounds_error=False, fill_value=np.nan)(timebase)
        psi_br_all = interp1d(efm_time, psi_bry_arr, bounds_error=False, fill_value=np.nan)(timebase)
        # Nearest-neighbour EFM and TS time indices for each timebase point
        t_eq_indices = np.argmin(np.abs(efm_time[:, None] - timebase[None, :]), axis=0)
        t_ts_indices = np.argmin(np.abs(ts_time[:, None] - timebase[None, :]), axis=0)

        for i, _t in enumerate(timebase):
            psi_ax = float(psi_ax_all[i])
            psi_br = float(psi_br_all[i])
            if not (np.isfinite(psi_ax) and np.isfinite(psi_br)):
                continue

            # psirz: (T_eq, n_z, n_r) - select nearest EFM timestep
            psi_2d = psirz[t_eq_indices[i], :, :]  # (n_z, n_r)

            psi_n_ts = _map_thomson_to_psi_n(r_ts, psi_2d, z_grid, r_grid_eq, psi_ax, psi_br)

            # te_raw/ne_raw dims: (major_radius, time)
            te_at_t = te_raw[:, t_ts_indices[i]]
            ne_at_t = ne_raw[:, t_ts_indices[i]]

            valid = (
                np.isfinite(te_at_t)
                & np.isfinite(ne_at_t)
                & np.isfinite(psi_n_ts)
                & (psi_n_ts >= 0)
                & (psi_n_ts <= 1.05)
                & (te_at_t > 0)
                & (ne_at_t > 0)
            )
            if valid.sum() < 4:
                continue

            psi_v = psi_n_ts[valid]
            te_v = te_at_t[valid] / 1e3  # eV -> keV
            ne_v = ne_at_t[valid] / 1e20  # m^-3 -> 1e20 m^-3

            sort_idx = np.argsort(psi_v)
            psi_v = psi_v[sort_idx]
            te_v = te_v[sort_idx]
            ne_v = ne_v[sort_idx]

            te_out[i, :] = np.interp(self.psi_n_grid, psi_v, te_v, left=np.nan, right=np.nan)
            ne_out[i, :] = np.interp(self.psi_n_grid, psi_v, ne_v, left=np.nan, right=np.nan)

        return te_out, ne_out

    # ------------------------------------------------------------------
    def make_raw_data_files(self):
        """Create one netCDF per shot in the raw_data directory."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        processed_shots = 0
        for shot in self.shotlist:
            if self.max_num_shots is not None and processed_shots >= self.max_num_shots:
                logger.info(f"Reached maximum shots: {self.max_num_shots}")
                break

            ds_path = Path(self.raw_data_dir) / f"{shot}.nc"
            if ds_path.exists():
                logger.info(f"Raw dataset for shot {shot} already exists at {ds_path}")
                processed_shots += 1
                continue

            if not _check_required_signals(shot, self.level2_cfg):
                continue

            dt = _open_level2(shot, self.level2_cfg)
            if dt is None:
                continue

            try:
                summ = dt["summary"].ds
                summ_time = summ.coords["time"].values
                ip_vals = summ["ip"].values
                min_ip = self.config["shot_filters"]["min_ip"]
                ip_mask = np.abs(ip_vals) > min_ip
                if ip_mask.sum() < 2:
                    logger.warning(f"Shot {shot}: no valid IP above {min_ip} A, skipping")
                    continue
                t_start = float(summ_time[ip_mask][0])
                t_end = float(summ_time[ip_mask][-1])
                if (t_end - t_start) < self.config["shot_filters"]["min_duration"]:
                    logger.warning(f"Shot {shot}: plasma duration too short, skipping")
                    continue
            except Exception as e:
                logger.warning(f"Shot {shot}: failed to determine timebase: {e}")
                continue

            timebase = make_uniform_1khz_timebase(t_end)
            params = _make_params(shot, dt, timebase)

            raw_0d = self._get_0d_dataset(shot, params)
            if raw_0d is None:
                continue

            profiles = self._get_profile_dataset(shot, params, timebase)
            if profiles is None:
                logger.warning(f"Shot {shot}: failed to compute Thomson profiles, skipping")
                continue
            te_psi, ne_psi = profiles

            data_vars = {}
            for name, vals in raw_0d.items():
                data_vars[name] = ([TIME_DIM], vals.astype(np.float32))

            data_vars["Te_keV_psi"] = ([TIME_DIM, "psi_n"], te_psi)
            data_vars["ne20_psi"] = ([TIME_DIM, "psi_n"], ne_psi)

            coords = {
                TIME_DIM: np.arange(len(timebase)),
                TIME_COORD: (TIME_DIM, timebase.astype(np.float32)),
                "psi_n": self.psi_n_grid.astype(np.float32),
            }

            ds = xr.Dataset(data_vars, coords=coords)
            ds = ds.expand_dims(shot=[shot])

            ds_standardized = self.standardize_signal_names(ds)
            if ds_standardized is None:
                logger.warning(f"Standardization failed for shot {shot}, skipping")
                continue

            ds_standardized.to_netcdf(ds_path)
            logger.info(f"Saved raw dataset for shot {shot} to {ds_path}")
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
        ds["beta_pol"] = ds["beta_p"]
        ds["delta_top"] = ds["tritop"]
        ds["delta_bot"] = ds["tribot"]

        if "ne20_psi" in ds and "psi_n" in ds.coords:
            ds["ne20_edge"] = ds["ne20_psi"].sel(psi_n=0.9, method="nearest")
        else:
            ds["ne20_edge"] = xr.zeros_like(ds["Ip_MA"])

        # Expose time as a data variable (needed by organize_data profile pipeline)
        if TIME_COORD in ds.coords and TIME_COORD not in ds.data_vars:
            ds = ds.reset_coords([TIME_COORD])

        kept_vars = {
            "Wtot_MJ",
            "Ip_MA",
            "B0",
            "R0",
            "a_minor",
            "kappa",
            "ne20_line_avg",
            "P_oh_MW",
            "P_NBI_MW",
            "P_ECRH_MW",
            "P_ICRF_MW",
            "P_LH_MW",
            "P_rad_MW",
            "Te_keV_psi",
            "ne20_psi",
            "betan",
            "ne20_edge",
            "delta_top",
            "delta_bot",
            "beta_pol",
            "li",
            "time",
        }

        present = kept_vars & set(ds.data_vars)
        missing_critical = {"Te_keV_psi", "ne20_psi", "Ip_MA", "Wtot_MJ"} - present
        if missing_critical:
            shot_id = ds["shot"].item() if "shot" in ds else "unknown"
            logger.warning(f"Shot {shot_id}: missing critical signals {missing_critical}")
            return None

        for signal in ["Te_keV_psi", "ne20_psi", "Ip_MA"]:
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
        """Clip unphysical values and fill ne20_edge from profile."""
        if "ne20_psi" in ds:
            ds["ne20_psi"] = ds["ne20_psi"].where(ds["ne20_psi"] > 0)
        if "Te_keV_psi" in ds:
            ds["Te_keV_psi"] = ds["Te_keV_psi"].where(ds["Te_keV_psi"] > 0)

        if "ne20_edge" in ds and "ne20_psi" in ds:
            ne_edge_from_profile = ds["ne20_psi"].sel(psi_n=0.9, method="nearest")
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
