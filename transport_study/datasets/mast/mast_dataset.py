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
from transport_study.datasets.cmod.gp_fit import fit_gp_hyperparameters, gp_profile
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
        self.gp_fit_psi = np.linspace(
            prof_cfg["psi_n_min"],
            prof_cfg["psi_n_max"],
            prof_cfg["num_psi_points"],
        )
        self.min_ts_points = int(prof_cfg["min_ts_points"])

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
    def _get_thomson_raw(
        self,
        shot: int,
        params: PhysicsMethodParams,
        timebase: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Get raw Thomson channel data mapped to psi_n coordinates.

        Returns (te_eV, ne_m3, psi_n_ts) each shaped (n_t, n_channels), or None
        on failure. Invalid channels at a given timestep are NaN.
        """
        try:
            ts_data = MastPhysicsMethods.get_ts_channels(params)
            te_da = ts_data["ts_te_eV"]  # (time, major_radius)
            ne_da = ts_data["ts_ne"]
            ts_time = te_da.coords["time"].values
            r_ts = te_da.coords["major_radius"].values
            # Transpose to (n_ch, n_ts_time) for per-timestep indexing
            te_raw = te_da.values.T
            ne_raw = ne_da.values.T
        except Exception as e:
            logger.warning(f"Shot {shot}: failed to read Thomson data: {e}")
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
        n_t = len(timebase)
        te_out = np.full((n_t, n_ch), np.nan, dtype=np.float32)
        ne_out = np.full((n_t, n_ch), np.nan, dtype=np.float32)
        psi_n_out = np.full((n_t, n_ch), np.nan, dtype=np.float32)

        psi_ax_all = interp1d(efm_time, psi_axis_arr, bounds_error=False, fill_value=np.nan)(timebase)
        psi_br_all = interp1d(efm_time, psi_bry_arr, bounds_error=False, fill_value=np.nan)(timebase)
        t_eq_indices = np.argmin(np.abs(efm_time[:, None] - timebase[None, :]), axis=0)
        t_ts_indices = np.argmin(np.abs(ts_time[:, None] - timebase[None, :]), axis=0)

        for i in range(n_t):
            psi_ax = float(psi_ax_all[i])
            psi_br = float(psi_br_all[i])
            if not (np.isfinite(psi_ax) and np.isfinite(psi_br)):
                continue

            psi_2d = psirz[t_eq_indices[i], :, :]  # (n_z, n_r)
            psi_n_ts = _map_thomson_to_psi_n(r_ts, psi_2d, z_grid, r_grid_eq, psi_ax, psi_br)

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
            psi_n_out[i, valid] = psi_n_ts[valid]
            te_out[i, valid] = te_at_t[valid]
            ne_out[i, valid] = ne_at_t[valid]

        return te_out, ne_out, psi_n_out

    # ------------------------------------------------------------------
    def _make_profile_dataset(
        self,
        te_eV: np.ndarray,
        ne_m3: np.ndarray,
        psi_n_ts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """GP fit raw Thomson channel data onto self.gp_fit_psi grid.

        Parameters
        ----------
        te_eV, ne_m3, psi_n_ts : (n_t, n_ch) - NaN where channel invalid.

        Returns
        -------
        (Te_keV_psi, ne20_psi) each (n_t, n_psi) on self.gp_fit_psi.
        """
        n_t, _ = te_eV.shape
        n_psi = len(self.gp_fit_psi)
        Te_out = np.full((n_t, n_psi), np.nan, dtype=np.float32)
        ne_out = np.full((n_t, n_psi), np.nan, dtype=np.float32)

        te_keV = te_eV / 1e3
        ne_20 = ne_m3 / 1e20

        for data_y_all, out_arr in [(te_keV, Te_out), (ne_20, ne_out)]:
            # Synthetic 10% fractional errors with a floor of 0.01 [keV or 1e20 m^-3]
            err_y_all = np.where(np.isfinite(data_y_all), 0.1 * np.abs(data_y_all), np.nan)
            err_y_all = np.where(err_y_all < 0.01, 0.01, err_y_all)

            cached_hp = None
            for i_seed in range(n_t):
                n_valid = int(np.sum(np.isfinite(psi_n_ts[i_seed, :]) & np.isfinite(data_y_all[i_seed, :])))
                if n_valid < self.min_ts_points:
                    continue
                seed_scale = float(np.nanmax(data_y_all[i_seed, :]))
                if not np.isfinite(seed_scale) or seed_scale < 1e-6:
                    continue
                cached_hp = fit_gp_hyperparameters(
                    data_X=psi_n_ts[i_seed, :],
                    data_y=data_y_all[i_seed, :] / seed_scale,
                    err_y=err_y_all[i_seed, :] / seed_scale,
                )
                if cached_hp is not None:
                    break

            for i_time in range(n_t):
                n_valid = int(np.sum(np.isfinite(psi_n_ts[i_time, :]) & np.isfinite(data_y_all[i_time, :])))
                if n_valid < self.min_ts_points:
                    continue
                # Normalize to O(1) before GP fit to prevent amplitude collapse
                # when channels don't cover the full psi_n range.
                scale = float(np.nanmax(data_y_all[i_time, :]))
                if not np.isfinite(scale) or scale < 1e-6:
                    continue
                y_star, _, _, _ = gp_profile(
                    data_X=psi_n_ts[i_time, :],
                    data_y=data_y_all[i_time, :] / scale,
                    err_y=err_y_all[i_time, :] / scale,
                    X_star=self.gp_fit_psi,
                    calc_gradient=False,
                    hyperparams=cached_hp,
                    optimize_hyperparams=cached_hp is None,
                )
                if y_star is None:
                    continue
                out_arr[i_time, :] = np.asarray(y_star).ravel() * scale

        return Te_out, ne_out

    # ------------------------------------------------------------------
    def _debug_plot_profiles(
        self,
        shot: int,
        te_keV: np.ndarray,
        ne_20: np.ndarray,
        psi_n_ts: np.ndarray,
        Te_out: np.ndarray,
        ne_out: np.ndarray,
        debug_plot_dir: Path | None = "/home/zkeith/proj/popsim_dirs/POPSIM-Transport-Predictor/scratch/datasets/mast/debug_plots",
    ) -> None:
        """Save a PDF of raw TS points vs GP fit for sampled timesteps."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        debug_plot_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = debug_plot_dir / f"{shot}_ts_gp_debug.pdf"

        n_t = te_keV.shape[0]
        step = max(1, n_t // 20)
        t_indices = range(0, n_t, step)

        with PdfPages(pdf_path) as pdf:
            for i_time in t_indices:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                for ax, data_y, gp_y, label, unit in [
                    (axes[0], te_keV[i_time, :], Te_out[i_time, :], "Te", "[keV]"),
                    (axes[1], ne_20[i_time, :], ne_out[i_time, :], "ne", "[1e20 m^-3]"),
                ]:
                    psi_raw = psi_n_ts[i_time, :]
                    valid = np.isfinite(psi_raw) & np.isfinite(data_y)
                    ax.scatter(psi_raw[valid], data_y[valid], s=14, zorder=3, label="raw TS")
                    gp_valid = np.isfinite(gp_y)
                    if gp_valid.any():
                        ax.plot(self.gp_fit_psi[gp_valid], gp_y[gp_valid], label="GP fit")
                    ax.set_xlabel("psi_n")
                    ax.set_ylabel(f"{label} {unit}")
                    ax.set_title(f"shot {shot}  t_idx={i_time}  n_valid={valid.sum()}")
                    ax.legend(fontsize=8)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        logger.info(f"Saved debug plot to {pdf_path}")

    # ------------------------------------------------------------------
    def make_raw_data_files(self, debug_plot_dir: Path | str | None = None):
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

            raw_ts = self._get_thomson_raw(shot, params, timebase)
            if raw_ts is None:
                logger.warning(f"Shot {shot}: failed to get raw Thomson data, skipping")
                continue
            te_eV, ne_m3, psi_n_ts = raw_ts

            Te_keV_psi, ne20_psi = self._make_profile_dataset(te_eV, ne_m3, psi_n_ts)

            debug_plot_dir = "/home/zkeith/proj/popsim_dirs/POPSIM-Transport-Predictor/scratch/datasets/mast/debug_plots"
            if debug_plot_dir is not None:
                self._debug_plot_profiles(
                    shot,
                    te_eV / 1e3,
                    ne_m3 / 1e20,
                    psi_n_ts,
                    Te_keV_psi,
                    ne20_psi,
                    Path(debug_plot_dir),
                )

            n_ch = te_eV.shape[1]
            data_vars = {}
            for name, vals in raw_0d.items():
                data_vars[name] = ([TIME_DIM], vals.astype(np.float32))

            data_vars["ts_te_eV"] = ([TIME_DIM, "ts_channel"], te_eV)
            data_vars["ts_ne"] = ([TIME_DIM, "ts_channel"], ne_m3)
            data_vars["ts_psi_n"] = ([TIME_DIM, "ts_channel"], psi_n_ts)
            data_vars["Te_keV_psi"] = ([TIME_DIM, "psi_n"], Te_keV_psi)
            data_vars["ne20_psi"] = ([TIME_DIM, "psi_n"], ne20_psi)

            coords = {
                TIME_DIM: np.arange(len(timebase)),
                TIME_COORD: (TIME_DIM, timebase.astype(np.float32)),
                "psi_n": self.gp_fit_psi.astype(np.float32),
                "ts_channel": np.arange(n_ch),
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
            "ts_te_eV",
            "ts_ne",
            "ts_psi_n",
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
        """Clip unphysical GP-fitted profile values and fill ne20_edge from profile."""
        if "ne20_psi" in ds:
            ds["ne20_psi"] = ds["ne20_psi"].where(ds["ne20_psi"] > 0)
        if "Te_keV_psi" in ds:
            ds["Te_keV_psi"] = ds["Te_keV_psi"].where(ds["Te_keV_psi"] > 0)

        if "ne20_edge" in ds and "ne20_psi" in ds and "psi_n" in ds["ne20_psi"].dims:
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
