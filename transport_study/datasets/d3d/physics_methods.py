"""Custom disruption-py physics methods for the DIII-D dataset.

Passed to disruption-py via RetrievalSettings.custom_physics_methods so the
submodule's machine files stay untouched. Also holds the uniform 1 kHz time
setting and the psi_n -> rho mapping helper.
"""

from pathlib import Path
from typing import ClassVar

import numpy as np
import xarray as xr
from disruption_py.core.physics_method.decorator import physics_method
from disruption_py.core.physics_method.errors import CalculationError
from disruption_py.core.physics_method.params import PhysicsMethodParams
from disruption_py.core.utils.math import interp1
from disruption_py.inout.mds import mdsExceptions
from disruption_py.machine.d3d.physics import D3DPhysicsMethods
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings.time_setting import (
    TimeSetting,
    TimeSettingParams,
    _postprocess,
)
from dynaconf import Dynaconf
from loguru import logger

from transport_study import PACKAGE_ROOT

config = Dynaconf(settings_files=[Path(PACKAGE_ROOT) / "datasets/d3d/config.toml"])

RHO_GRID = np.linspace(
    config["profile_grid"]["rho_min"],
    config["profile_grid"]["rho_max"],
    config["profile_grid"]["num_rho_points"],
)

# Programmed waveforms run past early plasma termination, keep them for predict-first
MIN_TIMEBASE_MS = 8000


def find_ida_path(shot: int, patterns: list[str] | None = None) -> Path | None:
    """First existing IDA file for the shot across the configured patterns, in priority order."""
    if patterns is None:
        patterns = config["data_sources"]["ida_path_patterns"]
    for pattern in patterns:
        path = Path(str(pattern).format(shot=shot))
        if path.exists():
            return path
    return None


class Uniform1kHzTimeSetting(TimeSetting):
    """Uniform 1 kHz timebase covering the EFIT range and the programmed waveforms."""

    def _get_times(self, params: TimeSettingParams) -> np.ndarray:
        (efit_time,) = params.get_dims(r"\efit_aeqdsk:ali", tree_name="_efit_tree")  # [ms]
        typical_delta = np.median(np.diff(efit_time))
        if typical_delta > 2:
            logger.warning(
                f"Shot {params.shot_id}: EFIT timebase slower than 1 kHz "
                f"(typical delta {typical_delta:.1f} ms), DISPY tree probably missing"
            )
        max_time = max(np.max(efit_time), MIN_TIMEBASE_MS)
        return _postprocess(times=np.round(np.arange(0, max_time + 1, 1), 0), units="ms")


def _rho_of_psi_n(psi_n_grid: np.ndarray, times: np.ndarray, efit_dict: dict) -> np.ndarray:
    """Map a psi_n grid to normalized midplane minor radius rho, per time slice.

    rho = (R - R_axis) / (R_lcfs - R_axis) on the outboard midplane at the
    magnetic axis height, the same definition as the C-Mod and MAST datasets
    (see _map_r_mid_to_psi_n_and_rho in disruption-py) but inverted: psi_n(R)
    along the midplane is computed from psin(z, r), the strictly increasing
    outboard branch is inverted to R(psi_n), then normalized by the psi_n = 1
    crossing radius.

    Parameters
    ----------
    psi_n_grid : np.ndarray
        Target psi_n values, shape (n_psi,)
    times : np.ndarray
        Times to evaluate at [s], shape (n_times,)
    efit_dict : dict
        Output of D3DPhysicsMethods._get_efit_dict, psin has dims (time, z, r)

    Returns
    -------
    np.ndarray
        rho values, shape (n_times, n_psi), NaN rows where the mapping fails
    """
    rgrid, zgrid = efit_dict["r"], efit_dict["z"]
    rho = np.full((len(times), len(psi_n_grid)), np.nan, dtype=np.float32)
    i_eq_arr = np.argmin(np.abs(efit_dict["time"][:, None] - times[None, :]), axis=0)
    in_range = (times >= efit_dict["time"].min()) & (times <= efit_dict["time"].max())
    for i, i_eq in enumerate(i_eq_arr):
        if not in_range[i]:
            continue
        psin_zr = efit_dict["psin"][i_eq]
        if not np.isfinite(psin_zr).all():
            continue
        # psi_n along the midplane at the magnetic axis height
        psi_n_mid = np.array([np.interp(efit_dict["zmaxis"][i_eq], zgrid, psin_zr[:, j]) for j in range(len(rgrid))])

        # Axis radius from the psi_n minimum, parabola-refined since the EFIT grid is a few cm
        i_axis = int(np.argmin(psi_n_mid))
        r_axis = float(rgrid[i_axis])
        if 0 < i_axis < len(rgrid) - 1:
            p_m, p_0, p_p = psi_n_mid[i_axis - 1], psi_n_mid[i_axis], psi_n_mid[i_axis + 1]
            curvature = p_m - 2 * p_0 + p_p
            if curvature > 0:
                dr = float(rgrid[i_axis + 1] - rgrid[i_axis - 1]) / 2.0
                r_axis += 0.5 * (p_m - p_p) / curvature * dr

        # Keep the strictly increasing outboard branch so the inversion is well defined
        psi_out, r_out = psi_n_mid[i_axis:], rgrid[i_axis:]
        non_increasing = np.where(np.diff(psi_out) <= 0)[0]
        end = non_increasing[0] + 1 if non_increasing.size else len(psi_out)
        psi_out, r_out = psi_out[:end], r_out[:end]
        if psi_out[-1] < psi_n_grid.max():
            continue  # branch does not cover the requested psi_n range

        r_lcfs = np.interp(1.0, psi_out, r_out)
        if not r_lcfs > r_axis:
            continue
        r_of_psi_n = np.interp(psi_n_grid, psi_out, r_out)
        # blue: clip tiny negatives at psi_n ~ 0 caused by the parabola-refined axis
        rho[i] = np.clip((r_of_psi_n - r_axis) / (r_lcfs - r_axis), 0.0, None)
    return rho


def _gradient_and_error(vals: np.ndarray, errs: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Numerical gradient along the last axis with propagated error."""
    grad = np.gradient(vals, x, axis=1)
    # blue: central difference error propagation, endpoints copy their neighbor.
    # No point covariance available from IDA, unlike the GP fits on C-Mod/MAST.
    grad_err = np.full_like(vals, np.nan)
    grad_err[:, 1:-1] = np.sqrt(errs[:, 2:] ** 2 + errs[:, :-2] ** 2) / (x[2:] - x[:-2])
    grad_err[:, 0], grad_err[:, -1] = grad_err[:, 1], grad_err[:, -2]
    return grad, grad_err


class D3DDatasetMethods:
    """Signals needed by the transport study that are not disruption-py built-ins."""

    PROGRAMMED_POINTNAMES: ClassVar[list[str]] = [
        "bttbt",  # programmed B0 [T]
        "bmtpwrtar",  # programmed betan target
        "dstdenp",  # programmed pedestal density
        "idtrp",  # programmed R0 [m]
        "ieeseg07",  # programmed inner gap [m]
        "idtrxbot",  # programmed bottom X point R [m]
        "idtzxbot",  # programmed bottom X point Z [m]
        "idtrxtop",  # programmed top X point R [m]
        "idtzxtop",  # programmed top X point Z [m]
    ]  # iptipp is covered by the built-in get_ip_parameters as ip_prog

    EXTENDED_EFIT_COLS: ClassVar[dict[str, str]] = {
        "betat": r"\efit_a_eqdsk:betat",
        "aminor": r"\efit_a_eqdsk:aminor",
        "tritop": r"\efit_a_eqdsk:tritop",
        "tribot": r"\efit_a_eqdsk:tribot",
        "rsurf": r"\efit_a_eqdsk:rsurf",
        "rxpt1": r"\efit_a_eqdsk:rxpt1",
        "zxpt1": r"\efit_a_eqdsk:zxpt1",
        "rxpt2": r"\efit_a_eqdsk:rxpt2",
        "zxpt2": r"\efit_a_eqdsk:zxpt2",
        "gapin": r"\efit_a_eqdsk:gapin",
    }

    @staticmethod
    @physics_method(columns=["wmhdf", "betanf", "betapf", "dssneped"], tokamak=Tokamak.D3D)
    def get_pedestal_parameters(params: PhysicsMethodParams):
        """Fast stored energy and betas from the pedestal tree, plus the pedestal density."""
        out = {}
        for col, node in [("wmhdf", r"\wmhdf"), ("betanf", r"\betanf"), ("betapf", r"\betapf")]:
            try:
                sig, t = params.get_data_with_dims(node, tree_name="pedestal")
                out[col] = interp1(t / 1e3, sig, params.times)
            except mdsExceptions.MdsException:
                params.logger.warning("pedestal node {node} missing", node=node)
                out[col] = np.full(len(params.times), np.nan)
        if not np.isfinite(out["betanf"]).any():
            # blue: efsbetan is the rt-EFIT beta_n, close enough when the pedestal calc is missing
            try:
                sig, t = params.get_data_with_dims(f"ptdata('efsbetan', {params.shot_id})")
                out["betanf"] = interp1(t / 1e3, sig, params.times)
            except mdsExceptions.MdsException:
                pass
        try:
            sig, t = params.get_data_with_dims(f"ptdata('dssneped', {params.shot_id})")
            out["dssneped"] = interp1(t / 1e3, sig, params.times)
        except mdsExceptions.MdsException:
            params.logger.warning("ptdata dssneped missing")
            out["dssneped"] = np.full(len(params.times), np.nan)
        return out

    @staticmethod
    @physics_method(columns=PROGRAMMED_POINTNAMES, tokamak=Tokamak.D3D)
    def get_programmed_parameters(params: PhysicsMethodParams):
        """PCS programmed waveforms for the predict-first trajectory optimization."""
        out = {}
        for name in D3DDatasetMethods.PROGRAMMED_POINTNAMES:
            try:
                sig, t = params.get_data_with_dims(f"ptdata('{name}', {params.shot_id})")
                out[name] = interp1(t / 1e3, sig, params.times)
            except mdsExceptions.MdsException:
                params.logger.warning("ptdata {name} missing", name=name)
                out[name] = np.full(len(params.times), np.nan)
        return out

    @staticmethod
    @physics_method(columns=list(EXTENDED_EFIT_COLS), tokamak=Tokamak.D3D)
    def get_extended_efit_parameters(params: PhysicsMethodParams):
        """A-eqdsk shape and beta signals not covered by the built-in get_efit_parameters."""
        data = {k: params.get_data(v, tree_name="_efit_tree") for k, v in D3DDatasetMethods.EXTENDED_EFIT_COLS.items()}
        efit_time = params.get_data(r"\efit_a_eqdsk:atime", tree_name="_efit_tree") / 1e3
        chisq = params.get_data(r"\efit_a_eqdsk:chisq", tree_name="_efit_tree")
        invalid = np.where(chisq > 50)  # same validity criterion as get_efit_parameters
        for values in data.values():
            values[invalid] = np.nan
        # X points read -9.99 when EFIT finds no X point, mask the whole set when either is bad
        xpt_bad = ~((data["rxpt1"] > 0) & (data["rxpt2"] > 0))
        for k in ["rxpt1", "zxpt1", "rxpt2", "zxpt2"]:
            data[k][xpt_bad] = np.nan
        return {k: interp1(efit_time, v, params.times) for k, v in data.items()}

    @staticmethod
    @physics_method(
        columns=[
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
        ],
        tokamak=Tokamak.D3D,
    )
    def get_ida_profiles(params: PhysicsMethodParams):
        """IDA Te/ne profiles on the native psi_n grid and mapped to a uniform rho grid.

        IDA files carry no equilibrium data, so the psi_n -> rho mapping uses the
        geqdsk equilibrium fetched through the same connection. Profiles are
        forward-filled onto params.times (NaN before the first IDA slice), the
        same hold-last-value behavior the raw 1 kHz assembly always used.
        """
        ida_path = find_ida_path(params.shot_id)
        if ida_path is None:
            raise CalculationError(f"no IDA file for shot {params.shot_id}")
        ida = xr.load_dataset(ida_path)
        ida = ida.sortby("time")
        ida_times = ida["time"].values / 1e3  # [ms] -> [s]
        psi_n_grid = ida["psi_n"].values

        efit_dict = D3DPhysicsMethods._get_efit_dict(params)  # cached geqdsk nodes
        rho_map = _rho_of_psi_n(psi_n_grid, ida_times, efit_dict)  # (n_ida, n_psi)

        n_t, n_rho = len(ida_times), len(RHO_GRID)
        prof = {}
        for src, dst in [
            ("T_e", "te_rho"),
            ("T_e_err", "te_rho_error"),
            ("n_e", "ne_rho"),
            ("n_e_err", "ne_rho_error"),
        ]:
            vals = np.full((n_t, n_rho), np.nan, dtype=np.float32)
            for i in range(n_t):
                if np.isfinite(rho_map[i]).all():
                    vals[i] = np.interp(RHO_GRID, rho_map[i], ida[src].values[i], left=np.nan, right=np.nan)
            prof[dst] = vals
        for var in ["te", "ne"]:
            grad, grad_err = _gradient_and_error(prof[f"{var}_rho"], prof[f"{var}_rho_error"], RHO_GRID)
            prof[f"{var}_rho_grad"], prof[f"{var}_rho_grad_error"] = grad, grad_err

        # ffill onto params.times, NaN before the first IDA slice
        idx_prev = np.searchsorted(ida_times, params.times, side="right") - 1

        def onto_times(arr: np.ndarray) -> np.ndarray:
            res = arr[np.clip(idx_prev, 0, None)].astype(np.float32)
            res[idx_prev < 0] = np.nan
            return res

        return xr.Dataset(
            data_vars={
                "te_psi": (("idx", "psi_n"), onto_times(ida["T_e"].values)),
                "ne_psi": (("idx", "psi_n"), onto_times(ida["n_e"].values)),
                **{name: (("idx", "rho"), onto_times(vals)) for name, vals in prof.items()},
            },
            coords={
                **params.to_coords(),
                "psi_n": psi_n_grid.astype(np.float32),
                "rho": RHO_GRID.astype(np.float32),
            },
        )
