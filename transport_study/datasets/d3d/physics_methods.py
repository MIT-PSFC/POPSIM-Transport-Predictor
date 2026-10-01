"""Custom disruption-py physics methods, EFIT tree and timebase for the DIII-D dataset.

Written against the PyPI disruption-py 0.14.0 API (params.mds_conn) and passed in through
RetrievalSettings, so nothing in disruption-py is patched.
"""

from typing import ClassVar

import numpy as np
import xarray as xr
from disruption_py.core.physics_method.decorator import physics_method
from disruption_py.core.physics_method.errors import CalculationError
from disruption_py.core.physics_method.params import PhysicsMethodParams
from disruption_py.core.utils.math import interp1
from disruption_py.inout.mds import mdsExceptions
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import TimeSetting, TimeSettingParams
from disruption_py.settings.nickname_setting import (
    NicknameSetting,
    NicknameSettingParams,
)
from loguru import logger

from transport_study.datasets import make_uniform_1khz_timebase
from transport_study.datasets.d3d import config
from transport_study.datasets.d3d.profiles import (
    IDA_PSI_COLUMNS,
    IDA_RHO_COLUMNS,
    find_ida_path,
    ida_profiles_on_grids,
)

# Programmed waveforms run past early plasma termination, keep them for predict-first [s]
MIN_TIMEBASE_S = 8.0


class DispyEfitNicknameSetting(NicknameSetting):
    """The shot's _efit_tree: its latest run under the configured runtag (DISPY, the 1 kHz disruption-efit).

    disruption-py's own DIII-D nickname falls back to the 50 Hz efit01 when the runtag has no run,
    and forces runtag DIS under pytest. This raises instead, so no shot is built on another EFIT.
    """

    def _get_tree_name(self, params: NicknameSettingParams) -> str:
        runtag = config["efit"]["runtag"]
        efit_runs = params.database.query(
            f"select tree from code_rundb.dbo.plasmas where shot = {params.shot_id} and runtag = '{runtag}' and deleted = 0 order by idx",
            use_pandas=False,
        )
        if not efit_runs:
            raise ValueError(f"Shot {params.shot_id} has no EFIT run under runtag {runtag}")
        efit_tree = efit_runs[-1][0]
        logger.info(f"Shot {params.shot_id}: EFIT tree {efit_tree} (runtag {runtag})")
        return efit_tree


class Uniform1kHzTimeSetting(TimeSetting):
    """Uniform 1 kHz timebase [s] from 0 to the end of the EFIT, and at least MIN_TIMEBASE_S long.

    Raises when the EFIT is slower than 1 kHz, the mark of a reconstruction that is not DISPY.
    """

    def _get_times(self, params: TimeSettingParams) -> np.ndarray:
        efit_time_ms = params.mds_conn.get_data(r"\efit_a_eqdsk:atime", tree_name="_efit_tree")
        efit_steps_ms = np.diff(efit_time_ms)
        efit_step_median_ms = np.median(efit_steps_ms)
        if efit_step_median_ms > config["efit"]["max_step_ms"]:
            raise ValueError(f"Shot {params.shot_id}: median EFIT step {efit_step_median_ms:.1f} ms, not a 1 kHz reconstruction")
        efit_end_s = np.max(efit_time_ms) / 1e3
        timebase_end_s = max(efit_end_s, MIN_TIMEBASE_S)
        return make_uniform_1khz_timebase(timebase_end_s)


def _efit_signals(params: PhysicsMethodParams, nodes: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """A-eqdsk nodes on the EFIT timebase [s], NaN on slices whose chi-squared exceeds chisq_max.

    Reads DIII-D's own a-eqdsk node names only. The a-file-format names in the same tree are NID
    aliases, and about half of them point at a different quantity.
    """
    efit_time_ms = params.mds_conn.get_data(r"\efit_a_eqdsk:atime", tree_name="_efit_tree")
    efit_time = efit_time_ms / 1e3
    chisq = params.mds_conn.get_data(r"\efit_a_eqdsk:chisq", tree_name="_efit_tree")
    mask_invalid = chisq > config["efit"]["chisq_max"]
    signals = {}
    for node in nodes:
        values = params.mds_conn.get_data(rf"\efit_a_eqdsk:{node}", tree_name="_efit_tree")
        values[mask_invalid] = np.nan
        signals[node] = values
    return efit_time, signals


class D3DDatasetMethods:
    """Signals the transport study needs that disruption-py 0.14 has no built-in for."""

    # Measured: bt (vacuum toroidal field at R = 1.6955 m) and dssneped (PCS pedestal density estimate).
    # Programmed (PCS targets): bttbt (B0), bmtpwrtar (beta_N), dstdenp (density), idtrp (R0),
    # ieeseg07 (inner gap), idtrxbot / idtzxbot / idtrxtop / idtzxtop (X points).
    # iptipp (Ip) comes from the built-in get_ip_parameters as ip_prog.
    PTDATA_POINTNAMES: ClassVar[list[str]] = [
        "bt",
        "dssneped",
        "bttbt",
        "bmtpwrtar",
        "dstdenp",
        "idtrp",
        "ieeseg07",
        "idtrxbot",
        "idtzxbot",
        "idtrxtop",
        "idtzxtop",
    ]
    # Store geometry the built-in get_efit_parameters lacks (it has kappa)
    BOUNDARY_NODES: ClassVar[list[str]] = ["aminor", "rsurf", "tritop", "tribot"]
    # Trajopt geometry. rxpt1 / zxpt1 is the lower X point, rxpt2 / zxpt2 the upper.
    XPOINT_GAP_NODES: ClassVar[list[str]] = ["gapin", "rxpt1", "zxpt1", "rxpt2", "zxpt2"]

    @staticmethod
    @physics_method(columns=PTDATA_POINTNAMES, tokamak=Tokamak.D3D)
    def get_ptdata_parameters(params: PhysicsMethodParams):
        """PTDATA pointnames on the timebase, each NaN when its pointname is missing from the shot."""
        signals = {}
        for pointname in D3DDatasetMethods.PTDATA_POINTNAMES:
            try:
                values, times_ms = params.mds_conn.get_data_with_dims(f"ptdata('{pointname}', {params.shot_id})")
            except mdsExceptions.MdsException:
                params.logger.warning("ptdata {pointname} missing", pointname=pointname)
                signals[pointname] = np.full(len(params.times), np.nan)
                continue
            times_s = times_ms / 1e3
            signals[pointname] = interp1(times_s, values, params.times)
        return signals

    @staticmethod
    @physics_method(columns=BOUNDARY_NODES, tokamak=Tokamak.D3D)
    def get_boundary_parameters(params: PhysicsMethodParams):
        """Plasma boundary minor radius, geometric axis R and triangularities from the DISPY EFIT."""
        efit_time, signals = _efit_signals(params, D3DDatasetMethods.BOUNDARY_NODES)
        return {node: interp1(efit_time, values, params.times) for node, values in signals.items()}

    @staticmethod
    @physics_method(columns=XPOINT_GAP_NODES, tokamak=Tokamak.D3D)
    def get_xpoint_gap_parameters(params: PhysicsMethodParams):
        """Inner gap and both X points from the DISPY EFIT, each X point NaN where EFIT finds none."""
        efit_time, signals = _efit_signals(params, D3DDatasetMethods.XPOINT_GAP_NODES)
        # EFIT writes R = -9.99 when it finds no X point
        for r_node, z_node in [("rxpt1", "zxpt1"), ("rxpt2", "zxpt2")]:
            mask_no_xpoint = ~(signals[r_node] > 0)
            signals[r_node][mask_no_xpoint] = np.nan
            signals[z_node][mask_no_xpoint] = np.nan
        return {node: interp1(efit_time, values, params.times) for node, values in signals.items()}

    @staticmethod
    @physics_method(columns=[*IDA_RHO_COLUMNS, *IDA_PSI_COLUMNS], tokamak=Tokamak.D3D)
    def get_ida_profiles(params: PhysicsMethodParams):
        """IDA Te/ne on the rho_tor_norm grid (mapped through the DISPY EFIT q profile) and on the psi_norm grid."""
        ida_path = find_ida_path(params.shot_id)
        if ida_path is None:
            raise CalculationError(f"no IDA file for shot {params.shot_id}")
        ida = xr.load_dataset(ida_path)

        efit_time_ms = params.mds_conn.get_data(r"\efit_a_eqdsk:atime", tree_name="_efit_tree")
        efit_time = efit_time_ms / 1e3
        qpsi = params.mds_conn.get_data(r"\top.results.geqdsk:qpsi", tree_name="_efit_tree")
        if qpsi.shape[0] != efit_time.size:
            raise CalculationError(f"qpsi shape {qpsi.shape} does not match {efit_time.size} EFIT slices")
        chisq = params.mds_conn.get_data(r"\efit_a_eqdsk:chisq", tree_name="_efit_tree")
        mask_q_finite = np.isfinite(qpsi).all(axis=1)
        mask_efit_valid = (chisq <= config["efit"]["chisq_max"]) & mask_q_finite

        ida_profiles = ida_profiles_on_grids(ida, efit_time, qpsi, mask_efit_valid, params.times)
        idx_coords = params.to_coords()
        return ida_profiles.assign_coords(idx_coords)
