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
from disruption_py.inout.mds import mdsExceptions
from disruption_py.machine.d3d.util import D3DUtilMethods
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
from transport_study.datasets.profile_grids import held_signal_on_grid
from transport_study.datasets.rho_tor_norm import geqdsk_psi_n_grid, mappable_q_profiles

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


def _injected_power(params: PhysicsMethodParams, node: str, tree_name: str) -> np.ndarray:
    """An injected heating power record held onto the timebase (held_signal_on_grid), in the record's units.

    0 outside the record and when the shot has none (that heating system did not run).
    """
    try:
        power, power_time_ms = params.mds_conn.get_data_with_dims(node, tree_name=tree_name)
    except mdsExceptions.MdsException:
        params.logger.debug("no {node} record, taking 0", node=node)
        return np.zeros(len(params.times))
    # echpwrc can end with a stray t = 0 sample
    if power_time_ms.size > 1 and power_time_ms[-1] == 0:
        power_time_ms = power_time_ms[:-1]
        power = power[:-1]
    if power_time_ms.size <= 2:
        return np.zeros(len(params.times))
    power_time = power_time_ms / 1e3
    power_on_timebase = held_signal_on_grid(power_time, power, params.times)
    mask_outside_record = (params.times < power_time[0]) | (params.times > power_time[-1])
    power_on_timebase[mask_outside_record] = 0.0
    return power_on_timebase


class D3DDatasetMethods:
    """Signals the transport study needs that disruption-py 0.14 has no built-in for, or no usable one.

    Every signal is held onto the timebase from its last sample (held_signal_on_grid), never interpolated,
    so no grid time draws on a later sample.
    The disruption-py built-ins interpolate, so the EFIT scalars and the plasma current are read here too.
    """

    # Measured: bt (vacuum toroidal field at R = 1.6955 m) and dssneped (PCS pedestal density estimate).
    # PCS targets: bmtpwrtar (beta_N), idtrp (R0), idtrxbot / idtzxbot / idtrxtop / idtzxtop (X points).
    # Unverified, see D3D_TRAJOPT_STORE_SIGNALS: bttbt, dstdenp, ieeseg07.
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
    # Store geometry
    BOUNDARY_NODES: ClassVar[list[str]] = ["aminor", "rsurf", "kappa", "tritop", "tribot"]
    # Store EFIT scalars, under the column names of the built-in get_efit_parameters
    EFIT_SCALAR_NODES: ClassVar[dict[str, str]] = {"wmhd": "wmhd", "beta_n": "betan"}
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
            signals[pointname] = held_signal_on_grid(times_s, values, params.times)
        return signals

    @staticmethod
    @physics_method(columns=["ip", "ip_prog"], tokamak=Tokamak.D3D)
    def get_plasma_current(params: PhysicsMethodParams):
        """Measured (PTDATA ip) and programmed (PTDATA iptipp, times the current polarity) plasma current [A].

        The nodes of the built-in get_ip_parameters, which interpolates them.
        """
        ip, ip_time_ms = params.mds_conn.get_data_with_dims(f"ptdata('ip', {params.shot_id})")
        ip_time = ip_time_ms / 1e3
        ip_on_timebase = held_signal_on_grid(ip_time, ip, params.times)
        try:
            ip_prog, ip_prog_time_ms = params.mds_conn.get_data_with_dims(f"ptdata('iptipp', {params.shot_id})")
            polarity = D3DUtilMethods.get_polarity(params)
            ip_prog_signed = ip_prog * polarity
            ip_prog_time = ip_prog_time_ms / 1e3
            ip_prog_on_timebase = held_signal_on_grid(ip_prog_time, ip_prog_signed, params.times)
        except mdsExceptions.MdsException:
            params.logger.warning("ptdata iptipp missing")
            ip_prog_on_timebase = np.full(len(params.times), np.nan)
        return {"ip": ip_on_timebase, "ip_prog": ip_prog_on_timebase}

    @staticmethod
    @physics_method(columns=["wmhd", "beta_n"], tokamak=Tokamak.D3D)
    def get_efit_scalars(params: PhysicsMethodParams):
        """Stored energy and normalized beta of the DISPY EFIT, NaN on slices with chi-squared above chisq_max.

        The nodes of the built-in get_efit_parameters, which interpolates them.
        """
        efit_time, signals = _efit_signals(params, list(D3DDatasetMethods.EFIT_SCALAR_NODES.values()))
        return {
            column: held_signal_on_grid(efit_time, signals[node], params.times)
            for column, node in D3DDatasetMethods.EFIT_SCALAR_NODES.items()
        }

    @staticmethod
    @physics_method(columns=["n_e_line_average"], tokamak=Tokamak.D3D)
    def get_line_average_density(params: PhysicsMethodParams):
        """Line-averaged electron density [m^-3], \\density of the DISPY EFIT tree, else the PCS estimate dssdenest.

        Replaces the built-in get_density_parameters, which falls back to \\d3d::denv2,
        and that reads about 3x higher. dssdenest [1e19 m^-3] matches \\density within 1 percent where both exist.
        """
        try:
            density_cm3, density_time_ms = params.mds_conn.get_data_with_dims(r"\density", tree_name="_efit_tree")
            density = density_cm3 * 1e6
        except mdsExceptions.MdsException:
            density = np.array([np.nan])
        if not np.isfinite(density).any():
            params.logger.warning("EFIT tree has no density, using PCS dssdenest")
            density_1e19, density_time_ms = params.mds_conn.get_data_with_dims(f"ptdata('dssdenest', {params.shot_id})")
            density = density_1e19 * 1e19
        density_time = density_time_ms / 1e3
        n_e_line_average = held_signal_on_grid(density_time, density, params.times)
        return {"n_e_line_average": n_e_line_average}

    @staticmethod
    @physics_method(columns=["p_ohm"], tokamak=Tokamak.D3D)
    def get_ohmic_power(params: PhysicsMethodParams):
        """Ohmic power [W] from the DISPY EFIT: poh = Ip V_surf - dW_pol/dt, with V_surf = -2 pi dpsi_bdy/dt.

        Replaces the built-in get_ohmic_parameters,
        whose 20 kHz vloopb with a 0.55 ms median filter is noise at 1 kHz.
        """
        efit_time, signals = _efit_signals(params, ["poh"])
        p_ohm = held_signal_on_grid(efit_time, signals["poh"], params.times)
        return {"p_ohm": p_ohm}

    @staticmethod
    @physics_method(columns=["p_rad"], tokamak=Tokamak.D3D)
    def get_radiated_power(params: PhysicsMethodParams):
        """Total radiated power [W] including the divertor, \\bolom::prad_tot of the standard bolometer analysis.

        Sampled every 4 ms and smoothed non-causally over 50 ms, the one stored signal that is not causal.
        Its units label reads MW, but the values are W (they match the built-in pwrmix to a few percent).
        Replaces the built-in pwrmix,
        a causal 10 ms sum of the 48 raw channels that resolves ELMs and goes negative.
        """
        p_rad, p_rad_time_ms = params.mds_conn.get_data_with_dims(r"\top.prad_01.prad:prad_tot", tree_name="bolom")
        p_rad_time = p_rad_time_ms / 1e3
        p_rad_on_timebase = held_signal_on_grid(p_rad_time, p_rad, params.times)
        return {"p_rad": p_rad_on_timebase}

    @staticmethod
    @physics_method(columns=["p_nbi", "p_ech"], tokamak=Tokamak.D3D)
    def get_heating_powers(params: PhysicsMethodParams):
        """Injected neutral beam and electron cyclotron powers [W], the nodes of the built-in get_power_parameters.

        The built-in also rebuilds pwrmix from the 48 raw bolometer channels and P_oh from vloopb,
        so a bolometer failure there would NaN p_nbi and skip the shot.
        """
        p_nbi_kW = _injected_power(params, r"\top.nb:pinj", "d3d")
        p_nbi = p_nbi_kW * 1e3
        p_ech = _injected_power(params, r"\top.ech.total:echpwrc", "rf")
        return {"p_nbi": p_nbi, "p_ech": p_ech}

    @staticmethod
    @physics_method(columns=BOUNDARY_NODES, tokamak=Tokamak.D3D)
    def get_boundary_parameters(params: PhysicsMethodParams):
        """Plasma boundary minor radius, geometric axis R, elongation and triangularities from the DISPY EFIT."""
        efit_time, signals = _efit_signals(params, D3DDatasetMethods.BOUNDARY_NODES)
        return {node: held_signal_on_grid(efit_time, values, params.times) for node, values in signals.items()}

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
        return {node: held_signal_on_grid(efit_time, values, params.times) for node, values in signals.items()}

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
        psi_n_grid_efit = geqdsk_psi_n_grid(qpsi.shape[1])
        mask_q_mappable = mappable_q_profiles(psi_n_grid_efit, qpsi)
        mask_efit_valid = (chisq <= config["efit"]["chisq_max"]) & mask_q_mappable

        ida_profiles = ida_profiles_on_grids(ida, efit_time, qpsi, mask_efit_valid, params.times)
        idx_coords = params.to_coords()
        return ida_profiles.assign_coords(idx_coords)
