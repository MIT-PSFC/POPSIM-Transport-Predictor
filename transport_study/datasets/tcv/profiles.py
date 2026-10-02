"""DEFUSE Te/ne profiles on the store rho_tor_norm grid, and the LIUQE rho_pol -> rho_tor_norm map that puts them there.

DEFUSE fits its profiles on rho_pol = sqrt(psi_N), the store grid is rho_tor_norm = sqrt(Phi_N).
No file access here, so all of it is testable offline.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicHermiteSpline
from scipy.special import xlogy

from transport_study.datasets.profile_grids import held_on_times, held_slice_index
from transport_study.datasets.tcv import config

RHO_TOR_NORM_GRID = np.linspace(
    config["profile_grid"]["rho_min"],
    config["profile_grid"]["rho_max"],
    config["profile_grid"]["num_rho_points"],
)
RHO_TOR_NORM_DEFINITION = (
    "Normalized toroidal flux coordinate rho_tor_norm = sqrt(Phi_N): 0 at the magnetic axis, 1 at the LCFS. "
    "Phi_N is the integral of the LIUQE q over psi_N. Where q diverges at the LCFS of a diverted plasma, "
    "the integral past the last surface of finite q uses q = a - b ln(1 - psi_N) fit to the four surfaces inside it. "
    "The DEFUSE profile fits end at the LCFS, so there is nothing past it."
)
# Outermost finite-q surfaces the logarithmic q tail of a diverted plasma is fit to
Q_TAIL_FIT_SURFACES = 4


@dataclass(frozen=True)
class DefuseProfile:
    """One DEFUSE profile fit of one shot."""

    time: np.ndarray  # (n_slices,) sorted slice times [s]
    rho_pol: np.ndarray  # (n_points,) increasing fit points, rho_pol = sqrt(psi_N)
    values: np.ndarray  # (n_slices, n_points)


@dataclass(frozen=True)
class LiuqeEquilibria:
    """The LIUQE reconstructions of one shot, only what the rho_tor_norm map needs."""

    time: np.ndarray  # (n_eq,) reconstruction times [s]
    rho_pol: np.ndarray  # (n_surfaces,) rho_pol of the flux surfaces, 0 at the axis to 1 at the LCFS (L.pQ)
    inverse_q: np.ndarray  # (n_eq, n_surfaces) 1/q on the surfaces, 0 where q diverges (LY.iqQ)


def _q_tail_integral(psi_n_from, psi_n_to, q_offset: float, q_log_slope: float):
    """Integral of q = q_offset - q_log_slope ln(1 - psi_N) over psi_N, finite up to psi_N = 1.

    The antiderivative of -ln(1 - psi_N) is (1 - psi_N) ln(1 - psi_N) - (1 - psi_N),
    and xlogy keeps it 0 at psi_N = 1.
    """
    one_minus_from = 1.0 - psi_n_from
    one_minus_to = 1.0 - psi_n_to
    antiderivative_from = q_offset * psi_n_from + q_log_slope * (xlogy(one_minus_from, one_minus_from) - one_minus_from)
    antiderivative_to = q_offset * psi_n_to + q_log_slope * (xlogy(one_minus_to, one_minus_to) - one_minus_to)
    return antiderivative_to - antiderivative_from


def phi_n_from_liuqe(rho_pol_surfaces: np.ndarray, inverse_q: np.ndarray, psi_n: np.ndarray) -> np.ndarray | None:
    """Phi_N at the given psi_N of one LIUQE reconstruction, the integral of q over psi_N normalized at the LCFS.

    LIUQE's own enclosed toroidal flux (FtPQ) is quantized, staircased near the axis and jittery at the edge,
    so it is not used.
    Inside the last surface of finite q the integral is Simpson's rule over the surfaces,
    interpolated between them by a cubic Hermite spline whose slope is q itself,
    so dPhi_N/dpsi_N stays continuous and the profile gradients have no kinks at the surfaces.
    A limited plasma has finite q everywhere, so that is all of it.
    In a diverted plasma q diverges logarithmically at the LCFS (1/q = 0 there),
    so past the last surface of finite q it is integrated analytically
    as q = a - b ln(1 - psi_N), fit to the Q_TAIL_FIT_SURFACES surfaces inside it.
    The sign of q cancels.

    Args:
        rho_pol_surfaces: (n_surfaces,) rho_pol of the surfaces, 0 at the axis to 1 at the LCFS.
        inverse_q: (n_surfaces,) 1/q on the surfaces.
        psi_n: Points to evaluate at, any shape, clipped to [0, 1].

    Returns:
        Phi_N shaped like psi_n, exactly 1 at the LCFS,
        or None when the reconstruction is unusable:
        a non-finite 1/q, fewer than Q_TAIL_FIT_SURFACES surfaces of finite q from the axis,
        a q integral that is not increasing, or a tail fit with q not positive and increasing.
    """
    if not np.isfinite(inverse_q).all():
        return None
    psi_n_surfaces = rho_pol_surfaces**2
    mask_q_infinite = inverse_q == 0
    num_q_finite = int(np.argmax(mask_q_infinite)) if mask_q_infinite.any() else inverse_q.size
    if num_q_finite < Q_TAIL_FIT_SURFACES:
        return None
    psi_n_inside = psi_n_surfaces[:num_q_finite]
    q_inside = 1.0 / np.abs(inverse_q[:num_q_finite])
    q_integral_inside = cumulative_simpson(q_inside, x=psi_n_inside, initial=0.0)
    q_integral_steps = np.diff(q_integral_inside)
    if not (q_integral_steps > 0).all():
        return None

    psi_n_join = psi_n_inside[-1]
    q_offset, q_log_slope = 0.0, 0.0
    if num_q_finite < psi_n_surfaces.size:
        psi_n_fit = psi_n_inside[-Q_TAIL_FIT_SURFACES:]
        q_fit = q_inside[-Q_TAIL_FIT_SURFACES:]
        log_term_fit = -np.log(1.0 - psi_n_fit)
        design = np.stack([np.ones_like(psi_n_fit), log_term_fit], axis=1)
        (q_offset, q_log_slope), *_ = np.linalg.lstsq(design, q_fit, rcond=None)
        q_at_join = q_offset - q_log_slope * np.log(1.0 - psi_n_join)
        if q_log_slope <= 0 or q_at_join <= 0:
            return None
    q_integral_total = q_integral_inside[-1] + _q_tail_integral(psi_n_join, 1.0, q_offset, q_log_slope)

    psi_n_clipped = np.clip(psi_n, 0.0, 1.0)
    q_integral_spline = CubicHermiteSpline(psi_n_inside, q_integral_inside, q_inside)
    q_integral_interior = q_integral_spline(np.minimum(psi_n_clipped, psi_n_join))
    q_integral_tail = q_integral_inside[-1] + _q_tail_integral(psi_n_join, psi_n_clipped, q_offset, q_log_slope)
    q_integral_points = np.where(psi_n_clipped <= psi_n_join, q_integral_interior, q_integral_tail)
    phi_n = q_integral_points / q_integral_total
    phi_n[psi_n_clipped == 1.0] = 1.0
    return phi_n


def liuqe_usable(equilibria: LiuqeEquilibria) -> np.ndarray:
    """(n_eq,) True where phi_n_from_liuqe can map through the reconstruction."""
    psi_n_surfaces = equilibria.rho_pol**2
    mask_usable = [phi_n_from_liuqe(equilibria.rho_pol, inverse_q, psi_n_surfaces) is not None for inverse_q in equilibria.inverse_q]
    return np.array(mask_usable, dtype=bool)


def defuse_profile_on_grid(profile: DefuseProfile, equilibria: LiuqeEquilibria, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One DEFUSE profile and its d/drho_tor_norm gradient on RHO_TOR_NORM_GRID, held onto the timebase.

    Each slice maps to rho_tor_norm through the nearest usable reconstruction no farther than match_max_ms away,
    otherwise it is NaN.
    A slice with any NaN fit point inside the LCFS is a failed fit and NaN too.
    The gradient is taken on the DEFUSE points before interpolating,
    so it is one-sided and finite at the LCFS, where the fits end.
    Grid points past the LCFS are NaN.
    Each slice is held until the next one, for at most max_hold_defuse_steps median DEFUSE steps.

    Args:
        profile: The DEFUSE fit, at least two slices.
        equilibria: The shot's LIUQE reconstructions.
        times: (n_t,) timebase [s].

    Returns:
        (n_t, n_grid) profile and gradient, float32.

    Raises:
        ValueError: If the profile has fewer than two slices.
    """
    num_slices = profile.time.size
    if num_slices < 2:
        raise ValueError(f"DEFUSE profile has {num_slices} slice(s), at least two are needed to set the hold")

    # Nearest usable reconstruction of each slice
    mask_eq_usable = liuqe_usable(equilibria)
    eq_distance = np.abs(equilibria.time[np.newaxis, :] - profile.time[:, np.newaxis])
    eq_distance[:, ~mask_eq_usable] = np.inf
    eq_nearest = np.argmin(eq_distance, axis=1)
    slice_rows = np.arange(num_slices)
    eq_nearest_distance = eq_distance[slice_rows, eq_nearest]
    match_max = config["equilibrium"]["match_max_ms"] / 1e3
    mask_slice_mapped = eq_nearest_distance <= match_max

    psi_n_points = profile.rho_pol**2
    mask_points_inside = profile.rho_pol <= 1.0
    values_grid = np.full((num_slices, RHO_TOR_NORM_GRID.size), np.nan)
    gradient_grid = np.full_like(values_grid, np.nan)
    mask_slice_complete = np.isfinite(profile.values[:, mask_points_inside]).all(axis=1)
    for i_slice in np.flatnonzero(mask_slice_mapped & mask_slice_complete):
        inverse_q = equilibria.inverse_q[eq_nearest[i_slice]]
        phi_n_points = phi_n_from_liuqe(equilibria.rho_pol, inverse_q, psi_n_points[mask_points_inside])
        # Usability does not depend on the points, so a usable reconstruction always maps
        assert phi_n_points is not None
        rho_tor_norm_points = np.sqrt(phi_n_points)
        values_inside = profile.values[i_slice, mask_points_inside]
        gradient_inside = np.gradient(values_inside, rho_tor_norm_points)
        values_grid[i_slice] = np.interp(RHO_TOR_NORM_GRID, rho_tor_norm_points, values_inside, right=np.nan)
        gradient_grid[i_slice] = np.interp(RHO_TOR_NORM_GRID, rho_tor_norm_points, gradient_inside, right=np.nan)

    slice_index, mask_held = held_slice_index(profile.time, times, config["profile_grid"]["max_hold_defuse_steps"])
    values_on_times = held_on_times(values_grid, slice_index, mask_held)
    gradient_on_times = held_on_times(gradient_grid, slice_index, mask_held)
    return values_on_times, gradient_on_times
