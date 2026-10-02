"""DEFUSE Te/ne profiles on the store rho_tor_norm grid, and the LIUQE rho_pol -> rho_tor_norm map that puts them there.

DEFUSE fits its profiles on rho_pol = sqrt(psi_N), the store grid is rho_tor_norm = sqrt(Phi_N).
No file access here, so all of it is testable offline.
"""

from dataclasses import dataclass

import numpy as np
from transport_validation_datasets.machine.generic import (
    hold_onto_grid,
    mappable_q_profiles,
    phi_n_map,
    values_on_grid,
)

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


def liuqe_q_profiles(equilibria: LiuqeEquilibria) -> tuple[np.ndarray, np.ndarray]:
    """The psi_N surfaces and the q profiles of the reconstructions, as phi_n_map takes them.

    LIUQE stores 1/q, which is 0 where q diverges at the LCFS of a diverted plasma,
    so q is infinite there, and a NaN 1/q stays NaN.
    LIUQE's own enclosed toroidal flux (FtPQ) is quantized, staircased near the axis and jittery at the edge,
    so Phi_N is integrated from q instead.

    Args:
        equilibria: The shot's LIUQE reconstructions.

    Returns:
        (n_surfaces,) psi_N = rho_pol^2 of the surfaces, and (n_eq, n_surfaces) |q| on them.
    """
    psi_n_surfaces = equilibria.rho_pol**2
    inverse_q_magnitude = np.abs(equilibria.inverse_q)
    with np.errstate(divide="ignore"):
        q_surfaces = 1.0 / inverse_q_magnitude
    return psi_n_surfaces, q_surfaces


def liuqe_usable(equilibria: LiuqeEquilibria) -> np.ndarray:
    """(n_eq,) True where phi_n_map can map through the reconstruction."""
    psi_n_surfaces, q_surfaces = liuqe_q_profiles(equilibria)
    return mappable_q_profiles(psi_n_surfaces, q_surfaces)


def defuse_profile_on_grid(
    profile: DefuseProfile, equilibria: LiuqeEquilibria, mask_eq_usable: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One DEFUSE profile and its d/drho_tor_norm gradient on RHO_TOR_NORM_GRID, held onto the timebase (hold_onto_grid).

    Each slice maps to rho_tor_norm through the nearest usable reconstruction no farther than match_max_ms away.
    The gradient is taken on the DEFUSE points before interpolating,
    so it is one-sided and finite at the LCFS, where the fits end.
    Grid points past the LCFS are NaN.
    Each slice is held until the next one, for at most max_hold_defuse_steps median steps of the usable slices.
    A slice that does not map, or has a NaN fit point inside the LCFS (a failed fit), is dropped before the hold,
    so the slice before it holds over it, as transport-validation-datasets does with its fits.

    Args:
        profile: The DEFUSE fit, at least two slices.
        equilibria: The shot's LIUQE reconstructions.
        mask_eq_usable: (n_eq,) liuqe_usable of equilibria, computed once per shot.
        times: (n_t,) timebase [s].

    Returns:
        (n_t, n_grid) profile and gradient, float32,
        and (n_t,) True where a usable slice lands.

    Raises:
        ValueError: If the profile has fewer than two slices.
    """
    num_slices = profile.time.size
    if num_slices < 2:
        raise ValueError(f"DEFUSE profile has {num_slices} slice(s), at least two are needed to set the hold")

    # Nearest usable reconstruction of each slice
    eq_distance = np.abs(equilibria.time[np.newaxis, :] - profile.time[:, np.newaxis])
    eq_distance[:, ~mask_eq_usable] = np.inf
    eq_nearest = np.argmin(eq_distance, axis=1)
    slice_rows = np.arange(num_slices)
    eq_nearest_distance = eq_distance[slice_rows, eq_nearest]
    match_max = config["equilibrium"]["match_max_ms"] / 1e3
    mask_slice_mapped = eq_nearest_distance <= match_max

    psi_n_surfaces, q_surfaces = liuqe_q_profiles(equilibria)
    psi_n_points = profile.rho_pol**2
    mask_points_inside = profile.rho_pol <= 1.0
    values_grid = np.full((num_slices, RHO_TOR_NORM_GRID.size), np.nan)
    gradient_grid = np.full_like(values_grid, np.nan)
    mask_slice_complete = np.isfinite(profile.values[:, mask_points_inside]).all(axis=1)
    for i_slice in np.flatnonzero(mask_slice_mapped & mask_slice_complete):
        # The DEFUSE points stop at the LCFS, so the SOL extension never enters
        phi_n_mapping = phi_n_map(psi_n_surfaces, q_surfaces[eq_nearest[i_slice]], "secant")
        phi_n_points = phi_n_mapping.phi_n(psi_n_points[mask_points_inside])
        rho_tor_norm_points = np.sqrt(phi_n_points)
        values_inside = profile.values[i_slice, mask_points_inside]
        gradient_inside = np.gradient(values_inside, rho_tor_norm_points)
        values_grid[i_slice] = np.interp(RHO_TOR_NORM_GRID, rho_tor_norm_points, values_inside, right=np.nan)
        gradient_grid[i_slice] = np.interp(RHO_TOR_NORM_GRID, rho_tor_norm_points, gradient_inside, right=np.nan)

    mask_slice_usable = mask_slice_mapped & mask_slice_complete
    max_hold_steps = config["profile_grid"]["max_hold_defuse_steps"]
    slice_index, fresh = hold_onto_grid(times, profile.time[mask_slice_usable], True, max_hold_periods=max_hold_steps)
    values_on_times = values_on_grid(values_grid[mask_slice_usable], slice_index)
    gradient_on_times = values_on_grid(gradient_grid[mask_slice_usable], slice_index)
    return values_on_times.astype(np.float32), gradient_on_times.astype(np.float32), fresh
