"""IDA Te/ne profiles on the store grids, and the psi_n -> rho_tor_norm map that puts them there.

No disruption-py or MDSplus here, so all of it is testable offline.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger
from transport_validation_datasets.machine.generic import (
    geqdsk_psi_n_grid,
    hold_onto_grid,
    phi_n_map,
    values_on_grid,
)

from transport_study import PACKAGE_ROOT, RADIAL_DIM
from transport_study.datasets import read_shotlist
from transport_study.datasets.d3d import config

D3D_DIR = Path(PACKAGE_ROOT) / "datasets" / "d3d"

RHO_TOR_NORM_GRID = np.linspace(
    config["profile_grid"]["rho_min"],
    config["profile_grid"]["rho_max"],
    config["profile_grid"]["num_rho_points"],
)
PSI_NORM_DIM = "psi_norm"
RHO_TOR_NORM_DEFINITION = (
    "Normalized toroidal flux coordinate rho_tor_norm = sqrt(Phi_N): 0 at the magnetic axis, 1 at the LCFS. "
    "Outside the LCFS Phi_N continues linearly in psi_N, see the sol_extension attribute."
)
PSI_NORM_GRID = np.linspace(0.0, config["profile_grid"]["psi_norm_max"], config["profile_grid"]["num_psi_norm_points"])

# IDA file variable (with an _err companion) -> prefix of its raw columns
IDA_PROFILE_PREFIXES = {"T_e": "te", "n_e": "ne"}
IDA_RHO_SUFFIXES = ("_rho", "_rho_error", "_rho_grad", "_rho_grad_error")
IDA_RHO_COLUMNS = [f"{prefix}{suffix}" for prefix in IDA_PROFILE_PREFIXES.values() for suffix in IDA_RHO_SUFFIXES]
IDA_PSI_COLUMNS = [f"{prefix}_psi" for prefix in IDA_PROFILE_PREFIXES.values()]


@dataclass(frozen=True)
class IdaDatabase:
    """An IDA profile database: a file pattern (may hold * wildcards) and, when set, the only shots it serves."""

    pattern: str
    shots: frozenset[int] | None = None

    def serves(self, shot: int) -> bool:
        return self.shots is None or shot in self.shots

    def path(self, shot: int) -> Path | None:
        """The shot's file in this database, None when it has none or does not serve the shot."""
        if not self.serves(shot):
            return None
        candidate_name = self.pattern.format(shot=shot)
        candidate = Path(candidate_name)
        matches = sorted(candidate.parent.glob(candidate.name))
        if len(matches) > 1:
            logger.warning(f"Shot {shot}: {len(matches)} IDA files match {candidate}, using {matches[0]}")
        return matches[0] if matches else None

    def available_shots(self) -> set[int]:
        """Every shot this database has a file for and serves."""
        pattern_path = Path(self.pattern)
        name_escaped = re.escape(pattern_path.name)
        name_regex = name_escaped.replace(re.escape("{shot}"), r"(\d+)").replace(re.escape("*"), ".*")
        name_re = re.compile(name_regex)
        name_glob = pattern_path.name.format(shot="*")
        shots = set()
        for path in pattern_path.parent.glob(name_glob):
            match = name_re.fullmatch(path.name)
            if match:
                shots.add(int(match.group(1)))
        if self.shots is not None:
            shots &= self.shots
        return shots


def _ida_databases() -> list[IdaDatabase]:
    """The configured IDA databases in priority order, shotlist names resolved in this directory."""
    databases = []
    for entry in config["data_sources"]["ida_databases"]:
        shotlist_name = entry.get("shotlist")
        shots = None if shotlist_name is None else frozenset(read_shotlist(D3D_DIR / shotlist_name))
        databases.append(IdaDatabase(pattern=entry["pattern"], shots=shots))
    return databases


def find_ida_path(shot: int, databases: list[IdaDatabase] | None = None) -> Path | None:
    """The shot's IDA file from the first database that has one, in priority order."""
    if databases is None:
        databases = _ida_databases()
    for database in databases:
        path = database.path(shot)
        if path is not None:
            return path
    return None


def find_ida_shots(databases: list[IdaDatabase] | None = None) -> list[int]:
    """Every shot some database has a file for and serves, sorted."""
    if databases is None:
        databases = _ida_databases()
    shots: set[int] = set()
    for database in databases:
        shots |= database.available_shots()
    return sorted(shots)


def gradient_and_error(values: np.ndarray, errors: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradient along the last axis, with the central-difference error assuming independent points.

    IDA gives no point covariance (unlike the GP fits of the published stores), so this overestimates the
    error of a smooth fit. The endpoints copy their neighbor's error.
    """
    gradient = np.gradient(values, x, axis=-1)
    errors_squared = errors**2
    x_spacing = x[2:] - x[:-2]
    gradient_error = np.full_like(values, np.nan)
    gradient_error[..., 1:-1] = np.sqrt(errors_squared[..., 2:] + errors_squared[..., :-2]) / x_spacing
    gradient_error[..., 0] = gradient_error[..., 1]
    gradient_error[..., -1] = gradient_error[..., -2]
    return gradient, gradient_error


def ida_profiles_on_grids(
    ida: xr.Dataset,
    efit_time: np.ndarray,
    qpsi: np.ndarray,
    mask_efit_valid: np.ndarray,
    times: np.ndarray,
) -> xr.Dataset:
    """IDA Te/ne on RHO_TOR_NORM_GRID (with gradients) and PSI_NORM_GRID, held onto the timebase (hold_onto_grid).

    Each IDA slice maps to rho_tor_norm through the q profile of the nearest valid EFIT slice no farther than
    match_max_ms away. The interpolation clamps to the innermost IDA value at the axis and is NaN past the IDA
    psi_n domain. Each slice is held onto the timebase until the next one,
    for at most max_hold_ida_steps median steps of the usable slices,
    and the timebase is NaN before the first slice.
    A slice that does not map, or lacks a Te or ne fit, is dropped before the rho_tor_norm hold,
    so the slice before it holds over it, as transport-validation-datasets does with its fits.
    fresh_profile marks the grid times where a usable slice lands.

    Args:
        ida: IDA file contents, T_e [eV] and n_e [m^-3] with 1-sigma _err companions on (time [ms], psi_n).
        efit_time: (n_eq,) EFIT slice times [s].
        qpsi: (n_eq, n_psi) safety factor on the GEQDSK psi_N grid.
        mask_efit_valid: (n_eq,) True where the EFIT slice is usable,
            chisq within its maximum and the q profile mappable (mappable_q_profiles).
        times: (n_t,) timebase [s].

    Returns:
        IDA_RHO_COLUMNS on ("idx", RADIAL_DIM), IDA_PSI_COLUMNS on ("idx", PSI_NORM_DIM),
        and fresh_profile on ("idx",), all float32.

    Raises:
        ValueError: If the IDA file has fewer than two slices.
    """
    ida_sorted = ida.sortby("time")
    ida_time = ida_sorted["time"].values / 1e3
    psi_n = ida_sorted["psi_n"].values
    num_slices = ida_time.size
    if num_slices < 2:
        raise ValueError(f"IDA has {num_slices} slice(s), at least two are needed to set the hold")

    # Nearest valid EFIT slice of each IDA slice
    efit_offset = efit_time[np.newaxis, :] - ida_time[:, np.newaxis]
    efit_distance = np.abs(efit_offset)
    efit_distance[:, ~mask_efit_valid] = np.inf
    efit_nearest = np.argmin(efit_distance, axis=1)
    slice_rows = np.arange(num_slices)
    efit_nearest_distance = efit_distance[slice_rows, efit_nearest]
    match_max = config["efit"]["match_max_ms"] / 1e3
    mask_slice_mapped = efit_nearest_distance <= match_max

    sol_extension = config["profile_grid"]["sol_extension"]
    psi_n_grid_efit = geqdsk_psi_n_grid(qpsi.shape[1])
    rho_tor_norm_slices = np.full((num_slices, psi_n.size), np.nan)
    for i_mapped in np.flatnonzero(mask_slice_mapped):
        qpsi_slice = qpsi[efit_nearest[i_mapped]]
        phi_n_mapping = phi_n_map(psi_n_grid_efit, qpsi_slice, sol_extension)
        phi_n_slice = phi_n_mapping.phi_n(psi_n)
        rho_tor_norm_slices[i_mapped] = np.sqrt(phi_n_slice)

    # Slice profiles, interpolated per slice. np.interp's default left clamps at the axis.
    slice_profiles = {}
    for ida_name, prefix in IDA_PROFILE_PREFIXES.items():
        ida_values = ida_sorted[ida_name].values
        ida_errors = ida_sorted[f"{ida_name}_err"].values
        values_rho = np.full((num_slices, RHO_TOR_NORM_GRID.size), np.nan)
        errors_rho = np.full_like(values_rho, np.nan)
        values_psi = np.full((num_slices, PSI_NORM_GRID.size), np.nan)
        for i in range(num_slices):
            values_psi[i] = np.interp(PSI_NORM_GRID, psi_n, ida_values[i], right=np.nan)
            if not mask_slice_mapped[i]:
                continue
            values_rho[i] = np.interp(RHO_TOR_NORM_GRID, rho_tor_norm_slices[i], ida_values[i], right=np.nan)
            errors_rho[i] = np.interp(RHO_TOR_NORM_GRID, rho_tor_norm_slices[i], ida_errors[i], right=np.nan)
        gradient_rho, gradient_error_rho = gradient_and_error(values_rho, errors_rho, RHO_TOR_NORM_GRID)
        slice_profiles[f"{prefix}_rho"] = values_rho
        slice_profiles[f"{prefix}_rho_error"] = errors_rho
        slice_profiles[f"{prefix}_rho_grad"] = gradient_rho
        slice_profiles[f"{prefix}_rho_grad_error"] = gradient_error_rho
        slice_profiles[f"{prefix}_psi"] = values_psi

    # The psi_norm profiles need no map, so every slice holds them
    max_hold_steps = config["profile_grid"]["max_hold_ida_steps"]
    mask_te_fit = np.isfinite(slice_profiles["te_rho"]).any(axis=1)
    mask_ne_fit = np.isfinite(slice_profiles["ne_rho"]).any(axis=1)
    mask_slice_usable = mask_slice_mapped & mask_te_fit & mask_ne_fit
    usable_index, fresh = hold_onto_grid(times, ida_time[mask_slice_usable], True, max_hold_periods=max_hold_steps)
    slice_index, _ = hold_onto_grid(times, ida_time, True, max_hold_periods=max_hold_steps)

    data_vars = {"fresh_profile": (("idx",), fresh.astype(np.float32))}
    for name, profiles in slice_profiles.items():
        if name in IDA_PSI_COLUMNS:
            profiles_on_times = values_on_grid(profiles, slice_index)
            data_vars[name] = (("idx", PSI_NORM_DIM), profiles_on_times.astype(np.float32))
        else:
            profiles_on_times = values_on_grid(profiles[mask_slice_usable], usable_index)
            data_vars[name] = (("idx", RADIAL_DIM), profiles_on_times.astype(np.float32))
    coords = {
        RADIAL_DIM: RHO_TOR_NORM_GRID.astype(np.float32),
        PSI_NORM_DIM: PSI_NORM_GRID.astype(np.float32),
    }
    return xr.Dataset(data_vars=data_vars, coords=coords)
