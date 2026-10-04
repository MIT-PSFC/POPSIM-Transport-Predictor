"""Prepare a device store (signals.STUDY_STORE_SIGNALS) for popsim-autocheck.

Writes two Zarr stores next to each other:
    <label>_indep.zarr: the shots as episodes on the store's time_idx layout, for time_indep mode.
    <label>_dep.zarr: every contiguous clean piece of a shot as an episode, 0D signals only, for time_dep mode.

The bookkeeping flags, the profile error bars, the per-shot statics like r0
and every signal constant over the whole store are not modeled.
The signed signals are taken as magnitudes, as the study takes them (signals.SIGNED_SIGNALS).
time_dep leaves the profiles out, since they step at every fresh slice.
A piece ends wherever the time step exceeds MAX_CONTIGUOUS_DT_S (slices removed by the dataset filters)
or a modeled 0D signal is NaN, so no time_dep segment spans a gap or gets dropped for a NaN.
Piece episode ids are shot * PIECE_ID_FACTOR + piece index.

Then run popsim-autocheck with --mode time_indep on the first store and --mode time_dep on the second.
"""

from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger
from transport_validation_datasets.machine.generic import UNIFORM_TIMEBASE_DT

from transport_study import EPISODE_DIM, RADIAL_DIM, TIME_COORD, TIME_DIM
from transport_study.signals import SIGNED_SIGNALS

# Bookkeeping flags and the profile error bars, only the profiles and gradients are checked.
# The study stores leave fresh_equilibrium out, the TVD stores keep it.
EXCLUDED_VARS = (
    "fresh_profile",
    "fresh_equilibrium",
    "t_e_error",
    "n_e_error",
    "t_e_gradient_error",
    "n_e_gradient_error",
)
# Longest step between two slices of one piece, half a step of slack on the uniform timebase
MAX_CONTIGUOUS_DT_S = 1.5 * UNIFORM_TIMEBASE_DT
PIECE_ID_FACTOR = 1000
# Shots per chunk of the written stores
STORE_CHUNK_SHOTS = 32


def _constant_vars(ds: xr.Dataset) -> list[str]:
    """Variables whose every feature holds one value over all shots and times, ignoring NaN."""
    constant = []
    for name, da in ds.data_vars.items():
        n_rows = da.sizes[EPISODE_DIM] * da.sizes[TIME_DIM]
        values_rows = da.transpose(EPISODE_DIM, TIME_DIM, ...).values.reshape(n_rows, -1)
        feature_range = np.nanmax(values_rows, axis=0) - np.nanmin(values_rows, axis=0)
        mask_feature_finite = ~np.isnan(feature_range)
        if np.all(feature_range[mask_feature_finite] == 0):
            constant.append(str(name))
    return constant


def contiguous_piece_bounds(time: np.ndarray, mask_slice_clean: np.ndarray) -> list[tuple[int, int, int]]:
    """(shot index, start, stop) of every run of at least two clean slices without a time gap.

    A run ends at an unclean slice or where the step from the previous slice exceeds MAX_CONTIGUOUS_DT_S.

    Args:
        time: (n_shots, n_times) slice times [s], NaN on the padding.
        mask_slice_clean: (n_shots, n_times) True where the slice is finite in every modeled signal.
    """
    bounds = []
    for shot_idx in range(time.shape[0]):
        time_shot = time[shot_idx]
        mask_clean = mask_slice_clean[shot_idx]
        dt = np.diff(time_shot)
        # A NaN step counts as a gap
        mask_gap_before = np.concatenate([[True], ~(dt <= MAX_CONTIGUOUS_DT_S)])
        start = None
        for idx in range(time_shot.size):
            if not mask_clean[idx]:
                if start is not None:
                    bounds.append((shot_idx, start, idx))
                    start = None
            elif start is None:
                start = idx
            elif mask_gap_before[idx]:
                bounds.append((shot_idx, start, idx))
                start = idx
        if start is not None:
            bounds.append((shot_idx, start, time_shot.size))
    return [(shot_idx, start, stop) for shot_idx, start, stop in bounds if stop - start >= 2]


def load_modeled_store(store_path: Path | str) -> xr.Dataset:
    """A device store with time as a coordinate and magnitudes of the signed signals,
    without EXCLUDED_VARS, the per-shot statics and the constant variables.
    """
    ds = xr.open_zarr(store_path)
    excluded_present = [name for name in EXCLUDED_VARS if name in ds.data_vars]
    static_vars = [name for name, da in ds.data_vars.items() if TIME_DIM not in da.dims]
    ds = ds.drop_vars([*excluded_present, *static_vars]).set_coords(TIME_COORD).load()
    # The source chunk encodings do not fit the rechunked stores written here
    for variable in ds.variables.values():
        variable.encoding = {}
    for name in SIGNED_SIGNALS:
        ds[name] = abs(ds[name]).assign_attrs(ds[name].attrs)
    constant = _constant_vars(ds)
    logger.info(f"Dropping the constant variables {constant}")
    ds = ds.drop_vars(constant)
    return _drop_empty_radial_points(ds)


def _drop_empty_radial_points(ds: xr.Dataset) -> xr.Dataset:
    """ds without the radial points where some radial variable is NaN in every slice.

    TCV profiles end at the LCFS, so their points past rho_tor_norm 1 would leave no slice clean.
    """
    radial_vars = [name for name, da in ds.data_vars.items() if RADIAL_DIM in da.dims]
    mask_point_kept = np.ones(ds.sizes.get(RADIAL_DIM, 0), dtype=bool)
    for name in radial_vars:
        mask_finite = ds[name].notnull()
        other_dims = [dim for dim in mask_finite.dims if dim != RADIAL_DIM]
        mask_point_finite = mask_finite.any(other_dims)
        mask_point_kept &= mask_point_finite.transpose(RADIAL_DIM).values
    if mask_point_kept.all():
        return ds
    radial_dropped = ds[RADIAL_DIM].values[~mask_point_kept]
    logger.info(f"Dropping the {radial_dropped.size} {RADIAL_DIM} points no slice fills, {radial_dropped}")
    return ds.isel({RADIAL_DIM: mask_point_kept})


def _piece_dataset(ds: xr.Dataset, bounds: list[tuple[int, int, int]]) -> xr.Dataset:
    """Every piece of bounds as an episode, padded with NaN to the longest piece."""
    piece_lengths = [stop - start for _, start, stop in bounds]
    n_times_max = max(piece_lengths)

    shots = ds[EPISODE_DIM].values
    piece_counts: dict[int, int] = {}
    piece_ids = []
    for shot_idx, _, _ in bounds:
        shot = int(shots[shot_idx])
        piece_counts[shot] = piece_counts.get(shot, 0) + 1
        piece_ids.append(shot * PIECE_ID_FACTOR + piece_counts[shot] - 1)

    data_vars = {}
    for name in [*ds.data_vars, TIME_COORD]:
        da = ds[name].transpose(EPISODE_DIM, TIME_DIM, ...)
        values = da.values
        values_pieces = np.full((len(bounds), n_times_max, *values.shape[2:]), np.nan, dtype=values.dtype)
        for piece_idx, (shot_idx, start, stop) in enumerate(bounds):
            values_pieces[piece_idx, : stop - start] = values[shot_idx, start:stop]
        data_vars[name] = (da.dims, values_pieces, da.attrs)
    coords: dict = {name: ds[name] for name in ds.coords if name not in (EPISODE_DIM, TIME_COORD)}
    coords[EPISODE_DIM] = np.array(piece_ids, dtype=np.int64)
    ds_pieces = xr.Dataset(data_vars, coords=coords, attrs=ds.attrs)
    return ds_pieces.set_coords(TIME_COORD)


def prep_autocheck(store_path: Path | str, label: str, out_dir: Path | str):
    """Write <label>_indep.zarr and <label>_dep.zarr for a device store into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_modeled_store(store_path)
    ds.attrs = {"source": str(store_path), "modeled_vars": ",".join(str(name) for name in ds.data_vars)}
    logger.info(f"{label}: modeled variables {list(ds.data_vars)}")
    ds.chunk({EPISODE_DIM: STORE_CHUNK_SHOTS}).to_zarr(out_dir / f"{label}_indep.zarr", mode="w")

    radial_vars = [name for name, da in ds.data_vars.items() if RADIAL_DIM in da.dims]
    ds_0d = ds.drop_vars(radial_vars)
    if RADIAL_DIM in ds_0d.dims:
        ds_0d = ds_0d.drop_dims(RADIAL_DIM)
    ds_0d.attrs = {**ds.attrs, "modeled_vars": ",".join(str(name) for name in ds_0d.data_vars)}
    logger.info(f"{label}: time_dep modeled variables {list(ds_0d.data_vars)}")
    time = ds_0d[TIME_COORD].values
    mask_slice_clean = ~np.isnan(time)
    for da in ds_0d.data_vars.values():
        mask_nan = da.isnull().transpose(EPISODE_DIM, TIME_DIM).values
        mask_slice_clean &= ~mask_nan
    bounds = contiguous_piece_bounds(time, mask_slice_clean)
    piece_lengths = np.array([stop - start for _, start, stop in bounds])
    logger.info(
        f"{label}: {mask_slice_clean.sum()} clean slices in {len(bounds)} pieces, "
        f"length median {np.median(piece_lengths)} max {piece_lengths.max()}"
    )

    ds_pieces = _piece_dataset(ds_0d, bounds)
    ds_pieces.chunk({EPISODE_DIM: STORE_CHUNK_SHOTS}).to_zarr(out_dir / f"{label}_dep.zarr", mode="w")
    logger.info(f"{label}: wrote both autocheck stores to {out_dir}")
