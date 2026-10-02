"""Slow samples held onto the 1 kHz timebase, shared by the raw-file devices (DIII-D, TCV).

Profile diagnostics, and many 0D signals, sample slower than 1 kHz,
so each sample is held forward over the grid times that follow it, never interpolated,
and no grid time draws on a later sample.
A capped hold carries nothing across the end of a shot or a diagnostic dropping out.
"""

import numpy as np

# Longest a 0D sample is held onto the timebase, in median sample steps of its signal.
# The same as transport-validation-datasets' MAX_HOLD_PERIODS.
MAX_HOLD_STEPS_0D = 1.5


def held_slice_index(slice_times: np.ndarray, times: np.ndarray, max_hold_steps: float) -> tuple[np.ndarray, np.ndarray]:
    """Which slice each time holds, and where that hold is valid.

    Each slice holds until the next one, for at most max_hold_steps median slice steps.
    Times before the first slice hold nothing.

    Args:
        slice_times: (n_slices,) sorted slice times [s], at least two.
        times: (n_t,) timebase [s].
        max_hold_steps: Longest hold, in median slice steps.

    Returns:
        (n_t,) index of the latest slice at or before each time (0 before the first slice),
        and (n_t,) True where a slice holds.
    """
    slice_steps = np.diff(slice_times)
    max_hold = max_hold_steps * np.median(slice_steps)
    slice_index = np.searchsorted(slice_times, times, side="right") - 1
    slice_index_clipped = np.clip(slice_index, 0, None)
    hold_duration = times - slice_times[slice_index_clipped]
    mask_held = (slice_index >= 0) & (hold_duration <= max_hold)
    return slice_index_clipped, mask_held


def held_on_times(slice_profiles: np.ndarray, slice_index: np.ndarray, mask_held: np.ndarray) -> np.ndarray:
    """Per-slice profiles picked onto the timebase by slice index, float32, NaN where no slice holds."""
    profiles_on_times = slice_profiles[slice_index].astype(np.float32)
    profiles_on_times[~mask_held] = np.nan
    return profiles_on_times


def held_signal_on_grid(source_times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Place a 0D signal on the grid by holding each finite sample forward (held_slice_index).

    Each grid time takes the last finite sample at or before it, held for at most MAX_HOLD_STEPS_0D
    median steps of the finite samples, so a gap in the source stays NaN.

    Args:
        source_times: (n_source,) ascending sample times of the source [s].
        values: (n_source,) the signal at those times, NaN where missing.
        grid: (n_t,) the shot's 1 kHz timebase [s].

    Returns:
        (n_t,) the signal on the grid, NaN where nothing is held.
    """
    mask_finite = np.isfinite(values)
    if mask_finite.sum() < 2:
        return np.full(grid.size, np.nan)
    sample_index, mask_held = held_slice_index(source_times[mask_finite], grid, MAX_HOLD_STEPS_0D)
    values_finite = values[mask_finite]
    values_on_grid = values_finite[sample_index].astype(float)
    values_on_grid[~mask_held] = np.nan
    return values_on_grid
