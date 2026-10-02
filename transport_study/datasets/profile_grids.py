"""Samples placed causally onto the 1 kHz timebase, shared by the raw-file devices (DIII-D, TCV).

Profile diagnostics, and many 0D signals, sample slower than 1 kHz,
so each sample is held forward over the grid times that follow it, never interpolated.
A 0D signal sampled faster than 1 kHz is averaged over each grid step instead.
Either way no grid time draws on a later sample.
A capped hold carries nothing across the end of a shot or a diagnostic dropping out.
"""

import numpy as np

# Longest a 0D sample is held onto the timebase, in median sample steps of its signal.
# The same as transport-validation-datasets' MAX_HOLD_PERIODS.
MAX_HOLD_STEPS_0D = 1.5

# A time takes a slice this close after it as its own [s], the same as transport-validation-datasets' SAMPLE_TIME_TOL.
# The timebase is float32, so a source sample at the same millisecond can sit just above its time.
# float32 round-off stays under this below 16 s.
SAMPLE_TIME_TOL_S = 1e-6


def held_slice_index(slice_times: np.ndarray, times: np.ndarray, max_hold_steps: float) -> tuple[np.ndarray, np.ndarray]:
    """Which slice each time holds, and where that hold is valid.

    Each slice holds until the next one, for at most max_hold_steps median slice steps.
    A slice within SAMPLE_TIME_TOL_S after a time counts as at that time.
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
    times_float64 = times.astype(np.float64)
    slice_index = np.searchsorted(slice_times, times_float64 + SAMPLE_TIME_TOL_S, side="right") - 1
    slice_index_clipped = np.clip(slice_index, 0, None)
    hold_duration = times_float64 - slice_times[slice_index_clipped]
    mask_held = (slice_index >= 0) & (hold_duration <= max_hold)
    return slice_index_clipped, mask_held


def held_on_times(slice_profiles: np.ndarray, slice_index: np.ndarray, mask_held: np.ndarray) -> np.ndarray:
    """Per-slice profiles picked onto the timebase by slice index, float32, NaN where no slice holds."""
    profiles_on_times = slice_profiles[slice_index].astype(np.float32)
    profiles_on_times[~mask_held] = np.nan
    return profiles_on_times


def signal_on_grid(source_times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Place a 0D signal on the grid causally, so no grid value draws on a later sample.

    A source sampled faster than the grid is averaged over each grid step (_window_mean_on_grid).
    A slower one is held forward from its last sample (_held_on_grid).
    Only finite samples count.
    The source's period is the median spacing of its finite samples,
    so a fast clock populated only at a slower cadence is held on that cadence.
    Fewer than two finite samples have no period, and give all NaN.
    Kept identical to transport-validation-datasets' machine/generic.py signal_on_grid.

    Args:
        source_times: (n_source,) ascending sample times of the source [s].
        values: (n_source,) the signal at those times, NaN where missing.
        grid: (n_t,) the shot's 1 kHz timebase [s].

    Returns:
        (n_t,) the signal on the grid, NaN where it has no value.
    """
    mask_finite = np.isfinite(values)
    if mask_finite.sum() < 2:
        return np.full(grid.size, np.nan)
    times_finite = source_times[mask_finite]
    values_finite = values[mask_finite]
    source_steps = np.diff(times_finite)
    source_period = float(np.median(source_steps))
    grid_float64 = grid.astype(np.float64)
    grid_steps = np.diff(grid_float64)
    grid_step = float(np.median(grid_steps))
    if source_period < grid_step - SAMPLE_TIME_TOL_S:
        return _window_mean_on_grid(times_finite, values_finite, grid_float64, grid_step)
    return _held_on_grid(times_finite, values_finite, grid_float64)


def _held_on_grid(source_times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Each grid time takes the last sample at or before it, for at most MAX_HOLD_STEPS_0D median sample steps (held_slice_index).

    Args:
        source_times: (n_source,) ascending finite sample times [s], at least two.
        values: (n_source,) the samples.
        grid: (n_t,) the timebase [s].

    Returns:
        (n_t,) the held signal, NaN where nothing is held.
    """
    sample_index, mask_held = held_slice_index(source_times, grid, MAX_HOLD_STEPS_0D)
    values_on_grid = values[sample_index].astype(float)
    values_on_grid[~mask_held] = np.nan
    return values_on_grid


def _window_mean_on_grid(source_times: np.ndarray, values: np.ndarray, grid: np.ndarray, grid_step: float) -> np.ndarray:
    """Each grid time t takes the mean of the samples in (t - grid_step, t].

    A sample within SAMPLE_TIME_TOL_S after a grid time counts as at it, as in held_slice_index.

    Args:
        source_times: (n_source,) ascending finite sample times [s].
        values: (n_source,) the samples.
        grid: (n_t,) the uniform timebase [s], float64.
        grid_step: Its step [s].

    Returns:
        (n_t,) the window means, NaN where a window holds no sample.
    """
    first_window_start = grid[0] - grid_step
    window_edges = np.concatenate([[first_window_start], grid])
    window_index = np.searchsorted(window_edges, source_times - SAMPLE_TIME_TOL_S, side="left") - 1
    mask_on_grid = (window_index >= 0) & (window_index < grid.size)
    window_index_on_grid = window_index[mask_on_grid]
    values_in_windows = values[mask_on_grid]
    window_sums = np.bincount(window_index_on_grid, weights=values_in_windows, minlength=grid.size)
    window_counts = np.bincount(window_index_on_grid, minlength=grid.size)
    window_means = np.full(grid.size, np.nan)
    mask_sampled = window_counts > 0
    window_means[mask_sampled] = window_sums[mask_sampled] / window_counts[mask_sampled]
    return window_means
