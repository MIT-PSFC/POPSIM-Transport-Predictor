"""Profile slices held onto the 1 kHz timebase, shared by the raw-file devices (DIII-D, TCV).

Profile diagnostics sample far slower than 1 kHz,
so each slice is held forward over the grid times that follow it.
A capped hold carries nothing across the end of a shot or a diagnostic dropping out.
"""

import numpy as np


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
