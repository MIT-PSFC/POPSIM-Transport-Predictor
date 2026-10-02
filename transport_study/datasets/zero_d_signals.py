"""0D signal helpers of the raw-file devices (DIII-D, TCV).

Kept identical to their namesakes in transport-validation-datasets (machine/generic.py),
which builds the C-Mod and MAST stores, so every device shares one definition.
"""

import numpy as np
import xarray as xr

# Vacuum permeability [H/m]
MU0 = 4e-7 * np.pi


def trailing_boxcar_mean(values: np.ndarray, window: float, dt: float) -> np.ndarray:
    """Smooth a uniformly sampled signal causally, each sample the mean of itself and those before it in the window.

    NaN samples are skipped, and the first samples average over what is there.

    Args:
        values: (n,) the signal.
        window: Width of the boxcar [s].
        dt: The sample spacing [s].

    Returns:
        (n,) the smoothed signal, NaN only where the whole window is.
    """
    n_samples = max(1, round(window / dt))
    signal = xr.DataArray(values, dims="time")
    smoothed = signal.rolling(time=n_samples, min_periods=1).mean()
    return smoothed.values


def ohmic_power(
    times: np.ndarray,
    ip: np.ndarray,
    v_loop: np.ndarray,
    li: np.ndarray,
    major_radius: np.ndarray,
) -> np.ndarray:
    """Ohmic power P_oh = Ip V_loop - dW_pol/dt, causal.

    The internal poloidal field energy is W_pol = L_i Ip^2 / 2 with L_i = mu0 R0 li / 2,
    R0 the geometric major radius.
    dW_pol/dt is a backward difference, so no sample draws on a later one,
    and the first sample is NaN.
    The sign of Ip and V_loop cancels as long as they share a convention.

    Args:
        times: (n,) ascending sample times [s].
        ip: (n,) plasma current [A].
        v_loop: (n,) loop voltage [V].
        li: (n,) internal inductance.
        major_radius: (n,) major radius of the geometric center of the boundary [m].

    Returns:
        (n,) ohmic power [W].
    """
    w_pol = MU0 * major_radius * li * ip**2 / 4.0
    dw_pol = np.diff(w_pol, prepend=np.nan)
    dt = np.diff(times, prepend=np.nan)
    dw_pol_dt = dw_pol / dt
    return ip * v_loop - dw_pol_dt


def greenwald_fraction(ip, minor_radius, n_e_line_average):
    """Line-averaged density over the Greenwald density n_GW = Ip / (pi a^2), in 1e20 m^-3, MA and m.

    Args:
        ip: Plasma current [A], either sign.
        minor_radius: Minor radius [m].
        n_e_line_average: Line-averaged electron density [m^-3].

    Returns:
        The Greenwald fraction, shaped like the broadcast inputs.
    """
    ip_magnitude_ma = abs(ip) / 1e6
    n_greenwald = 1e20 * ip_magnitude_ma / (np.pi * minor_radius**2)
    return n_e_line_average / n_greenwald


def end_of_shot_index(ip_magnitude: np.ndarray, times: np.ndarray, ip_min: float, end_margin: float) -> int | None:
    """Index of the first grid time the end-of-shot cut removes.

    The plasma ends at the last grid time with |ip| at or above ip_min,
    since a record can hold ip near 0 long after the plasma is gone.
    Everything after end_margin before that is cut, to leave out the termination, often a disruption.
    The margin is counted in grid steps, so float round-off in the times never moves the cut.

    Args:
        ip_magnitude: (n_t,) |ip| on the uniform grid [A], NaN where unknown.
        times: (n_t,) the grid [s].
        ip_min: The ip min_filter threshold [A].
        end_margin: Margin before the end of the plasma [s].

    Returns:
        The index, or None when |ip| never reaches ip_min.
    """
    with np.errstate(invalid="ignore"):
        mask_plasma = ip_magnitude >= ip_min
    if not mask_plasma.any():
        return None
    last_plasma_idx = int(np.flatnonzero(mask_plasma)[-1])
    grid_steps = np.diff(times)
    margin_steps = round(end_margin / float(np.median(grid_steps)))
    return last_plasma_idx - margin_steps + 1


def _kept_segments(keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the runs of kept samples.

    Args:
        keep: Mask over the uniform 1 kHz grid.

    Returns:
        (starts, ends): the index of each run's first sample and one past its last.
    """
    # Pad with False on both sides so a run touching either end still has an edge
    keep_padded = np.concatenate(([False], keep, [False]))
    edges = np.diff(keep_padded.astype(np.int8))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def keep_longest_segment(keep: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Clear every run of kept samples but the longest.

    A run is measured from its first to its last sample, the same way kept_span measures the one left,
    so a run of n samples on the 1 kHz grid is n - 1 milliseconds long.
    The earliest of equally long runs is kept.

    Args:
        keep: Mask over times, True where the sample survived the filters. Not modified.
        times: The shot's timebase [s].

    Returns:
        The mask with only the longest run left, and the lengths [s] of the runs cleared.
    """
    keep = np.asarray(keep, dtype=bool)
    keep_longest = np.zeros(keep.size, dtype=bool)
    starts, ends = _kept_segments(keep)
    if starts.size == 0:
        return keep_longest, []
    run_lengths = times[ends - 1] - times[starts]
    longest = int(np.argmax(run_lengths))
    keep_longest[starts[longest] : ends[longest]] = True
    dropped_lengths = np.delete(run_lengths, longest)
    return keep_longest, dropped_lengths.tolist()


def kept_span(keep: np.ndarray, times: np.ndarray) -> float:
    """Time from the first to the last kept sample, for the min_pulse_length gate.

    Args:
        keep: Mask over times, True where the sample survived the filters.
        times: The shot's timebase [s].

    Returns:
        The span [s], 0 when fewer than two samples are kept.
    """
    keep = np.asarray(keep, dtype=bool)
    kept_times = times[keep]
    if kept_times.size < 2:
        return 0.0
    return float(kept_times[-1] - kept_times[0])
