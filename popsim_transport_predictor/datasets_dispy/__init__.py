import numpy as np


def make_uniform_1khz_timebase(max_time: float) -> np.ndarray:
    """
    Create a uniform timebase at 1 kHz up to the specified maximum time.
    This is the timebase used for all datasets throughout the study.
    Using arange and round ensures the timebase is consistent across all shots.

    Parameters
    ----------
    max_time : float
        The maximum time for the timebase [s].
    """
    times = np.round(np.arange(0, max_time + 1e-3, 1e-3), 3)
    times = np.unique(times).astype("float32")
    return times
