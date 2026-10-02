from pathlib import Path

import numpy as np

# Nominal step of the uniform timebase every device workflow builds with
# make_uniform_1khz_timebase
UNIFORM_TIMEBASE_DT_S = 1e-3


def make_uniform_1khz_timebase(max_time: float) -> np.ndarray:
    """
    Create a uniform timebase at 1 kHz up to the specified maximum time.
    This is the timebase used for all datasets throughout the study.
    Built from an integer millisecond count so no sample can be dropped or duplicated by float accumulation.

    Parameters
    ----------
    max_time : float
        The maximum time for the timebase [s].
    """
    last_ms = int(np.ceil(np.round(max_time * 1000, 6)))
    times = np.round(np.arange(last_ms + 1, dtype=np.float64) * 1e-3, 3).astype("float32")
    return times


def read_shotlist(shotlist_file: Path | str) -> list[int]:
    """Shots of a shotlist file, one per line, in file order. Lines that are not a shot number are skipped."""
    with open(shotlist_file) as f:
        lines = [line.strip() for line in f]
    return [int(line) for line in lines if line.isdigit()]
