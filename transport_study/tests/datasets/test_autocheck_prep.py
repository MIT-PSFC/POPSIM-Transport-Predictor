"""Tests for the contiguous piece split of the autocheck time_dep store."""

import numpy as np

from transport_study.datasets.autocheck_prep import contiguous_piece_bounds


def test_pieces_split_at_time_gaps_and_unclean_slices():
    """A piece ends at a time gap or an unclean slice, and runs of one slice are dropped.

    Shot 0 has a 3 ms gap before slice 3, then padding.
    Shot 1 has unclean slices 2 and 5, which leave slice 6 alone at the end.
    """
    time = np.array(
        [
            [0.000, 0.001, 0.002, 0.005, 0.006, np.nan, np.nan],
            [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
        ]
    )
    mask_slice_clean = ~np.isnan(time)
    mask_slice_clean[1, [2, 5]] = False

    bounds = contiguous_piece_bounds(time, mask_slice_clean)

    assert bounds == [(0, 0, 3), (0, 3, 5), (1, 0, 2), (1, 3, 5)]
