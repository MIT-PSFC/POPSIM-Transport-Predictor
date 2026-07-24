import numpy as np
import xarray as xr

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.modules.power_balance.trb import (
    mask_to_largest_contiguous_segment,
)

nan = np.nan


def _make_ds(times, wtot, ip):
    return xr.Dataset(
        data_vars={
            "Wtot_MJ": ((EPISODE_DIM, TIME_DIM), np.asarray(wtot, dtype=np.float32)),
            "Ip_MA": ((EPISODE_DIM, TIME_DIM), np.asarray(ip, dtype=np.float32)),
            "performance": ((EPISODE_DIM,), np.arange(len(wtot), dtype=np.float32)),
        },
        coords={
            EPISODE_DIM: np.arange(len(wtot)),
            TIME_COORD: ((EPISODE_DIM, TIME_DIM), np.asarray(times, dtype=np.float32)),
        },
    )


def test_keeps_only_longest_contiguous_run():
    # Shot 0: run of 2, gap, run of 3 -> keep the run of 3
    # Shot 1: fully contiguous -> unchanged
    times = [
        [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
        [0.000, 0.001, 0.002, 0.003, 0.004, nan, nan],
    ]
    wtot = [
        [1.0, 2.0, nan, 4.0, 5.0, 6.0, nan],
        [1.0, 2.0, 3.0, 4.0, 5.0, nan, nan],
    ]
    ip = [
        [1.0, 1.0, nan, 1.0, 1.0, 1.0, nan],
        [1.0, 1.0, 1.0, 1.0, 1.0, nan, nan],
    ]
    ds = mask_to_largest_contiguous_segment(_make_ds(times, wtot, ip), ["Wtot_MJ", "Ip_MA"])

    got = ds["Wtot_MJ"].values
    assert np.allclose(got[0], [nan, nan, nan, 4.0, 5.0, 6.0, nan], equal_nan=True)
    assert np.allclose(got[1], [1.0, 2.0, 3.0, 4.0, 5.0, nan, nan], equal_nan=True)

    # Time coordinate is masked alongside so the run reads as the episode window
    t = ds[TIME_COORD].values
    assert np.isnan(t[0, :3]).all()
    assert np.allclose(t[0, 3:6], [0.003, 0.004, 0.005], atol=1e-6)
    assert np.isnan(t[0, 6])


def test_gap_in_any_training_var_splits_the_run():
    # Wtot is contiguous but Ip has a gap, the mask must respect the union
    times = [[0.000, 0.001, 0.002, 0.003, 0.004]]
    wtot = [[1.0, 2.0, 3.0, 4.0, 5.0]]
    ip = [[1.0, nan, 1.0, 1.0, 1.0]]
    ds = mask_to_largest_contiguous_segment(_make_ds(times, wtot, ip), ["Wtot_MJ", "Ip_MA"])
    assert np.allclose(ds["Wtot_MJ"].values[0], [nan, nan, 3.0, 4.0, 5.0], equal_nan=True)


def test_per_shot_vars_keep_their_dims():
    times = [[0.000, 0.001, 0.002]]
    wtot = [[1.0, nan, 3.0]]
    ip = [[1.0, nan, 1.0]]
    ds = mask_to_largest_contiguous_segment(_make_ds(times, wtot, ip), ["Wtot_MJ", "Ip_MA"])
    # force_drop_nans-style ds.where would broadcast this to (shot, time)
    assert ds["performance"].dims == (EPISODE_DIM,)
    assert np.allclose(ds["performance"].values, [0.0])
