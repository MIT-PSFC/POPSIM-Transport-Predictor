"""Tests for the helpers shared by every study's TrainRunBuilder."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from popsim.ml.dataloading import make_dataloaders

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.modules.trb_utils import (
    FixedStepsDataLoader,
    mask_to_largest_contiguous_segment,
    scalar_study_results,
)

nan = np.nan


def _make_ds(times, wtot, ip):
    return xr.Dataset(
        data_vars={
            "energy_mhd_MJ": ((EPISODE_DIM, TIME_DIM), np.asarray(wtot, dtype=np.float32)),
            "ip_MA": ((EPISODE_DIM, TIME_DIM), np.asarray(ip, dtype=np.float32)),
            "hazard": ((EPISODE_DIM,), np.arange(len(wtot), dtype=np.float32)),
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
    ds = mask_to_largest_contiguous_segment(_make_ds(times, wtot, ip), ["energy_mhd_MJ", "ip_MA"])

    got = ds["energy_mhd_MJ"].values
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
    ds = mask_to_largest_contiguous_segment(_make_ds(times, wtot, ip), ["energy_mhd_MJ", "ip_MA"])
    assert np.allclose(ds["energy_mhd_MJ"].values[0], [nan, nan, 3.0, 4.0, 5.0], equal_nan=True)


def test_per_shot_vars_keep_their_dims():
    times = [[0.000, 0.001, 0.002]]
    wtot = [[1.0, nan, 3.0]]
    ip = [[1.0, nan, 1.0]]
    ds = mask_to_largest_contiguous_segment(_make_ds(times, wtot, ip), ["energy_mhd_MJ", "ip_MA"])
    # force_drop_nans-style ds.where would broadcast this to (shot, time)
    assert ds["hazard"].dims == (EPISODE_DIM,)
    assert np.allclose(ds["hazard"].values, [0.0])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_scalar_results_flag_non_finite_predictions(bad_value):
    """A non-finite prediction is flagged diverged with NaN errors, never inf, and a missing target is neither."""
    targ = np.array([[1.0, 1.0, 1.0, nan]])
    pred = np.array([[1.5, bad_value, 1.0, nan]])
    coords = {EPISODE_DIM: [7], TIME_DIM: np.arange(4), TIME_COORD: ((EPISODE_DIM, TIME_DIM), 1e-3 * np.arange(4)[None, :])}
    dims = (EPISODE_DIM, TIME_DIM)
    stack = {"sample": dims}
    input_ds = xr.Dataset({"energy_mhd_MJ": (dims, targ)}, coords=coords).assign_coords(ds_source="mast").stack(stack)
    output_ds = xr.Dataset({"output.energy_mhd_MJ_pred": (dims, pred)}, coords=coords).stack(stack)

    ds = scalar_study_results(SimpleNamespace(input_ds=input_ds, output_ds=output_ds), "energy_mhd_MJ", "output.energy_mhd_MJ_pred")

    np.testing.assert_array_equal(ds["error_diverged_ts"].values[0], [0.0, 1.0, 0.0, nan])
    np.testing.assert_array_equal(np.isnan(ds["error_abs_ts"].values[0]), [False, True, False, True])
    assert np.isfinite(ds["error_abs_shot"].values).all()


def _timeslice_loader(n_shots: int, n_times: int, batch_size: int):
    """A shuffled time-independent training loader whose sample_id variable names each timeslice."""
    sample_id = np.arange(n_shots * n_times, dtype=np.float32).reshape(n_shots, n_times)
    times = np.tile(1e-3 * np.arange(n_times), (n_shots, 1))
    ds = xr.Dataset(
        {"sample_id": ((EPISODE_DIM, TIME_DIM), sample_id), "target": ((EPISODE_DIM, TIME_DIM), sample_id)},
        coords={EPISODE_DIM: np.arange(n_shots), TIME_COORD: ((EPISODE_DIM, TIME_DIM), times)},
    )
    (train_dl,) = make_dataloaders(
        datasets=(ds,),
        time_coord=TIME_COORD,
        episode_coord=EPISODE_DIM,
        input_vars=["sample_id"],
        target_vars=["target"],
        convert_xr_to_jnp=False,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return train_dl


def test_fixed_steps_epochs_chain_reshuffled_passes():
    """Every epoch is exactly steps_per_epoch full batches, and each complete pass inside it covers every sample once in a new order."""
    natural_dl = _timeslice_loader(n_shots=3, n_times=4, batch_size=4)
    assert len(natural_dl) == 3
    fixed_dl = FixedStepsDataLoader(natural_dl, steps_per_epoch=7)

    assert len(fixed_dl) == 7
    for _ in range(2):
        batch_ids = [batch.ds["sample_id"].values for batch in fixed_dl]
        assert [ids.size for ids in batch_ids] == [4] * 7
        first_pass = np.concatenate(batch_ids[0:3])
        second_pass = np.concatenate(batch_ids[3:6])
        for pass_ids in (first_pass, second_pass):
            assert sorted(pass_ids.tolist()) == list(range(12))
        assert first_pass.tolist() != second_pass.tolist()


def test_fixed_steps_refuse_a_training_set_above_the_study_steps():
    natural_dl = _timeslice_loader(n_shots=3, n_times=4, batch_size=4)
    with pytest.raises(ValueError, match="more than the study's 2 steps per epoch"):
        FixedStepsDataLoader(natural_dl, steps_per_epoch=2)
