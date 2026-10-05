"""The padded rollout tail stays out of the analysis metrics.

The time-dep rollout batches pad every shot to a common length by repeating its final timeslice with a clamped time (not NaN).
Unmasked, those repeats would weight the per-timeslice stats and the stage-resolved time averages toward each shot's final error.
"""

import numpy as np
import pytest
import xarray as xr

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.config import StudyConfig, load_config
from transport_study.modules.trb_utils import integrate_error_over_time
from transport_study.orchestration.study import real_timeslice_mask, write_netcdf_atomic
from transport_study.power_balance_transfer.study_metrics import (
    compute_case_timeslice_metrics,
    load_stage_dataset,
    shot_time_averaged_errors,
)
from transport_study.tests.datasets.synthetic_store import DT, SHOT_LENGTHS, T_START
from transport_study.tests.stubs import StubCase

N_PAD_REPEATS = 20


def padded_result(errors_real: np.ndarray, time_real: np.ndarray, shot: int) -> xr.Dataset:
    """One-shot result file whose final timeslice repeats N_PAD_REPEATS times at a clamped time, as the rollout batches pad."""
    errors = np.concatenate([errors_real, np.full(N_PAD_REPEATS, errors_real[-1])])
    time = np.concatenate([time_real, np.full(N_PAD_REPEATS, time_real[-1])])
    dims = (EPISODE_DIM, TIME_DIM)
    return xr.Dataset(
        {
            "error_abs_ts": (dims, errors[None, :]),
            "error_rel_ts": (dims, 0.1 * errors[None, :]),
            "error_abs_shot": ((EPISODE_DIM,), [1.0]),
            "error_rel_shot": ((EPISODE_DIM,), [0.1]),
            TIME_COORD: (dims, time[None, :]),
        },
        coords={EPISODE_DIM: [shot]},
    ).assign_coords(ds_source=(EPISODE_DIM, ["mast"]))


def test_real_timeslice_mask_drops_padded_tails():
    time_ms = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            # Clamped repeats of the final time
            [0.0, 1.0, 2.0, 2.0, 2.0],
            # A tail advancing only by float jitter, below PAD_TIME_STEP_S
            [0.0, 1.0, 2.0, 2.0 + 1e-10, 2.0 + 2e-10],
            [0.0, 1.0, np.nan, np.nan, np.nan],
        ]
    )
    time_2d = xr.DataArray(1e-3 * time_ms, dims=(EPISODE_DIM, TIME_DIM))

    mask_real = real_timeslice_mask(time_2d)

    expected = [
        [True, True, True, True, True],
        [True, True, True, False, False],
        [True, True, True, False, False],
        [True, True, False, False, False],
    ]
    np.testing.assert_array_equal(mask_real.values, expected)


def test_shot_time_integral_unchanged_by_padding():
    """The repeats have zero dt, so the per-shot integrals need no masking."""
    errors_real = np.array([1.0, 2.0, 3.0])
    time_real = np.array([0.0, 1e-3, 2e-3])
    ds_padded = padded_result(errors_real, time_real, shot=101)
    ds_unpadded = padded_result(errors_real, time_real, shot=101).isel({TIME_DIM: slice(0, errors_real.size)})

    integral_padded = integrate_error_over_time(ds_padded["error_abs_ts"], ds_padded[TIME_COORD])
    integral_unpadded = integrate_error_over_time(ds_unpadded["error_abs_ts"], ds_unpadded[TIME_COORD])

    np.testing.assert_allclose(integral_padded.values, integral_unpadded.values)


def test_collected_timeslice_stats_ignore_padded_tail(make_stub_study):
    """collect_results summarizes only the real timeslices, the outlier final one counting once."""
    case = StubCase(name="case.a")
    study = make_stub_study([case])
    errors_real = np.array([1.0, 2.0, 3.0, 10.0])
    time_real = 1e-3 * np.arange(errors_real.size)
    write_netcdf_atomic(padded_result(errors_real, time_real, shot=101), study.result_path(case))

    summary = study.collect_results().isel(case_idx=0)

    assert float(summary["err_abs_ts_mean"]) == pytest.approx(errors_real.mean())
    assert float(summary["err_abs_ts_med"]) == pytest.approx(np.median(errors_real))
    assert float(summary["err_abs_ts_max"]) == pytest.approx(errors_real.max())


def test_stage_metrics_count_each_real_timeslice_once(synthetic_device_stores):
    """One record per real timeslice, so the time average is free of the padded repeats of the outlier final error."""
    load_config(StudyConfig(study_name="test-padded-tail", dataset_paths=synthetic_device_stores, target_device="mast"))
    # The store path is per session, a cached dataset of an earlier one would not join
    load_stage_dataset.cache_clear()
    shot = next(iter(SHOT_LENGTHS))
    errors_real = np.array([1.0, 1.0, 1.0, 1.0, 9.0])
    time_real = T_START + DT * np.arange(errors_real.size)

    ts_metrics = compute_case_timeslice_metrics(padded_result(errors_real, time_real, shot))
    _shots, avg_abs, _avg_rel, n_ts = shot_time_averaged_errors(ts_metrics)

    assert len(ts_metrics.shot) == errors_real.size
    assert n_ts.tolist() == [errors_real.size]
    assert avg_abs[0] == pytest.approx(errors_real.mean())
