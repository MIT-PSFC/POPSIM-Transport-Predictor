"""Unit tests for the chi per-case metrics in profile_transfer.study_metrics,
on synthetic data with no training involved.

The chi itself is computed by the test suite (tests/transport_transfer/test_study_results.py),
these tests cover reading it back per retained checkpoint, the stage join and the per-case cache.
The stage segmentation these join against is shared machinery covered by
tests/orchestration/test_stages.py.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from transport_study import RADIAL_DIM
from transport_study.config import RHO_GRID
from transport_study.orchestration.case_metrics import (
    case_metrics_path,
    compute_and_save_case_metrics,
)
from transport_study.orchestration.case_reports import analysis_case_done
from transport_study.orchestration.stages import STAGE_AGG_NAMES
from transport_study.orchestration.topk_results import aggregate_topk_results
from transport_study.profile_transfer import study_metrics
from transport_study.profile_transfer.profile_study import ProfileStudy
from transport_study.profile_transfer.study_metrics import (
    compute_case_timeslice_metrics,
)

GRADIENT_WEIGHT = 0.1
BEST_EPOCH = 4
OTHER_EPOCH = 8

RESULT_TIMES = np.array([0.1, 0.2, 0.3, 0.4])
EVAL_TIMES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
# Flattop threshold 0.9 * p95: eval indices 2..4 are flattop, 0..1 rampup, 5 rampdown
EVAL_IP = np.array([0.2, 0.5, 1.0, 1.0, 1.0, 0.3])
EVAL_P_NBI = np.array([0.0, 0.0, 0.05, 0.5, 0.0, 0.0])
# Per-timeslice chi of the best checkpoint, the other retained checkpoint scores twice as much
CHI_VALUE = np.array([1.0, 2.0, 3.0, 4.0])
CHI_GRAD = np.array([10.0, 0.0, 10.0, 0.0])


def _step_result(times: np.ndarray, chi_scale: float) -> xr.Dataset:
    """One checkpoint's study_results of one shot, shaped like the test suite output."""
    n_ts = len(times)
    profile = np.broadcast_to(1.0 - 0.5 * RHO_GRID, (1, n_ts, len(RHO_GRID))).copy()
    dims_profile = ("shot", "time_idx", RADIAL_DIM)
    dims_time = ("shot", "time_idx")
    chi_value = chi_scale * CHI_VALUE[None, :n_ts]
    chi_grad = chi_scale * CHI_GRAD[None, :n_ts]
    return xr.Dataset(
        data_vars={
            "n_e_1e20_pred": (dims_profile, profile),
            "n_e_1e20_targ": (dims_profile, profile),
            "t_e_keV_pred": (dims_profile, profile),
            "t_e_keV_targ": (dims_profile, profile),
            "error_chi_value_ts": (dims_time, chi_value),
            "error_chi_grad_ts": (dims_time, chi_grad),
            "error_chi_ts": (dims_time, chi_value + GRADIENT_WEIGHT * chi_grad),
            "error_diverged_ts": (dims_time, np.zeros((1, n_ts))),
        },
        coords={
            "shot": [1],
            "time_idx": np.arange(n_ts),
            RADIAL_DIM: RHO_GRID,
            "time": (dims_time, times[None, :]),
            "ds_source": (("shot",), np.array(["devA"], dtype=object)),
        },
    )


def _make_result_ds(times: np.ndarray = RESULT_TIMES) -> xr.Dataset:
    """A case result of two retained checkpoints."""
    return aggregate_topk_results({BEST_EPOCH: _step_result(times, 1.0), OTHER_EPOCH: _step_result(times, 2.0)}, BEST_EPOCH)


def _make_eval_ds() -> xr.Dataset:
    """Device dataset with only what the stage labels read."""
    return xr.Dataset(
        data_vars={
            "ip_MA": (("shot", "time_idx"), EVAL_IP[None, :]),
            "power_additional_MW": (("shot", "time_idx"), EVAL_P_NBI[None, :]),
        },
        coords={"shot": [1], "time_idx": np.arange(len(EVAL_TIMES)), "time": (("shot", "time_idx"), EVAL_TIMES[None, :])},
    )


@pytest.fixture()
def patched_eval(monkeypatch):
    monkeypatch.setattr(study_metrics, "load_eval_dataset", lambda device: _make_eval_ds())


class TestComputeCaseTimesliceMetrics:
    def test_reads_the_suite_chi_of_every_checkpoint(self, patched_eval):
        ts = compute_case_timeslice_metrics(_make_result_ds())
        assert len(ts) == 4
        np.testing.assert_allclose(ts.best("value"), CHI_VALUE)
        np.testing.assert_allclose(ts.best("combined"), CHI_VALUE + GRADIENT_WEIGHT * CHI_GRAD)
        np.testing.assert_allclose(ts.metric("grad"), np.stack([CHI_GRAD, 2.0 * CHI_GRAD], axis=1))
        np.testing.assert_allclose(ts.best("diverged"), 0.0)

    def test_time_join_and_stage_labels(self, patched_eval):
        ts = compute_case_timeslice_metrics(_make_result_ds())
        # Result times 0.1..0.4 match eval indices 1..4
        np.testing.assert_array_equal(ts.eval_time_idx, [1, 2, 3, 4])
        np.testing.assert_array_equal(ts.result_time_idx, [0, 1, 2, 3])
        np.testing.assert_array_equal(ts.stage, ["rampup", "flattop", "flattop", "flattop"])
        np.testing.assert_array_equal(ts.aux_heated, [False, False, True, False])
        # Flattop subdivision via the stage masks
        np.testing.assert_array_equal(ts.stage_mask("flattop_ohmic"), [False, True, False, True])
        np.testing.assert_array_equal(ts.stage_mask("flattop_aux"), [False, False, True, False])
        np.testing.assert_array_equal(ts.stage_mask("all"), [True, True, True, True])

    def test_nan_padded_timeslices_are_dropped(self, patched_eval):
        times = RESULT_TIMES.copy()
        times[-1] = np.nan
        ts = compute_case_timeslice_metrics(_make_result_ds(times))
        assert len(ts) == 3


class _StubStudy:
    """Just enough of the Study interface for the per-case metric cache."""

    ANALYSIS_METRICS_MODULE = ProfileStudy.ANALYSIS_METRICS_MODULE
    ANALYSIS_REPORTS_MODULE = ProfileStudy.ANALYSIS_REPORTS_MODULE

    def __init__(self, tmp_path: Path):
        self.result_dir = tmp_path / "results"
        self.figure_dir = tmp_path / "figures"

    def result_path(self, case) -> Path:
        return self.result_dir / str(case) / "result_data.nc"

    def is_borrowed(self, case) -> bool:
        return False


class TestCaseMetricsCache:
    def _stub_with_result(self, tmp_path: Path, case: str = "case.test") -> _StubStudy:
        study = _StubStudy(tmp_path)
        result_path = study.result_path(case)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        _make_result_ds().to_netcdf(result_path)
        return study

    def test_missing_result_returns_empty_without_caching(self, tmp_path):
        study = _StubStudy(tmp_path)
        case_ds = compute_and_save_case_metrics(study, "case.test")
        assert not case_ds.data_vars
        assert not case_metrics_path(study, "case.test").exists()

    def test_computes_aggregates_and_caches(self, tmp_path, patched_eval):
        study = self._stub_with_result(tmp_path)

        case_ds = compute_and_save_case_metrics(study, "case.test")
        assert case_metrics_path(study, "case.test").exists()
        assert list(case_ds["stage"].values) == list(STAGE_AGG_NAMES)
        best_mean = CHI_VALUE.mean()
        # The best checkpoint, the top-K mean and std of the two checkpoints' case scores
        np.testing.assert_allclose(case_ds["value_mean_best"].sel(stage="all").item(), best_mean)
        np.testing.assert_allclose(case_ds["value_mean"].sel(stage="all").item(), 1.5 * best_mean)
        np.testing.assert_allclose(case_ds["value_mean_ckpt_std"].sel(stage="all").item(), 0.5 * best_mean)
        assert case_ds["value_count"].sel(stage="all").item() == 4
        # From EVAL_IP / EVAL_P_NBI: rampup at 0.1s, flattop at 0.2-0.4s with one aux slice
        assert case_ds["value_count"].sel(stage="rampup").item() == 1
        assert case_ds["value_count"].sel(stage="flattop").item() == 3
        assert case_ds["value_count"].sel(stage="flattop_aux").item() == 1
        assert case_ds["value_count"].sel(stage="rampdown").item() == 0

    def test_second_call_uses_cache(self, tmp_path, patched_eval, monkeypatch):
        study = self._stub_with_result(tmp_path)
        first = compute_and_save_case_metrics(study, "case.test")

        def _boom(*args, **kwargs):
            raise AssertionError("cache should have been used")

        monkeypatch.setattr(study_metrics, "compute_case_timeslice_metrics", _boom)
        second = compute_and_save_case_metrics(study, "case.test")
        xr.testing.assert_allclose(first, second)

    def test_analysis_case_done(self, tmp_path, patched_eval):
        study = _StubStudy(tmp_path)
        case = "case.test"
        # No metrics cache yet
        assert not analysis_case_done(study, case, study.figure_dir)

        # Empty marker means nothing valid to report, so the case is done
        marker_path = case_metrics_path(study, case)
        marker_path.parent.mkdir(parents=True)
        xr.Dataset().to_netcdf(marker_path)
        assert analysis_case_done(study, case, study.figure_dir)

        # Non-empty metrics additionally require the report artifacts
        marker_path.unlink()
        study = self._stub_with_result(tmp_path)
        compute_and_save_case_metrics(study, case)
        assert not analysis_case_done(study, case, study.figure_dir)

        case_dir = study.figure_dir / "case_reports" / case
        case_dir.mkdir(parents=True)
        (case_dir / "best_worst_timeslices.pdf").touch()
        assert not analysis_case_done(study, case, study.figure_dir)
        (case_dir / "shot_1_evolution.gif").touch()
        assert analysis_case_done(study, case, study.figure_dir)
