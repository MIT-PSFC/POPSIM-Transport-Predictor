"""Unit tests for the chi per-case metrics in profile_transfer.study_metrics,
on synthetic data with no training involved.

The stage segmentation these join against is shared machinery covered by
tests/orchestration/test_stages.py.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from transport_study import RADIAL_DIM
from transport_study.config import RHO_GRID
from transport_study.modules.trb_utils import CHI_ERROR_VARS, GRAD_RHO_MAX
from transport_study.orchestration.case_metrics import (
    case_metrics_path,
    compute_and_save_case_metrics,
)
from transport_study.orchestration.case_reports import analysis_case_done
from transport_study.orchestration.stages import STAGE_AGG_NAMES
from transport_study.profile_transfer import study_metrics
from transport_study.profile_transfer.profile_study import ProfileStudy
from transport_study.profile_transfer.study_metrics import (
    compute_case_timeslice_metrics,
)

# Peak-normalized error-bar floor, it applies wherever the synthetic error bars are zero
SIGMA_FLOOR = 0.01


def _make_result_ds(times: np.ndarray, ne_pred, ne_targ, te_pred, te_targ, shot: int = 1) -> xr.Dataset:
    """Result file shaped like the study_results eval suite output, one shot."""
    n_ts = len(times)

    def _var(arr):
        return (("shot", "time_idx", RADIAL_DIM), np.broadcast_to(arr, (1, n_ts, len(RHO_GRID))).copy())

    return xr.Dataset(
        data_vars={
            "n_e_1e20_pred": _var(ne_pred),
            "n_e_1e20_targ": _var(ne_targ),
            "t_e_keV_pred": _var(te_pred),
            "t_e_keV_targ": _var(te_targ),
        },
        coords={
            "shot": [shot],
            "time_idx": np.arange(n_ts),
            RADIAL_DIM: RHO_GRID,
            "time": (("shot", "time_idx"), times[None, :]),
            "ds_source": (("shot",), np.array(["devA"], dtype=object)),
        },
    )


def _make_eval_ds(
    times: np.ndarray,
    ip: np.ndarray,
    p_nbi: np.ndarray,
    ne_err=0.0,
    te_err=0.0,
    ne_grad=None,
    te_grad=None,
    ne_grad_err=0.0,
    te_grad_err=0.0,
    shot: int = 1,
) -> xr.Dataset:
    n_ts = len(times)
    n_rho = len(RHO_GRID)

    def _profile_var(arr):
        return (("shot", "time_idx", RADIAL_DIM), np.broadcast_to(arr, (1, n_ts, n_rho)).copy())

    def _scalar_var(arr):
        return (("shot", "time_idx"), np.asarray(arr, dtype=float)[None, :])

    return xr.Dataset(
        data_vars={
            "n_e_1e20_error": _profile_var(ne_err),
            "t_e_keV_error": _profile_var(te_err),
            "n_e_1e20_gradient": _profile_var(ne_grad if ne_grad is not None else 0.0),
            "t_e_keV_gradient": _profile_var(te_grad if te_grad is not None else 0.0),
            "n_e_1e20_gradient_error": _profile_var(ne_grad_err),
            "t_e_keV_gradient_error": _profile_var(te_grad_err),
            "ip_MA": _scalar_var(ip),
            "power_additional_MW": _scalar_var(p_nbi),
        },
        coords={
            "shot": [shot],
            "time_idx": np.arange(n_ts),
            RADIAL_DIM: RHO_GRID,
            "time": (("shot", "time_idx"), times[None, :]),
        },
    )


# Linear profiles: finite-difference gradients are exact, so the GP gradient
# targets can be set to the analytic slope and a perfect prediction scores 0
NE_TARG = 2.0 - 1.0 * RHO_GRID  # peak 2.0, slope -1
TE_TARG = 4.0 - 2.0 * RHO_GRID  # peak 4.0, slope -2

RESULT_TIMES = np.array([0.1, 0.2, 0.3, 0.4])
EVAL_TIMES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
# Flattop threshold 0.9 * p95: eval indices 2..4 are flattop, 0..1 rampup, 5 rampdown
EVAL_IP = np.array([0.2, 0.5, 1.0, 1.0, 1.0, 0.3])
EVAL_P_NBI = np.array([0.0, 0.0, 0.05, 0.5, 0.0, 0.0])

LOSS_CONFIG = {
    "gradient_weight": 0.1,
    "chi_sigma_floors": {"devA": {var: SIGMA_FLOOR for error_vars in CHI_ERROR_VARS.values() for var in error_vars}},
}


@pytest.fixture()
def patched_eval(monkeypatch):
    def _patch(eval_ds):
        monkeypatch.setattr(study_metrics, "load_eval_dataset", lambda device: eval_ds)

    return _patch


class TestComputeCaseTimesliceMetrics:
    def test_perfect_prediction_scores_zero(self, patched_eval):
        result_ds = _make_result_ds(RESULT_TIMES, NE_TARG, NE_TARG, TE_TARG, TE_TARG)
        eval_ds = _make_eval_ds(EVAL_TIMES, EVAL_IP, EVAL_P_NBI, ne_grad=-1.0, te_grad=-2.0)
        patched_eval(eval_ds)

        ts = compute_case_timeslice_metrics(result_ds, LOSS_CONFIG)
        assert len(ts) == 4
        np.testing.assert_allclose(ts.metric_value, 0.0, atol=1e-12)
        np.testing.assert_allclose(ts.metric_grad, 0.0, atol=1e-9)
        np.testing.assert_allclose(ts.metric_combined, 0.0, atol=1e-9)

    def test_time_join_and_stage_labels(self, patched_eval):
        result_ds = _make_result_ds(RESULT_TIMES, NE_TARG, NE_TARG, TE_TARG, TE_TARG)
        eval_ds = _make_eval_ds(EVAL_TIMES, EVAL_IP, EVAL_P_NBI, ne_grad=-1.0, te_grad=-2.0)
        patched_eval(eval_ds)

        ts = compute_case_timeslice_metrics(result_ds, LOSS_CONFIG)
        # Result times 0.1..0.4 match eval indices 1..4
        np.testing.assert_array_equal(ts.eval_time_idx, [1, 2, 3, 4])
        np.testing.assert_array_equal(ts.stage, ["rampup", "flattop", "flattop", "flattop"])
        np.testing.assert_array_equal(ts.aux_heated, [False, False, True, False])
        # Flattop subdivision via the stage masks
        np.testing.assert_array_equal(ts.stage_mask("flattop_ohmic"), [False, True, False, True])
        np.testing.assert_array_equal(ts.stage_mask("flattop_aux"), [False, False, True, False])
        np.testing.assert_array_equal(ts.stage_mask("all"), [True, True, True, True])

    def test_value_chi_counts_offset_in_error_bars(self, patched_eval):
        # Constant offsets of half the ne error bar and a quarter of the Te one,
        # chi integrates to that fraction over rho in [0, 1]
        result_ds = _make_result_ds(RESULT_TIMES, NE_TARG + 0.1, NE_TARG, TE_TARG + 0.1, TE_TARG)
        eval_ds = _make_eval_ds(EVAL_TIMES, EVAL_IP, EVAL_P_NBI, ne_err=0.2, te_err=0.4, ne_grad=-1.0, te_grad=-2.0)
        patched_eval(eval_ds)

        ts = compute_case_timeslice_metrics(result_ds, LOSS_CONFIG)
        np.testing.assert_allclose(ts.metric_value, 0.5 + 0.25, rtol=1e-10)
        # A constant offset leaves the gradients untouched
        np.testing.assert_allclose(ts.metric_grad, 0.0, atol=1e-9)
        np.testing.assert_allclose(ts.metric_combined, ts.metric_value + LOSS_CONFIG["gradient_weight"] * ts.metric_grad, rtol=1e-12)

    def test_zero_error_bars_count_as_the_floor(self, patched_eval):
        # Zero-width error bars fall back to the device floor on the peak-normalized residual
        result_ds = _make_result_ds(RESULT_TIMES, NE_TARG + 0.1, NE_TARG, TE_TARG + 0.1, TE_TARG)
        eval_ds = _make_eval_ds(EVAL_TIMES, EVAL_IP, EVAL_P_NBI, ne_grad=-1.0, te_grad=-2.0)
        patched_eval(eval_ds)

        ts = compute_case_timeslice_metrics(result_ds, LOSS_CONFIG)
        np.testing.assert_allclose(ts.metric_value, (0.1 / 2.0 + 0.1 / 4.0) / SIGMA_FLOOR, rtol=1e-10)

    def test_gradient_error_beyond_rho_max_is_masked(self, patched_eval):
        # GP gradient targets disagree with the (perfect-value) prediction only
        # beyond GRAD_RHO_MAX, where the mask must zero the contribution
        rho_mid = 0.5 * (RHO_GRID[:-1] + RHO_GRID[1:])
        ne_grad = np.full(len(RHO_GRID), -1.0)
        te_grad = np.full(len(RHO_GRID), -2.0)
        # Corrupt the gradient targets only where every midpoint average of
        # adjacent points is beyond the mask
        edge = RHO_GRID > 0.97
        assert rho_mid[np.flatnonzero(edge)[0] - 1] >= GRAD_RHO_MAX
        ne_grad[edge] = 50.0
        te_grad[edge] = 50.0

        result_ds = _make_result_ds(RESULT_TIMES, NE_TARG, NE_TARG, TE_TARG, TE_TARG)
        eval_ds = _make_eval_ds(EVAL_TIMES, EVAL_IP, EVAL_P_NBI, ne_grad=ne_grad, te_grad=te_grad)
        patched_eval(eval_ds)

        ts = compute_case_timeslice_metrics(result_ds, LOSS_CONFIG)
        np.testing.assert_allclose(ts.metric_grad, 0.0, atol=1e-9)

    def test_combined_uses_gradient_weight(self, patched_eval):
        # Wrong gradient targets everywhere plus a value offset: combined must
        # be exactly value + gradient_weight * grad
        result_ds = _make_result_ds(RESULT_TIMES, NE_TARG + 0.1, NE_TARG, TE_TARG, TE_TARG)
        eval_ds = _make_eval_ds(EVAL_TIMES, EVAL_IP, EVAL_P_NBI, ne_grad=0.0, te_grad=-2.0)
        patched_eval(eval_ds)

        ts = compute_case_timeslice_metrics(result_ds, LOSS_CONFIG)
        assert (ts.metric_value > 0).all()
        assert (ts.metric_grad > 0).all()
        np.testing.assert_allclose(
            ts.metric_combined,
            ts.metric_value + LOSS_CONFIG["gradient_weight"] * ts.metric_grad,
            rtol=1e-12,
        )

    def test_nan_padded_timeslices_are_dropped(self, patched_eval):
        times = RESULT_TIMES.copy()
        times[-1] = np.nan
        result_ds = _make_result_ds(times, NE_TARG, NE_TARG, TE_TARG, TE_TARG)
        eval_ds = _make_eval_ds(EVAL_TIMES, EVAL_IP, EVAL_P_NBI, ne_grad=-1.0, te_grad=-2.0)
        patched_eval(eval_ds)

        ts = compute_case_timeslice_metrics(result_ds, LOSS_CONFIG)
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

    def make_train_config(self, case):
        return SimpleNamespace(loss_config=dict(LOSS_CONFIG))


class TestCaseMetricsCache:
    def _stub_with_result(self, tmp_path: Path, case: str = "case.test") -> _StubStudy:
        study = _StubStudy(tmp_path)
        result_ds = _make_result_ds(RESULT_TIMES, NE_TARG + 0.1, NE_TARG, TE_TARG + 0.1, TE_TARG)
        result_path = study.result_path(case)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_ds.to_netcdf(result_path)
        return study

    def test_missing_result_returns_empty_without_caching(self, tmp_path):
        study = _StubStudy(tmp_path)
        case_ds = compute_and_save_case_metrics(study, "case.test")
        assert not case_ds.data_vars
        assert not case_metrics_path(study, "case.test").exists()

    def test_computes_aggregates_and_caches(self, tmp_path, patched_eval):
        patched_eval(_make_eval_ds(EVAL_TIMES, EVAL_IP, EVAL_P_NBI, ne_grad=-1.0, te_grad=-2.0))
        study = self._stub_with_result(tmp_path)

        case_ds = compute_and_save_case_metrics(study, "case.test")
        assert case_metrics_path(study, "case.test").exists()
        assert list(case_ds["stage"].values) == list(STAGE_AGG_NAMES)
        expected_value = (0.1 / 2.0 + 0.1 / 4.0) / SIGMA_FLOOR
        np.testing.assert_allclose(case_ds["value_mean"].sel(stage="all").item(), expected_value, rtol=1e-10)
        assert case_ds["value_count"].sel(stage="all").item() == 4
        # From EVAL_IP / EVAL_P_NBI: rampup at 0.1s, flattop at 0.2-0.4s with one aux slice
        assert case_ds["value_count"].sel(stage="rampup").item() == 1
        assert case_ds["value_count"].sel(stage="flattop").item() == 3
        assert case_ds["value_count"].sel(stage="flattop_aux").item() == 1
        assert case_ds["value_count"].sel(stage="rampdown").item() == 0

    def test_second_call_uses_cache(self, tmp_path, patched_eval, monkeypatch):
        patched_eval(_make_eval_ds(EVAL_TIMES, EVAL_IP, EVAL_P_NBI, ne_grad=-1.0, te_grad=-2.0))
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
        patched_eval(_make_eval_ds(EVAL_TIMES, EVAL_IP, EVAL_P_NBI, ne_grad=-1.0, te_grad=-2.0))
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
