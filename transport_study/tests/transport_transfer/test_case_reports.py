"""Transport case reports and metrics on the result files of every case kind the study writes."""

from types import SimpleNamespace

import numpy as np
import xarray as xr

from transport_study import RADIAL_DIM
from transport_study.orchestration.case_metrics import aggregate_case_metrics
from transport_study.orchestration.case_reports import generate_case_report
from transport_study.orchestration.topk_results import aggregate_topk_results
from transport_study.power_balance_transfer import study_metrics
from transport_study.transport_transfer.study_metrics import (
    METRIC_NAMES,
    compute_case_timeslice_metrics,
)
from transport_study.transport_transfer.transport_transfer_study import TransportStudy

N_TS = 40
SHOTS = [101, 102]
RHO = np.linspace(0.0, 1.0, 5)
STALE_TIME_IDX = 10
DIVERGED_TIME_IDX = 20


def _device_ds() -> xr.Dataset:
    ip = np.concatenate([np.linspace(0.0, 1.0, 10), np.full(20, 1.0), np.linspace(1.0, 0.0, 10)])
    time = np.arange(N_TS) * 1e-3
    data_vars = {"power_additional_MW": (("shot", "time_idx"), np.zeros((len(SHOTS), N_TS)))}
    data_vars["ip_MA"] = (("shot", "time_idx"), np.stack([ip, ip]))
    return xr.Dataset(data_vars, coords={"shot": SHOTS, "time": (("shot", "time_idx"), np.stack([time, time]))})


def _scalar_result_ds() -> xr.Dataset:
    """The result file of a power_balance / p_oh / p_rad prereq case, no profile variables."""
    time = np.arange(N_TS) * 1e-3
    targ = np.ones((N_TS, len(SHOTS)))
    err = np.stack([np.full(N_TS, 0.1), np.full(N_TS, 0.3)], axis=1)
    dims = ("time_idx", "shot")
    step_ds = xr.Dataset(
        data_vars={
            "power_ohm_MW_targ": (dims, targ),
            "power_ohm_MW_pred": (dims, targ + err),
            "error_abs_ts": (dims, err),
            "error_rel_ts": (dims, err / 1.1),
            "error_diverged_ts": (dims, np.zeros_like(err)),
        },
        coords={"shot": SHOTS, "time": (dims, np.stack([time, time], axis=1)), "ds_source": ("shot", ["testdev", "testdev"])},
    )
    return aggregate_topk_results({1: step_ds}, best_step=1)


def _profile_result_ds() -> xr.Dataset:
    """A top-level transport result: chi 2.0 on fresh timeslices, NaN at the stale one, shot 101 diverged at one timeslice."""
    time = np.arange(N_TS) * 1e-3
    dims = ("time_idx", "shot")
    chi = np.full((N_TS, len(SHOTS)), 2.0)
    chi[STALE_TIME_IDX, :] = np.nan
    chi[DIVERGED_TIME_IDX, 0] = np.nan
    diverged = np.zeros((N_TS, len(SHOTS)))
    diverged[DIVERGED_TIME_IDX, 0] = 1.0
    profile = np.broadcast_to(1.0 - 0.5 * RHO, (N_TS, len(SHOTS), RHO.size)).copy()
    step_ds = xr.Dataset(
        data_vars={
            **{f"{signal}_{kind}": ((*dims, RADIAL_DIM), profile) for signal in ("n_e_1e20", "t_e_keV") for kind in ("targ", "pred")},
            "error_chi_ts": (dims, chi),
            "error_chi_value_ts": (dims, 0.5 * chi),
            "error_chi_grad_ts": (dims, 15.0 * chi),
            "error_abs_ts": (dims, 0.1 * chi),
            "error_rel_ts": (dims, 0.2 * chi),
            "error_diverged_ts": (dims, diverged),
        },
        coords={
            "shot": SHOTS,
            RADIAL_DIM: RHO,
            "time": (dims, np.stack([time, time], axis=1)),
            "ds_source": ("shot", ["testdev", "testdev"]),
        },
    )
    return aggregate_topk_results({1: step_ds}, best_step=1)


def _report_study(result_path) -> SimpleNamespace:
    return SimpleNamespace(
        result_path=lambda case: result_path,
        is_borrowed=lambda case: False,
        ANALYSIS_METRICS_MODULE=TransportStudy.ANALYSIS_METRICS_MODULE,
        ANALYSIS_REPORTS_MODULE=TransportStudy.ANALYSIS_REPORTS_MODULE,
    )


def test_profile_result_scores_chi_and_counts_divergence(monkeypatch):
    """Stale timeslices are NaN in chi but stay records, so the diverged fraction counts every real timeslice."""
    monkeypatch.setattr(study_metrics, "load_stage_dataset", lambda device: _device_ds())
    ts_metrics = compute_case_timeslice_metrics(_profile_result_ds())
    case_ds = aggregate_case_metrics(ts_metrics, METRIC_NAMES).sel(stage="all")

    assert len(ts_metrics) == N_TS * len(SHOTS)
    assert np.isclose(case_ds["combined_mean"], 2.0)
    assert case_ds["combined_count"] == N_TS * len(SHOTS) - len(SHOTS) - 1
    assert np.isclose(case_ds["rel_mean"], 0.4)
    assert np.isclose(case_ds["diverged_mean"], 1.0 / (N_TS * len(SHOTS)))


def test_scalar_prereq_result_scores_nan_chi(monkeypatch):
    """The scalar prereq cases carry no chi, which reads as NaN next to their abs / rel errors."""
    monkeypatch.setattr(study_metrics, "load_stage_dataset", lambda device: _device_ds())
    ts_metrics = compute_case_timeslice_metrics(_scalar_result_ds())

    assert np.isnan(ts_metrics.metric("combined")).all()
    assert np.allclose(ts_metrics.best("abs")[ts_metrics.shot == 101], 0.1)


def test_every_case_kind_gets_a_report(tmp_path, monkeypatch):
    """The profile results get the transport page ranked by chi,
    the scalar prereq cases the power balance page instead of failing on missing profile targets."""
    monkeypatch.setattr(study_metrics, "load_stage_dataset", lambda device: _device_ds())
    for case, result_ds in (("case.p_oh.td_cmod.freeze_True", _scalar_result_ds()), ("case.transformer.td_cmod", _profile_result_ds())):
        result_path = tmp_path / case / "result_data.nc"
        result_path.parent.mkdir(parents=True)
        result_ds.to_netcdf(result_path)

        generate_case_report(_report_study(result_path), case, tmp_path / "figures")

        pdf_path = tmp_path / "figures" / "case_reports" / case / "best_worst_shots.pdf"
        assert pdf_path.stat().st_size > 0
