"""Unit tests for the power balance stage-resolved metrics, best/worst shot
PDFs, and per-axis comparison tables, all on synthetic data (the device
dataset join is monkeypatched, so no config or real datasets are needed)."""

import re
from itertools import product

import numpy as np
import pytest
import xarray as xr

from transport_study.orchestration.case_metrics import aggregate_case_metrics
from transport_study.orchestration.stages import STAGE_AGG_NAMES
from transport_study.orchestration.tables import (
    TOPK_SUFFIXES,
    write_summary_comparison_tables,
)
from transport_study.orchestration.topk_results import aggregate_topk_results
from transport_study.power_balance_transfer import study_metrics
from transport_study.power_balance_transfer.case_reports import shot_pdf
from transport_study.power_balance_transfer.study_metrics import (
    METRIC_NAMES,
    compute_case_timeslice_metrics,
    shot_time_averaged,
)
from transport_study.power_balance_transfer.tables import SPEC

N_TS = 100
SHOTS = [101, 102]
# The best checkpoint's errors are the constants below, the other retained checkpoint's are twice them
BEST_EPOCH = 7
OTHER_EPOCH = 3


@pytest.fixture()
def device_ds() -> xr.Dataset:
    """Synthetic device dataset: trapezoid Ip with NBI heating mid-flattop."""
    ip = np.concatenate([np.linspace(0.0, 1.0, 20), np.full(60, 1.0), np.linspace(1.0, 0.0, 20)])
    p_nbi = np.zeros(N_TS)
    p_nbi[40:60] = 5.0
    time = np.arange(N_TS) * 1e-3

    data_vars = {
        "ip_MA": (("shot", "time_idx"), np.stack([ip, ip])),
        "power_additional_MW": (("shot", "time_idx"), np.stack([p_nbi, p_nbi])),
    }
    return xr.Dataset(
        data_vars=data_vars,
        coords={"shot": SHOTS, "time": (("shot", "time_idx"), np.stack([time, time]))},
    )


def _step_result(error_scale: float, n_diverged_101: int = 0) -> xr.Dataset:
    """One checkpoint's study_results: constant abs error 0.1 on shot 101 and 0.3 on shot 102 times error_scale,
    the last 20 timeslices of shot 102 NaN-padded, and the last n_diverged_101 of shot 101 diverged."""
    time = np.arange(N_TS) * 1e-3
    targ = np.full((N_TS, 2), 1.0)
    err = np.stack([np.full(N_TS, 0.1), np.full(N_TS, 0.3)], axis=1) * error_scale
    err[80:, 1] = np.nan
    targ[80:, 1] = np.nan
    pred = targ + err
    diverged = np.where(np.isfinite(targ), 0.0, np.nan)
    if n_diverged_101:
        pred[N_TS - n_diverged_101 :, 0] = np.nan
        err[N_TS - n_diverged_101 :, 0] = np.nan
        diverged[N_TS - n_diverged_101 :, 0] = 1.0

    return xr.Dataset(
        data_vars={
            "energy_mhd_MJ_targ": (("time_idx", "shot"), targ),
            "energy_mhd_MJ_pred": (("time_idx", "shot"), pred),
            "error_abs_ts": (("time_idx", "shot"), err),
            "error_rel_ts": (("time_idx", "shot"), err / 1.1),
            "error_diverged_ts": (("time_idx", "shot"), diverged),
        },
        coords={
            "shot": SHOTS,
            "time": (("time_idx", "shot"), np.stack([time, time], axis=1)),
            "ds_source": ("shot", ["testdev", "testdev"]),
        },
    )


@pytest.fixture()
def result_ds() -> xr.Dataset:
    """Synthetic case result file of two retained checkpoints, the best one with the constant errors."""
    return aggregate_topk_results({BEST_EPOCH: _step_result(1.0), OTHER_EPOCH: _step_result(2.0)}, BEST_EPOCH)


@pytest.fixture()
def ts_metrics(monkeypatch, device_ds, result_ds):
    monkeypatch.setattr(study_metrics, "load_stage_dataset", lambda device: device_ds)
    return compute_case_timeslice_metrics(result_ds)


def test_compute_case_timeslice_metrics(ts_metrics):
    # Shot 101 keeps all timeslices, shot 102 loses the 20 NaN-padded ones
    assert len(ts_metrics) == N_TS + 80
    assert (ts_metrics.ds_source == "testdev").all()
    assert set(ts_metrics.stage) == {"rampup", "flattop", "rampdown"}

    # Aux heating flag follows the NBI trace
    shot_101 = ts_metrics.shot == 101
    assert ts_metrics.aux_heated[shot_101][40:60].all()
    assert not ts_metrics.aux_heated[shot_101][:40].any()

    # Errors ride along unchanged, one column per retained checkpoint
    assert np.allclose(ts_metrics.best("abs")[shot_101], 0.1)
    assert np.allclose(ts_metrics.best("abs")[~shot_101], 0.3)
    assert np.allclose(ts_metrics.metric("abs")[shot_101], [0.2, 0.1])


def test_first_real_timeslice_after_leading_padding_is_kept(monkeypatch, device_ds, result_ds):
    """A shot whose result starts with NaN times keeps its first real timeslice.

    Same rule as real_timeslice_mask: a timeslice after a NaN time is real, only clock-stalled repeats are padding.
    """
    monkeypatch.setattr(study_metrics, "load_stage_dataset", lambda device: device_ds)
    n_leading_nan = 10
    time = result_ds["time"].values.copy()
    time[:n_leading_nan, 1] = np.nan
    result_ds = result_ds.assign_coords(time=(("time_idx", "shot"), time))

    ts_metrics = compute_case_timeslice_metrics(result_ds)

    times_102 = ts_metrics.time[ts_metrics.shot == 102]
    assert times_102.min() == pytest.approx(time[n_leading_nan, 1])
    assert len(times_102) == 80 - n_leading_nan


def test_aggregate_case_metrics(ts_metrics):
    case_ds = aggregate_case_metrics(ts_metrics, METRIC_NAMES)

    assert list(case_ds["stage"].values) == list(STAGE_AGG_NAMES)
    counts = case_ds["abs_count"].values
    by_stage = dict(zip(STAGE_AGG_NAMES, counts, strict=True))
    assert by_stage["all"] == len(ts_metrics)
    assert by_stage["flattop_ohmic"] + by_stage["flattop_aux"] == by_stage["flattop"]
    assert by_stage["rampup"] + by_stage["flattop"] + by_stage["rampdown"] == by_stage["all"]

    # Pooled mean over both shots sits between the per-shot constants
    best_mean = float(case_ds["abs_mean_best"].sel(stage="all"))
    assert 0.1 < best_mean < 0.3

    # Each checkpoint scores the case on its own, the other checkpoint's errors are twice the best one's
    other_mean = 2.0 * best_mean
    assert np.isclose(case_ds["abs_mean"].sel(stage="all"), 0.5 * (best_mean + other_mean))
    assert np.isclose(case_ds["abs_mean_ckpt_std"].sel(stage="all"), 0.5 * (other_mean - best_mean))
    assert np.isclose(case_ds["diverged_mean"].sel(stage="all"), 0.0)


def test_diverged_timeslices_stay_records_and_count_as_the_fraction(monkeypatch, device_ds):
    """A rollout that diverges keeps its timeslices as records:
    NaN in the errors, which the error means skip, and 1 in the diverged flag, whose stage mean is the fraction."""
    monkeypatch.setattr(study_metrics, "load_stage_dataset", lambda device: device_ds)
    n_diverged = 15
    result_ds = aggregate_topk_results({BEST_EPOCH: _step_result(1.0, n_diverged), OTHER_EPOCH: _step_result(2.0)}, BEST_EPOCH)

    ts_metrics = compute_case_timeslice_metrics(result_ds)
    case_ds = aggregate_case_metrics(ts_metrics, METRIC_NAMES)

    assert len(ts_metrics) == N_TS + 80
    assert np.isclose(case_ds["diverged_mean_best"].sel(stage="all"), n_diverged / (N_TS + 80))
    assert np.isclose(case_ds["diverged_mean"].sel(stage="all"), 0.5 * n_diverged / (N_TS + 80))
    assert case_ds["abs_count_best"].sel(stage="all") == N_TS + 80 - n_diverged


def test_shot_time_averaged(ts_metrics):
    shots, avg_abs, n_ts = shot_time_averaged(ts_metrics, "abs")
    _, avg_rel, _ = shot_time_averaged(ts_metrics, "rel")

    assert list(shots) == SHOTS
    # Time-averaged errors are duration-free: the best checkpoint's constants come back exactly
    assert np.allclose(avg_abs, [0.1, 0.3])
    assert np.allclose(avg_rel, np.array([0.1, 0.3]) / 1.1)
    assert list(n_ts) == [N_TS, 80]


def test_shot_pdf(tmp_path, result_ds, ts_metrics):
    pdf_path = tmp_path / "case_reports" / "best_worst_shots.pdf"
    shot_pdf(result_ds, ts_metrics, pdf_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


@pytest.fixture()
def collected_datasets() -> tuple[xr.Dataset, xr.Dataset]:
    """Synthetic collected_results.nc / collected_metrics.nc pair sharing
    case_idx, with a p_oh submodule case and one case missing from metrics."""
    model_types = ["sciml-taue-nn", "transformer", "p_oh"]
    normalizations = ["raw", "coral"]
    shots_options = [0, 3, 10]

    rows = [(mt, "cmod_tcv", dn, "addition", True, n) for mt, dn, n in product(model_types, normalizations, shots_options)]
    n = len(rows)
    rng = np.random.default_rng(0)

    results = xr.Dataset(coords={"case_idx": np.arange(n)})
    for coord_name, idx in zip(
        ("model_type", "training_data", "data_normalization", "domain_adaptation", "freeze_submodules", "num_target_shots"),
        range(6),
        strict=True,
    ):
        results = results.assign_coords({coord_name: ("case_idx", [row[idx] for row in rows])})
    # A single configured value is a scalar coord in real collected files
    results = results.assign_coords(target_shot_order="ascending", multiobjective=False)
    for err in ("err_abs", "err_rel"):
        for domain in ("shot", "ts"):
            for stat in ("mean", "std", "med", "p25", "p75", "min", "max"):
                results[f"{err}_{domain}_{stat}"] = ("case_idx", rng.uniform(0.01, 1.0, n))

    # Metrics cover every case except the last (simulates a diverged case)
    n_metrics = n - 1
    metrics = xr.Dataset(coords={"case_idx": np.arange(n_metrics), "stage": list(STAGE_AGG_NAMES)})
    for metric in METRIC_NAMES:
        for suffix in TOPK_SUFFIXES:
            metrics[f"{metric}_mean{suffix}"] = (
                ("case_idx", "stage"),
                rng.uniform(0.01, 1.0, (n_metrics, len(STAGE_AGG_NAMES))),
            )
    return results, metrics


def test_write_comparison_tables(tmp_path, collected_datasets):
    results, metrics = collected_datasets
    write_summary_comparison_tables(results, metrics, SPEC, tmp_path)

    tables_dir = tmp_path / "tables"
    assert (tables_dir / "case_stats.csv").exists()

    # One table family per axis with grouped members
    for axis in ("model_type", "data_normalization", "num_target_shots"):
        axis_tables = list((tables_dir / axis).glob("*.md"))
        assert axis_tables, f"No {axis} tables were generated"

    # Submodule prereq cases are excluded everywhere ("flattop_ohmic" contains
    # the substring p_oh, so match on word boundaries)
    for table in tables_dir.rglob("*"):
        if table.is_file():
            assert not re.search(r"\bp_oh\b", table.read_text()), f"p_oh leaked into {table}"

    # A model_type table lists both main models and every column
    model_table = next(iter((tables_dir / "model_type").glob("*.md"))).read_text()
    assert "sciml-taue-nn" in model_table
    assert "transformer" in model_table
    assert "rel err (time avg)" in model_table
    assert "flattop ohmic" in model_table
    assert "| ckpt std |" in model_table
    assert "| diverged |" in model_table

    # Axes with a single value (training_data, domain_adaptation) produce no tables
    assert not (tables_dir / "training_data").exists()
    assert not (tables_dir / "domain_adaptation").exists()


def test_write_comparison_tables_empty(tmp_path):
    write_summary_comparison_tables(xr.Dataset(), xr.Dataset(), SPEC, tmp_path)
    assert not (tmp_path / "tables").exists()
