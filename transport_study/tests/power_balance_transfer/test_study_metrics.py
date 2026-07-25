"""Unit tests for the power balance stage-resolved metrics, best/worst shot
PDFs, and per-axis comparison tables, all on synthetic data (the device
dataset join is monkeypatched, so no config or real datasets are needed)."""

import re
from itertools import product

import numpy as np
import pytest
import xarray as xr

from transport_study.orchestration.organize_data import INPUT_POWER_SIGNALS
from transport_study.power_balance_transfer import study_metrics
from transport_study.power_balance_transfer.case_reports import best_worst_pdf
from transport_study.power_balance_transfer.study_metrics import (
    STAGE_AGG_NAMES,
    aggregate_case_metrics,
    compute_case_timeslice_metrics,
    shot_time_averaged_errors,
)
from transport_study.power_balance_transfer.tables import write_comparison_tables

N_TS = 100
SHOTS = [101, 102]


@pytest.fixture()
def device_ds() -> xr.Dataset:
    """Synthetic device dataset: trapezoid Ip with NBI heating mid-flattop."""
    ip = np.concatenate([np.linspace(0.0, 1.0, 20), np.full(60, 1.0), np.linspace(1.0, 0.0, 20)])
    p_nbi = np.zeros(N_TS)
    p_nbi[40:60] = 5.0
    time = np.arange(N_TS) * 1e-3

    data_vars = {
        "Ip_MA": (("shot", "time_idx"), np.stack([ip, ip])),
        "P_NBI_MW": (("shot", "time_idx"), np.stack([p_nbi, p_nbi])),
    }
    for sig in INPUT_POWER_SIGNALS:
        if sig not in data_vars:
            data_vars[sig] = (("shot", "time_idx"), np.zeros((2, N_TS)))
    return xr.Dataset(
        data_vars=data_vars,
        coords={"shot": SHOTS, "time": (("shot", "time_idx"), np.stack([time, time]))},
    )


@pytest.fixture()
def result_ds() -> xr.Dataset:
    """Synthetic case result file: constant abs error 0.1 on shot 101,
    0.3 on shot 102, with the last 20 timeslices of shot 102 NaN-padded."""
    time = np.arange(N_TS) * 1e-3
    targ = np.full((N_TS, 2), 1.0)
    err = np.stack([np.full(N_TS, 0.1), np.full(N_TS, 0.3)], axis=1)
    err[80:, 1] = np.nan
    targ[80:, 1] = np.nan
    pred = targ + err

    return xr.Dataset(
        data_vars={
            "Wtot_MJ_targ": (("time_idx", "shot"), targ),
            "Wtot_MJ_pred": (("time_idx", "shot"), pred),
            "error_abs_ts": (("time_idx", "shot"), err),
            "error_rel_ts": (("time_idx", "shot"), err / 1.1),
        },
        coords={
            "shot": SHOTS,
            "time": (("time_idx", "shot"), np.stack([time, time], axis=1)),
            "ds_source": ("shot", ["testdev", "testdev"]),
        },
    )


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

    # Errors ride along unchanged
    assert np.allclose(ts_metrics.err_abs[shot_101], 0.1)
    assert np.allclose(ts_metrics.err_abs[~shot_101], 0.3)


def test_aggregate_case_metrics(ts_metrics):
    case_ds = aggregate_case_metrics(ts_metrics)

    assert list(case_ds["stage"].values) == list(STAGE_AGG_NAMES)
    counts = case_ds["abs_count"].values
    by_stage = dict(zip(STAGE_AGG_NAMES, counts, strict=True))
    assert by_stage["all"] == len(ts_metrics)
    assert by_stage["flattop_ohmic"] + by_stage["flattop_aux"] == by_stage["flattop"]
    assert by_stage["rampup"] + by_stage["flattop"] + by_stage["rampdown"] == by_stage["all"]

    # Pooled mean over both shots sits between the per-shot constants
    all_mean = float(case_ds["abs_mean"].sel(stage="all"))
    assert 0.1 < all_mean < 0.3


def test_shot_time_averaged_errors(ts_metrics):
    shots, avg_abs, avg_rel, n_ts = shot_time_averaged_errors(ts_metrics)

    assert list(shots) == SHOTS
    # Time-averaged errors are duration-free: the constants come back exactly
    assert np.allclose(avg_abs, [0.1, 0.3])
    assert np.allclose(avg_rel, np.array([0.1, 0.3]) / 1.1)
    assert list(n_ts) == [N_TS, 80]


def test_best_worst_pdf(tmp_path, result_ds, ts_metrics):
    pdf_path = tmp_path / "case_reports" / "best_worst_shots.pdf"
    best_worst_pdf(result_ds, ts_metrics, pdf_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


@pytest.fixture()
def collected_datasets() -> tuple[xr.Dataset, xr.Dataset]:
    """Synthetic collected_results.nc / collected_metrics.nc pair sharing
    case_idx, with a p_oh submodule case and one case missing from metrics."""
    model_types = ["sciml", "transformer", "p_oh"]
    normalizations = ["raw", "coral"]
    shots_options = [0, 3, -1]

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
    for err in ("err_abs", "err_rel"):
        for domain in ("shot", "ts"):
            for stat in ("mean", "std", "med", "p25", "p75", "min", "max"):
                results[f"{err}_{domain}_{stat}"] = ("case_idx", rng.uniform(0.01, 1.0, n))

    # Metrics cover every case except the last (simulates a diverged case)
    n_metrics = n - 1
    metrics = xr.Dataset(coords={"case_idx": np.arange(n_metrics), "stage": list(STAGE_AGG_NAMES)})
    for metric in ("abs", "rel"):
        for stat in ("mean", "std", "med"):
            metrics[f"{metric}_{stat}"] = (
                ("case_idx", "stage"),
                rng.uniform(0.01, 1.0, (n_metrics, len(STAGE_AGG_NAMES))),
            )
        metrics[f"{metric}_count"] = (("case_idx", "stage"), np.full((n_metrics, len(STAGE_AGG_NAMES)), 10))
    return results, metrics


def test_write_comparison_tables(tmp_path, collected_datasets):
    results, metrics = collected_datasets
    write_comparison_tables(results, metrics, tmp_path)

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
    assert "sciml" in model_table
    assert "transformer" in model_table
    assert "rel err (time avg)" in model_table
    assert "flattop ohmic" in model_table

    # Axes with a single value (training_data, domain_adaptation) produce no tables
    assert not (tables_dir / "training_data").exists()
    assert not (tables_dir / "domain_adaptation").exists()


def test_write_comparison_tables_empty(tmp_path):
    write_comparison_tables(xr.Dataset(), xr.Dataset(), tmp_path)
    assert not (tmp_path / "tables").exists()
