"""Tests for the power balance per-axis comparison tables."""

import numpy as np
import pandas as pd
import xarray as xr

from transport_study.orchestration.stages import STAGE_AGG_NAMES
from transport_study.orchestration.tables import (
    summary_case_stats_frame,
    write_summary_comparison_tables,
)
from transport_study.power_balance_transfer.tables import SPEC


def _results_ds() -> xr.Dataset:
    # Three cases: two main models plus a p_oh submodule prereq that must be
    # excluded. freeze_submodules is a scalar coord like in real collected
    # files where every case shares the single configured value.
    coords = {
        "case_idx": [0, 1, 2],
        "model_type": ("case_idx", np.array(["sciml-taue-nn", "sciml-taue-scalinglaw", "p_oh"])),
        "training_data": ("case_idx", np.array(["cmod"] * 3)),
        "data_normalization": ("case_idx", np.array(["physics"] * 3)),
        "domain_adaptation": ("case_idx", np.array(["none"] * 3)),
        "num_target_shots": ("case_idx", np.array([0, 0, 0])),
        "freeze_submodules": "none",
    }
    return xr.Dataset(
        data_vars={
            "err_rel_shot_med": ("case_idx", np.array([0.1, 0.2, 0.9])),
            "err_abs_shot_med": ("case_idx", np.array([1.0, 2.0, 9.0])),
        },
        coords=coords,
    )


def _metrics_ds() -> xr.Dataset:
    n_stages = len(STAGE_AGG_NAMES)
    data_vars = {}
    for i, metric in enumerate(("abs", "rel")):
        values = np.arange(3 * n_stages, dtype=float).reshape(3, n_stages) + 10 * i
        data_vars[f"{metric}_mean"] = (("case_idx", "stage"), values)
    return xr.Dataset(data_vars=data_vars, coords={"case_idx": [0, 1, 2], "stage": list(STAGE_AGG_NAMES)})


def test_case_stats_frame_excludes_submodules_and_merges_metrics():
    df = summary_case_stats_frame(_results_ds(), _metrics_ds(), SPEC)
    assert set(df["model_type"]) == {"sciml-taue-nn", "sciml-taue-scalinglaw"}
    # The scalar freeze_submodules coord is broadcast back to a column
    assert (df["freeze_submodules"] == "none").all()
    df = df.set_index("case_idx")
    assert df.loc[0, "abs_mean_all"] == 0.0
    assert df.loc[1, "rel_mean_all"] == float(len(STAGE_AGG_NAMES)) + 10


def test_write_comparison_tables(tmp_path):
    write_summary_comparison_tables(_results_ds(), _metrics_ds(), SPEC, tmp_path)

    csv_path = tmp_path / "tables" / "case_stats.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert "p_oh" not in set(df["model_type"])

    table_path = tmp_path / "tables" / "model_type" / "td_cmod.norm_physics.da_none.freeze_none.targ_0.md"
    content = table_path.read_text()
    assert "| sciml-taue-nn |" in content
    assert "| sciml-taue-scalinglaw |" in content


def test_missing_metrics_render_as_dash(tmp_path):
    write_summary_comparison_tables(_results_ds(), xr.Dataset(), SPEC, tmp_path)
    table_path = tmp_path / "tables" / "model_type" / "td_cmod.norm_physics.da_none.freeze_none.targ_0.md"
    content = table_path.read_text()
    assert "| sciml-taue-nn | - |" in content


def test_empty_results_skip(tmp_path):
    write_summary_comparison_tables(xr.Dataset(), xr.Dataset(), SPEC, tmp_path)
    assert not (tmp_path / "tables").exists()
