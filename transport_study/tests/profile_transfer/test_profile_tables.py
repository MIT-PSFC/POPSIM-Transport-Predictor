"""Tests for the profile transfer per-axis comparison tables."""

import numpy as np
import pandas as pd
import xarray as xr

from transport_study.orchestration.stages import STAGE_AGG_NAMES
from transport_study.profile_transfer.tables import (
    CASE_FIELD_ORDER,
    case_stats_frame,
    write_comparison_tables,
)

# Two cases differing only in model_type, three shots each. Every other case
# field is held fixed, so model_type is the only axis with a table to write.
FIXED_CASE_FIELDS = {
    "training_data": "cmod",
    "data_normalization": "physics",
    "domain_adaptation": "none",
    "freeze_shapes": True,
    "geometry_builder": "circular",
    "num_target_shots": 0,
}
MODEL_TYPES = ["shape-init-pca", "mlp"]
SHOTS = [101, 102, 103]


def _results_ds() -> xr.Dataset:
    n_records = len(MODEL_TYPES) * len(SHOTS)
    coords = {name: ("record", np.array([value] * n_records)) for name, value in FIXED_CASE_FIELDS.items()}
    coords["case_idx"] = ("record", np.repeat(np.arange(len(MODEL_TYPES)), len(SHOTS)))
    coords["model_type"] = ("record", np.repeat(MODEL_TYPES, len(SHOTS)))
    coords["shot"] = ("record", np.tile(SHOTS, len(MODEL_TYPES)))
    coords["ds_source"] = ("record", np.array(["cmod"] * n_records))
    return xr.Dataset(
        data_vars={
            "err_abs_shot": ("record", np.arange(1.0, n_records + 1.0)),
            "err_rel_shot": ("record", np.arange(1.0, n_records + 1.0) / 10.0),
        },
        coords=coords,
    )


def _metrics_ds() -> xr.Dataset:
    n_stages = len(STAGE_AGG_NAMES)
    data_vars = {}
    for i, metric in enumerate(("value", "grad", "combined")):
        values = np.arange(len(MODEL_TYPES) * n_stages, dtype=float).reshape(len(MODEL_TYPES), n_stages) + 10 * i
        data_vars[f"{metric}_mean"] = (("case_idx", "stage"), values)
    return xr.Dataset(data_vars=data_vars, coords={"case_idx": np.arange(len(MODEL_TYPES)), "stage": list(STAGE_AGG_NAMES)})


def test_case_stats_frame_covers_every_case_field():
    """Every axis in CASE_FIELD_ORDER must survive into the frame, or the
    grouping in write_case_comparison_tables raises on the missing column."""
    df = case_stats_frame(_results_ds(), _metrics_ds())
    assert set(CASE_FIELD_ORDER) <= set(df.columns)


def test_case_stats_frame_medians_and_stage_merge():
    df = case_stats_frame(_results_ds(), _metrics_ds()).set_index("case_idx")
    assert len(df) == len(MODEL_TYPES)
    assert df.loc[0, "err_abs_shot_med"] == 2.0
    assert df.loc[1, "err_rel_shot_med"] == 0.5
    # Stage "all" is index 0 in STAGE_AGG_NAMES, combined metric offset is 20
    assert df.loc[0, "combined_mean_all"] == 20.0
    assert df.loc[1, "value_mean_all"] == float(len(STAGE_AGG_NAMES))


def test_case_stats_frame_without_metrics():
    df = case_stats_frame(_results_ds(), xr.Dataset())
    assert np.isnan(df["combined_mean_all"]).all()
    assert df["err_abs_shot_med"].notna().all()


def test_write_comparison_tables(tmp_path):
    write_comparison_tables(_results_ds(), _metrics_ds(), tmp_path)

    csv_path = tmp_path / "tables" / "case_stats.csv"
    assert csv_path.exists()
    assert len(pd.read_csv(csv_path)) == len(MODEL_TYPES)

    # The two cases differ only along model_type, so exactly that axis gets a table
    table_path = tmp_path / "tables" / "model_type" / "td_cmod.norm_physics.da_none.freeze_True.geom_circular.targ_0.md"
    content = table_path.read_text()
    for model_type in MODEL_TYPES:
        assert f"| {model_type} |" in content
    for axis in set(CASE_FIELD_ORDER) - {"model_type"}:
        assert not (tmp_path / "tables" / axis).exists()


def test_empty_results_skip(tmp_path):
    write_comparison_tables(xr.Dataset(), xr.Dataset(), tmp_path)
    assert not (tmp_path / "tables").exists()
