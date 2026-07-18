"""Per-axis comparison tables for profile transfer study results.

For every case axis (model type, training dataset, domain adaptation, freeze
shapes, number of target shots) and every combination of the remaining axes,
one markdown table comparing the cases that differ only along that axis.
Columns combine the stage-resolved TIME-AVERAGED loss metrics from
collected_metrics.nc (per-timeslice means of the value / gradient / combined
validation loss components, free of the shot-duration confound in the
time-integrated per-shot errors) with the time-integrated per-shot error
medians from the long-form collected_results.nc.

A flat case_stats.csv with one row per case and every column is written next
to the tables for ad hoc analysis. The table and csv writing itself is the
shared orchestration.tables machinery; this module only declares the spec and
builds the per-case stats frame.
"""

from pathlib import Path

import pandas as pd
import xarray as xr
from loguru import logger

from transport_study.orchestration.tables import (
    ComparisonTableSpec,
    merge_stage_metrics,
    write_case_comparison_tables,
)

# Stage-resolved metric variables in collected_metrics.nc
# (see profile_transfer.study_metrics.METRIC_NAMES)
_STAGE_METRICS = ("value", "grad", "combined")

# Per-shot time-integrated errors kept from the long-form collected results,
# reduced to a per-case median
_SHOT_ERROR_VARS = ("err_abs_shot", "err_rel_shot")

CASE_FIELD_ORDER = (
    "model_type",
    "training_data",
    "data_normalization",
    "domain_adaptation",
    "freeze_shapes",
    "geometry_builder",
    "num_target_shots",
)

SPEC = ComparisonTableSpec(
    axis_names=CASE_FIELD_ORDER,
    case_field_order=CASE_FIELD_ORDER,
    # Filename tokens per grouping field, mirroring the case-string vocabulary
    field_tokens={
        "model_type": "{}",
        "training_data": "td_{}",
        "data_normalization": "norm_{}",
        "domain_adaptation": "da_{}",
        "freeze_shapes": "freeze_{}",
        "geometry_builder": "geom_{}",
        "num_target_shots": "targ_{}",
    },
    columns=(
        ("combined (time avg)", "combined_mean_all"),
        ("rampup", "combined_mean_rampup"),
        ("flattop", "combined_mean_flattop"),
        ("flattop ohmic", "combined_mean_flattop_ohmic"),
        ("flattop aux", "combined_mean_flattop_aux"),
        ("rampdown", "combined_mean_rampdown"),
        ("value (time avg)", "value_mean_all"),
        ("grad (time avg)", "grad_mean_all"),
        ("rel err (integral, med)", "err_rel_shot_med"),
        ("abs err (integral, med)", "err_abs_shot_med"),
    ),
)


def _case_stats_frame(results_ds: xr.Dataset, metrics_ds: xr.Dataset) -> pd.DataFrame:
    """One row per case: case-grid coords, per-shot integral error medians from
    the long-form collected results, and per-stage time-averaged value /
    gradient / combined metrics from the collected metrics (NaN for cases
    without valid metrics)."""
    long_df = results_ds[list(_SHOT_ERROR_VARS)].to_dataframe().reset_index()
    grouped = long_df.groupby("case_idx")
    medians = grouped[list(_SHOT_ERROR_VARS)].median()
    medians.columns = [f"{name}_med" for name in medians.columns]
    case_fields = grouped[list(CASE_FIELD_ORDER)].first()
    df = pd.concat([case_fields, medians], axis=1).reset_index()
    df = df[[*CASE_FIELD_ORDER, "case_idx", *medians.columns]]
    return merge_stage_metrics(df, metrics_ds, _STAGE_METRICS)


def write_comparison_tables(results_ds: xr.Dataset, metrics_ds: xr.Dataset, figure_dir: Path):
    """One markdown table per (axis, combination of the remaining axes) with at
    least two member cases, plus a flat case_stats.csv of every case."""
    if not results_ds.data_vars or "record" not in results_ds.dims:
        logger.warning("No collected results available, skipping comparison tables")
        return
    df = _case_stats_frame(results_ds, metrics_ds)
    if df.empty:
        logger.warning("No cases in the collected results, skipping comparison tables")
        return
    write_case_comparison_tables(df, SPEC, figure_dir)
