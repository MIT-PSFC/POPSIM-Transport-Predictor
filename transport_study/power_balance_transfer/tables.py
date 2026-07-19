"""Per-axis comparison tables for power balance transfer study results.

For every case axis (model type, training dataset, data normalization, domain
adaptation, number of target shots) and every combination of the remaining
axes, one markdown table comparing the cases that differ only along that axis.
Columns combine the stage-resolved TIME-AVERAGED errors from
collected_metrics.nc (per-timeslice means, free of the shot-duration confound
in the raw time-integrated per-shot errors) with the legacy time-integrated
medians from collected_results.nc.

A flat case_stats.csv with one row per case and every column is written next
to the tables for ad hoc analysis. The table and csv writing itself is the
shared orchestration.tables machinery, so this module only declares the spec
and builds the per-case stats frame.
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

# Submodule prereq cases predict P_oh / P_rad, not Wtot, so their errors are
# not comparable to the main models and are excluded from every table
SUBMODULE_MODEL_TYPES = ("p_oh", "p_rad")

# Stage-resolved metric variables in collected_metrics.nc
_STAGE_METRICS = ("abs", "rel")

CASE_FIELD_ORDER = ("model_type", "training_data", "data_normalization", "domain_adaptation", "freeze_submodules", "num_target_shots")

SPEC = ComparisonTableSpec(
    # freeze_submodules only has one value in practice so it stays a grouping
    # field rather than an axis
    axis_names=("model_type", "training_data", "data_normalization", "domain_adaptation", "num_target_shots"),
    case_field_order=CASE_FIELD_ORDER,
    # Filename tokens per grouping field, mirroring the case-string vocabulary
    field_tokens={
        "model_type": "{}",
        "training_data": "td_{}",
        "data_normalization": "norm_{}",
        "domain_adaptation": "da_{}",
        "freeze_submodules": "freeze_{}",
        "num_target_shots": "targ_{}",
    },
    columns=(
        ("rel err (time avg)", "rel_mean_all"),
        ("rampup", "rel_mean_rampup"),
        ("flattop", "rel_mean_flattop"),
        ("flattop ohmic", "rel_mean_flattop_ohmic"),
        ("flattop aux", "rel_mean_flattop_aux"),
        ("rampdown", "rel_mean_rampdown"),
        ("abs err (time avg) [MJ]", "abs_mean_all"),
        ("rel err (integral, med)", "err_rel_shot_med"),
        ("abs err (integral, med)", "err_abs_shot_med"),
    ),
)


def case_stats_frame(results_ds: xr.Dataset, metrics_ds: xr.Dataset) -> pd.DataFrame:
    """One row per case: case-grid coords, integral error medians from the
    collected results, and per-stage time-averaged errors from the collected
    metrics (NaN for cases without valid metrics)."""
    keep_vars = ["err_rel_shot_med", "err_abs_shot_med"]
    df = results_ds[keep_vars].to_dataframe().reset_index()
    # Case coords shared by every case (e.g. freeze_submodules with a single
    # configured option) are scalar in the collected file and may not survive
    # to_dataframe as columns; broadcast them back
    for field in CASE_FIELD_ORDER:
        if field not in df.columns:
            df[field] = results_ds[field].item()
    df = df[[*CASE_FIELD_ORDER, "case_idx", *keep_vars]]
    df = merge_stage_metrics(df, metrics_ds, _STAGE_METRICS)
    return df[~df["model_type"].isin(SUBMODULE_MODEL_TYPES)]


def write_comparison_tables(results_ds: xr.Dataset, metrics_ds: xr.Dataset, figure_dir: Path):
    """One markdown table per (axis, combination of the remaining axes) with at
    least two member cases, plus a flat case_stats.csv of every case."""
    if not results_ds.data_vars or "case_idx" not in results_ds.dims:
        logger.warning("No collected results available, skipping comparison tables")
        return
    df = case_stats_frame(results_ds, metrics_ds)
    if df.empty:
        logger.warning("No main-model cases in the collected results, skipping comparison tables")
        return
    write_case_comparison_tables(df, SPEC, figure_dir)
