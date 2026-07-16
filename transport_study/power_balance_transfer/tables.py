"""Per-axis comparison tables for power balance transfer study results.

For every case axis (model type, training dataset, data normalization, domain
adaptation, number of target shots) and every combination of the remaining
axes, one markdown table comparing the cases that differ only along that axis.
Columns combine the stage-resolved TIME-AVERAGED errors from
collected_metrics.nc (per-timeslice means, free of the shot-duration confound
in the raw time-integrated per-shot errors) with the legacy time-integrated
medians from collected_results.nc.

A flat case_stats.csv with one row per case and every column is written next
to the tables for ad hoc analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from transport_study.orchestration.stages import STAGE_AGG_NAMES

# Submodule prereq cases predict P_oh / P_rad, not Wtot, so their errors are
# not comparable to the main models and are excluded from every table
SUBMODULE_MODEL_TYPES = ("p_oh", "p_rad")

# The case axes a table can compare across; freeze_submodules only has one
# value in practice so it stays a grouping field rather than an axis
AXIS_NAMES = ("model_type", "training_data", "data_normalization", "domain_adaptation", "num_target_shots")

CASE_FIELD_ORDER = ("model_type", "training_data", "data_normalization", "domain_adaptation", "freeze_submodules", "num_target_shots")

# Filename tokens per grouping field, mirroring the case-string vocabulary
_FIELD_TOKENS = {
    "model_type": "{}",
    "training_data": "td_{}",
    "data_normalization": "norm_{}",
    "domain_adaptation": "da_{}",
    "freeze_submodules": "freeze_{}",
    "num_target_shots": "targ_{}",
}

# Table columns: (header, dataframe column)
_COLUMNS = (
    ("rel err (time avg)", "rel_mean_all"),
    ("rampup", "rel_mean_rampup"),
    ("flattop", "rel_mean_flattop"),
    ("flattop ohmic", "rel_mean_flattop_ohmic"),
    ("flattop aux", "rel_mean_flattop_aux"),
    ("rampdown", "rel_mean_rampdown"),
    ("abs err (time avg) [MJ]", "abs_mean_all"),
    ("rel err (integral, med)", "err_rel_shot_med"),
    ("abs err (integral, med)", "err_abs_shot_med"),
)


def _case_stats_frame(results_ds: xr.Dataset, metrics_ds: xr.Dataset) -> pd.DataFrame:
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

    if metrics_ds.data_vars and "case_idx" in metrics_ds.dims:
        stage_cols = {}
        for metric in ("abs", "rel"):
            for stage in STAGE_AGG_NAMES:
                stage_cols[f"{metric}_mean_{stage}"] = metrics_ds[f"{metric}_mean"].sel(stage=stage).values
        stage_df = pd.DataFrame({"case_idx": metrics_ds["case_idx"].values, **stage_cols})
        df = df.merge(stage_df, on="case_idx", how="left")
    else:
        for metric in ("abs", "rel"):
            for stage in STAGE_AGG_NAMES:
                df[f"{metric}_mean_{stage}"] = np.nan

    return df[~df["model_type"].isin(SUBMODULE_MODEL_TYPES)]


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return f"{value:.4g}"


def _group_filename(axis: str, group_key: tuple) -> str:
    fields = [f for f in CASE_FIELD_ORDER if f != axis]
    tokens = [_FIELD_TOKENS[field].format(value) for field, value in zip(fields, group_key, strict=True)]
    return ".".join(tokens) + ".md"


def _axis_sort_key(axis: str, value):
    if axis == "num_target_shots":
        # The -1 sentinel means all target shots, list it after the real counts
        return (1, 0) if value == -1 else (0, int(value))
    return (0, str(value))


def _write_group_table(axis: str, group_key: tuple, group: pd.DataFrame, out_path: Path):
    fields = [f for f in CASE_FIELD_ORDER if f != axis]
    lines = [
        f"# {axis} comparison",
        "",
        "Fixed: " + ", ".join(f"{field}={value}" for field, value in zip(fields, group_key, strict=True)),
        "",
        "| " + " | ".join([axis, *(header for header, _ in _COLUMNS)]) + " |",
        "|" + "---|" * (len(_COLUMNS) + 1),
    ]
    group = group.sort_values(axis, key=lambda s: s.map(lambda v: _axis_sort_key(axis, v)))
    for _, row in group.iterrows():
        cells = [str(row[axis]), *(_fmt(row[col]) for _, col in _COLUMNS)]
        lines.append("| " + " | ".join(cells) + " |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def write_comparison_tables(results_ds: xr.Dataset, metrics_ds: xr.Dataset, figure_dir: Path):
    """One markdown table per (axis, combination of the remaining axes) with at
    least two member cases, plus a flat case_stats.csv of every case."""
    if not results_ds.data_vars or "case_idx" not in results_ds.dims:
        logger.warning("No collected results available, skipping comparison tables")
        return
    out_dir = Path(figure_dir) / "tables"
    df = _case_stats_frame(results_ds, metrics_ds)
    if df.empty:
        logger.warning("No main-model cases in the collected results, skipping comparison tables")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    df.sort_values(list(CASE_FIELD_ORDER)).to_csv(out_dir / "case_stats.csv", index=False)

    n_tables = 0
    for axis in AXIS_NAMES:
        group_fields = [f for f in CASE_FIELD_ORDER if f != axis]
        for group_key, group in df.groupby(group_fields):
            if len(group) < 2:
                continue
            _write_group_table(axis, group_key, group, out_dir / axis / _group_filename(axis, group_key))
            n_tables += 1
    logger.info(f"Saved {n_tables} comparison tables to {out_dir}")
