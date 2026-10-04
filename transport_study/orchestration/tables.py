"""Generic per-axis comparison tables for study results.

For every case axis (a case-grid field worth comparing across) and every
combination of the remaining case fields, one markdown table comparing the
cases that differ only along that axis, plus a flat case_stats.csv with one
row per case for ad hoc analysis.

Each study declares a ComparisonTableSpec in its tables module.
The studies whose collect_results is the per-case scalar summary (dims case_idx) build their stats frame here too,
the profile study builds its own from its long-form per-shot records.
Everything downstream of that frame (grouping, markdown, csv) is shared here.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from transport_study.orchestration.stages import STAGE_AGG_NAMES


@dataclass(frozen=True)
class ComparisonTableSpec:
    """Study-specific vocabulary for the comparison tables.

    axis_names: case fields a table can compare across (subset of case_field_order)
    case_field_order: every case-grid field, in filename and csv column order
    field_tokens: per-field filename token template (e.g. "td_{}"), mirroring
        the case-string vocabulary so table filenames read like case names
    columns: (table header, dataframe column) pairs
    stage_metrics: the metrics of the collected metrics dataset, merged in as <metric>_mean_<stage> columns
    excluded_model_types: submodule prereq case types, whose errors are not comparable to the main models
    """

    axis_names: tuple[str, ...]
    case_field_order: tuple[str, ...]
    field_tokens: dict[str, str]
    columns: tuple[tuple[str, str], ...]
    stage_metrics: tuple[str, ...]
    excluded_model_types: tuple[str, ...] = ()


def merge_stage_metrics(df: pd.DataFrame, metrics_ds: xr.Dataset, metric_names: tuple[str, ...]) -> pd.DataFrame:
    """Merge the per-stage means from a collected metrics dataset (dims
    case_idx x stage, variables <metric>_mean) into a per-case frame as
    <metric>_mean_<stage> columns, NaN for cases without valid metrics."""
    if metrics_ds.data_vars and "case_idx" in metrics_ds.dims:
        stage_cols = {}
        for metric in metric_names:
            for stage in STAGE_AGG_NAMES:
                stage_cols[f"{metric}_mean_{stage}"] = metrics_ds[f"{metric}_mean"].sel(stage=stage).values
        stage_df = pd.DataFrame({"case_idx": metrics_ds["case_idx"].values, **stage_cols})
        return df.merge(stage_df, on="case_idx", how="left")
    for metric in metric_names:
        for stage in STAGE_AGG_NAMES:
            df[f"{metric}_mean_{stage}"] = np.nan
    return df


# Integral error medians of the per-case scalar summary (Study.collect_results)
SUMMARY_ERROR_VARS = ("err_rel_shot_med", "err_abs_shot_med")


def summary_case_stats_frame(results_ds: xr.Dataset, metrics_ds: xr.Dataset, spec: ComparisonTableSpec) -> pd.DataFrame:
    """One row per main-model case of a per-case scalar summary.

    Columns: the case-grid coords, the integral error medians,
    and the per-stage time-averaged metrics (NaN for cases without valid metrics).
    """
    df = results_ds[list(SUMMARY_ERROR_VARS)].to_dataframe().reset_index()
    # Case coords shared by every case (e.g. freeze_submodules with a single configured option)
    # are scalar in the collected file and may not survive to_dataframe as columns, broadcast them back
    for field in spec.case_field_order:
        if field not in df.columns:
            df[field] = results_ds[field].item()
    df = df[[*spec.case_field_order, "case_idx", *SUMMARY_ERROR_VARS]]
    df = merge_stage_metrics(df, metrics_ds, spec.stage_metrics)
    return df[~df["model_type"].isin(spec.excluded_model_types)]


def write_summary_comparison_tables(results_ds: xr.Dataset, metrics_ds: xr.Dataset, spec: ComparisonTableSpec, figure_dir: Path):
    """write_case_comparison_tables over the summary_case_stats_frame of a per-case scalar summary."""
    if not results_ds.data_vars or "case_idx" not in results_ds.dims:
        logger.warning("No collected results available, skipping comparison tables")
        return
    df = summary_case_stats_frame(results_ds, metrics_ds, spec)
    write_case_comparison_tables(df, spec, figure_dir)


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return f"{value:.4g}"


def _group_filename(spec: ComparisonTableSpec, axis: str, group_key: tuple) -> str:
    fields = [f for f in spec.case_field_order if f != axis]
    tokens = [spec.field_tokens[field].format(value) for field, value in zip(fields, group_key, strict=True)]
    return ".".join(tokens) + ".md"


def _axis_sort_key(axis: str, value):
    if axis == "num_target_shots":
        # The -1 sentinel means all target shots, list it after the real counts
        return (1, 0) if value == -1 else (0, int(value))
    return (0, str(value))


def _write_group_table(spec: ComparisonTableSpec, axis: str, group_key: tuple, group: pd.DataFrame, out_path: Path):
    fields = [f for f in spec.case_field_order if f != axis]
    lines = [
        f"# {axis} comparison",
        "",
        "Fixed: " + ", ".join(f"{field}={value}" for field, value in zip(fields, group_key, strict=True)),
        "",
        "| " + " | ".join([axis, *(header for header, _ in spec.columns)]) + " |",
        "|" + "---|" * (len(spec.columns) + 1),
    ]
    group = group.sort_values(axis, key=lambda s: s.map(lambda v: _axis_sort_key(axis, v)))
    for _, row in group.iterrows():
        cells = [str(row[axis]), *(_fmt(row[col]) for _, col in spec.columns)]
        lines.append("| " + " | ".join(cells) + " |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def write_case_comparison_tables(df: pd.DataFrame, spec: ComparisonTableSpec, figure_dir: Path):
    """One markdown table per (axis, combination of the remaining case fields)
    with at least two member cases, plus a flat case_stats.csv of every case."""
    if df.empty:
        logger.warning("No cases in the per-case stats frame, skipping comparison tables")
        return
    out_dir = Path(figure_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.sort_values(list(spec.case_field_order)).to_csv(out_dir / "case_stats.csv", index=False)

    n_tables = 0
    for axis in spec.axis_names:
        group_fields = [f for f in spec.case_field_order if f != axis]
        for group_key, group in df.groupby(group_fields):
            if len(group) < 2:
                continue
            _write_group_table(spec, axis, group_key, group, out_dir / axis / _group_filename(spec, axis, group_key))
            n_tables += 1
    logger.info(f"Saved {n_tables} comparison tables to {out_dir}")
