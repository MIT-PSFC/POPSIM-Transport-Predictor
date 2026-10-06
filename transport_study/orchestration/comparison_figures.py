"""Case-comparison figures shared by every study.

Every figure is a grid of panels with the number of target-device shots in training on a semilog x axis.
A ComparisonLayout holds one study's panel rows and columns and its case-grid vocabulary,
a ComparisonFamily one kind of comparison:
a line per value of its series field, a figure per combination of the remaining case-grid fields.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loguru import logger

from transport_study.plot_style import (
    BACKGROUND_COLOR,
    LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    LEGEND_STYLE,
    TEXT_COLOR,
    TICK_FONTSIZE,
    style_axis,
)

TITLE_FONTSIZE = 14

# Metric value above which a case is treated as diverged and masked out
DIVERGED_THRESHOLD = 1e3

DA_COLORS = {
    "none": "#c0c0c0",
    "weighted": "#ad2cfe",
    "addition": "#ff2ea6",
    "transfer": "#00ff5e",
    "transfer_pretrain": "#00a33c",
}

DA_LABELS = {
    "none": "No adaptation",
    "weighted": "Weighted",
    "addition": "Addition",
    "transfer": "Transfer",
    "transfer_pretrain": "Transfer pretrain",
}

NORM_COLORS = {
    "raw": "#ff4d4d",
    "physics": "#8dff36",
    "zscore": "#0095ff",
    "coral": "#ff60ec",
    "physics-coral": "#ffb52e",
    "physics-zscore": "#40e0d0",
}

NORM_LABELS = {
    "raw": "Raw",
    "physics": "Physics",
    "zscore": "Z-score",
    "coral": "CORAL",
    "physics-coral": "Physics CORAL",
    "physics-zscore": "Physics z-score",
}

# The target shot orders (see orchestration/target_shots.py)
ORDER_COLORS = {
    "ascending": "#c0c0c0",
    "descending": "#ff8c1a",
    "spanning": "#1ad1ff",
}

ORDER_LABELS = {
    "ascending": "Lowest hazard first",
    "descending": "Highest hazard first",
    "spanning": "Spanning",
}

# Markers that tell the orders apart where the line color already encodes another field
ORDER_MARKERS = {
    "ascending": "o",
    "descending": "v",
    "spanning": "D",
}


@dataclass(frozen=True)
class ComparisonLayout:
    """One study's comparison grid and case-grid vocabulary.

    row_labels / col_labels: panel rows and columns, name -> axis label
    cell_size: (width, height) of one panel in inches
    series_stats(ds, row, col): per-case (means, stds) of one panel
    grid_fields: every case-grid field a figure compares along or holds fixed
    field_tokens: per-field filename token template (e.g. "td_{}"), mirroring the case-string vocabulary
    value_labels: per-field value -> display label, in titles and legends
    excluded_model_types: submodule prereq case types, whose errors are not comparable to the main models
    """

    row_labels: dict[str, str]
    col_labels: dict[str, str]
    cell_size: tuple[float, float]
    series_stats: Callable[[xr.Dataset, str, str], tuple[np.ndarray, np.ndarray]]
    grid_fields: tuple[str, ...]
    field_tokens: dict[str, str]
    value_labels: dict[str, dict] = field(default_factory=dict)
    excluded_model_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComparisonFamily:
    """A line per series_field value, a figure per combination of the remaining grid fields.

    series_colors None colors the values present with tab10.
    model_types restricts the cases to those model types, None keeps every main model.
    Figures with fewer than min_series lines are skipped.
    """

    name: str
    series_field: str
    title: str
    series_colors: dict | None = None
    model_types: tuple[str, ...] | None = None
    min_series: int = 1


def coord_values(ds: xr.Dataset, name: str) -> list:
    return sorted(set(np.atleast_1d(ds[name].values).tolist()))


def field_mask(ds: xr.Dataset, field_name: str, value) -> np.ndarray:
    """Boolean case_idx mask for field == value.

    Case coords shared by every case (e.g. freeze_submodules with a single configured option)
    are scalar in the collected file, so the comparison is broadcast back to the case_idx length.
    """
    matches = np.atleast_1d(ds[field_name].values) == value
    return np.broadcast_to(matches, (ds.sizes["case_idx"],))


def mask_select(ds: xr.Dataset, mask) -> xr.Dataset:
    return ds.isel(case_idx=np.asarray(mask))


def grid_figure(layout: ComparisonLayout) -> tuple[plt.Figure, np.ndarray]:
    n_rows, n_cols = len(layout.row_labels), len(layout.col_labels)
    cell_width, cell_height = layout.cell_size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell_width * n_cols, cell_height * n_rows), sharex=True, squeeze=False)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    for row, row_label in enumerate(layout.row_labels.values()):
        for col, col_label in enumerate(layout.col_labels.values()):
            ax = axes[row, col]
            style_axis(ax)
            if row == 0:
                ax.set_title(col_label, color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
            if col == 0:
                ax.set_ylabel(row_label, color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
            if row == n_rows - 1:
                ax.set_xlabel("Target shots in training", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
    return fig, axes


def plot_series(ax, shots: np.ndarray, means: np.ndarray, stds: np.ndarray, color, label: str, linestyle: str = "-") -> bool:
    """One comparison member on one axis.

    An errorbar line over num_target_shots.
    Diverged cases are masked out. Returns whether anything was drawn.
    """
    shots = np.atleast_1d(shots)
    means = np.atleast_1d(means).astype(float).copy()
    stds = np.atleast_1d(stds).astype(float).copy()
    diverged = means > DIVERGED_THRESHOLD
    means[diverged] = np.nan
    stds[diverged] = np.nan

    if not np.isfinite(means).any():
        return False
    order = np.argsort(shots)
    ax.errorbar(
        shots[order],
        means[order],
        yerr=stds[order],
        label=label,
        color=color,
        linestyle=linestyle,
        marker="o",
        markersize=4,
        linewidth=1.6,
        capsize=3,
        capthick=1.0,
    )
    return True


def finalize_grid(fig, axes, sub: xr.Dataset, title: str, save_path: Path):
    """Shared x ticks, one figure legend and the title, then save and close."""
    shots = np.atleast_1d(sub["num_target_shots"].values)
    tick_shots = sorted({int(s) for s in shots})
    for ax in axes.flat:
        ax.set_xscale("symlog", linthresh=1)
        if tick_shots:
            ax.set_xticks(tick_shots)
            ax.set_xticklabels([str(s) for s in tick_shots])
        ax.tick_params(colors=TEXT_COLOR, labelsize=TICK_FONTSIZE)

    handles, labels = [], []
    for ax in axes.flat:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncols=min(len(labels), 4), fontsize=LEGEND_FONTSIZE, **LEGEND_STYLE)
    fig.suptitle(title, color=TEXT_COLOR, fontsize=TITLE_FONTSIZE)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def _series_colors(family: ComparisonFamily, series_values: list) -> dict:
    if family.series_colors is not None:
        return family.series_colors
    cmap = plt.colormaps["tab10"].resampled(max(len(series_values), 1))
    return {value: cmap(i) for i, value in enumerate(series_values)}


def comparison_figures(ds: xr.Dataset, layout: ComparisonLayout, family: ComparisonFamily, figure_dir: Path):
    """Every figure of one comparison family, ds being the per-case collected results or metrics (dims case_idx)."""
    if not ds.data_vars or "case_idx" not in ds.dims:
        logger.warning(f"No collected cases available, skipping {family.name} figures")
        return
    excluded = set(layout.excluded_model_types)
    model_types = [mt for mt in coord_values(ds, "model_type") if mt not in excluded]
    if family.model_types is not None:
        model_types = [mt for mt in model_types if mt in family.model_types]
    mask_models = np.isin(np.atleast_1d(ds["model_type"].values), model_types)
    ds_models = mask_select(ds, np.broadcast_to(mask_models, (ds.sizes["case_idx"],)))
    if ds_models.sizes["case_idx"] == 0:
        return

    out_dir = Path(figure_dir) / "comparison" / family.name
    series_values = coord_values(ds_models, family.series_field)
    series_colors = _series_colors(family, series_values)
    series_labels = layout.value_labels.get(family.series_field, {})
    fixed_fields = [f for f in layout.grid_fields if f != family.series_field]
    fixed_values = [coord_values(ds_models, f) for f in fixed_fields]

    for combo in product(*fixed_values):
        mask = np.ones(ds_models.sizes["case_idx"], dtype=bool)
        for fixed_field, value in zip(fixed_fields, combo, strict=True):
            mask = mask & field_mask(ds_models, fixed_field, value)
        sub = mask_select(ds_models, mask)
        if sub.sizes["case_idx"] == 0:
            continue
        present = [value for value in series_values if value in coord_values(sub, family.series_field)]
        if len(present) < family.min_series:
            continue

        fig, axes = grid_figure(layout)
        drew = False
        for series_value in present:
            series_sub = mask_select(sub, field_mask(sub, family.series_field, series_value))
            shots = series_sub["num_target_shots"].values
            color = series_colors.get(series_value, "white")
            label = series_labels.get(series_value, str(series_value))
            for row, row_name in enumerate(layout.row_labels):
                for col, col_name in enumerate(layout.col_labels):
                    means, stds = layout.series_stats(series_sub, row_name, col_name)
                    drew |= plot_series(axes[row, col], shots, means, stds, color, label)
        if not drew:
            plt.close(fig)
            continue
        fixed_desc = " / ".join(
            f"{fixed_field}: {layout.value_labels.get(fixed_field, {}).get(value, value)}"
            for fixed_field, value in zip(fixed_fields, combo, strict=True)
        )
        filename = ".".join(layout.field_tokens[f].format(value) for f, value in zip(fixed_fields, combo, strict=True))
        finalize_grid(fig, axes, sub, f"{family.title} - {fixed_desc}", out_dir / f"{filename}.png")
    logger.info(f"Saved {family.name} figures to {out_dir}")
