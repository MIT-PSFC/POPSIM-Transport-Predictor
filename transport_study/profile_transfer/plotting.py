"""Case-comparison figures of profile transfer study results (see orchestration.comparison_figures).

A grid with one row per chi metric (value / gradient / combined, see study_metrics)
and one column per shot stage (all, rampup, flattop, flattop ohmic, flattop aux, rampdown).
Consumes the stage-resolved collected metrics (collected_metrics.nc, dims case_idx x stage).
freeze_shapes_comparison plots the frozen - unfrozen difference on the same grid.
"""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loguru import logger

from transport_study.modules.profile_predictor.module import MODEL_TYPES_WITH_SHAPES
from transport_study.modules.profile_predictor.torax_module import TORAX_MODEL_TYPES
from transport_study.orchestration.comparison_figures import (
    DA_COLORS,
    DA_LABELS,
    DIVERGED_THRESHOLD,
    NORM_COLORS,
    NORM_LABELS,
    ORDER_COLORS,
    ORDER_LABELS,
    ORDER_MARKERS,
    ComparisonFamily,
    ComparisonLayout,
    coord_values,
    finalize_grid,
    grid_figure,
    mask_select,
)
from transport_study.orchestration.stages import STAGE_AGG_NAMES
from transport_study.plot_style import LABEL_FONTSIZE, TEXT_COLOR
from transport_study.profile_transfer.study_metrics import METRIC_NAMES

MODEL_COLORS = {
    "shape-init-pca": "#0095ff",
    "shape-init-kmeans": "#00d5ff",
    "mlp": "#ff4d4d",
    "reservoir": "#ffb347",
    "torax-constant": "#8dff36",
    "torax-gyrobohm": "#b4ff9e",
    "torax-qlknn": "#0e8a5f",
}

MODEL_LABELS = {
    "shape-init-pca": "Shape init (PCA)",
    "shape-init-kmeans": "Shape init (k-means)",
    "mlp": "MLP",
    "reservoir": "Reservoir",
    "torax-constant": "TORAX constant",
    "torax-gyrobohm": "TORAX GyroBohm",
    "torax-qlknn": "TORAX QLKNN",
}

GEOM_COLORS = {
    "circular": "#0095ff",
    "miller": "#ffb347",
}

GEOM_LABELS = {
    "circular": "Circular",
    "miller": "Miller",
}

METRIC_LABELS = {
    "value": "Value loss",
    "grad": "Gradient loss",
    "combined": "Combined loss",
}

STAGE_LABELS = {
    "all": "All",
    "rampup": "Rampup",
    "flattop": "Flattop",
    "flattop_ohmic": "Flattop (ohmic)",
    "flattop_aux": "Flattop (aux)",
    "rampdown": "Rampdown",
}


def _stage_series_stats(ds: xr.Dataset, metric: str, stage: str) -> tuple[np.ndarray, np.ndarray]:
    stage_ds = ds.sel(stage=stage)
    return stage_ds[f"{metric}_mean"].values, stage_ds[f"{metric}_std"].values


LAYOUT = ComparisonLayout(
    row_labels={metric: METRIC_LABELS[metric] for metric in METRIC_NAMES},
    col_labels={stage: STAGE_LABELS[stage] for stage in STAGE_AGG_NAMES},
    cell_size=(3.4, 2.9),
    series_stats=_stage_series_stats,
    grid_fields=(
        "model_type",
        "training_data",
        "data_normalization",
        "domain_adaptation",
        "freeze_shapes",
        "geometry_builder",
        "target_shot_order",
    ),
    field_tokens={
        "model_type": "{}",
        "training_data": "td_{}",
        "data_normalization": "norm_{}",
        "domain_adaptation": "da_{}",
        "freeze_shapes": "freeze_{}",
        "geometry_builder": "geom_{}",
        "target_shot_order": "order_{}",
    },
    value_labels={
        "model_type": MODEL_LABELS,
        "data_normalization": NORM_LABELS,
        "domain_adaptation": DA_LABELS,
        "geometry_builder": GEOM_LABELS,
        "target_shot_order": ORDER_LABELS,
    },
)

COMPARISON_FAMILIES = (
    ComparisonFamily("training_dataset_comparison", "training_data", "Training dataset comparison"),
    ComparisonFamily("model_comparison", "model_type", "Model comparison", MODEL_COLORS),
    # The no-adaptation baseline only exists at num_target_shots = 0 for non-exnihilo training data,
    # so it typically shows up as a single point rather than a trend
    ComparisonFamily("domain_adaptation_comparison", "domain_adaptation", "Domain adaptation comparison", DA_COLORS),
    ComparisonFamily("data_normalization_comparison", "data_normalization", "Data normalization comparison", NORM_COLORS),
    # Only the torax model types carry the geometry_builder axis, the others are pinned to circular
    ComparisonFamily(
        "geometry_builder_comparison", "geometry_builder", "Geometry builder comparison", GEOM_COLORS, TORAX_MODEL_TYPES, min_series=2
    ),
    # Without target shots every case takes the base order, so the other orders start at 1 target shot
    ComparisonFamily("target_shot_order_comparison", "target_shot_order", "Target shot order comparison", ORDER_COLORS, min_series=2),
)


def _freeze_diff(frozen_by_shot: dict, unfrozen_by_shot: dict, shot) -> float:
    """Frozen minus unfrozen metric at one num_target_shots value, NaN when
    either side is missing, non-finite, or diverged."""
    f, u = frozen_by_shot[shot], unfrozen_by_shot[shot]
    if not np.isfinite(f) or not np.isfinite(u) or f > DIVERGED_THRESHOLD or u > DIVERGED_THRESHOLD:
        return np.nan
    return f - u


def freeze_shapes_comparison(metrics_ds: xr.Dataset, figure_dir: Path):
    """Effect of freezing shapes: (frozen - unfrozen) metric difference vs num_target_shots.

    One line per combination of the other case-grid fields, one figure per model type that has shapes.
    Positive difference means the unfrozen model is better.
    """
    if not metrics_ds.data_vars or "case_idx" not in metrics_ds.dims:
        logger.warning("No stage-resolved metrics available, skipping freeze shapes comparison figures")
        return
    out_dir = Path(figure_dir) / "comparison" / "freeze_shapes_comparison"
    training_datasets = coord_values(metrics_ds, "training_data")
    cmap = plt.colormaps["tab10"].resampled(max(len(training_datasets), 1))
    td_colors = {td: cmap(i) for i, td in enumerate(training_datasets)}
    da_linestyles = {"none": ":", "weighted": "-", "addition": (0, (3, 1, 1, 1)), "transfer": "--", "transfer_pretrain": "-."}

    for model_type in coord_values(metrics_ds, "model_type"):
        if model_type not in MODEL_TYPES_WITH_SHAPES:
            continue
        model_sub = mask_select(metrics_ds, metrics_ds["model_type"] == model_type)
        norms = coord_values(model_sub, "data_normalization")
        geoms = coord_values(model_sub, "geometry_builder")
        orders = coord_values(model_sub, "target_shot_order")
        line_fields = product(
            coord_values(model_sub, "training_data"),
            norms,
            coord_values(model_sub, "domain_adaptation"),
            geoms,
            orders,
        )
        fig, axes = grid_figure(LAYOUT)
        drew = False
        for td, dn, da, geom, order in line_fields:
            combo_mask = (
                (model_sub["training_data"] == td)
                & (model_sub["data_normalization"] == dn)
                & (model_sub["domain_adaptation"] == da)
                & (model_sub["geometry_builder"] == geom)
                & (model_sub["target_shot_order"] == order)
            )
            frozen = mask_select(model_sub, combo_mask & model_sub["freeze_shapes"])
            unfrozen = mask_select(model_sub, combo_mask & ~model_sub["freeze_shapes"])
            if frozen.sizes.get("case_idx", 0) == 0 or unfrozen.sizes.get("case_idx", 0) == 0:
                continue

            frozen_shots = np.atleast_1d(frozen["num_target_shots"].values)
            unfrozen_shots = np.atleast_1d(unfrozen["num_target_shots"].values)
            shared = sorted(set(frozen_shots.tolist()) & set(unfrozen_shots.tolist()))

            color = td_colors[td]
            linestyle = da_linestyles.get(da, "-")
            label = f"{td} / {DA_LABELS.get(da, da)}"
            if len(norms) > 1:
                label = f"{label} / {NORM_LABELS.get(dn, dn)}"
            if len(geoms) > 1:
                label = f"{label} / {GEOM_LABELS.get(geom, geom)}"
            if len(orders) > 1:
                label = f"{label} / {ORDER_LABELS.get(order, order)}"

            for row, metric in enumerate(METRIC_NAMES):
                for col, stage in enumerate(STAGE_AGG_NAMES):
                    frozen_means = np.atleast_1d(frozen.sel(stage=stage)[f"{metric}_mean"].values).astype(float)
                    unfrozen_means = np.atleast_1d(unfrozen.sel(stage=stage)[f"{metric}_mean"].values).astype(float)
                    frozen_by_shot = dict(zip(frozen_shots.tolist(), frozen_means.tolist(), strict=True))
                    unfrozen_by_shot = dict(zip(unfrozen_shots.tolist(), unfrozen_means.tolist(), strict=True))
                    diffs = np.array([_freeze_diff(frozen_by_shot, unfrozen_by_shot, s) for s in shared])
                    if np.isfinite(diffs).any():
                        axes[row, col].plot(
                            shared,
                            diffs,
                            color=color,
                            linestyle=linestyle,
                            linewidth=1.6,
                            marker=ORDER_MARKERS.get(order, "o"),
                            markersize=4,
                            label=label,
                        )
                        drew = True

        if not drew:
            plt.close(fig)
            continue
        for ax in axes.flat:
            ax.axhline(0, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
        for row, metric in enumerate(METRIC_NAMES):
            axes[row, 0].set_ylabel(
                f"Delta {METRIC_LABELS[metric].lower()}\n(frozen - unfrozen)",
                color=TEXT_COLOR,
                fontsize=LABEL_FONTSIZE - 1,
            )
        model_label = MODEL_LABELS.get(model_type, model_type)
        title = f"Shape freezing effect - {model_label} (positive: unfrozen better)"
        finalize_grid(fig, axes, model_sub, title, out_dir / f"{model_type}.png")
    logger.info(f"Saved freeze shapes comparison figures to {out_dir}")
