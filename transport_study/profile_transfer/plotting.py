"""Case-comparison figures for profile transfer study results.

All figures share one layout: a grid with one row per performance metric
(value / gradient / combined, see study_metrics) and one column per shot stage
(rampup, flattop, flattop ohmic, flattop aux, rampdown, all), with the number
of target-device shots included in training on a semilog x axis. Each family
draws one line per member of the comparison (model type, training dataset,
domain adaptation method, or freeze_shapes difference) and one figure per
combination of the remaining case dimensions.

The num_target_shots = -1 sentinel (all target shots in training, the cheating
reference) has no x position: it is drawn as a dashed horizontal line in the
series color.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loguru import logger

from transport_study.plot_style import BACKGROUND_COLOR, TEXT_COLOR, style_axis
from transport_study.profile_transfer.study_metrics import METRIC_NAMES, STAGE_AGG_NAMES

TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 11
TICK_FONTSIZE = 9
LEGEND_FONTSIZE = 10

MODEL_COLORS = {
    "shape_init_pca": "#0095ff",
    "shape_init_kmeans": "#00d5ff",
    "unstructured_nn": "#ff4d4d",
    "reservoir": "#ffb347",
    "torax-constant": "#8dff36",
    "torax-cgm": "#2fbf71",
    "torax-gyrobohm": "#b4ff9e",
    "torax-qlknn": "#0e8a5f",
}

MODEL_LABELS = {
    "shape_init_pca": "Shape init (PCA)",
    "shape_init_kmeans": "Shape init (k-means)",
    "unstructured_nn": "Unstructured NN",
    "reservoir": "Reservoir",
    "torax-constant": "TORAX constant",
    "torax-cgm": "TORAX CGM",
    "torax-gyrobohm": "TORAX GyroBohm",
    "torax-qlknn": "TORAX QLKNN",
}

DA_COLORS = {
    "none": "#c0c0c0",
    "mixing": "#ad2cfe",
    "transfer": "#00ff5e",
    "transfer_pretrain": "#00a33c",
}

DA_LABELS = {
    "none": "No adaptation",
    "mixing": "Mixing",
    "transfer": "Transfer",
    "transfer_pretrain": "Transfer pretrain",
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

# Metric value above which a case is treated as diverged and masked out
DIVERGED_THRESHOLD = 1e3

# Model families with shape bases that can be frozen or trained
MODELS_WITH_SHAPES = ("shape_init_pca", "shape_init_kmeans", "torax-constant", "torax-cgm", "torax-gyrobohm", "torax-qlknn")


def _training_data_colors(training_datasets: list[str]) -> dict[str, tuple]:
    cmap = plt.colormaps["tab10"].resampled(max(len(training_datasets), 1))
    return {td: cmap(i) for i, td in enumerate(sorted(training_datasets))}


def _mask_select(ds: xr.Dataset, mask) -> xr.Dataset:
    return ds.isel(case_idx=np.asarray(mask))


def _grid_figure() -> tuple[plt.Figure, np.ndarray]:
    n_rows, n_cols = len(METRIC_NAMES), len(STAGE_AGG_NAMES)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * n_cols, 2.9 * n_rows),
        sharex=True,
        squeeze=False,
    )
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    for row, metric in enumerate(METRIC_NAMES):
        for col, stage in enumerate(STAGE_AGG_NAMES):
            ax = axes[row, col]
            style_axis(ax, TICK_FONTSIZE)
            if row == 0:
                ax.set_title(STAGE_LABELS[stage], color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric], color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
            if row == len(METRIC_NAMES) - 1:
                ax.set_xlabel("Target shots in training", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
    return fig, axes


def _plot_series(ax, sub: xr.Dataset, stage: str, metric: str, color, label: str, linestyle: str = "-") -> bool:
    """One comparison member on one axis: errorbar line over num_target_shots >= 0,
    plus a dashed horizontal reference for the -1 (all target shots) sentinel.
    Returns whether anything was drawn."""
    stage_sub = sub.sel(stage=stage)
    shots = np.atleast_1d(stage_sub["num_target_shots"].values)
    means = np.atleast_1d(stage_sub[f"{metric}_mean"].values).astype(float).copy()
    stds = np.atleast_1d(stage_sub[f"{metric}_std"].values).astype(float).copy()

    diverged = means > DIVERGED_THRESHOLD
    means[diverged] = np.nan
    stds[diverged] = np.nan

    drew = False
    line_mask = shots >= 0
    if line_mask.any() and np.isfinite(means[line_mask]).any():
        order = np.argsort(shots[line_mask])
        ax.errorbar(
            shots[line_mask][order],
            means[line_mask][order],
            yerr=stds[line_mask][order],
            label=label,
            color=color,
            linestyle=linestyle,
            marker="o",
            markersize=4,
            linewidth=1.6,
            capsize=3,
            capthick=1.0,
        )
        drew = True

    for ref_mean in means[shots == -1]:
        if np.isfinite(ref_mean):
            ax.axhline(ref_mean, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
            drew = True

    return drew


def _finalize_grid(fig, axes, sub: xr.Dataset, title: str, save_path: Path):
    shots = np.atleast_1d(sub["num_target_shots"].values)
    tick_shots = sorted({int(s) for s in shots if s >= 0})
    for ax in axes.flat:
        ax.set_xscale("symlog", linthresh=1)
        if tick_shots:
            ax.set_xticks(tick_shots)
            ax.set_xticklabels([str(s) for s in tick_shots])
        ax.tick_params(colors=TEXT_COLOR, labelsize=TICK_FONTSIZE)

    handles, labels = [], []
    for ax in axes.flat:
        h, ell = ax.get_legend_handles_labels()
        for handle, lab in zip(h, ell, strict=True):
            if lab not in labels:
                handles.append(handle)
                labels.append(lab)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncols=min(len(labels), 4),
            fontsize=LEGEND_FONTSIZE,
            labelcolor=TEXT_COLOR,
            facecolor=BACKGROUND_COLOR,
            edgecolor=TEXT_COLOR,
        )
    fig.suptitle(title, color=TEXT_COLOR, fontsize=TITLE_FONTSIZE)
    fig.text(
        0.99,
        0.01,
        "dashed: trained on all target shots",
        color=TEXT_COLOR,
        fontsize=TICK_FONTSIZE,
        ha="right",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def _coord_values(ds: xr.Dataset, name: str) -> list:
    return sorted(set(np.atleast_1d(ds[name].values).tolist()))


def _freeze_diff(frozen_by_shot: dict, unfrozen_by_shot: dict, shot) -> float:
    """Frozen minus unfrozen metric at one num_target_shots value, NaN when
    either side is missing, non-finite, or diverged."""
    f, u = frozen_by_shot[shot], unfrozen_by_shot[shot]
    if not np.isfinite(f) or not np.isfinite(u) or f > DIVERGED_THRESHOLD or u > DIVERGED_THRESHOLD:
        return np.nan
    return f - u


def _check_metrics_ds(metrics_ds: xr.Dataset, family: str) -> bool:
    if not metrics_ds.data_vars or "case_idx" not in metrics_ds.dims:
        logger.warning(f"No stage-resolved metrics available, skipping {family} figures")
        return False
    return True


def model_comparison(metrics_ds: xr.Dataset, figure_dir: Path):
    """One line per model type, one figure per (training_data, domain_adaptation, freeze_shapes)."""
    if not _check_metrics_ds(metrics_ds, "model comparison"):
        return
    out_dir = Path(figure_dir) / "comparison" / "model_comparison"
    for td in _coord_values(metrics_ds, "training_data"):
        for da in _coord_values(metrics_ds, "domain_adaptation"):
            for freeze in _coord_values(metrics_ds, "freeze_shapes"):
                sub = _mask_select(
                    metrics_ds,
                    (metrics_ds["training_data"] == td) & (metrics_ds["domain_adaptation"] == da) & (metrics_ds["freeze_shapes"] == freeze),
                )
                if sub.sizes.get("case_idx", 0) == 0:
                    continue
                fig, axes = _grid_figure()
                drew = False
                for model_type in _coord_values(sub, "model_type"):
                    model_sub = _mask_select(sub, sub["model_type"] == model_type)
                    color = MODEL_COLORS.get(model_type, "white")
                    label = MODEL_LABELS.get(model_type, model_type)
                    for row, metric in enumerate(METRIC_NAMES):
                        for col, stage in enumerate(STAGE_AGG_NAMES):
                            drew |= _plot_series(axes[row, col], model_sub, stage, metric, color, label)
                if not drew:
                    plt.close(fig)
                    continue
                title = f"Model comparison - train: {td} / DA: {da} / freeze shapes: {freeze}"
                _finalize_grid(fig, axes, sub, title, out_dir / f"td_{td}.da_{da}.freeze_{freeze}.png")
    logger.info(f"Saved model comparison figures to {out_dir}")


def training_dataset_comparison(metrics_ds: xr.Dataset, figure_dir: Path):
    """One line per training dataset, one figure per (model_type, domain_adaptation, freeze_shapes)."""
    if not _check_metrics_ds(metrics_ds, "training dataset comparison"):
        return
    out_dir = Path(figure_dir) / "comparison" / "training_dataset_comparison"
    td_colors = _training_data_colors(_coord_values(metrics_ds, "training_data"))
    for model_type in _coord_values(metrics_ds, "model_type"):
        for da in _coord_values(metrics_ds, "domain_adaptation"):
            for freeze in _coord_values(metrics_ds, "freeze_shapes"):
                sub = _mask_select(
                    metrics_ds,
                    (metrics_ds["model_type"] == model_type)
                    & (metrics_ds["domain_adaptation"] == da)
                    & (metrics_ds["freeze_shapes"] == freeze),
                )
                if sub.sizes.get("case_idx", 0) == 0:
                    continue
                fig, axes = _grid_figure()
                drew = False
                for td in _coord_values(sub, "training_data"):
                    td_sub = _mask_select(sub, sub["training_data"] == td)
                    for row, metric in enumerate(METRIC_NAMES):
                        for col, stage in enumerate(STAGE_AGG_NAMES):
                            drew |= _plot_series(axes[row, col], td_sub, stage, metric, td_colors[td], str(td))
                if not drew:
                    plt.close(fig)
                    continue
                model_label = MODEL_LABELS.get(model_type, model_type)
                title = f"Training dataset comparison - {model_label} / DA: {da} / freeze shapes: {freeze}"
                _finalize_grid(fig, axes, sub, title, out_dir / f"{model_type}.da_{da}.freeze_{freeze}.png")
    logger.info(f"Saved training dataset comparison figures to {out_dir}")


def domain_adaptation_comparison(metrics_ds: xr.Dataset, figure_dir: Path):
    """One line per domain adaptation method, one figure per (model_type, training_data, freeze_shapes).

    The no-adaptation baseline only exists at num_target_shots = 0 for
    non-exnihilo training data, so it typically shows up as a single point
    rather than a trend.
    """
    if not _check_metrics_ds(metrics_ds, "domain adaptation comparison"):
        return
    out_dir = Path(figure_dir) / "comparison" / "domain_adaptation_comparison"
    for model_type in _coord_values(metrics_ds, "model_type"):
        for td in _coord_values(metrics_ds, "training_data"):
            for freeze in _coord_values(metrics_ds, "freeze_shapes"):
                sub = _mask_select(
                    metrics_ds,
                    (metrics_ds["model_type"] == model_type)
                    & (metrics_ds["training_data"] == td)
                    & (metrics_ds["freeze_shapes"] == freeze),
                )
                if sub.sizes.get("case_idx", 0) == 0:
                    continue
                fig, axes = _grid_figure()
                drew = False
                for da in _coord_values(sub, "domain_adaptation"):
                    da_sub = _mask_select(sub, sub["domain_adaptation"] == da)
                    color = DA_COLORS.get(da, "white")
                    label = DA_LABELS.get(da, da)
                    for row, metric in enumerate(METRIC_NAMES):
                        for col, stage in enumerate(STAGE_AGG_NAMES):
                            drew |= _plot_series(axes[row, col], da_sub, stage, metric, color, label)
                if not drew:
                    plt.close(fig)
                    continue
                model_label = MODEL_LABELS.get(model_type, model_type)
                title = f"Domain adaptation comparison - {model_label} / train: {td} / freeze shapes: {freeze}"
                _finalize_grid(fig, axes, sub, title, out_dir / f"{model_type}.td_{td}.freeze_{freeze}.png")
    logger.info(f"Saved domain adaptation comparison figures to {out_dir}")


def freeze_shapes_comparison(metrics_ds: xr.Dataset, figure_dir: Path):
    """Effect of freezing shapes: (frozen - unfrozen) metric difference vs
    num_target_shots, one line per (training_data, domain_adaptation), one
    figure per model type that has shapes. Positive difference means the
    unfrozen model is better."""
    if not _check_metrics_ds(metrics_ds, "freeze shapes comparison"):
        return
    out_dir = Path(figure_dir) / "comparison" / "freeze_shapes_comparison"
    td_colors = _training_data_colors(_coord_values(metrics_ds, "training_data"))
    da_linestyles = {"none": ":", "mixing": "-", "transfer": "--", "transfer_pretrain": "-."}

    for model_type in _coord_values(metrics_ds, "model_type"):
        if model_type not in MODELS_WITH_SHAPES:
            continue
        model_sub = _mask_select(metrics_ds, metrics_ds["model_type"] == model_type)
        fig, axes = _grid_figure()
        drew = False
        for td in _coord_values(model_sub, "training_data"):
            for da in _coord_values(model_sub, "domain_adaptation"):
                combo_mask = (model_sub["training_data"] == td) & (model_sub["domain_adaptation"] == da)
                frozen = _mask_select(model_sub, combo_mask & model_sub["freeze_shapes"])
                unfrozen = _mask_select(model_sub, combo_mask & ~model_sub["freeze_shapes"])
                if frozen.sizes.get("case_idx", 0) == 0 or unfrozen.sizes.get("case_idx", 0) == 0:
                    continue

                frozen_shots = np.atleast_1d(frozen["num_target_shots"].values)
                unfrozen_shots = np.atleast_1d(unfrozen["num_target_shots"].values)
                shared = sorted(set(frozen_shots.tolist()) & set(unfrozen_shots.tolist()))
                shared_line = [s for s in shared if s >= 0]

                color = td_colors[td]
                linestyle = da_linestyles.get(da, "-")
                label = f"{td} / {DA_LABELS.get(da, da)}"

                for row, metric in enumerate(METRIC_NAMES):
                    for col, stage in enumerate(STAGE_AGG_NAMES):
                        ax = axes[row, col]
                        frozen_means = np.atleast_1d(frozen.sel(stage=stage)[f"{metric}_mean"].values).astype(float)
                        unfrozen_means = np.atleast_1d(unfrozen.sel(stage=stage)[f"{metric}_mean"].values).astype(float)
                        frozen_by_shot = dict(zip(frozen_shots.tolist(), frozen_means.tolist(), strict=True))
                        unfrozen_by_shot = dict(zip(unfrozen_shots.tolist(), unfrozen_means.tolist(), strict=True))

                        if shared_line:
                            diffs = np.array([_freeze_diff(frozen_by_shot, unfrozen_by_shot, s) for s in shared_line])
                            if np.isfinite(diffs).any():
                                ax.plot(
                                    shared_line,
                                    diffs,
                                    color=color,
                                    linestyle=linestyle,
                                    linewidth=1.6,
                                    marker="o",
                                    markersize=4,
                                    label=label,
                                )
                                drew = True
                        if -1 in shared:
                            ref = _freeze_diff(frozen_by_shot, unfrozen_by_shot, -1)
                            if np.isfinite(ref):
                                ax.axhline(ref, color=color, linestyle="--", linewidth=1.2, alpha=0.6)
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
        _finalize_grid(fig, axes, model_sub, title, out_dir / f"{model_type}.png")
    logger.info(f"Saved freeze shapes comparison figures to {out_dir}")
