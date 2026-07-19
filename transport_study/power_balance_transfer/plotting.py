"""Case-comparison figures for power balance transfer study results.

All figures share one layout: a 2x2 grid with one row per error kind
(absolute / relative) and one column per error domain (per-shot time-integrated
/ per-timeslice), with the number of target-device shots included in training
on a semilog x axis. Each family draws one line per member of the comparison
(model type, training dataset, data normalization, or domain adaptation
method) and one figure per combination of the remaining case dimensions.

Consumes the per-case scalar summary dataset written by
PowerBalanceStudy.collect_results (collected_results.nc): dims case_idx,
coords model_type / training_data / data_normalization / domain_adaptation /
freeze_submodules / num_target_shots, data vars err_E_D_S.

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

TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 11
TICK_FONTSIZE = 9
LEGEND_FONTSIZE = 10

# Rows of the comparison grid: error kind
METRIC_NAMES = ("err_abs", "err_rel")
# Columns of the comparison grid: error domain
DOMAIN_NAMES = ("shot", "ts")

METRIC_LABELS = {
    "err_abs": "Absolute error",
    "err_rel": "Relative error",
}

DOMAIN_LABELS = {
    "shot": "Per shot (time-integrated)",
    "ts": "Per timeslice",
}

MODEL_COLORS = {
    "scaling_law": "#8dff36",
    "sciml": "#0095ff",
    "unstructured_nn": "#ff4d4d",
    "transformer": "#ffb347",
}

MODEL_LABELS = {
    "scaling_law": "Scaling law",
    "sciml": "SciML",
    "unstructured_nn": "Unstructured NN",
    "transformer": "Transformer",
}

NORM_COLORS = {
    "raw": "#ff4d4d",
    "physics": "#8dff36",
    "z_score": "#0095ff",
    "coral": "#ff60ec",
    "physics-coral": "#ffb52e",
}

NORM_LABELS = {
    "raw": "Raw",
    "physics": "Physics",
    "z_score": "Z-score",
    "coral": "CORAL",
    "physics-coral": "Physics CORAL",
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

# Submodule prereq cases predict P_oh / P_rad, not Wtot, so their errors are
# not comparable to the main models and are excluded from every comparison
SUBMODULE_MODEL_TYPES = ("p_oh", "p_rad")

# Metric value above which a case is treated as diverged and masked out
DIVERGED_THRESHOLD = 1e3


def mask_select(ds: xr.Dataset, mask) -> xr.Dataset:
    return ds.isel(case_idx=np.asarray(mask))


def grid_figure() -> tuple[plt.Figure, np.ndarray]:
    n_rows, n_cols = len(METRIC_NAMES), len(DOMAIN_NAMES)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 3.4 * n_rows),
        sharex=True,
        squeeze=False,
    )
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    for row, metric in enumerate(METRIC_NAMES):
        for col, domain in enumerate(DOMAIN_NAMES):
            ax = axes[row, col]
            style_axis(ax, TICK_FONTSIZE)
            if row == 0:
                ax.set_title(DOMAIN_LABELS[domain], color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric], color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
            if row == len(METRIC_NAMES) - 1:
                ax.set_xlabel("Target shots in training", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
    return fig, axes


def plot_series(ax, sub: xr.Dataset, metric: str, domain: str, color, label: str, linestyle: str = "-") -> bool:
    """One comparison member on one axis: errorbar line over num_target_shots >= 0,
    plus a dashed horizontal reference for the -1 (all target shots) sentinel.
    Returns whether anything was drawn."""
    shots = np.atleast_1d(sub["num_target_shots"].values)
    means = np.atleast_1d(sub[f"{metric}_{domain}_mean"].values).astype(float).copy()
    stds = np.atleast_1d(sub[f"{metric}_{domain}_std"].values).astype(float).copy()

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


def finalize_grid(fig, axes, sub: xr.Dataset, title: str, save_path: Path):
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


def coord_values(ds: xr.Dataset, name: str) -> list:
    return sorted(set(np.atleast_1d(ds[name].values).tolist()))


def _model_types(ds: xr.Dataset) -> list:
    return [mt for mt in coord_values(ds, "model_type") if mt not in SUBMODULE_MODEL_TYPES]


def check_results_ds(results_ds: xr.Dataset, family: str) -> bool:
    if not results_ds.data_vars or "case_idx" not in results_ds.dims:
        logger.warning(f"No collected results available, skipping {family} figures")
        return False
    return True


def model_comparison(results_ds: xr.Dataset, figure_dir: Path):
    """One line per model type, one figure per (training_data, data_normalization,
    domain_adaptation, freeze_submodules)."""
    if not check_results_ds(results_ds, "model comparison"):
        return
    out_dir = Path(figure_dir) / "comparison" / "model_comparison"
    for td in coord_values(results_ds, "training_data"):
        for dn in coord_values(results_ds, "data_normalization"):
            for da in coord_values(results_ds, "domain_adaptation"):
                for freeze in coord_values(results_ds, "freeze_submodules"):
                    sub = mask_select(
                        results_ds,
                        (results_ds["training_data"] == td)
                        & (results_ds["data_normalization"] == dn)
                        & (results_ds["domain_adaptation"] == da)
                        & (results_ds["freeze_submodules"] == freeze),
                    )
                    if sub.sizes.get("case_idx", 0) == 0:
                        continue
                    fig, axes = grid_figure()
                    drew = False
                    for model_type in _model_types(sub):
                        model_sub = mask_select(sub, sub["model_type"] == model_type)
                        color = MODEL_COLORS.get(model_type, "white")
                        label = MODEL_LABELS.get(model_type, model_type)
                        for row, metric in enumerate(METRIC_NAMES):
                            for col, domain in enumerate(DOMAIN_NAMES):
                                drew |= plot_series(axes[row, col], model_sub, metric, domain, color, label)
                    if not drew:
                        plt.close(fig)
                        continue
                    title = f"Model comparison - train: {td} / norm: {NORM_LABELS.get(dn, dn)} / DA: {da} / freeze submodules: {freeze}"
                    finalize_grid(fig, axes, sub, title, out_dir / f"td_{td}.dn_{dn}.da_{da}.freeze_{freeze}.png")
    logger.info(f"Saved model comparison figures to {out_dir}")


def training_dataset_comparison(results_ds: xr.Dataset, figure_dir: Path):
    """One line per training dataset, one figure per (model_type, data_normalization,
    domain_adaptation, freeze_submodules)."""
    if not check_results_ds(results_ds, "training dataset comparison"):
        return
    out_dir = Path(figure_dir) / "comparison" / "training_dataset_comparison"
    training_datasets = coord_values(results_ds, "training_data")
    cmap = plt.colormaps["tab10"].resampled(max(len(training_datasets), 1))
    td_colors = {td: cmap(i) for i, td in enumerate(training_datasets)}
    for model_type in _model_types(results_ds):
        for dn in coord_values(results_ds, "data_normalization"):
            for da in coord_values(results_ds, "domain_adaptation"):
                for freeze in coord_values(results_ds, "freeze_submodules"):
                    sub = mask_select(
                        results_ds,
                        (results_ds["model_type"] == model_type)
                        & (results_ds["data_normalization"] == dn)
                        & (results_ds["domain_adaptation"] == da)
                        & (results_ds["freeze_submodules"] == freeze),
                    )
                    if sub.sizes.get("case_idx", 0) == 0:
                        continue
                    fig, axes = grid_figure()
                    drew = False
                    for td in coord_values(sub, "training_data"):
                        td_sub = mask_select(sub, sub["training_data"] == td)
                        for row, metric in enumerate(METRIC_NAMES):
                            for col, domain in enumerate(DOMAIN_NAMES):
                                drew |= plot_series(axes[row, col], td_sub, metric, domain, td_colors[td], str(td))
                    if not drew:
                        plt.close(fig)
                        continue
                    model_label = MODEL_LABELS.get(model_type, model_type)
                    title = f"Training dataset comparison - {model_label} / norm: {NORM_LABELS.get(dn, dn)} / DA: {da} / freeze submodules: {freeze}"
                    finalize_grid(fig, axes, sub, title, out_dir / f"{model_type}.dn_{dn}.da_{da}.freeze_{freeze}.png")
    logger.info(f"Saved training dataset comparison figures to {out_dir}")


def data_normalization_comparison(results_ds: xr.Dataset, figure_dir: Path):
    """One line per data normalization method, one figure per (model_type,
    training_data, domain_adaptation, freeze_submodules)."""
    if not check_results_ds(results_ds, "data normalization comparison"):
        return
    out_dir = Path(figure_dir) / "comparison" / "data_normalization_comparison"
    for model_type in _model_types(results_ds):
        for td in coord_values(results_ds, "training_data"):
            for da in coord_values(results_ds, "domain_adaptation"):
                for freeze in coord_values(results_ds, "freeze_submodules"):
                    sub = mask_select(
                        results_ds,
                        (results_ds["model_type"] == model_type)
                        & (results_ds["training_data"] == td)
                        & (results_ds["domain_adaptation"] == da)
                        & (results_ds["freeze_submodules"] == freeze),
                    )
                    if sub.sizes.get("case_idx", 0) == 0:
                        continue
                    fig, axes = grid_figure()
                    drew = False
                    for dn in coord_values(sub, "data_normalization"):
                        dn_sub = mask_select(sub, sub["data_normalization"] == dn)
                        color = NORM_COLORS.get(dn, "white")
                        label = NORM_LABELS.get(dn, dn)
                        for row, metric in enumerate(METRIC_NAMES):
                            for col, domain in enumerate(DOMAIN_NAMES):
                                drew |= plot_series(axes[row, col], dn_sub, metric, domain, color, label)
                    if not drew:
                        plt.close(fig)
                        continue
                    model_label = MODEL_LABELS.get(model_type, model_type)
                    title = f"Data normalization comparison - {model_label} / train: {td} / DA: {da} / freeze submodules: {freeze}"
                    finalize_grid(fig, axes, sub, title, out_dir / f"{model_type}.td_{td}.da_{da}.freeze_{freeze}.png")
    logger.info(f"Saved data normalization comparison figures to {out_dir}")


def domain_adaptation_comparison(results_ds: xr.Dataset, figure_dir: Path):
    """One line per domain adaptation method, one figure per (model_type,
    training_data, data_normalization, freeze_submodules).

    The no-adaptation baseline only exists at num_target_shots = 0 for
    non-exnihilo training data, so it typically shows up as a single point
    rather than a trend.
    """
    if not check_results_ds(results_ds, "domain adaptation comparison"):
        return
    out_dir = Path(figure_dir) / "comparison" / "domain_adaptation_comparison"
    for model_type in _model_types(results_ds):
        for td in coord_values(results_ds, "training_data"):
            for dn in coord_values(results_ds, "data_normalization"):
                for freeze in coord_values(results_ds, "freeze_submodules"):
                    sub = mask_select(
                        results_ds,
                        (results_ds["model_type"] == model_type)
                        & (results_ds["training_data"] == td)
                        & (results_ds["data_normalization"] == dn)
                        & (results_ds["freeze_submodules"] == freeze),
                    )
                    if sub.sizes.get("case_idx", 0) == 0:
                        continue
                    fig, axes = grid_figure()
                    drew = False
                    for da in coord_values(sub, "domain_adaptation"):
                        da_sub = mask_select(sub, sub["domain_adaptation"] == da)
                        color = DA_COLORS.get(da, "white")
                        label = DA_LABELS.get(da, da)
                        for row, metric in enumerate(METRIC_NAMES):
                            for col, domain in enumerate(DOMAIN_NAMES):
                                drew |= plot_series(axes[row, col], da_sub, metric, domain, color, label)
                    if not drew:
                        plt.close(fig)
                        continue
                    model_label = MODEL_LABELS.get(model_type, model_type)
                    title = f"Domain adaptation comparison - {model_label} / train: {td} / norm: {NORM_LABELS.get(dn, dn)} / freeze submodules: {freeze}"
                    finalize_grid(fig, axes, sub, title, out_dir / f"{model_type}.td_{td}.dn_{dn}.freeze_{freeze}.png")
    logger.info(f"Saved domain adaptation comparison figures to {out_dir}")
