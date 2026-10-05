"""Per-case reports of profile transfer study results (its ANALYSIS_REPORTS_MODULE, see orchestration.case_reports).

For every finished case:
- A PDF of the 10 best and 10 worst test timeslices by the combined metric,
  each page a 2x2 panel of predicted vs measured Te / ne profiles and their
  gradients, with the GP-fit error bars.
- GIFs of the predicted profile evolution against the most recent fresh
  measurement for the best / median / worst test shot by shot-mean combined
  metric, one frame per fresh timeslice with the same 2x2 layout.

Plus the TORAX-specific relaxation figure: for the best-performing torax case
and its best timeslice, how the TORAX solve relaxes the profiles into the
measured target shape.
"""

import io
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loguru import logger
from PIL import Image

from transport_study import EPISODE_DIM, RADIAL_DIM, TIME_DIM
from transport_study.modules.trb_utils import CHI_GRAD_RHO_MAX
from transport_study.orchestration.case_reports import TITLE_FONTSIZE, best_worst_pdf
from transport_study.plot_style import (
    BACKGROUND_COLOR,
    LABEL_FONTSIZE,
    LEGEND_STYLE,
    TEXT_COLOR,
    TICK_FONTSIZE,
    style_axis,
)
from transport_study.profile_transfer.plot_torax_evolution import (
    load_timeslice,
    plot_relaxation,
)
from transport_study.profile_transfer.study_metrics import (
    CaseTimesliceMetrics,
    compute_case_timeslice_metrics,
    load_eval_dataset,
)

GIF_FRAME_DURATION_MS = 200
REPORT_FILENAME = "best_worst_timeslices.pdf"

VALUE_LABELS = {
    "t_e_keV": r"$T_e$ [keV]",
    "n_e_1e20": r"$n_e$ [$10^{20}$ m$^{-3}$]",
}
GRAD_LABELS = {
    "t_e_keV": r"$dT_e/d\rho$ [keV]",
    "n_e_1e20": r"$dn_e/d\rho$ [$10^{20}$ m$^{-3}$]",
}
# Panel order: Te value, ne value, Te gradient, ne gradient
PANEL_VARS = ("t_e_keV", "n_e_1e20")


def _axis_lims(*arrays: np.ndarray) -> tuple[float, float]:
    lo = min(np.nanmin(a) for a in arrays if np.isfinite(a).any())
    hi = max(np.nanmax(a) for a in arrays if np.isfinite(a).any())
    pad = 0.05 * max(hi - lo, 1e-6)
    return (lo - pad, hi + pad)


def _shade_ignored_grad_region(ax, rho_mid: np.ndarray):
    """Transparent red span over the rho region the gradient chi metric excludes (rho >= CHI_GRAD_RHO_MAX)."""
    ax.axvspan(CHI_GRAD_RHO_MAX, rho_mid[-1], color="red", alpha=0.12, linewidth=0, zorder=0)


def _record_title(ts_metrics: CaseTimesliceMetrics, record_idx: int) -> str:
    stage = ts_metrics.stage[record_idx] or "unknown stage"
    if stage == "flattop":
        stage = "flattop (aux)" if ts_metrics.aux_heated[record_idx] else "flattop (ohmic)"
    return (
        f"shot {ts_metrics.shot[record_idx]} @ t={ts_metrics.time[record_idx]:.3f}s - {stage}\n"
        f"value={ts_metrics.metric_value[record_idx]:.3f}  "
        f"grad={ts_metrics.metric_grad[record_idx]:.3f}  "
        f"combined={ts_metrics.metric_combined[record_idx]:.3f}"
    )


def _timeslice_panel(
    result_ds: xr.Dataset,
    ts_metrics: CaseTimesliceMetrics,
    record_idx: int,
    title_prefix: str = "",
    ylims: dict | None = None,
) -> plt.Figure:
    """2x2 predicted vs measured panel for one test timeslice.

    Top row: Te and ne profiles, measured with GP-fit error band plus prediction.
    Bottom row: their rho gradients, measured GP gradient with error band plus
    finite-difference gradient of the prediction at the rho midpoints.
    """
    shot = ts_metrics.shot[record_idx]
    device = ts_metrics.ds_source[record_idx]
    shot_res = result_ds.sel({EPISODE_DIM: shot}).isel({TIME_DIM: int(ts_metrics.result_time_idx[record_idx])})
    shot_eval = load_eval_dataset(device).sel({EPISODE_DIM: shot}).isel({TIME_DIM: int(ts_metrics.eval_time_idx[record_idx])})

    rho = result_ds[RADIAL_DIM].values
    rho_mid = 0.5 * (rho[:-1] + rho[1:])
    d_rho = np.diff(rho)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    for col, var in enumerate(PANEL_VARS):
        targ = shot_res[f"{var}_targ"].values
        pred = shot_res[f"{var}_pred"].values
        err = shot_eval[f"{var}_error"].values
        grad_targ = shot_eval[f"{var}_gradient"].values
        grad_err = shot_eval[f"{var}_gradient_error"].values
        grad_targ_mid = 0.5 * (grad_targ[:-1] + grad_targ[1:])
        grad_err_mid = 0.5 * (grad_err[:-1] + grad_err[1:])
        grad_pred = np.diff(pred) / d_rho

        ax_val = axes[0, col]
        style_axis(ax_val)
        ax_val.fill_between(rho, targ - err, targ + err, color="white", alpha=0.25, linewidth=0)
        ax_val.plot(rho, targ, color="white", linewidth=2, linestyle="--", label="Measured")
        ax_val.plot(rho, pred, color="#0095ff", linewidth=2, label="Predicted")
        ax_val.set_ylabel(VALUE_LABELS[var], color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
        if ylims and f"{var}_value" in ylims:
            ax_val.set_ylim(*ylims[f"{var}_value"])

        ax_grad = axes[1, col]
        style_axis(ax_grad)
        _shade_ignored_grad_region(ax_grad, rho_mid)
        ax_grad.fill_between(rho_mid, grad_targ_mid - grad_err_mid, grad_targ_mid + grad_err_mid, color="white", alpha=0.25, linewidth=0)
        ax_grad.plot(rho_mid, grad_targ_mid, color="white", linewidth=2, linestyle="--", label="Measured")
        ax_grad.plot(rho_mid, grad_pred, color="#0095ff", linewidth=2, label="Predicted")
        ax_grad.set_ylabel(GRAD_LABELS[var], color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
        ax_grad.set_xlabel(r"$\rho_{tor,N}$", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
        if ylims and f"{var}_gradient" in ylims:
            ax_grad.set_ylim(*ylims[f"{var}_gradient"])
        else:
            grad_rho_mask = rho_mid < CHI_GRAD_RHO_MAX
            ax_grad.set_ylim(
                *_axis_lims(
                    (grad_targ_mid - grad_err_mid)[grad_rho_mask],
                    (grad_targ_mid + grad_err_mid)[grad_rho_mask],
                    grad_pred[grad_rho_mask],
                )
            )

    axes[0, 0].legend(fontsize=TICK_FONTSIZE, loc="upper right", **LEGEND_STYLE)
    fig.suptitle(f"{title_prefix}{_record_title(ts_metrics, record_idx)}", color=TEXT_COLOR, fontsize=TITLE_FONTSIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def _shot_ylims(result_ds: xr.Dataset, ts_metrics: CaseTimesliceMetrics, record_idxs: np.ndarray) -> dict:
    """Fixed axis limits over one shot's frames so the GIF does not jump around.

    Gradient y-lims are computed only over rho < CHI_GRAD_RHO_MAX, the region the gradient metric counts,
    so a noisy edge gradient outside it does not blow out the axis scale.
    """
    ylims = {}
    rho = result_ds[RADIAL_DIM].values
    d_rho = np.diff(rho)
    rho_mid = 0.5 * (rho[:-1] + rho[1:])
    grad_rho_mask = rho_mid < CHI_GRAD_RHO_MAX
    shot = ts_metrics.shot[record_idxs[0]]
    device = ts_metrics.ds_source[record_idxs[0]]
    shot_res = result_ds.sel({EPISODE_DIM: shot})
    shot_eval = load_eval_dataset(device).sel({EPISODE_DIM: shot})
    res_idxs = ts_metrics.result_time_idx[record_idxs]
    eval_idxs = ts_metrics.eval_time_idx[record_idxs]

    for var in PANEL_VARS:
        targ = shot_res[f"{var}_targ"].transpose(TIME_DIM, RADIAL_DIM).values[res_idxs]
        pred = shot_res[f"{var}_pred"].transpose(TIME_DIM, RADIAL_DIM).values[res_idxs]
        err = shot_eval[f"{var}_error"].transpose(TIME_DIM, RADIAL_DIM).values[eval_idxs]
        grad_targ = shot_eval[f"{var}_gradient"].transpose(TIME_DIM, RADIAL_DIM).values[eval_idxs]
        grad_err = shot_eval[f"{var}_gradient_error"].transpose(TIME_DIM, RADIAL_DIM).values[eval_idxs]
        # Measured gradients live on the full rho grid, average to the rho
        # midpoints where the panel plots them so the mask lines up
        grad_targ_mid = 0.5 * (grad_targ[:, :-1] + grad_targ[:, 1:])
        grad_err_mid = 0.5 * (grad_err[:, :-1] + grad_err[:, 1:])
        grad_pred = np.diff(pred, axis=-1) / d_rho

        ylims[f"{var}_value"] = _axis_lims(targ - err, targ + err, pred)
        ylims[f"{var}_gradient"] = _axis_lims(
            (grad_targ_mid - grad_err_mid)[:, grad_rho_mask],
            (grad_targ_mid + grad_err_mid)[:, grad_rho_mask],
            grad_pred[:, grad_rho_mask],
        )
    return ylims


def _fig_to_image(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _select_gif_shots(ts_metrics: CaseTimesliceMetrics) -> list:
    """Best, median, and worst test shots by shot-mean combined metric."""
    shots = np.unique(ts_metrics.shot)

    def _shot_mean(shot) -> float:
        values = ts_metrics.metric_combined[ts_metrics.shot == shot]
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if len(values) else np.nan

    shot_means = np.array([_shot_mean(shot) for shot in shots])
    finite = np.flatnonzero(np.isfinite(shot_means))
    if len(finite) == 0:
        return []
    order = finite[np.argsort(shot_means[finite])]
    picks = [order[0], order[len(order) // 2], order[-1]]
    return list(dict.fromkeys(shots[i] for i in picks))


def evolution_gifs(result_ds: xr.Dataset, ts_metrics: CaseTimesliceMetrics, case_dir: Path):
    """Predicted profile evolution vs the fresh measurements for the best /
    median / worst test shot. Each frame is the 2x2 timeslice panel at one
    fresh timeslice, with axis limits fixed over the shot."""
    for shot in _select_gif_shots(ts_metrics):
        record_idxs = np.flatnonzero(ts_metrics.shot == shot)
        record_idxs = record_idxs[np.argsort(ts_metrics.time[record_idxs])]
        if len(record_idxs) == 0:
            continue
        ylims = _shot_ylims(result_ds, ts_metrics, record_idxs)
        frames = [
            _fig_to_image(
                _timeslice_panel(result_ds, ts_metrics, record_idx, title_prefix=f"frame {i + 1}/{len(record_idxs)} - ", ylims=ylims)
            )
            for i, record_idx in enumerate(record_idxs)
        ]
        gif_path = case_dir / f"shot_{shot}_evolution.gif"
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=GIF_FRAME_DURATION_MS,
            loop=0,
        )
        logger.info(f"Saved profile evolution GIF to {gif_path}")


def case_report_done(case_dir: Path) -> bool:
    return (case_dir / REPORT_FILENAME).exists() and any(case_dir.glob("shot_*_evolution.gif"))


def render_case_report(result_ds: xr.Dataset, ts_metrics: CaseTimesliceMetrics, case_dir: Path):
    """The best and worst test timeslices by the combined metric, then the profile evolution GIFs."""
    timeslice_page = partial(_timeslice_panel, result_ds, ts_metrics)
    record_idxs = np.arange(len(ts_metrics))
    best_worst_pdf(record_idxs, ts_metrics.metric_combined, timeslice_page, case_dir / REPORT_FILENAME)
    evolution_gifs(result_ds, ts_metrics, case_dir)


def torax_relaxation_report(study, metrics_ds: xr.Dataset, figure_dir: Path):
    """Relaxation figure for the best torax case at its best timeslice.

    Picks the torax-* case with the lowest combined-metric mean over all test
    timeslices, restores its trained module, reruns the TORAX solve step by
    step on the timeslice where the fit is best, and plots the profile
    relaxation into the measured target shape.
    """
    if not metrics_ds.data_vars or "case_idx" not in metrics_ds.dims:
        logger.info("No stage-resolved metrics available, skipping TORAX relaxation report")
        return

    model_types = np.atleast_1d(metrics_ds["model_type"].values).astype(str)
    torax_mask = np.char.startswith(model_types, "torax-")
    if not torax_mask.any():
        logger.info("No finished torax cases, skipping TORAX relaxation report")
        return

    torax_ds = metrics_ds.isel(case_idx=torax_mask).sel(stage="all")
    combined = np.atleast_1d(torax_ds["combined_mean"].values).astype(float)
    if not np.isfinite(combined).any():
        logger.info("No finite combined metrics for torax cases, skipping TORAX relaxation report")
        return
    best_pos = int(np.nanargmin(combined))
    best_case_idx = int(np.atleast_1d(torax_ds["case_idx"].values)[best_pos])
    case = study.cases[best_case_idx]
    logger.info(f"Best torax case by combined metric: {case}")

    result_ds = xr.load_dataset(study.result_path(case))
    train_config = study.make_train_config(case)
    ts_metrics = compute_case_timeslice_metrics(result_ds, train_config.loss_config)
    finite_records = np.flatnonzero(np.isfinite(ts_metrics.metric_combined))
    if len(finite_records) == 0:
        logger.warning(f"No valid test timeslices for torax case {case}, skipping relaxation report")
        return
    best_record = int(finite_records[np.argmin(ts_metrics.metric_combined[finite_records])])

    shot = ts_metrics.shot[best_record]
    device = ts_metrics.ds_source[best_record]
    time_s = float(ts_metrics.time[best_record])
    eval_time_idx = int(ts_metrics.eval_time_idx[best_record])

    transport_model = case.model_type.removeprefix("torax-")
    plot_path = Path(figure_dir) / "torax" / f"relaxation_{case}_shot{shot}_t{time_s:.3f}.png"
    # Skip the module restore and TORAX solve when the figure for this exact
    # case and timeslice is already on disk. If new results shift the best
    # case or timeslice the path changes and the figure regenerates
    if plot_path.exists():
        logger.info(f"TORAX relaxation figure already exists at {plot_path}, skipping")
        return

    # Function-level import: restore_predictor imports profile_study, which
    # imports this module at run_study time, so a top-level import would cycle
    from transport_study.profile_transfer.restore_predictor import (
        restore_profile_predictor,
    )

    try:
        # The result files keep the store's time_idx indexing (rows are never dropped)
        timeslice = load_timeslice(device, shot, eval_time_idx)
        module = restore_profile_predictor(train_config)
        steps, coeffs = module.evolve(timeslice)
    except Exception:
        logger.exception(f"Failed to load, restore or evolve torax module for case {case}, skipping relaxation report")
        return
    logger.info(f"TORAX relaxation recorded {len(steps)} states (initial + {len(steps) - 1} steps)")

    plot_relaxation(
        steps,
        coeffs,
        timeslice,
        transport_model,
        title_context=f"{case}\nshot {shot} @ t={time_s:.3f}s (best combined-metric timeslice)",
        plot_path=plot_path,
    )
