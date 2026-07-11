"""Per-case deep-dive reports for profile transfer study results.

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
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loguru import logger
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from transport_study import EPISODE_DIM, TIME_DIM
from transport_study.config import config
from transport_study.orchestration.slurm_utils import (
    get_running_job_names,
    launch_profile_analysis_parallel,
)
from transport_study.profile_transfer.plot_torax_evolution import plot_relaxation
from transport_study.profile_transfer.plotting import (
    BACKGROUND_COLOR,
    FACE_COLOR,
    TEXT_COLOR,
)
from transport_study.profile_transfer.study_metrics import (
    CaseTimesliceMetrics,
    case_metrics_path,
    compute_case_timeslice_metrics,
    load_eval_dataset,
)

N_BEST_WORST = 10
GIF_FRAME_DURATION_MS = 200

# Parallel analysis orchestration: resubmission cap per case and how often the
# driver rechecks for finished cases
ANALYSIS_MAX_ATTEMPTS = 3
ANALYSIS_POLL_INTERVAL_S = 30

LABEL_FONTSIZE = 11
TICK_FONTSIZE = 9
TITLE_FONTSIZE = 12

VALUE_LABELS = {
    "Te_keV_rho": r"$T_e$ [keV]",
    "ne20_rho": r"$n_e$ [$10^{20}$ m$^{-3}$]",
}
GRAD_LABELS = {
    "Te_keV_rho": r"$dT_e/d\rho$ [keV]",
    "ne20_rho": r"$dn_e/d\rho$ [$10^{20}$ m$^{-3}$]",
}
# Panel order: Te value, ne value, Te gradient, ne gradient
PANEL_VARS = ("Te_keV_rho", "ne20_rho")


def _style_axis(ax):
    ax.set_facecolor(FACE_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=TICK_FONTSIZE)
    for spine in ax.spines.values():
        spine.set_edgecolor(TEXT_COLOR)
    ax.grid(True, alpha=0.2, color=TEXT_COLOR)


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

    rho = result_ds["rho"].values
    rho_mid = 0.5 * (rho[:-1] + rho[1:])
    d_rho = np.diff(rho)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    for col, var in enumerate(PANEL_VARS):
        targ = shot_res[f"{var}_targ"].values
        pred = shot_res[f"{var}_pred"].values
        err = shot_eval[f"{var}_error"].values
        grad_targ = shot_eval[f"{var}_grad"].values
        grad_err = shot_eval[f"{var}_grad_error"].values
        grad_targ_mid = 0.5 * (grad_targ[:-1] + grad_targ[1:])
        grad_err_mid = 0.5 * (grad_err[:-1] + grad_err[1:])
        grad_pred = np.diff(pred) / d_rho

        ax_val = axes[0, col]
        _style_axis(ax_val)
        ax_val.fill_between(rho, targ - err, targ + err, color="white", alpha=0.25, linewidth=0)
        ax_val.plot(rho, targ, color="white", linewidth=2, linestyle="--", label="Measured")
        ax_val.plot(rho, pred, color="#0095ff", linewidth=2, label="Predicted")
        ax_val.set_ylabel(VALUE_LABELS[var], color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
        if ylims and f"{var}_value" in ylims:
            ax_val.set_ylim(*ylims[f"{var}_value"])

        ax_grad = axes[1, col]
        _style_axis(ax_grad)
        ax_grad.fill_between(rho_mid, grad_targ_mid - grad_err_mid, grad_targ_mid + grad_err_mid, color="white", alpha=0.25, linewidth=0)
        ax_grad.plot(rho_mid, grad_targ_mid, color="white", linewidth=2, linestyle="--", label="Measured")
        ax_grad.plot(rho_mid, grad_pred, color="#0095ff", linewidth=2, label="Predicted")
        ax_grad.set_ylabel(GRAD_LABELS[var], color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
        ax_grad.set_xlabel(r"$\rho$", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
        if ylims and f"{var}_grad" in ylims:
            ax_grad.set_ylim(*ylims[f"{var}_grad"])

    axes[0, 0].legend(
        fontsize=TICK_FONTSIZE,
        labelcolor=TEXT_COLOR,
        facecolor=BACKGROUND_COLOR,
        edgecolor=TEXT_COLOR,
        loc="upper right",
    )
    fig.suptitle(f"{title_prefix}{_record_title(ts_metrics, record_idx)}", color=TEXT_COLOR, fontsize=TITLE_FONTSIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def _shot_ylims(result_ds: xr.Dataset, ts_metrics: CaseTimesliceMetrics, record_idxs: np.ndarray) -> dict:
    """Fixed axis limits over one shot's frames so the GIF does not jump around."""
    ylims = {}
    rho = result_ds["rho"].values
    d_rho = np.diff(rho)
    shot = ts_metrics.shot[record_idxs[0]]
    device = ts_metrics.ds_source[record_idxs[0]]
    shot_res = result_ds.sel({EPISODE_DIM: shot})
    shot_eval = load_eval_dataset(device).sel({EPISODE_DIM: shot})
    res_idxs = ts_metrics.result_time_idx[record_idxs]
    eval_idxs = ts_metrics.eval_time_idx[record_idxs]

    for var in PANEL_VARS:
        targ = shot_res[f"{var}_targ"].transpose(TIME_DIM, "rho").values[res_idxs]
        pred = shot_res[f"{var}_pred"].transpose(TIME_DIM, "rho").values[res_idxs]
        err = shot_eval[f"{var}_error"].transpose(TIME_DIM, "rho").values[eval_idxs]
        grad_targ = shot_eval[f"{var}_grad"].transpose(TIME_DIM, "rho").values[eval_idxs]
        grad_err = shot_eval[f"{var}_grad_error"].transpose(TIME_DIM, "rho").values[eval_idxs]
        grad_pred = np.diff(pred, axis=-1) / d_rho

        def _lims(*arrays):
            lo = min(np.nanmin(a) for a in arrays if np.isfinite(a).any())
            hi = max(np.nanmax(a) for a in arrays if np.isfinite(a).any())
            pad = 0.05 * max(hi - lo, 1e-6)
            return (lo - pad, hi + pad)

        ylims[f"{var}_value"] = _lims(targ - err, targ + err, pred)
        ylims[f"{var}_grad"] = _lims(grad_targ - grad_err, grad_targ + grad_err, grad_pred)
    return ylims


def _fig_to_image(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def best_worst_pdf(result_ds: xr.Dataset, ts_metrics: CaseTimesliceMetrics, pdf_path: Path):
    """One PDF per case: the N_BEST_WORST best pages then the N_BEST_WORST
    worst pages, ranked by the per-timeslice combined metric."""
    finite = np.flatnonzero(np.isfinite(ts_metrics.metric_combined))
    if len(finite) == 0:
        logger.warning(f"No finite combined metrics, skipping {pdf_path}")
        return
    order = finite[np.argsort(ts_metrics.metric_combined[finite])]
    best = order[:N_BEST_WORST]
    worst = order[::-1][:N_BEST_WORST]

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for rank, record_idx in enumerate(best, start=1):
            fig = _timeslice_panel(result_ds, ts_metrics, record_idx, title_prefix=f"BEST #{rank} - ")
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)
        for rank, record_idx in enumerate(worst, start=1):
            fig = _timeslice_panel(result_ds, ts_metrics, record_idx, title_prefix=f"WORST #{rank} - ")
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)
    logger.info(f"Saved best/worst timeslice PDF to {pdf_path}")


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


def generate_case_reports(study, figure_dir: Path):
    """Best/worst timeslice PDFs and profile evolution GIFs for every finished
    case. Existing case report directories are left alone (GIF rendering is
    slow); clean_figures wipes the figure dir to force regeneration."""
    for case in study.cases:
        generate_case_report(study, case, figure_dir)


def _case_report_dir(figure_dir: Path, case) -> Path:
    return Path(figure_dir) / "case_reports" / str(case)


def _case_report_done(case_dir: Path) -> bool:
    return (case_dir / "best_worst_timeslices.pdf").exists() and any(case_dir.glob("shot_*_evolution.gif"))


def generate_case_report(study, case, figure_dir: Path):
    """Best/worst timeslice PDF and profile evolution GIFs for one case.
    No-op when the case has no result file or the report already exists."""
    result_path = study.result_path(case)
    if not result_path.exists():
        return
    case_dir = _case_report_dir(figure_dir, case)
    if _case_report_done(case_dir):
        logger.info(f"Case report already exists for {case}, skipping")
        return

    result_ds = xr.load_dataset(result_path)
    if "rho" not in result_ds.dims:
        logger.warning(f"Result file for case {case} predates the rho grid schema, skipping case report")
        return

    loss_config = study.make_train_config(case).loss_config
    ts_metrics = compute_case_timeslice_metrics(result_ds, loss_config)
    if len(ts_metrics) == 0:
        logger.warning(f"No valid test timeslices for case {case}, skipping case report")
        return

    best_worst_pdf(result_ds, ts_metrics, case_dir / "best_worst_timeslices.pdf")
    evolution_gifs(result_ds, ts_metrics, case_dir)


def analysis_case_done(study, case, figure_dir: Path) -> bool:
    """Whether a case needs no more analysis work: its metrics cache exists and
    either it is the empty 'nothing valid' marker or the case report is on disk."""
    cache_path = case_metrics_path(study, case)
    if not cache_path.exists():
        return False
    case_metrics = xr.load_dataset(cache_path)
    if not case_metrics.data_vars:
        return True
    return _case_report_done(_case_report_dir(figure_dir, case))


def run_analysis_parallel(study):
    """Fan the per-case analysis (stage metrics + case report) out over SLURM.

    Submits CPU jobs for finished cases that still need analysis (see
    launch_profile_analysis_parallel, which targets config.analysis_partition)
    and polls until every case is done or has exhausted its attempts. At most
    config.max_analysis_jobs analysis jobs are in the queue (running or
    pending) at once, so a study with hundreds of cases does not flood the
    scheduler; more jobs are submitted as earlier ones finish. Cases that
    exhaust their attempts fall back to the serial path in collect_metrics /
    generate_case_reports afterwards.
    """
    partition = config.analysis_partition or config.partition
    pending = [case for case in study.cases if study.result_path(case).exists() and not analysis_case_done(study, case, study.figure_dir)]
    if not pending:
        return
    logger.info(
        f"Launching parallel analysis for {len(pending)} cases on partition {partition} (at most {config.max_analysis_jobs} jobs at once)"
    )

    attempts: dict[str, int] = {}
    while pending:
        running_job_names = get_running_job_names(partition)
        if running_job_names is None:
            logger.warning("Could not query SLURM job state, waiting before trying again...")
            time.sleep(ANALYSIS_POLL_INTERVAL_S)
            continue

        in_flight = [case for case in pending if study.analysis_job_name(case) in running_job_names]
        launchable = [
            case
            for case in pending
            if study.analysis_job_name(case) not in running_job_names and attempts.get(str(case), 0) < ANALYSIS_MAX_ATTEMPTS
        ]
        capacity = max(config.max_analysis_jobs - len(in_flight), 0)
        for case in launchable[:capacity]:
            attempts[str(case)] = attempts.get(str(case), 0) + 1
            launch_profile_analysis_parallel(study, case)
            in_flight.append(case)

        if not in_flight:
            logger.error(
                f"{len(pending)} analysis cases did not finish after {ANALYSIS_MAX_ATTEMPTS} attempts each, "
                "they will be computed serially instead"
            )
            break
        logger.info(f"{len(pending)} analysis cases remain ({len(in_flight)} jobs in flight)")

        time.sleep(ANALYSIS_POLL_INTERVAL_S)
        pending = [case for case in pending if not analysis_case_done(study, case, study.figure_dir)]

    if not pending:
        logger.info("Parallel analysis finished for all cases")


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

    # The raw device dataset carries the scalar input vars the module needs.
    # load_eval_dataset keeps the same time_idx indexing (rows are never dropped)
    timeslice = xr.open_dataset(config.dataset_paths[device]).sel({EPISODE_DIM: shot}).isel({TIME_DIM: eval_time_idx})

    # Function-level import: restore_predictor imports profile_study, which
    # imports this module at run_study time, so a top-level import would cycle
    from transport_study.profile_transfer.restore_predictor import (
        restore_profile_predictor,
    )

    try:
        module = restore_profile_predictor(train_config)
        steps, coeffs = module.evolve(timeslice)
    except Exception:
        logger.exception(f"Failed to restore or evolve torax module for case {case}, skipping relaxation report")
        return
    logger.info(f"TORAX relaxation recorded {len(steps)} states (initial + {len(steps) - 1} steps)")

    transport_model = case.model_type.removeprefix("torax-")
    plot_path = Path(figure_dir) / "torax" / f"relaxation_{case}_shot{shot}_t{time_s:.3f}.png"
    plot_relaxation(
        steps,
        coeffs,
        timeslice,
        transport_model,
        title_context=f"{case}\nshot {shot} @ t={time_s:.3f}s (best combined-metric timeslice)",
        plot_path=plot_path,
    )
