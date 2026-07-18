"""Per-case deep-dive reports for power balance transfer study results.

For every finished case: a PDF of the 10 best and 10 worst holdout shots
ranked by TIME-AVERAGED relative error (the per-shot errors in the result
files are raw time integrals, which penalize long shots; the time-averaged
form removes that duration confound). Each page shows the predicted vs
measured stored energy trajectory with the discharge stages (rampup /
flattop / rampdown) shaded, plus the per-timeslice error traces with the
aux-heated spans marked.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loguru import logger
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from transport_study import EPISODE_DIM
from transport_study.plot_style import BACKGROUND_COLOR, TEXT_COLOR, style_axis
from transport_study.power_balance_transfer.study_metrics import (
    CaseTimesliceMetrics,
    case_metrics_path,
    compute_case_timeslice_metrics,
    shot_time_averaged_errors,
)

N_BEST_WORST = 10

LABEL_FONTSIZE = 11
TICK_FONTSIZE = 9
TITLE_FONTSIZE = 12

STAGE_SHADE_COLORS = {
    "rampup": "#4d94ff",
    "flattop": "#8dff36",
    "rampdown": "#ff4d4d",
}
AUX_SHADE_COLOR = "#ffb347"


def _result_signal(result_ds: xr.Dataset) -> str:
    """Base name of the predicted signal in a result file, e.g. Wtot_MJ for the
    full power balance cases or P_oh_MW / P_rad_MW for the submodule cases."""
    for name in result_ds.data_vars:
        if str(name).endswith("_targ"):
            return str(name)[: -len("_targ")]
    raise KeyError(f"No *_targ variable in result file, found {list(result_ds.data_vars)}")


def _stage_spans(times: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """(start, end) time spans of the contiguous True runs of mask."""
    spans: list[tuple[float, float]] = []
    idxs = np.flatnonzero(mask)
    if len(idxs) == 0:
        return spans
    breaks = np.flatnonzero(np.diff(idxs) > 1)
    starts = np.concatenate(([idxs[0]], idxs[breaks + 1]))
    ends = np.concatenate((idxs[breaks], [idxs[-1]]))
    for s, e in zip(starts, ends, strict=True):
        spans.append((float(times[s]), float(times[e])))
    return spans


def _shot_page(
    result_ds: xr.Dataset,
    ts_metrics: CaseTimesliceMetrics,
    shot,
    title_prefix: str = "",
) -> plt.Figure:
    """Two-panel trajectory page for one holdout shot.

    Top: measured vs predicted signal (Wtot, or the submodule power) over time
    with the shot stages shaded.
    Bottom: per-timeslice absolute and relative error with aux-heated spans.
    """
    signal = _result_signal(result_ds)
    unit = signal.rsplit("_", 1)[-1]
    shot_res = result_ds.sel({EPISODE_DIM: shot})
    res_time = shot_res["time"].values
    targ = shot_res[f"{signal}_targ"].values
    pred = shot_res[f"{signal}_pred"].values
    valid = np.isfinite(res_time) & np.isfinite(targ)

    rec = np.flatnonzero(ts_metrics.shot == shot)
    rec = rec[np.argsort(ts_metrics.time[rec])]
    rec_time = ts_metrics.time[rec]
    device = str(ts_metrics.ds_source[rec[0]]) if len(rec) else str(shot_res["ds_source"].values)

    fig, (ax_traj, ax_err) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    style_axis(ax_traj, TICK_FONTSIZE)
    ax_traj.plot(res_time[valid], targ[valid], color="white", linewidth=2, linestyle="--", label="Measured")
    ax_traj.plot(res_time[valid], pred[valid], color="#0095ff", linewidth=2, label="Predicted")
    ax_traj.set_ylabel(f"{signal.rsplit('_', 1)[0]} [{unit}]", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)

    style_axis(ax_err, TICK_FONTSIZE)
    ax_err.plot(rec_time, ts_metrics.err_abs[rec], color="#0095ff", linewidth=1.5, label=f"Abs error [{unit}]")
    ax_err.plot(rec_time, ts_metrics.err_rel[rec], color="#ff60ec", linewidth=1.5, label="Rel error")
    ax_err.set_ylabel("Error", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
    ax_err.set_xlabel("Time [s]", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)

    # Stage shading on both panels, aux-heated spans on the error panel only
    stage_handles = []
    for stage, color in STAGE_SHADE_COLORS.items():
        spans = _stage_spans(rec_time, ts_metrics.stage[rec] == stage)
        for start, end in spans:
            ax_traj.axvspan(start, end, color=color, alpha=0.10, linewidth=0, zorder=0)
            ax_err.axvspan(start, end, color=color, alpha=0.10, linewidth=0, zorder=0)
        if spans:
            stage_handles.append(Patch(facecolor=color, alpha=0.35, label=stage))
    aux_spans = _stage_spans(rec_time, ts_metrics.aux_heated[rec])
    for start, end in aux_spans:
        ax_err.axvspan(start, end, ymin=0.0, ymax=0.05, color=AUX_SHADE_COLOR, alpha=0.6, linewidth=0, zorder=0)
    if aux_spans:
        stage_handles.append(Patch(facecolor=AUX_SHADE_COLOR, alpha=0.6, label="aux heated"))

    handles, _ = ax_traj.get_legend_handles_labels()
    ax_traj.legend(
        handles=handles + stage_handles,
        fontsize=TICK_FONTSIZE,
        labelcolor=TEXT_COLOR,
        facecolor=BACKGROUND_COLOR,
        edgecolor=TEXT_COLOR,
        loc="upper right",
    )
    ax_err.legend(
        fontsize=TICK_FONTSIZE,
        labelcolor=TEXT_COLOR,
        facecolor=BACKGROUND_COLOR,
        edgecolor=TEXT_COLOR,
        loc="upper right",
    )

    finite_rel = ts_metrics.err_rel[rec][np.isfinite(ts_metrics.err_rel[rec])]
    finite_abs = ts_metrics.err_abs[rec][np.isfinite(ts_metrics.err_abs[rec])]
    avg_rel = float(np.mean(finite_rel)) if len(finite_rel) else np.nan
    avg_abs = float(np.mean(finite_abs)) if len(finite_abs) else np.nan
    duration = float(rec_time[-1] - rec_time[0]) if len(rec_time) > 1 else 0.0
    fig.suptitle(
        f"{title_prefix}shot {shot} ({device})\n"
        f"time-avg rel err={avg_rel:.3f}  time-avg abs err={avg_abs:.3f} {unit}  "
        f"{len(rec)} timeslices over {duration:.2f} s",
        color=TEXT_COLOR,
        fontsize=TITLE_FONTSIZE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def best_worst_pdf(result_ds: xr.Dataset, ts_metrics: CaseTimesliceMetrics, pdf_path: Path, page_fn=None):
    """One PDF per case: the N_BEST_WORST best pages then the N_BEST_WORST
    worst pages, holdout shots ranked by time-averaged relative error.

    page_fn renders one shot's page (defaults to this study's Wtot trajectory
    page); the transport study passes its profile-evolution page instead."""
    if page_fn is None:
        page_fn = _shot_page
    shots, _avg_abs, avg_rel, _n_ts = shot_time_averaged_errors(ts_metrics)
    finite = np.flatnonzero(np.isfinite(avg_rel))
    if len(finite) == 0:
        logger.warning(f"No finite time-averaged errors, skipping {pdf_path}")
        return
    order = finite[np.argsort(avg_rel[finite])]
    best = order[:N_BEST_WORST]
    worst = order[::-1][:N_BEST_WORST]

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for rank, shot_idx in enumerate(best, start=1):
            fig = page_fn(result_ds, ts_metrics, shots[shot_idx], title_prefix=f"BEST #{rank} - ")
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)
        for rank, shot_idx in enumerate(worst, start=1):
            fig = page_fn(result_ds, ts_metrics, shots[shot_idx], title_prefix=f"WORST #{rank} - ")
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)
    logger.info(f"Saved best/worst shot PDF to {pdf_path}")


def generate_case_reports(study, figure_dir: Path):
    """Best/worst shot PDFs for every finished case. Existing case report
    directories are left alone; clean_figures wipes the figure dir to force
    regeneration."""
    for case in study.cases:
        generate_case_report(study, case, figure_dir)


def _case_report_dir(figure_dir: Path, case) -> Path:
    return Path(figure_dir) / "case_reports" / str(case)


def _case_report_done(case_dir: Path) -> bool:
    return (case_dir / "best_worst_shots.pdf").exists()


def generate_case_report(study, case, figure_dir: Path):
    """Best/worst shot PDF for one case. No-op when the case has no result
    file or the report already exists."""
    result_path = study.result_path(case)
    if not result_path.exists():
        return
    case_dir = _case_report_dir(figure_dir, case)
    if _case_report_done(case_dir):
        logger.info(f"Case report already exists for {case}, skipping")
        return

    result_ds = xr.load_dataset(result_path)
    ts_metrics = compute_case_timeslice_metrics(result_ds)
    if len(ts_metrics) == 0:
        logger.warning(f"No valid test timeslices for case {case}, skipping case report")
        return

    best_worst_pdf(result_ds, ts_metrics, case_dir / "best_worst_shots.pdf")


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
