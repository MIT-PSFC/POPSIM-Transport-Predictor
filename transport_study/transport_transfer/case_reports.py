"""Per-case deep-dive reports for transport transfer study results.

For every finished case: a PDF of the 10 best and 10 worst holdout shots
ranked by TIME-AVERAGED relative error (the per-shot errors in the result
files are raw time integrals, which penalize long shots; the time-averaged
form removes that duration confound). Each page shows the measured vs
predicted Te and ne profile evolution as (time, rho) maps, plus the
per-timeslice error traces with the discharge stages shaded and aux-heated
spans marked.

The ranking, PDF assembly, and completion logic are the power balance study's
(fully generic, see best_worst_pdf's page_fn hook); only the per-shot page is
transport-specific.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loguru import logger
from matplotlib.patches import Patch

from transport_study import EPISODE_DIM, TIME_DIM
from transport_study.plot_style import BACKGROUND_COLOR, TEXT_COLOR, style_axis
from transport_study.power_balance_transfer.case_reports import (  # noqa: F401 re-exported for the ANALYSIS_REPORTS_MODULE contract
    AUX_SHADE_COLOR,
    LABEL_FONTSIZE,
    STAGE_SHADE_COLORS,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    analysis_case_done,
    best_worst_pdf,
    case_report_dir,
    case_report_done,
    stage_spans,
)
from transport_study.transport_transfer.study_metrics import (
    CaseTimesliceMetrics,
    compute_case_timeslice_metrics,
)

PROFILE_CMAPS = {"Te_keV_rho": "magma", "ne20_rho": "viridis"}
PROFILE_LABELS = {"Te_keV_rho": "Te [keV]", "ne20_rho": "ne [1e20 m^-3]"}


def _profile_map(ax, times: np.ndarray, rho: np.ndarray, values: np.ndarray, cmap: str, vmin: float, vmax: float):
    """One (time, rho) profile evolution map on a styled axis."""
    style_axis(ax, TICK_FONTSIZE)
    return ax.pcolormesh(times, rho, values.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")


def _shot_page(
    result_ds: xr.Dataset,
    ts_metrics: CaseTimesliceMetrics,
    shot,
    title_prefix: str = "",
) -> plt.Figure:
    """Profile-evolution page for one holdout shot.

    Rows 1-2: measured vs predicted Te and ne over (time, rho), each channel
    on a shared color scale taken from the measurement.
    Row 3: per-timeslice combined absolute and relative error with the shot
    stages shaded and aux-heated spans marked.
    """
    shot_res = result_ds.sel({EPISODE_DIM: shot})
    res_time = shot_res["time"].values
    rho = shot_res["rho"].values
    valid = np.isfinite(res_time)
    for signal in PROFILE_CMAPS:
        valid &= np.isfinite(shot_res[f"{signal}_targ"].transpose(TIME_DIM, "rho").values).any(axis=-1)

    rec = np.flatnonzero(ts_metrics.shot == shot)
    rec = rec[np.argsort(ts_metrics.time[rec])]
    rec_time = ts_metrics.time[rec]
    device = str(ts_metrics.ds_source[rec[0]]) if len(rec) else str(shot_res["ds_source"].values)

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    for row, signal in enumerate(PROFILE_CMAPS):
        targ = shot_res[f"{signal}_targ"].transpose(TIME_DIM, "rho").values[valid]
        pred = shot_res[f"{signal}_pred"].transpose(TIME_DIM, "rho").values[valid]
        finite_targ = targ[np.isfinite(targ)]
        vmin = float(finite_targ.min()) if len(finite_targ) else 0.0
        vmax = float(finite_targ.max()) if len(finite_targ) else 1.0
        cmap = PROFILE_CMAPS[signal]
        _profile_map(axes[row, 0], res_time[valid], rho, targ, cmap, vmin, vmax)
        im = _profile_map(axes[row, 1], res_time[valid], rho, pred, cmap, vmin, vmax)
        axes[row, 0].set_ylabel("rho", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
        axes[row, 0].set_title(f"Measured {PROFILE_LABELS[signal]}", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
        axes[row, 1].set_title(f"Predicted {PROFILE_LABELS[signal]}", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
        cbar = fig.colorbar(im, ax=axes[row, :].tolist(), pad=0.02)
        cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=TICK_FONTSIZE)

    # Error traces on the left, the right slot repeats them on a log scale
    # (relative errors span orders of magnitude across a discharge)
    for ax_err, log_scale in ((axes[2, 0], False), (axes[2, 1], True)):
        style_axis(ax_err, TICK_FONTSIZE)
        ax_err.plot(rec_time, ts_metrics.err_abs[rec], color="#0095ff", linewidth=1.5, label="Abs error (rho integral)")
        ax_err.plot(rec_time, ts_metrics.err_rel[rec], color="#ff60ec", linewidth=1.5, label="Rel error (rho integral)")
        ax_err.set_xlabel("Time [s]", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
        if log_scale:
            ax_err.set_yscale("log")
        else:
            ax_err.set_ylabel("Error", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)

    # Stage shading and aux-heated spans on the error panels
    stage_handles = []
    for stage, color in STAGE_SHADE_COLORS.items():
        spans = stage_spans(rec_time, ts_metrics.stage[rec] == stage)
        for start, end in spans:
            axes[2, 0].axvspan(start, end, color=color, alpha=0.10, linewidth=0, zorder=0)
            axes[2, 1].axvspan(start, end, color=color, alpha=0.10, linewidth=0, zorder=0)
        if spans:
            stage_handles.append(Patch(facecolor=color, alpha=0.35, label=stage))
    aux_spans = stage_spans(rec_time, ts_metrics.aux_heated[rec])
    for start, end in aux_spans:
        axes[2, 0].axvspan(start, end, ymin=0.0, ymax=0.05, color=AUX_SHADE_COLOR, alpha=0.6, linewidth=0, zorder=0)
    if aux_spans:
        stage_handles.append(Patch(facecolor=AUX_SHADE_COLOR, alpha=0.6, label="aux heated"))

    handles, _ = axes[2, 0].get_legend_handles_labels()
    axes[2, 0].legend(
        handles=handles + stage_handles,
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
        f"time-avg rel err={avg_rel:.3f}  time-avg abs err={avg_abs:.3f}  "
        f"{len(rec)} timeslices over {duration:.2f} s",
        color=TEXT_COLOR,
        fontsize=TITLE_FONTSIZE,
    )
    return fig


def generate_case_reports(study, figure_dir: Path):
    """Best/worst shot PDFs for every finished case. Existing case report
    directories are left alone; clean_figures wipes the figure dir to force
    regeneration."""
    for case in study.cases:
        generate_case_report(study, case, figure_dir)


def generate_case_report(study, case, figure_dir: Path):
    """Best/worst shot PDF for one case. No-op when the case has no result
    file or the report already exists."""
    result_path = study.result_path(case)
    if not result_path.exists():
        return
    case_dir = case_report_dir(figure_dir, case)
    if case_report_done(case_dir):
        logger.info(f"Case report already exists for {case}, skipping")
        return

    result_ds = xr.load_dataset(result_path)
    ts_metrics = compute_case_timeslice_metrics(result_ds)
    if len(ts_metrics) == 0:
        logger.warning(f"No valid test timeslices for case {case}, skipping case report")
        return

    best_worst_pdf(result_ds, ts_metrics, case_dir / "best_worst_shots.pdf", page_fn=_shot_page)
