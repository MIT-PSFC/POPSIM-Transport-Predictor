"""Per-case reports of transport transfer study results (its ANALYSIS_REPORTS_MODULE, see orchestration.case_reports).

The power balance study's shot PDF with a transport-specific page:
the best and worst holdout shots by TIME-AVERAGED relative error,
each page the measured vs predicted Te and ne profile evolution as (time, rho) maps,
plus the per-timeslice error traces with the discharge stages shaded and aux-heated spans marked.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from transport_study import EPISODE_DIM, RADIAL_DIM, TIME_DIM
from transport_study.orchestration.case_reports import TITLE_FONTSIZE, shade_stages
from transport_study.plot_style import (
    BACKGROUND_COLOR,
    LABEL_FONTSIZE,
    LEGEND_STYLE,
    TEXT_COLOR,
    TICK_FONTSIZE,
    style_axis,
)
from transport_study.power_balance_transfer.case_reports import (  # noqa: F401 case_report_done is part of the ANALYSIS_REPORTS_MODULE contract
    REPORT_FILENAME,
    case_report_done,
    shot_pdf,
    shot_records,
    shot_title,
)
from transport_study.power_balance_transfer.study_metrics import CaseTimesliceMetrics

PROFILE_CMAPS = {"t_e_keV": "magma", "n_e_1e20": "viridis"}
PROFILE_LABELS = {"t_e_keV": "Te [keV]", "n_e_1e20": "ne [1e20 m^-3]"}


def _profile_map(ax, times: np.ndarray, rho: np.ndarray, values: np.ndarray, cmap: str, vmin: float, vmax: float):
    """One (time, rho) profile evolution map on a styled axis."""
    style_axis(ax)
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
    rho = shot_res[RADIAL_DIM].values
    valid = np.isfinite(res_time)
    for signal in PROFILE_CMAPS:
        valid &= np.isfinite(shot_res[f"{signal}_targ"].transpose(TIME_DIM, RADIAL_DIM).values).any(axis=-1)

    records = shot_records(ts_metrics, shot)
    record_time = ts_metrics.time[records]
    device = str(ts_metrics.ds_source[records[0]]) if len(records) else str(shot_res["ds_source"].values)

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    for row, signal in enumerate(PROFILE_CMAPS):
        targ = shot_res[f"{signal}_targ"].transpose(TIME_DIM, RADIAL_DIM).values[valid]
        pred = shot_res[f"{signal}_pred"].transpose(TIME_DIM, RADIAL_DIM).values[valid]
        finite_targ = targ[np.isfinite(targ)]
        vmin = float(finite_targ.min()) if len(finite_targ) else 0.0
        vmax = float(finite_targ.max()) if len(finite_targ) else 1.0
        cmap = PROFILE_CMAPS[signal]
        _profile_map(axes[row, 0], res_time[valid], rho, targ, cmap, vmin, vmax)
        im = _profile_map(axes[row, 1], res_time[valid], rho, pred, cmap, vmin, vmax)
        axes[row, 0].set_ylabel(r"$\rho_{tor,N}$", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
        axes[row, 0].set_title(f"Measured {PROFILE_LABELS[signal]}", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
        axes[row, 1].set_title(f"Predicted {PROFILE_LABELS[signal]}", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
        cbar = fig.colorbar(im, ax=axes[row, :].tolist(), pad=0.02)
        cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=TICK_FONTSIZE)

    # Error traces on the left, the right slot repeats them on a log scale
    # (relative errors span orders of magnitude across a discharge)
    for ax_err, log_scale in ((axes[2, 0], False), (axes[2, 1], True)):
        style_axis(ax_err)
        ax_err.plot(record_time, ts_metrics.err_abs[records], color="#0095ff", linewidth=1.5, label="Abs error (rho integral)")
        ax_err.plot(record_time, ts_metrics.err_rel[records], color="#ff60ec", linewidth=1.5, label="Rel error (rho integral)")
        ax_err.set_xlabel("Time [s]", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
        if log_scale:
            ax_err.set_yscale("log")
        else:
            ax_err.set_ylabel("Error", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)

    # Stage shading and aux-heated spans on the error panels
    stage_handles = shade_stages((axes[2, 0], axes[2, 1]), axes[2, 0], ts_metrics, records)
    handles, _ = axes[2, 0].get_legend_handles_labels()
    axes[2, 0].legend(handles=handles + stage_handles, fontsize=TICK_FONTSIZE, loc="upper right", **LEGEND_STYLE)

    title = shot_title(ts_metrics, records, shot, device, title_prefix)
    fig.suptitle(title, color=TEXT_COLOR, fontsize=TITLE_FONTSIZE)
    return fig


def render_case_report(result_ds: xr.Dataset, ts_metrics: CaseTimesliceMetrics, case_dir: Path):
    # The power_balance / p_oh / p_rad prereq cases write scalar result files, which get the power balance page
    if all(f"{signal}_targ" in result_ds for signal in PROFILE_CMAPS):
        shot_pdf(result_ds, ts_metrics, case_dir / REPORT_FILENAME, page_fn=_shot_page)
    else:
        shot_pdf(result_ds, ts_metrics, case_dir / REPORT_FILENAME)
