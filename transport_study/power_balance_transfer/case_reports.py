"""Per-case reports of power balance transfer study results (its ANALYSIS_REPORTS_MODULE, see orchestration.case_reports).

For every finished case: a PDF of the best and worst holdout shots by TIME-AVERAGED relative error.
The per-shot errors in the result files are raw time integrals, which penalize long shots,
the time-averaged form removes that duration confound.
Each page shows the predicted vs measured trajectory with the discharge stages shaded,
plus the per-timeslice error traces with the aux-heated spans marked.
"""

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from transport_study import EPISODE_DIM
from transport_study.orchestration.case_reports import (
    TITLE_FONTSIZE,
    best_worst_pdf,
    shade_stages,
)
from transport_study.plot_style import (
    BACKGROUND_COLOR,
    LABEL_FONTSIZE,
    LEGEND_STYLE,
    TEXT_COLOR,
    TICK_FONTSIZE,
    style_axis,
)
from transport_study.power_balance_transfer.study_metrics import (
    CaseTimesliceMetrics,
    shot_time_averaged_errors,
)

REPORT_FILENAME = "best_worst_shots.pdf"


def _result_signal(result_ds: xr.Dataset) -> str:
    """Base name of the predicted signal in a result file, e.g. energy_mhd_MJ for the
    full power balance cases or power_ohm_MW / power_radiated_MW for the submodule cases."""
    for name in result_ds.data_vars:
        if str(name).endswith("_targ"):
            return str(name)[: -len("_targ")]
    raise KeyError(f"No *_targ variable in result file, found {list(result_ds.data_vars)}")


def shot_records(ts_metrics: CaseTimesliceMetrics, shot) -> np.ndarray:
    """Positions of one shot's records, sorted by time."""
    records = np.flatnonzero(ts_metrics.shot == shot)
    return records[np.argsort(ts_metrics.time[records])]


def shot_title(ts_metrics: CaseTimesliceMetrics, records: np.ndarray, shot, device: str, title_prefix: str, unit: str = "") -> str:
    """Page title with the time-averaged errors and the span of one shot's records."""
    finite_rel = ts_metrics.err_rel[records][np.isfinite(ts_metrics.err_rel[records])]
    finite_abs = ts_metrics.err_abs[records][np.isfinite(ts_metrics.err_abs[records])]
    avg_rel = float(np.mean(finite_rel)) if len(finite_rel) else np.nan
    avg_abs = float(np.mean(finite_abs)) if len(finite_abs) else np.nan
    record_time = ts_metrics.time[records]
    duration = float(record_time[-1] - record_time[0]) if len(record_time) > 1 else 0.0
    abs_unit = f" {unit}" if unit else ""
    return (
        f"{title_prefix}shot {shot} ({device})\n"
        f"time-avg rel err={avg_rel:.3f}  time-avg abs err={avg_abs:.3f}{abs_unit}  "
        f"{len(records)} timeslices over {duration:.2f} s"
    )


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

    records = shot_records(ts_metrics, shot)
    record_time = ts_metrics.time[records]
    device = str(ts_metrics.ds_source[records[0]]) if len(records) else str(shot_res["ds_source"].values)

    fig, (ax_traj, ax_err) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    style_axis(ax_traj)
    ax_traj.plot(res_time[valid], targ[valid], color="white", linewidth=2, linestyle="--", label="Measured")
    ax_traj.plot(res_time[valid], pred[valid], color="#0095ff", linewidth=2, label="Predicted")
    ax_traj.set_ylabel(f"{signal.rsplit('_', 1)[0]} [{unit}]", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)

    style_axis(ax_err)
    ax_err.plot(record_time, ts_metrics.err_abs[records], color="#0095ff", linewidth=1.5, label=f"Abs error [{unit}]")
    ax_err.plot(record_time, ts_metrics.err_rel[records], color="#ff60ec", linewidth=1.5, label="Rel error")
    ax_err.set_ylabel("Error", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
    ax_err.set_xlabel("Time [s]", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)

    # Stage shading on both panels, aux-heated spans on the error panel only
    stage_handles = shade_stages((ax_traj, ax_err), ax_err, ts_metrics, records)
    handles, _ = ax_traj.get_legend_handles_labels()
    ax_traj.legend(handles=handles + stage_handles, fontsize=TICK_FONTSIZE, loc="upper right", **LEGEND_STYLE)
    ax_err.legend(fontsize=TICK_FONTSIZE, loc="upper right", **LEGEND_STYLE)

    title = shot_title(ts_metrics, records, shot, device, title_prefix, unit)
    fig.suptitle(title, color=TEXT_COLOR, fontsize=TITLE_FONTSIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def shot_pdf(result_ds: xr.Dataset, ts_metrics: CaseTimesliceMetrics, pdf_path: Path, page_fn=_shot_page):
    """Best and worst holdout shot pages by time-averaged relative error.

    page_fn(result_ds, ts_metrics, shot, title_prefix=...) renders one shot,
    the transport study passes its profile-evolution page.
    """
    shots, _avg_abs, avg_rel, _n_ts = shot_time_averaged_errors(ts_metrics)
    shot_page = partial(page_fn, result_ds, ts_metrics)
    best_worst_pdf(shots, avg_rel, shot_page, pdf_path)


def case_report_done(case_dir: Path) -> bool:
    return (case_dir / REPORT_FILENAME).exists()


def render_case_report(result_ds: xr.Dataset, ts_metrics: CaseTimesliceMetrics, case_dir: Path):
    shot_pdf(result_ds, ts_metrics, case_dir / REPORT_FILENAME)
