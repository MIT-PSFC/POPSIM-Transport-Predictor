"""Per-case reports of transport transfer study results (its ANALYSIS_REPORTS_MODULE, see orchestration.case_reports).

The power balance study's shot PDF with a transport-specific page:
the best and worst holdout shots by the best checkpoint's TIME-AVERAGED chi (the validation loss),
each page the per-timeslice chi over the shot with the discharge stages shaded,
aux-heated spans and diverged timeslices marked.
The predicted profiles are left out: their (time, rho) maps took most of a report's rendering time.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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
from transport_study.transport_transfer.study_metrics import (
    METRIC_VARS,
    CaseTimesliceMetrics,
)

# Chi traces of a page: metric name, label, color, line width
CHI_TRACES = (
    ("combined", "Chi (value + gradient)", "#0095ff", 2.0),
    ("value", "Value chi", "#8dff36", 1.0),
    ("grad", "Gradient chi", "#ffb347", 1.0),
)
DIVERGED_COLOR = "#ff4d4d"


def _shot_page(
    result_ds: xr.Dataset,
    ts_metrics: CaseTimesliceMetrics,
    shot,
    title_prefix: str = "",
) -> plt.Figure:
    """Chi page for one holdout shot, the best checkpoint's.

    The combined chi and its value and gradient parts over time on a log scale
    (chi spans orders of magnitude across a discharge), each a line through the fresh timeslices,
    with the shot stages shaded, the aux-heated spans along the bottom and the diverged timeslices as red lines.
    """
    records = shot_records(ts_metrics, shot)
    record_time = ts_metrics.time[records]
    device = str(ts_metrics.ds_source[records[0]])

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    style_axis(ax)
    # Chi is NaN at the stale timeslices, a line through the finite ones only keeps it visible
    for name, label, color, linewidth in CHI_TRACES:
        values = ts_metrics.best(name)[records]
        mask_finite = np.isfinite(values) & (values > 0)
        ax.plot(record_time[mask_finite], values[mask_finite], color=color, linewidth=linewidth, marker=".", markersize=3, label=label)
    for diverged_time in record_time[ts_metrics.best("diverged")[records] == 1]:
        ax.axvline(diverged_time, color=DIVERGED_COLOR, linewidth=0.8, alpha=0.6, zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("Time [s]", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Chi", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)

    stage_handles = shade_stages((ax,), ax, ts_metrics, records)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + stage_handles, fontsize=TICK_FONTSIZE, loc="upper right", **LEGEND_STYLE)

    title = shot_title(ts_metrics, records, shot, device, title_prefix, title_metrics=(("combined", "chi"), ("rel", "rel err")))
    fig.suptitle(title, color=TEXT_COLOR, fontsize=TITLE_FONTSIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def render_case_report(result_ds: xr.Dataset, ts_metrics: CaseTimesliceMetrics, case_dir: Path):
    # The power_balance / p_oh / p_rad prereq cases write scalar result files without chi, which get the power balance page
    if METRIC_VARS["combined"] in result_ds:
        shot_pdf(result_ds, ts_metrics, case_dir / REPORT_FILENAME, page_fn=_shot_page, rank_metric="combined")
    else:
        shot_pdf(result_ds, ts_metrics, case_dir / REPORT_FILENAME)
