"""Per-case report driver and page pieces shared by every study.

Each study's ANALYSIS_REPORTS_MODULE exports two names:
    case_report_done(case_dir): whether the case's report files are on disk
    render_case_report(result_ds, ts_metrics, case_dir): write them
Everything else lives here:
the skip and no-result checks, the timeslice scoring through orchestration.case_metrics,
the best / worst PDF assembly and the stage shading of the trajectory pages.
"""

from collections.abc import Callable, Sequence
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loguru import logger
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from transport_study.orchestration.case_metrics import (
    StagedTimesliceMetrics,
    case_metrics_path,
    case_timeslice_metrics,
)
from transport_study.orchestration.study import Study

N_BEST_WORST = 10
TITLE_FONTSIZE = 12

STAGE_SHADE_COLORS = {
    "rampup": "#4d94ff",
    "flattop": "#8dff36",
    "rampdown": "#ff4d4d",
}
AUX_SHADE_COLOR = "#ffb347"


def stage_spans(times: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
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


def shade_stages(stage_axes: Sequence, aux_ax, ts_metrics: StagedTimesliceMetrics, records: np.ndarray) -> list[Patch]:
    """Shade the shot stages of time-sorted records on stage_axes and the aux-heated spans along the bottom of aux_ax.

    Returns the legend handles of the shaded stages and of the aux-heated marker.
    """
    record_time = ts_metrics.time[records]
    stage_handles = []
    for stage, color in STAGE_SHADE_COLORS.items():
        spans = stage_spans(record_time, ts_metrics.stage[records] == stage)
        for start, end in spans:
            for ax in stage_axes:
                ax.axvspan(start, end, color=color, alpha=0.10, linewidth=0, zorder=0)
        if spans:
            stage_handles.append(Patch(facecolor=color, alpha=0.35, label=stage))
    aux_spans = stage_spans(record_time, ts_metrics.aux_heated[records])
    for start, end in aux_spans:
        aux_ax.axvspan(start, end, ymin=0.0, ymax=0.05, color=AUX_SHADE_COLOR, alpha=0.6, linewidth=0, zorder=0)
    if aux_spans:
        stage_handles.append(Patch(facecolor=AUX_SHADE_COLOR, alpha=0.6, label="aux heated"))
    return stage_handles


def best_worst_pdf(items: np.ndarray, scores: np.ndarray, page_fn: Callable[..., plt.Figure], pdf_path: Path):
    """The N_BEST_WORST lowest-score pages then the N_BEST_WORST highest-score pages in one PDF.

    page_fn(item, title_prefix=...) renders one item's page, items with a non-finite score are left out.
    """
    finite = np.flatnonzero(np.isfinite(scores))
    if len(finite) == 0:
        logger.warning(f"No finite scores, skipping {pdf_path}")
        return
    order = finite[np.argsort(scores[finite])]
    best = order[:N_BEST_WORST]
    worst = order[::-1][:N_BEST_WORST]

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for label, ranked in (("BEST", best), ("WORST", worst)):
            for rank, item_idx in enumerate(ranked, start=1):
                fig = page_fn(items[item_idx], title_prefix=f"{label} #{rank} - ")
                pdf.savefig(fig, facecolor=fig.get_facecolor())
                plt.close(fig)
    logger.info(f"Saved best/worst PDF to {pdf_path}")


def case_report_dir(figure_dir: Path, case) -> Path:
    return Path(figure_dir) / "case_reports" / str(case)


def generate_case_report(study: Study, case, figure_dir: Path):
    """The study's report for one case, a no-op without a result file or when the report is already on disk."""
    result_path = study.result_path(case)
    if not result_path.exists():
        return
    reports_module = import_module(study.ANALYSIS_REPORTS_MODULE)
    case_dir = case_report_dir(figure_dir, case)
    if reports_module.case_report_done(case_dir):
        logger.info(f"Case report already exists for {case}, skipping")
        return

    result_ds = xr.load_dataset(result_path)
    ts_metrics = case_timeslice_metrics(study, case, result_ds)
    if len(ts_metrics) == 0:
        logger.warning(f"No valid test timeslices for case {case}, skipping case report")
        return
    reports_module.render_case_report(result_ds, ts_metrics, case_dir)


def generate_case_reports(study: Study, figure_dir: Path):
    """Reports of every finished case.

    Existing reports are left alone, clean_figures wipes the figure dir to force regeneration.
    """
    for case in study.cases:
        generate_case_report(study, case, figure_dir)


def analysis_case_done(study: Study, case, figure_dir: Path) -> bool:
    """Whether a case needs no more analysis work.

    Its metrics cache exists and either it is the empty 'nothing valid' marker or the case report is on disk.
    """
    cache_path = case_metrics_path(study, case)
    if not cache_path.exists():
        return False
    case_metrics = xr.load_dataset(cache_path)
    if not case_metrics.data_vars:
        return True
    reports_module = import_module(study.ANALYSIS_REPORTS_MODULE)
    case_dir = case_report_dir(figure_dir, case)
    return reports_module.case_report_done(case_dir)
