"""Study analysis drivers.

Every study dispatches its per-case analysis (stage metrics + case report,
CPU-bound matplotlib and numpy work) as one SLURM job per case.
The driver loop here is study-agnostic,
the per-study pieces come from the ANALYSIS_METRICS_MODULE / ANALYSIS_REPORTS_MODULE ClassVars
through orchestration.case_metrics and orchestration.case_reports.
run_summary_analysis is the whole analysis of the studies whose collect_results is the per-case scalar summary.
"""

import time

import xarray as xr
from loguru import logger

from transport_study.config import config
from transport_study.orchestration.case_metrics import collect_metrics
from transport_study.orchestration.case_reports import (
    analysis_case_done,
    generate_case_reports,
)
from transport_study.orchestration.comparison_figures import (
    ComparisonFamily,
    ComparisonLayout,
    comparison_figures,
)
from transport_study.orchestration.slurm_utils import (
    get_running_job_names,
    launch_case_analysis_parallel,
)
from transport_study.orchestration.tables import (
    ComparisonTableSpec,
    write_summary_comparison_tables,
)

# Resubmission cap per case and how often the driver rechecks for finished cases
ANALYSIS_MAX_ATTEMPTS = 3
ANALYSIS_POLL_INTERVAL_S = 30


def run_case_analysis_parallel(study) -> None:
    """Fan the per-case analysis out over SLURM.

    Submits CPU jobs for finished cases that still need analysis (via
    launch_case_analysis_parallel, which targets config.analysis_partition)
    and polls until every case is done or has exhausted its attempts.
    At most config.max_analysis_jobs analysis jobs are queued (running or pending) at once,
    so a study with hundreds of cases does not flood the scheduler.
    More jobs are submitted as earlier ones finish.
    Cases that exhaust their attempts fall back to the study's serial analysis path afterwards.
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
            launch_case_analysis_parallel(study, case)
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


def log_section(title: str) -> None:
    logger.opt(colors=True).info(f"<bold><magenta>{title.upper()}</magenta></bold>")


def run_summary_analysis(
    study,
    enable_parallelism: bool,
    layout: ComparisonLayout,
    families: tuple[ComparisonFamily, ...],
    table_spec: ComparisonTableSpec,
) -> None:
    """Analysis of a study whose collect_results is the per-case scalar summary (dims case_idx).

    The per-case stage metrics and reports fan out over SLURM with parallelism,
    the serial paths after it skip the completed cases.
    Then the comparison figures over the collected results, the case reports and the per-axis comparison tables.
    """
    if enable_parallelism:
        run_case_analysis_parallel(study)

    # Stage-resolved time-averaged errors of every finished case, cached to collected_metrics.nc
    metrics_ds = collect_metrics(study)
    results_ds = xr.load_dataset(study.collected_results_path())
    for family in families:
        log_section(family.title)
        comparison_figures(results_ds, layout, family, study.figure_dir)

    log_section("Case reports")
    generate_case_reports(study, study.figure_dir)

    log_section("Comparison tables")
    write_summary_comparison_tables(results_ds, metrics_ds, table_spec, study.figure_dir)
