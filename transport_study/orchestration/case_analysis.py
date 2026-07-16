"""Generic SLURM fan-out driver for per-case analysis work.

Both studies dispatch their per-case analysis (stage metrics + case report,
CPU-bound matplotlib and numpy work) as one SLURM job per case. The driver
loop here is study-agnostic; each Study subclass declares which modules carry
its per-case analysis via the ANALYSIS_METRICS_MODULE / ANALYSIS_REPORTS_MODULE
ClassVars. The metrics module must export
``compute_and_save_case_metrics(study, case)`` and the reports module
``generate_case_report(study, case, figure_dir)`` plus
``analysis_case_done(study, case, figure_dir)``.
"""

import time
from importlib import import_module

from loguru import logger

from transport_study.config import config
from transport_study.orchestration.slurm_utils import (
    get_running_job_names,
    launch_case_analysis_parallel,
)

# Resubmission cap per case and how often the driver rechecks for finished cases
ANALYSIS_MAX_ATTEMPTS = 3
ANALYSIS_POLL_INTERVAL_S = 30


def run_case_analysis_parallel(study) -> None:
    """Fan the per-case analysis out over SLURM.

    Submits CPU jobs for finished cases that still need analysis (via
    launch_case_analysis_parallel, which targets config.analysis_partition)
    and polls until every case is done or has exhausted its attempts. At most
    config.max_analysis_jobs analysis jobs are in the queue (running or
    pending) at once, so a study with hundreds of cases does not flood the
    scheduler; more jobs are submitted as earlier ones finish. Cases that
    exhaust their attempts fall back to the study's serial analysis path
    afterwards.
    """
    analysis_case_done = import_module(study.ANALYSIS_REPORTS_MODULE).analysis_case_done
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
