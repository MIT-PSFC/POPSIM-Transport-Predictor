"""Stage-resolved performance metrics for transport transfer study results.

The power balance study's stage-metric machinery is fully generic over the
error_abs_ts / error_rel_ts variables (it joins result timeslices back to the
device datasets for Ip / aux power stage labeling and never touches the
predicted signal itself), and the transport result files carry exactly those
variables (rho-integrated combined ne+Te errors, see
ProfilePredictorTRB.get_test_eval_suite). So the whole module is reused
wholesale; this module exists to give the study its own
ANALYSIS_METRICS_MODULE dotted path.

Like the power balance study, the per-shot errors in the result files are raw
time integrals (long shots score worse), so shots are ranked by the
TIME-AVERAGED errors computed here.
"""

from transport_study.power_balance_transfer.study_metrics import (
    CASE_METRICS_FILENAME,
    COLLECTED_METRICS_FILENAME,
    METRIC_NAMES,
    STAGE_AGG_NAMES,
    CaseTimesliceMetrics,
    case_metrics_path,
    collect_metrics,
    collected_metrics_path,
    compute_and_save_case_metrics,
    compute_case_timeslice_metrics,
    load_stage_dataset,
    shot_time_averaged_errors,
)

__all__ = [
    "CASE_METRICS_FILENAME",
    "COLLECTED_METRICS_FILENAME",
    "METRIC_NAMES",
    "STAGE_AGG_NAMES",
    "CaseTimesliceMetrics",
    "case_metrics_path",
    "collect_metrics",
    "collected_metrics_path",
    "compute_and_save_case_metrics",
    "compute_case_timeslice_metrics",
    "load_stage_dataset",
    "shot_time_averaged_errors",
]
