"""Stage-resolved metrics of transport transfer study results (its ANALYSIS_METRICS_MODULE, see orchestration.case_metrics).

The cases are scored the way their checkpoints were selected:
combined is the validation chi per fresh test timeslice (value chi + gradient_weight x gradient chi, summed over ne and Te,
written by the shared profile test suite as error_chi_ts), value and grad its two parts.
The power balance metrics ride along, abs and rel the plain profile errors and diverged the non-finite rollout flag.
Stale (forward-filled) timeslices are NaN in every error and still count in diverged, as in the validation penalty.
The power_balance / p_oh / p_rad prereq cases write scalar result files without chi, which reads as NaN.
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr

from transport_study.orchestration.case_metrics import staged_result_records
from transport_study.orchestration.study import Study
from transport_study.power_balance_transfer import (
    study_metrics as power_balance_metrics,
)

METRIC_NAMES = ("combined", "value", "grad", *power_balance_metrics.METRIC_NAMES)
METRIC_VARS = {
    "combined": "error_chi_ts",
    "value": "error_chi_value_ts",
    "grad": "error_chi_grad_ts",
    **power_balance_metrics.METRIC_VARS,
}


@dataclass
class CaseTimesliceMetrics(power_balance_metrics.CaseTimesliceMetrics):
    """The power balance metrics plus the chi of the validation loss, so the power balance report helpers apply."""

    err_combined: np.ndarray
    err_value: np.ndarray
    err_grad: np.ndarray


def compute_case_timeslice_metrics(result_ds: xr.Dataset) -> CaseTimesliceMetrics:
    """Stage-label every valid test timeslice of one case result file."""
    # Looked up at call time, the stage dataset is the power balance one
    records = staged_result_records(result_ds, power_balance_metrics.load_stage_dataset, METRIC_VARS)
    return CaseTimesliceMetrics.from_records(records, METRIC_NAMES)


def case_timeslice_metrics(study: Study, case, result_ds: xr.Dataset) -> CaseTimesliceMetrics:
    return compute_case_timeslice_metrics(result_ds)
