"""Stage-resolved metrics of profile transfer study results (its ANALYSIS_METRICS_MODULE, see orchestration.case_metrics).

Reads the chi validation loss (ProfilePredictorTRB.get_val_loss_fn, trb_utils.chi_value / chi_gradient)
that the test suite writes per test timeslice and retained checkpoint,
joined back to the device datasets for ip_MA and the heating powers of the stage labels.
Each test timeslice gets four metrics:

- metric_value: value chi integrated over rho, summed over the ne and Te channels
- metric_grad: gradient chi at the rho midpoints, masked to rho below GRAD_RHO_MAX, summed over the channels
- metric_combined: metric_value + gradient_weight * metric_grad, the validation loss itself less the device weight
- metric_diverged: 1 where the prediction went non-finite (NaN in the chi metrics), 0 otherwise

and a shot-stage label (rampup / flattop / rampdown, with an aux-heating flag subdividing the flattop).
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import ClassVar

import numpy as np
import xarray as xr

from transport_study import TIME_COORD
from transport_study.config import config
from transport_study.orchestration.case_metrics import (
    DIVERGED_VAR,
    StagedTimesliceMetrics,
    staged_result_records,
)
from transport_study.orchestration.organize_data import (
    PROFILE_TARGET_VARS,
    to_rho_grid,
)
from transport_study.orchestration.study import Study
from transport_study.signals import POWER_ADDITIONAL_MW, convert_to_working_units

METRIC_NAMES = ("value", "grad", "combined", "diverged")
METRIC_VARS = {
    "value": "error_chi_value_ts",
    "grad": "error_chi_grad_ts",
    "combined": "error_chi_ts",
    "diverged": DIVERGED_VAR,
}


@cache
def load_eval_dataset(device: str) -> xr.Dataset:
    """Device dataset the result timeslices join to, for the stage labels and the case reports' error bars.

    Profile signals and their gradient / error-bar companions are put on the
    same uniform 51-point rho grid as training (organize_data.to_rho_grid,
    the prep of get_ds). ip_MA, the auxiliary power, and the time coord are
    kept for stage segmentation. Non-fresh timeslices are kept: the join is by
    time, so only timeslices that appear in a result file are ever read.
    """
    ds_path = Path(config.dataset_paths[device])
    ds_store = xr.open_dataset(ds_path)
    ds_working = convert_to_working_units(ds_store)
    ds = ds_working[[*PROFILE_TARGET_VARS, POWER_ADDITIONAL_MW, "ip_MA", TIME_COORD]]
    ds_rho = to_rho_grid(ds)
    return ds_rho.load()


@dataclass
class CaseTimesliceMetrics(StagedTimesliceMetrics):
    """result_time_idx / eval_time_idx are positional indices into the case result file and the device eval dataset,
    so callers can fetch the corresponding profiles and error bars."""

    METRIC_PREFIX: ClassVar[str] = "metric_"

    result_time_idx: np.ndarray
    eval_time_idx: np.ndarray
    metric_value: np.ndarray
    metric_grad: np.ndarray
    metric_combined: np.ndarray
    metric_diverged: np.ndarray


def compute_case_timeslice_metrics(result_ds: xr.Dataset) -> CaseTimesliceMetrics:
    """Stage-label every valid test timeslice of one case result file, joined to its device eval dataset by (shot, nearest time)."""
    records = staged_result_records(result_ds, load_eval_dataset, METRIC_VARS)
    return CaseTimesliceMetrics.from_records(
        records,
        METRIC_NAMES,
        result_time_idx=records["result_time_idx"],
        eval_time_idx=records["device_time_idx"],
    )


def case_timeslice_metrics(study: Study, case, result_ds: xr.Dataset) -> CaseTimesliceMetrics:
    return compute_case_timeslice_metrics(result_ds)
