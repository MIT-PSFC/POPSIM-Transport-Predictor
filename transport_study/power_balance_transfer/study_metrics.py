"""Stage-resolved metrics of power balance transfer study results (its ANALYSIS_METRICS_MODULE, see orchestration.case_metrics).

The result files carry per-timeslice absolute and relative errors (error_abs_ts / error_rel_ts)
and the diverged flag (error_diverged_ts) of every retained checkpoint,
not where in the discharge each timeslice sits.
Each result timeslice is joined back to its device dataset by (shot, nearest time)
to pick up ip_MA and the auxiliary power for the shot-stage labels.
The diverged metric's stage means are the fraction of timeslices whose prediction went non-finite,
which the error metrics skip.

The per-shot errors in the result files are raw time integrals, so longer shots score worse at equal instantaneous error.
shot_time_averaged removes that duration confound and is what the case reports rank shots by.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import ClassVar

import numpy as np
import xarray as xr
from transport_validation_datasets.store_schema import HEATING_POWERS

from transport_study import TIME_COORD
from transport_study.config import config
from transport_study.orchestration.case_metrics import (
    DIVERGED_VAR,
    StagedTimesliceMetrics,
    staged_result_records,
)
from transport_study.orchestration.study import Study
from transport_study.signals import POWER_ADDITIONAL_MW, convert_to_working_units

METRIC_NAMES = ("abs", "rel", "diverged")
METRIC_VARS = {"abs": "error_abs_ts", "rel": "error_rel_ts", "diverged": DIVERGED_VAR}


@cache
def load_stage_dataset(device: str) -> xr.Dataset:
    """Device dataset with only the signals the stage labels need, ip_MA, the auxiliary power and time."""
    ds_path = Path(config.dataset_paths[device])
    ds_store = xr.open_dataset(ds_path)
    ds_store_selected = ds_store[["ip", *HEATING_POWERS, TIME_COORD]]
    ds = convert_to_working_units(ds_store_selected)
    return ds[["ip_MA", POWER_ADDITIONAL_MW, TIME_COORD]].load()


@dataclass
class CaseTimesliceMetrics(StagedTimesliceMetrics):
    """Valid test timeslices are clock-advancing (not a padded repeat of the final timeslice)
    with a target, joined to the device dataset."""

    METRIC_PREFIX: ClassVar[str] = "err_"

    err_abs: np.ndarray
    err_rel: np.ndarray
    err_diverged: np.ndarray


def compute_case_timeslice_metrics(result_ds: xr.Dataset) -> CaseTimesliceMetrics:
    """Stage-label every valid test timeslice of one case result file."""
    records = staged_result_records(result_ds, load_stage_dataset, METRIC_VARS)
    return CaseTimesliceMetrics.from_records(records, METRIC_NAMES)


def case_timeslice_metrics(study: Study, case, result_ds: xr.Dataset) -> CaseTimesliceMetrics:
    return compute_case_timeslice_metrics(result_ds)


def shot_time_averaged(ts_metrics: StagedTimesliceMetrics, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-shot time average of one metric at the best checkpoint, the duration-free shot ranking metric.

    Returns:
        (shots, averages, n_ts): unique shots with the mean of the metric over each shot's finite timeslices,
        and how many timeslices contributed. Shots without one get NaN.
    """
    values = ts_metrics.best(name)
    shots = np.unique(ts_metrics.shot)
    averages = np.full(len(shots), np.nan)
    n_ts = np.zeros(len(shots), dtype=int)
    for i, shot in enumerate(shots):
        shot_values = values[ts_metrics.shot == shot]
        finite_values = shot_values[np.isfinite(shot_values)]
        n_ts[i] = len(finite_values)
        if n_ts[i] > 0:
            averages[i] = float(np.mean(finite_values))
    return shots, averages, n_ts
