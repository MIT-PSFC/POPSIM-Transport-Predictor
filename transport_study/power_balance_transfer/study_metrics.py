"""Stage-resolved metrics of power balance transfer study results (its ANALYSIS_METRICS_MODULE, see orchestration.case_metrics).

The result files carry per-timeslice absolute and relative errors (error_abs_ts / error_rel_ts),
not where in the discharge each timeslice sits.
Each result timeslice is joined back to its device dataset by (shot, nearest time)
to pick up ip_MA and the auxiliary power for the shot-stage labels.
Nothing here reads the predicted signal itself,
so the transport study, whose result files carry the same error variables, scores its cases with this module too.

The per-shot errors in the result files are raw time integrals, so longer shots score worse at equal instantaneous error.
shot_time_averaged_errors removes that duration confound and is what the case reports rank shots by.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import ClassVar

import numpy as np
import xarray as xr
from loguru import logger
from transport_validation_datasets.store_schema import HEATING_POWERS

from transport_study import EPISODE_DIM, TIME_COORD
from transport_study.config import config
from transport_study.orchestration.case_metrics import (
    StagedTimesliceMetrics,
    concat_records,
    nearest_time_positions,
    shot_stages,
)
from transport_study.orchestration.study import Study, real_timeslice_mask
from transport_study.signals import POWER_ADDITIONAL_MW, convert_to_working_units

METRIC_NAMES = ("abs", "rel")


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
    with a finite relative error, joined to the device dataset."""

    METRIC_PREFIX: ClassVar[str] = "err_"

    err_abs: np.ndarray
    err_rel: np.ndarray


def compute_case_timeslice_metrics(result_ds: xr.Dataset) -> CaseTimesliceMetrics:
    """Stage-label every valid test timeslice of one case result file."""
    records: dict[str, list] = {name: [] for name in ("shot", "ds_source", "time", "stage", "aux_heated", "err_abs", "err_rel")}

    for shot_pos, shot in enumerate(result_ds[EPISODE_DIM].values):
        shot_res = result_ds.isel({EPISODE_DIM: shot_pos})
        device = str(shot_res["ds_source"].values)
        stage_ds = load_stage_dataset(device)
        if shot not in stage_ds[EPISODE_DIM].values:
            logger.warning(f"Shot {shot} not found in dataset for device {device}, skipping")
            continue
        shot_stage = stage_ds.sel({EPISODE_DIM: shot})

        res_time = shot_res["time"].values
        err_abs = shot_res["error_abs_ts"].values
        err_rel = shot_res["error_rel_ts"].values
        # Only clock-advancing rows count, the padded rollout tail would weight the shot-end error hundreds of times
        valid = real_timeslice_mask(shot_res["time"]).values & np.isfinite(err_rel)
        result_idxs = np.flatnonzero(valid)
        if len(result_idxs) == 0:
            continue

        mask_joined, stage_idxs = nearest_time_positions(shot_stage["time"].values, res_time[result_idxs], shot, device)
        result_idxs = result_idxs[mask_joined]
        if len(result_idxs) == 0:
            continue

        # Stage labels over the full shot, then picked at the joined timeslices
        stage_full, aux_full = shot_stages(shot_stage)

        n = len(result_idxs)
        records["shot"].append(np.full(n, shot))
        records["ds_source"].append(np.full(n, device, dtype=object))
        records["time"].append(res_time[result_idxs])
        records["stage"].append(stage_full[stage_idxs])
        records["aux_heated"].append(aux_full[stage_idxs])
        records["err_abs"].append(err_abs[result_idxs])
        records["err_rel"].append(err_rel[result_idxs])

    return CaseTimesliceMetrics(
        shot=concat_records(records["shot"]),
        ds_source=concat_records(records["ds_source"], dtype=object).astype(str),
        time=concat_records(records["time"]),
        stage=concat_records(records["stage"], dtype=object).astype(str),
        aux_heated=concat_records(records["aux_heated"], dtype=bool),
        err_abs=concat_records(records["err_abs"]),
        err_rel=concat_records(records["err_rel"]),
    )


def case_timeslice_metrics(study: Study, case, result_ds: xr.Dataset) -> CaseTimesliceMetrics:
    return compute_case_timeslice_metrics(result_ds)


def shot_time_averaged_errors(ts_metrics: CaseTimesliceMetrics) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-shot time-averaged errors, the duration-free shot ranking metric.

    Returns:
        (shots, avg_abs, avg_rel, n_ts): unique shots with the mean absolute
        and relative per-timeslice error over each shot's valid timeslices,
        and how many timeslices contributed. Shots whose errors are all
        non-finite get NaN averages.
    """
    shots = np.unique(ts_metrics.shot)
    avg_abs = np.full(len(shots), np.nan)
    avg_rel = np.full(len(shots), np.nan)
    n_ts = np.zeros(len(shots), dtype=int)
    for i, shot in enumerate(shots):
        mask = ts_metrics.shot == shot
        rel = ts_metrics.err_rel[mask]
        abs_ = ts_metrics.err_abs[mask]
        finite = np.isfinite(rel)
        n_ts[i] = int(finite.sum())
        if n_ts[i] > 0:
            avg_rel[i] = float(np.mean(rel[finite]))
            avg_abs[i] = float(np.nanmean(abs_[finite])) if np.isfinite(abs_[finite]).any() else np.nan
    return shots, avg_abs, avg_rel, n_ts
