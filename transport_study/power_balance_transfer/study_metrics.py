"""Stage-resolved performance metrics for power balance transfer study results.

The per-case result files already carry per-timeslice absolute and relative
Wtot errors (error_abs_ts / error_rel_ts, see PowerBalanceTRB); what they do
not carry is where in the discharge each timeslice sits. This module joins
each result timeslice back to its device dataset by (shot, nearest time) to
pick up Ip and the auxiliary heating powers, labels it with a shot stage
(rampup / flattop / rampdown, with an aux-heating flag subdividing the
flattop), and aggregates the errors per stage.

Per-shot TIME-AVERAGED errors (mean over the shot's valid timeslices) are also
computed here: the per-shot errors in the result files are raw time integrals,
so longer shots score worse at equal instantaneous error. The time-averaged
form removes that duration confound and is what the case reports rank shots by.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from transport_study import EPISODE_DIM
from transport_study.config import config
from transport_study.orchestration.organize_data import INPUT_POWER_SIGNALS
from transport_study.orchestration.stages import STAGE_AGG_NAMES, segment_stages
from transport_study.orchestration.study import PAD_TIME_STEP_S, Study

# Joined dataset timeslice must be within this of the result timeslice.
# Timebases are 1 kHz, so anything beyond half a sample is a bad join
TIME_JOIN_TOLERANCE_S = 6e-4

METRIC_NAMES = ("abs", "rel")

COLLECTED_METRICS_FILENAME = "collected_metrics.nc"
CASE_METRICS_FILENAME = "case_metrics.nc"

__all__ = [
    "METRIC_NAMES",
    "STAGE_AGG_NAMES",
    "CaseTimesliceMetrics",
    "case_metrics_path",
    "collect_metrics",
    "collected_metrics_path",
    "compute_and_save_case_metrics",
    "compute_case_timeslice_metrics",
    "shot_time_averaged_errors",
]


@cache
def load_stage_dataset(device: str) -> xr.Dataset:
    """Device dataset with the signals needed to stage-label result timeslices.

    Only Ip, the auxiliary heating powers, and the time coordinate are kept
    (the errors themselves already live in the result files). Missing aux
    power signals are filled with zeros, matching organize_data.
    """
    ds_path = Path(config.dataset_paths[device])
    ds = xr.open_dataset(ds_path)

    for sig in INPUT_POWER_SIGNALS:
        if sig not in ds:
            ds[sig] = xr.zeros_like(ds["Ip_MA"])

    ds = ds[[*INPUT_POWER_SIGNALS, "Ip_MA", "time"]]
    return ds.load()


@dataclass
class CaseTimesliceMetrics:
    """Long-form per-timeslice metrics for one case, one record per valid test
    timeslice (finite time and relative error, clock-advancing rather than a
    padded repeat of the shot's final timeslice, joined to the device dataset)."""

    shot: np.ndarray
    ds_source: np.ndarray
    time: np.ndarray
    stage: np.ndarray
    aux_heated: np.ndarray
    err_abs: np.ndarray
    err_rel: np.ndarray

    def __len__(self) -> int:
        return len(self.shot)

    def stage_mask(self, stage: str) -> np.ndarray:
        if stage == "all":
            return np.ones(len(self), dtype=bool)
        if stage == "flattop_ohmic":
            return (self.stage == "flattop") & ~self.aux_heated
        if stage == "flattop_aux":
            return (self.stage == "flattop") & self.aux_heated
        return self.stage == stage

    def metric(self, name: str) -> np.ndarray:
        return getattr(self, f"err_{name}")


def compute_case_timeslice_metrics(result_ds: xr.Dataset) -> CaseTimesliceMetrics:
    """Stage-label every valid test timeslice of one case result file.

    Joins each result timeslice to its device dataset by (shot, nearest time)
    to pick up the Ip and auxiliary power traces the result file does not
    carry, then labels it with segment_stages.
    """
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
        valid = np.isfinite(res_time) & np.isfinite(err_rel)
        # The rollout batches pad every shot to a common length by repeating its
        # final timeslice with a clamped time value (not NaN). Keep only rows that
        # advance the shot clock, otherwise the shot-end error is counted hundreds
        # of times in the stage aggregates and time averages
        valid[1:] &= np.diff(res_time) > PAD_TIME_STEP_S
        result_idxs = np.flatnonzero(valid)
        if len(result_idxs) == 0:
            continue

        # Nearest-time join into the device dataset
        stage_time = shot_stage["time"].values
        finite_stage = np.isfinite(stage_time)
        stage_positions = np.flatnonzero(finite_stage)
        if len(stage_positions) == 0:
            logger.warning(f"Shot {shot} has no finite times in the {device} dataset, skipping")
            continue
        stage_times_finite = stage_time[stage_positions]
        order = np.argsort(stage_times_finite)
        sorted_times = stage_times_finite[order]
        sorted_positions = stage_positions[order]

        target_times = res_time[result_idxs]
        insert = np.searchsorted(sorted_times, target_times)
        insert = np.clip(insert, 1, len(sorted_times) - 1)
        left, right = sorted_times[insert - 1], sorted_times[insert]
        pick_right = np.abs(right - target_times) < np.abs(target_times - left)
        nearest_sorted = np.where(pick_right, insert, insert - 1)
        joined = np.abs(sorted_times[nearest_sorted] - target_times) <= TIME_JOIN_TOLERANCE_S
        if not joined.all():
            logger.warning(f"Shot {shot}: {np.sum(~joined)} result timeslices had no dataset time within tolerance, dropping them")
        result_idxs = result_idxs[joined]
        stage_idxs = sorted_positions[nearest_sorted[joined]]
        if len(result_idxs) == 0:
            continue

        # Stage labels over the full shot, then picked at the joined timeslices
        p_aux = sum(np.nan_to_num(shot_stage[sig].values, nan=0.0) for sig in INPUT_POWER_SIGNALS)
        stage_full, aux_full = segment_stages(shot_stage["Ip_MA"].values, p_aux)

        n = len(result_idxs)
        records["shot"].append(np.full(n, shot))
        records["ds_source"].append(np.full(n, device, dtype=object))
        records["time"].append(res_time[result_idxs])
        records["stage"].append(stage_full[stage_idxs])
        records["aux_heated"].append(aux_full[stage_idxs])
        records["err_abs"].append(err_abs[result_idxs])
        records["err_rel"].append(err_rel[result_idxs])

    def _cat(chunks: list, dtype=None) -> np.ndarray:
        if not chunks:
            return np.array([], dtype=dtype if dtype is not None else float)
        return np.concatenate(chunks)

    return CaseTimesliceMetrics(
        shot=_cat(records["shot"]),
        ds_source=_cat(records["ds_source"], dtype=object).astype(str),
        time=_cat(records["time"]),
        stage=_cat(records["stage"], dtype=object).astype(str),
        aux_heated=_cat(records["aux_heated"], dtype=bool),
        err_abs=_cat(records["err_abs"]),
        err_rel=_cat(records["err_rel"]),
    )


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


def collected_metrics_path(study: Study) -> Path:
    return study.result_dir / COLLECTED_METRICS_FILENAME


def case_metrics_path(study: Study, case) -> Path:
    """Per-case stage-aggregate cache, next to the case's result_data.nc.
    An empty dataset is the marker for 'computed, but no valid timeslices',
    so parallel analysis jobs can signal completion either way."""
    return study.result_path(case).parent / CASE_METRICS_FILENAME


def _aggregate_case_metrics(ts_metrics: CaseTimesliceMetrics) -> xr.Dataset:
    """Reduce one case's per-timeslice errors to per-stage statistics.

    Dims: stage (STAGE_AGG_NAMES). Data variables <metric>_<stat> for metric in
    abs/rel and stat in mean/std/med/count. Case-identifying coords are
    attached later by collect_metrics, which knows the case_idx.
    """
    data_vars = {}
    for metric in METRIC_NAMES:
        values = ts_metrics.metric(metric)
        means, stds, meds, counts = [], [], [], []
        for stage in STAGE_AGG_NAMES:
            stage_values = values[ts_metrics.stage_mask(stage)]
            stage_values = stage_values[np.isfinite(stage_values)]
            counts.append(len(stage_values))
            if len(stage_values) == 0:
                means.append(np.nan)
                stds.append(np.nan)
                meds.append(np.nan)
            else:
                means.append(float(np.mean(stage_values)))
                stds.append(float(np.std(stage_values)))
                meds.append(float(np.median(stage_values)))
        data_vars[f"{metric}_mean"] = ("stage", np.array(means))
        data_vars[f"{metric}_std"] = ("stage", np.array(stds))
        data_vars[f"{metric}_med"] = ("stage", np.array(meds))
        data_vars[f"{metric}_count"] = ("stage", np.array(counts))

    return xr.Dataset(data_vars=data_vars, coords={"stage": list(STAGE_AGG_NAMES)})


def compute_and_save_case_metrics(study, case) -> xr.Dataset:
    """Stage-aggregate metrics for one case, cached to case_metrics_path.

    Returns the cached dataset when present, otherwise computes from the case
    result file and saves. A case whose result file exists but yields no valid
    timeslices caches an empty dataset so the work is not retried. A case with
    no result file returns an empty dataset without caching (results may still
    appear later).
    """
    cache_path = case_metrics_path(study, case)
    if cache_path.exists():
        return xr.load_dataset(cache_path)

    result_path = study.result_path(case)
    if not result_path.exists():
        return xr.Dataset()

    result_ds = xr.load_dataset(result_path)
    ts_metrics = compute_case_timeslice_metrics(result_ds)
    if len(ts_metrics) == 0:
        logger.warning(f"No valid test timeslices for case {case}, caching empty metrics marker")
        case_ds = xr.Dataset()
    else:
        case_ds = _aggregate_case_metrics(ts_metrics)
        logger.info(f"Computed stage-resolved metrics for case {case} ({len(ts_metrics)} timeslices)")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    case_ds.to_netcdf(cache_path)
    return case_ds


def collect_metrics(study) -> xr.Dataset:
    """Aggregate stage-resolved metrics for every finished case.

    Dims: (case_idx, stage) with stage in STAGE_AGG_NAMES. Data variables
    <metric>_<stat> for metric in abs/rel and stat in mean/std/med/count.
    Case-identifying coords along case_idx match collect_results.

    Reads the per-case caches written by compute_and_save_case_metrics
    (parallel analysis jobs fill them ahead of time) and computes any that are
    still missing in-process. The per-case caches are the real cache; the
    combined dataset is rebuilt (cheap concat) and written to
    collected_metrics.nc in the study result dir every call, so a partial file
    from an interrupted run can never mask newly finished cases.
    """
    cache_path = collected_metrics_path(study)

    results = []
    for case_idx, case in enumerate(study.cases):
        case_ds = compute_and_save_case_metrics(study, case)
        if not case_ds.data_vars:
            continue
        results.append(case_ds.assign_coords(study._case_coords(case_idx, case)))

    if not results:
        logger.warning("No finished cases with valid metrics, stage-resolved metrics are empty")
        return xr.Dataset()

    # coords="different" stacks the per-case scalar coords (model_type, ...)
    metrics_ds = xr.concat(results, dim="case_idx", coords="different")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_ds.to_netcdf(cache_path)
    logger.info(f"Saved stage-resolved metrics to {cache_path}")
    return metrics_ds
