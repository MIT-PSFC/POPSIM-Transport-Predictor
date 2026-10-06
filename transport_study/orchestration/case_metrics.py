"""Stage-resolved case metrics shared by every study.

Each study's ANALYSIS_METRICS_MODULE scores the test timeslices of one case result file
and exports two names:
    METRIC_NAMES: the per-timeslice metrics it scores
    case_timeslice_metrics(study, case, result_ds): a StagedTimesliceMetrics subclass holding them
Everything else lives here:
the nearest-time join back to the device datasets, the shot-stage labels, the per-stage aggregation,
the per-case cache and the collected metrics of a whole study.
"""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import ClassVar

import numpy as np
import xarray as xr
from loguru import logger

from transport_study.orchestration.stages import STAGE_AGG_NAMES, segment_stages
from transport_study.orchestration.study import Study, write_netcdf_atomic
from transport_study.signals import POWER_ADDITIONAL_MW

# Joined dataset timeslice must be within this of the result timeslice.
# Timebases are 1 kHz, so anything beyond half a sample is a bad join.
TIME_JOIN_TOLERANCE_S = 6e-4

COLLECTED_METRICS_FILENAME = "collected_metrics.nc"
CASE_METRICS_FILENAME = "case_metrics.nc"


@dataclass
class StagedTimesliceMetrics:
    """Long-form per-timeslice metrics of one case, one record per valid test timeslice.

    Subclasses add one array per metric, named METRIC_PREFIX + metric name.
    """

    METRIC_PREFIX: ClassVar[str]

    shot: np.ndarray
    ds_source: np.ndarray
    time: np.ndarray
    stage: np.ndarray
    aux_heated: np.ndarray

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
        return getattr(self, f"{self.METRIC_PREFIX}{name}")


def nearest_time_positions(dataset_time: np.ndarray, result_times: np.ndarray, shot, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Join result timeslices to a shot of a device dataset by nearest time.

    Returns the mask of result_times that joined within TIME_JOIN_TOLERANCE_S
    and the dataset positions of the joined ones.
    """
    dataset_positions = np.flatnonzero(np.isfinite(dataset_time))
    if len(dataset_positions) == 0:
        logger.warning(f"Shot {shot} has no finite times in the {device} dataset, skipping")
        return np.zeros(len(result_times), dtype=bool), np.array([], dtype=int)
    dataset_times_finite = dataset_time[dataset_positions]
    order = np.argsort(dataset_times_finite)
    sorted_times = dataset_times_finite[order]
    sorted_positions = dataset_positions[order]

    insert = np.searchsorted(sorted_times, result_times)
    insert = np.clip(insert, 1, len(sorted_times) - 1)
    left, right = sorted_times[insert - 1], sorted_times[insert]
    pick_right = np.abs(right - result_times) < np.abs(result_times - left)
    nearest_sorted = np.where(pick_right, insert, insert - 1)
    mask_joined = np.abs(sorted_times[nearest_sorted] - result_times) <= TIME_JOIN_TOLERANCE_S
    if not mask_joined.all():
        logger.warning(f"Shot {shot}: {np.sum(~mask_joined)} result timeslices had no dataset time within tolerance, dropping them")
    return mask_joined, sorted_positions[nearest_sorted[mask_joined]]


def shot_stages(shot_ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """segment_stages labels over every timeslice of one shot of a working-unit device dataset."""
    p_aux_MW = np.nan_to_num(shot_ds[POWER_ADDITIONAL_MW].values, nan=0.0)
    return segment_stages(shot_ds["ip_MA"].values, p_aux_MW)


def concat_records(chunks: list, dtype=None) -> np.ndarray:
    """Concatenated per-shot record arrays, an empty array of dtype (float by default) without any."""
    if not chunks:
        return np.array([], dtype=dtype if dtype is not None else float)
    return np.concatenate(chunks)


def aggregate_case_metrics(ts_metrics: StagedTimesliceMetrics, metric_names: tuple[str, ...]) -> xr.Dataset:
    """Reduce one case's per-timeslice metrics to per-stage statistics.

    Dims: stage (STAGE_AGG_NAMES).
    Data variables <metric>_<stat> for every metric and stat in mean / std / med / count.
    collect_metrics attaches the case-identifying coords, it knows the case_idx.
    """
    data_vars = {}
    for metric in metric_names:
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


def case_timeslice_metrics(study: Study, case, result_ds: xr.Dataset) -> StagedTimesliceMetrics:
    """The study's own per-timeslice scoring of one case result file (its ANALYSIS_METRICS_MODULE)."""
    metrics_module = import_module(study.ANALYSIS_METRICS_MODULE)
    return metrics_module.case_timeslice_metrics(study, case, result_ds)


def collected_metrics_path(study: Study) -> Path:
    return study.result_dir / COLLECTED_METRICS_FILENAME


def case_metrics_path(study: Study, case) -> Path:
    """Per-case stage-aggregate cache, always in this study's own result dir.

    A borrowed case's result file sits in its home study (see orchestration/lineage.py), which this study never writes.
    An empty dataset marks 'computed, but no valid timeslices',
    so parallel analysis jobs can signal completion either way.
    """
    return study.result_dir / str(case) / CASE_METRICS_FILENAME


def cached_case_metrics_path(study: Study, case) -> Path | None:
    """The case's metrics cache, this study's own or else, read-only, the one its home study wrote next to the result file.

    None when neither exists yet.
    """
    home_cache_path = study.result_path(case).parent / CASE_METRICS_FILENAME
    for cache_path in (case_metrics_path(study, case), home_cache_path):
        if cache_path.exists():
            return cache_path
    return None


def compute_and_save_case_metrics(study: Study, case) -> xr.Dataset:
    """Stage-aggregate metrics for one case, cached to case_metrics_path.

    Returns the cached dataset when present, otherwise computes it from the case result file and saves it.
    A result file without valid timeslices caches an empty dataset so the work is not retried.
    A case without a result file returns an empty dataset without caching, results may still appear later.
    """
    existing_cache_path = cached_case_metrics_path(study, case)
    if existing_cache_path is not None:
        return xr.load_dataset(existing_cache_path)

    cache_path = case_metrics_path(study, case)
    result_path = study.result_path(case)
    if not result_path.exists():
        return xr.Dataset()

    result_ds = xr.load_dataset(result_path)
    ts_metrics = case_timeslice_metrics(study, case, result_ds)
    if len(ts_metrics) == 0:
        logger.warning(f"No valid test timeslices for case {case}, caching empty metrics marker")
        case_ds = xr.Dataset()
    else:
        metric_names = import_module(study.ANALYSIS_METRICS_MODULE).METRIC_NAMES
        case_ds = aggregate_case_metrics(ts_metrics, metric_names)
        logger.info(f"Computed stage-resolved metrics for case {case} ({len(ts_metrics)} timeslices)")

    write_netcdf_atomic(case_ds, cache_path)
    return case_ds


def collect_metrics(study: Study) -> xr.Dataset:
    """Stage-resolved metrics of every finished case.

    Dims: (case_idx, stage), data variables as in aggregate_case_metrics,
    case-identifying coords along case_idx as in collect_results.
    Reads the per-case caches (parallel analysis jobs fill them ahead of time) and computes the missing ones in-process.
    The combined dataset is rebuilt and written to collected_metrics.nc on every call,
    so a partial file from an interrupted run never masks newly finished cases.
    """
    cache_path = collected_metrics_path(study)

    results = []
    for case_idx, case in enumerate(study.cases):
        case_ds = compute_and_save_case_metrics(study, case)
        if not case_ds.data_vars:
            continue
        results.append(case_ds.assign_coords(study.case_coords(case_idx, case)))

    if not results:
        logger.warning("No finished cases with valid metrics, stage-resolved metrics are empty")
        return xr.Dataset()

    # coords="different" stacks the per-case scalar coords (model_type, ...).
    # compat is pinned, the xarray default is changing to "override", which coords="different" rejects.
    metrics_ds = xr.concat(results, dim="case_idx", coords="different", compat="equals")
    write_netcdf_atomic(metrics_ds, cache_path)
    logger.info(f"Saved stage-resolved metrics to {cache_path}")
    return metrics_ds
