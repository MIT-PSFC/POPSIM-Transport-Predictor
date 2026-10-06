"""Stage-resolved case metrics shared by every study.

Each study's ANALYSIS_METRICS_MODULE scores the test timeslices of one case result file
and exports two names:
    METRIC_NAMES: the per-timeslice metrics it scores
    case_timeslice_metrics(study, case, result_ds): a StagedTimesliceMetrics subclass holding them
Everything else lives here:
the per-timeslice records read off a result file (staged_result_records),
the nearest-time join back to the device datasets, the shot-stage labels, the per-stage aggregation,
the per-case cache and the collected metrics of a whole study.
Every metric keeps one value per retained checkpoint (orchestration.topk_results),
and the per-stage aggregation reduces each case score to its best, top-K mean and top-K std.
"""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import ClassVar

import numpy as np
import xarray as xr
from loguru import logger

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.orchestration.stages import STAGE_AGG_NAMES, segment_stages
from transport_study.orchestration.study import (
    Study,
    real_timeslice_mask,
    write_netcdf_atomic,
)
from transport_study.orchestration.topk_results import (
    BEST_EPOCH_ATTR,
    CKPT_DIM,
    topk_statistics,
)
from transport_study.signals import POWER_ADDITIONAL_MW

# Joined dataset timeslice must be within this of the result timeslice.
# Timebases are 1 kHz, so anything beyond half a sample is a bad join.
TIME_JOIN_TOLERANCE_S = 6e-4

COLLECTED_METRICS_FILENAME = "collected_metrics.nc"
CASE_METRICS_FILENAME = "case_metrics.nc"

# Result variable every test suite writes, 1 diverged, 0 finite and NaN without a target
DIVERGED_VAR = "error_diverged_ts"

# The StagedTimesliceMetrics fields every staged_result_records call fills
RECORD_FIELDS = ("shot", "ds_source", "time", "stage", "aux_heated", "ckpt_epochs", "best_epoch")


@dataclass
class StagedTimesliceMetrics:
    """Long-form per-timeslice metrics of one case, one record per valid test timeslice.

    Subclasses add one (records, checkpoints) array per metric, named METRIC_PREFIX + metric name,
    one column per retained checkpoint epoch in ckpt_epochs.
    """

    METRIC_PREFIX: ClassVar[str]

    shot: np.ndarray
    ds_source: np.ndarray
    time: np.ndarray
    stage: np.ndarray
    aux_heated: np.ndarray
    ckpt_epochs: np.ndarray
    best_epoch: int

    @classmethod
    def from_records(cls, records: dict, metric_names: tuple[str, ...], **extra_fields):
        """The metrics of staged_result_records, metric_names becoming the METRIC_PREFIX fields."""
        base_fields = {name: records[name] for name in RECORD_FIELDS}
        metric_fields = {f"{cls.METRIC_PREFIX}{name}": records[name] for name in metric_names}
        return cls(**base_fields, **metric_fields, **extra_fields)

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
        """One metric of every record at every retained checkpoint, (records, checkpoints)."""
        return getattr(self, f"{self.METRIC_PREFIX}{name}")

    def best(self, name: str) -> np.ndarray:
        """One metric of every record at the best checkpoint, the one whose predictions the case reports plot."""
        best_column = int(np.flatnonzero(self.ckpt_epochs == self.best_epoch)[0])
        return self.metric(name)[:, best_column]


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


def staged_result_records(result_ds: xr.Dataset, device_dataset: Callable[[str], xr.Dataset], metric_vars: dict[str, str]) -> dict:
    """The per-timeslice records of one case result file, the shared part of every study's case_timeslice_metrics.

    device_dataset(device) is a working-unit device dataset with ip_MA, the auxiliary power and time,
    each result timeslice is joined to it by (shot, nearest time) for its stage label.
    metric_vars maps each metric name to the result variable it reads, NaN when a result file lacks it.
    A record is a timeslice that advances its shot's clock (the padded rollout tail would weight the shot end
    hundreds of times) and has a target, read off the best checkpoint's DIVERGED_VAR.
    A diverged timeslice therefore stays a record, NaN in the errors and 1 in the diverged flag.

    Returns the RECORD_FIELDS, result_time_idx / device_time_idx (positions in the result file and the device dataset)
    and one (records, checkpoints) array per metric name.
    """
    ckpt_epochs = result_ds[CKPT_DIM].values
    best_epoch = int(result_ds.attrs[BEST_EPOCH_ATTR])
    record_names = ("shot", "ds_source", "time", "stage", "aux_heated", "result_time_idx", "device_time_idx")
    chunks: dict[str, list] = {name: [] for name in (*record_names, *metric_vars)}

    for shot_pos, shot in enumerate(result_ds[EPISODE_DIM].values):
        shot_res = result_ds.isel({EPISODE_DIM: shot_pos})
        device = str(shot_res["ds_source"].values)
        device_ds = device_dataset(device)
        if shot not in device_ds[EPISODE_DIM].values:
            logger.warning(f"Shot {shot} not found in dataset for device {device}, skipping")
            continue
        shot_device = device_ds.sel({EPISODE_DIM: shot})

        res_time = shot_res[TIME_COORD].values
        diverged_best = shot_res[DIVERGED_VAR].sel({CKPT_DIM: best_epoch}).values
        mask_valid = real_timeslice_mask(shot_res[TIME_COORD]).values & np.isfinite(diverged_best)
        result_idxs = np.flatnonzero(mask_valid)
        if len(result_idxs) == 0:
            continue

        mask_joined, device_idxs = nearest_time_positions(shot_device[TIME_COORD].values, res_time[result_idxs], shot, device)
        result_idxs = result_idxs[mask_joined]
        if len(result_idxs) == 0:
            continue

        # Stage labels over the full shot, then picked at the joined timeslices
        stage_full, aux_full = shot_stages(shot_device)

        n = len(result_idxs)
        chunks["shot"].append(np.full(n, shot))
        chunks["ds_source"].append(np.full(n, device, dtype=object))
        chunks["time"].append(res_time[result_idxs])
        chunks["stage"].append(stage_full[device_idxs])
        chunks["aux_heated"].append(aux_full[device_idxs])
        chunks["result_time_idx"].append(result_idxs)
        chunks["device_time_idx"].append(device_idxs)
        for name, var in metric_vars.items():
            if var in shot_res:
                values = shot_res[var].transpose(TIME_DIM, CKPT_DIM).values[result_idxs]
            else:
                values = np.full((n, len(ckpt_epochs)), np.nan)
            chunks[name].append(values)

    records = {
        "shot": concat_records(chunks["shot"]),
        "ds_source": concat_records(chunks["ds_source"], dtype=object).astype(str),
        "time": concat_records(chunks["time"]),
        "stage": concat_records(chunks["stage"], dtype=object).astype(str),
        "aux_heated": concat_records(chunks["aux_heated"], dtype=bool),
        "result_time_idx": concat_records(chunks["result_time_idx"], dtype=int),
        "device_time_idx": concat_records(chunks["device_time_idx"], dtype=int),
        "ckpt_epochs": ckpt_epochs,
        "best_epoch": best_epoch,
    }
    for name in metric_vars:
        records[name] = np.concatenate(chunks[name]) if chunks[name] else np.full((0, len(ckpt_epochs)), np.nan)
    return records


def aggregate_case_metrics(ts_metrics: StagedTimesliceMetrics, metric_names: tuple[str, ...]) -> xr.Dataset:
    """Reduce one case's per-timeslice metrics to per-stage statistics.

    Each stat (mean / std / med / count over the finite values of a stage's records) is computed per checkpoint,
    one case score per retained checkpoint,
    then reduced by topk_statistics to <metric>_<stat> (the top-K mean), <metric>_<stat>_best and <metric>_<stat>_ckpt_std.
    Dims: stage (STAGE_AGG_NAMES).
    collect_metrics attaches the case-identifying coords, it knows the case_idx.
    """
    n_ckpt = len(ts_metrics.ckpt_epochs)
    per_ckpt_vars = {}
    for metric in metric_names:
        values = ts_metrics.metric(metric)
        stats = {stat: np.full((len(STAGE_AGG_NAMES), n_ckpt), np.nan) for stat in ("mean", "std", "med", "count")}
        for stage_pos, stage in enumerate(STAGE_AGG_NAMES):
            stage_values = values[ts_metrics.stage_mask(stage)]
            for ckpt_pos in range(n_ckpt):
                ckpt_values = stage_values[:, ckpt_pos]
                ckpt_values = ckpt_values[np.isfinite(ckpt_values)]
                stats["count"][stage_pos, ckpt_pos] = len(ckpt_values)
                if len(ckpt_values):
                    stats["mean"][stage_pos, ckpt_pos] = np.mean(ckpt_values)
                    stats["std"][stage_pos, ckpt_pos] = np.std(ckpt_values)
                    stats["med"][stage_pos, ckpt_pos] = np.median(ckpt_values)
        for stat, per_ckpt_values in stats.items():
            per_ckpt_vars[f"{metric}_{stat}"] = (("stage", CKPT_DIM), per_ckpt_values)

    per_ckpt = xr.Dataset(per_ckpt_vars, coords={"stage": list(STAGE_AGG_NAMES), CKPT_DIM: ts_metrics.ckpt_epochs})
    return topk_statistics(per_ckpt, ts_metrics.best_epoch)


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
