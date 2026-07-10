"""Stage-resolved performance metrics for profile transfer study results.

Recomputes the delta-free validation loss components (see
ProfilePredictorTRB._make_profile_loss_fn with use_huber=False) in numpy from
the per-case result files, joined back to the device datasets for measurement
error bars, GP-fit gradients, Ip, and auxiliary heating power. Each test
timeslice gets three metrics:

- metric_value: peak-normalized, error-bar-softened profile residual integrated
  over rho, summed over the ne and Te channels
- metric_grad: same for the profile gradients at the rho midpoints, masked to
  rho below GRAD_LOSS_RHO_MAX
- metric_combined: metric_value + gradient_weight * metric_grad, the same
  weighting the training loss uses

and a shot-stage label (rampup / flattop / rampdown, with an aux-heating flag
subdividing the flattop).
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from transport_study import EPISODE_DIM, TIME_DIM
from transport_study.config import config
from transport_study.modules.profile_predictor.trb import ProfilePredictorTRB
from transport_study.orchestration.organize_data import (
    INPUT_POWER_SIGNALS,
    PROFILE_BASE_SIGNALS,
    PROFILE_ERROR_SIGNALS,
    PROFILE_TARGET_VARS,
)
from transport_study.orchestration.study import Study

# Flattop is the contiguous span where |Ip| is at least this fraction of the
# shot's 95th percentile |Ip|. Rampup is everything before, rampdown after
FLATTOP_IP_FRACTION = 0.9

# Auxiliary heating (NBI/LH/ECRH/ICRF) above this total power counts as
# significant, splitting the flattop into ohmic and aux-heated timeslices
AUX_SIGNIFICANT_MW = 0.1

# Joined eval timeslice must be within this of the result timeslice.
# Timebases are 1 kHz, so anything beyond half a sample is a bad join
TIME_JOIN_TOLERANCE_S = 6e-4

# The uniform rho grid the profile transfer workflow trains on
# (see organize_data.get_ds._profile_transfer)
RHO_GRID = np.linspace(0, 1, 51)

METRIC_NAMES = ("value", "grad", "combined")
STAGE_AGG_NAMES = ("all", "rampup", "flattop", "flattop_ohmic", "flattop_aux", "rampdown")

COLLECTED_METRICS_FILENAME = "collected_metrics.nc"
CASE_METRICS_FILENAME = "case_metrics.nc"


@cache
def load_eval_dataset(device: str) -> xr.Dataset:
    """Device dataset with the signals needed to score result timeslices.

    Profile signals and their gradient / error-bar companions are put on the
    same uniform 51-point rho grid as training (mirrors the preprocessing in
    organize_data.get_ds). Ip, auxiliary heating powers, and the time coord are
    kept for stage segmentation. Non-fresh timeslices are kept: the join is by
    time, so only timeslices that appear in a result file are ever read.
    """
    ds_path = Path(config.dataset_paths[device])
    ds = xr.open_dataset(ds_path)

    for base in PROFILE_BASE_SIGNALS:
        if f"{base}_grad" not in ds:
            ds[f"{base}_grad"] = ds[base].differentiate("rho")
        for err in (f"{base}_error", f"{base}_grad_error"):
            if err not in ds:
                ds[err] = xr.zeros_like(ds[base])

    for sig in INPUT_POWER_SIGNALS:
        if sig not in ds:
            ds[sig] = xr.zeros_like(ds["Ip_MA"])

    keep = [*PROFILE_TARGET_VARS, *INPUT_POWER_SIGNALS, "Ip_MA", "time"]
    if "fresh_profiles" in ds:
        keep.append("fresh_profiles")
    ds = ds[keep]

    ds = ds.interp(rho=RHO_GRID, kwargs={"fill_value": "extrapolate"})
    for err_sig in PROFILE_ERROR_SIGNALS:
        ds[err_sig] = ds[err_sig].clip(min=0.0)

    return ds.load()


def segment_stages(ip_ma: np.ndarray, p_aux_mw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label each timeslice of one shot as rampup / flattop / rampdown.

    Flattop is the contiguous index span between the first and last timeslice
    where |Ip| >= FLATTOP_IP_FRACTION * p95(|Ip|). Timeslices with NaN Ip get
    an empty stage label and are excluded from the stage aggregations.

    Returns:
        (stage, aux_heated): stage is an array of "rampup" / "flattop" /
        "rampdown" / "" labels, aux_heated a boolean array marking timeslices
        where the total auxiliary heating power exceeds AUX_SIGNIFICANT_MW.
    """
    abs_ip = np.abs(ip_ma)
    stage = np.full(abs_ip.shape, "", dtype=object)
    valid = np.isfinite(abs_ip)
    aux_heated = np.nan_to_num(p_aux_mw, nan=0.0) > AUX_SIGNIFICANT_MW

    if not valid.any():
        return stage.astype(str), aux_heated

    ip_p95 = np.nanpercentile(abs_ip, 95)
    at_flattop = valid & (abs_ip >= FLATTOP_IP_FRACTION * ip_p95)
    if not at_flattop.any():
        # Degenerate Ip trace, call every valid timeslice rampup
        stage[valid] = "rampup"
        return stage.astype(str), aux_heated

    flattop_idxs = np.flatnonzero(at_flattop)
    start, end = flattop_idxs[0], flattop_idxs[-1]
    idxs = np.arange(abs_ip.shape[0])
    stage[valid & (idxs < start)] = "rampup"
    stage[valid & (idxs >= start) & (idxs <= end)] = "flattop"
    stage[valid & (idxs > end)] = "rampdown"
    return stage.astype(str), aux_heated


def _to_mid(arr: np.ndarray) -> np.ndarray:
    """Grid-point signal averaged to the rho midpoints."""
    return 0.5 * (arr[..., :-1] + arr[..., 1:])


def _channel_value_metric(pred: np.ndarray, targ: np.ndarray, err: np.ndarray, rho: np.ndarray, within_error_weight: float) -> np.ndarray:
    """Peak-normalized softened value residual integrated over rho. Shapes (..., rho)."""
    floor = ProfilePredictorTRB.PROFILE_SCALE_FLOOR
    scale = np.maximum(np.max(np.abs(targ), axis=-1, keepdims=True), floor)
    abs_residual = np.abs(pred / scale - targ / scale)
    sigma = np.maximum(err / scale, 0.0)
    outside = np.maximum(abs_residual - sigma, 0.0)
    inside = np.minimum(abs_residual, sigma)
    softened = outside + within_error_weight * inside
    return np.trapezoid(softened, x=rho, axis=-1)


def _channel_grad_metric(
    pred: np.ndarray, targ: np.ndarray, grad_targ: np.ndarray, grad_err: np.ndarray, rho: np.ndarray, within_error_weight: float
) -> np.ndarray:
    """Softened gradient residual at the rho midpoints, masked to
    rho < GRAD_LOSS_RHO_MAX, integrated over rho. Shapes (..., rho)."""
    floor = ProfilePredictorTRB.PROFILE_SCALE_FLOOR
    scale = np.maximum(np.max(np.abs(targ), axis=-1, keepdims=True), floor)

    d_rho = np.diff(rho)
    rho_mid = 0.5 * (rho[:-1] + rho[1:])

    grad_pred = np.diff(pred / scale, axis=-1) / d_rho
    grad_targ_mid = _to_mid(grad_targ) / scale
    sigma = np.maximum(_to_mid(grad_err) / scale, 0.0)

    abs_residual = np.abs(grad_pred - grad_targ_mid)
    outside = np.maximum(abs_residual - sigma, 0.0)
    inside = np.minimum(abs_residual, sigma)
    softened = outside + within_error_weight * inside

    grad_rho_mask = rho_mid < ProfilePredictorTRB.GRAD_LOSS_RHO_MAX
    return np.trapezoid(grad_rho_mask * softened, x=rho_mid, axis=-1)


@dataclass
class CaseTimesliceMetrics:
    """Long-form per-timeslice metrics for one case, one record per valid test
    timeslice. result_time_idx / eval_time_idx are positional indices into the
    case result file and the device eval dataset respectively, so callers can
    fetch the corresponding profiles and error bars."""

    shot: np.ndarray
    ds_source: np.ndarray
    time: np.ndarray
    result_time_idx: np.ndarray
    eval_time_idx: np.ndarray
    stage: np.ndarray
    aux_heated: np.ndarray
    metric_value: np.ndarray
    metric_grad: np.ndarray
    metric_combined: np.ndarray

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
        return getattr(self, f"metric_{name}")


def compute_case_timeslice_metrics(result_ds: xr.Dataset, loss_config: dict) -> CaseTimesliceMetrics:
    """Score every valid test timeslice of one case result file.

    Joins each result timeslice to its device dataset by (shot, nearest time)
    to pick up the measurement error bars, GP gradients, Ip, and aux power the
    result file does not carry.
    """
    gradient_weight = loss_config.get("gradient_weight", 0.0)
    within_error_weight = loss_config.get("within_error_weight", ProfilePredictorTRB.WITHIN_ERROR_WEIGHT)

    rho = result_ds["rho"].values

    records: dict[str, list] = {
        name: [] for name in ("shot", "ds_source", "time", "result_time_idx", "eval_time_idx", "stage", "aux_heated")
    }
    metric_lists: dict[str, list] = {name: [] for name in METRIC_NAMES}

    for shot_pos, shot in enumerate(result_ds[EPISODE_DIM].values):
        shot_res = result_ds.isel({EPISODE_DIM: shot_pos})
        device = str(shot_res["ds_source"].values)
        eval_ds = load_eval_dataset(device)
        if shot not in eval_ds[EPISODE_DIM].values:
            logger.warning(f"Shot {shot} not found in dataset for device {device}, skipping")
            continue
        shot_eval = eval_ds.sel({EPISODE_DIM: shot})
        if not np.allclose(shot_eval["rho"].values, rho):
            raise ValueError(f"rho grid mismatch between result file and {device} dataset")

        res_time = shot_res["time"].values
        ne_pred_all = shot_res["ne20_rho_pred"].transpose(TIME_DIM, "rho").values
        te_pred_all = shot_res["Te_keV_rho_pred"].transpose(TIME_DIM, "rho").values
        valid = np.isfinite(res_time) & ~np.all(np.isnan(ne_pred_all), axis=-1) & ~np.all(np.isnan(te_pred_all), axis=-1)
        result_idxs = np.flatnonzero(valid)
        if len(result_idxs) == 0:
            continue

        # Nearest-time join into the device dataset
        eval_time = shot_eval["time"].values
        finite_eval = np.isfinite(eval_time)
        eval_positions = np.flatnonzero(finite_eval)
        if len(eval_positions) == 0:
            logger.warning(f"Shot {shot} has no finite times in the {device} dataset, skipping")
            continue
        eval_times_finite = eval_time[eval_positions]
        order = np.argsort(eval_times_finite)
        sorted_times = eval_times_finite[order]
        sorted_positions = eval_positions[order]

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
        eval_idxs = sorted_positions[nearest_sorted[joined]]
        if len(result_idxs) == 0:
            continue

        # Stage labels over the full shot, then picked at the joined timeslices
        p_aux = sum(np.nan_to_num(shot_eval[sig].values, nan=0.0) for sig in INPUT_POWER_SIGNALS)
        stage_full, aux_full = segment_stages(shot_eval["Ip_MA"].values, p_aux)

        ne_pred = ne_pred_all[result_idxs]
        te_pred = te_pred_all[result_idxs]
        ne_targ = shot_res["ne20_rho_targ"].transpose(TIME_DIM, "rho").values[result_idxs]
        te_targ = shot_res["Te_keV_rho_targ"].transpose(TIME_DIM, "rho").values[result_idxs]

        ne_err = shot_eval["ne20_rho_error"].transpose(TIME_DIM, "rho").values[eval_idxs]
        te_err = shot_eval["Te_keV_rho_error"].transpose(TIME_DIM, "rho").values[eval_idxs]
        ne_grad = shot_eval["ne20_rho_grad"].transpose(TIME_DIM, "rho").values[eval_idxs]
        te_grad = shot_eval["Te_keV_rho_grad"].transpose(TIME_DIM, "rho").values[eval_idxs]
        ne_grad_err = shot_eval["ne20_rho_grad_error"].transpose(TIME_DIM, "rho").values[eval_idxs]
        te_grad_err = shot_eval["Te_keV_rho_grad_error"].transpose(TIME_DIM, "rho").values[eval_idxs]

        metric_value = _channel_value_metric(ne_pred, ne_targ, ne_err, rho, within_error_weight) + _channel_value_metric(
            te_pred, te_targ, te_err, rho, within_error_weight
        )
        metric_grad = _channel_grad_metric(ne_pred, ne_targ, ne_grad, ne_grad_err, rho, within_error_weight) + _channel_grad_metric(
            te_pred, te_targ, te_grad, te_grad_err, rho, within_error_weight
        )
        metric_combined = metric_value + gradient_weight * metric_grad

        n = len(result_idxs)
        records["shot"].append(np.full(n, shot))
        records["ds_source"].append(np.full(n, device, dtype=object))
        records["time"].append(res_time[result_idxs])
        records["result_time_idx"].append(result_idxs)
        records["eval_time_idx"].append(eval_idxs)
        records["stage"].append(stage_full[eval_idxs])
        records["aux_heated"].append(aux_full[eval_idxs])
        metric_lists["value"].append(metric_value)
        metric_lists["grad"].append(metric_grad)
        metric_lists["combined"].append(metric_combined)

    def _cat(chunks: list, dtype=None) -> np.ndarray:
        if not chunks:
            return np.array([], dtype=dtype if dtype is not None else float)
        return np.concatenate(chunks)

    return CaseTimesliceMetrics(
        shot=_cat(records["shot"]),
        ds_source=_cat(records["ds_source"], dtype=object).astype(str),
        time=_cat(records["time"]),
        result_time_idx=_cat(records["result_time_idx"], dtype=int),
        eval_time_idx=_cat(records["eval_time_idx"], dtype=int),
        stage=_cat(records["stage"], dtype=object).astype(str),
        aux_heated=_cat(records["aux_heated"], dtype=bool),
        metric_value=_cat(metric_lists["value"]),
        metric_grad=_cat(metric_lists["grad"]),
        metric_combined=_cat(metric_lists["combined"]),
    )


def collected_metrics_path(study: Study) -> Path:
    return study.result_dir / COLLECTED_METRICS_FILENAME


def case_metrics_path(study: Study, case) -> Path:
    """Per-case stage-aggregate cache, next to the case's result_data.nc.
    An empty dataset is the marker for 'computed, but no valid timeslices',
    so parallel analysis jobs can signal completion either way."""
    return study.result_path(case).parent / CASE_METRICS_FILENAME


def _aggregate_case_metrics(ts_metrics: CaseTimesliceMetrics) -> xr.Dataset:
    """Reduce one case's per-timeslice metrics to per-stage statistics.

    Dims: stage (STAGE_AGG_NAMES). Data variables <metric>_<stat> for metric in
    value/grad/combined and stat in mean/std/med/count. Case-identifying coords
    are attached later by collect_metrics, which knows the case_idx.
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
    loss_config = study.make_train_config(case).loss_config
    ts_metrics = compute_case_timeslice_metrics(result_ds, loss_config)
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
    <metric>_<stat> for metric in value/grad/combined and stat in
    mean/std/med/count. Case-identifying coords along case_idx match
    collect_results.

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

    metrics_ds = xr.concat(results, dim="case_idx")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_ds.to_netcdf(cache_path)
    logger.info(f"Saved stage-resolved metrics to {cache_path}")
    return metrics_ds
