"""Stage-resolved metrics of profile transfer study results (its ANALYSIS_METRICS_MODULE, see orchestration.case_metrics).

Scores every test timeslice with the chi validation loss (ProfilePredictorTRB.get_val_loss_fn, trb_utils.chi_value / chi_gradient),
joined back to the device datasets for the measurement error bars, GP-fit gradients, ip_MA and the heating powers.
Each test timeslice gets three metrics:

- metric_value: value chi integrated over rho, summed over the ne and Te channels
- metric_grad: gradient chi at the rho midpoints, masked to rho below GRAD_RHO_MAX, summed over the channels
- metric_combined: metric_value + gradient_weight * metric_grad, the validation loss itself

and a shot-stage label (rampup / flattop / rampdown, with an aux-heating flag subdividing the flattop).
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import ClassVar

import numpy as np
import xarray as xr
from loguru import logger

from transport_study import EPISODE_DIM, RADIAL_DIM, TIME_COORD, TIME_DIM
from transport_study.config import config
from transport_study.modules.trb_utils import CHI_ERROR_VARS, chi_gradient, chi_value
from transport_study.orchestration.case_metrics import (
    StagedTimesliceMetrics,
    concat_records,
    nearest_time_positions,
    shot_stages,
)
from transport_study.orchestration.organize_data import (
    PROFILE_TARGET_VARS,
    to_rho_grid,
)
from transport_study.orchestration.study import Study
from transport_study.signals import POWER_ADDITIONAL_MW, convert_to_working_units

METRIC_NAMES = ("value", "grad", "combined")


@cache
def load_eval_dataset(device: str) -> xr.Dataset:
    """Device dataset with the signals needed to score result timeslices.

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


def compute_case_timeslice_metrics(result_ds: xr.Dataset, loss_config: dict) -> CaseTimesliceMetrics:
    """Score every valid test timeslice of one case result file with the chi validation loss.

    Joins each result timeslice to its device dataset by (shot, nearest time)
    to pick up the measurement error bars, GP gradients, Ip, and aux power the
    result file does not carry.
    loss_config is the case's own, for gradient_weight and the per-device chi_sigma_floors.
    """
    gradient_weight = loss_config["gradient_weight"]
    sigma_floors = loss_config["chi_sigma_floors"]

    rho = result_ds[RADIAL_DIM].values

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
        if not np.allclose(shot_eval[RADIAL_DIM].values, rho):
            raise ValueError(f"rho grid mismatch between result file and {device} dataset")

        res_time = shot_res["time"].values
        ne_pred_all = shot_res["n_e_1e20_pred"].transpose(TIME_DIM, RADIAL_DIM).values
        te_pred_all = shot_res["t_e_keV_pred"].transpose(TIME_DIM, RADIAL_DIM).values
        valid = np.isfinite(res_time) & ~np.all(np.isnan(ne_pred_all), axis=-1) & ~np.all(np.isnan(te_pred_all), axis=-1)
        result_idxs = np.flatnonzero(valid)
        if len(result_idxs) == 0:
            continue

        mask_joined, eval_idxs = nearest_time_positions(shot_eval["time"].values, res_time[result_idxs], shot, device)
        result_idxs = result_idxs[mask_joined]
        if len(result_idxs) == 0:
            continue

        # Stage labels over the full shot, then picked at the joined timeslices
        stage_full, aux_full = shot_stages(shot_eval)

        preds = {"n_e_1e20": ne_pred_all[result_idxs], "t_e_keV": te_pred_all[result_idxs]}
        device_floors = sigma_floors[device]
        metric_value = np.zeros(len(result_idxs))
        metric_grad = np.zeros(len(result_idxs))
        for channel, (value_error_var, grad_error_var) in CHI_ERROR_VARS.items():
            targ, sigma, grad_targ, grad_sigma = (
                profiles.transpose(TIME_DIM, RADIAL_DIM).values[idxs]
                for profiles, idxs in (
                    (shot_res[f"{channel}_targ"], result_idxs),
                    (shot_eval[value_error_var], eval_idxs),
                    (shot_eval[f"{channel}_gradient"], eval_idxs),
                    (shot_eval[grad_error_var], eval_idxs),
                )
            )
            metric_value += np.asarray(chi_value(preds[channel], targ, sigma, device_floors[value_error_var], rho))
            metric_grad += np.asarray(chi_gradient(preds[channel], targ, grad_targ, grad_sigma, device_floors[grad_error_var], rho))
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

    return CaseTimesliceMetrics(
        shot=concat_records(records["shot"]),
        ds_source=concat_records(records["ds_source"], dtype=object).astype(str),
        time=concat_records(records["time"]),
        result_time_idx=concat_records(records["result_time_idx"], dtype=int),
        eval_time_idx=concat_records(records["eval_time_idx"], dtype=int),
        stage=concat_records(records["stage"], dtype=object).astype(str),
        aux_heated=concat_records(records["aux_heated"], dtype=bool),
        metric_value=concat_records(metric_lists["value"]),
        metric_grad=concat_records(metric_lists["grad"]),
        metric_combined=concat_records(metric_lists["combined"]),
    )


def case_timeslice_metrics(study: Study, case, result_ds: xr.Dataset) -> CaseTimesliceMetrics:
    """Scored with the case's own loss config (gradient_weight and the per-device chi_sigma_floors)."""
    loss_config = study.make_train_config(case).loss_config
    return compute_case_timeslice_metrics(result_ds, loss_config)
