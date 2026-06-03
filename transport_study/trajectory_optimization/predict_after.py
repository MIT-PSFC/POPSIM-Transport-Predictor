"""Post-shot profile prediction: evaluate the model over perturbed measured trajectories.

Like predict_first, but the center of the perturbation ensemble is the *measured*
trajectory of the shot rather than its programmed trajectory.  This gives a
retrospective answer to the question "how well can the model predict the profiles
that were actually observed?"

Usage:
    python predict_after.py run_postshot_prediction \
        --shot 206364 \
        --profile_predictor_checkpoint_dir /path/to/checkpoint
"""

import os

import fire
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from loguru import logger

from transport_study.config import config
from transport_study.datasets.d3d.d3d_dataset import INNER_WALL
from transport_study.modules.profile_predictor.module import Inputs
from transport_study.modules.profile_trajectory.data import (
    add_gapin_prog,
    correct_B0_prog,
)
from transport_study.profile_transfer.restore_predictor import (
    restore_profile_predictor_from_checkpoint,
)
from transport_study.trajectory_optimization.predict_first import (
    _make_profile_gif,
    _plot_prediction_vs_measured,
    _plot_predictor_inputs,
    get_traj_shot_data,
)
from transport_study.trajectory_optimization.setup_data import make_augmented_dataset

# Mapping from measured variable → programmed variable (and optional unit scale factor)
# Used to overwrite the programmed trajectory with the measured one.
_MEAS_TO_PROG: list[tuple[str, str, float]] = [
    ("Ip_MA", "Ip_MA_prog", 1.0),
    ("B0", "B0_prog", 1.0),
    ("betan", "betan_prog", 1.0),
    ("ne20_edge", "ne20_edge_prog", 1.0),
    ("R0", "R0_prog", 1.0),
    ("rxbot", "rxbot_prog", 1.0),
    ("zxbot", "zxbot_prog", 1.0),
    ("rxtop", "rxtop_prog", 1.0),
    ("zxtop", "zxtop_prog", 1.0),
]


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill NaN in a 1-D array.

    Measured signals (EFIT, etc.) are sampled at lower frequency than the full
    time grid and typically contain NaN at intermediate time steps.  Filling
    ensures make_augmented_dataset does not drop those time steps.
    """
    arr = arr.copy().astype(float)
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return arr
    valid = np.where(~nan_mask)[0]
    if len(valid) == 0:
        return arr
    # forward-fill then backward-fill via nearest-neighbour extrapolation
    arr[nan_mask] = arr[valid[np.searchsorted(valid, np.where(nan_mask)[0]).clip(0, len(valid) - 1)]]
    # backward-fill any leading NaN (where searchsorted returns 0 for indices before valid[0])
    still_nan = np.isnan(arr)
    if still_nan.any():
        arr[still_nan] = arr[valid[0]]
    return arr


def _overwrite_prog_with_measured(ds: xr.Dataset, shot: int) -> xr.Dataset:
    """Replace all programmed trajectory variables with their measured equivalents.

    For shape control signals (rxbot/zxbot/rxtop/zxtop) the measured R-Z positions
    are used directly.  gapin_prog is derived from the measured a_minor:
        gapin = R0 - a_minor - INNER_WALL

    All measured values are forward/backward-filled to match the full time grid
    before assignment, because EFIT and other diagnostics are sampled at lower
    frequency and would otherwise leave NaN gaps that make_augmented_dataset drops.
    """
    updates: dict[str, xr.DataArray] = {}

    for meas_var, prog_var, scale in _MEAS_TO_PROG:
        if meas_var in ds and prog_var in ds:
            orig = ds[prog_var].load()
            new_vals = orig.copy()
            filled = _fill_nan(ds[meas_var].sel(shot=shot).values.astype(float)) * scale
            new_vals.loc[{"shot": shot}] = filled
            updates[prog_var] = new_vals

    # Derive gapin_prog from measured R0 and a_minor so that the augmented shape
    # variables are consistent with what was actually measured in the plasma.
    if "R0" in ds and "a_minor" in ds and "gapin_prog" in ds:
        R0_meas = _fill_nan(ds["R0"].sel(shot=shot).values)
        a_minor_meas = _fill_nan(ds["a_minor"].sel(shot=shot).values)
        gapin_meas = R0_meas - a_minor_meas - INNER_WALL
        orig_gapin = ds["gapin_prog"].load()
        new_gapin = orig_gapin.copy()
        new_gapin.loc[{"shot": shot}] = gapin_meas
        updates["gapin_prog"] = new_gapin

    return ds.assign(updates)


def run_postshot_prediction(
    shot: int,
    profile_predictor_checkpoint_dir: str,
    scratch_dir: str | None = config.scratch_dir,
):
    """Evaluate the profile predictor over an ensemble of perturbed *measured* trajectories.

    This is the post-shot counterpart to run_preshot_prediction.  Instead of
    perturbing around the programmed waveform, the perturbation ensemble is
    centered on what was actually measured during the discharge.  This allows
    a direct comparison of the predicted profile distribution against the
    measured profiles.

    Args:
        shot: Shot number to evaluate.
        profile_predictor_checkpoint_dir: Checkpoint directory for the profile predictor.
        scratch_dir: Scratch directory for output files.
    """
    working_dir = os.path.join(scratch_dir, "predict_first")
    shot_data_dir = os.path.join(working_dir, "raw_data")
    result_dir = os.path.join(
        working_dir,
        f"after_shot_{shot}",
        os.path.basename(profile_predictor_checkpoint_dir),
    )
    os.makedirs(shot_data_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Load (or fetch) shot data
    ds_path = os.path.join(shot_data_dir, f"{shot}.nc")
    ds = get_traj_shot_data(shot, ds_path)

    # Build programmed-trajectory variables first (gapin_prog needs B0_prog to exist)
    ds = correct_B0_prog(ds)
    ds = add_gapin_prog(ds)

    # Overwrite programmed trajectory with measured — this centres the augmentation
    # ensemble on the actual measured discharge rather than the pre-shot plan.
    logger.info(f"Replacing programmed trajectory with measured values for shot {shot}")
    ds = _overwrite_prog_with_measured(ds, shot)

    # Step 1: Make augmented dataset with perturbed (measured) inputs
    logger.info("Building augmented dataset with perturbed measured inputs")
    ds_aug = make_augmented_dataset(ds)

    for var in ds_aug.data_vars:
        if ds_aug[var].isnull().any() and var.endswith("_prog"):
            logger.critical(f"Variable {var} contains NaNs after augmentation!")

    # Step 2: Derive shape variables from the (now measurement-centred) prog signals
    R0_prog = ds_aug["R0_prog"].values
    gapin_prog = ds_aug["gapin_prog"].values
    rxbot_prog = ds_aug["rxbot_prog"].values
    zxbot_prog = ds_aug["zxbot_prog"].values
    rxtop_prog = ds_aug["rxtop_prog"].values
    zxtop_prog = ds_aug["zxtop_prog"].values

    a_minor = R0_prog - gapin_prog - INNER_WALL
    kappa = np.abs(zxtop_prog - zxbot_prog) / (a_minor * 2)
    delta_bot = (R0_prog - rxbot_prog) / a_minor
    delta_top = (R0_prog - rxtop_prog) / a_minor

    # Step 3: Restore profile predictor
    profile_predictor = restore_profile_predictor_from_checkpoint(profile_predictor_checkpoint_dir)

    # Build batched Inputs for vmap
    n_shot_alt, n_time = R0_prog.shape
    psi_grid = np.array(profile_predictor.psigrid)
    n_psi = len(psi_grid)

    Ip_flat = ds_aug["Ip_MA_prog"].values.reshape(-1)
    valid_mask = ~np.isnan(Ip_flat)

    psi_tiled = jnp.tile(jnp.array(psi_grid), (n_shot_alt * n_time, 1))

    inputs_batched = Inputs(
        Ip=ds_aug["Ip_MA_prog"].values.reshape(-1),
        B0=ds_aug["B0_prog"].values.reshape(-1),
        betan=ds_aug["betan_prog"].values.reshape(-1),
        ne20=ds_aug["ne20_edge_prog"].values.reshape(-1),
        R0=R0_prog.reshape(-1),
        a_minor=a_minor.reshape(-1),
        kappa=kappa.reshape(-1),
        delta_top=delta_top.reshape(-1),
        delta_bot=delta_bot.reshape(-1),
        psi=psi_tiled,
    )

    # Step 4: Run profile predictor
    logger.info(f"Running profile predictor on {n_shot_alt * n_time} samples ({n_shot_alt} shot_alts x {n_time} time steps)")

    def _predict(inputs: Inputs):
        outputs = profile_predictor(inputs)
        return outputs.ne.data, outputs.te.data

    ne_flat, te_flat = jax.vmap(_predict)(inputs_batched)

    valid_jnp = jnp.array(valid_mask)[:, None]
    ne_flat = jnp.where(valid_jnp, ne_flat, jnp.nan)
    te_flat = jnp.where(valid_jnp, te_flat, jnp.nan)

    ne_arr = np.array(ne_flat).reshape(n_shot_alt, n_time, n_psi)
    te_arr = np.array(te_flat).reshape(n_shot_alt, n_time, n_psi)

    # Step 5: Save output dataset
    ds_pred = xr.Dataset(
        {
            "ne": (["shot_alt", "time_idx", "psi_n"], ne_arr),
            "te": (["shot_alt", "time_idx", "psi_n"], te_arr),
        },
        coords={
            "shot_alt": ds_aug["shot_alt"].values,
            "time_idx": ds_aug["time_idx"].values,
            "psi_n": psi_grid,
        },
    )
    if "time" in ds_aug:
        ds_pred["time"] = ds_aug["time"]

    pred_path = os.path.join(result_dir, "predicted_profiles.nc")
    ds_pred.to_netcdf(pred_path)
    logger.info(f"Saved predicted profiles to {pred_path}")

    # Step 6: Plots — reuse the same helpers as predict_first
    _plot_predictor_inputs(ds_aug, a_minor, kappa, delta_top, delta_bot, result_dir)
    _plot_prediction_vs_measured(ds_pred, ds, shot, result_dir)
    _make_profile_gif(ds_pred, ds, shot, result_dir)


if __name__ == "__main__":
    fire.Fire(
        {
            "run_postshot_prediction": run_postshot_prediction,
        }
    )
