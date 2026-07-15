"""Shared helpers for the study TrainRunBuilders."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from popsim.ml.eval import EvalData, EvaluationSuite, batched_model_eval_and_loss

from transport_study import TIME_DIM


def trapezoid_dropna(y, x):
    """Trapezoid-integrate y over x ignoring NaN pairs, NaN when fewer than 2 valid points."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    y_valid, x_valid = y[mask], x[mask]
    sort_idx = np.argsort(x_valid)
    return np.trapezoid(y_valid[sort_idx], x_valid[sort_idx])


def integrate_error_over_time(error_ts: xr.DataArray, time_2d: xr.DataArray) -> xr.DataArray:
    """Per-shot time integral of a per-timeslice error, ignoring NaN-padded entries."""
    return xr.apply_ufunc(
        trapezoid_dropna,
        error_ts,
        time_2d,
        input_core_dims=[[TIME_DIM], [TIME_DIM]],
        vectorize=True,
    )


def make_exponential_adamw(optimizer_config: dict) -> optax.GradientTransformation:
    """AdamW on an exponentially decaying learning rate schedule."""
    schedule = optax.exponential_decay(
        init_value=optimizer_config["lr0"],
        transition_steps=optimizer_config["transition_steps"],
        decay_rate=optimizer_config["decay_rate"],
        end_value=optimizer_config["lrf"],
    )
    return optax.adamw(learning_rate=schedule, weight_decay=optimizer_config["weight_decay"])


def make_loss_eval_suite(loss_fn) -> EvaluationSuite:
    """Validation suite computing the loss mean plus a <=100 point sampled loss vector.

    The datasets are large and uploading the full per-sample loss vector to
    wandb every validation is too much data, so the vector is sorted and
    subsampled evenly - the distribution stays visible without all the data.

    A fresh closure is created and jitted per suite so the compiled forward
    persists across every validation of a training run while its cache stays
    isolated from other cases in the same process. Equinox keys
    eqx.filter_jit's cache off the wrapped function's identity, so jitting a
    shared module-level function would let unrelated cases collide in the
    same cache entry, which can raise instead of just retracing.
    """

    def _eval_and_loss(model, loss_fn, inputs, targets):
        return batched_model_eval_and_loss(model, loss_fn, inputs, targets)

    jit_eval_and_loss = eqx.filter_jit(_eval_and_loss)

    def eval_fn(inp: EvalData) -> dict:
        loss_vecs = []
        for batch in inp.dataloader:
            inputs, targets = batch.get_inputs_and_targets()
            loss_vecs.append(jit_eval_and_loss(inp.model, loss_fn, inputs, targets))
        loss_vec = jnp.concatenate(loss_vecs)
        # Drop padded duplicate samples from the pad_last validation dataloader
        loss_vec = loss_vec[: inp.dataloader.dataset.n_samples]
        loss_vec_mean = loss_vec.mean()
        # Sort and sample at most 100 points evenly for logging
        if loss_vec.shape[0] > 100:
            sorted_indices = jnp.argsort(loss_vec)
            selected_indices = sorted_indices[jnp.linspace(0, loss_vec.shape[0] - 1, num=100, dtype=int)]
            loss_vec = loss_vec[selected_indices]
        return {
            "mean": loss_vec_mean,
            "vec": loss_vec,
        }

    return {"loss": eval_fn}
