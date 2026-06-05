"""Tests for ProfilePredictorTorax training pipeline.

Covers model initialization (which JIT-compiles the TORAX step function),
a single forward pass through TORAX, and a short Stage 2 training run using
synthetic ``fitted_params`` labels (Stage 1 offline fitting is skipped to keep
the test fast).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import xarray as xr

from transport_study import TIME_COORD
from transport_study.modules.profile_predictor.module import Inputs
from transport_study.modules.profile_predictor.torax_module import (
    _NN_OUT_SIZE,
    ProfilePredictorTorax,
)

# ── Shared synthetic dataset ───────────────────────────────────────────────────

_N_SHOTS = 3
_N_TIMES = 2  # time slices per shot (real datasets have multiple per shot)
_N_PSI = 13
_PSI_N = np.linspace(0.0, 1.1, _N_PSI)


def _make_synthetic_dataset() -> xr.Dataset:
    """Build a minimal in-memory dataset with synthetic profiles and fitted_params.

    Structure mirrors the real profile-predictor dataset: ``(shot, time)`` for
    scalars and ``(shot, time, psi_n)`` for profiles.  ``make_dataloaders``
    stacks ``(shot, time)`` into flat samples.

    ``fitted_params`` plays the role of Stage-1 output: random values in the
    raw NN-output space (log/logit; zeros ≈ all-ones in physical space).
    """
    rng = np.random.default_rng(0)
    shape = (_N_SHOTS, _N_TIMES)

    def scalar(lo, hi):
        return xr.DataArray(rng.uniform(lo, hi, shape), dims=["shot", TIME_COORD])

    def profile(lo, hi):
        return xr.DataArray(
            rng.uniform(lo, hi, (*shape, _N_PSI)),
            dims=["shot", TIME_COORD, "psi_n"],
        )

    return xr.Dataset(
        {
            "Ip_MA": scalar(0.5, 2.0),
            "B0": scalar(1.0, 6.0),
            "betan": scalar(0.5, 3.0),
            "ne20_edge": scalar(0.5, 5.0),
            "R0": scalar(0.5, 1.0),
            "a_minor": scalar(0.15, 0.25),
            "kappa": scalar(1.3, 1.8),
            "delta_top": scalar(0.0, 0.4),
            "delta_bot": scalar(0.0, 0.4),
            "ne20_psi": profile(0.5, 5.0),
            "Te_keV_psi": profile(0.1, 5.0),
            # Synthetic fitted_params — skips Stage 1
            "fitted_params": xr.DataArray(
                rng.normal(0.0, 0.3, (*shape, _NN_OUT_SIZE)).astype(np.float32),
                dims=["shot", TIME_COORD, "fitted_param"],
            ),
        },
        coords={
            "shot": np.arange(_N_SHOTS),
            TIME_COORD: np.arange(_N_TIMES, dtype=float),
            "psi_n": _PSI_N,
        },
    )


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def small_model():
    """ProfilePredictorTorax with a tiny NN and coarse grid for fast testing."""
    return ProfilePredictorTorax.init(
        psigrid=_PSI_N,
        nn_width=8,
        nn_depth=2,
        dt_steady=5.0,
        n_rho=10,
        t_edge_keV=0.2,
    )


@pytest.fixture(scope="module")
def sample_inputs():
    ds = _make_synthetic_dataset()
    s = ds.isel(shot=0, time=0)  # pick one (shot, time) sample
    return Inputs(
        Ip=float(s["Ip_MA"]),
        B0=float(s["B0"]),
        betan=float(s["betan"]),
        ne20=float(s["ne20_edge"]),
        R0=float(s["R0"]),
        a_minor=float(s["a_minor"]),
        kappa=float(s["kappa"]),
        delta_top=float(s["delta_top"]),
        delta_bot=float(s["delta_bot"]),
        psi=jnp.array(_PSI_N),
    )


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_model_init():
    """Model initialisation compiles the TORAX step function without error."""
    model = ProfilePredictorTorax.init(
        psigrid=_PSI_N,
        nn_width=8,
        nn_depth=2,
        dt_steady=5.0,
        n_rho=10,
        t_edge_keV=0.2,
    )
    assert model._step_fn is not None
    assert model._base_provider is not None
    assert model._ref_torax_mesh is not None
    assert len(model.psigrid) == _N_PSI


def test_forward_pass(small_model, sample_inputs):
    """Single forward pass returns finite Te and ne profiles on the output grid."""
    out = small_model(sample_inputs)

    assert out.te.shape == (_N_PSI,), f"Expected ({_N_PSI},), got {out.te.shape}"
    assert out.ne.shape == (_N_PSI,), f"Expected ({_N_PSI},), got {out.ne.shape}"
    assert np.all(np.isfinite(out.te.values)), "Te contains non-finite values"
    assert np.all(np.isfinite(out.ne.values)), "ne contains non-finite values"
    assert np.all(out.te.values > 0), "Te should be positive"
    assert np.all(out.ne.values > 0), "ne should be positive"


def test_get_transport_params(small_model, sample_inputs):
    """Transport parameter extraction returns the correct keys and finite values."""
    params = small_model.get_transport_params(sample_inputs)
    expected_keys = {
        "chi_e",
        "chi_i",
        "source_rho",
        "source_width",
        "D_e",
        "ne_peaking",
        "P_scale",
        "P_physics",
        "P_total",
    }
    assert expected_keys == set(params.keys())
    for key, val in params.items():
        assert np.isfinite(val), f"Transport param {key} = {val} is not finite"


def test_stage2_training(small_model):
    """Stage 2 training: the NN weights update correctly via gradient descent.

    The full ``ProfilePredictorTorax.__call__`` runs TORAX, which is not
    vmap-compatible.  Stage 2 only requires gradient updates on the NN sub-
    module (not through TORAX), so this test drives the NN directly.

    Stage 1 (offline TORAX fitting) is skipped — ``fitted_params`` are random
    values that serve as supervised targets.  The test verifies:
      - Gradient computation through the NN succeeds (no tracer errors)
      - The MSE loss is finite before and after training
      - NN weights change after taking gradient steps
    """
    rng = np.random.default_rng(42)
    n_samples = 6

    # Random (inputs, fitted_params) pairs in the raw NN output space
    nn_inputs_batch = jnp.array(rng.normal(0.0, 0.5, (n_samples, 9)).astype(np.float32))
    fitted_params = jnp.array(rng.normal(0.0, 0.3, (n_samples, _NN_OUT_SIZE)).astype(np.float32))

    @eqx.filter_jit
    def loss_and_grad(nn_module):
        """MSE over the batch — same objective as ProfilePredictorToraxTRB.get_loss_fn."""

        def per_sample(x):
            return nn_module(x)

        preds = jax.vmap(per_sample)(nn_inputs_batch)  # (n_samples, 7)
        return jnp.mean((preds - fitted_params) ** 2)

    grad_fn = eqx.filter_grad(loss_and_grad)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(small_model.nn, eqx.is_array))

    nn_before = small_model.nn
    loss_before = float(loss_and_grad(nn_before))
    assert np.isfinite(loss_before), f"Initial loss is not finite: {loss_before}"

    # Take a few gradient steps on the NN
    nn = small_model.nn
    for _ in range(5):
        grads = grad_fn(nn)
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array),
            opt_state,
            eqx.filter(nn, eqx.is_array),
        )
        nn = eqx.apply_updates(nn, updates)

    loss_after = float(loss_and_grad(nn))
    assert np.isfinite(loss_after), f"Final loss is not finite: {loss_after}"

    # NN weights should have changed
    leaves_before = jnp.concatenate([x.ravel() for x in jax.tree.leaves(nn_before)])
    leaves_after = jnp.concatenate([x.ravel() for x in jax.tree.leaves(nn)])
    assert not jnp.allclose(leaves_before, leaves_after), "NN weights did not change after training"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
