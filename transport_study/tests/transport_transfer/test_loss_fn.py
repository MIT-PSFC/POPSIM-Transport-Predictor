"""Tests for the transport predictor training and validation losses.

Hand-built predictions and targets so the expected values are exact. Covers
the freshness masking: the time-dependent rollouts keep forward-filled
(stale) profile timeslices in the data for segment contiguity, so both
losses must zero them out, the anchor terms that are exempt from it,
the value chi of the validation loss and the gradient terms of both losses.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from transport_study import RADIAL_DIM
from transport_study.config import RHO_GRID, StudyConfig, load_config
from transport_study.modules.transport_predictor.module import Output
from transport_study.modules.transport_predictor.trb import TransportPredictorTRB
from transport_study.modules.trb_utils import (
    CHI_ERROR_VARS,
    GRAD_RHO_MAX,
    to_mid,
)

RHO = RHO_GRID
RHO_MID = np.asarray(to_mid(RHO))
# Peak-normalized error bar of every target profile, far above SIGMA_FLOOR
SIGMA_FRAC = 0.05
SIGMA_FLOOR = 1e-3
LOSS_CONFIG = {
    "huber_delta": 0.1,
    "huber_delta_grad": 1.0,
    "gradient_weight": 0.1,
    "divergence_penalty": 10.0,
    "divergence_penalty_val": 1e4,
    "anchor_weight_energy_mhd": 0.0,
    "anchor_weight_power_ohm": 0.0,
    "anchor_weight_power_radiated": 0.0,
}


@pytest.fixture(autouse=True)
def loaded_config():
    load_config(
        StudyConfig(
            study_name="test_transport_loss_fn",
            dataset_paths={
                "cmod-low": Path("path/to/cmod_low.nc"),
                "cmod-high": Path("path/to/cmod_high.nc"),
            },
            target_device="cmod-high",
        )
    )


def _loss_config(**overrides) -> dict:
    floors = {var: SIGMA_FLOOR for error_vars in CHI_ERROR_VARS.values() for var in error_vars}
    return {**LOSS_CONFIG, "chi_sigma_floors": dict.fromkeys(("cmod-low", "cmod-high"), floors), **overrides}


def _error_bars(profile: np.ndarray) -> np.ndarray:
    return np.full_like(profile, SIGMA_FRAC * np.max(np.abs(profile)))


# Target profiles (quadratic, so the finite-difference gradient at a midpoint is exact) and their analytic gradients
NE_TARG = 1.5 * (1 - RHO**2) + 0.5
TE_TARG = 3.0 * (1 - RHO**2) + 0.5
NE_GRAD = -3.0 * RHO
TE_GRAD = -6.0 * RHO


def _profile_targets(dims: tuple[str, ...], n_rows: int | None = None, grad_offset: float = 0.0) -> dict:
    """The profile target variables, the GP-fit gradients offset by grad_offset, optionally stacked n_rows times along time."""
    profiles = {
        "n_e_1e20": NE_TARG,
        "t_e_keV": TE_TARG,
        "n_e_1e20_error": _error_bars(NE_TARG),
        "t_e_keV_error": _error_bars(TE_TARG),
        "n_e_1e20_gradient": NE_GRAD + grad_offset,
        "t_e_keV_gradient": TE_GRAD + grad_offset,
        "n_e_1e20_gradient_error": _error_bars(NE_GRAD),
        "t_e_keV_gradient_error": _error_bars(TE_GRAD),
    }
    if n_rows is not None:
        profiles = {name: np.stack([values] * n_rows) for name, values in profiles.items()}
    return {name: xr.DataArray(values, dims=dims) for name, values in profiles.items()}


def _pred_and_targ(fresh: float, offset: float = 0.3, grad_offset: float = 0.0, **pred_extra):
    """Single-timeslice prediction/target pair with a nonzero profile residual.

    fresh sets the fresh_profile flag on the target side. pred_extra sets
    extra Output fields by name (the sciml anchor predictions).
    """
    pred = Output(
        ne=jnp.asarray(NE_TARG + offset),
        te=jnp.asarray(TE_TARG - offset),
        rho=jnp.asarray(RHO),
        **pred_extra,
    )
    targ = {
        **_profile_targets((RADIAL_DIM,), grad_offset=grad_offset),
        "ds_source_idx": xr.DataArray(0.0),
        "fresh_profile": xr.DataArray(float(fresh)),
    }
    return pred, targ


def test_stale_timeslice_contributes_zero_loss():
    """A slice whose profiles are forward-filled (fresh_profile 0) must steer
    neither training nor checkpoint selection, whatever its residual."""
    loss_config = _loss_config()
    train_loss = TransportPredictorTRB.get_loss_fn(loss_config).instantaneous_loss
    val_loss = TransportPredictorTRB.get_val_loss_fn(loss_config).instantaneous_loss

    pred_fresh, targ_fresh = _pred_and_targ(fresh=1.0)
    pred_stale, targ_stale = _pred_and_targ(fresh=0.0)

    assert float(train_loss(pred_fresh, targ_fresh)) > 0.0
    assert float(val_loss(pred_fresh, targ_fresh)) > 0.0
    assert float(train_loss(pred_stale, targ_stale)) == 0.0
    assert float(val_loss(pred_stale, targ_stale)) == 0.0


def test_fresh_mask_zeroes_only_stale_slices():
    """With a time axis the mask acts elementwise: two identical slices where
    one is stale average to exactly half the all-fresh loss."""
    pred = Output(
        ne=jnp.stack([jnp.asarray(NE_TARG + 0.3)] * 2),
        te=jnp.stack([jnp.asarray(TE_TARG - 0.3)] * 2),
        rho=jnp.asarray(RHO),
    )

    def targ_with_fresh(fresh_flags):
        return {
            **_profile_targets(("time_idx", RADIAL_DIM), n_rows=2, grad_offset=0.5),
            "ds_source_idx": xr.DataArray([0.0, 0.0], dims=("time_idx",)),
            "fresh_profile": xr.DataArray(fresh_flags, dims=("time_idx",)),
        }

    loss_config = _loss_config()
    for loss in (
        TransportPredictorTRB.get_loss_fn(loss_config).instantaneous_loss,
        TransportPredictorTRB.get_val_loss_fn(loss_config).instantaneous_loss,
    ):
        all_fresh = float(loss(pred, targ_with_fresh([1.0, 1.0])))
        half_fresh = float(loss(pred, targ_with_fresh([1.0, 0.0])))
        assert all_fresh > 0.0
        assert half_fresh == pytest.approx(0.5 * all_fresh, rel=1e-6)


def test_anchor_terms_exempt_from_fresh_mask():
    """The sciml anchor signals are measured at every timeslice, so a stale
    profile slice keeps its anchor loss: with fresh_profile 0 the training
    loss reduces to exactly the weighted anchor errors, and the validation
    loss (no anchors) stays zero."""
    loss_config = _loss_config(anchor_weight_energy_mhd=0.2, anchor_weight_power_ohm=0.3, anchor_weight_power_radiated=0.5)
    train_loss = TransportPredictorTRB.get_loss_fn(loss_config).instantaneous_loss
    val_loss = TransportPredictorTRB.get_val_loss_fn(loss_config).instantaneous_loss

    pred, targ = _pred_and_targ(
        fresh=0.0,
        energy_mhd_MJ_pred=jnp.asarray(0.10),
        power_ohm_MW_pred=jnp.asarray(1.5),
        power_radiated_MW_pred=jnp.asarray(0.9),
    )
    targ["energy_mhd_MJ"] = xr.DataArray(0.15)
    targ["power_ohm_MW"] = xr.DataArray(1.0)
    targ["power_radiated_MW"] = xr.DataArray(0.4)

    expected = 0.2 * abs(0.10 - 0.15) + 0.3 * abs(1.5 - 1.0) + 0.5 * abs(0.9 - 0.4)
    assert float(train_loss(pred, targ)) == pytest.approx(expected, rel=1e-6)
    assert float(val_loss(pred, targ)) == 0.0


def test_val_loss_is_value_chi():
    """A fresh slice offset by k error bars in each channel scores chi k per channel over rho in [0, 1],
    and a diverged slice pays the chi-scale divergence penalty."""
    k = 2.0
    pred, targ = _pred_and_targ(fresh=1.0)
    # A constant offset leaves the gradients exact, so only the value chi scores
    pred = Output(
        ne=jnp.asarray(NE_TARG + k * _error_bars(NE_TARG)),
        te=jnp.asarray(TE_TARG - k * _error_bars(TE_TARG)),
        rho=jnp.asarray(RHO),
    )
    val_loss = TransportPredictorTRB.get_val_loss_fn(_loss_config()).instantaneous_loss

    assert float(val_loss(pred, targ)) == pytest.approx(2.0 * k, rel=1e-6)

    pred_diverged = Output(ne=pred.ne.at[3].set(jnp.nan), te=pred.te, rho=pred.rho)
    assert float(val_loss(pred_diverged, targ)) == pytest.approx(2.0 * k + LOSS_CONFIG["divergence_penalty_val"], rel=1e-3)


def test_gradient_terms_score_a_gradient_only_residual():
    """Exact profile values with GP-fit gradients offset by g score only the gradient terms.

    Training: half the channel sum of gradient_weight x the huber of the peak-normalized offset below GRAD_RHO_MAX.
    Validation: gradient_weight x the offset in gradient error bars below GRAD_RHO_MAX, summed over the channels.
    A stale slice scores neither.
    """
    grad_offset = 0.5
    loss_config = _loss_config()
    train_loss = TransportPredictorTRB.get_loss_fn(loss_config).instantaneous_loss
    val_loss = TransportPredictorTRB.get_val_loss_fn(loss_config).instantaneous_loss
    pred, targ = _pred_and_targ(fresh=1.0, offset=0.0, grad_offset=grad_offset)

    expected_train = 0.0
    expected_val = 0.0
    for targ_profile, targ_grad in ((NE_TARG, NE_GRAD), (TE_TARG, TE_GRAD)):
        scale = np.max(np.abs(targ_profile))
        normalized_offset = grad_offset / scale
        huber = np.where(normalized_offset <= 1.0, 0.5 * normalized_offset**2, normalized_offset - 0.5)
        expected_train += 0.5 * 0.1 * np.trapezoid((RHO_MID < GRAD_RHO_MAX) * huber, x=RHO_MID)
        grad_sigma = SIGMA_FRAC * np.max(np.abs(targ_grad))
        expected_val += 0.1 * np.trapezoid((RHO_MID < GRAD_RHO_MAX) * grad_offset / grad_sigma, x=RHO_MID)
    assert float(train_loss(pred, targ)) == pytest.approx(expected_train, rel=1e-6)
    assert float(val_loss(pred, targ)) == pytest.approx(expected_val, rel=1e-6)

    pred_stale, targ_stale = _pred_and_targ(fresh=0.0, offset=0.0, grad_offset=grad_offset)
    assert float(train_loss(pred_stale, targ_stale)) == 0.0
    assert float(val_loss(pred_stale, targ_stale)) == 0.0
