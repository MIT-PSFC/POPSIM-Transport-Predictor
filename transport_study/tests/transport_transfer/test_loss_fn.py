"""Tests for the transport predictor training and validation losses.

Hand-built predictions and targets so the expected values are exact. Covers
the freshness masking: the time-dependent rollouts keep forward-filled
(stale) profile timeslices in the data for segment contiguity, so both
losses must zero them out, the anchor terms that are exempt from it,
and the value chi of the validation loss.
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
from transport_study.modules.trb_utils import CHI_ERROR_VARS

RHO = RHO_GRID
# Peak-normalized error bar of every target profile, far above SIGMA_FLOOR
SIGMA_FRAC = 0.05
SIGMA_FLOOR = 1e-3
LOSS_CONFIG = {
    "huber_delta": 0.1,
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


def _pred_and_targ(fresh: float, offset: float = 0.3, **pred_extra):
    """Single-timeslice prediction/target pair with a nonzero profile residual.

    fresh sets the fresh_profile flag on the target side. pred_extra sets
    extra Output fields by name (the sciml anchor predictions).
    """
    ne_targ = 1.5 * (1 - RHO**2) + 0.5
    te_targ = 3.0 * (1 - RHO**2) + 0.5
    pred = Output(
        ne=jnp.asarray(ne_targ + offset),
        te=jnp.asarray(te_targ - offset),
        rho=jnp.asarray(RHO),
        **pred_extra,
    )
    targ = {
        "n_e_1e20": xr.DataArray(ne_targ, dims=(RADIAL_DIM,)),
        "t_e_keV": xr.DataArray(te_targ, dims=(RADIAL_DIM,)),
        "n_e_1e20_error": xr.DataArray(_error_bars(ne_targ), dims=(RADIAL_DIM,)),
        "t_e_keV_error": xr.DataArray(_error_bars(te_targ), dims=(RADIAL_DIM,)),
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
    ne_targ = 1.5 * (1 - RHO**2) + 0.5
    te_targ = 3.0 * (1 - RHO**2) + 0.5
    pred = Output(
        ne=jnp.stack([jnp.asarray(ne_targ + 0.3)] * 2),
        te=jnp.stack([jnp.asarray(te_targ - 0.3)] * 2),
        rho=jnp.asarray(RHO),
    )

    def targ_with_fresh(fresh_flags):
        return {
            "n_e_1e20": xr.DataArray([ne_targ] * 2, dims=("time_idx", RADIAL_DIM)),
            "t_e_keV": xr.DataArray([te_targ] * 2, dims=("time_idx", RADIAL_DIM)),
            "n_e_1e20_error": xr.DataArray([_error_bars(ne_targ)] * 2, dims=("time_idx", RADIAL_DIM)),
            "t_e_keV_error": xr.DataArray([_error_bars(te_targ)] * 2, dims=("time_idx", RADIAL_DIM)),
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
    ne_targ = 1.5 * (1 - RHO**2) + 0.5
    te_targ = 3.0 * (1 - RHO**2) + 0.5
    k = 2.0
    pred, targ = _pred_and_targ(fresh=1.0)
    pred = Output(
        ne=jnp.asarray(ne_targ + k * _error_bars(ne_targ)),
        te=jnp.asarray(te_targ - k * _error_bars(te_targ)),
        rho=jnp.asarray(RHO),
    )
    val_loss = TransportPredictorTRB.get_val_loss_fn(_loss_config()).instantaneous_loss

    assert float(val_loss(pred, targ)) == pytest.approx(2.0 * k, rel=1e-6)

    pred_diverged = Output(ne=pred.ne.at[3].set(jnp.nan), te=pred.te, rho=pred.rho)
    assert float(val_loss(pred_diverged, targ)) == pytest.approx(2.0 * k + LOSS_CONFIG["divergence_penalty_val"], rel=1e-3)
