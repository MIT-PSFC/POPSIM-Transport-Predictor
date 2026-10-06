"""The profile test suite, shared by the profile and transport studies.

It scores the validation chi per test timeslice, flags diverged predictions,
and in the transport study scores only timeslices whose target is a fresh profile measurement.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from transport_study import EPISODE_DIM, RADIAL_DIM, TIME_COORD, TIME_DIM
from transport_study.modules.profile_predictor.trb import ProfilePredictorTRB
from transport_study.modules.transport_predictor.trb import TransportPredictorTRB
from transport_study.modules.trb_utils import CHI_ERROR_VARS, GRAD_RHO_MAX

RHO = np.linspace(0.0, 1.0, 11)
RHO_MID = 0.5 * (RHO[:-1] + RHO[1:])
SHOTS = [101, 102]
N_TIME = 4
STALE_TIME_IDX = 2
# Linear profiles with peaks 2 and 4, so the peak-normalized gradient is -0.5 in both channels
NE_TARG = 2.0 * (1.0 - 0.5 * RHO)
TE_TARG = 4.0 * (1.0 - 0.5 * RHO)
NE_GRAD = np.full(RHO.size, -1.0)
TE_GRAD = np.full(RHO.size, -2.0)
# Peak-normalized error-bar floor, it applies wherever the synthetic error bars are zero
SIGMA_FLOOR = 0.01
GRADIENT_WEIGHT = 0.1
SUITE_CONFIG = {
    "result_path": "unused",
    "loss_config": {
        "gradient_weight": GRADIENT_WEIGHT,
        "chi_sigma_floors": {"mast": {var: SIGMA_FLOOR for error_vars in CHI_ERROR_VARS.values() for var in error_vars}},
    },
}


def _eval_data(ne_pred=NE_TARG, te_pred=TE_TARG, ne_grad=NE_GRAD, te_grad=TE_GRAD) -> SimpleNamespace:
    """Stacked EvalData of two shots with zero-width error bars, the timeslice STALE_TIME_IDX forward-filled.

    Every argument is a profile over RHO broadcast to every timeslice, or a full (shot, time, rho) array.
    """
    shape = (len(SHOTS), N_TIME, RHO.size)
    fresh = np.ones((len(SHOTS), N_TIME))
    fresh[:, STALE_TIME_IDX] = 0.0
    dims_profile = (EPISODE_DIM, TIME_DIM, RADIAL_DIM)
    dims_time = (EPISODE_DIM, TIME_DIM)
    coords = {
        EPISODE_DIM: SHOTS,
        TIME_DIM: np.arange(N_TIME),
        RADIAL_DIM: RHO,
        TIME_COORD: (dims_time, np.tile(1e-3 * np.arange(N_TIME), (len(SHOTS), 1))),
    }
    profiles = {
        "n_e_1e20": NE_TARG,
        "t_e_keV": TE_TARG,
        "n_e_1e20_gradient": ne_grad,
        "t_e_keV_gradient": te_grad,
        **{var: np.zeros(RHO.size) for error_vars in CHI_ERROR_VARS.values() for var in error_vars},
    }
    input_vars = {name: (dims_profile, np.broadcast_to(profile, shape).copy()) for name, profile in profiles.items()}
    input_ds = xr.Dataset({**input_vars, "fresh_profile": (dims_time, fresh)}, coords=coords)
    input_ds = input_ds.assign_coords(ds_source=(EPISODE_DIM, ["mast", "mast"]))
    output_ds = xr.Dataset(
        {"ne": (dims_profile, np.broadcast_to(ne_pred, shape).copy()), "te": (dims_profile, np.broadcast_to(te_pred, shape).copy())},
        coords=coords,
    )
    stack = {"sample": (EPISODE_DIM, TIME_DIM)}
    return SimpleNamespace(input_ds=input_ds.stack(stack), output_ds=output_ds.stack(stack))


def _profile_results(eval_data: SimpleNamespace) -> xr.Dataset:
    return ProfilePredictorTRB.get_test_eval_suite(SUITE_CONFIG)["study_results"](eval_data)


def test_perfect_prediction_scores_zero_chi():
    ds = _profile_results(_eval_data())
    for var in ("error_chi_value_ts", "error_chi_grad_ts", "error_chi_ts", "error_diverged_ts"):
        np.testing.assert_allclose(ds[var], 0.0, atol=1e-6)


def test_zero_error_bars_count_as_the_floor():
    """A constant offset of 0.1 is 0.1 / peak in peak-normalized units, in floors of 0.01, over the unit rho interval."""
    ds = _profile_results(_eval_data(ne_pred=NE_TARG + 0.1, te_pred=TE_TARG + 0.1))
    np.testing.assert_allclose(ds["error_chi_value_ts"], (0.1 / 2.0 + 0.1 / 4.0) / SIGMA_FLOOR, rtol=1e-5)
    # A constant offset leaves the gradients untouched
    np.testing.assert_allclose(ds["error_chi_grad_ts"], 0.0, atol=1e-6)


def test_gradient_error_beyond_rho_max_is_masked():
    """GP gradient targets wrong only at the edge point corrupt only the last midpoint, beyond GRAD_RHO_MAX."""
    assert RHO_MID[-1] > GRAD_RHO_MAX > RHO_MID[-2]
    ne_grad = NE_GRAD.copy()
    te_grad = TE_GRAD.copy()
    ne_grad[-1] = 50.0
    te_grad[-1] = 50.0
    ds = _profile_results(_eval_data(ne_grad=ne_grad, te_grad=te_grad))
    np.testing.assert_allclose(ds["error_chi_grad_ts"], 0.0, atol=1e-6)


def test_combined_is_value_plus_weighted_gradient_chi():
    """A zero ne gradient target misses the prediction's -0.5 by 0.5 / floor at every midpoint below GRAD_RHO_MAX."""
    ds = _profile_results(_eval_data(ne_pred=NE_TARG + 0.1, ne_grad=np.zeros(RHO.size)))
    expected_grad = np.trapezoid((RHO_MID < GRAD_RHO_MAX) * 0.5 / SIGMA_FLOOR, x=RHO_MID)
    np.testing.assert_allclose(ds["error_chi_grad_ts"], expected_grad, rtol=1e-5)
    np.testing.assert_allclose(ds["error_chi_ts"], ds["error_chi_value_ts"] + GRADIENT_WEIGHT * ds["error_chi_grad_ts"], rtol=1e-6)


def test_stale_timeslices_score_nan_only_in_the_transport_suite():
    """A large miss at the forward-filled timeslice is NaN in every transport error, chi included, but scored by the profile suite."""
    ne_pred = np.broadcast_to(NE_TARG, (len(SHOTS), N_TIME, RHO.size)).copy()
    ne_pred[:, STALE_TIME_IDX, :] += 5.0
    eval_data = _eval_data(ne_pred=ne_pred)

    ds_transport = TransportPredictorTRB.get_test_eval_suite(SUITE_CONFIG)["study_results"](eval_data)
    ds_profile = _profile_results(eval_data)

    stale = ds_transport[TIME_DIM] == STALE_TIME_IDX
    for var in ("error_abs_ts", "error_chi_ts"):
        assert ds_transport[var].where(stale, drop=True).isnull().all()
        np.testing.assert_allclose(ds_transport[var].where(~stale, drop=True), 0.0, atol=1e-6)
    # Fresh timeslices are perfect, so the shot integrals carry no error from the stale miss
    np.testing.assert_allclose(ds_transport["error_abs_shot"], 0.0, atol=1e-12)
    # Divergence is flagged outside the freshness mask, a finite stale miss is no divergence
    np.testing.assert_allclose(ds_transport["error_diverged_ts"], 0.0)
    # The profile study scores every timeslice it is given
    assert (ds_profile["error_abs_ts"].where(stale, drop=True) > 1.0).all()
    assert (ds_profile["error_chi_ts"].where(stale, drop=True) > 1.0).all()


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_non_finite_predictions_are_flagged_and_never_score_inf(bad_value):
    """One non-finite point diverges the whole timeslice: flag 1, every error NaN, the shot integrals finite."""
    te_pred = np.broadcast_to(TE_TARG, (len(SHOTS), N_TIME, RHO.size)).copy()
    te_pred[0, 1, 3] = bad_value
    ds = _profile_results(_eval_data(te_pred=te_pred))

    diverged = ds["error_diverged_ts"].sel({EPISODE_DIM: SHOTS[0]})
    assert diverged.values.tolist() == [0.0, 1.0, 0.0, 0.0]
    for var in ("error_abs_ts", "error_rel_ts", "ne_error_abs_ts", "error_chi_value_ts", "error_chi_grad_ts", "error_chi_ts"):
        values = ds[var].sel({EPISODE_DIM: SHOTS[0]}).values
        assert np.isnan(values[1]), var
        assert np.isfinite(np.delete(values, 1)).all(), var
    assert np.isfinite(ds["error_abs_shot"].values).all()
