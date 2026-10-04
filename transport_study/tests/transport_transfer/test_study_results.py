"""The transport test suite scores only timeslices whose target is a fresh profile measurement."""

from types import SimpleNamespace

import numpy as np
import xarray as xr

from transport_study import EPISODE_DIM, RADIAL_DIM, TIME_COORD, TIME_DIM
from transport_study.modules.profile_predictor.trb import ProfilePredictorTRB
from transport_study.modules.transport_predictor.trb import TransportPredictorTRB

RHO = np.linspace(0.0, 1.0, 5)
SHOTS = [101, 102]
N_TIME = 4
STALE_TIME_IDX = 2


def _eval_data() -> SimpleNamespace:
    """Perfect predictions except a large miss at one forward-filled timeslice of each shot."""
    shape = (len(SHOTS), N_TIME, RHO.size)
    profile = np.broadcast_to(1.0 - 0.5 * RHO, shape)
    fresh = np.ones((len(SHOTS), N_TIME))
    fresh[:, STALE_TIME_IDX] = 0.0
    pred = profile.copy()
    pred[:, STALE_TIME_IDX, :] += 5.0
    dims_profile = (EPISODE_DIM, TIME_DIM, RADIAL_DIM)
    dims_time = (EPISODE_DIM, TIME_DIM)
    coords = {
        EPISODE_DIM: SHOTS,
        TIME_DIM: np.arange(N_TIME),
        RADIAL_DIM: RHO,
        TIME_COORD: (dims_time, np.tile(1e-3 * np.arange(N_TIME), (len(SHOTS), 1))),
    }
    input_ds = xr.Dataset(
        {"n_e_1e20": (dims_profile, profile), "t_e_keV": (dims_profile, profile), "fresh_profile": (dims_time, fresh)},
        coords=coords,
    ).assign_coords(ds_source=(EPISODE_DIM, ["mast", "mast"]))
    output_ds = xr.Dataset({"ne": (dims_profile, pred), "te": (dims_profile, pred)}, coords=coords)
    stack = {"sample": (EPISODE_DIM, TIME_DIM)}
    return SimpleNamespace(input_ds=input_ds.stack(stack), output_ds=output_ds.stack(stack))


def test_stale_timeslices_score_nan_only_in_the_transport_suite():
    eval_data = _eval_data()

    ds_transport = TransportPredictorTRB.get_test_eval_suite({"result_path": "unused"})["study_results"](eval_data)
    ds_profile = ProfilePredictorTRB.get_test_eval_suite({"result_path": "unused"})["study_results"](eval_data)

    stale = ds_transport[TIME_DIM] == STALE_TIME_IDX
    assert ds_transport["error_abs_ts"].where(stale, drop=True).isnull().all()
    np.testing.assert_allclose(ds_transport["error_abs_ts"].where(~stale, drop=True), 0.0, atol=1e-12)
    # Fresh timeslices are perfect, so the shot integrals carry no error from the stale miss
    np.testing.assert_allclose(ds_transport["error_abs_shot"], 0.0, atol=1e-12)
    # The profile study scores every timeslice it is given
    assert (ds_profile["error_abs_ts"].where(stale, drop=True) > 1.0).all()
