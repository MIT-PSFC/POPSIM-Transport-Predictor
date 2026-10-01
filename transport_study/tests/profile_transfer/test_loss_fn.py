"""Tests for the profile predictor training and validation losses.

Everything runs on hand-built profiles so the expected loss values are exact,
plus one end-to-end check against real GP-fit signals from a sample dataset.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from popsim.ml.dataloading import make_dataloaders

from transport_study import EPISODE_DIM, RADIAL_DIM, TIME_COORD
from transport_study.config import RHO_GRID, config, load_config
from transport_study.modules.profile_predictor.module import Outputs
from transport_study.modules.profile_predictor.trb import ProfilePredictorTRB
from transport_study.orchestration.organize_data import (
    PROFILE_TARGET_VARS,
    get_ds,
    get_loaded_shot_count,
)
from transport_study.profile_transfer.profile_study import ProfileStudy
from transport_study.tests.sample_data import SAMPLE_DIR, requires_sample_data

RHO = RHO_GRID


@pytest.fixture
def study_config(tmp_path) -> ProfileStudy.Config:
    """Loaded ProfileStudy config, so the case-grid tests can build a study."""
    cfg = ProfileStudy.Config(
        study_name="test_loss_fn",
        working_dir_base=tmp_path,
        dataset_paths={
            "cmod-low": SAMPLE_DIR / "cmod-low1.nc",
            "cmod-high": SAMPLE_DIR / "cmod-high.nc",
        },
        target_device="cmod-high",
        model_types=("mlp",),
        training_datasets=("cmod-low",),
        target_test_set_size=60,
    )
    load_config(cfg)
    return cfg


@pytest.fixture(autouse=True)
def loaded_config(request):
    """Every test needs a loaded config; study tests get theirs from study_config."""
    if "study_config" not in request.fixturenames:
        request.getfixturevalue("study_config")


def _profile_da(values):
    return xr.DataArray(data=jnp.asarray(values), dims=(RADIAL_DIM,), coords={RADIAL_DIM: RHO})


def _pred_and_targ(ne_pred, te_pred, ne_targ, te_targ, **extra_targ_vars):
    """Build a prediction/target pair for the loss.

    extra_targ_vars adds optional target vars by name, e.g. the error-bar and
    gradient signals n_e_1e20_error / t_e_keV_gradient / etc. Tests that omit
    them exercise the defaults: zero-width error bars and finite-difference
    gradient targets.
    """
    pred = Outputs(ne=_profile_da(ne_pred), te=_profile_da(te_pred))
    targ = {
        "n_e_1e20": _profile_da(ne_targ),
        "t_e_keV": _profile_da(te_targ),
        "ds_source_idx": xr.DataArray(data=jnp.asarray(0.0)),
    }
    for name, values in extra_targ_vars.items():
        targ[name] = _profile_da(values)
    return pred, targ


def test_loss_zero_for_perfect_prediction():
    loss_fn = ProfilePredictorTRB.get_loss_fn({"huber_delta": 0.5, "gradient_weight": 0.1})
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    pred, targ = _pred_and_targ(ne, te, ne, te)
    assert float(loss_fn(pred, targ)) == pytest.approx(0.0)


def test_gradient_term_penalizes_slope_mismatch():
    """Two predictions with identical pointwise value error but different gradients.

    A constant offset keeps the target's slope, an alternating offset of the same
    magnitude corrupts it. The value-only loss cannot tell them apart, the
    gradient term must.
    """
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    c = 0.2
    offset_const = np.full_like(RHO, c)
    offset_alt = c * np.where(np.arange(RHO.size) % 2 == 0, 1.0, -1.0)

    pred_const, targ = _pred_and_targ(ne + offset_const, te + offset_const, ne, te)
    pred_alt, _ = _pred_and_targ(ne + offset_alt, te + offset_alt, ne, te)

    value_only = ProfilePredictorTRB.get_loss_fn({"huber_delta": 0.5, "gradient_weight": 0.0})
    with_grad = ProfilePredictorTRB.get_loss_fn({"huber_delta": 0.5, "gradient_weight": 0.1, "huber_delta_grad": 5.0})

    assert float(value_only(pred_const, targ)) == pytest.approx(float(value_only(pred_alt, targ)), rel=1e-5)
    assert float(with_grad(pred_alt, targ)) > float(with_grad(pred_const, targ))
    # Constant offset leaves gradients untouched, so the gradient term adds nothing
    assert float(with_grad(pred_const, targ)) == pytest.approx(float(value_only(pred_const, targ)), rel=1e-5)


def test_gradient_weight_defaults_off():
    """A loss config without gradient keys weights the gradient term at zero."""
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    pred, targ = _pred_and_targ(ne + 0.3 * RHO, te - 0.3 * RHO, ne, te)
    implicit_off = ProfilePredictorTRB.get_loss_fn({"huber_delta": 0.5})
    explicit_off = ProfilePredictorTRB.get_loss_fn({"huber_delta": 0.5, "gradient_weight": 0.0})
    assert float(implicit_off(pred, targ)) == pytest.approx(float(explicit_off(pred, targ)), rel=1e-6)


def test_device_weights_scale_loss():
    """device_weights multiplies the loss for samples from the matching device only.

    Samples in _pred_and_targ carry ds_source_idx 0, which maps to the
    alphabetically-first device. Weighting that device by 3 must triple the
    loss (value and gradient terms alike), weighting any other device must
    leave it unchanged.
    """
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    pred, targ = _pred_and_targ(ne + 0.3 * RHO, te - 0.2 * RHO, ne, te)

    devices = sorted(config.dataset_paths)
    loss_base = {"huber_delta": 0.5, "gradient_weight": 0.1, "huber_delta_grad": 5.0}
    unweighted = ProfilePredictorTRB.get_loss_fn(loss_base)

    weights_sample_device = dict.fromkeys(devices, 1.0)
    weights_sample_device[devices[0]] = 3.0
    weighted = ProfilePredictorTRB.get_loss_fn({**loss_base, "device_weights": weights_sample_device})
    assert float(weighted(pred, targ)) == pytest.approx(3.0 * float(unweighted(pred, targ)), rel=1e-5)

    weights_other_device = dict.fromkeys(devices, 1.0)
    weights_other_device[devices[-1]] = 3.0
    other_weighted = ProfilePredictorTRB.get_loss_fn({**loss_base, "device_weights": weights_other_device})
    assert float(other_weighted(pred, targ)) == pytest.approx(float(unweighted(pred, targ)), rel=1e-6)


def test_channels_balanced_by_peak_normalization():
    """Equal fractional error must cost the same in ne and Te.

    The loss runs on peak-normalized profiles, so a 10% error on a tiny ne
    profile (0.05e20 m^-3 peak) and a 10% error on a large Te profile (8 keV
    peak) must contribute identically - without normalization the Te channel
    would dominate by orders of magnitude.
    """
    ne = 0.05 * (1 - RHO**2)
    te = 8.0 * (1 - RHO**2)

    pred_ne_off, targ = _pred_and_targ(1.1 * ne, te, ne, te)
    pred_te_off, _ = _pred_and_targ(ne, 1.1 * te, ne, te)

    for loss_fn in (
        ProfilePredictorTRB.get_loss_fn({"huber_delta": 0.1, "gradient_weight": 0.1, "huber_delta_grad": 1.0}),
        ProfilePredictorTRB.get_val_loss_fn({"gradient_weight": 0.1, "within_error_weight": 0.5}),
    ):
        ne_off_loss = float(loss_fn(pred_ne_off, targ))
        te_off_loss = float(loss_fn(pred_te_off, targ))
        assert ne_off_loss > 0.0
        assert ne_off_loss == pytest.approx(te_off_loss, rel=1e-5)


def test_val_loss_independent_of_huber_delta():
    """The sweep metric must not depend on the swept deltas.

    Huber loss shrinks monotonically as delta -> 0, so if the validation loss
    read the deltas the Bayes sweep would drive them to their minimum to make
    the reported number small. The validation loss must be identical across
    delta values, while the training loss must actually respond to them.
    """
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    pred, targ = _pred_and_targ(ne + 0.3, te - 0.4, ne, te)

    small = {"huber_delta": 0.01, "huber_delta_grad": 0.1, "gradient_weight": 0.1, "within_error_weight": 0.5}
    large = {"huber_delta": 1.0, "huber_delta_grad": 10.0, "gradient_weight": 0.1, "within_error_weight": 0.5}

    val_small = float(ProfilePredictorTRB.get_val_loss_fn(small)(pred, targ))
    val_large = float(ProfilePredictorTRB.get_val_loss_fn(large)(pred, targ))
    assert val_small == pytest.approx(val_large, rel=1e-6)
    assert val_small > 0.0

    train_small = float(ProfilePredictorTRB.get_loss_fn(small)(pred, targ))
    train_large = float(ProfilePredictorTRB.get_loss_fn(large)(pred, targ))
    assert train_small != pytest.approx(train_large, rel=1e-3)


def test_normalized_loss_magnitude_is_numerically_safe():
    """Peak normalization shrinks loss values - check they stay well above
    float32 resolution.

    A typical 5% profile error gives a training huber loss of order 1e-3 and a
    validation abs-error loss of order 1e-2. float32 has ~1e-38 normal range
    and ~1e-7 relative precision, so these are far from underflow, and the
    swept learning rate absorbs the overall loss-scale change.
    """
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    pred, targ = _pred_and_targ(1.05 * ne, 1.05 * te, ne, te)

    train_loss = float(ProfilePredictorTRB.get_loss_fn({"huber_delta": 0.1, "gradient_weight": 0.1, "huber_delta_grad": 1.0})(pred, targ))
    val_loss = float(ProfilePredictorTRB.get_val_loss_fn({"gradient_weight": 0.1, "within_error_weight": 0.5})(pred, targ))

    for loss in (train_loss, val_loss):
        assert np.isfinite(loss)
        assert loss > 1e-6


def weighted_case(study: ProfileStudy, num_target_shots: int):
    return next(c for c in study.cases if c.domain_adaptation == "weighted" and c.num_target_shots == num_target_shots)


@pytest.mark.slow
@requires_sample_data
def test_weighted_case_weights_reach_loss_config(study_config):
    """The per-device weights of a weighted case land in loss_config as device_weights.

    Regression test for the bug where make_train_config wrote them to
    dataloader_config["dataset_weights"], which nothing reads, so every
    weighted case trained unweighted.
    """
    study = ProfileStudy(study_config)
    train_config = study.make_train_config(weighted_case(study, num_target_shots=10))

    weights = train_config.loss_config["device_weights"]
    # Weights mirror the actual training set: all loaded source shots plus
    # num_target_shots target shots. Without configured dataset_fractions the
    # target's share of the loss is sqrt-scaled in its share of the shots
    n_source = get_loaded_shot_count("cmod-low", study_type=ProfileStudy.STUDY_TYPE)
    n_target = 10
    n_total = n_source + n_target
    target_fraction = min(0.5, np.sqrt(n_target / n_total))
    assert set(weights) == {"cmod-high", "cmod-low"}
    assert weights["cmod-high"] == pytest.approx(target_fraction * n_total / n_target)
    assert weights["cmod-low"] == pytest.approx((1 - target_fraction) * n_total / n_source)
    # Effective contributions split in the configured proportion and the mean
    # per-sample weight over the training set is exactly 1
    assert weights["cmod-high"] * n_target == pytest.approx(target_fraction * n_total)
    assert weights["cmod-high"] * n_target + weights["cmod-low"] * n_source == pytest.approx(n_total)
    # Scarce target data must be weighted more heavily than plentiful source data
    assert weights["cmod-high"] > weights["cmod-low"]
    # Validation eval must see the same weighting as training
    assert train_config.val_eval_suite_config["loss_config"]["device_weights"] == weights

    baseline_case = next(c for c in study.cases if c.domain_adaptation is None)
    assert "device_weights" not in study.make_train_config(baseline_case).loss_config


@pytest.mark.slow
@requires_sample_data
def test_weighted_zero_target_shots_disables_patience(study_config):
    """A weighted case with num_target_shots == 0 must run to max_epochs.

    With no target samples in training the weighted validation loss cannot
    track target improvement, so early stopping is disabled (patience None).
    Every other case keeps the configured patience.
    """
    study = ProfileStudy(study_config)

    assert study.make_train_config(weighted_case(study, num_target_shots=0)).patience is None
    assert study.make_train_config(weighted_case(study, num_target_shots=10)).patience == config.patience

    baseline_case = next(c for c in study.cases if c.domain_adaptation is None)
    assert study.make_train_config(baseline_case).patience == config.patience


def test_error_bars_soften_loss_within_them():
    """Residuals inside the measurement error bars are down-weighted, not free.

    The part of the residual inside the error bar is scaled by
    within_error_weight, the part beyond it is penalized at full weight. A
    prediction offset by half the error bar must therefore cost something (it
    is still pulled toward the GP fit mean) but exactly within_error_weight
    times what it would cost without error bars. Outside the bar only the
    within-bar part is discounted.
    """
    w = 0.25
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    ne_sigma = np.full_like(RHO, 0.2)
    te_sigma = np.full_like(RHO, 0.3)
    errors = {"n_e_1e20_error": ne_sigma, "t_e_keV_error": te_sigma}

    train_loss = ProfilePredictorTRB.get_loss_fn({"huber_delta": 0.1, "within_error_weight": w})
    val_loss = ProfilePredictorTRB.get_val_loss_fn({"within_error_weight": w})

    # Half-sigma offset: entirely inside the bar, so the delta-free validation
    # loss is exactly w times the unbanded loss on the same offset
    pred_inside, targ = _pred_and_targ(ne + 0.5 * ne_sigma, te - 0.5 * te_sigma, ne, te, **errors)
    _, targ_no_err = _pred_and_targ(ne, te, ne, te)
    inside_loss = float(val_loss(pred_inside, targ))
    assert inside_loss > 0.0
    assert inside_loss == pytest.approx(w * float(val_loss(pred_inside, targ_no_err)), rel=1e-5)
    assert float(train_loss(pred_inside, targ)) > 0.0

    # Three-sigma offset: sigma of it discounted by w, two sigma at full
    # weight, so it equals the unbanded loss on a (2 + w) sigma offset
    pred_outside, _ = _pred_and_targ(ne + 3.0 * ne_sigma, te - 3.0 * te_sigma, ne, te, **errors)
    pred_equiv, _ = _pred_and_targ(ne + (2.0 + w) * ne_sigma, te - (2.0 + w) * te_sigma, ne, te)
    assert float(val_loss(pred_outside, targ)) == pytest.approx(float(val_loss(pred_equiv, targ_no_err)), rel=1e-5)

    # Missing the error bar costs much more than landing inside it
    assert float(val_loss(pred_outside, targ)) > float(val_loss(pred_inside, targ))
    assert float(train_loss(pred_outside, targ)) > float(train_loss(pred_inside, targ))


def test_zero_error_sentinel_matches_missing_error_signals():
    """Error bars of 0 (the no-error-quantification sentinel) must score exactly
    the same as omitting the error signals entirely, gradient term included.

    The profiles are quadratic, so the analytic gradient signal averaged to
    the rho midpoints equals the finite differences of the values exactly and
    the comparison isolates the sentinel handling.
    """
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    zeros = np.zeros_like(RHO)
    sentinel_extras = {
        "n_e_1e20_error": zeros,
        "t_e_keV_error": zeros,
        "n_e_1e20_gradient": -3.0 * RHO,
        "t_e_keV_gradient": -6.0 * RHO,
        "n_e_1e20_gradient_error": zeros,
        "t_e_keV_gradient_error": zeros,
    }
    loss_base = {"huber_delta": 0.5, "gradient_weight": 0.1, "huber_delta_grad": 5.0, "within_error_weight": 0.5}

    pred, targ_sentinel = _pred_and_targ(ne + 0.3, te - 0.4, ne, te, **sentinel_extras)
    _, targ_no_signals = _pred_and_targ(ne + 0.3, te - 0.4, ne, te)

    for loss_fn in (ProfilePredictorTRB.get_loss_fn(loss_base), ProfilePredictorTRB.get_val_loss_fn(loss_base)):
        assert float(loss_fn(pred, targ_sentinel)) == pytest.approx(float(loss_fn(pred, targ_no_signals)), rel=1e-5)
        assert float(loss_fn(pred, targ_sentinel)) > 0.0


def test_measured_gradient_signal_is_the_gradient_target():
    """When gradient signals are present they define the gradient target.

    The prediction matches the target values exactly, so any gradient loss can
    only come from the measured gradient signal disagreeing with the
    prediction's slope. A gradient error bar covering that disagreement must
    down-weight it to within_error_weight times the unbanded gradient loss.
    """
    w = 0.25
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    grad_offset = 1.0
    extras = {
        "n_e_1e20_gradient": -3.0 * RHO + grad_offset,
        "t_e_keV_gradient": -6.0 * RHO + grad_offset,
    }
    loss_base = {"gradient_weight": 0.1, "within_error_weight": w}
    with_grad = ProfilePredictorTRB.get_val_loss_fn(loss_base)
    value_only = ProfilePredictorTRB.get_val_loss_fn({**loss_base, "gradient_weight": 0.0})

    pred, targ = _pred_and_targ(ne, te, ne, te, **extras)
    assert float(value_only(pred, targ)) == pytest.approx(0.0, abs=1e-9)
    unbanded = float(with_grad(pred, targ))
    assert unbanded > 0.0

    # Gradient error bars at least as large as the disagreement discount the
    # whole gradient residual by w. The offset is in channel units and the
    # error bar comparison runs on normalized quantities, so any error bar
    # >= the offset works for both channels
    big_sigma = np.full_like(RHO, 2.0 * grad_offset)
    _, targ_banded = _pred_and_targ(ne, te, ne, te, **extras, n_e_1e20_gradient_error=big_sigma, t_e_keV_gradient_error=big_sigma)
    assert float(with_grad(pred, targ_banded)) == pytest.approx(w * unbanded, rel=1e-5)


def test_gradient_loss_only_below_rho_09():
    """Gradient mismatch at rho >= 0.9 must not contribute to the loss.

    The measured gradients in the pedestal / edge are unreliable, so the
    gradient term is masked to rho < GRAD_LOSS_RHO_MAX. A large gradient-signal
    disagreement confined to rho >= 0.92 (whose midpoint contributions all sit
    at rho >= 0.9) must leave the loss at zero, while the same disagreement in
    the core must not.
    """
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    ne_grad_true = -3.0 * RHO
    te_grad_true = -6.0 * RHO
    loss_fn = ProfilePredictorTRB.get_loss_fn({"huber_delta": 0.5, "gradient_weight": 0.1, "huber_delta_grad": 5.0})

    edge_offset = np.where(RHO >= 0.92, 50.0, 0.0)
    pred, targ_edge = _pred_and_targ(
        ne, te, ne, te, n_e_1e20_gradient=ne_grad_true + edge_offset, t_e_keV_gradient=te_grad_true + edge_offset
    )
    assert float(loss_fn(pred, targ_edge)) == pytest.approx(0.0, abs=1e-9)

    core_offset = np.where(RHO <= 0.5, 50.0, 0.0)
    _, targ_core = _pred_and_targ(ne, te, ne, te, n_e_1e20_gradient=ne_grad_true + core_offset, t_e_keV_gradient=te_grad_true + core_offset)
    assert float(loss_fn(pred, targ_core)) > 0.0


@pytest.mark.slow
@requires_sample_data
def test_loss_on_prepared_sample_dataset():
    """Full-pipeline check: signals prepared by get_ds feed the loss directly.

    Pulls one fully-finite timeslice of the prepared cmod-high sample dataset
    (real GP-fit profiles, gradients and error bars on the uniform 51-point
    rho grid) and checks the error-bar semantics on real error bars: perfect
    predictions cost nothing, within-error-bar predictions cost exactly
    within_error_weight times their unbanded loss, predictions outside cost
    more, and the gradient term stays finite on real gradient signals.
    """
    w = 0.25
    ds, _ = get_ds("cmod-high", "profile_transfer")

    finite = np.ones((ds.sizes[EPISODE_DIM], ds.sizes["time_idx"]), dtype=bool)
    for var in PROFILE_TARGET_VARS:
        finite &= np.isfinite(ds[var].values).all(axis=-1)
    # The deadband assertions need real (nonzero) error bars on every point
    finite &= (ds["n_e_1e20_error"].values > 0).all(axis=-1)
    finite &= (ds["t_e_keV_error"].values > 0).all(axis=-1)
    assert finite.any(), "sample dataset has no fully-finite timeslice with error bars"
    i_shot, i_time = np.argwhere(finite)[0]
    ts = ds.isel({EPISODE_DIM: i_shot, "time_idx": i_time})

    ne = ts["n_e_1e20"].values
    te = ts["t_e_keV"].values
    ne_sigma = ts["n_e_1e20_error"].values
    te_sigma = ts["t_e_keV_error"].values
    extras = {var: ts[var].values for var in PROFILE_TARGET_VARS if var not in ("n_e_1e20", "t_e_keV")}

    val_loss = ProfilePredictorTRB.get_val_loss_fn({"within_error_weight": w})
    train_loss = ProfilePredictorTRB.get_loss_fn(
        {"huber_delta": 0.1, "gradient_weight": 0.1, "huber_delta_grad": 1.0, "within_error_weight": w}
    )

    pred_perfect, targ = _pred_and_targ(ne, te, ne, te, **extras)
    assert float(val_loss(pred_perfect, targ)) == pytest.approx(0.0, abs=1e-9)

    # Half-sigma offset: fully inside the real error bars, so the loss is
    # exactly w times the same offset's loss without error bars
    pred_inside, _ = _pred_and_targ(ne + 0.5 * ne_sigma, te - 0.5 * te_sigma, ne, te, **extras)
    _, targ_no_err = _pred_and_targ(ne, te, ne, te)
    inside_loss = float(val_loss(pred_inside, targ))
    assert inside_loss > 0.0
    assert inside_loss == pytest.approx(w * float(val_loss(pred_inside, targ_no_err)), rel=1e-5)

    pred_outside, _ = _pred_and_targ(ne + 2.0 * ne_sigma, te - 2.0 * te_sigma, ne, te, **extras)
    assert float(val_loss(pred_outside, targ)) > inside_loss

    for pred in (pred_perfect, pred_inside, pred_outside):
        assert np.isfinite(float(train_loss(pred, targ)))


def test_nan_target_samples_never_reach_loss():
    """Samples with NaN targets must be dropped before batching.

    fresh_profile is computed from ne20 only (workflow.py), so a slice where
    the ne GP fit succeeded but the Te fit failed is labeled fresh with an
    all-NaN Te profile. The loss has no NaN mask, so it relies on the
    time-indep dataloader path (no state_init_vars, dropna how="any" on the
    stacked sample dim) removing any sample with a NaN in any var. This pins
    that contract: if get_dataloaders ever passes state_init_vars or the
    popsim dropna behavior changes, NaNs would poison every gradient in the
    batch and this test must catch it.
    """
    n_shot, n_t = 2, 5
    rng = np.random.default_rng(0)
    te = rng.random((n_shot, n_t, RHO.size))
    ne = rng.random((n_shot, n_t, RHO.size))
    ip = rng.random((n_shot, n_t))
    # failed Te GP fit on a fresh slice: Te all NaN, ne valid
    te[0, 2, :] = np.nan
    # single point removed by an individual range filter
    te[1, 3, 1] = np.nan
    time = np.tile(np.arange(n_t, dtype=float), (n_shot, 1))

    ds = xr.Dataset(
        {
            "t_e_keV": ((EPISODE_DIM, "time_idx", RADIAL_DIM), te),
            "n_e_1e20": ((EPISODE_DIM, "time_idx", RADIAL_DIM), ne),
            "ip_MA": ((EPISODE_DIM, "time_idx"), ip),
        },
        coords={EPISODE_DIM: [1, 2], RADIAL_DIM: RHO, TIME_COORD: ((EPISODE_DIM, "time_idx"), time)},
    )

    # Mirror the ProfilePredictorTRB.get_dataloaders call: no state_init_vars,
    # so make_dataloaders takes the time-indep path
    train_dl, val_dl = make_dataloaders(
        datasets=(ds, ds),
        time_coord=TIME_COORD,
        episode_coord=EPISODE_DIM,
        input_vars=["ip_MA"],
        target_vars=["t_e_keV", "n_e_1e20"],
        batch_size=None,
        shuffle=[True, False],
        convert_xr_to_jnp=False,
    )

    for dl in (train_dl, val_dl):
        sample_ds = dl.dataset.ds
        # Both poisoned slices dropped, clean ones kept
        assert sample_ds.sizes["sample"] == n_shot * n_t - 2
        assert not bool(sample_ds["t_e_keV"].isnull().any())
        assert not bool(sample_ds["n_e_1e20"].isnull().any())
        assert not bool(sample_ds["ip_MA"].isnull().any())
