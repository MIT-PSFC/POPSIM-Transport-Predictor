"""Tests for the profile predictor training and validation losses.

Everything runs on hand-built profiles so the expected loss values are exact,
plus one end-to-end check against real GP-fit signals from a sample dataset.
The training loss is huber on the peak-normalized residual, the validation loss is chi.
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
from transport_study.modules.trb_utils import (
    CHI_ERROR_VARS,
    GRAD_RHO_MAX,
    chi_sigma_floors,
)
from transport_study.orchestration.organize_data import (
    PROFILE_TARGET_VARS,
    get_ds,
    get_loaded_shot_count,
)
from transport_study.profile_transfer.profile_study import ProfileStudy
from transport_study.tests.sample_data import SAMPLE_DIR, requires_sample_data

RHO = RHO_GRID
RHO_MID = 0.5 * (RHO[:-1] + RHO[1:])
TRAIN_CONFIG = {"huber_delta": 0.5, "gradient_weight": 0.1, "huber_delta_grad": 5.0}
# Fractional (peak-normalized) error bar of the hand-built targets, far above SIGMA_FLOOR
SIGMA_FRAC = 0.05
SIGMA_FLOOR = 1e-3


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
    """Every test needs a loaded config, study tests get theirs from study_config."""
    if "study_config" not in request.fixturenames:
        request.getfixturevalue("study_config")


def _val_config(gradient_weight: float = 0.1, floor: float = SIGMA_FLOOR, **overrides) -> dict:
    """Chi validation loss config with the same sigma floor for every device and error bar."""
    floors = {var: floor for error_vars in CHI_ERROR_VARS.values() for var in error_vars}
    return {"gradient_weight": gradient_weight, "chi_sigma_floors": dict.fromkeys(config.dataset_paths, floors), **overrides}


def _profile_da(values):
    return xr.DataArray(data=jnp.asarray(values), dims=(RADIAL_DIM,), coords={RADIAL_DIM: RHO})


def _pred_and_targ(ne_pred, te_pred, ne_targ, te_targ, **targ_overrides):
    """Build a prediction/target pair for the loss.

    The targets carry every companion the losses read:
    gradients exact for the quadratic test profiles (second-order differences)
    and flat error bars of SIGMA_FRAC times each profile's peak.
    targ_overrides replaces any of them by name.
    """
    pred = Outputs(ne=_profile_da(ne_pred), te=_profile_da(te_pred))
    targ_values = {"n_e_1e20": np.asarray(ne_targ), "t_e_keV": np.asarray(te_targ)}
    for channel in ("n_e_1e20", "t_e_keV"):
        profile = targ_values[channel]
        targ_values[f"{channel}_gradient"] = np.gradient(profile, RHO, edge_order=2)
        sigma = np.full_like(RHO, SIGMA_FRAC * np.max(np.abs(profile)))
        targ_values[f"{channel}_error"] = sigma
        targ_values[f"{channel}_gradient_error"] = sigma
    targ_values.update(targ_overrides)
    targ = {name: _profile_da(values) for name, values in targ_values.items()}
    targ["ds_source_idx"] = xr.DataArray(data=jnp.asarray(0.0))
    return pred, targ


def test_losses_zero_for_perfect_prediction():
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    pred, targ = _pred_and_targ(ne, te, ne, te)
    assert float(ProfilePredictorTRB.get_loss_fn(TRAIN_CONFIG)(pred, targ)) == pytest.approx(0.0)
    assert float(ProfilePredictorTRB.get_val_loss_fn(_val_config())(pred, targ)) == pytest.approx(0.0)


def test_gradient_term_penalizes_slope_mismatch():
    """Two predictions with identical pointwise value error but different gradients.

    A constant offset keeps the target's slope, an alternating offset of the same
    magnitude corrupts it. The value-only loss cannot tell them apart, the
    gradient term must, in training and in validation alike.
    """
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    c = 0.2
    offset_const = np.full_like(RHO, c)
    offset_alt = c * np.where(np.arange(RHO.size) % 2 == 0, 1.0, -1.0)

    pred_const, targ = _pred_and_targ(ne + offset_const, te + offset_const, ne, te)
    pred_alt, _ = _pred_and_targ(ne + offset_alt, te + offset_alt, ne, te)

    for value_only, with_grad in (
        (ProfilePredictorTRB.get_loss_fn({**TRAIN_CONFIG, "gradient_weight": 0.0}), ProfilePredictorTRB.get_loss_fn(TRAIN_CONFIG)),
        (ProfilePredictorTRB.get_val_loss_fn(_val_config(gradient_weight=0.0)), ProfilePredictorTRB.get_val_loss_fn(_val_config())),
    ):
        assert float(value_only(pred_const, targ)) == pytest.approx(float(value_only(pred_alt, targ)), rel=1e-5)
        assert float(with_grad(pred_alt, targ)) > float(with_grad(pred_const, targ))
        # Constant offset leaves gradients untouched, so the gradient term adds nothing
        assert float(with_grad(pred_const, targ)) == pytest.approx(float(value_only(pred_const, targ)), rel=1e-5)


def test_device_weights_scale_loss():
    """device_weights multiplies the loss for samples from the matching device only.

    Samples in _pred_and_targ carry ds_source_idx 0, which maps to the
    alphabetically-first device. Weighting that device by 3 must triple both losses,
    weighting any other device must leave them unchanged.
    """
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    pred, targ = _pred_and_targ(ne + 0.3 * RHO, te - 0.2 * RHO, ne, te)

    devices = sorted(config.dataset_paths)
    weights_sample_device = dict.fromkeys(devices, 1.0)
    weights_sample_device[devices[0]] = 3.0
    weights_other_device = dict.fromkeys(devices, 1.0)
    weights_other_device[devices[-1]] = 3.0

    for make_loss_fn, loss_config in (
        (ProfilePredictorTRB.get_loss_fn, TRAIN_CONFIG),
        (ProfilePredictorTRB.get_val_loss_fn, _val_config()),
    ):
        unweighted = float(make_loss_fn(loss_config)(pred, targ))
        weighted = float(make_loss_fn({**loss_config, "device_weights": weights_sample_device})(pred, targ))
        other_weighted = float(make_loss_fn({**loss_config, "device_weights": weights_other_device})(pred, targ))
        assert weighted == pytest.approx(3.0 * unweighted, rel=1e-5)
        assert other_weighted == pytest.approx(unweighted, rel=1e-6)


def test_channels_balanced_by_peak_normalization():
    """Equal fractional error must cost the same in ne and Te.

    Both losses run on peak-normalized profiles, so a 10% error on a tiny ne
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
        ProfilePredictorTRB.get_val_loss_fn(_val_config()),
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

    small = {"huber_delta": 0.01, "huber_delta_grad": 0.1, "gradient_weight": 0.1}
    large = {"huber_delta": 1.0, "huber_delta_grad": 10.0, "gradient_weight": 0.1}

    val_small = float(ProfilePredictorTRB.get_val_loss_fn(_val_config(**small))(pred, targ))
    val_large = float(ProfilePredictorTRB.get_val_loss_fn(_val_config(**large))(pred, targ))
    assert val_small == pytest.approx(val_large, rel=1e-6)
    assert val_small > 0.0

    train_small = float(ProfilePredictorTRB.get_loss_fn(small)(pred, targ))
    train_large = float(ProfilePredictorTRB.get_loss_fn(large)(pred, targ))
    assert train_small != pytest.approx(train_large, rel=1e-3)


def test_normalized_loss_magnitude_is_numerically_safe():
    """Peak normalization shrinks the training loss - check it stays well above float32 resolution.

    A typical 5% profile error gives a training huber loss of order 1e-3,
    and a chi of order one when the error bars are 5% too.
    """
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    pred, targ = _pred_and_targ(1.05 * ne, 1.05 * te, ne, te)

    train_loss = float(ProfilePredictorTRB.get_loss_fn({"huber_delta": 0.1, "gradient_weight": 0.1, "huber_delta_grad": 1.0})(pred, targ))
    val_loss = float(ProfilePredictorTRB.get_val_loss_fn(_val_config())(pred, targ))

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


def test_chi_counts_the_residual_in_error_bars():
    """A prediction offset by k error bars at every point scores chi k per channel, in units of the GP-fit error bar.

    The offset is constant, so the gradients are untouched and only the value chi counts.
    The training loss never reads the error bars.
    """
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    val_loss = ProfilePredictorTRB.get_val_loss_fn(_val_config())
    train_loss = ProfilePredictorTRB.get_loss_fn(TRAIN_CONFIG)
    ne_sigma = SIGMA_FRAC * 1.5
    te_sigma = SIGMA_FRAC * 3.0

    for k in (0.5, 3.0):
        pred, targ = _pred_and_targ(ne + k * ne_sigma, te - k * te_sigma, ne, te)
        # Value chi is k at every point, integrated over rho in [0, 1], for each of the two channels
        assert float(val_loss(pred, targ)) == pytest.approx(2.0 * k, rel=1e-6)

    pred, targ = _pred_and_targ(ne + ne_sigma, te - te_sigma, ne, te)
    _, targ_wide_bars = _pred_and_targ(
        ne, te, ne, te, n_e_1e20_error=np.full_like(RHO, 10 * ne_sigma), t_e_keV_error=np.full_like(RHO, 10 * te_sigma)
    )
    assert float(train_loss(pred, targ)) == pytest.approx(float(train_loss(pred, targ_wide_bars)), rel=1e-9)


def test_chi_floor_caps_tight_error_bars():
    """Error bars below the device floor count as the floor, so one overconfident fit point has bounded weight."""
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    floor = 0.02
    offset = 0.01  # Normalized by the 1.5 / 3.0 peaks, 0.0067 and 0.0033 fractional
    tight = {var: np.full_like(RHO, 1e-6) for var in ("n_e_1e20_error", "t_e_keV_error")}
    pred, targ = _pred_and_targ(ne + offset, te + offset, ne, te, **tight)

    val_loss = ProfilePredictorTRB.get_val_loss_fn(_val_config(floor=floor))

    assert float(val_loss(pred, targ)) == pytest.approx((offset / 1.5 + offset / 3.0) / floor, rel=1e-6)


def test_measured_gradient_signal_is_the_gradient_target():
    """The GP-fit gradient signals define the gradient target, in both losses.

    The prediction matches the target values exactly, so any loss can only come
    from the measured gradient signal disagreeing with the prediction's slope.
    In chi that disagreement counts in units of the gradient error bar, out to GRAD_RHO_MAX like in training.
    """
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    grad_offset = 1.0
    grad_sigma = 2.0 * grad_offset
    pred, targ = _pred_and_targ(
        ne,
        te,
        ne,
        te,
        n_e_1e20_gradient=-3.0 * RHO + grad_offset,
        t_e_keV_gradient=-6.0 * RHO + grad_offset,
        n_e_1e20_gradient_error=np.full_like(RHO, grad_sigma),
        t_e_keV_gradient_error=np.full_like(RHO, grad_sigma),
    )

    assert float(ProfilePredictorTRB.get_loss_fn({**TRAIN_CONFIG, "gradient_weight": 0.0})(pred, targ)) == pytest.approx(0.0, abs=1e-9)
    assert float(ProfilePredictorTRB.get_loss_fn(TRAIN_CONFIG)(pred, targ)) > 0.0

    # Peak normalization divides offset and error bar alike, so each midpoint scores offset / sigma
    chi_grad_channel = np.trapezoid((RHO_MID < GRAD_RHO_MAX) * grad_offset / grad_sigma, x=RHO_MID)
    expected = 0.1 * 2.0 * chi_grad_channel
    assert float(ProfilePredictorTRB.get_val_loss_fn(_val_config())(pred, targ)) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    ("make_loss_fn", "loss_config", "rho_cut"),
    [
        (ProfilePredictorTRB.get_loss_fn, TRAIN_CONFIG, GRAD_RHO_MAX),
        (ProfilePredictorTRB.get_val_loss_fn, None, GRAD_RHO_MAX),
    ],
    ids=["train", "chi"],
)
def test_gradient_terms_count_only_inside_their_rho_cut(make_loss_fn, loss_config, rho_cut):
    """Gradient mismatch beyond GRAD_RHO_MAX must not contribute, in training or in chi, the same mismatch in the core must.

    The measured gradients at the edge are unreliable.
    The offset sits on the grid points at and beyond the next point past the cut,
    so every midpoint it reaches lies beyond the cut.
    """
    ne = 1.5 * (1 - RHO**2)
    te = 3.0 * (1 - RHO**2)
    loss_fn = make_loss_fn(_val_config() if loss_config is None else loss_config)
    first_offset_point = RHO[np.flatnonzero(RHO > rho_cut)[0] + 1]

    edge_offset = np.where(RHO >= first_offset_point, 50.0, 0.0)
    pred, targ_edge = _pred_and_targ(ne, te, ne, te, n_e_1e20_gradient=-3.0 * RHO + edge_offset, t_e_keV_gradient=-6.0 * RHO + edge_offset)
    assert float(loss_fn(pred, targ_edge)) == pytest.approx(0.0, abs=1e-9)

    core_offset = np.where(RHO <= 0.5, 50.0, 0.0)
    _, targ_core = _pred_and_targ(ne, te, ne, te, n_e_1e20_gradient=-3.0 * RHO + core_offset, t_e_keV_gradient=-6.0 * RHO + core_offset)
    assert float(loss_fn(pred, targ_core)) > 0.0


@pytest.mark.slow
@requires_sample_data
def test_loss_on_prepared_sample_dataset():
    """Full-pipeline check: signals prepared by get_ds feed the losses directly.

    Pulls one fully-finite timeslice of the prepared cmod-high sample dataset
    (real GP-fit profiles, gradients and error bars on the uniform 51-point
    rho grid) with the device's real chi_sigma_floors:
    perfect predictions cost nothing, a two-sigma miss costs more than a half-sigma miss,
    and both losses stay finite on real signals.
    """
    ds = get_ds("cmod-high", "profile_transfer")

    finite = np.ones((ds.sizes[EPISODE_DIM], ds.sizes["time_idx"]), dtype=bool)
    for var in PROFILE_TARGET_VARS:
        finite &= np.isfinite(ds[var].values).all(axis=-1)
    assert finite.any(), "sample dataset has no fully-finite timeslice"
    i_shot, i_time = np.argwhere(finite)[0]
    ts = ds.isel({EPISODE_DIM: i_shot, "time_idx": i_time})

    ne = ts["n_e_1e20"].values
    te = ts["t_e_keV"].values
    ne_sigma = ts["n_e_1e20_error"].values
    te_sigma = ts["t_e_keV_error"].values
    extras = {var: ts[var].values for var in PROFILE_TARGET_VARS if var not in ("n_e_1e20", "t_e_keV")}

    floors = {device: chi_sigma_floors(device) for device in config.dataset_paths}
    val_loss = ProfilePredictorTRB.get_val_loss_fn({"gradient_weight": 0.1, "chi_sigma_floors": floors})
    train_loss = ProfilePredictorTRB.get_loss_fn({"huber_delta": 0.1, "gradient_weight": 0.1, "huber_delta_grad": 1.0})

    pred_perfect, targ = _pred_and_targ(ne, te, ne, te, **extras)
    assert float(val_loss(pred_perfect, targ)) == pytest.approx(0.0, abs=1e-9)

    pred_inside, _ = _pred_and_targ(ne + 0.5 * ne_sigma, te - 0.5 * te_sigma, ne, te, **extras)
    pred_outside, _ = _pred_and_targ(ne + 2.0 * ne_sigma, te - 2.0 * te_sigma, ne, te, **extras)
    assert float(val_loss(pred_outside, targ)) > float(val_loss(pred_inside, targ)) > 0.0

    for pred in (pred_perfect, pred_inside, pred_outside):
        assert np.isfinite(float(train_loss(pred, targ)))
        assert np.isfinite(float(val_loss(pred, targ)))


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
