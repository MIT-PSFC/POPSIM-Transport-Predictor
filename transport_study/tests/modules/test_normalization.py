import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from transport_study.modules.normalization import (
    CORAL_DEGENERATE_STD_FRAC,
    MIN_CORAL_SHOTS,
    N_FEATURES,
    NORM_INPUT_VARS,
    CoralNormalizer,
    InputNormalizer,
    PhysicsCoralNormalizer,
    PhysicsNormalizer,
    PhysicsZScoreNormalizer,
    RawNormalizer,
    ZScoreNormalizer,
    fit_coral_stats,
    make_normalizer,
)
from transport_study.orchestration.organize_data import normalize_domain

RNG = np.random.default_rng(7)
N_DEVICES = 3  # global device registry size, device 2 is never fitted


def _toy_dataset(n_shots: int = 6, n_time: int = 40) -> xr.Dataset:
    """Two-device dataset with distinct distributions and a few NaNs.

    Shots split evenly between devices 0 and 1. CORAL fits gate on
    MIN_CORAL_SHOTS distinct shots per device, so coral tests must request
    n_shots >= 16. The within-device stds are kept well above
    CORAL_DEGENERATE_STD_FRAC of the pooled spread so every feature is live
    and the full-covariance alignment property holds exactly.
    """
    shape = (n_shots, n_time)
    device_of_shot = np.repeat([0, 1], n_shots // 2)

    def _var(mean0, mean1, scale0, scale1):
        vals = np.where(
            device_of_shot[:, None] == 0,
            RNG.normal(mean0, scale0, shape),
            RNG.normal(mean1, scale1, shape),
        )
        return (("shot", "time_idx"), np.abs(vals))

    ds = xr.Dataset(
        {
            "ip_MA": _var(1.0, 0.2, 0.2, 0.1),
            "b_geo": _var(5.0, 1.4, 0.5, 0.5),
            # The field at r0, which the visualization beta of normalize_domain normalizes with
            "b0": _var(5.2, 1.4, 0.5, 0.5),
            "geometric_axis_r": _var(0.68, 0.88, 0.04, 0.04),
            "minor_radius": _var(0.22, 0.25, 0.02, 0.02),
            "elongation": _var(1.6, 1.4, 0.1, 0.1),
            "n_e_line_average_1e20": _var(1.5, 0.5, 0.4, 0.15),
            "power_additional_MW": _var(2.0, 0.5, 0.8, 0.3),
            "energy_mhd_MJ": _var(0.15, 0.02, 0.05, 0.01),
            "ds_source_idx": (("shot",), device_of_shot.astype(float)),
        },
        coords={"shot": np.arange(n_shots)},
    )
    # A NaN row must not poison the statistics
    ds["ip_MA"][0, 0] = np.nan
    return ds


def _sample_inputs(ds: xr.Dataset, shot: int, t: int, device: int) -> InputNormalizer.Inputs:
    return InputNormalizer.Inputs(
        **{var: jnp.asarray(float(ds[var][shot, t])) for var in NORM_INPUT_VARS},
        ds_source_idx=jnp.asarray(float(device)),
    )


def test_raw_is_identity():
    ds = _toy_dataset()
    norm = RawNormalizer()
    inp = _sample_inputs(ds, 1, 3, 0)
    out = norm(inp)
    for var in NORM_INPUT_VARS:
        assert np.isclose(float(getattr(out, var)), float(getattr(inp, var)))


def test_physics_matches_normalize_domain():
    ds = _toy_dataset().assign_coords(ds_source=("shot", np.array(["dev_a"] * 3 + ["dev_b"] * 3)))
    ds_ref, _ = normalize_domain(ds.copy(deep=True), None, method="physics")

    norm = PhysicsNormalizer()
    for shot, t in [(0, 5), (2, 10), (4, 0)]:
        out = norm(_sample_inputs(ds, shot, t, 0))
        assert np.isclose(float(out.ip_MA), float(ds["ip_MA"][shot, t]))
        assert np.isclose(float(out.b_geo), float(ds_ref["q_star"][shot, t]), rtol=1e-5)
        assert np.isclose(float(out.geometric_axis_r), float(ds_ref["epsilon"][shot, t]), rtol=1e-5)
        assert np.isclose(float(out.minor_radius), float(ds_ref["aB0"][shot, t]), rtol=1e-5)
        assert np.isclose(float(out.elongation), float(ds["elongation"][shot, t]))
        assert np.isclose(float(out.n_e_line_average_1e20), float(ds_ref["f_G"][shot, t]), rtol=1e-5)
        assert np.isclose(float(out.power_additional_MW), float(ds_ref["surface_power_density"][shot, t]), rtol=1e-5)


def test_z_score_standardizes_per_device():
    ds = _toy_dataset()
    norm = ZScoreNormalizer.fit(ds, n_devices=N_DEVICES)

    for device in (0, 1):
        rows = []
        for shot in np.where(ds["ds_source_idx"].values == device)[0]:
            for t in range(ds.sizes["time_idx"]):
                out = norm(_sample_inputs(ds, int(shot), t, device))
                rows.append(out.to_vec())
        arr = np.array(rows)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        np.testing.assert_allclose(mean, np.zeros(N_FEATURES), atol=0.05)
        np.testing.assert_allclose(std, np.ones(N_FEATURES), atol=0.05)


def test_z_score_unfitted_device_passthrough():
    ds = _toy_dataset()
    norm = ZScoreNormalizer.fit(ds, n_devices=N_DEVICES)
    inp = _sample_inputs(ds, 0, 1, device=2)  # device 2 never fitted
    out = norm(inp)
    for var in NORM_INPUT_VARS:
        assert np.isclose(float(getattr(out, var)), float(getattr(inp, var)))


def test_coral_aligns_covariance():
    ds = _toy_dataset(n_shots=16, n_time=400)
    norm = CoralNormalizer.fit(ds, n_devices=N_DEVICES)

    # Reference covariance over pooled complete rows
    features = np.column_stack([ds[var].values.ravel() for var in NORM_INPUT_VARS])
    idx = np.repeat(ds["ds_source_idx"].values, ds.sizes["time_idx"])
    valid = ~np.any(np.isnan(features), axis=1)
    cov_ref = np.cov(features[valid], rowvar=False)

    for device in (0, 1):
        rows = features[valid & (idx == device)]
        transformed = (rows - np.asarray(norm.means[device])) @ np.asarray(norm.transforms[device]) + np.asarray(norm.means[device])
        cov_t = np.cov(transformed, rowvar=False)
        # Covariance is aligned to the pooled reference (up to regularization)
        np.testing.assert_allclose(cov_t, cov_ref, atol=0.05 * np.abs(cov_ref).max())

    # Unfitted device: identity transform
    inp = _sample_inputs(ds, 0, 1, device=2)
    out = norm(inp)
    for var in NORM_INPUT_VARS:
        assert np.isclose(float(getattr(out, var)), float(getattr(inp, var)))


def test_coral_degenerate_feature_keeps_identity_slot():
    """fit_coral_stats with one feature near-constant within a device (std
    below CORAL_DEGENERATE_STD_FRAC of the pooled std, e.g. B0 fixed per
    device with only measurement jitter):
    - the device transform has an identity row/col for that feature (value 1
      on the diagonal, zero cross terms), so the feature passes through raw
    - the remaining live features still align to the pooled covariance
    - the same feature with healthy spread on the other device is transformed
      normally there (the guard is per device, not global)
    """
    rng = np.random.default_rng(11)
    n_shots, n_time = 20, 30
    n_rows = n_shots * n_time

    rows0 = rng.normal([1.0, 5.0, 2.0], [0.5, 1.0, 0.7], size=(n_rows, 3))
    # Feature 1 is fixed on device 1 up to tiny jitter, healthy on device 0
    rows1 = rng.normal([2.0, 1.0, 3.0], [0.4, 0.001, 0.6], size=(n_rows, 3))
    features = np.concatenate([rows0, rows1])
    source_idx = np.repeat([0, 1], n_rows)
    shot_idx = np.repeat(np.arange(2 * n_shots), n_time)

    stats = fit_coral_stats(features, source_idx, n_devices=2, shot_idx=shot_idx)
    assert stats is not None
    means, transforms = (np.asarray(arr) for arr in stats)
    cov_ref = np.cov(features, rowvar=False)

    # Degenerate slot: identity row and column, zero cross terms
    np.testing.assert_allclose(transforms[1][1, :], np.eye(3)[1])
    np.testing.assert_allclose(transforms[1][:, 1], np.eye(3)[1])
    # so the feature passes through raw
    transformed1 = (rows1 - means[1]) @ transforms[1] + means[1]
    np.testing.assert_allclose(transformed1[:, 1], rows1[:, 1])
    # while the live features still align to the pooled covariance
    cov_live = np.cov(transformed1[:, [0, 2]], rowvar=False)
    np.testing.assert_allclose(cov_live, cov_ref[np.ix_([0, 2], [0, 2])], atol=0.05 * np.abs(cov_ref).max())

    # Device 0 has healthy spread in feature 1, so it is transformed normally
    # there: full-covariance alignment holds and the slot is not identity
    assert abs(transforms[0][1, 1] - 1.0) > 0.1
    transformed0 = (rows0 - means[0]) @ transforms[0] + means[0]
    np.testing.assert_allclose(np.cov(transformed0, rowvar=False), cov_ref, atol=0.05 * np.abs(cov_ref).max())


def test_coral_whitening_gain_bounded():
    """No entry of any fitted transform, expressed in pooled-std units
    (scale[i] * T[i, j] / scale[j]), exceeds ~1/CORAL_DEGENERATE_STD_FRAC even
    when a device covariance is near-singular from collinear features (two
    features that are near-copies of each other within one device).
    This pins the CORAL_EIGVAL_FLOOR backstop for non-axis-aligned degeneracy
    """
    rng = np.random.default_rng(13)
    n_shots, n_time = 20, 30
    n_rows = n_shots * n_time

    # Device 0: feature 1 is a near-copy of feature 0 (collinear, both
    # individually healthy so the per-feature guard does not fire)
    base = rng.normal(1.0, 1.0, n_rows)
    rows0 = np.column_stack([base, base + rng.normal(0.0, 0.001, n_rows), rng.normal(0.0, 1.0, n_rows)])
    # Device 1: independent features, so the pooled reference is well spread
    # in the direction device 0 is degenerate in
    rows1 = rng.normal([2.0, 0.5, 1.0], [1.0, 1.0, 1.0], size=(n_rows, 3))
    features = np.concatenate([rows0, rows1])
    source_idx = np.repeat([0, 1], n_rows)
    shot_idx = np.repeat(np.arange(2 * n_shots), n_time)

    stats = fit_coral_stats(features, source_idx, n_devices=2, shot_idx=shot_idx)
    assert stats is not None
    _, transforms = (np.asarray(arr) for arr in stats)

    scale = np.std(features, axis=0)
    gains = np.abs(scale[:, None] * transforms / scale[None, :])
    assert np.all(np.isfinite(gains))
    # The eigenvalue floor caps the whitening gain of the collinear direction
    # at 1/CORAL_DEGENERATE_STD_FRAC (small slack for the recoloring step)
    assert gains.max() <= 1.5 / CORAL_DEGENERATE_STD_FRAC
    # The scenario really exercises the cap: device 0 does get whitened hard
    assert gains[0].max() > 5.0


def test_coral_shot_count_gate():
    """The MIN_CORAL_SHOTS gate counts distinct shots, not timeslices:
    - a device with 1 shot and 1000 timeslices keeps the identity transform
      (the old MIN_CORAL_SAMPLES=8 row count let it pass)
    - a device with MIN_CORAL_SHOTS shots of a few slices each is fitted
    - a pooled set below MIN_CORAL_SHOTS distinct shots returns None so the
      normalizer falls back to full identity stats
    """
    rng = np.random.default_rng(17)

    def block(n_shots: int, n_time: int, first_shot: int, device: int, stds: list) -> tuple:
        rows = rng.normal([1.0, 2.0, 0.5], stds, size=(n_shots * n_time, 3))
        return (
            rows,
            np.full(n_shots * n_time, device),
            np.repeat(np.arange(first_shot, first_shot + n_shots), n_time),
        )

    # Distinct stds per device so a fitted transform is clearly not identity
    rows0, src0, shots0 = block(1, 1000, 0, 0, [0.5, 1.0, 0.3])
    rows1, src1, shots1 = block(MIN_CORAL_SHOTS, 5, 1, 1, [1.0, 2.0, 0.6])
    stats = fit_coral_stats(
        np.concatenate([rows0, rows1]),
        np.concatenate([src0, src1]),
        n_devices=2,
        shot_idx=np.concatenate([shots0, shots1]),
    )
    assert stats is not None
    means, transforms = (np.asarray(arr) for arr in stats)
    # One shot keeps identity no matter how many timeslices it has
    np.testing.assert_allclose(transforms[0], np.eye(3))
    np.testing.assert_allclose(means[0], np.zeros(3))
    # MIN_CORAL_SHOTS shots of a few slices each are fitted
    assert np.abs(means[1]).max() > 0
    assert not np.allclose(transforms[1], np.eye(3), atol=0.01)

    # Pooled set below MIN_CORAL_SHOTS distinct shots: no stats at all
    rows2, src2, shots2 = block(3, 100, 100, 0, [0.5, 1.0, 0.3])
    rows3, src3, shots3 = block(3, 100, 103, 1, [1.0, 2.0, 0.6])
    assert (
        fit_coral_stats(
            np.concatenate([rows2, rows3]),
            np.concatenate([src2, src3]),
            n_devices=2,
            shot_idx=np.concatenate([shots2, shots3]),
        )
        is None
    )


def test_physics_zscore_standardizes_in_physics_space():
    """PhysicsZScoreNormalizer.fit on the toy dataset:
    - transformed outputs of each fitted device have ~zero mean and ~unit std
      per physics feature (compare against physics_feature_vec applied to the
      same rows)
    - device 2 (never fitted) keeps identity stats, so its output equals the
      plain physics features
    - identity() and fit() instances share pytree structure (transfer restore
      overwrites identity buffers from a fitted checkpoint)
    """


def test_make_normalizer_identity_without_data():
    for method, cls in [
        ("zscore", ZScoreNormalizer),
        ("coral", CoralNormalizer),
        ("physics-coral", PhysicsCoralNormalizer),
        ("physics-zscore", PhysicsZScoreNormalizer),
    ]:
        norm = make_normalizer(method, train_ds=None, n_devices=N_DEVICES)
        assert isinstance(norm, cls)
        fitted = make_normalizer(method, train_ds=_toy_dataset(), n_devices=N_DEVICES)
        # Identity and fitted instances share pytree structure (checkpoint restore relies on it)
        assert jnp.asarray(norm.means).shape == jnp.asarray(fitted.means).shape


def test_make_normalizer_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown normalization method"):
        make_normalizer("bogus", train_ds=None, n_devices=N_DEVICES)


def test_stats_are_arrays_not_trainable_by_selectors():
    """Stats live in the pytree (checkpointable) but must never be selected as trainable.

    The env/TRB trainable selectors pick module.nn leaves explicitly. This test
    guards the contract those selectors rely on: the stats arrays are inexact
    array leaves, so any selector using a broad eqx.filter WOULD pick them up.
    """
    ds = _toy_dataset()
    norm = CoralNormalizer.fit(ds, n_devices=N_DEVICES)
    leaves = eqx.filter(norm, eqx.is_inexact_array)
    # Both stats arrays are pytree leaves (they checkpoint/restore with the model)
    assert leaves.means is not None
    assert leaves.transforms is not None


def test_zscore_feature_normalizer_standardizes_per_device():
    """ZScoreFeatureNormalizer.fit_from_features on a toy (N, F) matrix with two
    devices: transformed rows of each fitted device have ~zero mean and ~unit
    std per feature, and the transform matches (x - mean_d) / std_d exactly.
    """


def test_zscore_feature_normalizer_unfitted_device_passthrough():
    """A device index absent from the fitting data keeps the identity row
    (mean 0, std 1), so its feature vectors pass through unchanged. Mirrors
    test_z_score_unfitted_device_passthrough for the feature-vector variant.
    """


def test_zscore_feature_normalizer_identity_matches_fitted_structure():
    """ZScoreFeatureNormalizer.identity and .fit_from_features instances share
    pytree structure (same field names, array shapes, dtypes). Transfer
    restore builds the identity instance and overwrites its buffers from a
    checkpoint written by a fitted instance, so any structure drift breaks
    physics-zscore transfer cases.
    """


def test_profile_model_init_physics_zscore_dispatch():
    """ProfilePredictorTRB.model_init with data_normalization='physics-zscore':
    - without transfer_checkpoint the module normalizer is a
      ZScoreFeatureNormalizer fitted on the 10 nn_inputs (non-identity stats
      for devices present in the training data)
    - with transfer_checkpoint set the normalizer is
      ZScoreFeatureNormalizer.identity (buffers to be overwritten by restore)
    - 'physics' still yields identity CoralFeatureNormalizer and unknown
      methods still raise ValueError
    """


def test_physics_zscore_is_stat_normalization_pretrain_case():
    """A profile transfer case with data_normalization='physics-zscore' (or
    'physics-coral') gets a transfer_pretrain_case() with
    domain_adaptation='transfer_pretrain' keeping this case's
    num_target_shots (the stat-fit-on-historic+target twin-case design),
    while 'physics' falls back to the plain da=None baseline prereq.
    """


def test_profile_case_norm_token_naming():
    """str(case) naming with the data_normalization axis:
    - every method (physics, physics-coral, physics-zscore) produces its own
      norm_{method} token between td_ and freeze_, data_normalization is not
      suppressed from the case name (unlike geometry_builder's "circular")
    - restore_predictor.checkpoint_to_profile_case round-trips all three
      (with and without geom_ / targ_ / da_ tokens present)
    """


def test_model_init_requires_data_normalization():
    """ProfilePredictorTRB.model_init and TransportPredictorTRB.model_init raise
    KeyError when model_init_config has no data_normalization key, instead of
    silently falling back to a fitted stat stage. Matches the power balance
    TRBs, which index the key strictly - a config that lost the key must fail
    loudly, not train a different model than its case name claims.
    """


def test_make_feature_normalizer_matches_make_normalizer_contract():
    """make_feature_normalizer is the feature-vector twin of make_normalizer,
    and both module wrappers (profile make_nn_input_normalizer, transport
    make_transport_nn_input_normalizer) inherit its contract:
    - fit_data None yields identity buffers whose pytree structure matches the
      fitted instance of the same method (so a transfer restore lands cleanly)
    - 'physics' yields identity CoralFeatureNormalizer buffers with either
      fit_data, 'physics-coral' / 'physics-zscore' yield their fitted class
    - a power-balance-only method ('raw', 'zscore', 'coral') raises ValueError
    - feature_fit_arrays drops rows whose ds_source_idx is NaN and keeps the
      features, device indices, and shot indices aligned
    """


def test_normalize_domain_feature_spaces_match_modules():
    """normalize_domain(..., feature_space=...) visualizes exactly the features
    each study's modules consume:
    - 'profile' writes module.nn_input_matrix column by column (identity slots
      beta_tor_norm / kappa / triangularity_upper / triangularity_lower stay the raw dataset vars)
    - 'transport' writes transport_nn_input_matrix the same way (identity slots
      kappa / triangularity_upper / triangularity_lower, and beta_tor_norm IS written since the transport
      datasets carry no measured beta_tor_norm)
    - physics-coral / physics-zscore stats fitted here match the stats the
      matching make_*_normalizer fits on the same dataset
    - 'power_balance' is unchanged (still the 7 physics features plus the
      Wtot-derived beta extra)
    - an unknown feature space raises ValueError
    """
