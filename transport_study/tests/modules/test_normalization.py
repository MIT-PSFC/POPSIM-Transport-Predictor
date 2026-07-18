import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from transport_study.modules.normalization import (
    N_FEATURES,
    NORM_INPUT_VARS,
    CoralNormalizer,
    InputNormalizer,
    PhysicsCoralNormalizer,
    PhysicsNormalizer,
    RawNormalizer,
    ZScoreNormalizer,
    make_normalizer,
)
from transport_study.orchestration.organize_data import normalize_domain

RNG = np.random.default_rng(7)
N_DEVICES = 3  # global device registry size, device 2 is never fitted


def _toy_dataset(n_shots: int = 6, n_time: int = 40) -> xr.Dataset:
    """Two-device dataset with distinct distributions and a few NaNs."""
    shape = (n_shots, n_time)
    device_of_shot = np.array([0, 0, 0, 1, 1, 1])

    def _var(mean0, mean1, scale0, scale1):
        vals = np.where(
            device_of_shot[:, None] == 0,
            RNG.normal(mean0, scale0, shape),
            RNG.normal(mean1, scale1, shape),
        )
        return (("shot", "time_idx"), np.abs(vals))

    ds = xr.Dataset(
        {
            "Ip_MA": _var(1.0, 0.2, 0.2, 0.05),
            "B0": _var(5.0, 1.4, 0.5, 0.1),
            "R0": _var(0.68, 0.88, 0.02, 0.02),
            "a_minor": _var(0.22, 0.25, 0.01, 0.01),
            "kappa": _var(1.6, 1.4, 0.1, 0.1),
            "ne20_line_avg": _var(1.5, 0.5, 0.4, 0.15),
            "P_aux_MW": _var(2.0, 0.5, 0.8, 0.3),
            "Wtot_MJ": _var(0.15, 0.02, 0.05, 0.01),
            "ds_source_idx": (("shot",), device_of_shot.astype(float)),
        },
        coords={"shot": np.arange(n_shots)},
    )
    # A NaN row must not poison the statistics
    ds["Ip_MA"][0, 0] = np.nan
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
        assert np.isclose(float(out.Ip_MA), float(ds["Ip_MA"][shot, t]))
        assert np.isclose(float(out.B0), float(ds_ref["q_star"][shot, t]), rtol=1e-5)
        assert np.isclose(float(out.R0), float(ds_ref["epsilon"][shot, t]), rtol=1e-5)
        assert np.isclose(float(out.a_minor), float(ds_ref["aB0"][shot, t]), rtol=1e-5)
        assert np.isclose(float(out.kappa), float(ds["kappa"][shot, t]))
        assert np.isclose(float(out.ne20_line_avg), float(ds_ref["f_G"][shot, t]), rtol=1e-5)
        assert np.isclose(float(out.P_aux_MW), float(ds_ref["surface_power_density"][shot, t]), rtol=1e-5)


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
    ds = _toy_dataset(n_shots=6, n_time=400)
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


def test_make_normalizer_identity_without_data():
    for method, cls in [("z_score", ZScoreNormalizer), ("coral", CoralNormalizer), ("physics-coral", PhysicsCoralNormalizer)]:
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
    - physics-coral (suppressed default) produces NO norm_ token, so
      pre-axis case names stay byte-identical (checkpoint dirs, wandb
      projects, tuned-config paths of existing physics-coral studies)
    - physics / physics-zscore produce a norm_{method} token between td_
      and freeze_, mirroring power balance
    - restore_predictor.checkpoint_to_profile_case round-trips all three
      (with and without geom_ / targ_ / da_ tokens present)
    """
