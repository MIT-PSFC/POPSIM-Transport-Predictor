"""Input normalization implemented as POPSIM modules.

Each normalizer maps the 7 physical power-balance inputs
(Ip_MA, B0, R0, a_minor, kappa, ne20_line_avg, P_aux_MW) to 7 NN-ready
features. Dimensionality is maintained so no new information is provided.
Stats-bearing methods (z_score, coral) are fitted from TRAINING data only, at
model_init time, and store their statistics as frozen buffers on the module.
The arrays live in the pytree (so they checkpoint and restore with the model,
which is what carries source-fitted stats through transfer learning), but the
trainable selectors never include them so they are never updated by the
optimizer.

Methods:
- raw: identity, features are the physical values
- physics: dimensionless / device-invariant combinations computed in-graph
- z_score: per-device zero mean and unit variance
- coral: per-device covariance alignment to the pooled training covariance
  (https://arxiv.org/abs/1612.01939)
- physics-coral: the physics transform followed by CORAL alignment fitted in
  the dimensionless physics feature space

organize_data.normalize_domain (data visualization) is a thin wrapper around
the fit/apply helpers in this file, so the visualized feature spaces are the
ones the models actually consume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from loguru import logger

if TYPE_CHECKING:
    from jaxtyping import ArrayLike
from popsim.cfspopcon_jax.current_drive import calc_f_shaping, calc_q_star
from popsim.cfspopcon_jax.geometry import calc_plasma_surface_area
from popsim.module_base import TimeIndepModule
from scipy.linalg import fractional_matrix_power

NORM_INPUT_VARS = (
    "Ip_MA",
    "B0",
    "R0",
    "a_minor",
    "kappa",
    "ne20_line_avg",
    "P_aux_MW",
)
N_FEATURES = len(NORM_INPUT_VARS)

# Regularization for covariance matrix roots
CORAL_REG = 1e-6
# A device needs at least this many complete samples for a meaningful
# covariance, below it the device keeps the identity transform
MIN_CORAL_SAMPLES = 8

# Methods whose per-device statistics are fitted from data
# Their transfer cases pretrain through a dedicated transfer_pretrain prereq
# case whose stats are fitted on the combined historic + target data and
# inherited via the checkpoint restore (see Study.Case.transfer_pretrain_case)
STAT_NORMALIZATIONS = ("z_score", "coral", "physics-coral")


class InputNormalizer(TimeIndepModule):
    """Base class mapping the 7 physical inputs to 7 NN features.

    ds_source_idx selects the per-device statistics for stats-bearing methods
    (the global device index from config.ds_source_to_idx), raw and physics
    ignore it.
    """

    @chex.dataclass
    class Inputs:
        Ip_MA: float
        B0: float
        R0: float
        a_minor: float
        kappa: float
        ne20_line_avg: float
        P_aux_MW: float
        ds_source_idx: float

    @chex.dataclass
    class Output:
        Ip_MA: float
        B0: float
        R0: float
        a_minor: float
        kappa: float
        ne20_line_avg: float
        P_aux_MW: float

        def to_vec(self) -> jnp.ndarray:
            return jnp.stack(
                [
                    self.Ip_MA,
                    self.B0,
                    self.R0,
                    self.a_minor,
                    self.kappa,
                    self.ne20_line_avg,
                    self.P_aux_MW,
                ]
            )

    @classmethod
    def inputs_from_dict(cls, data: dict[str, ArrayLike] | xr.Dataset) -> InputNormalizer.Inputs:
        """Build Inputs from dataset variables keyed by their NORM_INPUT_VARS names."""
        if isinstance(data, xr.Dataset):
            data = {var: data[var].data for var in data.data_vars}
        return cls.Inputs(
            Ip_MA=data["Ip_MA"],
            B0=data["B0"],
            R0=data["R0"],
            a_minor=data["a_minor"],
            kappa=data["kappa"],
            ne20_line_avg=data["ne20_line_avg"],
            P_aux_MW=data["P_aux_MW"],
            ds_source_idx=data["ds_source_idx"],
        )

    def _normalize_vec(self, vec: jnp.ndarray, ds_source_idx: ArrayLike) -> jnp.ndarray:
        raise NotImplementedError

    def __call__(self, inputs: Inputs) -> Output:
        vec = jnp.stack(
            [
                inputs.Ip_MA,
                inputs.B0,
                inputs.R0,
                inputs.a_minor,
                inputs.kappa,
                inputs.ne20_line_avg,
                inputs.P_aux_MW,
            ]
        )
        out = self._normalize_vec(vec, inputs.ds_source_idx)
        return self.Output(
            Ip_MA=out[0],
            B0=out[1],
            R0=out[2],
            a_minor=out[3],
            kappa=out[4],
            ne20_line_avg=out[5],
            P_aux_MW=out[6],
        )


class RawNormalizer(InputNormalizer):
    """Identity, features are the raw physical values."""

    def _normalize_vec(self, vec: jnp.ndarray, ds_source_idx: ArrayLike) -> jnp.ndarray:
        return vec


# Names of the physics_feature_vec output slots, in order
PHYSICS_FEATURE_NAMES = (
    "Ip_MA",
    "q_star",
    "epsilon",
    "aB0",
    "kappa",
    "f_G",
    "surface_power_density",
)


def physics_feature_vec(vec: jnp.ndarray) -> jnp.ndarray:
    """The dimensionless physics features for a stacked 7-input vector.

    Slot mapping (slot name -> feature):
    - Ip_MA          -> Ip_MA (kept raw, sufficiently device-invariant)
    - B0             -> q_star (zero triangularity, consistent with H89/H98)
    - R0             -> epsilon = a_minor / R0
    - a_minor        -> aB0 = a_minor * B0 (dimensional, but the dimensionless
                        alternatives like normalized gyroradius need a temperature,
                        which the scaling-law baselines do not have. A fair
                        comparison keeps the same information budget)
    - kappa          -> kappa
    - ne20_line_avg  -> Greenwald fraction f_G
    - P_aux_MW       -> P_aux / plasma surface area
    """
    ip_ma, b0, r0, a_minor, kappa, ne20, p_aux = vec
    epsilon = a_minor / r0
    f_shaping = calc_f_shaping(epsilon, kappa, jnp.zeros_like(epsilon))
    q_star = calc_q_star(b0, r0, epsilon, ip_ma, f_shaping)
    greenwald_limit = ip_ma / (jnp.pi * a_minor**2)
    f_g = ne20 / greenwald_limit
    a_b0 = a_minor * b0
    surface_area = calc_plasma_surface_area(r0, epsilon, kappa)
    surface_power_density = p_aux / surface_area
    return jnp.stack([ip_ma, q_star, epsilon, a_b0, kappa, f_g, surface_power_density])


class PhysicsNormalizer(InputNormalizer):
    """Dimensionless / device-invariant features computed in-graph.

    Stateless, nothing is fitted. See physics_feature_vec for the slot
    mapping.
    """

    def _normalize_vec(self, vec: jnp.ndarray, ds_source_idx: ArrayLike) -> jnp.ndarray:
        return physics_feature_vec(vec)


def fit_z_score_stats(features: np.ndarray, source_idx: np.ndarray, n_devices: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-device mean/std from an (N, F) feature matrix.

    Per-feature statistics ignore NaNs independently (z-scoring is
    elementwise, so incomplete rows still contribute their valid entries).
    Devices absent from the data keep the identity row (mean 0, std 1), as do
    features with zero or undefined spread.
    """
    n_features = features.shape[1]
    means = np.zeros((n_devices, n_features))
    stds = np.ones((n_devices, n_features))
    for device_val in np.unique(source_idx):
        device = int(device_val)
        rows = features[source_idx == device]
        with np.errstate(all="ignore"):
            mean = np.nanmean(rows, axis=0)
            std = np.nanstd(rows, axis=0)
        means[device] = np.where(np.isfinite(mean), mean, 0.0)
        stds[device] = np.where(np.isfinite(std) & (std != 0), std, 1.0)
    return jnp.asarray(means), jnp.asarray(stds)


def apply_z_score(vec: jnp.ndarray, ds_source_idx: ArrayLike, means: jnp.ndarray, stds: jnp.ndarray) -> jnp.ndarray:
    """Apply the device's z-score to a feature vector (batched inputs broadcast)."""
    idx = jnp.asarray(ds_source_idx).astype(jnp.int32)
    mean = jnp.take(means, idx, axis=0)
    std = jnp.take(stds, idx, axis=0)
    return (vec - mean) / std


class ZScoreNormalizer(InputNormalizer):
    """Per-device zero mean, unit variance.

    means/stds have shape (n_devices, 7), row order follows the global
    config.ds_source_to_idx. Devices absent from the fitting data keep the
    identity row (mean 0, std 1) so their features pass through raw. Sizing by
    the global device registry keeps the pytree structure identical across
    cases, so checkpoints restore cleanly regardless of which devices a case
    was trained on.
    """

    means: jnp.ndarray
    stds: jnp.ndarray

    def _normalize_vec(self, vec: jnp.ndarray, ds_source_idx: ArrayLike) -> jnp.ndarray:
        return apply_z_score(vec, ds_source_idx, self.means, self.stds)

    @classmethod
    def identity(cls, n_devices: int) -> ZScoreNormalizer:
        return cls(means=jnp.zeros((n_devices, N_FEATURES)), stds=jnp.ones((n_devices, N_FEATURES)))

    @classmethod
    def fit(cls, ds: xr.Dataset, n_devices: int) -> ZScoreNormalizer:
        """Fit per-device mean/std over exactly the 7 input vars."""
        features, source_idx = _feature_matrix(ds)
        means, stds = fit_z_score_stats(features, source_idx, n_devices)
        return cls(means=means, stds=stds)


def identity_coral_stats(n_devices: int, n_features: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Identity CORAL statistics: zero means, identity transforms."""
    return (
        jnp.zeros((n_devices, n_features)),
        jnp.tile(jnp.eye(n_features), (n_devices, 1, 1)),
    )


def fit_coral_stats(features: np.ndarray, source_idx: np.ndarray, n_devices: int) -> tuple[jnp.ndarray, jnp.ndarray] | None:
    """Per-device CORAL statistics from an (N, F) feature matrix.

    For device d the transform is: center with the device mean, whiten with
    C_d^{-1/2}, re-color with C_ref^{1/2} (C_ref the pooled covariance over
    all fitting data), then re-add the device mean. Only rows complete in all
    F features contribute (covariances need complete rows). Devices with fewer
    than MIN_CORAL_SAMPLES complete rows keep the identity transform. Returns
    None when the whole pooled set is below MIN_CORAL_SAMPLES.
    """
    n_features = features.shape[1]
    valid = ~np.any(np.isnan(features), axis=1)
    pooled = features[valid]
    if len(pooled) < MIN_CORAL_SAMPLES:
        logger.warning(f"CORAL fit got only {len(pooled)} complete samples, using identity transforms")
        return None
    cov_ref = np.cov(pooled, rowvar=False)
    cr_pos_half = np.real(fractional_matrix_power(cov_ref + CORAL_REG * np.eye(n_features), 0.5))

    means = np.zeros((n_devices, n_features))
    transforms = np.tile(np.eye(n_features), (n_devices, 1, 1))
    for device_val in np.unique(source_idx[valid]):
        device = int(device_val)
        rows = features[valid & (source_idx == device)]
        if len(rows) < MIN_CORAL_SAMPLES:
            continue
        cov_device = np.cov(rows, rowvar=False)
        cd_neg_half = np.real(fractional_matrix_power(cov_device + CORAL_REG * np.eye(n_features), -0.5))
        means[device] = np.mean(rows, axis=0)
        transforms[device] = cd_neg_half @ cr_pos_half
    return jnp.asarray(means), jnp.asarray(transforms)


def apply_coral(vec: jnp.ndarray, ds_source_idx: ArrayLike, means: jnp.ndarray, transforms: jnp.ndarray) -> jnp.ndarray:
    """Apply the device's CORAL transform to a feature vector."""
    idx = jnp.asarray(ds_source_idx).astype(jnp.int32)
    mean = jnp.take(means, idx, axis=0)
    transform = jnp.take(transforms, idx, axis=0)
    return (vec - mean) @ transform + mean


class CoralNormalizer(InputNormalizer):
    """Per-device CORAL alignment to the pooled training covariance.

    transforms has shape (n_devices, 7, 7) and means (n_devices, 7), rows
    follow the global config.ds_source_to_idx. Unfitted devices keep the
    identity transform. See fit_coral_stats for the math.
    """

    means: jnp.ndarray
    transforms: jnp.ndarray

    def _normalize_vec(self, vec: jnp.ndarray, ds_source_idx: ArrayLike) -> jnp.ndarray:
        return apply_coral(vec, ds_source_idx, self.means, self.transforms)

    @classmethod
    def identity(cls, n_devices: int) -> CoralNormalizer:
        means, transforms = identity_coral_stats(n_devices, N_FEATURES)
        return cls(means=means, transforms=transforms)

    @classmethod
    def fit(cls, ds: xr.Dataset, n_devices: int) -> CoralNormalizer:
        """Fit per-device CORAL transforms over exactly the 7 input vars."""
        features, source_idx = _feature_matrix(ds)
        stats = fit_coral_stats(features, source_idx, n_devices)
        if stats is None:
            return cls.identity(n_devices)
        means, transforms = stats
        return cls(means=means, transforms=transforms)


class PhysicsCoralNormalizer(InputNormalizer):
    """The physics transform followed by CORAL alignment in physics space.

    The CORAL statistics are fitted on the physics-transformed training
    features, so the alignment corrects the per-device distribution of the
    dimensionless parameters rather than the raw inputs.
    Buffer shapes match CoralNormalizer.
    """

    means: jnp.ndarray
    transforms: jnp.ndarray

    def _normalize_vec(self, vec: jnp.ndarray, ds_source_idx: ArrayLike) -> jnp.ndarray:
        phys = physics_feature_vec(vec)
        return apply_coral(phys, ds_source_idx, self.means, self.transforms)

    @classmethod
    def identity(cls, n_devices: int) -> PhysicsCoralNormalizer:
        means, transforms = identity_coral_stats(n_devices, N_FEATURES)
        return cls(means=means, transforms=transforms)

    @classmethod
    def fit(cls, ds: xr.Dataset, n_devices: int) -> PhysicsCoralNormalizer:
        """Fit per-device CORAL transforms in the physics feature space."""
        features, source_idx = _feature_matrix(ds)
        phys_rows = np.asarray(jax.vmap(physics_feature_vec)(jnp.asarray(features)))
        stats = fit_coral_stats(phys_rows, source_idx, n_devices)
        if stats is None:
            return cls.identity(n_devices)
        means, transforms = stats
        return cls(means=means, transforms=transforms)


class CoralFeatureNormalizer(TimeIndepModule):
    """Generic per-device CORAL stage over an arbitrary feature vector.

    Used by the profile predictor on its 10 dimensionless nn_inputs. The
    statistics are frozen buffers exactly like the InputNormalizer methods:
    they checkpoint with the model and are never in a trainable selection.
    """

    means: jnp.ndarray
    transforms: jnp.ndarray

    def __call__(self, vec: jnp.ndarray, ds_source_idx: ArrayLike) -> jnp.ndarray:
        return apply_coral(vec, ds_source_idx, self.means, self.transforms)

    @classmethod
    def identity(cls, n_devices: int, n_features: int) -> CoralFeatureNormalizer:
        means, transforms = identity_coral_stats(n_devices, n_features)
        return cls(means=means, transforms=transforms)

    @classmethod
    def fit_from_features(cls, features: np.ndarray, source_idx: np.ndarray, n_devices: int) -> CoralFeatureNormalizer:
        stats = fit_coral_stats(features, source_idx, n_devices)
        if stats is None:
            return cls.identity(n_devices, features.shape[1])
        means, transforms = stats
        return cls(means=means, transforms=transforms)


def _feature_matrix(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the 7 input vars to an (N, 7) matrix plus the matching (N,) device index."""
    reference = ds[NORM_INPUT_VARS[0]]
    columns = [np.asarray(ds[var].broadcast_like(reference).values, dtype=float).ravel() for var in NORM_INPUT_VARS]
    source_idx = np.asarray(ds["ds_source_idx"].broadcast_like(reference).values, dtype=float).ravel()
    # NaN device indices (from NaN-padded concatenation) can't be attributed to a device
    features = np.column_stack(columns)
    attributed = ~np.isnan(source_idx)
    return features[attributed], source_idx[attributed].astype(int)


def make_normalizer(
    method: str,
    train_ds: xr.Dataset | None,
    n_devices: int,
) -> InputNormalizer:
    """Build the normalizer for a case.

    train_ds is required for the stats-bearing methods unless the caller is
    about to overwrite the module from a checkpoint (transfer restore), then
    passing None yields identity stats with the correct pytree structure.
    """
    if method == "raw":
        return RawNormalizer()
    if method == "physics":
        return PhysicsNormalizer()
    if method == "z_score":
        return ZScoreNormalizer.identity(n_devices) if train_ds is None else ZScoreNormalizer.fit(train_ds, n_devices)
    if method == "coral":
        return CoralNormalizer.identity(n_devices) if train_ds is None else CoralNormalizer.fit(train_ds, n_devices)
    if method == "physics-coral":
        return PhysicsCoralNormalizer.identity(n_devices) if train_ds is None else PhysicsCoralNormalizer.fit(train_ds, n_devices)
    raise ValueError(f"Unknown normalization method: {method}")
