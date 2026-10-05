"""Input normalization implemented as POPSIM modules.

Each normalizer maps the 7 physical power-balance inputs (NORM_INPUT_VARS) to 7 NN-ready features.
Dimensionality is maintained so no new information is provided.
Stats-bearing methods (zscore, coral, and their physics- variants) are fitted
from TRAINING data only, at model_init time, and store their statistics as
frozen buffers on the module.
The arrays live in the pytree (so they checkpoint and restore with the model,
which is what carries source-fitted stats through transfer learning), but the
trainable selectors never include them so they are never updated by the
optimizer.

Methods:
- raw: identity, features are the physical values
- physics: dimensionless / device-invariant combinations computed in-graph
- zscore: per-device zero mean and unit variance
- coral: per-device covariance alignment to the target device covariance
  (https://arxiv.org/abs/1612.01939)
- physics-coral: the physics transform followed by CORAL alignment fitted in
  the dimensionless physics feature space
- physics-zscore: the physics transform followed by a per-device z-score
  fitted in the dimensionless physics feature space

The power balance mlp also normalizes its own state, the stored energy, as an 8th slot
(normalize_with_energy, a normalizer built with_energy),
and energy_rate_scale maps its network output back to dW/dt.

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
from popsim.cfspopcon_jax.geometry import calc_plasma_surface_area
from popsim.module_base import TimeIndepModule

from transport_study.modules import plasma_parameters

NORM_INPUT_VARS = (
    "ip_MA",
    "b_geo",
    "geometric_axis_r",
    "minor_radius",
    "elongation",
    "n_e_line_average_1e20",
    "power_additional_MW",
)
N_FEATURES = len(NORM_INPUT_VARS)
# The stored energy, the 8th slot of a normalizer built with_energy
ENERGY_VAR = "energy_mhd_MJ"
# Time unit [s] of the dW/dt output of the methods that normalize the stored energy,
# ~tau_E on C-Mod and MAST so the network output sits near 1
TAU_REF_S = 0.03

# A feature whose within-device spread is below this fraction of the target spread
# is treated as constant for that device.
# Whitening it would amplify measurement noise by more than 1/frac, so it keeps the identity slot.
CORAL_DEGENERATE_STD_FRAC = 0.05
# Eigenvalue floor for the device covariance in target-std units.
# Guards non-axis-aligned degeneracy (collinear features) the same way,
# capping the whitening gain of any direction at 1/CORAL_DEGENERATE_STD_FRAC.
CORAL_EIGVAL_FLOOR = CORAL_DEGENERATE_STD_FRAC**2
# A source device needs at least this many distinct shots for a meaningful covariance,
# below it the device keeps the identity transform.
# The samples of one shot are strongly correlated, so the gate counts shots, not samples.
# The target is exempt: its few-shot statistics are poor but still the alignment reference.
MIN_CORAL_SHOTS = 8

# Methods whose per-device statistics are fitted from data
# Their transfer cases pretrain through a dedicated transfer_pretrain prereq
# case whose stats are fitted on the combined historic + target data and
# inherited via the checkpoint restore (see Study.Case.transfer_pretrain_case)
# physics-zscore doubles as the profile/transport feature-stage method (a
# z-score over the predictor's dimensionless nn_inputs, see
# ZScoreFeatureNormalizer) and as the power-balance InputNormalizer
# method PhysicsZScoreNormalizer
STAT_NORMALIZATIONS = ("zscore", "coral", "physics-coral", "physics-zscore")


class InputNormalizer(TimeIndepModule):
    """Base class mapping the 7 physical inputs to 7 NN features.

    ds_source_idx selects the per-device statistics for stats-bearing methods
    (the global device index from config.ds_source_to_idx), raw and physics
    ignore it.
    """

    @chex.dataclass
    class Inputs:
        ip_MA: float
        b_geo: float
        geometric_axis_r: float
        minor_radius: float
        elongation: float
        n_e_line_average_1e20: float
        power_additional_MW: float
        ds_source_idx: float

    @chex.dataclass
    class Output:
        ip_MA: float
        b_geo: float
        geometric_axis_r: float
        minor_radius: float
        elongation: float
        n_e_line_average_1e20: float
        power_additional_MW: float

        def to_vec(self) -> jnp.ndarray:
            return jnp.stack(
                [
                    self.ip_MA,
                    self.b_geo,
                    self.geometric_axis_r,
                    self.minor_radius,
                    self.elongation,
                    self.n_e_line_average_1e20,
                    self.power_additional_MW,
                ]
            )

    @staticmethod
    def input_vec(inputs: InputNormalizer.Inputs) -> jnp.ndarray:
        """The 7 physical inputs stacked in NORM_INPUT_VARS order."""
        return jnp.stack(
            [
                inputs.ip_MA,
                inputs.b_geo,
                inputs.geometric_axis_r,
                inputs.minor_radius,
                inputs.elongation,
                inputs.n_e_line_average_1e20,
                inputs.power_additional_MW,
            ]
        )

    @classmethod
    def inputs_from_dict(cls, data: dict[str, ArrayLike] | xr.Dataset) -> InputNormalizer.Inputs:
        """Build Inputs from dataset variables keyed by their NORM_INPUT_VARS names."""
        if isinstance(data, xr.Dataset):
            data = {var: data[var].data for var in data.data_vars}
        return cls.Inputs(
            ip_MA=data["ip_MA"],
            b_geo=data["b_geo"],
            geometric_axis_r=data["geometric_axis_r"],
            minor_radius=data["minor_radius"],
            elongation=data["elongation"],
            n_e_line_average_1e20=data["n_e_line_average_1e20"],
            power_additional_MW=data["power_additional_MW"],
            ds_source_idx=data["ds_source_idx"],
        )

    def _normalize_vec(self, vec: jnp.ndarray, ds_source_idx: ArrayLike) -> jnp.ndarray:
        raise NotImplementedError

    def normalize_with_energy(self, inputs: Inputs, energy_mhd_MJ: ArrayLike) -> jnp.ndarray:
        """The 7 normalized inputs followed by the normalized stored energy, from a normalizer built with_energy."""
        input_vec = self.input_vec(inputs)
        vec = jnp.append(input_vec, energy_mhd_MJ)
        return self._normalize_vec(vec, inputs.ds_source_idx)

    def energy_rate_scale(self, inputs: Inputs) -> ArrayLike:
        """dW/dt [MW] per unit of a network output predicting it.

        Methods that keep the stored energy in MJ (raw, coral) take the output in MW.
        The others take it in their normalized energy units per TAU_REF_S.
        """
        return 1.0

    def __call__(self, inputs: Inputs) -> Output:
        vec = self.input_vec(inputs)
        out = self._normalize_vec(vec, inputs.ds_source_idx)
        return self.Output(
            ip_MA=out[0],
            b_geo=out[1],
            geometric_axis_r=out[2],
            minor_radius=out[3],
            elongation=out[4],
            n_e_line_average_1e20=out[5],
            power_additional_MW=out[6],
        )


class RawNormalizer(InputNormalizer):
    """Identity, features are the raw physical values."""

    def _normalize_vec(self, vec: jnp.ndarray, ds_source_idx: ArrayLike) -> jnp.ndarray:
        return vec


# Names of the physics_feature_vec output slots, in order
PHYSICS_FEATURE_NAMES = (
    "ip_MA",
    "q_star",
    "epsilon",
    "aB0",
    "elongation",
    "f_G",
    "surface_power_density",
)


def energy_at_unit_beta_n_MJ(ip_MA, b_geo, geometric_axis_r, minor_radius, elongation):
    """Stored energy [MJ] at beta_N = 1, the energy unit of the physics methods.

    b_geo stands in for b0, the power balance inputs carry no b0.
    """
    volume_m3 = plasma_parameters.volume_approx(geometric_axis_r, minor_radius, elongation)
    return plasma_parameters.energy_mhd_MJ_from_beta_tor_norm(1.0, volume_m3, minor_radius, b_geo, ip_MA)


def physics_energy_unit_MJ(inputs: InputNormalizer.Inputs) -> ArrayLike:
    """energy_at_unit_beta_n_MJ at the physical inputs."""
    return energy_at_unit_beta_n_MJ(inputs.ip_MA, inputs.b_geo, inputs.geometric_axis_r, inputs.minor_radius, inputs.elongation)


def physics_energy_rate_scale(inputs: InputNormalizer.Inputs) -> ArrayLike:
    """dW/dt [MW] per unit rate of the beta_N slot, one beta_N of stored energy per TAU_REF_S."""
    energy_unit_MJ = physics_energy_unit_MJ(inputs)
    return energy_unit_MJ / TAU_REF_S


def physics_feature_vec(vec: jnp.ndarray) -> jnp.ndarray:
    """The dimensionless physics features for a stacked 7-input vector, or 8 with the stored energy.

    Slot mapping (slot name -> feature):
    - ip_MA                  -> ip_MA (kept raw, sufficiently device-invariant)
    - b_geo                  -> q_star (zero triangularity, consistent with H89/H98)
    - geometric_axis_r       -> epsilon = minor_radius / geometric_axis_r
    - minor_radius           -> aB0 = minor_radius * b_geo (dimensional, but the dimensionless
                                alternatives like normalized gyroradius need a temperature,
                                which the scaling-law baselines do not have.
                                A fair comparison keeps the same information budget)
    - elongation             -> elongation
    - n_e_line_average_1e20  -> Greenwald fraction f_G
    - power_additional_MW    -> P_aux / plasma surface area
    - energy_mhd_MJ (8th)    -> beta_N, the stored energy over energy_at_unit_beta_n_MJ
    """
    ip_ma, b_geo, r_geo, a_minor, kappa, ne20, p_aux = vec[:N_FEATURES]
    epsilon = plasma_parameters.inverse_aspect_ratio(a_minor, r_geo)
    no_triangularity = jnp.zeros_like(epsilon)
    q_star = plasma_parameters.q_star(ip_ma, b_geo, r_geo, a_minor, kappa, no_triangularity, no_triangularity)
    f_g = plasma_parameters.greenwald_fraction(ne20, ip_ma, a_minor)
    a_b0 = plasma_parameters.a_b0(a_minor, b_geo)
    surface_area = calc_plasma_surface_area(r_geo, epsilon, kappa)
    surface_power_density = p_aux / surface_area
    features = [ip_ma, q_star, epsilon, a_b0, kappa, f_g, surface_power_density]
    if vec.shape[0] > N_FEATURES:
        energy_unit_MJ = energy_at_unit_beta_n_MJ(ip_ma, b_geo, r_geo, a_minor, kappa)
        features.append(vec[N_FEATURES] / energy_unit_MJ)
    return jnp.stack(features)


class PhysicsNormalizer(InputNormalizer):
    """Dimensionless / device-invariant features computed in-graph.

    Stateless, nothing is fitted. See physics_feature_vec for the slot
    mapping.
    """

    def _normalize_vec(self, vec: jnp.ndarray, ds_source_idx: ArrayLike) -> jnp.ndarray:
        return physics_feature_vec(vec)

    def energy_rate_scale(self, inputs: InputNormalizer.Inputs) -> ArrayLike:
        return physics_energy_rate_scale(inputs)


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
    """Per-device zero mean, unit variance of the 7 inputs.

    means/stds have shape (n_devices, 7), row order follows the global
    config.ds_source_to_idx. Devices absent from the fitting data keep the
    identity row (mean 0, std 1) so their features pass through raw. Sizing by
    the global device registry keeps the pytree structure identical across
    cases, so checkpoints restore cleanly regardless of which devices a case
    was trained on.
    """

    means: jnp.ndarray
    stds: jnp.ndarray

    @staticmethod
    def features(vec: jnp.ndarray) -> jnp.ndarray:
        """The feature vector the statistics standardize, the inputs themselves."""
        return vec

    @staticmethod
    def energy_unit_MJ(inputs: InputNormalizer.Inputs) -> ArrayLike:
        """MJ per unit of the energy slot before standardization."""
        return 1.0

    def _normalize_vec(self, vec: jnp.ndarray, ds_source_idx: ArrayLike) -> jnp.ndarray:
        return apply_z_score(self.features(vec), ds_source_idx, self.means, self.stds)

    def energy_rate_scale(self, inputs: InputNormalizer.Inputs) -> ArrayLike:
        """One device std of the energy slot per TAU_REF_S."""
        if self.stds.shape[1] != N_FEATURES + 1:
            raise ValueError("energy_rate_scale needs a normalizer built with_energy")
        idx = jnp.asarray(inputs.ds_source_idx).astype(jnp.int32)
        energy_std = jnp.take(self.stds[:, N_FEATURES], idx)
        energy_unit_MJ = self.energy_unit_MJ(inputs)
        return energy_std * energy_unit_MJ / TAU_REF_S

    @classmethod
    def identity(cls, n_devices: int, n_features: int = N_FEATURES) -> ZScoreNormalizer:
        return cls(means=jnp.zeros((n_devices, n_features)), stds=jnp.ones((n_devices, n_features)))

    @classmethod
    def fit(cls, ds: xr.Dataset, n_devices: int, variables: tuple[str, ...] = NORM_INPUT_VARS) -> ZScoreNormalizer:
        """Fit per-device mean/std of the features of variables over the training rows."""
        inputs, source_idx, _ = _feature_matrix(ds, variables)
        means, stds = fit_z_score_stats(_feature_rows(cls.features, inputs), source_idx, n_devices)
        return cls(means=means, stds=stds)


class PhysicsZScoreNormalizer(ZScoreNormalizer):
    """The physics transform followed by a per-device z-score in physics space.

    The standardization removes the per-device offset and scale of the dimensionless parameters
    rather than of the raw inputs. Features come out centered, z-scoring does not re-add the device mean the way CORAL does.
    """

    features = staticmethod(physics_feature_vec)
    energy_unit_MJ = staticmethod(physics_energy_unit_MJ)


def identity_coral_stats(n_devices: int, n_features: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Identity CORAL statistics: zero means, identity transforms."""
    return (
        jnp.zeros((n_devices, n_features)),
        jnp.tile(jnp.eye(n_features), (n_devices, 1, 1)),
    )


def _sym_matrix_power(mat: np.ndarray, power: float, eigval_floor: float) -> np.ndarray:
    """Symmetric matrix power via eigh with an eigenvalue floor.

    The floor bounds the gain of negative powers on near-singular matrices
    (and clips roundoff-negative eigenvalues for positive powers).
    """
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.maximum(eigvals, eigval_floor)
    return (eigvecs * eigvals**power) @ eigvecs.T


def fit_coral_stats(
    features: np.ndarray, source_idx: np.ndarray, n_devices: int, shot_idx: np.ndarray, target_idx: int
) -> tuple[jnp.ndarray, jnp.ndarray] | None:
    """Per-device CORAL statistics from an (N, F) feature matrix, aligned to the target device.

    For device d the transform is: center with the device mean, whiten with C_d^{-1/2},
    re-color with C_t^{1/2} (C_t the target device covariance), then re-add the device mean.
    CORAL aligns second moments only, so every device keeps its own mean.
    Only rows complete in all F features contribute (covariances need complete rows).

    The target keeps the identity transform, it is the reference.
    Any number of complete target rows is accepted.
    Other devices need MIN_CORAL_SHOTS distinct shots, below that they keep the identity transform.
    Returns None when the target has no complete rows or no spread in any feature,
    every device then keeps the identity transform.

    Guards, all in target-std units:
    - a feature without target spread has no covariance to align to, it keeps the identity slot on every device
    - a feature whose within-device std is below CORAL_DEGENERATE_STD_FRAC of the target std
      keeps the identity slot on that device (whitening it would amplify noise by more than 1/frac)
    - CORAL_EIGVAL_FLOOR bounds the whitening gain of any remaining near-degenerate direction
    """
    n_features = features.shape[1]
    valid = ~np.any(np.isnan(features), axis=1)
    target_rows = features[valid & (source_idx == target_idx)]
    target_std = np.std(target_rows, axis=0) if len(target_rows) else np.zeros(n_features)
    mask_target_spread = target_std > 0
    if not mask_target_spread.any():
        logger.warning(f"CORAL fit has no target device spread ({len(target_rows)} complete target rows), using identity transforms")
        return None
    scale = np.where(mask_target_spread, target_std, 1.0)
    cov_target = np.atleast_2d(np.cov(target_rows / scale, rowvar=False))

    means = np.zeros((n_devices, n_features))
    transforms = np.tile(np.eye(n_features), (n_devices, 1, 1))
    for device_val in np.unique(source_idx[valid]):
        device = int(device_val)
        if device == target_idx:
            continue
        mask_device_rows = valid & (source_idx == device)
        rows = features[mask_device_rows]
        n_device_shots = len(np.unique(shot_idx[mask_device_rows]))
        if n_device_shots < MIN_CORAL_SHOTS:
            logger.warning(f"CORAL fit for device {device} got only {n_device_shots} complete shots, keeping identity")
            continue
        means[device] = np.mean(rows, axis=0)
        live = mask_target_spread & (np.std(rows, axis=0) > CORAL_DEGENERATE_STD_FRAC * target_std)
        if not live.any():
            logger.warning(f"CORAL fit for device {device} found no non-degenerate features, keeping identity")
            continue
        cov_device = np.atleast_2d(np.cov(rows[:, live] / scale[live], rowvar=False))
        cd_neg_half = _sym_matrix_power(cov_device, -0.5, eigval_floor=CORAL_EIGVAL_FLOOR)
        ct_pos_half = _sym_matrix_power(np.atleast_2d(cov_target[np.ix_(live, live)]), 0.5, eigval_floor=0.0)
        # Compose back to raw units: the transform rows and columns carry the per-feature scale,
        # so apply_coral stays a single (vec - mean) @ T + mean.
        # The identity slots keep the initial identity rows and columns
        block = (cd_neg_half @ ct_pos_half) * scale[live][None, :] / scale[live][:, None]
        transforms[device][np.ix_(live, live)] = block
    return jnp.asarray(means), jnp.asarray(transforms)


def apply_coral(vec: jnp.ndarray, ds_source_idx: ArrayLike, means: jnp.ndarray, transforms: jnp.ndarray) -> jnp.ndarray:
    """Apply the device's CORAL transform to a feature vector."""
    idx = jnp.asarray(ds_source_idx).astype(jnp.int32)
    mean = jnp.take(means, idx, axis=0)
    transform = jnp.take(transforms, idx, axis=0)
    return (vec - mean) @ transform + mean


class CoralNormalizer(InputNormalizer):
    """Per-device CORAL alignment of the 7 inputs to the target device covariance.

    transforms has shape (n_devices, 7, 7) and means (n_devices, 7), rows
    follow the global config.ds_source_to_idx. Unfitted devices keep the
    identity transform. See fit_coral_stats for the math.
    """

    means: jnp.ndarray
    transforms: jnp.ndarray

    @staticmethod
    def features(vec: jnp.ndarray) -> jnp.ndarray:
        """The feature vector the transforms align, the 7 inputs themselves."""
        return vec

    def _normalize_vec(self, vec: jnp.ndarray, ds_source_idx: ArrayLike) -> jnp.ndarray:
        return apply_coral(self.features(vec), ds_source_idx, self.means, self.transforms)

    @classmethod
    def identity(cls, n_devices: int, n_features: int = N_FEATURES) -> CoralNormalizer:
        means, transforms = identity_coral_stats(n_devices, n_features)
        return cls(means=means, transforms=transforms)

    @classmethod
    def fit(cls, ds: xr.Dataset, n_devices: int, target_idx: int, variables: tuple[str, ...] = NORM_INPUT_VARS) -> CoralNormalizer:
        """Fit per-device CORAL transforms of the features of variables over the training rows."""
        inputs, source_idx, shot_idx = _feature_matrix(ds, variables)
        stats = fit_coral_stats(_feature_rows(cls.features, inputs), source_idx, n_devices, shot_idx, target_idx)
        if stats is None:
            return cls.identity(n_devices, len(variables))
        means, transforms = stats
        return cls(means=means, transforms=transforms)


class PhysicsCoralNormalizer(CoralNormalizer):
    """The physics transform followed by CORAL alignment in physics space.

    The alignment corrects the per-device distribution of the dimensionless parameters rather than of the raw inputs.
    """

    features = staticmethod(physics_feature_vec)

    def energy_rate_scale(self, inputs: InputNormalizer.Inputs) -> ArrayLike:
        return physics_energy_rate_scale(inputs)


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
    def fit_from_features(
        cls, features: np.ndarray, source_idx: np.ndarray, n_devices: int, shot_idx: np.ndarray, target_idx: int
    ) -> CoralFeatureNormalizer:
        stats = fit_coral_stats(features, source_idx, n_devices, shot_idx, target_idx)
        if stats is None:
            return cls.identity(n_devices, features.shape[1])
        means, transforms = stats
        return cls(means=means, transforms=transforms)


class ZScoreFeatureNormalizer(TimeIndepModule):
    """Generic per-device z-score stage over an arbitrary feature vector.

    The z-score counterpart of CoralFeatureNormalizer, used by the profile
    predictor under the physics-zscore method: per-feature mean/std only, no
    covariance alignment, and features come out centered (z-scoring does not
    re-add the device mean the way CORAL does). Same frozen-buffer semantics:
    the stats checkpoint with the model and are never in a trainable
    selection.
    """

    means: jnp.ndarray
    stds: jnp.ndarray

    def __call__(self, vec: jnp.ndarray, ds_source_idx: ArrayLike) -> jnp.ndarray:
        return apply_z_score(vec, ds_source_idx, self.means, self.stds)

    @classmethod
    def identity(cls, n_devices: int, n_features: int) -> ZScoreFeatureNormalizer:
        return cls(means=jnp.zeros((n_devices, n_features)), stds=jnp.ones((n_devices, n_features)))

    @classmethod
    def fit_from_features(cls, features: np.ndarray, source_idx: np.ndarray, n_devices: int) -> ZScoreFeatureNormalizer:
        means, stds = fit_z_score_stats(features, source_idx, n_devices)
        return cls(means=means, stds=stds)


# Either per-device stat stage a profile or transport predictor module can
# hold. The two classes have different pytree structures, so a study must keep
# one method for its whole lifetime (data_normalization is config-lock guarded)
FeatureNormalizer = CoralFeatureNormalizer | ZScoreFeatureNormalizer

# Methods available to the studies whose models normalize their own
# dimensionless feature vector rather than the 7 physical inputs
FEATURE_NORMALIZATIONS = ("physics", "physics-coral", "physics-zscore")


def flat_columns(ds: xr.Dataset, variables: tuple[str, ...] | list[str]) -> np.ndarray:
    """(N, len(variables)) float matrix of dataset variables flattened in step.

    Every variable is broadcast against ip_MA first, so per-shot variables line up with the per-timeslice ones.
    """
    reference = ds["ip_MA"]
    return np.column_stack([np.asarray(ds[var].broadcast_like(reference).values, dtype=float).ravel() for var in variables])


def feature_fit_arrays(ds: xr.Dataset, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Attach an (N, F) feature matrix built with flat_columns to its device and shot indices.

    NaN device indices (from NaN-padded concatenation) can't be attributed to a device, so their rows are dropped.
    """
    source_idx = flat_columns(ds, ["ds_source_idx"])[:, 0]
    shot_idx = np.asarray(ds["shot"].broadcast_like(ds["ip_MA"]).values).ravel()
    attributed = ~np.isnan(source_idx)
    return features[attributed], source_idx[attributed].astype(int), shot_idx[attributed]


def make_feature_normalizer(
    method: str,
    fit_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    n_devices: int,
    n_features: int,
    target_idx: int,
) -> FeatureNormalizer:
    """Build the per-device stat stage over an arbitrary dimensionless feature vector.

    The counterpart of make_normalizer for the profile and transport predictors,
    whose models normalize their own nn_inputs instead of the 7 physical inputs,
    with the same contract: fit_data None yields identity statistics with the
    correct pytree structure, for callers about to overwrite the buffers from a
    checkpoint (transfer restore).

    fit_data is the (features, source_idx, shot_idx) triple from
    feature_fit_arrays. "physics" feeds the features to the network unchanged,
    so it keeps the CORAL identity buffers (the class is arbitrary for an
    identity transform, but it is fixed here because it names the checkpointed
    pytree). target_idx is the device physics-coral aligns every other device to.
    """
    if method == "physics":
        return CoralFeatureNormalizer.identity(n_devices, n_features)
    if method == "physics-coral":
        if fit_data is None:
            return CoralFeatureNormalizer.identity(n_devices, n_features)
        features, source_idx, shot_idx = fit_data
        return CoralFeatureNormalizer.fit_from_features(features, source_idx, n_devices, shot_idx, target_idx)
    if method == "physics-zscore":
        if fit_data is None:
            return ZScoreFeatureNormalizer.identity(n_devices, n_features)
        features, source_idx, _ = fit_data
        return ZScoreFeatureNormalizer.fit_from_features(features, source_idx, n_devices)
    raise ValueError(f"Unknown feature normalization method: {method}. Must be one of {FEATURE_NORMALIZATIONS}.")


def _feature_matrix(ds: xr.Dataset, variables: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """variables as an (N, len(variables)) matrix with the matching (N,) device and shot indices, see feature_fit_arrays."""
    return feature_fit_arrays(ds, flat_columns(ds, variables))


def _feature_rows(features_fn, inputs: np.ndarray) -> np.ndarray:
    """features_fn applied to every row of an input matrix."""
    return np.asarray(jax.vmap(features_fn)(jnp.asarray(inputs)))


# The InputNormalizer classes per method, grouped by what their fit needs
_STATELESS_NORMALIZERS: dict[str, type[InputNormalizer]] = {"raw": RawNormalizer, "physics": PhysicsNormalizer}
_ZSCORE_NORMALIZERS: dict[str, type[ZScoreNormalizer]] = {
    "zscore": ZScoreNormalizer,
    "physics-zscore": PhysicsZScoreNormalizer,
}
_CORAL_NORMALIZERS: dict[str, type[CoralNormalizer]] = {
    "coral": CoralNormalizer,
    "physics-coral": PhysicsCoralNormalizer,
}
# Every method make_normalizer builds
INPUT_NORMALIZATIONS = (*_STATELESS_NORMALIZERS, *_ZSCORE_NORMALIZERS, *_CORAL_NORMALIZERS)


def make_normalizer(
    method: str,
    train_ds: xr.Dataset | None,
    n_devices: int,
    target_idx: int,
    with_energy: bool = False,
) -> InputNormalizer:
    """Build the normalizer for a case.

    train_ds is required for the stats-bearing methods unless the caller is
    about to overwrite the module from a checkpoint (transfer restore), then
    passing None yields identity stats with the correct pytree structure.
    target_idx is the device the CORAL methods align every other device to.
    with_energy adds the stored energy as an 8th slot (see normalize_with_energy).
    """
    variables = (*NORM_INPUT_VARS, ENERGY_VAR) if with_energy else NORM_INPUT_VARS
    if method in _STATELESS_NORMALIZERS:
        return _STATELESS_NORMALIZERS[method]()
    if method in _ZSCORE_NORMALIZERS:
        zscore_cls = _ZSCORE_NORMALIZERS[method]
        if train_ds is None:
            return zscore_cls.identity(n_devices, len(variables))
        return zscore_cls.fit(train_ds, n_devices, variables)
    if method in _CORAL_NORMALIZERS:
        coral_cls = _CORAL_NORMALIZERS[method]
        if train_ds is None:
            return coral_cls.identity(n_devices, len(variables))
        return coral_cls.fit(train_ds, n_devices, target_idx, variables)
    raise ValueError(f"Unknown normalization method: {method}")
