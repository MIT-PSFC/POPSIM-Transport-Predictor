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

organize_data.normalize_domain implements the same math on whole datasets, it
remains for data visualization only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chex
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

# Regularization for covariance matrix roots (matches normalize_domain)
CORAL_REG = 1e-6
# A device needs at least this many complete samples for a meaningful
# covariance, below it the device keeps the identity transform
MIN_CORAL_SAMPLES = 8


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


class PhysicsNormalizer(InputNormalizer):
    """Dimensionless / device-invariant features computed in-graph.

    Stateless, nothing is fitted. Output slot mapping (slot name -> feature):
    - Ip_MA          -> Ip_MA (kept raw, sufficiently device-invariant)
    - B0             -> q_star (zero triangularity, consistent with H89/H98)
    - R0             -> epsilon = a_minor / R0
    - a_minor        -> a_minor * B0
    - kappa          -> kappa
    - ne20_line_avg  -> Greenwald fraction f_G
    - P_aux_MW       -> P_aux / plasma surface area

    beta is excluded, it needs Wtot which is module state, not an input.
    """

    def _normalize_vec(self, vec: jnp.ndarray, ds_source_idx: ArrayLike) -> jnp.ndarray:
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
        idx = jnp.asarray(ds_source_idx).astype(jnp.int32)
        mean = jnp.take(self.means, idx, axis=0)
        std = jnp.take(self.stds, idx, axis=0)
        return (vec - mean) / std

    @classmethod
    def identity(cls, n_devices: int) -> ZScoreNormalizer:
        return cls(means=jnp.zeros((n_devices, N_FEATURES)), stds=jnp.ones((n_devices, N_FEATURES)))

    @classmethod
    def fit(cls, ds: xr.Dataset, n_devices: int) -> ZScoreNormalizer:
        """Fit per-device mean/std over exactly the 7 input vars.

        Per-var statistics ignore NaNs independently, matching
        normalize_domain's z-score. (normalize_domain also normalized Wtot_MJ,
        the module fits only the model inputs - intentional cleanup.)
        """
        features, source_idx = _feature_matrix(ds)
        means = np.zeros((n_devices, N_FEATURES))
        stds = np.ones((n_devices, N_FEATURES))
        for device_val in np.unique(source_idx):
            device = int(device_val)
            rows = features[source_idx == device]
            with np.errstate(all="ignore"):
                mean = np.nanmean(rows, axis=0)
                std = np.nanstd(rows, axis=0)
            mean = np.where(np.isfinite(mean), mean, 0.0)
            std = np.where(np.isfinite(std) & (std != 0), std, 1.0)
            means[device] = mean
            stds[device] = std
        return cls(means=jnp.asarray(means), stds=jnp.asarray(stds))


class CoralNormalizer(InputNormalizer):
    """Per-device CORAL alignment to the pooled training covariance.

    For device d the transform is: center with the device mean, whiten with
    C_d^{-1/2}, re-color with C_ref^{1/2} (C_ref the pooled covariance over
    all fitting data), then re-add the device mean. transforms has shape
    (n_devices, 7, 7) and means (n_devices, 7), rows follow the global
    config.ds_source_to_idx. Unfitted devices keep the identity transform.
    """

    means: jnp.ndarray
    transforms: jnp.ndarray

    def _normalize_vec(self, vec: jnp.ndarray, ds_source_idx: ArrayLike) -> jnp.ndarray:
        idx = jnp.asarray(ds_source_idx).astype(jnp.int32)
        mean = jnp.take(self.means, idx, axis=0)
        transform = jnp.take(self.transforms, idx, axis=0)
        return (vec - mean) @ transform + mean

    @classmethod
    def identity(cls, n_devices: int) -> CoralNormalizer:
        return cls(
            means=jnp.zeros((n_devices, N_FEATURES)),
            transforms=jnp.tile(jnp.eye(N_FEATURES), (n_devices, 1, 1)),
        )

    @classmethod
    def fit(cls, ds: xr.Dataset, n_devices: int) -> CoralNormalizer:
        """Fit per-device CORAL transforms over exactly the 7 input vars.

        Only rows complete in all 7 vars contribute (covariances need complete
        rows). Devices with fewer than MIN_CORAL_SAMPLES complete rows keep the
        identity transform, a covariance from a handful of samples is
        ill-conditioned. (normalize_domain also included Wtot_MJ in its
        feature matrix, the module fits only the model inputs - intentional
        cleanup.)
        """
        features, source_idx = _feature_matrix(ds)
        valid = ~np.any(np.isnan(features), axis=1)
        pooled = features[valid]
        if len(pooled) < MIN_CORAL_SAMPLES:
            logger.warning(f"CORAL fit got only {len(pooled)} complete samples, using identity transforms")
            return cls.identity(n_devices)
        cov_ref = np.cov(pooled, rowvar=False)
        cr_pos_half = np.real(fractional_matrix_power(cov_ref + CORAL_REG * np.eye(N_FEATURES), 0.5))

        means = np.zeros((n_devices, N_FEATURES))
        transforms = np.tile(np.eye(N_FEATURES), (n_devices, 1, 1))
        for device_val in np.unique(source_idx[valid]):
            device = int(device_val)
            rows = features[valid & (source_idx == device)]
            if len(rows) < MIN_CORAL_SAMPLES:
                continue
            cov_device = np.cov(rows, rowvar=False)
            cd_neg_half = np.real(fractional_matrix_power(cov_device + CORAL_REG * np.eye(N_FEATURES), -0.5))
            means[device] = np.mean(rows, axis=0)
            transforms[device] = cd_neg_half @ cr_pos_half
        return cls(means=jnp.asarray(means), transforms=jnp.asarray(transforms))


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
    raise ValueError(f"Unknown normalization method: {method}")
