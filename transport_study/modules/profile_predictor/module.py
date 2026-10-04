from enum import IntEnum

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from jaxtyping import Array, ArrayLike
from loguru import logger
from popsim import TimeIndepModule
from popsim.basis import Basis1DProtocol, BSplineBasis, InterpedLinearBasis
from popsim.math_utils import safe_log
from popsim.ml.rtd_mlp import Activation, RtdMLP

from transport_study import RADIAL_DIM
from transport_study.modules import plasma_parameters
from transport_study.modules.normalization import (
    FeatureNormalizer,
    feature_fit_arrays,
    flat_columns,
    make_feature_normalizer,
)


class ProfileShape(TimeIndepModule):
    """
    A module defining a profile shape on the rho_tor_norm grid [0, 1].

    The profile shape can be specified either directly with points on the rho grid or with a set of coefficients for a B-spline basis.
    """

    basis: Basis1DProtocol  # B-spline basis used to define the profile shape
    coeffs: Array  # Coefficients for the B-spline basis
    normalize: bool = eqx.field(static=True)  # Whether to normalize the profile shape

    def __init__(self, basis: Basis1DProtocol, coeffs: Array, normalize: bool = True):
        self.basis = basis
        self.coeffs = coeffs
        self.normalize = normalize

    def __call__(self, rho: Array, n_normalize: int = 100) -> Array:
        """Evaluate the profile shape at the given rho values.

        Args:
            rho (Array): Grid values at which to evaluate the profile shape.
            n_normalize (int, optional): Number of points to use for normalization. Defaults to 100.

        Returns:
            Array: The profile shape evaluated at each rho value.
        """

        # Check that rho values are between 0 and 1.2
        rho = eqx.error_if(
            rho,
            jnp.any(jnp.logical_or(rho < 0.0, rho > 1.2)),
            "rho values must be in the range [0, 1.2]",
        )

        vals = self.basis(self.coeffs, rho)
        if self.normalize:
            vals = vals / self.integral(n_normalize=n_normalize)
        return vals

    def integral(
        self,
        n_normalize: int = 100,
    ) -> float:
        """Calculate the integral of the profile shape."""
        rho_norm = jnp.linspace(0, 1.2, n_normalize)
        vals_rho_norm = self.basis(self.coeffs, rho_norm)
        integral = jnp.trapezoid(vals_rho_norm, rho_norm)
        return integral

    def visualize(self, rho: Array = None, ax=None):
        """Visualize the profile shape and the components"""
        if rho is None:
            rho = jnp.linspace(0, 1.2, 100)
        # Calculate the profile shape and its components
        profile_shape = self(rho)

        # Create a new figure and axis if none are provided
        if ax is None:
            _fig, ax = plt.subplots()

        # Plot the overall profile shape
        ax.plot(rho, profile_shape, label="Profile Shape", color="black", linewidth=2)

        # Add labels and legend
        ax.set_xlabel(r"$\rho_{tor,N}$")
        ax.set_ylabel("Profile value")
        ax.legend()

        # Display the plot if a new figure was created
        if ax is None:
            plt.show()

    @classmethod
    def make_bspline(cls, coeffs: Array, normalize: bool = True) -> "ProfileShape":
        """Create a ProfileShape with a B-spline basis.

        Args:
            coeffs (Array): Coefficients for the B-spline basis.
            normalize (bool, optional): Whether to normalize the profile shape. Defaults to True.

        Returns:
            ProfileShape: A ProfileShape with the specified B-spline basis and coefficients.
        """
        n_splines = coeffs.size
        basis = BSplineBasis(n_comps=n_splines)

        return cls(basis=basis, coeffs=coeffs, normalize=normalize)

    @classmethod
    def make_points(cls, points: Array, grid: Array | np.ndarray, normalize: bool = True) -> "ProfileShape":
        """Create a ProfileShape with points on the grid. To evaluate the profile shape on an arbitrary grid, we use interpolation.

        Args:
            points (Array): points defining the profile shape.
            grid (Array): grid points at which the profile shape is defined.
            normalize (bool, optional): Whether to normalize the profile shape. Defaults to True.

        Returns:
            ProfileShape: A ProfileShape with the specified points and grid.
        """
        assert points.size == grid.size
        # In this case, the "coeffs" are just the points on the grid.
        basis = InterpedLinearBasis(
            grid=grid,
        )
        return cls(basis=basis, coeffs=points, normalize=normalize)


# Size of the dimensionless nn_inputs feature vector every profile model consumes
N_NN_INPUTS = 10

# Names of the nn_inputs slots, in order (see Inputs.nn_inputs).
# The data visualization plots these, so it shows exactly the feature space the models consume
NN_INPUT_NAMES = (
    "beta",
    "q_star",
    "epsilon",
    "f_G",
    "aB0",
    "beta_tor_norm",
    "elongation",
    "triangularity_upper",
    "triangularity_lower",
    "log_nu_star",
)

# Dataset variables Inputs.nn_inputs is derived from
NN_INPUT_SOURCE_VARS = (
    "ip_MA",
    "b0",
    "b_geo",
    "beta_tor_norm",
    "n_e_line_average_1e20",
    "geometric_axis_r",
    "minor_radius",
    "elongation",
    "triangularity_upper",
    "triangularity_lower",
)


class DerivedPlasmaParameters:
    """Dimensionless parameters shared by the profile and transport Inputs, which both carry the fields read here."""

    @property
    def epsilon(self):
        return plasma_parameters.inverse_aspect_ratio(self.minor_radius, self.geometric_axis_r)

    @property
    def q_star(self):
        return plasma_parameters.q_star(
            self.ip_MA,
            self.b_geo,
            self.geometric_axis_r,
            self.minor_radius,
            self.elongation,
            self.triangularity_upper,
            self.triangularity_lower,
        )

    @property
    def fGW(self):
        return plasma_parameters.greenwald_fraction(self.n_e_line_average_1e20, self.ip_MA, self.minor_radius)

    @property
    def aB0(self):
        return plasma_parameters.a_b0(self.minor_radius, self.b_geo)

    @property
    def volume_approx(self):
        return plasma_parameters.volume_approx(self.geometric_axis_r, self.minor_radius, self.elongation)

    def beta_tor_of(self, beta_tor_norm: ArrayLike) -> ArrayLike:
        """Toroidal beta as a fraction, from the IMAS percent beta_tor_norm with b0 at r0."""
        return plasma_parameters.beta_tor_from_beta_tor_norm(beta_tor_norm, self.ip_MA, self.minor_radius, self.b0)

    def te_approx_of(self, beta_tor: ArrayLike) -> ArrayLike:
        """Single-fluid temperature estimate <p> / n_e [keV] of a toroidal beta."""
        return plasma_parameters.te_approx_keV(beta_tor, self.b0, self.n_e_line_average_1e20)

    def nu_star_of(self, te_keV: ArrayLike) -> ArrayLike:
        return plasma_parameters.nu_star(te_keV, self.n_e_line_average_1e20, self.q_star, self.geometric_axis_r, self.epsilon)


@chex.dataclass
class Inputs(DerivedPlasmaParameters):
    ip_MA: float  # Plasma current [MA]
    b0: float  # Vacuum toroidal field at r0, the one IMAS normalizes beta_tor_norm with [T]
    b_geo: float  # Vacuum toroidal field at the geometric axis [T]
    beta_tor_norm: float  # Normalized beta [-]
    n_e_line_average_1e20: float  # line-averaged electron density [10^20 m^-3]
    geometric_axis_r: float  # Geometric major radius [m]
    minor_radius: float  # Minor radius [m]
    elongation: float  # Elongation [-]
    triangularity_upper: float  # Upper triangularity [-]
    triangularity_lower: float  # Bottom triangularity [-]
    ds_source_idx: float  # Device index selecting per-device normalization statistics

    # Other
    rho: Array  # rho_tor_norm values to evaluate the profiles at

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, rho: Array) -> "Inputs":
        return cls(
            ip_MA=ds["ip_MA"].data,
            b0=ds["b0"].data,
            b_geo=ds["b_geo"].data,
            beta_tor_norm=ds["beta_tor_norm"].data,
            n_e_line_average_1e20=ds["n_e_line_average_1e20"].data,
            geometric_axis_r=ds["geometric_axis_r"].data,
            minor_radius=ds["minor_radius"].data,
            elongation=ds["elongation"].data,
            triangularity_upper=ds["triangularity_upper"].data,
            triangularity_lower=ds["triangularity_lower"].data,
            ds_source_idx=ds["ds_source_idx"].data,
            rho=rho,
        )

    @property
    def beta(self):
        """Toroidal beta as a fraction of the measured beta_tor_norm."""
        return self.beta_tor_of(self.beta_tor_norm)

    @property
    def te_approx(self):
        """Single-fluid temperature estimate <p> / n_e [keV]."""
        return self.te_approx_of(self.beta)

    @property
    def w_approx(self):
        """Stored energy [MJ] of beta_tor_norm through the store's own inverse, with volume_approx as the volume.

        The store's betan used the reconstruction volume,
        so this carries the volume_approx / reconstruction volume ratio.
        """
        return plasma_parameters.energy_mhd_MJ_from_beta_tor_norm(
            self.beta_tor_norm,
            self.volume_approx,
            self.minor_radius,
            self.b0,
            self.ip_MA,
        )

    @property
    def nu_star(self):
        return self.nu_star_of(self.te_approx)

    @property
    def nn_inputs(self):
        # N_NN_INPUTS dimensionless parameters derived from original inputs
        inp_array = jnp.array(
            [
                self.beta,
                self.q_star,
                self.epsilon,
                self.fGW,
                self.aB0,
                self.beta_tor_norm,
                self.elongation,
                self.triangularity_upper,
                self.triangularity_lower,
                safe_log(self.nu_star),
            ]
        )
        return inp_array


# Inputs fields nn_input_matrix reads from a dataset
_NN_INPUT_MATRIX_VARS = (
    "ip_MA",
    "b0",
    "b_geo",
    "beta_tor_norm",
    "n_e_line_average_1e20",
    "geometric_axis_r",
    "minor_radius",
    "elongation",
    "triangularity_upper",
    "triangularity_lower",
)


def nn_input_matrix(ds: xr.Dataset) -> np.ndarray:
    """(N, N_NN_INPUTS) matrix of the dimensionless nn_inputs over a flattened dataset (see flat_columns).

    Shared by the normalizer fit and the data visualization,
    so both see exactly the feature space the modules consume.
    """
    columns = flat_columns(ds, _NN_INPUT_MATRIX_VARS)
    input_columns = dict(zip(_NN_INPUT_MATRIX_VARS, columns.T, strict=True))
    # ds_source_idx and rho are unused by nn_inputs
    inputs = Inputs(**input_columns, ds_source_idx=np.zeros(len(columns)), rho=jnp.zeros(1))
    return np.asarray(inputs.nn_inputs).T


def make_nn_input_normalizer(method: str, fit_ds: xr.Dataset | None, n_devices: int, target_idx: int) -> FeatureNormalizer:
    """Build the per-device stat stage over the 10 dimensionless nn_inputs.

    Thin wrapper around normalization.make_feature_normalizer
    (which holds the method dispatch shared with the transport predictor),
    so fit_ds None yields identity statistics with the correct pytree structure
    for callers about to overwrite the buffers from a checkpoint.
    """
    fit_data = None if fit_ds is None else feature_fit_arrays(fit_ds, nn_input_matrix(fit_ds))
    return make_feature_normalizer(method, fit_data, n_devices, N_NN_INPUTS, target_idx)


@chex.dataclass
class Outputs:
    ne: xr.DataArray  # Electron density profile [10^20 m^-3]
    te: xr.DataArray  # Electron temperature profile [keV]
    debug_info: dict | None = None


def static_rhogrid(rhogrid: ArrayLike) -> tuple:
    """The rho grid as a tuple of floats, since arrays in static fields break pytree metadata equality."""
    return tuple(np.asarray(rhogrid).tolist())


def profile_outputs(rhogrid: tuple, ne: Array, te: Array, debug_info: dict | None = None) -> Outputs:
    """Outputs holding the ne and te profiles labeled with the rho grid."""
    rho_coords = {RADIAL_DIM: list(rhogrid)}
    ne_da = xr.DataArray(data=ne, dims=(RADIAL_DIM,), coords=rho_coords)
    te_da = xr.DataArray(data=te, dims=(RADIAL_DIM,), coords=rho_coords)
    return Outputs(ne=ne_da, te=te_da, debug_info=debug_info)


class ShapeType(IntEnum):
    PCA_LIKE = 0
    CONVEX_COMBINATION = 1


# Model types built as ProfilePredictorShapeInit, the only ones with shape bases to freeze
MODEL_TYPES_WITH_SHAPES = ("shape-init-pca", "shape-init-kmeans")


def kmeans_initial_guess(
    n_shapes: int,
    te_data: xr.DataArray,
    ne_data: xr.DataArray,
    sample_dim: str,
    seed: int = 0,
):
    from sklearn.cluster import KMeans

    te_data = te_data.transpose(sample_dim, ...)
    ne_data = ne_data.transpose(sample_dim, ...)

    # Small datasets may have fewer samples than shapes
    # (e.g. exnihilo with 1 target shot and it happens to only have 2 or 3 valid profile fits)
    n_clusters = min(n_shapes, te_data.sizes[sample_dim])
    if n_clusters < n_shapes:
        logger.warning(
            f"Only {te_data.sizes[sample_dim]} samples available for {n_shapes} k-means shapes, clustering into {n_clusters} and repeating centers"
        )

    te_kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(te_data.values)
    ne_kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(ne_data.values)

    te_shapes = [
        ProfileShape.make_points(points=te_kmeans.cluster_centers_[i % n_clusters], grid=te_data[RADIAL_DIM].values)
        for i in range(n_shapes)
    ]
    ne_shapes = [
        ProfileShape.make_points(points=ne_kmeans.cluster_centers_[i % n_clusters], grid=ne_data[RADIAL_DIM].values)
        for i in range(n_shapes)
    ]
    return te_shapes, ne_shapes


def pca_initial_guess(n_shapes: int, te_data: xr.DataArray, ne_data: xr.DataArray, sample_dim: str):
    from xeofs.single import EOF

    # Small datasets may have fewer samples than shapes
    # (e.g. exnihilo with 1 target shot and it happens to only have 2 or 3 valid profile fits)
    n_modes = min(n_shapes, te_data.sizes[sample_dim])
    if n_modes < n_shapes:
        logger.warning(
            f"Only {te_data.sizes[sample_dim]} samples available for {n_shapes} PCA shapes, fitting {n_modes} modes and repeating them"
        )

    te_eof = EOF(n_modes=n_modes)
    te_eof.fit(te_data, dim=sample_dim)
    te_components = te_eof.components()
    te_shapes = [
        ProfileShape.make_points(
            points=te_components.sel(mode=te_components.mode.values[i % n_modes]).values,
            grid=te_data[RADIAL_DIM].values,
            normalize=False,
        )
        for i in range(n_shapes)
    ]
    ne_eof = EOF(n_modes=n_modes)
    ne_eof.fit(ne_data, dim=sample_dim)
    ne_components = ne_eof.components()
    ne_shapes = [
        ProfileShape.make_points(
            points=ne_components.sel(mode=ne_components.mode.values[i % n_modes]).values,
            grid=ne_data[RADIAL_DIM].values,
            normalize=False,
        )
        for i in range(n_shapes)
    ]
    return te_shapes, ne_shapes


class ProfilePredictor(TimeIndepModule):
    rhogrid: tuple = eqx.field(static=True)  # The rho grid on which the profiles are evaluated

    nn: RtdMLP
    # Per-device stat stage over the 10 dimensionless nn_inputs
    # Identity CORAL buffers when data_normalization is 'physics', fitted CORAL for 'physics-coral', fitted z-score for 'physics-zscore'
    # Frozen like every normalizer, the trainable getters never include it
    normalizer: FeatureNormalizer

    def outputs_from_points(self, nn_outputs: Array, inputs: Inputs) -> Outputs:
        """Profiles from an output vector of ne points, te points, then the te and ne scale corrections.

        The points are scaled by the line-averaged density and the beta-implied temperature.
        """
        n_pred_points = len(self.rhogrid)
        ne_points = nn_outputs[:n_pred_points]
        te_points = nn_outputs[n_pred_points : 2 * n_pred_points]
        ne_correction = jnp.abs(nn_outputs[-1])
        te_correction = jnp.abs(nn_outputs[-2])
        ne = ne_points * inputs.n_e_line_average_1e20 * ne_correction
        te = te_points * inputs.te_approx * te_correction
        return profile_outputs(self.rhogrid, ne, te)


class ProfilePredictorShapeInit(ProfilePredictor):
    te_shapes: list[ProfileShape]
    ne_shapes: list[ProfileShape]

    shape_type: ShapeType = eqx.field(static=True)
    softmax_temp: float = eqx.field(static=True, default=1.0)

    def __init__(
        self,
        te_shapes: list[ProfileShape],
        ne_shapes: list[ProfileShape],
        nn_width: int,
        nn_depth: int,
        in_size: int,
        softmax_temp: float,
        shape_type: ShapeType,
        rhogrid: tuple,
        key: jax.random.PRNGKey,
        normalizer: FeatureNormalizer,
    ):
        self.te_shapes = te_shapes
        self.ne_shapes = ne_shapes
        self.normalizer = normalizer

        key, subkey = jax.random.split(key)
        self.nn = RtdMLP(
            in_size=in_size,
            out_size=len(te_shapes) + len(ne_shapes) + 1,
            width_size=nn_width,
            depth=nn_depth,
            activation=Activation.RELU,
            final_activation=Activation.IDENTITY,
            key=subkey,
        )
        self.softmax_temp = softmax_temp
        self.shape_type = shape_type
        self.rhogrid = rhogrid

    def __call__(self, inputs: Inputs | xr.Dataset, debug: bool = False) -> Outputs:
        if isinstance(inputs, xr.Dataset):
            inputs = Inputs.from_dataset(inputs, jnp.array(self.rhogrid))

        nn_inputs = self.normalizer(inputs.nn_inputs, inputs.ds_source_idx)

        # Predict the coefficients for the shapes and the correction factor.
        coeffs = self.nn(nn_inputs)
        te_coeffs = coeffs[: len(self.te_shapes)]
        ne_coeffs = coeffs[len(self.te_shapes) : -1]
        te_correction = jnp.abs(coeffs[-1])

        if self.shape_type == ShapeType.CONVEX_COMBINATION:
            te_coeffs = jax.nn.softmax(te_coeffs / self.softmax_temp)
            ne_coeffs = jax.nn.softmax(ne_coeffs / self.softmax_temp)
        elif self.shape_type == ShapeType.PCA_LIKE:
            # NN outputs are already ready to be used as coefficients.
            pass
        else:
            raise ValueError(f"Invalid shape type: {self.shape_type}")

        # Compute the shapes.
        ne_shapes = jnp.stack(
            [w * shape(inputs.rho) for shape, w in zip(self.ne_shapes, ne_coeffs, strict=True)],
            axis=0,
        )
        te_shapes = jnp.stack(
            [w * shape(inputs.rho) for shape, w in zip(self.te_shapes, te_coeffs, strict=True)],
            axis=0,
        )

        # Compute the ne profile.
        ne = jnp.sum(ne_shapes, axis=0) * inputs.n_e_line_average_1e20

        # Compute the te profile using the learned correction.
        te = jnp.sum(te_shapes, axis=0) * inputs.te_approx * te_correction

        if debug:
            debug_info = {
                "te_coeffs": te_coeffs,
                "ne_coeffs": ne_coeffs,
                "te_correction": te_correction,
            }
        else:
            debug_info = None

        return profile_outputs(self.rhogrid, ne, te, debug_info)

    @classmethod
    def init(
        cls,
        n_shapes: int,
        rhogrid: Array,
        nn_width: int,
        nn_depth: int,
        in_size: int,
        shape_type: ShapeType,
        softmax_temp: float,
        prng_seed: int,
        normalizer: FeatureNormalizer,
    ) -> "ProfilePredictor":
        rhogrid_jax = jnp.array(rhogrid)
        rhogrid_tuple = static_rhogrid(rhogrid)
        te_shapes = [
            ProfileShape.make_points(points=jnp.zeros_like(rhogrid_jax), grid=rhogrid_jax, normalize=False) for _ in range(n_shapes)
        ]
        ne_shapes = [
            ProfileShape.make_points(points=jnp.zeros_like(rhogrid_jax), grid=rhogrid_jax, normalize=False) for _ in range(n_shapes)
        ]
        return cls(
            te_shapes=te_shapes,
            ne_shapes=ne_shapes,
            nn_width=nn_width,
            nn_depth=nn_depth,
            in_size=in_size,
            softmax_temp=softmax_temp,
            shape_type=shape_type,
            rhogrid=rhogrid_tuple,
            key=jax.random.PRNGKey(prng_seed),
            normalizer=normalizer,
        )


class ProfilePredictorReservoir(ProfilePredictor):
    """Reservoir computing (echo state network) profile predictor.

    Same input/output contract as ProfilePredictorUnstructuredNN, but instead of an MLP
    the physics inputs are expanded through a fixed random reservoir.
    The reservoir state is iterated to a washed-out state with a leaky tanh update,
    and only the linear readout (self.nn, an RtdMLP with depth=0) is trained.
    The reservoir weights (w_in, w_res, res_bias) are drawn once at init
    and kept frozen by the trainable getter, which only exposes self.nn leaves.
    """

    w_in: Array  # Fixed random input weights (reservoir_size, 10)
    w_res: Array  # Fixed random recurrent weights (reservoir_size, reservoir_size)
    res_bias: Array  # Fixed random bias (reservoir_size,)
    n_steps: int = eqx.field(static=True)  # Reservoir update iterations before readout
    leak_rate: float = eqx.field(static=True)  # Leaky integration rate in (0, 1]

    def __init__(
        self,
        reservoir_size: int,
        rhogrid: tuple,
        key: jax.random.PRNGKey,
        normalizer: FeatureNormalizer,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.5,
        leak_rate: float = 1.0,
        n_steps: int = 20,
    ):
        rhogrid_tuple = static_rhogrid(rhogrid)
        self.normalizer = normalizer

        key_in, key_res, key_bias, key_out = jax.random.split(key, 4)
        self.w_in = input_scaling * jax.random.uniform(key_in, (reservoir_size, N_NN_INPUTS), minval=-1.0, maxval=1.0)
        w_res = jax.random.normal(key_res, (reservoir_size, reservoir_size))
        # Rescale recurrent weights to the requested spectral radius so the state
        # update is contracting (echo state property)
        # Done with numpy at init time since general eigvals is host-side anyway.
        # TODO(ZanderKeith): double check this
        eig_max = float(np.max(np.abs(np.linalg.eigvals(np.asarray(w_res)))))
        self.w_res = w_res * (spectral_radius / eig_max)
        self.res_bias = input_scaling * jax.random.uniform(key_bias, (reservoir_size,), minval=-1.0, maxval=1.0)

        # Trainable linear readout, depth=0 makes RtdMLP a single Linear layer
        self.nn = RtdMLP(
            in_size=reservoir_size,
            out_size=(len(rhogrid_tuple) * 2) + 2,  # +2 for the correction factors
            width_size=reservoir_size,
            depth=0,
            activation=Activation.RELU,
            final_activation=Activation.IDENTITY,
            key=key_out,
        )
        self.n_steps = n_steps
        self.leak_rate = leak_rate
        self.rhogrid = rhogrid_tuple

    def reservoir_state(self, nn_inputs: Array) -> Array:
        """Iterate the leaky tanh reservoir update to a washed-out state."""
        drive = self.w_in @ nn_inputs + self.res_bias

        def step(_i, h):
            return (1.0 - self.leak_rate) * h + self.leak_rate * jnp.tanh(drive + self.w_res @ h)

        h0 = jnp.zeros(self.res_bias.shape, dtype=drive.dtype)
        return jax.lax.fori_loop(0, self.n_steps, step, h0)

    def __call__(self, inputs: Inputs | xr.Dataset, debug: bool = False) -> Outputs:
        if isinstance(inputs, xr.Dataset):
            inputs = Inputs.from_dataset(inputs, jnp.array(self.rhogrid))

        nn_inputs = self.normalizer(inputs.nn_inputs, inputs.ds_source_idx)
        state = self.reservoir_state(nn_inputs)
        # Profile values directly on the rhogrid
        nn_outputs = self.nn(state)
        return self.outputs_from_points(nn_outputs, inputs)


class ProfilePredictorUnstructuredNN(ProfilePredictor):
    def __init__(
        self,
        nn_width: int,
        nn_depth: int,
        rhogrid: tuple,
        key: jax.random.PRNGKey,
        normalizer: FeatureNormalizer,
    ):
        rhogrid_tuple = static_rhogrid(rhogrid)
        self.normalizer = normalizer

        key, subkey = jax.random.split(key)
        self.nn = RtdMLP(
            in_size=N_NN_INPUTS,
            out_size=(len(rhogrid_tuple) * 2) + 2,  # +2 for the correction factors
            width_size=nn_width,
            depth=nn_depth,
            activation=Activation.RELU,
            final_activation=Activation.IDENTITY,
            key=subkey,
        )
        self.rhogrid = rhogrid_tuple

    def __call__(self, inputs: Inputs | xr.Dataset, debug: bool = False) -> Outputs:
        if isinstance(inputs, xr.Dataset):
            inputs = Inputs.from_dataset(inputs, jnp.array(self.rhogrid))

        nn_inputs = self.normalizer(inputs.nn_inputs, inputs.ds_source_idx)
        # Profile values directly on the rhogrid
        nn_outputs = self.nn(nn_inputs)
        return self.outputs_from_points(nn_outputs, inputs)
