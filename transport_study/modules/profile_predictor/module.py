from enum import IntEnum

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import xarray as xr
from jaxtyping import Array
from popsim import TimeIndepModule
from popsim.basis import Basis1DProtocol, BSplineBasis, InterpedLinearBasis
from popsim.cfspopcon_jax.current_drive import calc_f_shaping, calc_q_star
from popsim.cfspopcon_jax.geometry import calc_plasma_volume
from popsim.ml.rtd_mlp import Activation, RtdMLP
from scipy.constants import eV, mu_0


class ProfileShape(TimeIndepModule):
    """
    A module defining a profile shape on the normalized minor radius (rho) grid [0, 1].

    The profile shape can either be specified either directly with points on the rho grid or with a set of coefficients for a B-spline basis.
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
        ax.set_xlabel("rho")
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
    def make_points(cls, points: Array, grid: Array, normalize: bool = True) -> "ProfileShape":
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


@chex.dataclass
class Inputs:
    Ip: float  # Plasma current [MA]
    B0: float  # On-axis toroidal field [T]
    betan: float  # Normalized beta
    ne20_line_avg: float  # line-averaged electron density [10^20 m^-3]
    R0: float  # Geometric major radius [m]
    a_minor: float  # Minor radius [m]
    kappa: float  # Elongation
    delta_top: float  # Upper triangularity
    delta_bot: float  # Bottom triangularity

    # Other
    rho: Array  # Normalized minor radius coordinate to evaluate the profiles at

    @property
    def epsilon(self):
        return self.a_minor / self.R0

    @property
    def q_star(self):
        delta = (self.delta_top + self.delta_bot) / 2
        f_shaping = calc_f_shaping(
            self.epsilon,
            self.kappa,
            delta,
        )
        q_star = calc_q_star(self.B0, self.R0, self.epsilon, self.Ip, f_shaping)
        return q_star

    @property
    def fGW(self):
        greenwald_limit = self.Ip / (jnp.pi * self.a_minor**2)
        return self.ne20_line_avg / greenwald_limit

    @property
    def aB0(self):
        return self.a_minor * self.B0

    @property
    def volume_approx(self):
        return calc_plasma_volume(
            major_radius=self.R0,
            inverse_aspect_ratio=self.epsilon,
            areal_elongation=self.kappa,
        )

    @property
    def beta(self):
        return self.betan * self.Ip / (self.a_minor * self.B0)

    @property
    def te_approx(self):
        pressure_Pa = self.beta * self.B0**2 / (2 * mu_0)
        pressure_eV = pressure_Pa / eV
        pressure_keV20 = pressure_eV / 1e3 / 1e20
        temp_keV = pressure_keV20 / self.ne20_line_avg
        return temp_keV

    @property
    def nn_inputs(self):
        # 9 unitless parameters that maintain the same dimensionality as the original inputs
        inp_array = jnp.array(
            [
                self.beta,
                self.q_star,
                self.epsilon,
                self.fGW,
                self.aB0,
                self.betan,
                self.kappa,
                self.delta_top,
                self.delta_bot,
            ]
        )
        return inp_array


@chex.dataclass
class Outputs:
    ne: xr.DataArray  # Electron density profile [10^20 m^-3]
    te: xr.DataArray  # Electron temperature profile [keV]
    debug_info: dict | None = None


class ShapeType(IntEnum):
    PCA_LIKE = 0
    CONVEX_COMBINATION = 1


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

    te_kmeans = KMeans(n_clusters=n_shapes, random_state=seed).fit(te_data.values)

    ne_kmeans = KMeans(n_clusters=n_shapes, random_state=seed).fit(ne_data.values)

    te_shapes = [ProfileShape.make_points(points=te_kmeans.cluster_centers_[i], grid=te_data.rho.values) for i in range(n_shapes)]
    ne_shapes = [ProfileShape.make_points(points=ne_kmeans.cluster_centers_[i], grid=ne_data.rho.values) for i in range(n_shapes)]
    return te_shapes, ne_shapes


def pca_initial_guess(n_shapes: int, te_data: xr.DataArray, ne_data: xr.DataArray, sample_dim: str):
    from xeofs.single import EOF

    te_eof = EOF(n_modes=n_shapes)
    te_eof.fit(te_data, dim=sample_dim)
    te_components = te_eof.components()
    te_shapes = [
        ProfileShape.make_points(
            points=te_components.sel(mode=i).values,
            grid=te_data.rho.values,
            normalize=False,
        )
        for i in te_components.mode.values
    ]
    ne_eof = EOF(n_modes=n_shapes)
    ne_eof.fit(ne_data, dim=sample_dim)
    ne_components = ne_eof.components()
    ne_shapes = [
        ProfileShape.make_points(
            points=ne_components.sel(mode=i).values,
            grid=ne_data.rho.values,
            normalize=False,
        )
        for i in ne_components.mode.values
    ]
    return te_shapes, ne_shapes


class ProfilePredictor(TimeIndepModule):
    rhogrid: tuple = eqx.field(static=True)  # The rho grid on which the profiles are evaluated

    nn: RtdMLP


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
    ):
        self.te_shapes = te_shapes
        self.ne_shapes = ne_shapes

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
            inputs = Inputs(
                Ip=inputs["Ip_MA"].data,
                B0=inputs["B0"].data,
                betan=inputs["betan"].data,
                ne20_line_avg=inputs["ne20_line_avg"].data,
                R0=inputs["R0"].data,
                a_minor=inputs["a_minor"].data,
                kappa=inputs["kappa"].data,
                delta_top=inputs["delta_top"].data,
                delta_bot=inputs["delta_bot"].data,
                rho=jnp.array(self.rhogrid),
            )

        nn_inputs = inputs.nn_inputs

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
        ne = jnp.sum(ne_shapes, axis=0) * inputs.ne20_line_avg

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

        out = Outputs(
            ne=xr.DataArray(data=ne, dims=("rho",), coords={"rho": list(self.rhogrid)}),
            te=xr.DataArray(data=te, dims=("rho",), coords={"rho": list(self.rhogrid)}),
            debug_info=debug_info,
        )

        return out

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
    ) -> "ProfilePredictor":
        rhogrid_jax = jnp.array(rhogrid)
        rhogrid_tuple = tuple(rhogrid.tolist())
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
        )


class ProfilePredictorUnstructuredNN(ProfilePredictor):
    rho_points: tuple = eqx.field(static=True)  # Hashable points at which the NN predicts

    def __init__(
        self,
        n_points: int,
        nn_width: int,
        nn_depth: int,
        rhogrid: tuple,
        key: jax.random.PRNGKey,
    ):
        rhogrid_tuple = tuple(rhogrid.tolist()) if hasattr(rhogrid, "tolist") else tuple(rhogrid)
        self.rho_points = tuple(jnp.linspace(min(rhogrid_tuple), max(rhogrid_tuple), n_points).tolist())

        key, subkey = jax.random.split(key)
        self.nn = RtdMLP(
            in_size=9,
            out_size=(n_points * 2) + 2,  # +2 for the correction factors
            width_size=nn_width,
            depth=nn_depth,
            activation=Activation.RELU,
            final_activation=Activation.IDENTITY,
            key=subkey,
        )
        self.rhogrid = rhogrid_tuple

    def __call__(self, inputs: Inputs | xr.Dataset, debug: bool = False) -> Outputs:
        if isinstance(inputs, xr.Dataset):
            inputs = Inputs(
                Ip=inputs["Ip_MA"].data,
                B0=inputs["B0"].data,
                betan=inputs["betan"].data,
                ne20_line_avg=inputs["ne20_line_avg"].data,
                R0=inputs["R0"].data,
                a_minor=inputs["a_minor"].data,
                kappa=inputs["kappa"].data,
                delta_top=inputs["delta_top"].data,
                delta_bot=inputs["delta_bot"].data,
                rho=jnp.array(self.rhogrid),
            )

        nn_inputs = inputs.nn_inputs
        rho_points = jnp.asarray(self.rho_points)
        n_pred_points = len(self.rho_points)

        # Predict the profile values at the specified points.
        outputs = self.nn(nn_inputs)
        ne_points = outputs[:n_pred_points]
        te_points = outputs[n_pred_points : 2 * n_pred_points]
        ne_correction = jnp.abs(outputs[-1])
        te_correction = jnp.abs(outputs[-2])

        # Interpolate the predicted points to the rhogrid
        ne = jnp.interp(inputs.rho, rho_points, ne_points) * inputs.ne20_line_avg * ne_correction
        te = jnp.interp(inputs.rho, rho_points, te_points) * te_correction

        out = Outputs(
            ne=xr.DataArray(data=ne, dims=("rho",), coords={"rho": list(self.rhogrid)}),
            te=xr.DataArray(data=te, dims=("rho",), coords={"rho": list(self.rhogrid)}),
            debug_info=None,
        )

        return out
