"""Similar implementation as the original POPSIM version, but using specifically psi_n with a larger range to be more portable and capture
more interesting profile behavior at the edge"""

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


class ProfileShape(TimeIndepModule):
    """
    A module defining a profile shape on the psi_n grid [0, 1.2].

    The profile shape can either be specified either directly with points on the psi grid or with a set of coefficients for a B-spline basis.
    """

    basis: Basis1DProtocol  # B-spline basis used to define the profile shape
    coeffs: Array  # Coefficients for the B-spline basis
    normalize: bool = eqx.field(static=True)  # Whether to normalize the profile shape

    def __init__(self, basis: Basis1DProtocol, coeffs: Array, normalize: bool = True):
        self.basis = basis
        self.coeffs = coeffs
        self.normalize = normalize

    def __call__(self, psi: Array, n_normalize: int = 100) -> Array:
        """Evaluate the profile shape at the given psi values.

        Args:
            psi (Array): Grid values at which to evaluate the profile shape.
            n_normalize (int, optional): Number of points to use for normalization. Defaults to 100.

        Returns:
            Array: The profile shape evaluated at each psi value.
        """

        # Check that psi values are between 0 and 1.2
        assert jnp.all(psi >= 0) and jnp.all(psi <= 1.2), (
            "Psi values must be between 0 and 1.2"
        )

        vals = self.basis(self.coeffs, psi)
        if self.normalize:
            vals = vals / self.integral(n_normalize=n_normalize)
        return vals

    def integral(
        self,
        n_normalize: int = 100,
    ) -> float:
        """Calculate the integral of the profile shape."""
        psi_norm = jnp.linspace(0, 1.2, n_normalize)
        vals_psi_norm = self.basis(self.coeffs, psi_norm)
        integral = jnp.trapezoid(vals_psi_norm, psi_norm)
        return integral

    def visualize(self, psi: Array = None, ax=None):
        """Visualize the profile shape and the components"""
        if psi is None:
            psi = jnp.linspace(0, 1.2, 100)
        # Calculate the profile shape and its components
        profile_shape = self(psi)

        # Create a new figure and axis if none are provided
        if ax is None:
            _fig, ax = plt.subplots()

        # Plot the overall profile shape
        ax.plot(psi, profile_shape, label="Profile Shape", color="black", linewidth=2)

        # Add labels and legend
        ax.set_xlabel("psi_n")
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
    def make_points(
        cls, points: Array, grid: Array, normalize: bool = True
    ) -> "ProfileShape":
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
    R0: float  # Major radius [m]
    B0: float  # On-axis toroidal field [T]
    Ip: float  # Plasma current [MA]
    a_minor: float  # Minor radius [m]
    kappa: float  # Elongation [-]
    delta_top: float  # Upper triangularity [-]
    delta_bottom: float  # Lower triangularity [-]
    Paux: float  # Auxiliary heating power [MW]
    ne20_edge: float  # Edge electron density [10^20 m^-3]
    beta: float  # Plasma beta [%]
    psi: Array  # Toroidal flux coordinate to evaluate the profiles at

    @property
    def epsilon(self):
        return self.a_minor / self.R0

    @property
    def volume_approx(self):
        return calc_plasma_volume(
            major_radius=self.R0,
            inverse_aspect_ratio=self.epsilon,
            areal_elongation=self.kappa,
        )

    @property
    def q_star(self):
        f_shaping = calc_f_shaping(
            self.epsilon, self.kappa, self.delta_top, self.delta_bottom
        )
        return calc_q_star(
            magnetic_field_on_axis=self.B0,
            major_radius=self.R0,
            inverse_aspect_ratio=self.epsilon,
            plasma_current=self.Ip,
            f_shaping=f_shaping,
        )

    @property
    def nn_inputs(self):
        """An incomplete attempt at having maximally device-independent normalized inputs."""
        inp_array = jnp.array(
            [
                self.B0,
                self.q_star,
                self.epsilon,
                self.kappa,
                self.delta_top,
                self.delta_bottom,
                self.Paux,
                self.ne20_edge,
                self.beta,
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

    te_shapes = [
        ProfileShape.make_points(
            points=te_kmeans.cluster_centers_[i], grid=te_data.psi.values
        )
        for i in range(n_shapes)
    ]
    ne_shapes = [
        ProfileShape.make_points(
            points=ne_kmeans.cluster_centers_[i], grid=ne_data.psi.values
        )
        for i in range(n_shapes)
    ]
    return te_shapes, ne_shapes


def pca_initial_guess(
    n_shapes: int, te_data: xr.DataArray, ne_data: xr.DataArray, sample_dim: str
):
    from xeofs.single import EOF

    te_eof = EOF(n_modes=n_shapes)
    te_eof.fit(te_data, dim=sample_dim)
    te_components = te_eof.components()
    te_shapes = [
        ProfileShape.make_points(
            points=te_components.sel(mode=i).values,
            grid=te_data.psi.values,
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
            grid=ne_data.psi.values,
            normalize=False,
        )
        for i in ne_components.mode.values
    ]
    return te_shapes, ne_shapes


class ProfilePredictor(TimeIndepModule):
    te_shapes: list[ProfileShape]
    ne_shapes: list[ProfileShape]

    nn: RtdMLP
    psigrid: tuple = eqx.field(
        static=True
    )  # The psi grid on which the profiles are evaluated
    shape_type: ShapeType = eqx.field(static=True)
    softmax_temp: float = eqx.field(static=True, default=1.0)
    use_ne_edge: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        te_shapes: list[ProfileShape],
        ne_shapes: list[ProfileShape],
        nn_width: int,
        nn_depth: int,
        softmax_temp: float,
        shape_type: ShapeType,
        use_ne_edge: bool,
        psigrid: tuple,
        key: jax.random.PRNGKey,
    ):
        self.te_shapes = te_shapes
        self.ne_shapes = ne_shapes

        key, subkey = jax.random.split(key)
        self.nn = RtdMLP(
            in_size=9,
            out_size=len(te_shapes) + len(ne_shapes) + 1,
            width_size=nn_width,
            depth=nn_depth,
            activation=Activation.RELU,
            final_activation=Activation.IDENTITY,
            key=subkey,
        )
        self.softmax_temp = softmax_temp
        self.shape_type = shape_type
        self.use_ne_edge = use_ne_edge
        self.psigrid = psigrid

    def __call__(self, inputs: Inputs | xr.Dataset, debug: bool = False) -> Outputs:
        if isinstance(inputs, xr.Dataset):
            inputs = Inputs(
                R0=inputs["R0"].data,
                B0=inputs["B0"].data,
                Ip=inputs["Ip_MA"].data,
                a_minor=inputs["a_minor"].data,
                kappa=inputs["kappa"].data,
                delta=inputs["delta"].data,
                Paux=inputs["Paux_MW"].data,
                ne20_line_avg=inputs["ne20_line_avg"].data,
                Wtot_MJ=inputs["Wtot_MJ"].data,
                psi=jnp.array(self.psigrid),
                ne_edge=inputs["ne20_edge"].data if "ne20_edge" in inputs else None,
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
            [
                w * shape(inputs.psi)
                for shape, w in zip(self.ne_shapes, ne_coeffs, strict=True)
            ],
            axis=0,
        )
        te_shapes = jnp.stack(
            [
                w * shape(inputs.psi)
                for shape, w in zip(self.te_shapes, te_coeffs, strict=True)
            ],
            axis=0,
        )

        # Compute the ne profile. If we are using the edge density as an input, we subtract out the predicted edge density and add the input edge density.
        ne = jnp.sum(ne_shapes, axis=0) * inputs.ne20_line_avg
        if self.use_ne_edge:
            ne = ne - ne[-1] + inputs.ne_edge

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
            ne=xr.DataArray(data=ne, dims=("psi",), coords={"psi": list(self.psigrid)}),
            te=xr.DataArray(data=te, dims=("psi",), coords={"psi": list(self.psigrid)}),
            debug_info=debug_info,
        )

        return out

    @classmethod
    def init(
        cls,
        n_shapes: int,
        psigrid: Array,
        nn_width: int,
        nn_depth: int,
        shape_type: ShapeType,
        softmax_temp: float,
        use_ne_edge: bool,
        prng_seed: int,
    ) -> "ProfilePredictor":
        psigrid_jax = jnp.array(psigrid)
        psigrid_tuple = tuple(psigrid.tolist())
        te_shapes = [
            ProfileShape.make_points(
                points=jnp.zeros_like(psigrid_jax), grid=psigrid_jax, normalize=False
            )
            for _ in range(n_shapes)
        ]
        ne_shapes = [
            ProfileShape.make_points(
                points=jnp.zeros_like(psigrid_jax), grid=psigrid_jax, normalize=False
            )
            for _ in range(n_shapes)
        ]
        return cls(
            te_shapes=te_shapes,
            ne_shapes=ne_shapes,
            nn_width=nn_width,
            nn_depth=nn_depth,
            softmax_temp=softmax_temp,
            shape_type=shape_type,
            use_ne_edge=use_ne_edge,
            psigrid=psigrid_tuple,
            key=jax.random.PRNGKey(prng_seed),
        )
