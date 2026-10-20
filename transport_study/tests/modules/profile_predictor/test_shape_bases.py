"""The PCA shape basis must be able to represent the profiles it was fit on."""

import numpy as np
import xarray as xr

from transport_study import RADIAL_DIM
from transport_study.config import RHO_GRID
from transport_study.modules.profile_predictor.module import pca_initial_guess


def _unit_integral_profiles(seed: int) -> xr.DataArray:
    """Rank-2 profiles a (1 - rho^2) + b normalized to unit integral, as the t_e_shape / n_e_shape prep does."""
    rng = np.random.default_rng(seed)
    peaked_weights = rng.uniform(0.5, 2.0, size=(40, 1))
    flat_weights = rng.uniform(0.05, 0.5, size=(40, 1))
    profiles = peaked_weights * (1 - RHO_GRID**2) + flat_weights
    integrals = np.trapezoid(profiles, RHO_GRID, axis=1)
    profiles_unit_integral = profiles / integrals[:, np.newaxis]
    sample_coords = np.arange(len(profiles_unit_integral))
    return xr.DataArray(profiles_unit_integral, dims=("sample", RADIAL_DIM), coords={"sample": sample_coords, RADIAL_DIM: RHO_GRID})


def test_pca_basis_spans_the_mean_profile():
    """Unit-integral profiles have a zero-integral spread about their mean,
    so a centered basis spans only that spread and misses the mean itself.
    """
    te_data = _unit_integral_profiles(seed=0)
    ne_data = _unit_integral_profiles(seed=1)

    te_shapes, ne_shapes = pca_initial_guess(2, te_data, ne_data, "sample")

    for shapes, data in ((te_shapes, te_data), (ne_shapes, ne_data)):
        basis = np.stack([np.asarray(shape.coeffs) for shape in shapes], axis=1)
        mean_profile = data.mean("sample").values
        coeffs_lstsq, *_ = np.linalg.lstsq(basis, mean_profile, rcond=None)
        residual_relative = np.linalg.norm(basis @ coeffs_lstsq - mean_profile) / np.linalg.norm(mean_profile)
        assert residual_relative < 1e-4
