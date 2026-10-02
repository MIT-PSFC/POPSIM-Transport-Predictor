"""rho_tor_norm = sqrt(Phi_N), with Phi_N the integral of q over psi_N, for the raw-file devices (DIII-D, TCV).

geqdsk_psi_n_grid through phi_n_map are kept identical to transport-validation-datasets (machine/generic.py),
which builds the C-Mod and MAST stores,
and tests/datasets/test_rho_tor_norm.py holds the same cases as its tests, so all four device stores share one map.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicHermiteSpline
from scipy.special import xlogy

# How Phi_N continues past the LCFS, see phi_n_map.
SOL_EXTENSIONS = ("secant", "tangent")

# The secant SOL extension takes its slope over psi_N from here to the LCFS.
SECANT_PSI_N = 0.95

# Outermost finite-q surfaces the logarithmic q tail of a diverted plasma is fit to, see phi_n_map.
Q_TAIL_FIT_SURFACES = 4

# Points of the dense psi_N table PhiNMap.psi_n inverts Phi_N on
NUM_INVERSE_POINTS = 4097


def geqdsk_psi_n_grid(num_psi: int) -> np.ndarray:
    """The psi_N grid of a GEQDSK profile such as qpsi, uniform from 0 at the axis to 1 at the LCFS by the format's definition.

    Args:
        num_psi: Number of points of the profile.

    Returns:
        (num_psi,) psi_N grid.
    """
    return np.linspace(0.0, 1.0, num_psi)


def cumulative_q_integral(psi_n_grid: np.ndarray, qpsi: np.ndarray) -> np.ndarray:
    """Integrate the safety factor over normalized poloidal flux, outward from the axis, by Simpson's rule.

    The toroidal flux is phi = integral q dpsi,
    so this is phi in units of (psi_boundary - psi_axis),
    and dividing it by its last value gives the normalized toroidal flux Phi_N.

    Args:
        psi_n_grid: (n_psi,) increasing psi_N grid, 0 at the axis.
        qpsi: (..., n_psi) safety factor on psi_n_grid.

    Returns:
        (..., n_psi) integral of q dpsi_N from 0 to each grid point, starting at 0.
    """
    return cumulative_simpson(qpsi, x=psi_n_grid, initial=0.0)


def _q_tail_integral(
    psi_n_from: np.ndarray | float,
    psi_n_to: np.ndarray | float,
    q_offset: float,
    q_log_slope: float,
) -> np.ndarray | float:
    """Integral of q = q_offset - q_log_slope ln(1 - psi_N) over psi_N, finite up to psi_N = 1.

    The antiderivative of -ln(1 - psi_N) is (1 - psi_N) ln(1 - psi_N) - (1 - psi_N),
    and xlogy keeps it 0 at psi_N = 1.

    Args:
        psi_n_from: Lower limit.
        psi_n_to: Upper limit, any shape broadcasting with psi_n_from.
        q_offset: q_offset of the tail.
        q_log_slope: q_log_slope of the tail.

    Returns:
        The integral, shaped like the broadcast limits.
    """
    one_minus_from = 1.0 - psi_n_from
    one_minus_to = 1.0 - psi_n_to
    antiderivative_from = q_offset * psi_n_from + q_log_slope * (xlogy(one_minus_from, one_minus_from) - one_minus_from)
    antiderivative_to = q_offset * psi_n_to + q_log_slope * (xlogy(one_minus_to, one_minus_to) - one_minus_to)
    return antiderivative_to - antiderivative_from


@dataclass(frozen=True)
class PhiNMap:
    """Phi_N(psi_N) of one equilibrium, built by phi_n_map.

    Phi_N is the integral of |q| over psi_N, normalized to 1 at the LCFS.
    Inside the last surface of finite q the integral is Simpson's rule over the surfaces,
    interpolated between them by a cubic Hermite spline whose slope is q itself,
    so dPhi_N/dpsi_N stays continuous and gradients have no kinks at the surfaces.
    From there to the LCFS it is the analytic integral of the logarithmic q tail,
    zero width when q is finite at the LCFS.
    Outside the LCFS Phi_N continues linearly in psi_N with sol_slope.
    """

    q_integral_spline: CubicHermiteSpline
    psi_n_join: float  # last surface of finite q, 1 when q is finite at the LCFS
    q_integral_join: float  # integral of |q| from the axis to psi_n_join
    q_offset: float  # q_offset of the tail, 0 when q is finite at the LCFS
    q_log_slope: float  # q_log_slope of the tail, 0 when q is finite at the LCFS
    q_integral_total: float  # integral of |q| from the axis to the LCFS
    sol_slope: float  # dPhi_N/dpsi_N outside the LCFS

    def phi_n(self, psi_n: np.ndarray) -> np.ndarray:
        """Phi_N at psi_N, any shape, NaN where psi_n is.

        psi_N below 0, which interpolation can give next to the axis, maps to 0.

        Args:
            psi_n: Normalized poloidal flux.

        Returns:
            Phi_N shaped like psi_n, exactly 1 at the LCFS.
        """
        psi_n_clipped = np.maximum(psi_n, 0.0)
        psi_n_inside = np.minimum(psi_n_clipped, 1.0)
        psi_n_interior = np.minimum(psi_n_inside, self.psi_n_join)
        q_integral_interior = self.q_integral_spline(psi_n_interior)
        q_integral_tail = self.q_integral_join + _q_tail_integral(self.psi_n_join, psi_n_inside, self.q_offset, self.q_log_slope)
        with np.errstate(invalid="ignore"):
            q_integral = np.where(psi_n_inside <= self.psi_n_join, q_integral_interior, q_integral_tail)
            phi_n_inside = q_integral / self.q_integral_total
            phi_n_inside = np.where(psi_n_inside == 1.0, 1.0, phi_n_inside)
            phi_n_outside = 1.0 + self.sol_slope * (psi_n_clipped - 1.0)
            return np.where(psi_n_clipped <= 1.0, phi_n_inside, phi_n_outside)

    def psi_n(self, phi_n: np.ndarray) -> np.ndarray:
        """psi_N at Phi_N, the inverse of PhiNMap.phi_n.

        Inside the LCFS Phi_N is inverted by interpolation on a dense table of NUM_INVERSE_POINTS,
        outside it through the linear continuation.

        Args:
            phi_n: Normalized toroidal flux, any shape, NaN where unknown.

        Returns:
            psi_N shaped like phi_n, NaN where phi_n is.
        """
        psi_n_table = np.linspace(0.0, 1.0, NUM_INVERSE_POINTS)
        phi_n_table = self.phi_n(psi_n_table)
        psi_n_inside = np.interp(phi_n, phi_n_table, psi_n_table)
        psi_n_outside = 1.0 + (phi_n - 1.0) / self.sol_slope
        with np.errstate(invalid="ignore"):
            return np.where(phi_n <= 1.0, psi_n_inside, psi_n_outside)


def phi_n_map(psi_n_grid: np.ndarray, qpsi: np.ndarray, sol_extension: str) -> PhiNMap | None:
    """Build the Phi_N(psi_N) map of one equilibrium from its q profile.

    The sign of q cancels in Phi_N, so |q| is integrated once q is known to keep one sign.
    q is infinite on the surfaces of a diverted plasma where it diverges at the LCFS.
    Past the last surface of finite q it is integrated analytically
    as q = a - b ln(1 - psi_N), fit to the Q_TAIL_FIT_SURFACES surfaces inside it.
    Outside the LCFS Phi_N continues linearly in psi_N,
    with the secant slope (1 - Phi_N(SECANT_PSI_N)) / (1 - SECANT_PSI_N)
    or the tangent slope |q(1)| / integral_0^1 |q| dpsi_N,
    where q(1) is the outermost finite q when q diverges.

    Args:
        psi_n_grid: (n_psi,) increasing psi_N grid, 0 at the axis to 1 at the LCFS.
        qpsi: (n_psi,) safety factor on psi_n_grid, infinite where it diverges.
        sol_extension: One of SOL_EXTENSIONS.

    Returns:
        The map, or None when the q profile is unusable:
        a NaN q, a finite q that changes sign, fewer than 2 surfaces of finite q from the axis,
        a diverging q with fewer than Q_TAIL_FIT_SURFACES finite surfaces to fit the tail to,
        a q integral that is not increasing, or a tail fit with q not positive and increasing.

    Raises:
        ValueError: If sol_extension is not one of SOL_EXTENSIONS.
    """
    if sol_extension not in SOL_EXTENSIONS:
        raise ValueError(f"sol_extension must be one of {SOL_EXTENSIONS}, got {sol_extension!r}")
    if np.isnan(qpsi).any():
        return None
    mask_q_infinite = np.isinf(qpsi)
    num_q_finite = int(np.argmax(mask_q_infinite)) if mask_q_infinite.any() else qpsi.size
    q_diverges = num_q_finite < qpsi.size
    if num_q_finite < 2 or (q_diverges and num_q_finite < Q_TAIL_FIT_SURFACES):
        return None
    q_finite = qpsi[:num_q_finite]
    if not ((q_finite > 0).all() or (q_finite < 0).all()):
        return None
    psi_n_inside = psi_n_grid[:num_q_finite]
    q_inside = np.abs(q_finite)
    q_integral_inside = cumulative_q_integral(psi_n_inside, q_inside)
    q_integral_steps = np.diff(q_integral_inside)
    if not (q_integral_steps > 0).all():
        return None

    psi_n_join = float(psi_n_inside[-1])
    q_offset, q_log_slope = 0.0, 0.0
    if q_diverges:
        psi_n_fit = psi_n_inside[-Q_TAIL_FIT_SURFACES:]
        q_fit = q_inside[-Q_TAIL_FIT_SURFACES:]
        log_term_fit = -np.log(1.0 - psi_n_fit)
        constant_term_fit = np.ones_like(psi_n_fit)
        design = np.stack([constant_term_fit, log_term_fit], axis=1)
        (q_offset, q_log_slope), *_ = np.linalg.lstsq(design, q_fit, rcond=None)
        log_term_join = -np.log(1.0 - psi_n_join)
        q_at_join = q_offset + q_log_slope * log_term_join
        if q_log_slope <= 0 or q_at_join <= 0:
            return None
    q_integral_join = float(q_integral_inside[-1])
    q_integral_tail = _q_tail_integral(psi_n_join, 1.0, q_offset, q_log_slope)
    q_integral_total = q_integral_join + float(q_integral_tail)
    q_integral_spline = CubicHermiteSpline(psi_n_inside, q_integral_inside, q_inside)

    # The secant slope needs Phi_N inside the LCFS, which does not depend on the slope
    phi_n_mapping_inside = PhiNMap(
        q_integral_spline=q_integral_spline,
        psi_n_join=psi_n_join,
        q_integral_join=q_integral_join,
        q_offset=float(q_offset),
        q_log_slope=float(q_log_slope),
        q_integral_total=q_integral_total,
        sol_slope=np.nan,
    )
    if sol_extension == "secant":
        phi_n_secant_start = phi_n_mapping_inside.phi_n(SECANT_PSI_N)
        sol_slope = (1.0 - float(phi_n_secant_start)) / (1.0 - SECANT_PSI_N)
    else:
        sol_slope = q_inside[-1] / q_integral_total
    return replace(phi_n_mapping_inside, sol_slope=float(sol_slope))


def mappable_q_profiles(psi_n_grid: np.ndarray, qpsi: np.ndarray) -> np.ndarray:
    """Mark the reconstructions whose q profile gives a Phi_N map (phi_n_map).

    Whether a q profile maps does not depend on the SOL extension.

    Args:
        psi_n_grid: (n_psi,) increasing psi_N grid, 0 at the axis to 1 at the LCFS.
        qpsi: (n_eq, n_psi) safety factor of each reconstruction on psi_n_grid, infinite where it diverges.

    Returns:
        (n_eq,) True where the q profile maps.
    """
    mask_mappable = [phi_n_map(psi_n_grid, qpsi_slice, "secant") is not None for qpsi_slice in qpsi]
    return np.array(mask_mappable, dtype=bool)
