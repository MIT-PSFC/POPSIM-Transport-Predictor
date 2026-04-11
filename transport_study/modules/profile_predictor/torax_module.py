"""TORAX-based profile predictor for the transport study.

Predicts quasi-steady-state electron temperature and density profiles by running
the TORAX transport code for a single large timestep.  A small neural network
infers the transport coefficients and a heating-power correction factor from the
equilibrium inputs; all other TORAX inputs (geometry, current, density scale)
are derived directly from physics.

Architecture summary
--------------------
1. **NN forward pass** - an MLP maps the 9 ``nn_inputs`` to 7 latent
   transport parameters (see ``_NN_OUT_SIZE`` and :meth:`ProfilePredictorTorax.__call__`).
2. **Physics derivation** - geometry (R0, a_minor, B0, kappa), plasma current
   (Ip), and a baseline heating-power estimate (from beta and volume) are
   computed analytically from the input ``Inputs`` dataclass.
3. **TORAX simulation** - a single step of ``dt_steady`` seconds is taken with
   the constant-diffusivity transport model.  This approximates the quasi-
   steady-state profile because ``dt_steady >> tau_E`` means the implicit
   solver finds the near-equilibrium solution.
4. **Coordinate mapping** - the resulting ``T_e(rho_norm)`` and ``n_e(rho_norm)``
   profiles are mapped to the ``psi_n`` output grid via the circular-geometry
   relation ``psi_n = rho_norm²``.

Neural-network outputs (7 values)
----------------------------------
Index  Raw output  Decoded parameter
-----  ----------  -----------------
  0    log_chi_e       chi_e  [m²/s]  electron thermal diffusivity
  1    log_chi_ratio   chi_i  = chi_e x exp(raw[1]) [m^2/s]
  2    logit_src_rho   source_rho  = sigmoid(raw[2]) ∈ (0, 1)  [rho_norm]
  3    log_src_width   source_width = exp(raw[3]) [rho_norm]
  4    log_D_e         D_e [m²/s]  electron particle diffusivity
  5    log_ne_peak     ne_peaking = exp(raw[5])  core/edge density ratio (initial cond.)
  6    log_P_scale     P_scale = exp(raw[6])  multiplicative correction on physics power

JIT-compilation and differentiability
--------------------------------------
This module is JIT-compilable and supports end-to-end gradients from the output
profiles back through the NN weights.  This works because:

- ``SimulationStepFn.__call__`` and ``get_initial_state_and_post_processed_outputs``
  are both ``@jax.jit`` decorated in TORAX.
- All TORAX pydantic config classes (``BaseModelFrozen`` subclasses) are
  registered JAX pytrees, with non-``JAX_STATIC`` fields as dynamic leaves.
- ``TimeVaryingScalar.value`` is a dynamic JAX leaf that can be patched with
  traced JAX arrays using ``eqx.tree_at`` without rebuilding pydantic objects.

The key design is: the ``SimulationStepFn`` and base ``RuntimeParamsProvider``
are built **once** at ``init`` time and stored as static fields.  Each
``__call__`` patches the chi / source / power leaves of the base provider with
the NN-predicted JAX arrays using ``eqx.tree_at``, then passes the patched
provider to the already-compiled step function.

Gradient path: NN weights → chi_e/chi_i/D_e/P_total/source_rho/source_width
AND R0/a_minor/B0/kappa (all traced JAX scalars) → eqx.tree_at patch into
RuntimeParamsProvider + _build_circular_geometry_jax → JIT step → final T_e/n_e
profiles (JAX arrays) → Huber loss → jax.grad.

Non-differentiable parts:
- Initial density/temperature profiles: built with concrete (non-traced) values.
  The steady-state profile shape is dominated by the transport coefficients and
  heating power, not the initial condition, so this is acceptable.
"""

from __future__ import annotations

import logging
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torax
import xarray as xr
from popsim import TimeIndepModule
from popsim.ml.rtd_mlp import Activation, RtdMLP
from scipy.constants import mu_0
from torax._src.config.build_runtime_params import RuntimeParamsProvider
from torax._src.geometry import circular_geometry
from torax._src.geometry import geometry as torax_geometry
from torax._src.geometry.geometry_provider import ConstantGeometryProvider
from torax._src.orchestration.initial_state import (
    get_initial_state_and_post_processed_outputs,
)
from torax._src.orchestration.run_simulation import make_step_fn
from torax._src.torax_pydantic import model_config as torax_model_config

from transport_study.modules.profile_predictor.module import Inputs, Outputs

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────────────────────────────────────

_NN_OUT_SIZE: int = 7  # number of raw NN output values

# Safety bounds applied after decoding to prevent TORAX divergence.
_CHI_MIN: float = 0.01  # m²/s
_CHI_MAX: float = 100.0  # m²/s
_D_E_MIN: float = 0.01  # m²/s
_D_E_MAX: float = 50.0  # m²/s
_SRC_RHO_MIN: float = 0.01
_SRC_RHO_MAX: float = 0.95
_SRC_WIDTH_MIN: float = 0.02
_SRC_WIDTH_MAX: float = 0.50
_NE_PEAK_MIN: float = 1.0
_NE_PEAK_MAX: float = 10.0
_P_SCALE_MIN: float = 0.1
_P_SCALE_MAX: float = 10.0
_P_TOTAL_MIN: float = 1e3  # W  (1 kW floor)
_P_TOTAL_MAX: float = 5e8  # W  (500 MW ceiling)

# Reference plasma parameters used to build the step_fn once at init time.
_REF_R0: float = 0.67  # C-Mod-like major radius [m]
_REF_A: float = 0.22  # C-Mod-like minor radius [m]
_REF_B0: float = 5.4  # C-Mod-like field [T]
_REF_KAPPA: float = 1.6  # C-Mod-like elongation


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate mapping helper
# ──────────────────────────────────────────────────────────────────────────────


def _rho_to_psigrid(
    te_rho: np.ndarray,
    ne_rho: np.ndarray,
    rho_norm_grid: np.ndarray,
    psigrid: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Map TORAX profiles from rho_norm to the psi_n output grid.

    For circular geometry, the normalised poloidal flux satisfies
    ``psi_n = rho_norm²``, i.e. ``rho_norm = sqrt(psi_n)`` for ``psi_n ∈ [0, 1]``.

    Points with ``psi_n > 1`` (scrape-off-layer region outside TORAX's domain)
    are obtained by linear extrapolation using the gradient at the boundary.

    Args:
        te_rho: Electron temperature [keV] on rho_norm_grid.
        ne_rho: Electron density [10²⁰ m⁻³] on rho_norm_grid.
        rho_norm_grid: Normalised toroidal flux coordinate array ∈ [0, 1], shape (n,).
        psigrid: Target psi_n grid values (tuple of floats).

    Returns:
        te_psi: T_e [keV] on psigrid.
        ne_psi: n_e [10²⁰ m⁻³] on psigrid.
    """
    psi_vals = np.asarray(psigrid, dtype=float)
    te_rho = np.asarray(te_rho, dtype=float)
    ne_rho = np.asarray(ne_rho, dtype=float)

    rho_for_psi = np.sqrt(np.clip(psi_vals, 0.0, 1.0))

    te_out = np.interp(rho_for_psi, rho_norm_grid, te_rho)
    ne_out = np.interp(rho_for_psi, rho_norm_grid, ne_rho)

    # Linear extrapolation into the SOL (psi_n > 1)
    mask_sol = psi_vals > 1.0
    if np.any(mask_sol) and len(rho_norm_grid) >= 2:
        drho = float(rho_norm_grid[-1] - rho_norm_grid[-2])
        if drho > 0.0:
            dTe = (float(te_rho[-1]) - float(te_rho[-2])) / drho
            dne = (float(ne_rho[-1]) - float(ne_rho[-2])) / drho
            rho_extrap = np.sqrt(psi_vals[mask_sol])
            delta_rho = rho_extrap - float(rho_norm_grid[-1])
            te_out[mask_sol] = float(te_rho[-1]) + dTe * delta_rho
            ne_out[mask_sol] = float(ne_rho[-1]) + dne * delta_rho

    te_out = np.maximum(te_out, 0.001)
    ne_out = np.maximum(ne_out, 1e-6)

    return te_out, ne_out


# ──────────────────────────────────────────────────────────────────────────────
# Physics-based power estimation
# ──────────────────────────────────────────────────────────────────────────────


def _estimate_P_total(inputs: Inputs) -> jnp.ndarray:
    """Estimate total heating power [W] from stored energy and confinement scaling.

    Returns a JAX scalar so it can participate in the gradient graph.

    The stored thermal energy is W_th = (3/2) β B₀² / (2μ₀) V.  Inverting the
    ITER L-mode confinement scaling (τ_E ∝ P^-0.69) gives P = (W_th / τ_base)^(1/1.69).

    Args:
        inputs: Equilibrium parameters (JAX-traceable dataclass).

    Returns:
        Estimated heating power [W] as a JAX scalar.
    """
    W_th = jnp.maximum(
        1.5 * inputs.beta * inputs.B0**2 / (2.0 * mu_0) * inputs.volume_approx,
        1e3,
    )

    _C_LMODE = 0.0325
    tau_E_base = jnp.maximum(
        _C_LMODE
        * inputs.Ip**0.96
        * inputs.B0**0.03
        * jnp.maximum(inputs.ne20, 1e-3) ** 0.41
        * inputs.R0**1.97
        * inputs.kappa**0.64
        * inputs.epsilon**0.42,
        1e-6,
    )
    return (W_th / tau_E_base) ** (1.0 / 1.69)


# ──────────────────────────────────────────────────────────────────────────────
# Reference config for compiling the step function once
# ──────────────────────────────────────────────────────────────────────────────


def _make_reference_config(
    n_rho: int,
    dt_steady: float,
    t_edge_keV: float,
) -> dict[str, Any]:
    """Config dict for building the reference step function.

    Uses representative plasma parameters close to C-Mod / small tokamaks.
    The exact values don't matter — they only determine the JIT-compiled
    simulation structure (grid size, equation flags, physics models), not the
    parameter values that the NN will override.
    """
    return {
        "profile_conditions": {
            "Ip": {0: 1.5e6},  # 1.5 MA
            "T_e": {0: {0.0: 3.0, 1.0: t_edge_keV}},
            "T_i": {0: {0.0: 3.0, 1.0: t_edge_keV}},
            "T_e_right_bc": t_edge_keV,
            "T_i_right_bc": t_edge_keV,
            "n_e": {0: {0.0: 3e20, 1.0: 1e20}},
            "n_e_right_bc": 1e20,
        },
        "numerics": {
            "t_final": dt_steady,
            "fixed_dt": dt_steady,
            "evolve_ion_heat": True,
            "evolve_electron_heat": True,
            "evolve_density": True,
            "evolve_current": False,
            "exact_t_final": True,
            "adaptive_dt": False,
        },
        "plasma_composition": {},
        "geometry": {
            "geometry_type": "circular",
            "R_major": _REF_R0,
            "a_minor": _REF_A,
            "B_0": _REF_B0,
            "elongation_LCFS": _REF_KAPPA,
            "n_rho": n_rho,
        },
        "sources": {
            "generic_heat": {
                "P_total": 3e6,
                "gaussian_location": 0.3,
                "gaussian_width": 0.1,
                "electron_heat_fraction": 0.5,
            },
        },
        "transport": {
            "model_name": "constant",
            "chi_e": 1.0,
            "chi_i": 1.0,
            "D_e": 0.5,
            "V_e": 0.0,
        },
        "solver": {},
        "pedestal": {},
        "time_step_calculator": {"calculator_type": "fixed"},
    }


# ──────────────────────────────────────────────────────────────────────────────
# JAX-native circular geometry builder (differentiable w.r.t. R0, a, B0, kappa)
# ──────────────────────────────────────────────────────────────────────────────


def _build_circular_geometry_jax(
    R0: jnp.ndarray,
    a: jnp.ndarray,
    B0: jnp.ndarray,
    kappa: jnp.ndarray,
    torax_mesh: Any,
    rho_hires_norm: np.ndarray,
) -> torax_geometry.Geometry:
    """JAX-native circular geometry builder.

    Replicates :func:`torax._src.geometry.circular_geometry._build_circular_geometry`
    using ``jnp`` instead of ``np``, so that ``R0``, ``a``, ``B0``, and
    ``kappa`` can be JAX traced arrays — enabling JIT compilation and
    end-to-end differentiation through plasma shape parameters.

    ``torax_mesh`` (the normalised rho grid) and ``rho_hires_norm`` (the
    high-resolution grid used for poloidal-flux computations) are structural:
    they only depend on ``n_rho`` and the hires factor, not on plasma parameters.
    They must be pre-built as concrete numpy / pydantic objects and passed in.

    Args:
        R0: Major radius [m] — JAX traced scalar.
        a: Minor radius [m] — JAX traced scalar.
        B0: On-axis toroidal field [T] — JAX traced scalar.
        kappa: Elongation at LCFS — JAX traced scalar.
        torax_mesh: Pre-built concrete ``Grid1D`` (structural, not traced).
        rho_hires_norm: Pre-built concrete numpy array of normalised rho on
            the high-resolution grid (structural, not traced).

    Returns:
        A ``torax_geometry.Geometry`` whose array fields are JAX-traced
        functions of ``R0``, ``a``, ``B0``, and ``kappa``.
    """
    # ── Grid arrays (structural — same for all calls) ────────────────────
    rho_face_norm = jnp.asarray(torax_mesh.face_centers)  # (n+1,)
    rho_norm = jnp.asarray(torax_mesh.cell_centers)  # (n,)
    rho_hi = jnp.asarray(rho_hires_norm)  # (n*hires+1,)

    # ── Unnormalised rho (traced) ────────────────────────────────────────
    rho_b = a  # boundary rho = minor radius
    rho = rho_norm * rho_b
    rho_face = rho_face_norm * rho_b
    rho_hires = rho_hi * rho_b

    # ── Toroidal flux ────────────────────────────────────────────────────
    Phi = jnp.pi * B0 * rho**2
    Phi_face = jnp.pi * B0 * rho_face**2

    # ── Elongation (linear from 1 at axis to kappa at LCFS) ─────────────
    elongation = 1.0 + rho_norm * (kappa - 1.0)
    elongation_face = 1.0 + rho_face_norm * (kappa - 1.0)
    elongation_hires = 1.0 + rho_hi * (kappa - 1.0)

    # ── Volume and area (elongated circular) ─────────────────────────────
    volume = 2.0 * jnp.pi**2 * R0 * rho**2 * elongation
    volume_face = 2.0 * jnp.pi**2 * R0 * rho_face**2 * elongation_face
    volume_hires = 2.0 * jnp.pi**2 * R0 * rho_hires**2 * elongation_hires
    area = jnp.pi * rho**2 * elongation
    area_face = jnp.pi * rho_face**2 * elongation_face
    area_hires = jnp.pi * rho_hires**2 * elongation_hires

    # ── vpr = dV/drho_norm, spr = dS/drho_norm ───────────────────────────
    vpr = 4.0 * jnp.pi**2 * R0 * rho * elongation * rho_b + volume / elongation * (
        kappa - 1.0
    )
    vpr_face = (
        4.0 * jnp.pi** 2 * R0 * rho_face * elongation_face * rho_b
        + volume_face / elongation_face * (kappa - 1.0)
    )
    vpr_hires = (
        4.0 * jnp.pi** 2 * R0 * rho_hires * elongation_hires * rho_b
        + volume_hires / elongation_hires * (kappa - 1.0)
    )
    spr = 2.0 * jnp.pi * rho * elongation * rho_b + area / elongation * (kappa - 1.0)
    spr_face = (
        2.0 * jnp.pi * rho_face * elongation_face * rho_b
        + area_face / elongation_face * (kappa - 1.0)
    )
    spr_hires = (
        2.0 * jnp.pi * rho_hires * elongation_hires * rho_b
        + area_hires / elongation_hires * (kappa - 1.0)
    )

    delta_face = jnp.zeros_like(rho_face)

    # ── Metric coefficients ───────────────────────────────────────────────
    g0, g0_face = vpr / rho_b, vpr_face / rho_b
    g1, g1_face = vpr**2 / rho_b**2, vpr_face**2 / rho_b**2
    g2, g2_face = g1 / R0**2, g1_face / R0**2

    # g3 = <1/R²> for circular geometry
    g3 = 1.0 / (R0**2 * (1.0 - (rho / R0) ** 2) ** 1.5)
    g3_face = 1.0 / (R0**2 * (1.0 - (rho_face / R0) ** 2) ** 1.5)
    g3_hires = 1.0 / (R0**2 * (1.0 - (rho_hires / R0) ** 2) ** 1.5)

    # Simplified J = R*B/(R0*B0) = 1, F = R0*B0 for circular geometry
    J = jnp.ones_like(rho)
    J_face = jnp.ones_like(rho_face)
    F = jnp.full_like(rho, R0 * B0)
    F_face = jnp.full_like(rho_face, R0 * B0)
    F_hires = jnp.full_like(rho_hires, R0 * B0)

    g2g3_over_rhon = 4.0 * jnp.pi**2 * vpr * g3 / (J * R0)
    g2g3_over_rhon_face = 4.0 * jnp.pi**2 * vpr_face * g3_face / (J_face * R0)
    g2g3_over_rhon_hires = 4.0 * jnp.pi**2 * vpr_hires * g3_hires * B0 / F_hires

    # Inboard / outboard radii
    R_out = R0 + rho
    R_out_face = R0 + rho_face
    R_in = R0 - rho
    R_in_face = R0 - rho_face

    # gm4 = <1/B²>, gm5 = <B²>
    eps = (R_out - R_in) / (R_out + R_in)
    eps_face = (R_out_face - R_in_face) / (R_out_face + R_in_face)
    gm4 = B0 ** (-2) * (1.0 + 1.5 * eps**2)
    gm4_face = B0 ** (-2) * (1.0 + 1.5 * eps_face**2)
    gm5 = B0**2 / jnp.sqrt(1.0 - eps**2)
    gm5_face = B0**2 / jnp.sqrt(1.0 - eps_face**2)

    return torax_geometry.Geometry(
        geometry_type=torax_geometry.GeometryType.CIRCULAR,
        torax_mesh=torax_mesh,
        Phi=Phi,
        Phi_face=Phi_face,
        R_major=R0,
        a_minor=rho_b,
        B_0=B0,
        volume=volume,
        volume_face=volume_face,
        area=area,
        area_face=area_face,
        vpr=vpr,
        vpr_face=vpr_face,
        spr=spr,
        spr_face=spr_face,
        delta_face=delta_face,
        g0=g0,
        g0_face=g0_face,
        g1=g1,
        g1_face=g1_face,
        g2=g2,
        g2_face=g2_face,
        g3=g3,
        g3_face=g3_face,
        gm4=gm4,
        gm4_face=gm4_face,
        gm5=gm5,
        gm5_face=gm5_face,
        g2g3_over_rhon=g2g3_over_rhon,
        g2g3_over_rhon_face=g2g3_over_rhon_face,
        g2g3_over_rhon_hires=g2g3_over_rhon_hires,
        F=F,
        F_face=F_face,
        F_hires=F_hires,
        R_in=R_in,
        R_in_face=R_in_face,
        R_out=R_out,
        R_out_face=R_out_face,
        elongation=elongation,
        elongation_face=elongation_face,
        spr_hires=spr_hires,
        rho_hires_norm=rho_hi,
        rho_hires=rho_hires,
        Phi_b_dot=jnp.asarray(0.0),
        _z_magnetic_axis=jnp.asarray(0.0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-sample config builder (used by Stage-1 offline fitting in torax_trb.py)
# ──────────────────────────────────────────────────────────────────────────────


def _build_torax_config(
    inputs: Inputs,
    chi_e: float,
    chi_i: float,
    source_rho: float,
    source_width: float,
    D_e: float,
    ne_peaking: float,
    P_total: float,
    t_edge_keV: float,
    dt_steady: float,
    n_rho: int,
) -> dict[str, Any]:
    """Build a per-sample TORAX config dict from concrete physics inputs.

    All parameters must be plain Python floats (no JAX traced arrays).
    The result can be passed directly to ``torax.ToraxConfig.from_dict()``.

    This is used by Stage-1 offline fitting in ``torax_trb.py`` where
    ``scipy.optimize`` calls the objective function many times with concrete
    values.  The JIT-compiled :meth:`ProfilePredictorTorax.__call__` uses a
    different path (``eqx.tree_at`` patching of a pre-built provider).

    Args:
        inputs: Equilibrium parameters with concrete float fields.
        chi_e: Electron thermal diffusivity [m²/s].
        chi_i: Ion thermal diffusivity [m²/s].
        source_rho: Heat source Gaussian centre [rho_norm].
        source_width: Heat source Gaussian width [rho_norm].
        D_e: Electron particle diffusivity [m²/s].
        ne_peaking: Core-to-edge density ratio (sets initial profile shape).
        P_total: Total heating power [W].
        t_edge_keV: Edge temperature boundary condition [keV].
        dt_steady: TORAX timestep [s].
        n_rho: Radial grid resolution.

    Returns:
        Config dict suitable for ``torax.ToraxConfig.from_dict()``.
    """
    ne_edge_m3 = float(inputs.ne20) * 1e20
    ne_core_m3 = ne_edge_m3 * float(ne_peaking)
    T_core = max(float(inputs.te_approx), t_edge_keV + 0.1)

    return {
        "profile_conditions": {
            "Ip": {0: float(inputs.Ip) * 1e6},
            "T_e": {0: {0.0: T_core, 1.0: t_edge_keV}},
            "T_i": {0: {0.0: T_core, 1.0: t_edge_keV}},
            "T_e_right_bc": t_edge_keV,
            "T_i_right_bc": t_edge_keV,
            "n_e": {0: {0.0: ne_core_m3, 1.0: ne_edge_m3}},
            "n_e_right_bc": ne_edge_m3,
        },
        "numerics": {
            "t_final": dt_steady,
            "fixed_dt": dt_steady,
            "evolve_ion_heat": True,
            "evolve_electron_heat": True,
            "evolve_density": True,
            "evolve_current": False,
            "exact_t_final": True,
            "adaptive_dt": False,
        },
        "plasma_composition": {},
        "geometry": {
            "geometry_type": "circular",
            "R_major": float(inputs.R0),
            "a_minor": float(inputs.a_minor),
            "B_0": float(inputs.B0),
            "elongation_LCFS": float(inputs.kappa),
            "n_rho": n_rho,
        },
        "sources": {
            "generic_heat": {
                "P_total": float(P_total),
                "gaussian_location": float(source_rho),
                "gaussian_width": float(source_width),
                "electron_heat_fraction": 0.5,
            },
        },
        "transport": {
            "model_name": "constant",
            "chi_e": float(chi_e),
            "chi_i": float(chi_i),
            "D_e": float(D_e),
            "V_e": 0.0,
        },
        "solver": {},
        "pedestal": {},
        "time_step_calculator": {"calculator_type": "fixed"},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main module class
# ──────────────────────────────────────────────────────────────────────────────


class ProfilePredictorTorax(TimeIndepModule):
    """Profile predictor using TORAX as a physics-based forward model.

    A neural network maps the 9 equilibrium ``nn_inputs`` to 7 transport and
    source parameters.  TORAX then solves the coupled heat and particle transport
    PDEs for a single large timestep, approximating the quasi-steady-state
    profile shapes.

    **JIT and gradients**: The ``SimulationStepFn`` is built once at ``init``
    time and stored as a static field.  Per ``__call__``, the NN-predicted
    chi / D / source / power values (JAX traced scalars) are injected into a
    pre-built ``RuntimeParamsProvider`` via ``eqx.tree_at``, which patches the
    ``TimeVaryingScalar.value`` leaves without rebuilding any pydantic objects.
    The patched provider is then passed to the already-compiled ``@jax.jit``
    step function.  This makes the module JIT-compilable and end-to-end
    differentiable through chi_e, chi_i, D_e, P_total, source_rho, and
    source_width.

    Non-differentiable inputs: initial density/temperature profiles (built
    from concrete values).  Steady-state profile shapes are dominated by the
    transport coefficients and heating power, not the initial condition.

    Attributes:
        nn: RtdMLP mapping 9 ``nn_inputs`` → 7 raw transport parameters.
        psigrid: Tuple of psi_n values at which output profiles are evaluated.
        dt_steady: Single TORAX timestep used to approximate steady state [s].
        n_rho: Number of radial cells in the TORAX simulation.
        t_edge_keV: Fixed temperature boundary condition at the LCFS [keV].
        _step_fn: Pre-compiled TORAX ``SimulationStepFn`` (static).
        _base_provider: Reference ``RuntimeParamsProvider`` whose leaves are
            patched per call with the NN-predicted values (static).
        _ref_torax_mesh: Pre-built ``Grid1D`` for the normalised rho grid
            (structural, depends only on ``n_rho`` — not on plasma params).
        _rho_hires_norm: Pre-built numpy array of normalised rho on the
            high-resolution grid used by TORAX for poloidal-flux integrals.
    """

    nn: RtdMLP
    psigrid: tuple = eqx.field(static=True)
    dt_steady: float = eqx.field(static=True)
    n_rho: int = eqx.field(static=True)
    t_edge_keV: float = eqx.field(static=True)
    # All four fields below are static: they encode the compiled simulation
    # structure.  Marking them static means JAX treats them as compile-time
    # constants; any change (which never happens after init) triggers recompile.
    _step_fn: Any = eqx.field(static=True)
    _base_provider: Any = eqx.field(static=True)
    _ref_torax_mesh: Any = eqx.field(static=True)
    _rho_hires_norm: Any = eqx.field(static=True)

    def __init__(
        self,
        nn: RtdMLP,
        psigrid: tuple,
        dt_steady: float,
        n_rho: int,
        t_edge_keV: float,
    ) -> None:
        self.nn = nn
        self.psigrid = psigrid
        self.dt_steady = dt_steady
        self.n_rho = n_rho
        self.t_edge_keV = t_edge_keV

        # Build the step function and base provider once.
        ref_config_dict = _make_reference_config(n_rho, dt_steady, t_edge_keV)
        torax_cfg = torax_model_config.ToraxConfig.from_dict(ref_config_dict)
        self._step_fn = make_step_fn(torax_cfg)
        self._base_provider = RuntimeParamsProvider.from_config(torax_cfg)

        # Extract the structural grid objects from the reference geometry.
        # These only depend on n_rho and the hires factor — not on plasma
        # parameters — so they are shared across all per-call geometries.
        ref_geo = circular_geometry.CircularConfig(
            R_major=_REF_R0,
            a_minor=_REF_A,
            B_0=_REF_B0,
            elongation_LCFS=_REF_KAPPA,
            n_rho=n_rho,
        ).build_geometry()
        self._ref_torax_mesh = ref_geo.torax_mesh
        self._rho_hires_norm = np.asarray(ref_geo.rho_hires_norm)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _patch_provider(
        self,
        chi_e: jnp.ndarray,
        chi_i: jnp.ndarray,
        D_e: jnp.ndarray,
        P_total: jnp.ndarray,
        source_rho: jnp.ndarray,
        source_width: jnp.ndarray,
        Ip_A: jnp.ndarray,
    ) -> Any:
        """Patch the pre-built RuntimeParamsProvider with new JAX-traced values.

        Uses ``eqx.tree_at`` to replace the ``TimeVaryingScalar.value`` leaves
        in the base provider — this is a pure JAX pytree operation that works
        inside JIT and preserves the gradient graph.

        All arguments are JAX scalars (shape ``()``).  They are broadcast to
        shape ``(1,)`` to match the ``value`` leaf shape in ``TimeVaryingScalar``.

        Args:
            chi_e: Electron thermal diffusivity [m²/s].
            chi_i: Ion thermal diffusivity [m²/s].
            D_e: Electron particle diffusivity [m²/s].
            P_total: Total heating power [W].
            source_rho: Heat source deposition location [rho_norm].
            source_width: Heat source Gaussian width [rho_norm].
            Ip_A: Plasma current [A].

        Returns:
            A patched ``RuntimeParamsProvider`` with the new values.
        """

        # All TimeVaryingScalar.value leaves have shape (n_times,) = (1,) for
        # time-constant parameters.  Broadcast scalars to match.
        def to1d(x):
            return jnp.expand_dims(x, 0)

        provider = eqx.tree_at(
            lambda p: p.transport_model.chi_e.value,
            self._base_provider,
            to1d(chi_e),
        )
        provider = eqx.tree_at(
            lambda p: p.transport_model.chi_i.value,
            provider,
            to1d(chi_i),
        )
        provider = eqx.tree_at(
            lambda p: p.transport_model.D_e.value,
            provider,
            to1d(D_e),
        )
        provider = eqx.tree_at(
            lambda p: p.sources.generic_heat.P_total.value,
            provider,
            to1d(P_total),
        )
        provider = eqx.tree_at(
            lambda p: p.sources.generic_heat.gaussian_location.value,
            provider,
            to1d(source_rho),
        )
        provider = eqx.tree_at(
            lambda p: p.sources.generic_heat.gaussian_width.value,
            provider,
            to1d(source_width),
        )
        provider = eqx.tree_at(
            lambda p: p.profile_conditions.Ip.value,
            provider,
            to1d(Ip_A),
        )
        return provider

    def _build_geometry_provider(self, inputs: Inputs) -> ConstantGeometryProvider:
        """Build a per-call geometry provider from input plasma parameters.

        Uses :func:`_build_circular_geometry_jax` so that ``R0``, ``a_minor``,
        ``B0``, and ``kappa`` remain JAX traced arrays.  The structural grid
        (``torax_mesh``, ``rho_hires_norm``) is taken from the static reference
        built at init time and shared across all calls.

        Args:
            inputs: Equilibrium parameters.  ``R0``, ``a_minor``, ``B0``, and
                ``kappa`` may be JAX traced scalars.

        Returns:
            ``ConstantGeometryProvider`` wrapping a ``Geometry`` whose array
            fields are JAX-traced w.r.t. the four shape parameters.
        """
        geo = _build_circular_geometry_jax(
            R0=inputs.R0,
            a=inputs.a_minor,
            B0=inputs.B0,
            kappa=inputs.kappa,
            torax_mesh=self._ref_torax_mesh,
            rho_hires_norm=self._rho_hires_norm,
        )
        return ConstantGeometryProvider(geo)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def __call__(
        self,
        inputs: Inputs | xr.Dataset,
        debug: bool = False,
    ) -> Outputs:
        """Predict T_e and n_e profiles via a quasi-steady-state TORAX run.

        The six transport/source parameters (chi_e, chi_i, D_e, P_total,
        source_rho, source_width) and their gradients flow through TORAX's
        compiled PDE solver.  Geometry and initial profile conditions are set
        from concrete input values.

        Args:
            inputs: Either an ``Inputs`` dataclass or an ``xr.Dataset`` with
                the 9 equilibrium variables.
            debug: If True, populate ``Outputs.debug_info``.

        Returns:
            ``Outputs`` with ``te`` [keV] and ``ne`` [10²⁰ m⁻³] on ``psigrid``.

        Raises:
            RuntimeError: If the TORAX simulation exits with a non-zero
                ``SimError``.
        """
        if isinstance(inputs, xr.Dataset):
            inputs = Inputs(
                Ip=float(inputs["Ip_MA"].values),
                B0=float(inputs["B0"].values),
                betan=float(inputs["betan"].values),
                ne20=float(inputs["ne20_edge"].values),
                R0=float(inputs["R0"].values),
                a_minor=float(inputs["a_minor"].values),
                kappa=float(inputs["kappa"].values),
                delta_top=float(inputs["delta_top"].values),
                delta_bot=float(inputs["delta_bot"].values),
                psi=jnp.array(self.psigrid),
            )

        # ── NN forward pass (traced) ──────────────────────────────────────
        nn_raw = self.nn(inputs.nn_inputs)

        # Decode: all remain as JAX arrays (no float() conversions).
        chi_e = jnp.clip(jnp.exp(nn_raw[0]), _CHI_MIN, _CHI_MAX)
        chi_i = jnp.clip(chi_e * jnp.exp(nn_raw[1]), _CHI_MIN, _CHI_MAX)
        source_rho = jnp.clip(jax.nn.sigmoid(nn_raw[2]), _SRC_RHO_MIN, _SRC_RHO_MAX)
        source_width = jnp.clip(jnp.exp(nn_raw[3]), _SRC_WIDTH_MIN, _SRC_WIDTH_MAX)
        D_e = jnp.clip(jnp.exp(nn_raw[4]), _D_E_MIN, _D_E_MAX)
        ne_peaking = jnp.clip(jnp.exp(nn_raw[5]), _NE_PEAK_MIN, _NE_PEAK_MAX)
        P_scale = jnp.clip(jnp.exp(nn_raw[6]), _P_SCALE_MIN, _P_SCALE_MAX)

        P_physics = _estimate_P_total(inputs)
        P_total = jnp.clip(P_physics * P_scale, _P_TOTAL_MIN, _P_TOTAL_MAX)
        Ip_A = inputs.Ip * 1e6  # [A], JAX scalar

        # ── Build geometry (Python / numpy, not traced) ──────────────────
        # Concrete float conversions are intentional here: circular geometry
        # uses numpy and cannot be JIT-traced.  Gradients w.r.t. R0, a_minor,
        # B0, kappa are not available.
        geo_provider = self._build_geometry_provider(inputs)

        # ── Patch runtime provider leaves with traced NN values ───────────
        # eqx.tree_at replaces TimeVaryingScalar.value leaves in the static
        # base_provider with the traced NN-predicted scalars.  This operation
        # is pure JAX and preserves the gradient graph.
        patched_provider = self._patch_provider(
            chi_e=chi_e,
            chi_i=chi_i,
            D_e=D_e,
            P_total=P_total,
            source_rho=source_rho,
            source_width=source_width,
            Ip_A=Ip_A,
        )

        # Update the initial density profile with the NN-predicted peaking.
        # ne_peaking is used only as an initial condition (concrete here; the
        # steady-state n_e shape is determined primarily by D_e).
        ne_edge_m3 = float(inputs.ne20) * 1e20
        ne_core_m3 = ne_edge_m3 * float(ne_peaking)
        T_core = max(float(inputs.te_approx), self.t_edge_keV + 0.1)

        # Patch the initial profile conditions with per-call concrete values.
        # We use eqx.tree_at for the scalars; profile shapes are TimeVaryingArray
        # (2D) and need a different update path — for now we rebuild just the
        # n_e_right_bc scalar which is a TimeVaryingScalar.
        patched_provider = eqx.tree_at(
            lambda p: p.profile_conditions.n_e_right_bc.value,
            patched_provider,
            jnp.array([ne_edge_m3]),
        )

        # ── JIT-compiled initial state ────────────────────────────────────
        initial_state, initial_ppo = get_initial_state_and_post_processed_outputs(
            step_fn=self._step_fn,
            runtime_params_overrides=patched_provider,
            geometry_overrides=geo_provider,
        )

        # ── JIT-compiled single transport step ────────────────────────────
        final_state, _ = self._step_fn(
            input_state=initial_state,
            previous_post_processed_outputs=initial_ppo,
            runtime_params_overrides=patched_provider,
            geo_overrides=geo_provider,
        )

        # ── Check simulation health ───────────────────────────────────────
        sim_error = final_state.check_for_errors()
        if sim_error != torax.SimError.NO_ERROR:
            raise RuntimeError(
                f"TORAX simulation failed with {sim_error}. "
                f"Decoded params: chi_e={float(chi_e):.3f} m²/s, "
                f"chi_i={float(chi_i):.3f} m²/s, P_total={float(P_total):.3e} W"
            )

        # ── Extract profiles (JAX arrays — gradient-traceable) ────────────
        # Use the cell_centers from the structural reference mesh (concrete
        # numpy), not from the per-call JAX geometry, because _rho_to_psigrid
        # uses numpy interp and requires a concrete array.
        rho_norm_grid = np.asarray(self._ref_torax_mesh.cell_centers)
        te_rho = np.asarray(final_state.core_profiles.T_e.value)  # [keV]
        ne_rho = np.asarray(final_state.core_profiles.n_e.value) / 1e20  # [10²⁰ m⁻³]

        # ── Map to psi_n output grid ──────────────────────────────────────
        te_psi, ne_psi = _rho_to_psigrid(te_rho, ne_rho, rho_norm_grid, self.psigrid)

        debug_info: dict | None = None
        if debug:
            debug_info = {
                "chi_e": float(chi_e),
                "chi_i": float(chi_i),
                "source_rho": float(source_rho),
                "source_width": float(source_width),
                "D_e": float(D_e),
                "ne_peaking": float(ne_peaking),
                "P_total": float(P_total),
                "P_physics": float(P_physics),
                "P_scale": float(P_scale),
                "T_core_init": T_core,
                "ne_core_init_m3": ne_core_m3,
            }

        psi_n_list = list(self.psigrid)
        return Outputs(
            te=xr.DataArray(te_psi, dims=("psi_n",), coords={"psi_n": psi_n_list}),
            ne=xr.DataArray(ne_psi, dims=("psi_n",), coords={"psi_n": psi_n_list}),
            debug_info=debug_info,
        )

    def get_transport_params(self, inputs: Inputs) -> dict[str, float]:
        """Return the decoded NN transport parameters without running TORAX."""
        nn_raw = self.nn(inputs.nn_inputs)
        chi_e = jnp.clip(jnp.exp(nn_raw[0]), _CHI_MIN, _CHI_MAX)
        chi_i = jnp.clip(chi_e * jnp.exp(nn_raw[1]), _CHI_MIN, _CHI_MAX)
        source_rho = jnp.clip(jax.nn.sigmoid(nn_raw[2]), _SRC_RHO_MIN, _SRC_RHO_MAX)
        source_wid = jnp.clip(jnp.exp(nn_raw[3]), _SRC_WIDTH_MIN, _SRC_WIDTH_MAX)
        D_e = jnp.clip(jnp.exp(nn_raw[4]), _D_E_MIN, _D_E_MAX)
        ne_peaking = jnp.clip(jnp.exp(nn_raw[5]), _NE_PEAK_MIN, _NE_PEAK_MAX)
        P_scale = jnp.clip(jnp.exp(nn_raw[6]), _P_SCALE_MIN, _P_SCALE_MAX)
        P_physics = _estimate_P_total(inputs)
        P_total = jnp.clip(P_physics * P_scale, _P_TOTAL_MIN, _P_TOTAL_MAX)
        return {
            "chi_e": float(chi_e),
            "chi_i": float(chi_i),
            "source_rho": float(source_rho),
            "source_width": float(source_wid),
            "D_e": float(D_e),
            "ne_peaking": float(ne_peaking),
            "P_scale": float(P_scale),
            "P_physics": float(P_physics),
            "P_total": float(P_total),
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def init(
        cls,
        psigrid: np.ndarray,
        nn_width: int = 32,
        nn_depth: int = 3,
        dt_steady: float = 5.0,
        n_rho: int = 25,
        t_edge_keV: float = 0.2,
        prng_seed: int = 42,
    ) -> ProfilePredictorTorax:
        """Initialise a ``ProfilePredictorTorax`` with a fresh MLP.

        This call is slow the first time (compiles the TORAX step function) but
        subsequent ``__call__`` invocations reuse the compiled step.

        Args:
            psigrid: 1-D array of psi_n values for the output profile grid.
            nn_width: Width (neurons per layer) of the MLP.
            nn_depth: Number of hidden layers.
            dt_steady: Single TORAX timestep [s] for the steady-state
                approximation (should be >> tau_E, typically 1-10 s).
            n_rho: Number of radial cells in the TORAX simulation grid.
            t_edge_keV: Fixed edge temperature boundary condition [keV].
            prng_seed: Seed for JAX PRNG weight initialisation.

        Returns:
            A freshly initialised ``ProfilePredictorTorax``.
        """
        psigrid_tuple = tuple(np.asarray(psigrid).tolist())
        nn = RtdMLP(
            in_size=9,
            out_size=_NN_OUT_SIZE,
            width_size=nn_width,
            depth=nn_depth,
            activation=Activation.RELU,
            final_activation=Activation.IDENTITY,
            key=jax.random.PRNGKey(prng_seed),
        )
        return cls(
            nn=nn,
            psigrid=psigrid_tuple,
            dt_steady=dt_steady,
            n_rho=n_rho,
            t_edge_keV=t_edge_keV,
        )
