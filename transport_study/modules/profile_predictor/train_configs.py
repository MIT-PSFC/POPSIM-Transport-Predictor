import copy
from typing import Any

from transport_study.modules.profile_predictor.module import (
    ShapeType,
)
from transport_study.modules.profile_predictor.trb import (
    ProfilePredictorTRB,
)
from transport_study.orchestration.organize_data import PROFILE_TARGET_VARS

PROFILE_PREDICTOR_SHAPE_INIT_CONFIG = {
    "project": "profile_predictor_shape_init",
    "train_run_builder": ProfilePredictorTRB,
    "max_epochs": 2,
    "epochs_per_val": 2,
    "checkpoint_dir": None,
    "dataloader_config": {
        "input_vars": [
            "Ip_MA",
            "B0",
            "betan",
            "ne20_line_avg",
            "R0",
            "a_minor",
            "kappa",
            "delta_top",
            "delta_bot",
        ],
        "target_vars": [*PROFILE_TARGET_VARS, "ds_source_idx"],
        "extra_vars": ["Te_shape", "ne_shape"],
    },
    "model_init_config": {
        "model_type": "shape_init",
        "shape_type": ShapeType.CONVEX_COMBINATION.value,
        "te_shape_var": "Te_shape",
        "ne_shape_var": "ne_shape",
        "n_shapes": 3,
        "nn_depth": 2,
        "nn_width": 16,
        "in_size": 10,  # Dimensionless nn_inputs derived from the raw input_vars, includes log(nu_star)
        "softmax_temp": 1,
        "prng_seed": 42,
    },
    "loss_config": {
        "huber_delta": 0.5,
    },
    "optimizer_config": {
        "lr0": 3e-3,
        "transition_steps": 500,
        "decay_rate": 0.5,
        "lrf": 5e-4,
        "weight_decay": 2e-4,
    },
    "trainable_getter_config": {
        "freeze_shapes": True,
    },
}

# Transport blocks for the TORAX transport models the torax profile
# predictor can be benchmarked with. Values are placeholders that must pass
# pydantic validation; the NN-driven entries are overridden at call time.
TORAX_TRANSPORT_BLOCKS = {
    "constant": {
        # Prescribed (flat) transport coefficients, all predicted by the NN.
        "model_name": "constant",
        "chi_i": 1.0,  # Predicted by NN
        "chi_e": 1.0,  # Predicted by NN
        "D_e": 1.0,  # Predicted by NN
        "V_e": -0.33,  # Predicted by NN
    },
    "cgm": {
        # Critical Gradient Model: TORAX computes the critical ion temperature
        # gradient from the evolving state and geometry (known inputs); the NN
        # predicts the dimensionless free parameters.
        "model_name": "CGM",
        "alpha": 2.0,  # Predicted by NN
        "chi_stiff": 2.0,  # Predicted by NN
        "chi_e_i_ratio": 2.0,  # Predicted by NN
        "chi_D_ratio": 5.0,  # Predicted by NN
        "VR_D_ratio": 0.0,  # Predicted by NN
        # Subcritical CGM drops chi to chi_min. The TORAX default of 0.05 m^2/s
        # is near-zero transport, so ohmic heating in low-density plasmas runs
        # away within a 20 ms step and NaNs the solver.
        # A floor of 0.3 keeps some background transport.
        "chi_min": 0.3,
        "chi_max": 50.0,
        "D_e_min": 0.1,
    },
    "gyrobohm": {
        # Bohm-GyroBohm model: TORAX computes the Bohm and GyroBohm chi terms
        # from the evolving state and geometry; the NN predicts one multiplier
        # per term (applied to both species) plus the particle transport
        # weighting constants. The coeff prefactors stay at TORAX defaults.
        "model_name": "bohm-gyrobohm",
        "chi_e_bohm_multiplier": 1.0,  # Predicted by NN
        "chi_i_bohm_multiplier": 1.0,  # Predicted by NN
        "chi_e_gyrobohm_multiplier": 1.0,  # Predicted by NN
        "chi_i_gyrobohm_multiplier": 1.0,  # Predicted by NN
        "D_face_c1": 1.0,  # Predicted by NN
        "D_face_c2": 0.3,  # Predicted by NN
        "V_face_coeff": -0.1,  # Predicted by NN
        # Same stability clipping as the cgm, floor background transport
        # so cold low-density samples cannot run away, cap the stiff side.
        "chi_min": 0.3,
        "chi_max": 50.0,
        "D_e_min": 0.1,
    },
    "qlknn": {
        # QLKNN surrogate of QuaLiKiz turbulent transport. model_path and
        # qlknn_model_name left unset so fusion_surrogates loads its bundled
        # qlknn_7_11_v1 weights.
        "model_name": "qlknn",
        "ITG_flux_ratio_correction": 1.0,  # Predicted by NN
        "ETG_correction_factor": 0.333,  # Predicted by NN
        "collisionality_multiplier": 1.0,  # Predicted by NN
        # QLKNN is data-driven itself
        # Clip inputs so out-of-range samples saturate instead of extrapolating
        "clip_inputs": True,
        "clip_margin": 0.95,
        "DV_effective": False,
        "smoothing_width": 0.1,
        # Same stability clipping as cgm and gyrobohm blocks: subcritical QLKNN
        # drops chi toward zero and low-density ohmic samples run away.
        "chi_min": 0.3,
        "chi_max": 50.0,
        "D_e_min": 0.1,
    },
}

# TORAX config skeleton shared by the torax-backed profile predictor and, with
# a one-step numerics override, the torax-backed transport predictor, whose
# builder lives in the transport_predictor train_configs module. The
# "transport" block is filled per transport model from TORAX_TRANSPORT_BLOCKS.
TORAX_CONFIG_BASE: dict[str, Any] = {
    "profile_conditions": {
        "Ip": 9999,  # Overridden by dataloader input
        # Edge BCs predicted by NN as fractions of te_approx and ne20_line_avg.
        "T_i_right_bc": 0.2,  # [keV] Predicted by NN
        "T_e_right_bc": 0.2,  # [keV] Predicted by NN
        "n_e_right_bc": 0.5e20,  # [m^-3] Predicted by NN
        # Placeholder initial profiles; overridden per sample with
        # parabolic inits scaled to te_approx / ne20_line_avg and
        # continuous with the NN edge BCs (see build_provider_and_geo)
        "T_i": {0: {0: 0.3, 1: 0.2}},
        "T_e": {0: {0: 0.3, 1: 0.2}},
        "n_e": {0: {0: 1e20, 1: 0.5e20}},
        "normalize_n_e_to_nbar": False,
        # Initialize psi from Ip and geometry via the current_profile_nu formula.
        # Same as the legacy fallback for circular geometry, but explicit to
        # silence the TORAX deprecation warning.
        "initial_psi_mode": "j",
    },
    "numerics": {
        "t_initial": 0.0,
        "t_final": 0.1,  # Give it ~100 ms to relax, on order of energy confinement time
        # Linear theta solver is implicit / unconditionally stable, so we
        # can take large fixed steps to reach steady state cheaply
        "fixed_dt": 2e-2,
        "min_dt": 1e-3,
        # dt never changes with the fixed time-step calculator, so the
        # adaptive retry loop is pure overhead (1.4x, bit-identical results)
        "adaptive_dt": False,
        "evolve_ion_heat": True,
        "evolve_electron_heat": True,
        "evolve_current": True,
        "evolve_density": True,
    },
    "plasma_composition": {
        "main_ion": {"D": 1.0},  # Assuming DD and minor impurities
        "Z_eff": 1.1,
    },
    "geometry": {
        "geometry_type": "circular",
        "R_major": 9999,  # Overridden by dataloader input
        "a_minor": 3000,  # Overridden by dataloader input (must be less than R_major)
        "B_0": 9999,  # Overridden by dataloader input
        "elongation_LCFS": 9999,  # Overridden by dataloader input
        # Internal solver mesh, up from the TORAX default of 25 to
        # halve the piecewise-linear gradient staircase in the output
        # interpolation. The 51-point output rhogrid shared by all
        # model families is unaffected.
        "n_rho": 50,
    },
    # "transport" block filled per transport model from TORAX_TRANSPORT_BLOCKS
    "sources": {
        "ei_exchange": {},
        "bremsstrahlung": {},
        "cyclotron_radiation": {},
        "ohmic": {},
        "gas_puff": {"S_total": 9999},  # Predicted by NN
        # NN-inferred auxiliary heating. All entries except
        # absorption_fraction are per-sample overridden by the profile
        # module's sources network (absorption is fixed there, degenerate
        # with the NN-predicted P_total. The transport module overrides
        # absorption_fraction per sample too, since its P_total is the
        # measured input). Placeholders only need to pass pydantic validation
        "generic_heat": {
            "P_total": 1.0e6,  # Predicted by NN
            "gaussian_location": 0.3,  # Predicted by NN
            "gaussian_width": 0.25,  # Predicted by NN
            "electron_heat_fraction": 0.6,  # Predicted by NN
            "absorption_fraction": 0.9,
        },
        "generic_current": {},
    },
    "solver": {
        # INERT with the solver this config selects. TORAX only applies the
        # Pereverzev-Corrigan terms in the NONLINEAR (Newton-Raphson) solver,
        # to build its optional initial guess from a linear solve - see the
        # use_pereverzev docstring in torax._src.solver.pydantic_model. No
        # solver_type is set here, so this is a bare LinearThetaMethod and the
        # flag does nothing: every chi_face_*_pereverzev / d_face_el_pereverzev
        # / v_face_el_pereverzev term measured out as identically zero on both
        # healthy and diverging samples (2026-07-26 trace probe).
        # Kept True only so the intent survives if the solver is ever switched
        # to newton_raphson, which is what it would take to actually get the
        # stabilization the gradient-dependent models (stiff CGM, BgB chi
        # driven by the evolving gradients) would otherwise want at large
        # fixed steps. Do NOT rely on this flag for stability as written.
        "use_pereverzev": True,
        "use_predictor_corrector": True,
        # Picard iterations run n_corrector_steps + 1 times with no early exit.
        # Benchmarked at the 20ms dt above: 1/2/4/8 corrector steps all
        # converge to mean-best val losses within the seed spread, so extra
        # iterations buy nothing. 1 is the minimum TORAX accepts
        "n_corrector_steps": 1,
    },
    "time_step_calculator": {"calculator_type": "fixed"},
    "neoclassical": {},
    "pedestal": {},
}


_PROFILE_PREDICTOR_TORAX_CONFIG_BASE: dict[str, Any] = {
    "project": "profile_predictor_torax",
    "train_run_builder": ProfilePredictorTRB,
    "max_epochs": 10,
    "epochs_per_val": 2,
    "checkpoint_dir": None,
    "dataloader_config": {
        "input_vars": [
            "Ip_MA",
            "B0",
            "betan",
            "ne20_line_avg",
            "R0",
            "a_minor",
            "kappa",
            "delta_top",
            "delta_bot",
        ],
        "target_vars": [*PROFILE_TARGET_VARS, "ds_source_idx"],
    },
    "model_init_config": {
        "model_type": "torax-cgm",  # Overridden per transport model by the builder below
        "nn_depth": 2,
        "nn_width": 16,
        # The builder deep-copies the whole config, so the shared skeleton
        # reference here is never mutated
        "torax_config": TORAX_CONFIG_BASE,
        "prng_seed": 42,
        # Per-sample geometry builder: "circular" or "miller" (shaped,
        # uses delta_top/delta_bot with delta ~ rho_norm**delta_exponent)
        "geometry_builder": "circular",
        "delta_exponent": 2.0,
        # Relaxation window overrides, None keeps the torax_config numerics values.
        # Top-level keys so wandb sweeps can search them like nn_width.
        # n_solver_steps derives fixed_dt = t_final / n_solver_steps so sweeps
        # can widen the horizon without multiplying per-sample solver cost.
        # Mutually exclusive with an explicit fixed_dt
        "t_final": None,
        "fixed_dt": None,
        "n_solver_steps": None,
    },
    "loss_config": {
        "huber_delta": 0.5,
    },
    "optimizer_config": {
        "lr0": 1e-3,
        "transition_steps": 200,
        "decay_rate": 0.5,
        "lrf": 1e-4,
        "weight_decay": 1e-4,
    },
}


def make_profile_predictor_torax_config(transport_model: str, geometry_builder: str = "circular", delta_exponent: float = 2.0) -> dict:
    """Train config for the torax profile predictor with the given transport model.

    transport_model is one of "constant", "cgm", "gyrobohm", "qlknn"
    the corresponding model_type is "torax-<transport_model>".
    """
    if transport_model not in TORAX_TRANSPORT_BLOCKS:
        raise ValueError(f"Unknown transport model '{transport_model}', valid: {sorted(TORAX_TRANSPORT_BLOCKS)}")
    cfg = copy.deepcopy(_PROFILE_PREDICTOR_TORAX_CONFIG_BASE)
    cfg["project"] = f"profile_predictor_torax_{transport_model}"
    cfg["model_init_config"]["model_type"] = f"torax-{transport_model}"
    cfg["model_init_config"]["torax_config"]["transport"] = copy.deepcopy(TORAX_TRANSPORT_BLOCKS[transport_model])
    cfg["model_init_config"]["geometry_builder"] = geometry_builder
    cfg["model_init_config"]["delta_exponent"] = delta_exponent
    return cfg


PROFILE_PREDICTOR_TORAX_CONFIGS = {
    transport_model: make_profile_predictor_torax_config(transport_model) for transport_model in TORAX_TRANSPORT_BLOCKS
}

PROFILE_PREDICTOR_DIRECT_POINTS_CONFIG = {
    "project": "profile_predictor_direct_points",
    "train_run_builder": ProfilePredictorTRB,
    "max_epochs": 2,
    "epochs_per_val": 2,
    "checkpoint_dir": None,
    "dataloader_config": {
        "input_vars": [
            "Ip_MA",
            "B0",
            "betan",
            "ne20_line_avg",
            "R0",
            "a_minor",
            "kappa",
            "delta_top",
            "delta_bot",
        ],
        "target_vars": [*PROFILE_TARGET_VARS, "ds_source_idx"],
    },
    "model_init_config": {
        "model_type": "direct_points",
        "shape_type": ShapeType.CONVEX_COMBINATION.value,
        "n_points": 13,
        "nn_depth": 3,
        "nn_width": 20,
        "prng_seed": 42,
    },
    "loss_config": {
        "huber_delta": 0.5,
    },
    "optimizer_config": {
        "lr0": 3e-3,
        "transition_steps": 500,
        "decay_rate": 0.5,
        "lrf": 5e-4,
        "weight_decay": 2e-4,
    },
    "trainable_getter_config": {
        "freeze_shapes": True,
    },
}
