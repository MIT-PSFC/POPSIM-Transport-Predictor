import copy
from typing import Any

from transport_study.modules.profile_predictor.module import (
    NN_INPUT_SOURCE_VARS,
    ShapeType,
)
from transport_study.modules.profile_predictor.trb import (
    ProfilePredictorTRB,
    relaxation_numerics,
)
from transport_study.orchestration.organize_data import PROFILE_TARGET_VARS

# Solver steps of the profile relaxation without a tuned config
DEFAULT_N_SOLVER_STEPS = 8

PROFILE_PREDICTOR_SHAPE_INIT_CONFIG = {
    "project": "profile_predictor_shape_init",
    "train_run_builder": ProfilePredictorTRB,
    "max_epochs": 2,
    "epochs_per_val": 2,
    "checkpoint_dir": None,
    "dataloader_config": {
        "input_vars": list(NN_INPUT_SOURCE_VARS),
        "target_vars": [*PROFILE_TARGET_VARS, "ds_source_idx"],
        "extra_vars": ["t_e_shape", "n_e_shape"],
    },
    "model_init_config": {
        "model_type": "shape_init",
        "shape_type": ShapeType.CONVEX_COMBINATION.value,
        "te_shape_var": "t_e_shape",
        "ne_shape_var": "n_e_shape",
        "n_shapes": 3,
        "nn_depth": 2,
        "nn_width": 16,
        "in_size": 10,  # Dimensionless nn_inputs derived from the raw input_vars, includes log(nu_star)
        "prng_seed": 42,
    },
    "loss_config": {
        "huber_delta": 0.5,
    },
    "optimizer_config": {
        "lr0": 3e-3,
        "transition_steps": 1661,
        "decay_rate": 0.1,
        "lrf_frac": 0.17,
        "weight_decay": 2e-4,
    },
    "trainable_getter_config": {
        "freeze_shapes": True,
    },
}

# Top-level TORAX transport clipping shared by the gyrobohm and qlknn blocks.
# Flat-gradient BGB and subcritical QLKNN drop chi to chi_min.
# The TORAX default of 0.05 m^2/s is near-zero transport,
# so ohmic heating in low-density plasmas runs away within a 20 ms step and NaNs the solver.
# A floor of 0.3 keeps some background transport, the cap bounds the stiff side.
_STABILITY_CLIPPING = {
    "chi_min": 0.3,
    "chi_max": 50.0,
    "D_e_min": 0.1,
}

# Transport blocks for the TORAX transport models the torax predictors can be benchmarked with.
# Each block holds exactly one core transport model, keyed by our transport_model name,
# which is the key the NN overrides address (see transport_provider_mapping).
# Values are placeholders that must pass pydantic validation,
# the NN-driven entries are overridden per sample.
TORAX_TRANSPORT_BLOCKS = {
    "constant": {
        "core_transport_models": {
            "constant": {
                # Prescribed (flat) transport coefficients, all predicted by the NN
                "model_name": "prescribed",
                "chi_i": 1.0,  # Predicted by NN
                "chi_e": 1.0,  # Predicted by NN
                "D_e": 1.0,  # Predicted by NN
                "V_e": -0.33,  # Predicted by NN
            },
        },
        # No stability clipping here:
        # the NN bounds chi and D at 0.1 - 10 chi_ref of the device (bound_transport_coefficients),
        # the TORAX defaults (chi_min 0.05, chi_max 100, D_e_min 0.05) bind only at the bottom of that range on C-Mod and TCV.
    },
    "gyrobohm": {
        "core_transport_models": {
            "gyrobohm": {
                # Bohm-GyroBohm model:
                # TORAX computes the Bohm and GyroBohm chi terms from the evolving state and geometry,
                # the NN predicts one multiplier per term (applied to both species)
                # plus the particle transport weighting constants.
                # The coeff prefactors stay at TORAX defaults.
                "model_name": "bohm-gyrobohm",
                "chi_e_bohm_multiplier": 1.0,  # Predicted by NN
                "chi_i_bohm_multiplier": 1.0,  # Predicted by NN
                "chi_e_gyrobohm_multiplier": 1.0,  # Predicted by NN
                "chi_i_gyrobohm_multiplier": 1.0,  # Predicted by NN
                "D_face_c1": 1.0,  # Predicted by NN
                "D_face_c2": 0.3,  # Predicted by NN
                "V_face_coeff": -0.1,  # Predicted by NN
            },
        },
        **_STABILITY_CLIPPING,
    },
    "qlknn": {
        "core_transport_models": {
            "qlknn": {
                # QLKNN surrogate of QuaLiKiz turbulent transport.
                # model_path and qlknn_model_name are left unset,
                # so fusion_surrogates loads its bundled qlknn_7_11_v1 weights.
                "model_name": "qlknn",
                "ITG_flux_ratio_correction": 1.0,  # Predicted by NN
                "ETG_correction_factor": 0.333,  # Predicted by NN
                "collisionality_multiplier": 1.0,  # Predicted by NN
                # Clip inputs so out-of-range samples saturate instead of extrapolating
                "clip_inputs": True,
                "clip_margin": 0.95,
                "DV_effective": False,
            },
        },
        # Gaussian smoothing of the stiff QLKNN outputs
        "smoothing_width": 0.1,
        **_STABILITY_CLIPPING,
    },
}

# TORAX config skeleton shared by the torax-backed profile predictor and, with
# a one-step numerics override, the torax-backed transport predictor, whose
# builder lives in the transport_predictor train_configs module.
# The "transport" block is filled per transport model from TORAX_TRANSPORT_BLOCKS.
TORAX_CONFIG_BASE: dict[str, Any] = {
    "profile_conditions": {
        "Ip": 9999,  # Overridden by dataloader input
        # Edge BCs predicted by NN as fractions of te_approx and n_e_line_average_1e20.
        "T_i_right_bc": 0.2,  # [keV] Predicted by NN
        "T_e_right_bc": 0.2,  # [keV] Predicted by NN
        "n_e_right_bc": 0.5e20,  # [m^-3] Predicted by NN
        # Placeholder initial profiles.
        # Each sample overrides them with parabolic inits scaled to te_approx / n_e_line_average_1e20,
        # continuous with the NN edge BCs (see build_provider_and_geo).
        "T_i": {0: {0: 0.3, 1: 0.2}},
        "T_e": {0: {0: 0.3, 1: 0.2}},
        "n_e": {0: {0: 1e20, 1: 0.5e20}},
        "normalize_n_e_to_nbar": False,
        # Initialize psi from Ip and geometry via the current_profile_nu formula.
        # Same as the legacy fallback for circular geometry, but explicit to
        # silence the TORAX deprecation warning.
        "initial_psi_mode": "j",
        # No non-inductive current source and bootstrap off, so the formula current is the total current.
        # This skips TORAX's psi and source iteration, which the transport rebuild runs every step
        "initial_j_is_total_current": True,
    },
    "numerics": {
        "t_initial": 0.0,
        # With the Pereverzev linear step below, every step makes the same progress at any dt,
        # so the relaxation is a fixed number of damped steps from the initial profiles, not a steady state.
        # The TRB sets the window from n_solver_steps at the fixed RELAXATION_DT_S (trb.relaxation_numerics)
        **relaxation_numerics(DEFAULT_N_SOLVER_STEPS),
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
        # Sets the impurity density (TORAX's default neon impurity) and so the dilution,
        # bremsstrahlung, Spitzer resistivity and the collisionality the transport models see
        "Z_eff": 1.5,
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
    # No cyclotron radiation, it is negligible at these fields and temperatures (measured no output change).
    # No generic_current either, TORAX's default drives 20 percent of Ip at rho 0.4 and reverses the shear
    "sources": {
        "ei_exchange": {},
        "bremsstrahlung": {},
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
    },
    # The profile relaxation solver, the transport predictor replaces it per transport model.
    "solver": {
        # Pereverzev at the TORAX defaults (chi 30, D 15 m^2/s) damps every shape change by ~dt chi / a^2 per step.
        # That keeps the lagged linear step from oscillating, but it pins the progress per step,
        # so the relaxation output is n_solver_steps damped steps from the parabolic initial profiles
        "use_pereverzev": True,
        # One linearized solve per step: transport coefficients, sources and the transient n_e are taken at the old state
        "use_predictor_corrector": False,
        # Backward (implicit) Euler.
        # Forward Euler (0.0) violates the diffusion CFL bound dt <= dx^2 / (2 chi) by orders of magnitude here.
        "theta_implicit": 1.0,
    },
    "time_step_calculator": {"calculator_type": "fixed"},
    # Kim poloidal velocity only feeds the ExB shear of rotation-enabled QuaLiKiz-family models, and rotation is off
    "neoclassical": {"poloidal_velocity": {"model_name": "zeros"}},
    "pedestal": {},
}


_PROFILE_PREDICTOR_TORAX_CONFIG_BASE: dict[str, Any] = {
    "project": "profile_predictor_torax",
    "train_run_builder": ProfilePredictorTRB,
    "max_epochs": 10,
    "epochs_per_val": 2,
    "checkpoint_dir": None,
    "dataloader_config": {
        "input_vars": list(NN_INPUT_SOURCE_VARS),
        "target_vars": [*PROFILE_TARGET_VARS, "ds_source_idx"],
    },
    "model_init_config": {
        "model_type": "torax-gyrobohm",  # Overridden per transport model by the builder below
        "nn_depth": 2,
        "nn_width": 16,
        # The builder deep-copies the whole config, so the shared skeleton
        # reference here is never mutated
        "torax_config": TORAX_CONFIG_BASE,
        "prng_seed": 42,
        # Per-sample geometry builder: "circular" or "miller" (shaped,
        # uses triangularity_upper/triangularity_lower with delta ~ rho_norm**delta_exponent)
        "geometry_builder": "circular",
        "delta_exponent": 2.0,
        # Relaxation length in solver steps of RELAXATION_DT_S, the only window knob.
        # Top-level so wandb sweeps can search it like nn_width
        "n_solver_steps": DEFAULT_N_SOLVER_STEPS,
    },
    "loss_config": {
        "huber_delta": 0.5,
    },
    "optimizer_config": {
        "lr0": 1e-3,
        "transition_steps": 664,
        "decay_rate": 0.1,
        "lrf_frac": 0.1,
        "weight_decay": 1e-4,
    },
}


def make_profile_predictor_torax_config(
    transport_model: str,
    geometry_builder: str = "circular",
    delta_exponent: float = 2.0,
) -> dict:
    """Train config for the torax profile predictor with the given transport model.

    transport_model is one of "constant", "gyrobohm", "qlknn"
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
