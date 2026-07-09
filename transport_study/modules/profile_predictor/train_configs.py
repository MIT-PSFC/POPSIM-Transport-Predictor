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

# Transport blocks for the three TORAX transport models the torax profile
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
    },
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
        "torax_config": {
            "profile_conditions": {
                "Ip": 9999,  # Overridden by dataloader input
                # Edge BCs predicted by NN as fractions of te_approx and ne20_line_avg.
                "T_i_right_bc": 0.2,  # [keV] Predicted by NN
                "T_e_right_bc": 0.2,  # [keV] Predicted by NN
                "n_e_right_bc": 0.5e20,  # [m^-3] Predicted by NN
                # Near-flat initial profiles; will relax up under ohmic heating / NN transport.
                "T_i": {0: {0: 0.3, 1: 0.2}},
                "T_e": {0: {0: 0.3, 1: 0.2}},
                "n_e": {0: {0: 1e20, 1: 0.5e20}},
                "normalize_n_e_to_nbar": True,
                "nbar": 99,  # Overridden by dataloader input
                "n_e_nbar_is_fGW": True,
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
            },
            # "transport" block filled per transport model from TORAX_TRANSPORT_BLOCKS
            "sources": {
                "ei_exchange": {},
                "bremsstrahlung": {},
                "cyclotron_radiation": {},
                "ohmic": {},
                "gas_puff": {"S_total": 9999},  # Predicted by NN
                "generic_current": {},
            },
            "solver": {
                # Gradient-dependent transport (CGM is stiff, BgB chi depends on the
                # evolving gradients); the Pereverzev-Corrigan terms keep the linear
                # theta solver stable at large fixed steps. Harmless for the
                # constant model.
                "use_pereverzev": True,
                "use_predictor_corrector": True,
                # Picard iterations run n_corrector_steps + 1 times with no early exit.
                # TORAX default of 10 is overkill at 10ms dt, but the 20ms
                # dt above needs headroom: benchmarked on the cgm base case,
                # 8 matched baseline val loss per epoch at 3.2x overall speedup,
                # while 4 was 5.3x but converged to visibly worse loss per epoch.
                "n_corrector_steps": 8,
            },
            "time_step_calculator": {"calculator_type": "fixed"},
            "neoclassical": {},
            "pedestal": {},
        },
        "prng_seed": 42,
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


def make_profile_predictor_torax_config(transport_model: str) -> dict:
    """Train config for the torax profile predictor with the given transport model.

    transport_model is one of "constant", "cgm", "gyrobohm"
    the corresponding model_type is "torax-<transport_model>".
    """
    if transport_model not in TORAX_TRANSPORT_BLOCKS:
        raise ValueError(f"Unknown transport model '{transport_model}', valid: {sorted(TORAX_TRANSPORT_BLOCKS)}")
    cfg = copy.deepcopy(_PROFILE_PREDICTOR_TORAX_CONFIG_BASE)
    cfg["project"] = f"profile_predictor_torax_{transport_model}"
    cfg["model_init_config"]["model_type"] = f"torax-{transport_model}"
    cfg["model_init_config"]["torax_config"]["transport"] = copy.deepcopy(TORAX_TRANSPORT_BLOCKS[transport_model])
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
