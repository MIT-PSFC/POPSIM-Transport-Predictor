from transport_study.modules.profile_predictor.module import (
    ShapeType,
)
from transport_study.modules.profile_predictor.trb import (
    ProfilePredictorTRB,
)

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
        "target_vars": ["ne20_psi", "Te_keV_psi"],
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

PROFILE_PREDICTOR_TORAX_CONFIG = {
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
        "target_vars": ["ne20_psi", "Te_keV_psi"],
    },
    "model_init_config": {
        "model_type": "torax",
        "nn_depth": 2,
        "nn_width": 16,
        "torax_config": {
            "profile_conditions": {
                "Ip": 9999,  # Overridden by dataloader input
                # Mild edge BCs to avoid huge initial gradient at LCFS (was crashing solver).
                "T_i_right_bc": 0.2,  # [keV]
                "T_e_right_bc": 0.2,  # [keV]
                # Near-flat initial profiles; will relax up under ohmic heating / NN transport.
                "T_i": {0: {0: 0.3, 1: 0.2}},
                "T_e": {0: {0: 0.3, 1: 0.2}},
                "n_e": {0: {0: 1e20, 1: 0.5e20}},
                "normalize_n_e_to_nbar": True,
                "nbar": 99,  # Overridden by dataloader input
                "n_e_nbar_is_fGW": True,
                "n_e_right_bc": 0.5e20,
                # Initialize psi from Ip and geometry via the current_profile_nu formula.
                # Same as the legacy fallback for circular geometry, but explicit to
                # silence the TORAX deprecation warning.
                "initial_psi_mode": "j",
            },
            "numerics": {
                "t_initial": 0.0,
                "t_final": 0.1,  # Give it ~100 ms to relax, on order of energy confinement time
                # Linear theta solver is implicit / unconditionally stable, so we
                # can take large fixed steps to reach steady state cheaply. 10ms
                # dt -> 10 steps to cover t_final, vs ~30-100 with chi-based dt.
                "fixed_dt": 1e-2,
                "min_dt": 1e-3,
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
                "a_minor": 9999,  # Overridden by dataloader input
                "B_0": 9999,  # Overridden by dataloader input
                "elongation_LCFS": 9999,  # Overridden by dataloader input
            },
            "transport": {
                "model_name": "constant",
                "chi_i": 9999,  # Predicted by NN
                "chi_e": 9999,  # Predicted by NN
                "D_e": 9999,  # Predicted by NN
                "V_e": 9999,  # Predicted by NN
            },
            "sources": {
                "ei_exchange": {},
                "bremsstrahlung": {},
                "cyclotron_radiation": {},
                "ohmic": {},
                "gas_puff": {"S_total": 9999},  # Predicted by NN
                "generic_current": {},
            },
            # A bunch of stuff that we aren't using but we must include so it doesn't complain
            "solver": {},
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
        "target_vars": ["ne20_psi", "Te_keV_psi"],
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
