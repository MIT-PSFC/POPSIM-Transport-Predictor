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
            "ne20_edge",
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
            "ne20_edge",
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
                "Ip": None,  # Overridden by dataloader input
                "T_i_right_bc": None,
                "T_e_right_bc": None,
                # Initial profiles that will then relax under influence of transport and sources.
                "T_i": {0: {0: 1.0, 1: 0.01}},
                "T_e": {0: {0: 1.0, 1: 0.01}},
                "n_e": {0: {0: 1e20, 1: 0.01e20}},
                "normalize_n_e_to_nbar": True,
                "nbar": None,  # Overridden by dataloader input
                "n_e_nbar_is_fGW": True,
                "n_e_right_bc": None,
            },
            "numerics": {
                "t_initial": 0.0,
                "t_final": 0.1,  # Give it ~100 ms to relax, on order of energy confinement time
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
                "R_major": None,  # Overridden by dataloader input
                "a_minor": None,  # Overridden by dataloader input
                "B_0": None,  # Overridden by dataloader input
                "elongation_LCFS": None,  # Overridden by dataloader input
            },
            "transport": {
                "model_name": "constant",
                "chi_i": None,  # Predicted by NN
                "chi_e": None,  # Predicted by NN
                "D_e": None,  # Predicted by NN
                "V_e": None,  # Predicted by NN
            },
            "sources": {
                "ei_exchange": {},
                "bremsstrahlung": {},
                "cyclotron_radiation": {},
                "ohmic": {},
                "gas_puff": {"S_total": None},  # Predicted by NN
                "generic_current": {},
            },
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
            "ne20_edge",
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
