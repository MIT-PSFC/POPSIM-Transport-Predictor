from popsim_transport_predictor.modules.profile_trajectory.profile_predictor.module import (
    ShapeType,
)
from popsim_transport_predictor.modules.profile_trajectory.profile_predictor.trb import (
    ProfilePredictorTRB,
)

PROFILE_PREDICTOR_SHAPE_INIT_CONFIG = {
    "project": "profile_predictor_shape_init",
    "train_run_builder": ProfilePredictorTRB,
    "max_epochs": 2,
    "epochs_per_val": 2,
    "checkpoint_dir": None,
    "dataloader_config": {
        # DIII-D - specific keys, this will only ever be a submodule of the trajectory optimization
        "input_vars": [
            "iptipp_MA",
            "B0",
            "beta",
            "dstdenp",
            "gapin",
            "gapout",
            "rxpt1",
            "zxpt1",
            "rxpt2",
            "zxpt2",
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

PROFILE_PREDICTOR_DIRECT_POINTS_CONFIG = {
    "project": "profile_predictor_direct_points",
    "train_run_builder": ProfilePredictorTRB,
    "max_epochs": 2,
    "epochs_per_val": 2,
    "checkpoint_dir": None,
    "dataloader_config": {
        # DIII-D - specific keys, this will only ever be a submodule of the trajectory optimization
        "input_vars": [
            "iptipp_MA",
            "B0",
            "beta",
            "dstdenp",
            "gapin",
            "gapout",
            "rxpt1",
            "zxpt1",
            "rxpt2",
            "zxpt2",
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
