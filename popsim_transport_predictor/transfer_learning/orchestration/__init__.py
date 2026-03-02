from popsim_transport_predictor.transfer_learning.config import config

TRAINING_DATA_CASES = [
    "cmod",  # C-Mod only
    "tcv",  # TCV only
    "cmod_tcv",  # C-Mod + TCV
    "d3d_lp",  # DIII-D low-performance shots only
    "cmod_tcv_d3d_lp",  # C-Mod + TCV + DIII-D low-performance shots
    "exnihilo",  # No training data
]

DOMAIN_NORMALIZATION_METHODS = [
    "raw",  # No normalization, Ip, Wtot, etc. are in their original units
    "physics",  # Convert to typical dimensionless parameters like beta, q95, f_G, etc.
    "z_score",  # Within each device, normalize each variable to zero mean and unit variance.
    "coral",  # Use the CORAL method to align covariances of source and target domains (https://arxiv.org/abs/1612.01939)
]

MODEL_CASES = {
    # Different treatments of the full-shot transport predictor module
    "transport_predictor": [
        "simple_shapes",  # Simple analytic shapes
        "pedestal_shapes",  # Simple analytic shapes with pedestal region
        "gradient_shapes",  # Gradient-based analytic shapes
        "unstructured_nn",  # Unstructured neural network
    ],
    # Just the power balance module. (P_OH and P_RAD still get their own models since the data is atrocious)
    "power_balance": [
        "scaling_law",  # Based on scaling laws, H89, H98, and HL transition threshold between them
        "sciml",  # Neural network predicts tau_E, and we do the power balance calculation
        "unstructured_nn",  # Unstructured neural network directly predicts stored energy evolution
    ],
    # Profile predictor (needed for pre-shot trajectory optimization for upcoming DIII-D campaign)
    "profile_predictor": [
        "simple_shapes",  # Simple analytic shapes
        "pedestal_shapes",  # Simple analytic shapes with pedestal region
        "gradient_shapes",  # Gradient-based analytic shapes
        "unstructured_nn",  # Unstructured neural network
    ],
}

# Shots of high-performance data included in training
if config.debug:
    HP_SHOTS_INCLUDED = [0, 1, 10]
else:
    HP_SHOTS_INCLUDED = [0, 1, 3, 10, 30, 100]

# 80/20 between train/val
# 80/20 between train+val/test
TRAIN_VAL_SPLIT = (0.8, 0.2)
TRAIN_VAL_TEST_SPLIT = (0.64, 0.16, 0.2)
