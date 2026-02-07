TRAINING_DATA_CASES = [
    "cmod",  # C-Mod only
    "tcv",  # TCV only
    "cmod_tcv",  # C-Mod + TCV
    "d3d_lp",  # DIII-D low-performance shots only
    "cmod_tcv_d3d_lp",  # C-Mod + TCV + DIII-D low-performance shots
    "exnihilo",  # No training data
]

MODEL_CASES = {
    # Different treatments of the full-shot transport predictor module
    "transport_predictor": {
        "simple_shapes",  # Simple analytic shapes
        "pedestal_shapes",  # Simple analytic shapes with pedestal region
        "gradient_shapes",  # Gradient-based analytic shapes
        "unstructured_nn",  # Unstructured neural network
    },
    # Just the power balance module. (P_OH and P_RAD still get their own models since the data is atrocious)
    "power_balance": {
        "scaling_law",  # Based on the scaling law, between H89 and H98 HL transition threshold
        "sciml",  # Neural network predicts tau_E, and we do the power balance calculation
        "unstructured_nn",  # Unstructured neural network directly predicts stored energy evolution
    },
    # Profile predictor (needed for pre-shot trajectory optimization for upcoming DIII-D campaign)
    "profile_predictor": {
        "simple_shapes",  # Simple analytic shapes
        "pedestal_shapes",  # Simple analytic shapes with pedestal region
        "gradient_shapes",  # Gradient-based analytic shapes
        "unstructured_nn",  # Unstructured neural network
    },
}

# Shots of high-performance data included in training
HP_SHOTS_INCLUDED = [0, 1, 3, 10, 30, 100]

# 80/20 between training+validation and testing
# and 80/20 between training and validation
TRAIN_VAL_TEST_SPLIT = (0.64, 0.16, 0.2)
TRAIN_TEST_SPLIT = (0.8, 0.2)
