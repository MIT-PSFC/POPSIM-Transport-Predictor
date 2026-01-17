TRAINING_DATA_CASES = [
    "cmod",  # C-Mod only
    "tcv",  # TCV only
    "cmod_tcv",  # C-Mod + TCV
    "d3d_lp",  # DIII-D low-performance shots only
    "cmod_tcv_d3d_lp",  # C-Mod + TCV + DIII-D low-performance shots
    "exnihilo",  # No training data
]

MODEL_CASES = [
    "simple_shapes",  # Simple analytic shapes
    "pedestal_shapes",  # Simple analytic shapes with pedestal region
    "gradient_shapes",  # Gradient-based analytic shapes
    "unstructured_nn",  # Unstructured neural network
]

# Shots of high-performance data included in training
HP_SHOTS_INCLUDED = [0, 1, 3, 10, 30, 100]

# 80/20 between training+validation and testing
# and 80/20 between training and validation
TRAIN_VAL_TEST_SPLIT = (0.64, 0.16, 0.2)
TRAIN_TEST_SPLIT = (0.8, 0.2)
