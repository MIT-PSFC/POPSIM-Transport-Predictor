"""
POPSIM already has some training code in place, but for this study we need more flexibility
regarding how the datasets are constructed and how the models are trained.

Primary difference is that the dataloader_config doesn't have a dataset path,
rather it just has the training_data_case, normalization_method, and model_case,
so that the TrainRunBuilder can construct the appropriate datasets using
get_train_val_datasets and get_train_test_datasets_transfer from the organize_data module.
"""

import numpy as np
from popsim.ml import TrainConfig

from transport_study.orchestration.study import Study


def make_sweep_config(case: Study.Case) -> dict:
    """Make the wandb sweep config for a given case in this study."""


def make_train_config(
    study: Study, case: Study.Case, checkpoint_dir: str
) -> TrainConfig:
    """Make the base TrainConfig for a given case in this study"""
    # TODO(ZanderKeith): This needs a lot more thought put into it, just replacing with NaNs for the time being
    loss_config_base = {
        "huber_delta": 0.5,
        "device_weight": {
            "cmod": np.nan,
            "tcv": np.nan,
            "d3d_lp": np.nan,
            "d3d_hp": np.nan,  # Putting a higher weight on the high-performance shots should help with the extrapolation to those shots that we're ultimately interested in
        },
        "var_weight": {
            "Wtot_MJ": np.nan,  # Wtot is ~0.01 MJ so we need a higher weight to make sure this doesn't drift too much
            "P_oh_MW": np.nan,
            "P_rad_MW": np.nan,
        },
        "device_ignore": {
            # Ohmic power data is horrible for DIII-D, ignore it entirely in the loss function
            # TODO(ZanderKeith): Ask Oak Nelson why this is the case
            "d3d_lp": ["P_oh_MW"],
            "d3d_hp": ["P_oh_MW"],
        },
    }

    optimizer_config_base = {
        "lr0": 1e-4,
        "transition_steps": 500,
        "decay_rate": 0.5,
        "lrf": 5e-4,
        "weight_decay": 2e-4,
    }

    # TODO(ZanderKeith) A few things to consider:
    # Do you want to have a different loss function for training / evaluation?
    # That would be interesting for hyperparameter tuning
    # In this case, putting a higher weight on the importance of the P_rad network during training
    # prevents it from straying too much from actually predicting P_rad,
    # which may help with the extrapolation to higher-performance shots that you'd be evaluating on,

    return loss_config_base, optimizer_config_base
