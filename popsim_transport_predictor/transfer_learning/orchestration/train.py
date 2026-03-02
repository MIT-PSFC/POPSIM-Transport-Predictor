"""
POPSIM already has some training code in place, but for this study we need more flexibility
regarding how the datasets are constructed and how the models are trained.

Primary difference is that the dataloader_config doesn't have a dataset path,
rather it just has the training_data_case, normalization_method, and model_case,
so that the TrainRunBuilder can construct the appropriate datasets using
get_train_val_datasets and get_train_test_datasets_transfer from the organize_data module.
"""

import os

from popsim.ml import TrainConfig
from popsim.ml.launch import launch_train
from popsim.modules.transport_predictor.train_configs import update_submodule_configs

from popsim_transport_predictor.modules.power_balance.p_oh.trb import OhmicPowerTRB
from popsim_transport_predictor.modules.power_balance.p_rad.trb import RadiatedPowerTRB
from popsim_transport_predictor.modules.power_balance.trb import PowerBalanceTRB
from popsim_transport_predictor.transfer_learning.config import config


def _make_train_config(
    model_dir: str,
    training_data_case: str,
    normalization_method: str,
    model_case: str,
    num_hp_shots: int | None = None,
    transfer_learning: bool = False,
) -> tuple[TrainConfig, TrainConfig, TrainConfig]:
    """Make TrainConfigs for the power_balance, p_oh_predictor, and p_rad_predictor submodules"""

    input_vars_base = [
        "Ip_MA",
        "B0",
        "R0",
        "a_minor",
        "kappa",
        "ne20_line_avg",
        "P_aux_MW",
    ]
    target_vars_base = ["Wtot_MJ", "P_oh_MW", "P_rad_MW"]

    max_epochs = 1 if config.debug else 800
    epochs_per_val = 2 if config.debug else 20

    loss_config_base = {
        "huber_delta": 0.5,
        "device_weight": {
            "cmod": 0.1,
            "tcv": 0.1,
            "d3d_lp": 0.7,
            "d3d_hp": 1.0,  # Putting a higher weight on the high-performance shots should help with the extrapolation to those shots that we're ultimately interested in
        },
        "var_weight": {
            "Wtot_MJ": 800,  # Wtot is ~0.01 MJ so we need a higher weight to make sure this doesn't drift too much
            "P_oh_MW": 0.5,
            "P_rad_MW": 0.5,
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
    if normalization_method == "raw":
        input_vars = input_vars_base
    elif normalization_method == "physics":
        input_vars = [*input_vars_base, "q_star", "f_G", "aB0", "surface_power_density"]
    elif normalization_method == "z_score":
        input_vars = [*input_vars_base, *(f"{var}_z" for var in input_vars_base)]
    elif normalization_method == "coral":
        input_vars = [*input_vars_base, *(f"{var}_coral" for var in input_vars_base)]
    else:
        raise ValueError(f"Unknown normalization method: {normalization_method}")

    if model_case in ["scaling_law", "sciml"]:
        submodule_init_config = {
            "nn_depth": 2,
            "nn_width": 16,
            "min_val": 0,  # Minimum ohmic power in MW
            "max_val": None,  # Get max from training data
            "prng_seed": 42,
            "in_size": 7,  # B0, Ip, R0, a_minor, kappa, ne20_line_avg, P_aux_MW TODO(ZanderKeith): It'd be nice to put the predicted stored energy here, but that'd require passing the mapping functions into the submodules... Doable, but a bit of a pain, so maybe for a future study
            "out_size": 1,
            "normalization_method": normalization_method,
        }
        p_oh_config = TrainConfig(
            project="p_oh_predictor",
            train_run_builder=OhmicPowerTRB,
            max_epochs=max_epochs,
            epochs_per_val=epochs_per_val,
            checkpoint_dir=os.path.join(model_dir, "p_oh_predictor"),
            dataloader_config={
                "target_vars": ["P_oh_MW"],
                "input_vars": input_vars,
            },
            model_init_config=submodule_init_config,
            loss_config=loss_config_base,
            optimizer_config=optimizer_config_base,
        )
        p_rad_config = TrainConfig(
            project="p_rad_predictor",
            train_run_builder=RadiatedPowerTRB,
            max_epochs=max_epochs,
            epochs_per_val=epochs_per_val,
            checkpoint_dir=os.path.join(model_dir, "p_rad_predictor"),
            dataloader_config={
                "target_vars": ["P_rad_MW"],
                "input_vars": input_vars,
            },
            model_init_config=submodule_init_config,
            loss_config=loss_config_base,
            optimizer_config=optimizer_config_base,
        )
        submodules = ["p_oh_predictor", "p_rad_predictor"]
    elif model_case == "unstructured_nn":
        p_oh_config, p_rad_config = None, None
        submodules = []
    else:
        raise ValueError(f"Unknown model case: {model_case}")

    if model_case == "scaling_law":
        model_init_config = {
            "model_case": model_case,
            "normalization_method": normalization_method,
            "freeze_submodules": ["p_oh_predictor", "p_rad_predictor"],
            "submodules": {
                "p_oh_predictor": p_oh_config,
                "p_rad_predictor": p_rad_config,
            },
            "restore_submodules": True,  # Always restoring pre-trained submodules in this study
        }
    elif model_case in ["sciml", "unstructured_nn"]:
        model_init_config = {
            "model_case": model_case,
            "normalization_method": normalization_method,
            "freeze_submodules": [],
            "nn_depth": 2,
            "nn_width": 16,
            "in_size": 7,  # B0, Ip, R0, a_minor, kappa, ne20_line_avg, P_aux_MW TODO(ZanderKeith): It'd be nice to put the predicted stored energy here, but that'd require passing the mapping functions into the submodules... Doable, but a bit of a pain, so maybe for a future study
            "out_size": 1,
            "prng_seed": 42,
            "submodules": {
                "p_oh_predictor": p_oh_config,
                "p_rad_predictor": p_rad_config,
            },
            "restore_submodules": True,  # Always restoring pre-trained submodules in this study
        }
    else:
        raise ValueError(f"Unknown model case: {model_case}")

    power_balance_config = TrainConfig(
        project=f"{model_case}.{training_data_case}.{normalization_method}.power_balance",
        train_run_builder=PowerBalanceTRB,
        max_epochs=max_epochs,
        epochs_per_val=epochs_per_val,
        checkpoint_dir=os.path.join(model_dir, "power_balance"),
        dataloader_config={
            "training_data_case": training_data_case,
            "normalization_method": normalization_method,
            "num_hp_shots": num_hp_shots,
            "transfer_learning": transfer_learning,
            "state_vars": ["Wtot_MJ"],
            "input_vars": input_vars,
            "target_vars": target_vars_base,
            "prng_seed": 42,
            "debug": config.debug,
            # Hyperparameters
            "segment_length_train": 100,
            "segment_overlap_train": 50,
            "batch_size": 8192,
            # Part of validation, should be left alone during hyperparameter tuning
            "segment_length_val": None,
            "segment_overlap_val": 0,
        },
        model_init_config=model_init_config,
        # TODO(ZanderKeith): is the trainable getter only needed for time-independent modules?
        trainable_getter_config={},
        loss_config=loss_config_base,
        optimizer_config=optimizer_config_base,
    )

    power_balance_config = update_submodule_configs(
        power_balance_config.model_dump(), submodules=submodules
    )

    return power_balance_config


def train_model_standard(
    model_dir: str,
    training_data_case: str,
    normalization_method: str,
    model_case: str,
):
    """
    Train a model using standard learning (train and test on the same data distribution).
    """

    power_balance_config = _make_train_config(
        model_dir=model_dir,
        training_data_case=training_data_case,
        normalization_method=normalization_method,
        model_case=model_case,
        num_hp_shots=None,
        transfer_learning=False,
    )

    if model_case in ["scaling_law", "sciml"]:
        p_oh_config = power_balance_config.model_init_config["submodules"][
            "p_oh_predictor"
        ]
        p_rad_config = power_balance_config.model_init_config["submodules"][
            "p_rad_predictor"
        ]
        launch_train(p_oh_config)
        launch_train(p_rad_config)

    trainer, _, val_dl, _, _ = launch_train(power_balance_config)
    return trainer, val_dl


def train_model_transfer(
    model_dir: str,
    training_data_case: str,
    normalization_method: str,
    model_case: str,
    num_hp_shots: int,
):
    """
    Train a model using transfer learning (train on historic data, test on high-performance data).
    The number of high-performance shots included in training is specified by `num_hp_shots`.
    """

    power_balance_config = _make_train_config(
        model_dir=model_dir,
        training_data_case=training_data_case,
        normalization_method=normalization_method,
        model_case=model_case,
        num_hp_shots=num_hp_shots,
        transfer_learning=True,
    )

    if model_case in ["scaling_law", "sciml"]:
        p_oh_config = power_balance_config.model_init_config["submodules"][
            "p_oh_predictor"
        ]
        p_rad_config = power_balance_config.model_init_config["submodules"][
            "p_rad_predictor"
        ]
        launch_train(p_oh_config)
        launch_train(p_rad_config)

    trainer, _, val_dl, _, _ = launch_train(power_balance_config)
    return trainer, val_dl
