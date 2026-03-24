import os

from popsim.ml.checkpointing import create_default_checkpoint_manager, restore_model
from popsim.ml.train_config import TrainConfig

from transport_study.modules.profile_predictor.module import ProfilePredictor
from transport_study.modules.profile_predictor.trb import (
    ProfilePredictorTRB,
)
from transport_study.profile_transfer.run_study import ProfileStudy


def checkpoint_to_profile_case(checkpoint_dir: str) -> ProfileStudy.Case:
    # Extract the case name from the path, assuming it's the name of the last directory in the path
    case_name = checkpoint_dir.split("/")[-1]
    case_pieces = case_name.split(".")
    model_type = case_pieces[1]
    training_data = case_pieces[2][3:]  # remove "td_" prefix
    data_normalization = "physics"  # Always using this for profile predictor

    if len(case_pieces) == 4:
        raise NotImplementedError(
            "Don't have a case with 4 pieces, need to update the parsing logic if we want to add one"
        )
    elif len(case_pieces) == 5:
        domain_adaptation = None
        freeze_shapes = (
            case_pieces[4][7:] == "True"
        )  # remove "freeze_" prefix and convert to bool
        num_hp_shots = -1
    elif len(case_pieces) == 6:
        domain_adaptation = case_pieces[3][3:]  # remove "da_" prefix
        freeze_shapes = (
            case_pieces[4][7:] == "True"
        )  # remove "freeze_" prefix and convert to bool
        num_hp_shots = int(case_pieces[5][3:])  # remove "hp_" prefix and convert to int
    else:
        raise ValueError(f"Unexpected case name format: {case_name}")

    return ProfileStudy.Case(
        model_type=model_type,
        training_data=training_data,
        data_normalization=data_normalization,
        domain_adaptation=domain_adaptation,
        freeze_shapes=freeze_shapes,
        num_hp_shots=num_hp_shots,
    )


def checkpoint_to_profile_config(checkpoint_dir: str) -> TrainConfig:
    profile_working_dir = os.path.dirname(checkpoint_dir)
    case = checkpoint_to_profile_case(checkpoint_dir)
    hyperparam_case = case.get_hyperparam_prereq()
    tuned_config_path = os.path.join(
        profile_working_dir, str(hyperparam_case), "tuned_config.yaml"
    )
    if not os.path.exists(tuned_config_path):
        raise FileNotFoundError(
            f"Tuned config not found for profile predictor case {case} at path {tuned_config_path}"
        )
    profile_predictor_config = TrainConfig.load(tuned_config_path)

    return profile_predictor_config


def restore_profile_predictor(
    profile_predictor_config: dict | TrainConfig,
) -> ProfilePredictor:
    if isinstance(profile_predictor_config, TrainConfig):
        profile_predictor_config = profile_predictor_config.model_dump()

    # Don't need the full dataloader, only want psi grid
    profile_predictor_config["dataloader_config"]["debug"] = True

    # TODO(ZanderKeith) ensure this actually completely works for all types of profile predictors
    # Make a test that trains a thing and restores it and checks that the predictions are the same
    _, profile_predictor_train_dl, _, _ = ProfilePredictorTRB.get_dataloaders(
        profile_predictor_config["dataloader_config"]
    )
    profile_predictor = ProfilePredictorTRB.model_init(
        profile_predictor_train_dl,
        profile_predictor_config["model_init_config"],
    )
    profile_predictor_manager = create_default_checkpoint_manager(
        profile_predictor_config["checkpoint_dir"]
    )
    profile_predictor = restore_model(profile_predictor_manager, profile_predictor)
    return profile_predictor


def restore_profile_predictor_from_checkpoint(checkpoint_dir: str):
    """Restore the profile predictor from the given checkpoint directory"""
    profile_predictor_config = checkpoint_to_profile_config(checkpoint_dir)
    config_dict = profile_predictor_config.model_dump()
    config_dict["checkpoint_dir"] = checkpoint_dir
    profile_predictor = restore_profile_predictor(config_dict)
    return profile_predictor
