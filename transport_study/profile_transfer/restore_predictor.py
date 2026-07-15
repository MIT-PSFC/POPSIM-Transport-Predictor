from pathlib import Path

from popsim.ml.checkpointing import create_default_checkpoint_manager, restore_model
from popsim.ml.train_config import TrainConfig

from transport_study.modules.profile_predictor.module import ProfilePredictor
from transport_study.modules.profile_predictor.trb import (
    ProfilePredictorTRB,
)
from transport_study.profile_transfer.profile_study import (
    HYPERPARAM_TARGET_SHOTS,
    ProfileStudy,
)


def checkpoint_to_profile_case(checkpoint_dir: Path | str) -> ProfileStudy.Case:
    """Parse a case directory name back into a ProfileStudy.Case.

    Mirrors ProfileStudy.Case.__str__:
    case.{model_type}.td_{td}.freeze_{f}                   source-trained, no adaptation
    case.{model_type}.td_{td}.freeze_{f}.targ_{n}          exnihilo
    case.{model_type}.td_{td}.freeze_{f}.targ_{n}.da_{da}  domain adaptation
    """
    case_name = Path(checkpoint_dir).name
    pieces = case_name.split(".")
    if len(pieces) not in (4, 5, 6) or pieces[0] != "case" or not pieces[2].startswith("td_") or not pieces[3].startswith("freeze_"):
        raise ValueError(f"Unexpected case name format: {case_name}")

    return ProfileStudy.Case(
        model_type=pieces[1],
        training_data=pieces[2].removeprefix("td_"),
        domain_adaptation=pieces[5].removeprefix("da_") if len(pieces) == 6 else None,
        freeze_shapes=pieces[3].removeprefix("freeze_") == "True",
        num_target_shots=int(pieces[4].removeprefix("targ_")) if len(pieces) >= 5 else HYPERPARAM_TARGET_SHOTS,
    )


def checkpoint_to_profile_config(checkpoint_dir: Path | str) -> TrainConfig:
    profile_working_dir = Path(checkpoint_dir).parent
    case = checkpoint_to_profile_case(checkpoint_dir)
    hyperparam_case = case.get_hyperparam_prereq()
    tuned_config_path = profile_working_dir / str(hyperparam_case) / "tuned_config.yaml"
    if not tuned_config_path.exists():
        raise FileNotFoundError(f"Tuned config not found for profile predictor case {case} at path {tuned_config_path}")
    profile_predictor_config = TrainConfig.load(tuned_config_path)
    # Make sure the checkpoint_dir in the config matches the one we're trying to restore from
    profile_predictor_config = profile_predictor_config.model_copy(update={"checkpoint_dir": checkpoint_dir})
    return profile_predictor_config


def restore_profile_predictor(
    profile_predictor_config: dict | TrainConfig,
) -> ProfilePredictor:
    if isinstance(profile_predictor_config, TrainConfig):
        profile_predictor_config = profile_predictor_config.model_dump()

    # Don't need the full dataloader, only want rho grid
    profile_predictor_config["dataloader_config"]["debug"] = True

    _, profile_predictor_train_dl, _, _ = ProfilePredictorTRB.get_dataloaders(profile_predictor_config["dataloader_config"])
    profile_predictor = ProfilePredictorTRB.model_init(
        profile_predictor_train_dl,
        profile_predictor_config["model_init_config"],
    )
    profile_predictor_manager = create_default_checkpoint_manager(profile_predictor_config["checkpoint_dir"])
    profile_predictor = restore_model(profile_predictor_manager, profile_predictor)
    return profile_predictor


def restore_profile_predictor_from_checkpoint(checkpoint_dir: Path | str):
    """Restore the profile predictor from the given checkpoint directory"""
    profile_predictor_config = checkpoint_to_profile_config(checkpoint_dir)
    config_dict = profile_predictor_config.model_dump()
    config_dict["checkpoint_dir"] = checkpoint_dir
    profile_predictor = restore_profile_predictor(config_dict)
    return profile_predictor
