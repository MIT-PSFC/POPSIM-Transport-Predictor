from pathlib import Path

from popsim.ml.checkpointing import create_default_checkpoint_manager, restore_model
from popsim.ml.train_config import TrainConfig

from transport_study.modules.profile_predictor.module import ProfilePredictor
from transport_study.modules.profile_predictor.trb import (
    ProfilePredictorTRB,
)
from transport_study.orchestration.study import HYPERPARAM_TARGET_SHOTS
from transport_study.profile_transfer.profile_study import ProfileStudy


def checkpoint_to_profile_case(checkpoint_dir: Path | str) -> ProfileStudy.Case:
    """Parse a case directory name back into a ProfileStudy.Case.

    Mirrors ProfileStudy.Case.__str__:
    case.{model_type}.td_{td}.norm_{n}[.freeze_True].geom_{g}                   source-trained, no adaptation
    case.{model_type}.td_{td}.norm_{n}[.freeze_True].geom_{g}.targ_{n}          exnihilo
    case.{model_type}.td_{td}.norm_{n}[.freeze_True].geom_{g}.targ_{n}.da_{da}  domain adaptation

    The freeze_ token is suppressed for unfrozen shapes (see Study.Case.STR_TOKEN_FIELDS),
    norm_ and geom_ are mandatory and fixed in position.
    """
    case_name = Path(checkpoint_dir).name
    pieces = case_name.split(".")
    if len(pieces) < 5 or pieces[0] != "case" or not pieces[2].startswith("td_"):
        raise ValueError(f"Unexpected case name format: {case_name}")

    idx = 3
    if not pieces[idx].startswith("norm_"):
        raise ValueError(f"Unexpected case name format: {case_name}")
    data_normalization = pieces[idx].removeprefix("norm_")
    idx += 1

    freeze_shapes = pieces[idx].startswith("freeze_")
    if freeze_shapes:
        if pieces[idx] != "freeze_True":
            raise ValueError(f"Unexpected case name format: {case_name}")
        idx += 1

    if idx == len(pieces) or not pieces[idx].startswith("geom_"):
        raise ValueError(f"Unexpected case name format: {case_name}")
    geometry_builder = pieces[idx].removeprefix("geom_")
    idx += 1

    num_target_shots = HYPERPARAM_TARGET_SHOTS
    domain_adaptation = None
    if idx < len(pieces) and pieces[idx].startswith("targ_"):
        num_target_shots = int(pieces[idx].removeprefix("targ_"))
        idx += 1
        if idx < len(pieces) and pieces[idx].startswith("da_"):
            domain_adaptation = pieces[idx].removeprefix("da_")
            idx += 1

    if idx != len(pieces):
        raise ValueError(f"Unexpected case name format: {case_name}")

    return ProfileStudy.Case(
        model_type=pieces[1],
        training_data=pieces[2].removeprefix("td_"),
        data_normalization=data_normalization,
        domain_adaptation=domain_adaptation,
        freeze_shapes=freeze_shapes,
        num_target_shots=num_target_shots,
        geometry_builder=geometry_builder,
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
    profile_predictor_config = profile_predictor_config.model_copy(update={"checkpoint_dir": str(checkpoint_dir)})
    return profile_predictor_config


def restore_profile_predictor(
    profile_predictor_config: dict | TrainConfig,
) -> ProfilePredictor:
    if isinstance(profile_predictor_config, TrainConfig):
        profile_predictor_config = profile_predictor_config.model_dump()

    _, profile_predictor_train_dl, _, _ = ProfilePredictorTRB.get_dataloaders(profile_predictor_config["dataloader_config"])
    profile_predictor = ProfilePredictorTRB.model_init(
        profile_predictor_train_dl,
        profile_predictor_config["model_init_config"],
    )
    profile_predictor_manager = create_default_checkpoint_manager(profile_predictor_config["checkpoint_dir"])
    profile_predictor = restore_model(profile_predictor_manager, profile_predictor)
    return profile_predictor


def restore_profile_predictor_from_checkpoint(checkpoint_dir: Path | str) -> ProfilePredictor:
    """Restore the profile predictor from the given checkpoint directory"""
    # checkpoint_to_profile_config already points checkpoint_dir at the target
    return restore_profile_predictor(checkpoint_to_profile_config(checkpoint_dir))
