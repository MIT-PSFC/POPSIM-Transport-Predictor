"""Literal case-name regression tests for the profile study.

str(case) names checkpoint dirs, result files, tuned-config paths, wandb
projects, and SLURM job names, so its format must never drift. The parser in
restore_predictor must invert it exactly.
"""

from pathlib import Path

import pytest

from transport_study.config import load_config
from transport_study.profile_transfer.profile_study import ProfileStudy
from transport_study.profile_transfer.restore_predictor import (
    checkpoint_to_profile_case,
)


@pytest.fixture
def loaded_config():
    return load_config(
        ProfileStudy.Config(
            study_name="test-case-naming",
            dataset_paths={"cmod-low1": Path("path/to/cmod_low1.nc"), "cmod-high": Path("path/to/cmod_high.nc")},
            target_device="cmod-high",
            target_test_set_size=5,
            training_datasets=("cmod-low1",),
        )
    )


def _case(**overrides):
    kwargs = {
        "model_type": "shape-init-pca",
        "training_data": "cmod-low1",
        "domain_adaptation": None,
        "freeze_shapes": True,
        "num_target_shots": 0,
    }
    kwargs.update(overrides)
    return ProfileStudy.Case(**kwargs)


def test_source_trained_case_name(loaded_config):
    assert str(_case()) == "case.shape-init-pca.td_cmod-low1.norm_physics.freeze_True.geom_circular"


def test_exnihilo_case_name(loaded_config):
    case = _case(model_type="mlp", training_data="exnihilo", num_target_shots=7)
    assert str(case) == "case.mlp.td_exnihilo.norm_physics.freeze_True.geom_circular.targ_7"


def test_domain_adaptation_case_name(loaded_config):
    case = _case(model_type="torax-gyrobohm", domain_adaptation="addition", num_target_shots=10)
    assert str(case) == "case.torax-gyrobohm.td_cmod-low1.norm_physics.freeze_True.geom_circular.targ_10.da_addition"


def test_weighted_case_name(loaded_config):
    case = _case(data_normalization="physics-coral", domain_adaptation="weighted", num_target_shots=3)
    assert str(case) == "case.shape-init-pca.td_cmod-low1.norm_physics-coral.freeze_True.geom_circular.targ_3.da_weighted"


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"model_type": "mlp", "training_data": "exnihilo", "num_target_shots": 7},
        {"model_type": "torax-gyrobohm", "domain_adaptation": "transfer", "num_target_shots": 3},
        {"data_normalization": "physics-coral", "domain_adaptation": "weighted", "num_target_shots": 3},
        {"model_type": "torax-qlknn", "geometry_builder": "miller", "domain_adaptation": "addition", "num_target_shots": 1},
        # Unfrozen shapes leave no freeze token
        {"freeze_shapes": False},
        {"freeze_shapes": False, "domain_adaptation": "transfer", "num_target_shots": 3},
    ],
)
def test_checkpoint_parser_round_trip(loaded_config, overrides):
    case = _case(**overrides)
    parsed = checkpoint_to_profile_case(f"/checkpoints/{case}")
    assert str(parsed) == str(case)
    assert parsed == case
