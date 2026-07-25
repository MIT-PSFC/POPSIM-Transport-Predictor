"""Literal case-name regression tests for the profile study.

str(case) names checkpoint dirs, result files, tuned-config paths, wandb
projects, and SLURM job names, so its format must never drift. The parser in
restore_predictor must invert it exactly.
"""

from pathlib import Path

import pytest

from transport_study import PACKAGE_ROOT
from transport_study.config import load_config
from transport_study.profile_transfer.profile_study import ProfileStudy
from transport_study.profile_transfer.restore_predictor import (
    checkpoint_to_profile_case,
)

SAMPLE_DIR = Path(PACKAGE_ROOT) / "datasets" / "sample"


@pytest.fixture
def loaded_config():
    return load_config(
        ProfileStudy.Config(
            study_name="test-case-naming",
            dataset_paths={"cmod-low1": SAMPLE_DIR / "cmod-low1.nc", "cmod-high": SAMPLE_DIR / "cmod-high.nc"},
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
    case = _case(model_type="torax-cgm", domain_adaptation="addition", num_target_shots=10)
    assert str(case) == "case.torax-cgm.td_cmod-low1.norm_physics.freeze_True.geom_circular.targ_10.da_addition"


def test_weighted_case_name(loaded_config):
    """Pin the case-name string for the 'weighted' domain adaptation method.

    Should test that a case with domain_adaptation='weighted' names as
    case.{model_type}.td_{td}.norm_{n}.freeze_{f}.geom_{g}.targ_{n}.da_weighted, and that
    checkpoint_to_profile_case round-trips it (see
    test_checkpoint_parser_round_trip).
    """


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"model_type": "mlp", "training_data": "exnihilo", "num_target_shots": 7},
        {"model_type": "torax-cgm", "domain_adaptation": "transfer", "num_target_shots": 3},
    ],
)
def test_checkpoint_parser_round_trip(loaded_config, overrides):
    case = _case(**overrides)
    parsed = checkpoint_to_profile_case(f"/checkpoints/{case}")
    assert str(parsed) == str(case)
    assert parsed == case
