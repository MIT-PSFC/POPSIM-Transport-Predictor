"""Literal case-name regression tests.

str(case) names checkpoint dirs, result files, tuned-config paths, wandb
projects, and SLURM job names, so its format must never drift.
"""

from pathlib import Path

import pytest

from transport_study.config import load_config
from transport_study.power_balance_transfer.power_balance_study import PowerBalanceStudy


@pytest.fixture
def loaded_config():
    return load_config(
        PowerBalanceStudy.Config(
            study_name="test-case-naming",
            dataset_paths={"cmod-low1": Path("path/to/cmod_low1.nc"), "cmod-high": Path("path/to/cmod_high.nc")},
            target_device="cmod-high",
            target_test_set_size=5,
            training_datasets=("cmod-low1",),
        )
    )


def _case(**overrides):
    kwargs = {
        "model_type": "sciml-taue-nn",
        "training_data": "cmod-low1",
        "data_normalization": "coral",
        "domain_adaptation": None,
        "freeze_submodules": True,
        "num_target_shots": 0,
    }
    kwargs.update(overrides)
    return PowerBalanceStudy.Case(**kwargs)


def test_source_trained_case_name(loaded_config):
    assert str(_case()) == "case.sciml-taue-nn.td_cmod-low1.norm_coral.freeze_True"


def test_exnihilo_case_name(loaded_config):
    # Unfrozen is the default, so it leaves no freeze token
    case = _case(model_type="mlp", training_data="exnihilo", data_normalization="raw", freeze_submodules=False, num_target_shots=5)
    assert str(case) == "case.mlp.td_exnihilo.norm_raw.targ_5"


def test_domain_adaptation_case_name(loaded_config):
    case = _case(domain_adaptation="transfer", num_target_shots=3)
    assert str(case) == "case.sciml-taue-nn.td_cmod-low1.norm_coral.freeze_True.targ_3.da_transfer"


def test_target_shot_order_case_name(loaded_config):
    # The base (ascending) order leaves no token, every other order follows targ_
    case = _case(domain_adaptation="transfer", num_target_shots=3, target_shot_order="spanning")
    assert str(case) == "case.sciml-taue-nn.td_cmod-low1.norm_coral.freeze_True.targ_3.order_spanning.da_transfer"


def test_cases_are_hashable_and_set_stable(loaded_config):
    assert _case() in {_case()}
    assert hash(_case()) == hash(_case())


def test_multiobjective_case_name_and_shared_sweep(loaded_config):
    """A multiobjective transformer carries mo_True and tunes on the plain transformer's sweep."""
    case = _case(
        model_type="transformer",
        data_normalization="physics",
        freeze_submodules=False,
        domain_adaptation="weighted",
        num_target_shots=3,
        multiobjective=True,
    )
    assert str(case) == "case.transformer.td_cmod-low1.norm_physics.mo_True.targ_3.da_weighted"
    assert str(case.prereqs[0]) == "case.transformer.td_cmod-low1.norm_physics"


def test_sciml_rejects_multiobjective(loaded_config):
    # The sciml types anchor their submodules regardless, the axis does not apply to them
    with pytest.raises(ValueError, match="multiobjective should be a dummy value"):
        _case(multiobjective=True)
