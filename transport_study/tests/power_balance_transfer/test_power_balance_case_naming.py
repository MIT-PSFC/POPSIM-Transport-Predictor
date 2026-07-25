"""Literal case-name regression tests.

str(case) names checkpoint dirs, result files, tuned-config paths, wandb
projects, and SLURM job names, so its format must never drift.
"""

from pathlib import Path

import pytest

from transport_study import PACKAGE_ROOT
from transport_study.config import load_config
from transport_study.power_balance_transfer.power_balance_study import PowerBalanceStudy

SAMPLE_DIR = Path(PACKAGE_ROOT) / "datasets" / "sample"


@pytest.fixture
def loaded_config():
    return load_config(
        PowerBalanceStudy.Config(
            study_name="test-case-naming",
            dataset_paths={"cmod-low1": SAMPLE_DIR / "cmod-low1.nc", "cmod-high": SAMPLE_DIR / "cmod-high.nc"},
            target_device="cmod-high",
            target_test_set_size=5,
            training_datasets=("cmod-low1",),
        )
    )


def _case(**overrides):
    kwargs = {
        "model_type": "sciml",
        "training_data": "cmod-low1",
        "data_normalization": "coral",
        "domain_adaptation": None,
        "freeze_submodules": True,
        "num_target_shots": 0,
    }
    kwargs.update(overrides)
    return PowerBalanceStudy.Case(**kwargs)


def test_source_trained_case_name(loaded_config):
    assert str(_case()) == "case.sciml.td_cmod-low1.norm_coral.freeze_True"


def test_exnihilo_case_name(loaded_config):
    case = _case(model_type="unstructured_nn", training_data="exnihilo", data_normalization="raw", num_target_shots=5)
    assert str(case) == "case.unstructured_nn.td_exnihilo.norm_raw.freeze_True.targ_5"


def test_domain_adaptation_case_name(loaded_config):
    case = _case(domain_adaptation="transfer", num_target_shots=3)
    assert str(case) == "case.sciml.td_cmod-low1.norm_coral.freeze_True.targ_3.da_transfer"


def test_cases_are_hashable_and_set_stable(loaded_config):
    assert _case() in {_case()}
    assert hash(_case()) == hash(_case())
