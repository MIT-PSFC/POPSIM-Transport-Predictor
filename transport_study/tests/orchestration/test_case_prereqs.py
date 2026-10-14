"""Prereq chains the base Case builds through pinned replaces, under both hyperparam multiobjective values."""

import pytest

from transport_study.config import reset_config
from transport_study.power_balance_transfer.power_balance_study import PowerBalanceStudy
from transport_study.transport_transfer.transport_transfer_study import TransportStudy

DATASET_PATHS = {"cmod-low": "path/to/cmod_low.nc", "cmod-high": "path/to/cmod_high.nc"}


def _study(study_cls, tmp_path, **overrides):
    defaults = dict(
        study_name="test_case_prereqs",
        dataset_paths=dict(DATASET_PATHS),
        target_device="cmod-high",
        working_dir_base=tmp_path,
        training_datasets=("cmod-low",),
        domain_adaptation_methods=("weighted",),
        num_target_shots_options=(1,),
        target_test_set_size=1,
        multiobjective_options=(True,),
    )
    reset_config()
    return study_cls(study_cls.Config(**{**defaults, **overrides}))


@pytest.mark.parametrize("hyperparam_multiobjective", [False, True])
def test_multiobjective_anchor_prereqs_are_the_hyperparam_submodule_cases(tmp_path, hyperparam_multiobjective):
    """A multiobjective transformer depends on the hyperparam p_oh / p_rad cases, whose own flag the pin may set True.

    With hyperparam_multiobjective True the submodule prereqs carry multiobjective True too,
    and must not look for anchor prereqs of their own.
    """
    study = _study(PowerBalanceStudy, tmp_path, model_types=("transformer",), hyperparam_multiobjective=hyperparam_multiobjective)
    case = next(case for case in study.cases if case.model_type == "transformer" and case.multiobjective)
    anchor_prereqs = sorted(str(prereq) for prereq in case.prereqs if prereq.model_type in ("p_oh", "p_rad"))
    assert len(anchor_prereqs) == 2
    for prereq in case.prereqs:
        if prereq.model_type in ("p_oh", "p_rad"):
            assert prereq.is_hyperparam_case()
            assert prereq.prereqs is None


@pytest.mark.parametrize("hyperparam_multiobjective", [False, True])
def test_transport_multiobjective_prereqs_terminate(tmp_path, hyperparam_multiobjective):
    study = _study(TransportStudy, tmp_path, model_types=("transformer",), hyperparam_multiobjective=hyperparam_multiobjective)
    case = next(case for case in study.cases if case.model_type == "transformer" and case.multiobjective)
    anchor_types = sorted(prereq.model_type for prereq in case.prereqs if prereq.model_type in ("power_balance", "p_oh"))
    assert anchor_types == ["p_oh", "power_balance"]
    power_balance = next(prereq for prereq in case.prereqs if prereq.model_type == "power_balance")
    assert power_balance.is_hyperparam_case()
    assert sorted(prereq.model_type for prereq in power_balance.prereqs) == ["p_oh", "p_rad"]


def test_a_case_spelled_off_its_pinned_axes_is_rejected(tmp_path):
    """An mlp has no submodules to freeze, so a frozen mlp is not a case the grid could hold."""
    study = _study(PowerBalanceStudy, tmp_path, model_types=("mlp",), multiobjective_options=(False,))
    mlp_case = next(case for case in study.cases if case.model_type == "mlp")
    with pytest.raises(ValueError, match="freeze_submodules"):
        mlp_case.replace(freeze_submodules=not mlp_case.freeze_submodules)
    assert mlp_case.replace_pinned(freeze_submodules=not mlp_case.freeze_submodules) == mlp_case
