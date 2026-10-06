"""Literal case-name regression tests for the transport transfer study.

str(case) names checkpoint dirs, result files, tuned-config paths, wandb projects, and SLURM job names,
so its format must never drift.
"""

from pathlib import Path

import pytest

from transport_study.config import load_config
from transport_study.transport_transfer.transport_transfer_study import TransportStudy


@pytest.fixture
def loaded_config():
    return load_config(
        TransportStudy.Config(
            study_name="test-transport-case-naming",
            dataset_paths={"cmod-low1": Path("path/to/cmod_low1.nc"), "cmod-high": Path("path/to/cmod_high.nc")},
            target_device="cmod-high",
            target_test_set_size=5,
            training_datasets=("cmod-low1",),
        )
    )


def _case(**overrides):
    kwargs = {
        "model_type": "sciml",
        "training_data": "cmod-low1",
        "domain_adaptation": None,
        "freeze_submodules": True,
        "num_target_shots": 0,
    }
    kwargs.update(overrides)
    return TransportStudy.Case(**kwargs)


def test_source_trained_case_name(loaded_config):
    # Normalization is study-wide, and the circular / rebuild defaults are suppressed
    assert str(_case()) == "case.sciml.td_cmod-low1.freeze_True"


def test_exnihilo_case_name(loaded_config):
    case = _case(model_type="transformer", training_data="exnihilo", num_target_shots=5)
    assert str(case) == "case.transformer.td_exnihilo.freeze_True.targ_5"


def test_domain_adaptation_case_name(loaded_config):
    case = _case(domain_adaptation="transfer", num_target_shots=3)
    assert str(case) == "case.sciml.td_cmod-low1.freeze_True.targ_3.da_transfer"


@pytest.mark.parametrize(
    ("geometry_builder", "torax_state", "suffix"),
    [
        ("miller", "rebuild", ".geom_miller"),
        ("circular", "carry", ".tstate_carry"),
        ("miller", "carry", ".geom_miller.tstate_carry"),
    ],
)
def test_torax_axis_tokens(loaded_config, geometry_builder, torax_state, suffix):
    """A non-default torax axis gets its token in STR_TOKEN_FIELDS order, a default one stays suppressed."""
    case = _case(model_type="torax-gyrobohm", geometry_builder=geometry_builder, torax_state=torax_state)
    assert str(case) == f"case.torax-gyrobohm.td_cmod-low1.freeze_True{suffix}"


def test_submodule_prereq_case_names(loaded_config):
    """sciml needs a power_balance and a profile case, and the power_balance case needs p_oh and p_rad.

    Submodule cases sit at the unfrozen hyperparam default, so their names carry no freeze token.
    """
    prereq_names = {str(prereq) for prereq in _case().prereqs}
    assert {"case.power_balance.td_cmod-low1", "case.profile.td_cmod-low1"} <= prereq_names

    power_balance_case = next(prereq for prereq in _case().prereqs if prereq.model_type == "power_balance")
    power_balance_prereq_names = {str(prereq) for prereq in power_balance_case.prereqs}
    assert {"case.p_oh.td_cmod-low1", "case.p_rad.td_cmod-low1"} <= power_balance_prereq_names


def test_target_shot_order_reaches_every_prereq(loaded_config):
    """A sciml case's power_balance and profile prereqs, and their p_oh and p_rad, train on the same target shots."""
    case = _case(domain_adaptation="weighted", num_target_shots=3, target_shot_order="spanning")
    assert str(case) == "case.sciml.td_cmod-low1.freeze_True.targ_3.order_spanning.da_weighted"

    prereq_names = {str(prereq) for prereq in case.prereqs}
    assert "case.power_balance.td_cmod-low1.targ_3.order_spanning.da_weighted" in prereq_names
    assert "case.profile.td_cmod-low1.targ_3.order_spanning.da_weighted" in prereq_names
    power_balance_case = next(prereq for prereq in case.prereqs if prereq.model_type == "power_balance")
    power_balance_prereq_names = {str(prereq) for prereq in power_balance_case.prereqs}
    assert "case.p_oh.td_cmod-low1.targ_3.order_spanning.da_weighted" in power_balance_prereq_names


@pytest.mark.parametrize("torax_axis", [{"geometry_builder": "miller"}, {"torax_state": "carry"}])
def test_non_torax_rejects_torax_axes(loaded_config, torax_axis):
    with pytest.raises(ValueError, match="only applies to torax model types"):
        _case(model_type="transformer", **torax_axis)


def test_cases_are_hashable_and_set_stable(loaded_config):
    assert _case() in {_case()}
    assert hash(_case()) == hash(_case())
