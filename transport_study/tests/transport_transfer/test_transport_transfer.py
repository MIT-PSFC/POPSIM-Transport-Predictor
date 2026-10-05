"""Study-level wiring of the transport transfer study.

Module behavior lives in tests/modules/transport_predictor/, the losses in test_loss_fn.py.
"""

from collections import Counter
from pathlib import Path

import pytest

from transport_study.config import config, load_config
from transport_study.orchestration.study import HYPERPARAM_TARGET_SHOTS
from transport_study.transport_transfer.transport_transfer_study import TransportStudy


def _load_transport_config(**overrides):
    return load_config(
        TransportStudy.Config(
            study_name="test-transport-transfer",
            dataset_paths={"cmod-low1": Path("path/to/cmod_low1.nc"), "cmod-high": Path("path/to/cmod_high.nc")},
            target_device="cmod-high",
            target_test_set_size=5,
            training_datasets=("cmod-low1",),
            **overrides,
        )
    )


@pytest.mark.parametrize(
    ("data_normalization", "power_balance_data_normalization", "twin_targ"),
    [
        # Stateless power balance normalization under a stat study-wide one: one shared twin
        ("physics-coral", "physics", HYPERPARAM_TARGET_SHOTS),
        # Stat power balance normalization under a stateless study-wide one: the twin keeps n
        ("physics", "coral", 3),
    ],
)
@pytest.mark.parametrize("model_type", ["power_balance", "p_oh", "p_rad"])
def test_power_balance_submodule_twins_follow_their_own_normalization(
    model_type, data_normalization, power_balance_data_normalization, twin_targ
):
    """The power balance submodule cases train with power_balance_data_normalization,
    so their transfer twin keeps num_target_shots exactly when that method fits statistics,
    whatever the study-wide data_normalization is."""
    _load_transport_config(
        data_normalization=data_normalization,
        power_balance_data_normalization=power_balance_data_normalization,
    )
    case = TransportStudy.Case(
        model_type=model_type,
        training_data="cmod-low1",
        domain_adaptation="transfer",
        freeze_submodules=False,
        num_target_shots=3,
    )

    twin = case.transfer_pretrain_case()

    assert twin.domain_adaptation == "transfer_pretrain"
    assert twin.num_target_shots == twin_targ


def test_power_balance_data_normalization_is_validated():
    with pytest.raises(ValueError, match="power_balance_data_normalization"):
        _load_transport_config(power_balance_data_normalization="bogus")


@pytest.fixture
def grid_study(synthetic_device_stores, tmp_path) -> TransportStudy:
    """Every case-grid axis with two values, on the synthetic stores (cmod the source, mast the target)."""
    return TransportStudy(
        TransportStudy.Config(
            study_name="test-transport-transfer-grid",
            working_dir_base=tmp_path,
            dataset_paths=synthetic_device_stores,
            target_device="mast",
            target_test_set_size=1,
            training_datasets=("cmod",),
            model_types=("sciml", "transformer", "torax-gyrobohm"),
            freeze_submodules_options=(True, False),
            geometry_builders=("circular", "miller"),
            torax_state_options=("rebuild", "carry"),
            domain_adaptation_methods=("none", "weighted"),
            num_target_shots_options=(1,),
        )
    )


def test_case_grid_axes_apply_only_to_their_model_types(grid_study):
    """freeze_submodules varies only for sciml, geometry_builder and torax_state only for torax-*,
    and every sciml variant shares one case of each submodule per domain adaptation."""
    cases_by_type = Counter(case.model_type for case in grid_study.cases)
    torax_axes = {
        (case.geometry_builder, case.torax_state, case.domain_adaptation)
        for case in grid_study.cases
        if case.model_type.startswith("torax-")
    }

    for case in grid_study.cases:
        if not case.model_type.startswith("torax-"):
            assert (case.geometry_builder, case.torax_state) == ("circular", "rebuild"), case
        if case.model_type != "sciml":
            assert case.freeze_submodules == config.hyperparam_freeze_submodules, case
    assert len(torax_axes) == 2 * 2 * 2
    # The hyperparam case pins freeze_submodules, the weighted cases take both values
    assert cases_by_type["sciml"] == 3
    for submodule_type in ("power_balance", "profile", "p_oh", "p_rad"):
        assert cases_by_type[submodule_type] == 2, submodule_type


def test_sciml_submodule_configs_nest_two_deep(grid_study):
    """A sciml case restores its power_balance and profile prereqs, and the power_balance config restores its own p_oh / p_rad."""
    case = next(case for case in grid_study.cases if case.model_type == "sciml" and case.domain_adaptation is None)
    prereqs = {prereq.model_type: prereq for prereq in case.prereqs}
    power_balance_prereqs = {prereq.model_type: prereq for prereq in prereqs["power_balance"].prereqs}

    submodule_configs = grid_study.make_train_config(case).model_init_config["submodules"]
    power_balance_submodule_configs = submodule_configs["power_balance"].model_init_config["submodules"]

    assert submodule_configs["power_balance"].checkpoint_dir == str(grid_study.trained_model_dir(prereqs["power_balance"]))
    assert submodule_configs["profile_predictor"].checkpoint_dir == str(grid_study.trained_model_dir(prereqs["profile"]))
    for name, model_type in (("p_oh_predictor", "p_oh"), ("p_rad_predictor", "p_rad")):
        expected_dir = str(grid_study.trained_model_dir(power_balance_prereqs[model_type]))
        assert power_balance_submodule_configs[name].checkpoint_dir == expected_dir
