"""The profile case grid keeps every model family whatever geometry builders a study compares."""

from collections import Counter
from pathlib import Path

import pytest

from transport_study.orchestration.study import HYPERPARAM_TARGET_SHOTS
from transport_study.profile_transfer.profile_study import ProfileStudy


@pytest.mark.parametrize("geometry_builders", [("circular",), ("miller",), ("circular", "miller")])
def test_non_torax_cases_survive_any_geometry_axis(tmp_path, geometry_builders):
    study = ProfileStudy(
        ProfileStudy.Config(
            study_name="test-profile-case-grid",
            working_dir_base=tmp_path,
            dataset_paths={"cmod-low1": Path("path/to/cmod_low1.nc"), "cmod-high": Path("path/to/cmod_high.nc")},
            target_device="cmod-high",
            target_test_set_size=5,
            model_types=("mlp", "torax-gyrobohm"),
            training_datasets=("cmod-low1",),
            domain_adaptation_methods=(None,),
            geometry_builders=geometry_builders,
            num_target_shots_options=(HYPERPARAM_TARGET_SHOTS,),
        )
    )

    cases_by_model = Counter((case.model_type, case.geometry_builder) for case in study.cases)

    assert cases_by_model[("mlp", "circular")] == 1, "every non-torax case appears once, pinned to circular"
    assert sum(n for (model_type, _), n in cases_by_model.items() if model_type == "mlp") == 1
    for geometry_builder in geometry_builders:
        assert cases_by_model[("torax-gyrobohm", geometry_builder)] == 1
