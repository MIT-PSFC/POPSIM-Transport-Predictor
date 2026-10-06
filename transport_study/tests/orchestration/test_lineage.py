"""Parent studies on real PowerBalanceStudy grids.

A child borrows every case an ancestor has trained or will train, waits for the unfinished ones,
trains only the cases new to its grid, and never writes into an ancestor.
Each test builds the studies in order, as separate runs would, resetting the one-shot global config in between.
"""

import pytest
import xarray as xr
from loguru import logger

from transport_study.config import reset_config
from transport_study.orchestration import study as study_module
from transport_study.orchestration.study import write_netcdf_atomic
from transport_study.power_balance_transfer.power_balance_study import PowerBalanceStudy

DATASET_PATHS = {"cmod-low": "path/to/cmod_low.nc", "cmod-high": "path/to/cmod_high.nc"}


def make_config(tmp_path, study_name, parent_study=None, **overrides):
    defaults = dict(
        study_name=study_name,
        parent_study=parent_study,
        dataset_paths=dict(DATASET_PATHS),
        target_device="cmod-high",
        working_dir_base=tmp_path,
        training_datasets=("cmod-low",),
        domain_adaptation_methods=("none", "weighted"),
        model_types=("mlp",),
        data_normalization_methods=("physics",),
        num_target_shots_options=(0, 1),
        target_test_set_size=10,
    )
    defaults.update(overrides)
    return PowerBalanceStudy.Config(**defaults)


def build(cfg) -> PowerBalanceStudy:
    """The study as a fresh run builds it, detached from its log file so a study left behind writes nothing more."""
    reset_config()
    study = PowerBalanceStudy(cfg)
    logger.remove(study.log_sink_id)
    return study


def finish(study, case) -> None:
    """Stand in for a finished training job of a case the study trains itself."""
    write_netcdf_atomic(xr.Dataset(), study.result_path(case))


def cases_by_name(study) -> dict:
    return {str(case): case for case in study.cases}


def snapshot(directory):
    return {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in sorted(directory.rglob("*"))}


def test_child_borrows_the_parent_grid_and_trains_only_new_cases(tmp_path):
    parent = build(make_config(tmp_path, "primary"))
    parent_names = set(cases_by_name(parent))
    child = build(make_config(tmp_path, "secondary", parent_study="primary", model_types=("mlp", "transformer")))

    for name, case in cases_by_name(child).items():
        borrowed = name in parent_names
        assert child.is_borrowed(case) == borrowed, name
        home_dir = tmp_path / ("primary" if borrowed else "secondary")
        assert child.result_path(case).is_relative_to(home_dir)
        assert child.tuned_config_path(case).is_relative_to(home_dir)
    assert any(not child.is_borrowed(case) for case in child.cases)


def test_child_waits_for_borrowed_cases_and_launches_only_its_own(tmp_path, monkeypatch):
    parent = build(make_config(tmp_path, "primary"))
    parent_cases = list(parent.cases)
    child = build(make_config(tmp_path, "secondary", parent_study="primary", model_types=("mlp", "transformer")))

    launched = []

    def fake_run_case(case, skip_tuning, enable_parallelism):
        launched.append(str(case))
        finish(child, case)

    def parent_finishes_while_child_sleeps(_seconds):
        for case in parent_cases:
            finish(parent, case)

    monkeypatch.setattr(child, "run_case", fake_run_case)
    monkeypatch.setattr(study_module.time, "sleep", parent_finishes_while_child_sleeps)
    child.run_unfinished_cases(skip_tuning=True, enable_parallelism=False)

    assert sorted(launched) == sorted(str(case) for case in child.cases if case.model_type == "transformer")
    assert child.get_unfinished_cases() == []
    with pytest.raises(RuntimeError, match="must not write it"):
        child.launch_train(parent_cases[0])


def test_tertiary_resolves_each_case_to_the_root_most_claimant(tmp_path):
    build(make_config(tmp_path, "primary"))
    build(make_config(tmp_path, "secondary", parent_study="primary", model_types=("mlp", "transformer")))
    tertiary = build(
        make_config(tmp_path, "tertiary", parent_study="secondary", model_types=("mlp", "transformer"), num_target_shots_options=(0, 1, 3))
    )

    for case in tertiary.cases:
        if case.num_target_shots == 3:
            expected_home = "tertiary"
        elif case.model_type == "mlp":
            expected_home = "primary"
        else:
            expected_home = "secondary"
        assert tertiary.case_home(case).name == expected_home, str(case)


def test_reset_ancestor_stops_every_descendant(tmp_path):
    primary_cfg = make_config(tmp_path, "primary")
    secondary_cfg = make_config(tmp_path, "secondary", parent_study="primary", model_types=("mlp", "transformer"))
    tertiary_cfg = make_config(tmp_path, "tertiary", parent_study="secondary", num_target_shots_options=(0, 1, 3))
    build(primary_cfg)
    build(secondary_cfg)
    build(tertiary_cfg)

    PowerBalanceStudy.clean_working_dir(primary_cfg, clean_models=True, clean_results=True)
    build(primary_cfg)
    with pytest.raises(RuntimeError, match="primary was reset after study secondary"):
        build(secondary_cfg)
    with pytest.raises(RuntimeError, match="primary was reset after study secondary"):
        build(tertiary_cfg)

    # Cleaning the secondary brings it back on the primary's new results, and resets it for the tertiary in turn
    PowerBalanceStudy.clean_working_dir(secondary_cfg, clean_models=True, clean_results=True)
    build(secondary_cfg)
    with pytest.raises(RuntimeError, match="secondary was reset after study tertiary"):
        build(tertiary_cfg)


def test_locked_field_mismatch_with_parent_is_refused(tmp_path):
    build(make_config(tmp_path, "primary", max_epochs=500))
    with pytest.raises(RuntimeError, match="max_epochs"):
        build(make_config(tmp_path, "secondary", parent_study="primary", max_epochs=400))


def test_ancestor_grid_growing_over_a_trained_case_is_a_conflict(tmp_path):
    """Root-first ownership would silently swap the secondary's own case for the primary's, so it raises instead."""
    build(make_config(tmp_path, "primary"))
    secondary_cfg = make_config(tmp_path, "secondary", parent_study="primary", model_types=("mlp", "transformer"))
    secondary = build(secondary_cfg)
    transformer_case = next(case for case in secondary.cases if case.model_type == "transformer")
    finish(secondary, transformer_case)

    build(make_config(tmp_path, "primary", model_types=("mlp", "transformer"))).record_case_grid()
    with pytest.raises(RuntimeError, match="holds its own copy"):
        build(secondary_cfg)


def test_child_never_writes_into_its_parent(tmp_path):
    parent = build(make_config(tmp_path, "primary"))
    for case in parent.cases:
        finish(parent, case)
    parent_tree = snapshot(tmp_path / "primary")

    child_cfg = make_config(tmp_path, "secondary", parent_study="primary", model_types=("mlp", "transformer"))
    child = build(child_cfg)
    child.record_case_grid()
    child.setup_directories()
    child.refresh_lineage()
    for case in child.cases:
        child.result_path(case)
        child.trained_model_dir(case)
        child.tuned_config_path(case)
    PowerBalanceStudy.clean_working_dir(child_cfg, clean_models=True, clean_results=True, clean_figures=True)
    build(child_cfg)

    assert snapshot(tmp_path / "primary") == parent_tree
