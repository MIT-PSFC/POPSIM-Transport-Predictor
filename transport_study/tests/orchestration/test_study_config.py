"""Config-lock and TOML layout tests, shared by every case-grid study.

Study.__init__ writes the identity, locked and case-axis fields of its Config to config_lock.toml on the first run,
and refuses a later run whose identity or locked fields differ, so a study cannot silently mix results from two setups.
The rules are generic over the field roles (see transport_study/config.py), so they are checked once here for all three studies.
"""

import pytest
from pydantic import ValidationError

from transport_study.config import StudyConfig, reset_config
from transport_study.orchestration.config_lock import (
    CONFIG_LOCK_FILENAME,
    LOCK_ROLES,
    read_config_lock,
)
from transport_study.power_balance_transfer.power_balance_study import PowerBalanceStudy
from transport_study.profile_transfer.profile_study import ProfileStudy
from transport_study.transport_transfer.transport_transfer_study import TransportStudy

STUDIES = (ProfileStudy, PowerBalanceStudy, TransportStudy)

DATASET_PATHS = {"cmod-low": "path/to/cmod_low.nc", "cmod-high": "path/to/cmod_high.nc"}


def make_config(study, tmp_path, **overrides):
    defaults = dict(
        study_name="test_study",
        dataset_paths=dict(DATASET_PATHS),
        target_device="cmod-high",
        working_dir_base=tmp_path / "working_dir_base",
        training_datasets=("exnihilo", "cmod-low"),
        domain_adaptation_methods=("none", "weighted"),
        num_target_shots_options=(0, 1),
        target_test_set_size=60,
    )
    defaults.update(overrides)
    return study.Config(**defaults)


def build_study(study, cfg):
    """A fresh Study under cfg, as a new run of the study would build it."""
    reset_config()
    return study(cfg)


@pytest.fixture(params=STUDIES, ids=[s.STUDY_TYPE for s in STUDIES])
def study(request):
    return request.param


def test_locked_field_change_is_refused(study, tmp_path):
    """max_epochs changes what every case produces under an unchanged name, so a rerun with a new value must stop."""
    build_study(study, make_config(study, tmp_path, max_epochs=500))
    with pytest.raises(RuntimeError, match="max_epochs"):
        build_study(study, make_config(study, tmp_path, max_epochs=400))


def test_case_axis_change_is_accepted_and_recorded(study, tmp_path):
    """Adding cases must not invalidate a study, and the orchestrator's lock rewrite records the new grid."""
    build_study(study, make_config(study, tmp_path))
    grown = make_config(study, tmp_path, num_target_shots_options=(0, 1, 3))
    build_study(study, grown).record_case_grid()

    lock = read_config_lock(grown.working_dir_base / grown.study_name / CONFIG_LOCK_FILENAME, study.Config)
    assert lock.config.num_target_shots_options == (0, 1, 3)


def test_lock_round_trips_every_locked_and_axis_field(study, tmp_path, monkeypatch):
    """The lock reads back exactly what it recorded, without the env dataset paths a later run happens to have."""
    cfg = make_config(study, tmp_path, dataset_fractions={"cmod-low": 0.25, "cmod-high": 0.75}, max_ds_size=100)
    build_study(study, cfg)
    monkeypatch.setenv("PTPS_DATASET_PATHS", '{"mast": "path/to/mast.nc"}')

    lock = read_config_lock(cfg.working_dir_base / cfg.study_name / CONFIG_LOCK_FILENAME, study.Config)
    assert lock.study_type == study.STUDY_TYPE
    assert cfg.differing_fields(lock.config, *LOCK_ROLES) == []


def test_lock_is_never_created_over_existing_artifacts(study, tmp_path):
    """A hand-deleted lock must not vouch for results of an unknown setup, cleaning the study starts it over."""
    cfg = make_config(study, tmp_path)
    built = build_study(study, cfg)
    built.setup_directories()
    (built.result_dir / "some_case").mkdir()
    (built.working_dir / CONFIG_LOCK_FILENAME).unlink()

    with pytest.raises(RuntimeError, match="no config lock"):
        build_study(study, cfg)
    study.clean_working_dir(cfg, clean_models=True, clean_results=True)
    build_study(study, cfg)


def test_config_toml_round_trip(study, tmp_path, monkeypatch):
    # from_toml fills devices missing from [datasets] in from PTPS_DATASET_PATHS, which this round trip must not see
    monkeypatch.delenv("PTPS_DATASET_PATHS", raising=False)
    config = make_config(study, tmp_path, study_name="test_study_save_load", dataset_fractions={"cmod-low": 0.25, "cmod-high": 0.75})
    save_path = tmp_path / "config.toml"
    config.save(save_path)

    assert study.Config.from_toml(save_path) == config


def test_misplaced_toml_field_names_its_table(study, tmp_path):
    """A case axis at the top level of a study TOML raises, pointing at [cases]."""
    toml_path = tmp_path / "study.toml"
    toml_path.write_text(
        'study_name = "x"\ntarget_test_set_size = 2\nnum_target_shots_options = [0, 1]\n'
        '[cases]\ntraining_datasets = ["exnihilo"]\n[datasets]\ncmod-high = "path/to/cmod_high.nc"\ntarget = "cmod-high"\n'
    )
    with pytest.raises(ValueError, match=r"num_target_shots_options .* belongs in \[cases\]"):
        study.Config.from_toml(toml_path)


def test_base_config_rejects_study_specific_fields():
    """The base StudyConfig forbids extra fields, so it can never stand in for
    a study Config and silently drop its case-grid axes."""
    with pytest.raises(ValidationError):
        StudyConfig(study_name="x", target_device="y", model_types=("sciml",))
