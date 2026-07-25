"""Config-lock and TOML round-trip tests, shared by every case-grid study.

Study.__init__ writes its Config to a lock file on the first run and refuses to
start when a later run's config is not is_compatible with it, so a study cannot
silently mix results from two different data or tuning setups. The rules are
identical across studies (only COMPAT_HYPERPARAM_FIELDS differs), so they are
checked generically here rather than once per study.
"""

import pytest
from pydantic import ValidationError

from transport_study.config import StudyConfig
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
        target_test_set_size=60,
    )
    defaults.update(overrides)
    return study.Config(**defaults)


def mutate(config, field: str):
    """A copy differing only in `field`.

    model_copy skips validation, so the mutated value only has to differ, not
    to be a legal setting for that field.
    """
    value = getattr(config, field)
    changed: object
    if isinstance(value, bool):
        changed = not value
    elif isinstance(value, int):
        changed = value + 1
    else:
        changed = f"changed-{value}"
    return config.model_copy(update={field: changed})


@pytest.fixture(params=STUDIES, ids=[s.STUDY_TYPE for s in STUDIES])
def study(request):
    return request.param


def test_identical_configs_are_compatible(study, tmp_path):
    assert make_config(study, tmp_path).is_compatible(make_config(study, tmp_path))


@pytest.mark.parametrize("field", ["study_name", "target_device", "target_test_set_size"])
def test_study_identity_change_is_incompatible(study, tmp_path, field):
    config = make_config(study, tmp_path)
    assert not config.is_compatible(mutate(config, field))


def test_dataset_path_change_is_incompatible(study, tmp_path):
    config = make_config(study, tmp_path)
    other = make_config(study, tmp_path, dataset_paths={**DATASET_PATHS, "cmod-low": "new/path/to/cmod_low.nc"})
    assert not config.is_compatible(other)


def test_hyperparam_change_is_incompatible(study, tmp_path):
    """Every COMPAT_HYPERPARAM_FIELDS entry must invalidate an existing study."""
    config = make_config(study, tmp_path)
    assert study.Config.COMPAT_HYPERPARAM_FIELDS, f"{study.__name__} declares no compat hyperparam fields"
    for field in study.Config.COMPAT_HYPERPARAM_FIELDS:
        assert not config.is_compatible(mutate(config, field)), field


def test_case_grid_axes_do_not_affect_compatibility(study, tmp_path):
    """Adding cases to a study must not invalidate its existing results."""
    config = make_config(study, tmp_path)
    assert config.is_compatible(mutate(config, "model_types"))
    assert config.is_compatible(make_config(study, tmp_path, training_datasets=("exnihilo",)))


def test_config_toml_round_trip(study, tmp_path):
    config = make_config(study, tmp_path, study_name="test_study_save_load")
    save_path = tmp_path / "config.toml"
    config.save(save_path)

    assert study.Config.from_toml(save_path) == config


def test_config_toml_round_trip_with_dataset_fractions(study, tmp_path):
    config = make_config(study, tmp_path, dataset_fractions={"cmod-low": 0.25, "cmod-high": 0.75})
    save_path = tmp_path / "config.toml"
    config.save(save_path)

    reloaded = study.Config.from_toml(save_path)
    assert reloaded.dataset_fractions == config.dataset_fractions
    assert reloaded == config


def test_base_config_rejects_study_specific_fields():
    """The base StudyConfig forbids extra fields, so it can never stand in for
    a study Config and silently drop its case-grid axes."""
    with pytest.raises(ValidationError):
        StudyConfig(study_name="x", target_device="y", model_types=("sciml",))
