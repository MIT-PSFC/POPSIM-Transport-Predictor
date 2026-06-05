from pathlib import Path

import pytest
from pydantic import ValidationError

from transport_study.config import StudyConfig, _ConfigProxy, config, load_config


@pytest.fixture(autouse=True)
def reset_config():
    _ConfigProxy._cfg = None
    _ConfigProxy.initialized = False
    yield
    _ConfigProxy._cfg = None
    _ConfigProxy.initialized = False


def test_config_load():
    # Config not loaded yet, should raise error when trying to access
    with pytest.raises(RuntimeError):
        debug = config.debug

    cfg = StudyConfig(
        study_name="test_study",
        debug=True,
        dataset_paths={
            "cmod-low": "path/to/cmod_low.nc",
            "cmod-high": "path/to/cmod_high.nc",
        },
    )
    load_config(cfg)

    # After loading, should be able to access config values
    assert config.study_name == "test_study"
    assert config.debug is True


def test_config_immutable():
    cfg1 = StudyConfig(
        study_name="test_study1",
        debug=True,
        dataset_paths={
            "cmod-low": "path/to/cmod_low.nc",
            "cmod-high": "path/to/cmod_high.nc",
        },
    )
    load_config(cfg1)

    cfg2 = StudyConfig(
        study_name="test_study2",
        debug=False,
        dataset_paths={
            "cmod-low": "path/to/cmod_low.nc",
            "cmod-high": "path/to/cmod_high.nc",
        },
    )
    with pytest.raises(RuntimeError):
        load_config(cfg2)

    with pytest.raises(ValidationError):
        cfg1.study_name = "modified_study_name"

    with pytest.raises(TypeError):
        config.dataset_paths["cmod-low"] = "new/path/to/cmod_low.nc"

    assert config.dataset_paths["cmod-low"] == Path("path/to/cmod_low.nc"), "Dataset path should not have been modified"
