from pathlib import Path

import pytest
from pydantic import ValidationError

from transport_study.config import (
    DEBUG_EPOCHS_PER_VAL,
    DEBUG_MAX_EPOCHS,
    StudyConfig,
    config,
    load_config,
)

# Global config reset around each test is handled by the shared conftest fixture


def test_config_load():
    # Config not loaded yet, should raise error when trying to access
    with pytest.raises(RuntimeError):
        _ = config.debug

    cfg = StudyConfig(
        study_name="test_study",
        debug=True,
        dataset_paths={
            "cmod-low": "path/to/cmod_low.nc",
            "cmod-high": "path/to/cmod_high.nc",
        },
        target_device="cmod-high",
    )
    load_config(cfg)

    # After loading, should be able to access config values
    assert config.study_name == "test_study"
    assert config.debug is True
    # debug caps the epochs, the validation interval and the sweep size
    assert config.max_epochs == DEBUG_MAX_EPOCHS
    assert config.epochs_per_val == DEBUG_EPOCHS_PER_VAL
    assert config.hyperparam_sweeps == 1


def test_config_immutable():
    cfg1 = StudyConfig(
        study_name="test_study1",
        debug=True,
        dataset_paths={
            "cmod-low": "path/to/cmod_low.nc",
            "cmod-high": "path/to/cmod_high.nc",
        },
        target_device="cmod-high",
    )
    load_config(cfg1)

    cfg2 = StudyConfig(
        study_name="test_study2",
        debug=False,
        dataset_paths={
            "cmod-low": "path/to/cmod_low.nc",
            "cmod-high": "path/to/cmod_high.nc",
        },
        target_device="cmod-high",
    )
    with pytest.raises(RuntimeError):
        load_config(cfg2)

    with pytest.raises(ValidationError):
        cfg1.study_name = "modified_study_name"

    with pytest.raises(TypeError):
        config.dataset_paths["cmod-low"] = "new/path/to/cmod_low.nc"

    assert config.dataset_paths["cmod-low"] == Path("path/to/cmod_low.nc"), "Dataset path should not have been modified"
