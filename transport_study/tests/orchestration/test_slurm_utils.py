import os
import shutil

import pytest

from transport_study.config import StudyConfig, load_config
from transport_study.orchestration.slurm_utils import (
    count_idle_gpus,
    resources_available,
)

if shutil.which("squeue") is None:
    pytest.skip("slurm not available, skipping slurm utils tests", allow_module_level=True)


@pytest.mark.skipif(os.environ.get("PTPS_PARTITION", None) is None, reason="PTPS_PARTITION not set, skipping slurm utils tests")
def test_count_idle_gpus():
    test_config = StudyConfig(
        study_name="test_study",
        dataset_paths={"test_device": "/path/to/test_ds"},
        target_device="test_device",
    )
    load_config(test_config)
    idle_gpus = count_idle_gpus()
    assert isinstance(idle_gpus, int)
    assert idle_gpus >= 0


@pytest.mark.skipif(os.environ.get("PTPS_PARTITION", None) is None, reason="PTPS_PARTITION not set, skipping slurm utils tests")
def test_resources_available():
    test_config = StudyConfig(
        study_name="test_study",
        dataset_paths={"test_device": "/path/to/test_ds"},
        target_device="test_device",
    )
    load_config(test_config)
    available = resources_available()
    assert isinstance(available, bool)
