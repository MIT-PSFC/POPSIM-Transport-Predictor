import shutil

import pytest

from transport_study.orchestration.slurm_utils import (
    count_idle_gpus,
    resources_available,
)

if shutil.which("squeue") is None:
    pytest.skip("slurm not available, skipping slurm utils tests", allow_module_level=True)


def test_count_idle_gpus():
    idle_gpus = count_idle_gpus()
    assert isinstance(idle_gpus, int)
    assert idle_gpus >= 0


def test_resources_available():
    available = resources_available()
    assert isinstance(available, bool)


if __name__ == "__main__":
    test_count_idle_gpus()
    test_resources_available()
