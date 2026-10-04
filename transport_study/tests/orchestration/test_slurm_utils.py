import os
import shutil
import sys

import pytest

from transport_study.config import StudyConfig, load_config
from transport_study.orchestration.slurm_utils import (
    count_idle_gpus,
    importable_module,
    parse_slurm_time_s,
    resources_available,
)
from transport_study.profile_transfer import profile_study

needs_slurm = pytest.mark.skipif(
    shutil.which("squeue") is None or os.environ.get("PTPS_PARTITION") is None,
    reason="needs a SLURM install and PTPS_PARTITION",
)


@pytest.fixture
def loaded_config():
    return load_config(
        StudyConfig(
            study_name="test_slurm_utils",
            dataset_paths={"test_device": "/path/to/test_ds"},
            target_device="test_device",
        )
    )


@needs_slurm
def test_count_idle_gpus(loaded_config):
    idle_gpus = count_idle_gpus()
    assert isinstance(idle_gpus, int)
    assert idle_gpus >= 0


@needs_slurm
def test_resources_available(loaded_config):
    assert isinstance(resources_available(), bool)


def test_importable_module_normal():
    """A class from a normally imported module reports its own dotted path."""
    assert importable_module(profile_study.ProfileStudy.Config) == "transport_study.profile_transfer.profile_study"


def test_importable_module_main_recovers_dotted_path(monkeypatch):
    """When launched as a script the entry-point module is __main__.

    cls.__module__ is then "__main__" and a subprocess cannot import it, so the
    helper must recover the real dotted path from the source file. Emulate that
    state: point sys.modules["__main__"] at the profile_study module (which has
    a __file__) and mark the class as living in __main__.
    """
    cls = profile_study.ProfileStudy.Config
    monkeypatch.setitem(sys.modules, "__main__", profile_study)
    monkeypatch.setattr(cls, "__module__", "__main__")
    assert importable_module(cls) == "transport_study.profile_transfer.profile_study"


@pytest.mark.parametrize(
    ("time_str", "expected_s"),
    [
        ("30", 30 * 60),
        ("30:15", 30 * 60 + 15),
        ("06:00:00", 6 * 3600),
        ("2-12", 2 * 86400 + 12 * 3600),
        ("2-12:30", 2 * 86400 + 12 * 3600 + 30 * 60),
        ("2-12:30:15", 2 * 86400 + 12 * 3600 + 30 * 60 + 15),
        ("UNLIMITED", None),
    ],
)
def test_parse_slurm_time_s_reads_every_sbatch_form(time_str, expected_s):
    """A lone field is minutes, but after a day prefix it is hours (sbatch --time grammar)."""
    assert parse_slurm_time_s(time_str) == expected_s
