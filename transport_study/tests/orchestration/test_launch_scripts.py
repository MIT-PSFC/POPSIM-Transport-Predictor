"""The launch_*_parallel helpers embed file paths into generated Python scripts
with !r interpolation. A Path object reprs as PosixPath('...'), which is a
NameError in the generated script (only Path is imported there), so every
embedded path must be a str at interpolation time. These capture the generated
scripts without touching SLURM and check they compile.
"""

from pathlib import Path

import pytest
import yaml
from popsim.ml import TrainConfig

from transport_study.config import StudyConfig, load_config
from transport_study.orchestration import slurm_utils


@pytest.fixture
def loaded_config():
    return load_config(
        StudyConfig(
            study_name="launch_script_test",
            dataset_paths={},
            target_device="cmod",
            partition="fake_partition",
        )
    )


@pytest.fixture
def train_config() -> TrainConfig:
    return TrainConfig(
        project="launch_script_test",
        train_run_builder="transport_study.modules.profile_predictor.trb.ProfilePredictorTRB",
        max_epochs=1,
        epochs_per_val=1,
        dataloader_config={},
        model_init_config={},
        loss_config={},
        optimizer_config={},
    )


@pytest.fixture
def submitted_scripts(monkeypatch) -> list[str]:
    """Record the sbatch stdin of every submission, without touching SLURM."""
    submitted: list[str] = []

    class FakeResult:
        returncode = 0
        stdout = "Submitted batch job 1"
        stderr = ""

    def fake_run(cmd, **kwargs):
        # Launch helpers also shell out to scontrol/sacctmgr for partition
        # limits, only record actual sbatch submissions
        if cmd and cmd[0] == "sbatch":
            submitted.append(kwargs.get("input", ""))
        return FakeResult()

    monkeypatch.setattr(slurm_utils.subprocess, "run", fake_run)
    return submitted


def assert_script_valid(script_path: Path):
    src = script_path.read_text()
    assert "PosixPath" not in src, f"{script_path.name} embeds a Path repr instead of a str:\n{src}"
    compile(src, str(script_path), "exec")


def test_launch_agent_parallel_script_compiles(tmp_path, loaded_config, train_config, submitted_scripts):
    slurm_utils.launch_agent_parallel(train_config, "sweep123", {"count": 1}, "agent_test_job", tmp_path)

    assert len(submitted_scripts) == 1
    scripts = list(tmp_path.glob("agent_test_job_*/run_agent.py"))
    assert len(scripts) == 1
    assert_script_valid(scripts[0])
    # The config the agent reloads must round-trip through yaml
    config_yaml = scripts[0].parent / "config.yaml"
    TrainConfig(**yaml.safe_load(config_yaml.read_text()))


def test_launch_train_parallel_script_compiles(tmp_path, loaded_config, train_config, submitted_scripts):
    slurm_utils.launch_train_parallel(train_config, "train_test_job", tmp_path / "result.nc", tmp_path)

    assert len(submitted_scripts) == 1
    scripts = list(tmp_path.glob("train_test_job_script_*.py"))
    assert len(scripts) == 1
    assert_script_valid(scripts[0])
