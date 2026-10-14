"""The launch_*_parallel helpers embed file paths into generated Python scripts
with !r interpolation. A Path object reprs as PosixPath('...'), which is a
NameError in the generated script (only Path is imported there), so every
embedded path must be a str at interpolation time. These capture the generated
scripts without touching SLURM and check they compile.
"""

import runpy
from pathlib import Path

import pytest
import yaml
from popsim.ml import TrainConfig

from transport_study.config import StudyConfig, load_config
from transport_study.orchestration import slurm_utils
from transport_study.tests.stubs import StubCase


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
    slurm_utils.launch_train_parallel(train_config, "train_test_job", tmp_path / "result.nc", tmp_path / "train_test_job.log")

    assert len(submitted_scripts) == 1
    scripts = list(tmp_path.glob("train_test_job_script_*.py"))
    assert len(scripts) == 1
    assert_script_valid(scripts[0])


def test_cpu_partition_job_requests_cpus_and_no_gpu(tmp_path, train_config, submitted_scripts):
    """A training job on a CPU partition asks for no GPU and pins jax to the CPU, with the configured CPUs and memory."""
    load_config(
        StudyConfig(
            study_name="launch_script_test",
            dataset_paths={},
            target_device="cmod",
            partition="fake_gpu_partition",
            cpu_partition="fake_cpu_partition",
            cpu_train_cpus=8,
            train_mem="48G",
        )
    )
    slurm_utils.launch_train_parallel(
        train_config, "train_cpu_job", tmp_path / "result.nc", tmp_path / "train_cpu_job.log", partition="fake_cpu_partition"
    )

    (script,) = submitted_scripts
    assert "--gres" not in script
    assert "export JAX_PLATFORMS=cpu" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=48G" in script


def test_collect_results_job_script_compiles_and_asks_for_its_cpus(make_stub_study, submitted_scripts):
    """The collection job reads on one process per CPU, so it asks for that many."""
    study = make_stub_study([StubCase(name="case.a")], partition="fake_partition")

    slurm_utils.launch_collect_results_job(study, cpus=16)

    (script,) = submitted_scripts
    assert "#SBATCH --cpus-per-task=16" in script
    assert "export JAX_PLATFORMS=cpu" in script
    collect_scripts = list((study.working_dir / "logs" / "logs_collect").glob(f"{study.collect_job_name()}_*.py"))
    assert len(collect_scripts) == 1
    assert_script_valid(collect_scripts[0])


def test_collect_results_job_script_inert_when_rerun_as_mp_main(make_stub_study, submitted_scripts):
    """Each spawned reader process re-runs the job's script as __mp_main__.

    Without the main guard every reader built the study and spawned readers of its own,
    and multiprocessing aborted the job.
    """
    study = make_stub_study([StubCase(name="case.a")], partition="fake_partition")
    slurm_utils.launch_collect_results_job(study, cpus=16)
    log_dir = study.working_dir / "logs" / "logs_collect"
    (collect_script,) = log_dir.glob(f"{study.collect_job_name()}_*.py")
    (study_config,) = log_dir.glob(f"{study.collect_job_name()}_study_config_*.toml")

    runpy.run_path(str(collect_script), run_name="__mp_main__")

    # Only the main process deletes the config it reloads
    assert study_config.exists()
    assert not study.collected_results_path().exists()
