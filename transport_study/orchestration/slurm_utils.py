import getpass
import inspect
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from loguru import logger
from popsim.ml import TrainConfig

from transport_study.config import config


def _importable_module(cls: type) -> str:
    """Return a dotted module path a subprocess can import ``cls`` from.

    When the study is launched as a script (e.g. ``python .../profile_study.py``)
    the entry-point module is ``__main__``, so ``cls.__module__`` is ``"__main__"``
    and a subprocess cannot ``from __main__ import ...``. Recover the real dotted
    module by walking up the source-file path while ``__init__.py`` exists.
    """
    module = cls.__module__
    if module != "__main__":
        return module
    file = Path(inspect.getfile(cls)).resolve()
    parts = [file.stem]
    parent = file.parent
    while (parent / "__init__.py").exists():
        parts.append(parent.name)
        parent = parent.parent
    return ".".join(reversed(parts))


def _config_reload_script(study_config_path: Path) -> str:
    """Python source that reconstructs the current StudyConfig (or subclass) and loads it.

    `load_config(Path(...))` only knows how to build the base `StudyConfig`, but
    the actual config is often a subclass with extra fields and `extra="forbid"`
    (e.g. `ProfileStudy.Config`), so a subprocess must import that exact class
    and call its `from_toml` directly instead of going through `load_config`'s
    generic Path branch.

    So for example, this will turn into python source like:
    ```
    from transport_study.profile_transfer.profile_study import ProfileStudy
    from transport_study.config import load_config
    study_config = ProfileStudy.Config.from_toml(Path("/path/to/study_config.toml"))
    load_config(study_config)
    ```
    """
    cls = config.get_subclass()
    top_name = cls.__qualname__.split(".")[0]
    code_str = (
        f"from {_importable_module(cls)} import {top_name}\n"
        "from transport_study.config import load_config\n"
        f"study_config = {cls.__qualname__}.from_toml(Path({str(study_config_path)!r}))\n"
        "load_config(study_config)\n"
    )
    return code_str


def count_running_jobs(job_name: str, partition: str | None = None) -> int:
    """Run squeue to list running jobs on the partition with the specific name"""
    if partition is None:
        partition = config.partition
    result = subprocess.run(
        [
            "squeue",
            "-p",
            partition,
            "-n",
            job_name,
            "--state=RUNNING,PENDING",
            "--noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.critical(f"squeue failed: {result.stderr}")
        return 999999  # Return a large number to prevent launching more jobs if squeue fails
    # Count lines (each line is a job)
    return len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0


def get_running_job_names(partition: str | None = None) -> set[str] | None:
    """Names of this user's running and pending jobs on the partition, in one squeue call.

    Orchestration loops poll job state for every case each pass. One squeue
    call returning all names (checked by set membership) replaces hundreds of
    per-case squeue calls. Returns None when squeue fails, so callers can tell
    "no jobs" apart from "scheduler unreachable" and hold off launching.
    """
    if partition is None:
        partition = config.partition
    result = subprocess.run(
        [
            "squeue",
            "-p",
            partition,
            "-u",
            getpass.getuser(),
            "--state=RUNNING,PENDING",
            "--noheader",
            # Default %j truncates long names, and case names run long
            "--format=%512j",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.critical(f"squeue failed: {result.stderr}")
        return None
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def count_idle_gpus(partition: str | None = None, buffer_gpus: int | None = None) -> int:
    """Count the number of idle GPUs on this partition."""
    if partition is None:
        partition = config.partition
    if buffer_gpus is None:
        buffer_gpus = config.buffer_gpus
    result = subprocess.run(
        ["sinfo", "-p", partition, "-N", "--Format=gres,gresused", "--noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.critical(f"sinfo failed: {result.stderr}")
        return 0  # Return 0 to prevent launching more jobs if sinfo fails
    total = 0
    used = 0
    for line in result.stdout.strip().split("\n"):
        # Each line has two fixed-width columns (gres, gresused). When the gres
        # value is long it truncates and runs directly into gresused with no
        # whitespace, so we can't split on spaces. Instead, extract all
        # gpu:<type>:<count> counts from the line; first match = total,
        # second match = used (mirrors the grep-oP approach in grunk).
        counts = re.findall(r"gpu:\w+:(\d+)", line)
        if len(counts) >= 2:
            total += int(counts[0])
            used += int(counts[1])

    avail = total - used
    return max(avail - buffer_gpus, 0)  # Don't report negative available GPUs, just 0


def count_pending_jobs(partition: str | None = None) -> int:
    """Count this user's pending jobs on the partition."""
    if partition is None:
        partition = config.partition
    result = subprocess.run(
        [
            "squeue",
            "-p",
            partition,
            "-u",
            getpass.getuser(),
            "--state=PENDING",
            "--noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.critical(f"squeue failed: {result.stderr}")
        return 999999  # Return a large number to prevent launching more jobs if squeue fails
    return len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0


def resources_available(partition: str | None = None, buffer_gpus: int | None = None) -> bool:
    idle_gpus = count_idle_gpus(partition, buffer_gpus)
    pending_jobs = count_pending_jobs(partition)
    # Pending jobs will consume idle GPUs once scheduled, so only launch more
    # when there are more idle GPUs than jobs already queued.
    return idle_gpus > pending_jobs


def launch_train_parallel(
    train_config: TrainConfig,
    job_name: str,
    result_path: Path | str,
    log_dir: Path | str,
    partition: str | None = None,
) -> None:
    """Submit a SLURM job that runs training serially and saves results to result_path.

    Args:
        train_config: The training configuration.
        job_name: The SLURM job name.
        result_path: Path where the result xarray Dataset will be saved as NetCDF.
        log_dir: Directory for SLURM stdout/stderr logs.
        partition: The SLURM partition to submit to.
    """
    if partition is None:
        partition = config.partition
    # All temp files must live on the shared filesystem (not /tmp which is
    # node-local), so that compute nodes can read them.
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Serialize the train config so the job can reconstruct it
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        prefix=f"{job_name}_config_",
        dir=log_dir,
    ) as f:
        yaml.dump(train_config.model_dump(), f, indent=4)
        config_path = f.name

    # This runs in a fresh process, so the global transport_study.config
    # singleton isn't loaded there - serialize it too, or any code path that
    # touches config.* (e.g. resolving dataset_paths) raises "Config not loaded"
    study_config_path = Path(log_dir) / f"{job_name}_study_config.toml"
    config.save(study_config_path)

    # Write the serial training logic as a small Python script
    py_script = f"""\
import yaml
from pathlib import Path
from popsim.ml import TrainConfig
from popsim.ml.launch import launch_train
{_config_reload_script(study_config_path)}
with open({config_path!r}) as f:
    train_config = TrainConfig(**yaml.safe_load(f))

_, _, _, _, result_dict = launch_train(train_config)
if result_dict is None:
    # Trainer hit its wall-clock budget and saved the latest checkpoint. Exit
    # cleanly with no result file, the orchestrator resubmits and the next job
    # resumes from that checkpoint.
    print("Training stopped at the wall-clock budget before finishing, resubmitted job will resume.")
else:
    ds = result_dict["test/study_results"]
    Path({str(result_path)!r}).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf({str(result_path)!r})
Path({str(config_path)!r}).unlink()
Path({str(study_config_path)!r}).unlink()
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix=f"{job_name}_script_", dir=log_dir) as f:
        f.write(py_script)
        script_path = f.name
    log_path = log_dir / f"{job_name}.log"

    sbatch_script = f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={config.train_time_limit}
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --exclude=node2301,node2101
#SBATCH --output={log_path}
#SBATCH --error={log_path}

# Save results to netcdf only; no need to sync wandb runs online from batch jobs
export WANDB_MODE=offline

{sys.executable} {script_path}
exit_code=$?
rm -f {script_path}
exit $exit_code
"""

    result = subprocess.run(
        ["sbatch"],
        input=sbatch_script,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"sbatch failed for job {job_name}: {result.stderr}")
    else:
        logger.info(f"Submitted training job {job_name}: {result.stdout.strip()}")


def launch_agent_parallel(
    train_config: TrainConfig,
    sweep_id: str,
    kwargs_agent: dict,
    job_name: str,
    log_dir: Path | str,
    partition: str | None = None,
) -> None:
    """Submit a SLURM job that launches a W&B agent for an existing sweep.

    Args:
        train_config: The training configuration.
        sweep_id: Existing W&B sweep ID.
        kwargs_agent: Keyword arguments for ``wandb.agent``.
        job_name: The SLURM job name.
        log_dir: Directory for SLURM stdout/stderr logs.
        partition: The SLURM partition to submit to.
    """
    if partition is None:
        partition = config.partition
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_dir = tempfile.mkdtemp(prefix=f"{job_name}_", dir=log_dir)
    config_path = Path(run_dir) / "config.yaml"
    study_config_path = Path(run_dir) / "study_config.toml"
    script_path = Path(run_dir) / "run_agent.py"
    log_path = Path(run_dir) / "slurm.log"

    with open(config_path, "w") as f:
        yaml.dump(train_config.model_dump(), f, indent=4)

    # This runs in a fresh process, so the global transport_study.config
    # singleton isn't loaded there - serialize it too, or any code path that
    # touches config.* (e.g. resolving dataset_paths) raises "Config not loaded"
    config.save(study_config_path)

    py_script = f"""\
import yaml
from pathlib import Path
from popsim.ml import TrainConfig
from popsim.ml.launch import launch_agent
{_config_reload_script(study_config_path)}
with open({config_path!r}) as f:
    train_config = TrainConfig(**yaml.safe_load(f))

launch_agent(
    train_config,
    {sweep_id!r},
    kwargs_agent={kwargs_agent!r},
)
Path({config_path!r}).unlink()
Path({str(study_config_path)!r}).unlink()
"""

    with open(script_path, "w") as f:
        f.write(py_script)

    sbatch_script = f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={config.train_time_limit}
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --exclude=node2301,node2101
#SBATCH --output={log_path}
#SBATCH --error={log_path}

{sys.executable} {script_path}
exit_code=$?
rm -f {script_path}
exit $exit_code
"""

    result = subprocess.run(
        ["sbatch"],
        input=sbatch_script,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"sbatch failed for job {job_name}: {result.stderr}")
    else:
        logger.info(f"Submitted agent job {job_name}: {result.stdout.strip()}")


def launch_profile_analysis_parallel(study, case) -> None:
    """Submit a CPU SLURM job computing one case's stage metrics and case report.

    Analysis (metrics aggregation, PDF pages, GIF frames) is matplotlib and
    numpy bound with no GPU work, so it goes to config.analysis_partition when
    set (config.partition otherwise) and requests no GPU.
    """
    partition = config.analysis_partition or config.partition
    log_dir = study.working_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    job_name = study.analysis_job_name(case)
    case_str = str(case)

    # This runs in a fresh process, so the global transport_study.config
    # singleton isn't loaded there. The study class reloads it from this TOML
    # (per-job file: jobs clean up after themselves, so sharing one would race)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False, prefix=f"{job_name}_study_config_", dir=log_dir) as f:
        study_config_path = f.name
    config.save(Path(study_config_path))

    study_cls = type(study)
    py_script = f"""\
from pathlib import Path
from {_importable_module(study_cls)} import {study_cls.__name__}
from transport_study.profile_transfer.case_reports import generate_case_report
from transport_study.profile_transfer.study_metrics import compute_and_save_case_metrics

study = {study_cls.__name__}(Path({study_config_path!r}))
case = next(c for c in study.cases if str(c) == {case_str!r})

case_metrics = compute_and_save_case_metrics(study, case)
if case_metrics.data_vars:
    generate_case_report(study, case, study.figure_dir)
Path({study_config_path!r}).unlink()
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix=f"{job_name}_", dir=log_dir) as f:
        f.write(py_script)
        script_path = f.name
    log_path = log_dir / f"{job_name}.log"

    sbatch_script = f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={config.analysis_time_limit}
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --output={log_path}
#SBATCH --error={log_path}

export MPLBACKEND=Agg
{sys.executable} {script_path}
exit_code=$?
rm -f {script_path}
exit $exit_code
"""

    result = subprocess.run(["sbatch"], input=sbatch_script, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed for analysis job {job_name}: {result.stderr}")
    else:
        logger.info(f"Submitted analysis job {job_name}: {result.stdout.strip()}")


def launch_trajopt_case_parallel(
    trajopt,
    case,
) -> None:
    """Submit a SLURM job that trains and generates output for a single trajectory optimization case."""
    log_dir = trajopt.working_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    init_kwargs = {
        "name": trajopt.name,
        "working_dir_base": trajopt.working_dir.parent,
        "profile_module_checkpoint_dir": trajopt.profile_module_checkpoint_dir,
        "traj_times": trajopt.traj_times,
        "max_num_traj_times": trajopt.max_num_traj_times,
    }
    case_str = str(case)

    py_script = f"""\
from transport_study.trajectory_optimization.optimize import TrajectoryOptimization

trajopt = TrajectoryOptimization(**{init_kwargs!r})
case = next(c for c in trajopt.cases if str(c) == {case_str!r})

if not trajopt.checkpoint_dir(case).exists():
    trajopt.run_case(case)

if not trajopt.output_path(case).exists():
    trajopt.output_optimized_trajectory(case)
"""

    job_name = trajopt.train_job_name(case)
    log_path = log_dir / f"{job_name}.log"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix=f"{job_name}_", dir=log_dir) as f:
        f.write(py_script)
        script_path = f.name

    sbatch_script = f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={config.partition}
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --exclude=node2301,node2101
#SBATCH --output={log_path}
#SBATCH --error={log_path}

export WANDB_MODE=offline
{sys.executable} {script_path}
exit_code=$?
rm -f {script_path}
exit $exit_code
"""

    result = subprocess.run(["sbatch"], input=sbatch_script, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed for {job_name}: {result.stderr}")
    else:
        logger.info(f"Submitted SLURM job {job_name}: {result.stdout.strip()}")
