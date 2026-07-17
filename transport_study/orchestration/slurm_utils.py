import functools
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


def query_partitions() -> str:
    """Comma-separated partition list for squeue queries tracking this study's jobs.

    Covers the primary partition plus the spillover partitions when configured,
    so in-flight checks see jobs regardless of where they were submitted.
    squeue -p accepts a comma-separated list.
    """
    partitions = [config.partition, *config.spillover_partitions]
    return ",".join(p for p in partitions if p)


def count_running_jobs(job_name: str, partition: str | None = None) -> int:
    """Run squeue to list running jobs on the partition(s) with the specific name.

    Defaults to the primary plus spillover partitions so a case whose job
    spilled over still counts as in flight.
    """
    if partition is None:
        partition = query_partitions()
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
    """Names of this user's running and pending jobs, in one squeue call.

    Orchestration loops poll job state for every case each pass. One squeue
    call returning all names (checked by set membership) replaces hundreds of
    per-case squeue calls. Returns None when squeue fails, so callers can tell
    "no jobs" apart from "scheduler unreachable" and hold off launching.
    Defaults to the primary plus spillover partitions.
    """
    if partition is None:
        partition = query_partitions()
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


def get_running_job_elapsed_s(partition: str | None = None) -> dict[str, int] | None:
    """Elapsed running time in seconds for this user's RUNNING jobs, keyed by job name.

    Pending jobs are excluded, they haven't started accumulating epochs yet.
    Returns None when squeue fails, so callers can tell "no running jobs" apart
    from "scheduler unreachable" (mirrors get_running_job_names).
    """
    if partition is None:
        partition = query_partitions()
    result = subprocess.run(
        [
            "squeue",
            "-p",
            partition,
            "-u",
            getpass.getuser(),
            "--state=RUNNING",
            "--noheader",
            # Default %j truncates long names, and case names run long
            "--format=%512j %M",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.critical(f"squeue failed: {result.stderr}")
        return None
    elapsed: dict[str, int] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # %M has no internal whitespace, so the token after the last space is
        # the elapsed time and everything before it (stripped of the %512j
        # padding) is the job name
        name, _, time_str = line.rpartition(" ")
        elapsed_s = parse_slurm_time_s(time_str.strip())
        if elapsed_s is not None:
            elapsed[name.strip()] = elapsed_s
    return elapsed


def cancel_job(job_name: str, partition: str | None = None) -> None:
    """Cancel this user's running job(s) with the given name.

    Used by the stuck-job watchdog to kill a deadlocked training job so the
    orchestration loop's normal relaunch path can resubmit it fresh.
    """
    if partition is None:
        partition = query_partitions()
    # Unlike squeue -p, scancel -p takes a single partition name and treats a
    # comma-separated list as one literal (nonexistent) name, matching no jobs
    # while still exiting 0, so cancel each partition separately
    for single_partition in partition.split(","):
        result = subprocess.run(
            [
                "scancel",
                "-p",
                single_partition,
                "-u",
                getpass.getuser(),
                "-n",
                job_name,
                "--state=RUNNING",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"scancel failed for job {job_name}: {result.stderr}")


def count_idle_gpus(partition: str | None = None, buffer_gpus: int | None = None) -> int:
    """Count the number of idle GPUs on this partition.

    An 'idle' GPU is one that is either not allocated to any job, or is allocated to a job that is not part of this partition.
    This is so we actively boot preemptable jobs from this partition.
    """
    if partition is None:
        partition = config.partition
    if buffer_gpus is None:
        buffer_gpus = config.buffer_gpus
    sinfo_result = subprocess.run(
        ["sinfo", "-p", partition, "-N", "--Format=gres", "--noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if sinfo_result.returncode != 0:
        logger.critical(f"sinfo failed: {sinfo_result.stderr}")
        return 0  # Return 0 to prevent launching more jobs if sinfo fails
    total = 0
    for line in sinfo_result.stdout.strip().split("\n"):
        counts = re.findall(r"gpu:\w+:(\d+)", line)
        if counts:
            total += int(counts[0])

    # gresused (from sinfo) counts GPUs busy on the node regardless of which
    # partition the job landed in, so a node shared with another partition
    # would look fully busy even when this partition's jobs hold none of it.
    # Instead, sum GPU allocations only from jobs actually RUNNING in this
    # partition (squeue resolves %P to the single assigned partition for
    # running jobs, unlike the requested-partition list shown for pending ones).
    squeue_result = subprocess.run(
        ["squeue", "-p", partition, "--states=RUNNING", "-o", "%b", "--noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if squeue_result.returncode != 0:
        logger.critical(f"squeue failed: {squeue_result.stderr}")
        return 0  # Return 0 to prevent launching more jobs if squeue fails
    used = 0
    for line in squeue_result.stdout.strip().split("\n"):
        for count in re.findall(r"gpu:(?:\w+:)?(\d+)", line):
            used += int(count)

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


def count_user_jobs() -> int:
    """This user's running + pending jobs across all partitions.

    Every job counts toward the association/QOS MaxSubmit ceilings no matter
    which partition it went to, so the spillover budget is based on this total.
    """
    result = subprocess.run(
        [
            "squeue",
            "-u",
            getpass.getuser(),
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
    return len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0


@functools.cache
def _partition_info(partition: str) -> dict[str, str]:
    """key=value fields from scontrol show partition. Partition limits are
    static for the lifetime of a study run, so results are cached."""
    result = subprocess.run(
        ["scontrol", "show", "partition", partition],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning(f"scontrol show partition {partition} failed: {result.stderr}")
        return {}
    return dict(token.split("=", 1) for token in result.stdout.split() if "=" in token)


@functools.cache
def partition_user_gpu_cap(partition: str) -> int | None:
    """Per-user GPU cap on a partition (its QOS MaxTRESPU gres/gpu), None if uncapped.

    E.g. mit_preemptable's QOS allows 4 running GPUs per user, mit_normal_gpu's
    allows 2. Submitting more jobs than this just parks them pending on the QOS
    limit, so the spillover logic treats it as that partition's submission cap.
    """
    qos = _partition_info(partition).get("QoS")
    if qos in (None, "N/A"):
        return None
    result = subprocess.run(
        ["sacctmgr", "-nP", "show", "qos", qos, "format=MaxTRESPU"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning(f"sacctmgr show qos {qos} failed: {result.stderr}")
        return None
    match = re.search(r"gres/gpu=(\d+)", result.stdout)
    return int(match.group(1)) if match else None


def parse_slurm_time_s(time_str: str | None) -> int | None:
    """Seconds from a SLURM time string like 06:00:00 or 2-00:00:00, None if unlimited."""
    if not time_str or time_str.upper() in ("UNLIMITED", "NONE", "N/A"):
        return None
    days = 0
    if "-" in time_str:
        day_str, time_str = time_str.split("-", 1)
        days = int(day_str)
    parts = [int(p) for p in time_str.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def format_slurm_time(seconds: int) -> str:
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def partition_time_limit_s(partition: str) -> int | None:
    """Partition MaxTime in seconds, None if unlimited or unknown."""
    return parse_slurm_time_s(_partition_info(partition).get("MaxTime"))


def count_user_gpus(partition: str) -> int:
    """This user's allocated + requested GPUs among running and pending jobs on a partition."""
    result = subprocess.run(
        [
            "squeue",
            "-p",
            partition,
            "-u",
            getpass.getuser(),
            "--state=RUNNING,PENDING",
            "--noheader",
            "-O",
            "tres-alloc:200",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.critical(f"squeue failed: {result.stderr}")
        return 999999  # Return a large number to prevent launching more jobs if squeue fails
    # Generic gres/gpu=N only; the typed gres/gpu:<type>=N entry would double count
    return sum(int(n) for n in re.findall(r"gres/gpu=(\d+)", result.stdout))


def spillover_budget() -> int:
    """Jobs still submittable anywhere before hitting the user-wide MaxSubmit ceiling."""
    if not config.spillover_partitions:
        return 0
    return max(config.max_user_jobs - config.spillover_job_headroom - count_user_jobs(), 0)


def spillover_slots(partition: str) -> int:
    """Jobs still worth submitting to this spillover partition right now.

    Capped by the partition's per-user GPU allowance when its QOS has one
    (assumes one GPU per job, which matches every template here). Without a
    cap, only submit what could start immediately, so an unbounded partition
    doesn't accumulate a deep pending queue either.
    """
    cap = partition_user_gpu_cap(partition)
    if cap is not None:
        return max(cap - count_user_gpus(partition), 0)
    idle = count_idle_gpus(partition, buffer_gpus=0)
    pending = count_pending_jobs(partition)
    return max(idle - pending, 0)


def pick_partition() -> str | None:
    """Partition the next GPU job should go to, or None to hold off launching.

    The primary partition wins while it has idle GPUs beyond buffer_gpus.
    Otherwise the spillover partitions are tried in configured order, each
    limited to its own slot count, all limited by the user-wide job ceiling.
    """
    if resources_available():
        return config.partition
    if spillover_budget() <= 0:
        return None
    for spillover in config.spillover_partitions:
        if spillover_slots(spillover) > 0:
            return spillover
    return None


def clamp_time_for_partition(partition: str, train_config: TrainConfig) -> tuple[str, TrainConfig]:
    """sbatch --time and in-job wall budget fitted to the partition's MaxTime.

    Some spillover partitions cap walltime below train_time_limit
    and reject over-limit requests outright.
    The in-job budget shrinks by the same amount, preserving the
    margin the trainer needs to checkpoint and exit before SLURM kills the job.
    """
    requested_s = parse_slurm_time_s(config.train_time_limit)
    limit_s = partition_time_limit_s(partition)
    if requested_s is None or limit_s is None or limit_s >= requested_s:
        return config.train_time_limit, train_config
    checkpoint_margin_s = max(requested_s - config.train_wall_budget_s, 0)
    clamped_budget_s = float(max(limit_s - checkpoint_margin_s, 600))
    if train_config.max_wall_seconds is not None:
        clamped_budget_s = min(clamped_budget_s, train_config.max_wall_seconds)
    logger.info(
        f"Partition {partition} MaxTime {format_slurm_time(limit_s)} is below the requested "
        f"{config.train_time_limit}, clamping job time and wall budget ({clamped_budget_s:.0f} s)"
    )
    return format_slurm_time(limit_s), train_config.model_copy(update={"max_wall_seconds": clamped_budget_s})


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
    time_limit, train_config = clamp_time_for_partition(partition, train_config)
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
from loguru import logger
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
    logger.info("Training stopped at the wall-clock budget before finishing, resubmitted job will resume.")
else:
    ds = result_dict["test/study_results"]
    result_path = Path({str(result_path)!r})
    result_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp name then rename so a partially written file is never
    # visible at the result path, whose existence marks the case done
    tmp_path = result_path.with_name(result_path.name + ".tmp")
    ds.to_netcdf(tmp_path)
    tmp_path.replace(result_path)
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
#SBATCH --time={time_limit}
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --output={log_path}
#SBATCH --error={log_path}
#SBATCH --open-mode=append
#SBATCH --requeue

# Resubmitted attempts share this log path, mark where each one starts
echo "=== $(date) job $SLURM_JOB_ID ({job_name}) start ==="

# Save results to netcdf only; no need to sync wandb runs online from batch jobs
export WANDB_MODE=offline

# Single-thread host BLAS/OpenMP. Reservoir init runs np.linalg.eigvals whose
# OpenBLAS threadpool can deadlock nondeterministically under core contention.
# eigvals is tiny so single-threaded costs nothing.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Compile XLA GPU programs serially. Parallel compilation threads can deadlock
# under the 4-cpu cgroup, stalling the job during initial compilation.
export XLA_FLAGS="${{XLA_FLAGS:+$XLA_FLAGS }}--xla_gpu_force_compilation_parallelism=1"

# The driver may run with JAX_PLATFORMS=cpu, which leaks in via --export=ALL.
# Pin this GPU job to cuda. Listing platforms explicitly makes jax raise if cuda
# fails to init, so a broken GPU env fails loudly instead of training on cpu.
# cpu stays second in the list only so host-side helpers like jax.devices("cpu")
# keep working. All compute defaults to cuda.
export JAX_PLATFORMS=cuda,cpu

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
    time_limit, train_config = clamp_time_for_partition(partition, train_config)
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
with open({str(config_path)!r}) as f:
    train_config = TrainConfig(**yaml.safe_load(f))

launch_agent(
    train_config,
    {sweep_id!r},
    kwargs_agent={kwargs_agent!r},
)
Path({str(config_path)!r}).unlink()
Path({str(study_config_path)!r}).unlink()
"""

    with open(script_path, "w") as f:
        f.write(py_script)

    sbatch_script = f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --output={log_path}
#SBATCH --error={log_path}
#SBATCH --requeue

# Single-thread host BLAS/OpenMP. Reservoir init runs np.linalg.eigvals whose
# OpenBLAS threadpool can deadlock nondeterministically under core contention.
# eigvals is tiny so single-threaded costs nothing.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Compile XLA GPU programs serially. Parallel compilation threads can deadlock
# under the 4-cpu cgroup, stalling the job during initial compilation.
export XLA_FLAGS="${{XLA_FLAGS:+$XLA_FLAGS }}--xla_gpu_force_compilation_parallelism=1"

# The driver may run with JAX_PLATFORMS=cpu, which leaks in via --export=ALL.
# Pin this GPU job to cuda. Listing platforms explicitly makes jax raise if cuda
# fails to init, so a broken GPU env fails loudly instead of training on cpu.
# cpu stays second in the list only so host-side helpers like jax.devices("cpu")
# keep working. All compute defaults to cuda.
export JAX_PLATFORMS=cuda,cpu

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


def launch_case_analysis_parallel(study, case) -> None:
    """Submit a CPU SLURM job computing one case's stage metrics and case report.

    Analysis (metrics aggregation, PDF pages, GIF frames) is matplotlib and
    numpy bound with no GPU work, so it goes to config.analysis_partition when
    set (config.partition otherwise) and requests no GPU.

    The job imports the study's analysis modules from its
    ANALYSIS_METRICS_MODULE / ANALYSIS_REPORTS_MODULE ClassVars: the metrics
    module must export ``compute_and_save_case_metrics(study, case)`` and the
    reports module ``generate_case_report(study, case, figure_dir)``.
    """
    partition = config.analysis_partition or config.partition
    log_dir = study.working_dir / "logs" / "logs_analysis"
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
from {study_cls.ANALYSIS_REPORTS_MODULE} import generate_case_report
from {study_cls.ANALYSIS_METRICS_MODULE} import compute_and_save_case_metrics

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
#SBATCH --open-mode=append

# Resubmitted attempts share this log path, mark where each one starts
echo "=== $(date) job $SLURM_JOB_ID ({job_name}) start ==="

export MPLBACKEND=Agg
# Single-thread host BLAS/OpenMP. Reservoir init runs np.linalg.eigvals whose
# OpenBLAS threadpool can deadlock nondeterministically under core contention.
# eigvals is tiny so single-threaded costs nothing.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Analysis is cpu-only. Pin jax so behavior is identical regardless of driver env
# and cuda plugin init is skipped on cpu partitions.
export JAX_PLATFORMS=cpu

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
# init_kwargs contains Path objects whose repr is PosixPath('...')
from pathlib import PosixPath  # noqa: F401

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
#SBATCH --output={log_path}
#SBATCH --error={log_path}
#SBATCH --open-mode=append
#SBATCH --requeue

# Resubmitted attempts share this log path, mark where each one starts
echo "=== $(date) job $SLURM_JOB_ID ({job_name}) start ==="

export WANDB_MODE=offline
# Single-thread host BLAS/OpenMP. Reservoir init runs np.linalg.eigvals whose
# OpenBLAS threadpool can deadlock nondeterministically under core contention.
# eigvals is tiny so single-threaded costs nothing.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
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
