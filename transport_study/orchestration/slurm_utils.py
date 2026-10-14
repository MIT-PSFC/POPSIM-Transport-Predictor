import functools
import getpass
import inspect
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Collection
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import yaml
from loguru import logger
from popsim.ml import TrainConfig

from transport_study.config import config, slurm_mem_MB

# A hung SLURM controller must not freeze the orchestration loop
SLURM_COMMAND_TIMEOUT_S = 120

# Environment shared by every sbatch script
SINGLE_THREAD_BLAS_ENV = """\
# Single-thread host BLAS and OpenMP.
# Reservoir init runs np.linalg.eigvals, whose OpenBLAS threadpool can deadlock under core contention.
# eigvals is tiny, so a single thread costs nothing.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
"""

# Environment shared by every GPU sbatch script (training and sweep agents).
# Not an f-string, the shell parameter expansions must land literally.
GPU_JOB_ENV = (
    SINGLE_THREAD_BLAS_ENV
    + """
# Compile XLA GPU programs serially, parallel compile threads can deadlock under the 4-cpu cgroup.
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_force_compilation_parallelism=1"

# The driver may run with JAX_PLATFORMS=cpu, which leaks in through --export=ALL.
# Listing cuda first makes jax raise if cuda fails to init, instead of training on cpu.
# cpu stays listed so host-side helpers like jax.devices("cpu") keep working.
export JAX_PLATFORMS=cuda,cpu

# One preallocated XLA pool at 90% of VRAM, growth-mode allocation fragments the pool.
# The transport transformer at its sweep maximum peaks at 72 GB at TRANSPORT_BATCH_SIZE (2026-10-13),
# over the 64 GiB of an 80% pool on an A100-80GB.
# The other 10% holds the CUDA context and kernel images (~0.7 GB measured).
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
"""
)

# Environment of every CPU job (training, sweep agents, analysis).
# Pinning jax to the cpu skips the CUDA plugin, whose version check hangs at the first JAX op on a node without a GPU.
CPU_JOB_ENV = SINGLE_THREAD_BLAS_ENV + "export JAX_PLATFORMS=cpu\n"

# Start banner of a CPU training or agent job.
# nproc without the OpenMP variables is the cgroup's core count, which XLA's CPU thread pool sizes itself to.
# An OMP_NUM_THREADS inherited through --export=ALL would otherwise override it.
# Not an f-string, job_name is filled in with str.format.
CPU_JOB_BANNER = """\
echo "=== $(date) job $SLURM_JOB_ID ({job_name}) start ==="
echo "=== node $SLURMD_NODENAME cpus $(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc) ==="
"""

# Start banner of a GPU job.
# SLURM_JOB_GPUS is the physical index on the node, the uuid names the exact card for bad-hardware reports.
# Not an f-string, job_name is filled in with str.format.
GPU_JOB_BANNER = """\
echo "=== $(date) job $SLURM_JOB_ID ({job_name}) start ==="
echo "=== node $SLURMD_NODENAME gpu ${{SLURM_JOB_GPUS:-$CUDA_VISIBLE_DEVICES}} $(nvidia-smi --query-gpu=name,uuid --format=csv,noheader 2>/dev/null || echo nvidia-smi unavailable) ==="
"""


def _run_slurm(cmd: list[str], stdin: str | None = None) -> subprocess.CompletedProcess | None:
    """The completed SLURM command, None (logged) when it timed out or SLURM is not installed."""
    try:
        return subprocess.run(cmd, input=stdin, check=False, capture_output=True, text=True, timeout=SLURM_COMMAND_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        logger.critical(f"{cmd[0]} timed out after {SLURM_COMMAND_TIMEOUT_S} s")
        return None
    except FileNotFoundError:
        logger.critical(f"{cmd[0]} not found, SLURM is not installed on this machine")
        return None


def _slurm_stdout(cmd: list[str], level: str = "CRITICAL") -> str | None:
    """stdout of a SLURM query, None (logged at level) when it failed or timed out."""
    result = _run_slurm(cmd)
    if result is None:
        return None
    if result.returncode != 0:
        logger.log(level, f"{cmd[0]} failed: {result.stderr}")
        return None
    return result.stdout


def _count_lines(stdout: str) -> int:
    return len(stdout.strip().split("\n")) if stdout.strip() else 0


def _cache_resolved(fn):
    """Cache fn per argument tuple, except a None result (a failed SLURM query), which the next call retries.

    The partition inventory and limits are static for a study run, and so is the allocation of one run of a job,
    but a cached failure would disable the spillover limits or the GPU type exclusion for the rest of it.
    """
    resolved: dict[tuple, object] = {}

    @functools.wraps(fn)
    def wrapper(*args):
        if args not in resolved:
            result = fn(*args)
            if result is None:
                return None
            resolved[args] = result
        return resolved[args]

    return wrapper


def sbatch_script(
    job_name: str,
    partition: str,
    time_limit: str | None,
    log_path: Path | str,
    script_path: Path | str,
    setup: str,
    mem: str = "120G",
    gres: str | None = None,
    exclude: str | None = None,
    requeue: bool = False,
    cpus: int = 4,
) -> str:
    """sbatch script that runs script_path with this interpreter after the setup shell lines, then deletes it.

    Resubmitted attempts append to the same log_path.
    A time_limit of None leaves the partition default, exclude is a comma-separated node list.
    """
    time_line = f"\n#SBATCH --time={time_limit}" if time_limit else ""
    gres_line = f"\n#SBATCH --gres={gres}" if gres else ""
    exclude_line = f"\n#SBATCH --exclude={exclude}" if exclude else ""
    requeue_line = "\n#SBATCH --requeue" if requeue else ""
    return f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}{time_line}{gres_line}{exclude_line}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --export=ALL
#SBATCH --output={log_path}
#SBATCH --error={log_path}
#SBATCH --open-mode=append{requeue_line}

{setup}
{sys.executable} {script_path}
exit_code=$?
rm -f {script_path}
exit $exit_code
"""


# What sbatch prints for an accepted job
SUBMITTED_JOB_PATTERN = re.compile(r"Submitted batch job (\d+)")


def _submitted_job_id(sbatch_stdout: str, job_name: str) -> int | None:
    """The job id an accepted sbatch printed, None (logged) when its output has none."""
    match = SUBMITTED_JOB_PATTERN.search(sbatch_stdout)
    if match is None:
        logger.error(f"sbatch accepted {job_name} but printed no job id: {sbatch_stdout!r}")
        return None
    return int(match.group(1))


def submit_sbatch(script: str, job_name: str, kind: str) -> int | None:
    """Submit an sbatch script, returns the job id SLURM accepted it under, None when it did not."""
    result = _run_slurm(["sbatch"], stdin=script)
    if result is None:
        return None
    if result.returncode != 0:
        logger.error(f"sbatch failed for {kind} job {job_name}: {result.stderr}")
        return None
    logger.info(f"Submitted {kind} job {job_name}: {result.stdout.strip()}")
    return _submitted_job_id(result.stdout, job_name)


def importable_module(cls: type) -> str:
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
        f"from {importable_module(cls)} import {top_name}\n"
        "from transport_study.config import load_config\n"
        f"study_config = {cls.__qualname__}.from_toml(Path({str(study_config_path)!r}))\n"
        "load_config(study_config)\n"
    )
    return code_str


def cpu_partitions() -> list[str]:
    """The CPU partitions for the cpu_model_types, the primary first, empty without a cpu_partition."""
    if not config.cpu_partition:
        return []
    return [config.cpu_partition, *config.cpu_fallback_partitions]


def query_partitions() -> str:
    """Comma-separated partition list for squeue queries tracking this study's jobs.

    Covers the primary partition plus the GPU spillover and CPU partitions when configured,
    so in-flight checks see jobs regardless of where they were submitted.
    squeue -p accepts a comma-separated list.
    """
    partitions = [config.partition, *config.spillover_partitions, *cpu_partitions()]
    return ",".join(dict.fromkeys(p for p in partitions if p))


# squeue states of a job that has not left the queue yet.
# A COMPLETING job has finished training but still holds its name, a REQUEUED one runs again
IN_FLIGHT_JOB_STATES = "PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED,REQUEUED"


def count_running_jobs(job_name: str, partition: str | None = None) -> int | None:
    """Jobs with this name still in the queue on the partition(s) (IN_FLIGHT_JOB_STATES), None if squeue fails.

    Defaults to the primary plus spillover partitions so a case whose job
    spilled over still counts as in flight.
    """
    if partition is None:
        partition = query_partitions()
    stdout = _slurm_stdout(["squeue", "-p", partition, "-n", job_name, f"--state={IN_FLIGHT_JOB_STATES}", "--noheader"])
    if stdout is None:
        return None
    return _count_lines(stdout)


def get_running_job_names() -> set[str] | None:
    """Names of this user's jobs still in the queue (IN_FLIGHT_JOB_STATES) on every partition, in one squeue call.

    Orchestration loops poll job state for every case each pass. One squeue
    call returning all names (checked by set membership) replaces hundreds of
    per-case squeue calls. Returns None when squeue fails, so callers can tell
    "no jobs" apart from "scheduler unreachable" and hold off launching.
    Every job name ends with its study's name, so no partition filter is needed,
    and a job on a partition the config no longer names still counts.
    """
    # Default %j truncates long names, and case names run long
    stdout = _slurm_stdout(["squeue", "-u", getpass.getuser(), f"--state={IN_FLIGHT_JOB_STATES}", "--noheader", "--format=%512j"])
    if stdout is None:
        return None
    return {line.strip() for line in stdout.splitlines() if line.strip()}


# sacct states of a job that will not run again (CANCELLED prints as "CANCELLED by <uid>")
TERMINAL_JOB_STATES = ("BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT")


class JobEnd(NamedTuple):
    """A job's sacct state and wall-clock end time, the end None while the job can still run."""

    state: str
    end_time: float | None


def job_end_states(job_ids: Collection[int]) -> dict[int, JobEnd] | None:
    """sacct's state and end of each of job_ids, in one call, None when sacct fails.

    A job sacct has no record of yet is missing from the dict.
    The end is None until the state is one of TERMINAL_JOB_STATES.
    A terminal job without an end time reads as having ended at once (end 0).
    """
    job_id_list = ",".join(str(job_id) for job_id in job_ids)
    stdout = _slurm_stdout(["sacct", "-X", "-n", "-P", "-j", job_id_list, "-o", "JobID,State,End"], level="WARNING")
    if stdout is None:
        return None
    end_states = {}
    for line in stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) != 3 or not fields[0].isdigit():
            continue
        job_id_field, state_field, end_field = fields
        state = state_field.split(" ")[0]
        end_time = None
        if state in TERMINAL_JOB_STATES:
            # sacct prints local times without a zone, the same clock as time.time()
            end_time = datetime.fromisoformat(end_field).timestamp() if end_field[:1].isdigit() else 0.0
        end_states[int(job_id_field)] = JobEnd(state, end_time)
    return end_states


# Result files can lag job exit by ~1 min on NFS, so the work of a job waits this long after the job ended before it is relaunched
RELAUNCH_GRACE_S = 180

# sacct records a job seconds after its submission, a submitted job it still has no record of this long after counts as gone
SUBMIT_ACCOUNTING_LAG_S = 300


class SubmittedJob(NamedTuple):
    """A job SLURM accepted, its id and the wall-clock time of the submission."""

    job_id: int
    submitted_at: float


def held_job_names(submitted_jobs: dict[str, SubmittedJob], running_job_names: set[str]) -> set[str]:
    """The submitted jobs that left the queue too recently for their work to be relaunched.

    submitted_jobs maps each job name to the job, running_job_names is the pass's squeue snapshot.
    A job is held while sacct has no record of it yet (accounting lags submission by seconds),
    while sacct still shows it in a non-terminal state (squeue and sacct can disagree for a moment),
    until RELAUNCH_GRACE_S past its end, and through any call whose sacct query fails.
    A job sacct never records is released SUBMIT_ACCOUNTING_LAG_S after its submission.
    A released job is removed from submitted_jobs, so each is looked up only until it is resolved.
    The hold needs no snapshot to have seen the job, so a job submitted and finished between two snapshots is held too.
    """
    gone_jobs = {job_name: job for job_name, job in submitted_jobs.items() if job_name not in running_job_names}
    if not gone_jobs:
        return set()
    gone_job_ids = [job.job_id for job in gone_jobs.values()]
    end_states = job_end_states(gone_job_ids)
    if end_states is None:
        logger.warning("sacct failed, holding every submitted job that left the queue until the next pass")
        return set(gone_jobs)
    now = time.time()
    held = set()
    for job_name, job in gone_jobs.items():
        end_state = end_states.get(job.job_id)
        if end_state is None:
            still_held = now - job.submitted_at < SUBMIT_ACCOUNTING_LAG_S
        elif end_state.end_time is None:
            still_held = True
        else:
            still_held = now - end_state.end_time < RELAUNCH_GRACE_S
        if still_held:
            held.add(job_name)
        else:
            del submitted_jobs[job_name]
    return held


def recent_submitted_jobs(name_prefix: str, name_suffix: str) -> dict[str, SubmittedJob]:
    """This user's jobs named name_prefix...name_suffix that were queued or ran in the last RELAUNCH_GRACE_S.

    A restarted orchestrator tracks them in held_job_names, since its predecessor submitted them.
    The newest job wins when a name repeats.
    A failed sacct leaves them untracked (logged), as before the hold existed.
    """
    start_time = datetime.fromtimestamp(time.time() - RELAUNCH_GRACE_S).strftime("%Y-%m-%dT%H:%M:%S")
    stdout = _slurm_stdout(
        ["sacct", "-u", getpass.getuser(), "-X", "-n", "-P", "-S", start_time, "-o", "JobID,JobName%512"], level="WARNING"
    )
    if stdout is None:
        logger.warning(f"sacct failed, the {name_prefix}*{name_suffix} jobs of a previous orchestrator are not held")
        return {}
    job_ids: dict[str, int] = {}
    for line in stdout.splitlines():
        job_id_field, _, job_name = line.strip().partition("|")
        if job_id_field.isdigit() and job_name.startswith(name_prefix) and job_name.endswith(name_suffix):
            job_ids[job_name] = max(int(job_id_field), job_ids.get(job_name, 0))
    now = time.time()
    return {job_name: SubmittedJob(job_id, now) for job_name, job_id in job_ids.items()}


def get_running_job_elapsed_s(partition: str | None = None) -> dict[str, int] | None:
    """Elapsed running time in seconds for this user's RUNNING jobs, keyed by job name.

    Pending jobs are excluded, they haven't started accumulating epochs yet.
    Returns None when squeue fails, so callers can tell "no running jobs" apart
    from "scheduler unreachable" (mirrors get_running_job_names).
    Duplicate job names (e.g. multiple agent jobs) keep the largest value, as in get_pending_job_pending_s.
    """
    if partition is None:
        partition = query_partitions()
    # Default %j truncates long names, and case names run long
    stdout = _slurm_stdout(["squeue", "-p", partition, "-u", getpass.getuser(), "--state=RUNNING", "--noheader", "--format=%512j %M"])
    if stdout is None:
        return None
    elapsed: dict[str, int] = {}
    for line in stdout.splitlines():
        if not line.strip():
            continue
        # %M has no internal whitespace, so the token after the last space is
        # the elapsed time and everything before it (stripped of the %512j
        # padding) is the job name
        name, _, time_str = line.rpartition(" ")
        elapsed_s = parse_slurm_time_s(time_str.strip())
        if elapsed_s is not None:
            job_name = name.strip()
            elapsed[job_name] = max(elapsed_s, elapsed.get(job_name, 0))
    return elapsed


def get_pending_job_pending_s(partition: str | None = None) -> dict[str, int] | None:
    """Seconds spent in the PENDING state for this user's pending jobs, keyed by job name.

    Uses the squeue -O PendingTime field (no %-format code exists for it),
    which reports seconds pending directly instead of requiring submit-time
    parsing. Returns None when squeue fails, so callers can tell "no pending
    jobs" apart from "scheduler unreachable" (mirrors get_running_job_names).
    Duplicate job names (e.g. multiple agent jobs) keep the largest value.
    Defaults to the primary plus spillover partitions.
    """
    if partition is None:
        partition = query_partitions()
    # Wide Name field because case names run long and -O truncates at the given width.
    # PendingTime first so the name is the tail token.
    stdout = _slurm_stdout(
        ["squeue", "-p", partition, "-u", getpass.getuser(), "--state=PENDING", "--noheader", "-O", "PendingTime:20,Name:512"]
    )
    if stdout is None:
        return None
    pending: dict[str, int] = {}
    for line in stdout.splitlines():
        if not line.strip():
            continue
        # PendingTime has no internal whitespace, so the token before the
        # first space is the pending seconds and the rest is the job name
        time_str, _, name = line.strip().partition(" ")
        name = name.strip()
        if not name or not time_str.isdigit():
            continue
        pending[name] = max(int(time_str), pending.get(name, 0))
    return pending


def cancel_job(job_name: str, partition: str | None = None, state: str = "RUNNING") -> None:
    """Cancel this user's job(s) with the given name in the given state.

    Used by the stuck-job watchdogs: RUNNING for deadlocked training jobs,
    PENDING for jobs parked too long on a busy partition. Either way the
    orchestration loop's normal relaunch path can resubmit the case fresh.
    Pinning the state keeps the cancel from racing a legitimate job in the
    other state under the same name.
    """
    if partition is None:
        partition = query_partitions()
    # Unlike squeue -p, scancel -p takes a single partition name and treats a
    # comma-separated list as one literal (nonexistent) name, matching no jobs
    # while still exiting 0, so cancel each partition separately
    for single_partition in partition.split(","):
        _slurm_stdout(["scancel", "-p", single_partition, "-u", getpass.getuser(), "-n", job_name, f"--state={state}"], level="ERROR")


# One GPU entry of a sinfo gres field or a job's GRES, typed (gpu:a100:4(S:1,3,5,7)) or untyped (gpu:4).
# The groups are the type ("" when untyped) and the count.
GPU_GRES_ENTRY_PATTERN = r"gpu(?::([\w.]+))?:(\d+)"

# Node states that take new jobs.
# A trailing "-" (planned by backfill for a pending job) is stripped first, any other flag such as "*" excludes the node.
SCHEDULABLE_NODE_STATES = ("idle", "mix", "alloc", "comp")

# One per-node group of scontrol -d -o show job.
# The groups are the hostlist, and the CPU IDs, memory in MB and GRES (empty without any) held on each of its nodes.
JOB_NODE_DETAIL_PATTERN = r" Nodes=(\S+) CPU_IDs=(\S+) Mem=(\d+) GRES=(\S*)"

# What preempting a running job frees on each of its nodes, memory in MB
EVICTABLE_TRES = ("cpu", "mem", "gres/gpu")


def _gres_gpu_count(gres: str) -> int:
    """GPUs of every type in a sinfo gres or gresused field, typed or untyped."""
    type_counts = re.findall(GPU_GRES_ENTRY_PATTERN, gres)
    return sum(int(count) for _, count in type_counts)


def _schedulable_node_free_gpus(partition: str) -> dict[str, int] | None:
    """Unallocated GPUs on each node of the partition that takes new jobs and is not excluded, None if sinfo fails."""
    stdout = _slurm_stdout(["sinfo", "-p", partition, "-N", "--noheader", "--Format=nodehost:64,gres:128,gresused:128,statecompact:16"])
    if stdout is None:
        return None
    node_free_gpus: dict[str, int] = {}
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        node, gres_total, gres_used, state = fields
        # An excluded node's GPUs are not free to this study's jobs, which can never land there
        if node in config.exclude_nodes or state.removesuffix("-") not in SCHEDULABLE_NODE_STATES:
            continue
        gpus_total = _gres_gpu_count(gres_total)
        gpus_used = _gres_gpu_count(gres_used)
        node_free_gpus[node] = gpus_total - gpus_used
    return node_free_gpus


def _expand_hostlist(hostlist: str) -> list[str] | None:
    """Node names of a SLURM hostlist expression, None if scontrol fails."""
    if "[" not in hostlist:
        return hostlist.split(",")
    stdout = _slurm_stdout(["scontrol", "show", "hostnames", hostlist], level="WARNING")
    if stdout is None:
        return None
    return stdout.split()


def _cpu_id_count(cpu_ids: str) -> int:
    """IDs in an scontrol CPU ID list like 0-15,18-19,64."""
    id_count = 0
    for id_range in cpu_ids.split(","):
        first_id, _, last_id = id_range.partition("-")
        id_count += int(last_id or first_id) - int(first_id) + 1
    return id_count


@_cache_resolved
def _job_node_tres(job_id: str, start_time: str, job_cpus: int) -> tuple[tuple[str, dict[str, int]], ...] | None:
    """(node, EVICTABLE_TRES held there) of every node of one run of a multi-node job, None if a query fails.

    The split comes from the job's scontrol detail groups.
    A run's allocation never changes, so each is queried once.
    A requeued job keeps its id but starts a new run, so the start time is part of the key.
    A node's CPU ID count is not its allocated CPUs on every node type (64 IDs against 48 CPUs on one),
    so job_cpus, the job's tres-alloc cpu total, is split in proportion to the counts.
    """
    job_stdout = _slurm_stdout(["scontrol", "-d", "-o", "show", "job", job_id], level="WARNING")
    if job_stdout is None:
        return None
    node_details: list[tuple[str, int, int, int]] = []
    for group_hostlist, cpu_ids, mem_MB, gres in re.findall(JOB_NODE_DETAIL_PATTERN, job_stdout):
        group_nodes = _expand_hostlist(group_hostlist)
        if group_nodes is None:
            return None
        cpu_id_count = _cpu_id_count(cpu_ids)
        gpus = _gres_gpu_count(gres)
        node_details.extend((node, cpu_id_count, int(mem_MB), gpus) for node in group_nodes)
    total_cpu_ids = sum(cpu_id_count for _, cpu_id_count, _, _ in node_details)
    return tuple(
        (node, {"cpu": job_cpus * cpu_id_count // total_cpu_ids, "mem": mem_MB, "gres/gpu": gpus})
        for node, cpu_id_count, mem_MB, gpus in node_details
    )


def _evictable_node_tres(nodes: list[str], partitions: tuple[str, ...]) -> dict[str, dict[str, int]] | None:
    """EVICTABLE_TRES held on each of these nodes by running jobs of these partitions, None if a query fails.

    A one-node job's squeue tres-alloc is what it holds on its node.
    A multi-node job's is its total over every node, so its split comes from scontrol (_job_node_tres).
    """
    if not nodes or not partitions:
        return {}
    node_list = ",".join(nodes)
    partition_list = ",".join(partitions)
    # The | suffixes leave every field unpadded, so a long hostlist cannot run into its neighbor
    squeue_stdout = _slurm_stdout(
        [
            "squeue",
            "-w",
            node_list,
            "-p",
            partition_list,
            "--states=RUNNING",
            "--noheader",
            "-O",
            "JobID:|,StartTime:|,tres-alloc:|,NodeList:",
        ]
    )
    if squeue_stdout is None:
        return None
    node_tres: dict[str, dict[str, int]] = {}
    for line in squeue_stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) != 4:
            continue
        job_id, start_time, tres_alloc, hostlist = fields
        job_tres = _tres_amounts(tres_alloc)
        if job_tres["node"] == 1:
            job_node_tres = ((hostlist, job_tres),)
        else:
            job_node_tres = _job_node_tres(job_id, start_time, job_tres["cpu"])
            if job_node_tres is None:
                return None
        for node, held_tres in job_node_tres:
            node_totals = node_tres.setdefault(node, dict.fromkeys(EVICTABLE_TRES, 0))
            for key in EVICTABLE_TRES:
                node_totals[key] += held_tres.get(key, 0)
    # A multi-node job may also hold resources outside these nodes
    queried_nodes = set(nodes)
    return {node: held_tres for node, held_tres in node_tres.items() if node in queried_nodes}


def _available_gpus(partition: str) -> int | None:
    """GPUs a job submitted to this partition could take now, None when a query fails.

    A GPU counts when it sits on a node of the partition that takes new jobs,
    and either no job holds it or a job of a partition this one preempts holds it.
    sinfo gresused covers every job on a node, whatever its partition, GPU request style or node count.
    """
    node_free_gpus = _schedulable_node_free_gpus(partition)
    evictable_partitions = _preemptable_partitions(partition)
    if node_free_gpus is None or evictable_partitions is None:
        return None
    schedulable_nodes = list(node_free_gpus)
    node_evictable_tres = _evictable_node_tres(schedulable_nodes, evictable_partitions)
    if node_evictable_tres is None:
        return None
    free_gpus = sum(node_free_gpus.values())
    evictable_gpus = sum(held_tres["gres/gpu"] for held_tres in node_evictable_tres.values())
    return free_gpus + evictable_gpus


def count_idle_gpus(partition: str | None = None, buffer_gpus: int | None = None) -> int:
    """GPUs a job submitted to this partition could take now (_available_gpus), minus buffer_gpus.

    Any failed query counts as no idle GPUs, so nothing launches.
    """
    if partition is None:
        partition = config.partition
    if buffer_gpus is None:
        buffer_gpus = config.buffer_gpus
    available_gpus = _available_gpus(partition)
    if available_gpus is None:
        return 0
    return max(available_gpus - buffer_gpus, 0)


def count_pending_jobs(partition: str | None = None) -> int | None:
    """This user's pending jobs on the partition, None if squeue fails."""
    if partition is None:
        partition = config.partition
    stdout = _slurm_stdout(["squeue", "-p", partition, "-u", getpass.getuser(), "--state=PENDING", "--noheader"])
    if stdout is None:
        return None
    return _count_lines(stdout)


def open_gpu_slots(partition: str | None = None, buffer_gpus: int | None = None) -> int:
    """GPU jobs worth submitting to this partition now: idle GPUs beyond buffer_gpus less this user's pending jobs.

    Pending jobs will take idle GPUs once scheduled, so they claim slots already.
    Any failed query gives no slots, so nothing launches.
    """
    idle_gpus = count_idle_gpus(partition, buffer_gpus)
    pending_jobs = count_pending_jobs(partition)
    if pending_jobs is None:
        return 0
    return max(idle_gpus - pending_jobs, 0)


def resources_available(partition: str | None = None, buffer_gpus: int | None = None) -> bool:
    return open_gpu_slots(partition, buffer_gpus) > 0


def count_user_jobs() -> int | None:
    """This user's running + pending jobs across all partitions, None if squeue fails.

    Every job counts toward the association/QOS MaxSubmit ceilings no matter
    which partition it went to, so the spillover budget is based on this total.
    """
    stdout = _slurm_stdout(["squeue", "-u", getpass.getuser(), "--state=RUNNING,PENDING", "--noheader"])
    if stdout is None:
        return None
    return _count_lines(stdout)


def _scontrol_fields(text: str) -> dict[str, str]:
    """key=value fields of scontrol show output."""
    return dict(token.split("=", 1) for token in text.split() if "=" in token)


@_cache_resolved
def _partition_info(partition: str) -> dict[str, str] | None:
    """key=value fields from scontrol show partition, None if scontrol failed."""
    stdout = _slurm_stdout(["scontrol", "show", "partition", partition], level="WARNING")
    if not stdout or not stdout.strip():
        return None
    return _scontrol_fields(stdout)


def partition_exists(partition: str) -> bool:
    """Whether scontrol knows the partition, False also when the query failed."""
    return _partition_info(partition) is not None


@_cache_resolved
def _preemptable_partitions(partition: str) -> tuple[str, ...] | None:
    """Partitions whose running jobs a job in this partition preempts, None if scontrol fails.

    Under preempt/partition_prio that is every partition with a lower PriorityTier and a PreemptMode other than OFF.
    Any other preemption type gives none, so only unallocated GPUs count as idle.
    """
    config_stdout = _slurm_stdout(["scontrol", "show", "config"], level="WARNING")
    if config_stdout is None:
        return None
    preempt_type = re.search(r"^PreemptType\s*=\s*(\S+)", config_stdout, re.MULTILINE)
    if preempt_type is None:
        return None
    if preempt_type.group(1) != "preempt/partition_prio":
        return ()
    table_stdout = _slurm_stdout(["scontrol", "show", "partition", "-o"], level="WARNING")
    if table_stdout is None:
        return None
    partition_table = [_scontrol_fields(line) for line in table_stdout.splitlines() if "PartitionName=" in line]
    tiers = {fields["PartitionName"]: int(fields["PriorityTier"]) for fields in partition_table}
    if partition not in tiers:
        return None
    own_tier = tiers[partition]
    return tuple(
        fields["PartitionName"]
        for fields in partition_table
        if fields.get("PreemptMode") != "OFF" and tiers[fields["PartitionName"]] < own_tier
    )


def _tres_amounts(tres: str) -> dict[str, int]:
    """Amounts of a SLURM TRES string like cpu=4,mem=120G,node=1,gres/gpu=1, memory in MB.

    Typed entries like gres/gpu:a100=1 are skipped, they repeat the generic gres/gpu count.
    """
    amounts: dict[str, int] = {}
    for item in tres.strip().split(","):
        key, _, value = item.partition("=")
        if not value or ":" in key:
            continue
        if key == "mem":
            amounts[key] = slurm_mem_MB(value)
        elif value.isdigit():
            amounts[key] = int(value)
    return amounts


@_cache_resolved
def partition_user_tres_caps(partition: str) -> dict[str, int] | None:
    """Per-user caps of the partition's QOS MaxTRESPU, memory in MB, empty when uncapped, None if a query failed.

    E.g. mit_preemptable's QOS allows 4 running GPUs per user and mit_normal's 96 CPUs and 386G.
    Submitting more than this just parks jobs pending on the QOS limit,
    so the cap bounds the submissions to the partition.
    A failed query is never cached, the next call retries it.
    """
    info = _partition_info(partition)
    if not info:
        return None
    qos = info.get("QoS")
    if qos in (None, "N/A"):
        return {}
    # The name column tells a QOS without MaxTRESPU (a "<qos>|" row) from one sacctmgr does not know (no row)
    stdout = _slurm_stdout(["sacctmgr", "-nP", "show", "qos", qos, "format=Name,MaxTRESPU"], level="WARNING")
    if stdout is None:
        return None
    for line in stdout.splitlines():
        name, _, max_tres_per_user = line.partition("|")
        if name == qos:
            return _tres_amounts(max_tres_per_user)
    logger.warning(f"sacctmgr show qos {qos} returned no row for it, not caching")
    return None


# Clock fields of a SLURM time string by field count, without and with a day prefix (the sbatch --time grammar)
SLURM_CLOCK_FIELDS = {1: ("minutes",), 2: ("minutes", "seconds"), 3: ("hours", "minutes", "seconds")}
SLURM_DAY_CLOCK_FIELDS = {1: ("hours",), 2: ("hours", "minutes"), 3: ("hours", "minutes", "seconds")}


def parse_slurm_time_s(time_str: str | None) -> int | None:
    """Seconds from a SLURM time string, None if unlimited.

    Takes every sbatch --time form: M, M:S, H:M:S, D-H, D-H:M and D-H:M:S.
    squeue and scontrol print a subset of these.
    """
    if not time_str or time_str.upper() in ("UNLIMITED", "INFINITE", "NONE", "N/A"):
        return None
    day_str, _, clock_str = time_str.rpartition("-")
    clock_values = [int(value) for value in clock_str.split(":")]
    clock_fields = SLURM_DAY_CLOCK_FIELDS if day_str else SLURM_CLOCK_FIELDS
    clock = dict(zip(clock_fields[len(clock_values)], clock_values, strict=True))
    duration = timedelta(days=int(day_str or 0), **clock)
    return int(duration.total_seconds())


def format_slurm_time(seconds: int) -> str:
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def partition_time_limit_s(partition: str) -> int | None:
    """Partition MaxTime in seconds, None if unlimited or unknown."""
    info = _partition_info(partition)
    if info is None:
        return None
    return parse_slurm_time_s(info.get("MaxTime"))


def count_user_tres(partition: str) -> dict[str, int] | None:
    """This user's allocated + requested TRES summed over running and pending jobs on a partition, None if squeue fails."""
    stdout = _slurm_stdout(
        ["squeue", "-p", partition, "-u", getpass.getuser(), "--state=RUNNING,PENDING", "--noheader", "-O", "tres-alloc:200"]
    )
    if stdout is None:
        return None
    totals: dict[str, int] = {}
    for line in stdout.splitlines():
        for key, amount in _tres_amounts(line).items():
            totals[key] = totals.get(key, 0) + amount
    return totals


def user_job_budget() -> int:
    """Jobs still submittable anywhere before hitting the user-wide MaxSubmit ceiling, 0 if squeue fails."""
    user_jobs = count_user_jobs()
    if user_jobs is None:
        return 0
    return max(config.max_user_jobs - config.spillover_job_headroom - user_jobs, 0)


def spillover_slots(partition: str) -> int:
    """Jobs still worth submitting to this spillover partition right now.

    Capped by the partition's per-user GPU allowance when its QOS has one
    (assumes one GPU per job, which matches every template here). Without a
    cap, only submit what could start immediately, so an unbounded partition
    doesn't accumulate a deep pending queue either.
    """
    caps = partition_user_tres_caps(partition) or {}
    if "gres/gpu" not in caps:
        return open_gpu_slots(partition, buffer_gpus=0)
    used = count_user_tres(partition)
    if used is None:
        return 0
    return max(caps["gres/gpu"] - used.get("gres/gpu", 0), 0)


def _schedulable_node_free_cpus_mem(partition: str) -> dict[str, tuple[int, int]] | None:
    """Idle CPUs and unallocated memory in MB on each node of the partition that takes new jobs, None if sinfo fails.

    Unallocated memory is the configured memory less what jobs hold.
    The OS FreeMem is another number, off in both directions for scheduling.
    """
    stdout = _slurm_stdout(
        ["sinfo", "-p", partition, "-N", "--noheader", "--Format=nodehost:64,cpusstate:32,memory:16,allocmem:16,statecompact:16"]
    )
    if stdout is None:
        return None
    node_free: dict[str, tuple[int, int]] = {}
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        node, cpus_state, memory_MB, alloc_mem_MB, state = fields
        if state.removesuffix("-") not in SCHEDULABLE_NODE_STATES:
            continue
        # cpusstate is allocated/idle/other/total
        idle_cpus = int(cpus_state.split("/")[1])
        node_free[node] = (idle_cpus, int(memory_MB) - int(alloc_mem_MB))
    return node_free


def open_cpu_slots(partition: str) -> int:
    """CPU training or agent jobs worth submitting to this partition now, 0 if a query fails.

    Each node fits as many cpu_train_cpus x train_mem jobs as its idle CPUs and unallocated memory allow,
    counting what running jobs of a partition this one preempts hold there as free.
    This user's pending jobs there claim slots already, beyond the cpu_pending_jobs allowed to pend,
    so a freed slot is refilled without waiting for the next orchestration pass.
    The partition's per-user QOS cpu and mem caps bound the rest.
    """
    node_free = _schedulable_node_free_cpus_mem(partition)
    evictable_partitions = _preemptable_partitions(partition)
    caps = partition_user_tres_caps(partition)
    pending_jobs = count_pending_jobs(partition)
    if node_free is None or evictable_partitions is None or caps is None or pending_jobs is None:
        return 0
    node_evictable_tres = _evictable_node_tres(list(node_free), evictable_partitions)
    if node_evictable_tres is None:
        return 0
    job_mem_MB = slurm_mem_MB(config.train_mem)
    node_slots = []
    for node, (idle_cpus, free_mem_MB) in node_free.items():
        evictable_tres = node_evictable_tres.get(node, dict.fromkeys(EVICTABLE_TRES, 0))
        node_cpus = idle_cpus + evictable_tres["cpu"]
        node_mem_MB = free_mem_MB + evictable_tres["mem"]
        node_slots.append(min(node_cpus // config.cpu_train_cpus, node_mem_MB // job_mem_MB))
    slots = sum(node_slots) + config.cpu_pending_jobs - pending_jobs
    job_tres = {"cpu": config.cpu_train_cpus, "mem": job_mem_MB}
    capped_tres = [key for key in job_tres if key in caps]
    if capped_tres:
        used = count_user_tres(partition)
        if used is None:
            return 0
        for key in capped_tres:
            slots = min(slots, (caps[key] - used.get(key, 0)) // job_tres[key])
    return max(slots, 0)


class LaunchSlots:
    """The open slots of one orchestration pass, each partition queried at most once and debited per accepted job.

    Querying every partition again for every job cost about 2 s per launch in the first apsdpp launch,
    most of its 7-10 min passes.
    The primary GPU partition is read once (the GPUs a job could take now and this user's pending jobs there),
    and both kinds of job draw on that one count:
    a GPU-only job leaves buffer_gpus GPUs idle and may wait in the queue up to gpu_pending_jobs,
    a CPU-capable one leaves cpu_capable_buffer_gpus idle and never waits there.
    Every other partition is read the first time a job may go there,
    and the user-wide job budget, read once, bounds them all.
    A failed query gives no slots, so nothing launches there this pass.
    """

    def __init__(self):
        self._primary_read = False
        # (GPUs a job could take, this user's pending jobs) of the primary partition, None when a query failed
        self._primary_counts: tuple[int, int] | None = None
        # Open slots of every other partition read so far, before this pass's debits and the user-wide budget
        self._partition_slots: dict[str, int] = {}
        self._user_job_budget: int | None = None
        # Jobs SLURM accepted this pass, per partition
        self._taken: Counter[str] = Counter()

    def _primary_gpu_counts(self) -> tuple[int, int] | None:
        if not self._primary_read:
            self._primary_read = True
            available_gpus = _available_gpus(config.partition)
            pending_jobs = count_pending_jobs(config.partition)
            if available_gpus is not None and pending_jobs is not None:
                self._primary_counts = (available_gpus, pending_jobs)
        return self._primary_counts

    def available(self, partition: str, cpu_capable: bool = False) -> int:
        """Training or agent jobs worth submitting to this partition now."""
        if partition == config.partition:
            primary_counts = self._primary_gpu_counts()
            if primary_counts is None:
                return 0
            available_gpus, pending_jobs = primary_counts
            if cpu_capable:
                slots = max(available_gpus - config.cpu_capable_buffer_gpus, 0) - pending_jobs
            else:
                slots = max(available_gpus - config.buffer_gpus, 0) + config.gpu_pending_jobs - pending_jobs
            return max(slots - self._taken[partition], 0)
        if partition not in self._partition_slots:
            self._partition_slots[partition] = open_cpu_slots(partition) if partition in cpu_partitions() else spillover_slots(partition)
        if self._user_job_budget is None:
            self._user_job_budget = user_job_budget()
        taken_off_primary = sum(count for taken_partition, count in self._taken.items() if taken_partition != config.partition)
        partition_left = self._partition_slots[partition] - self._taken[partition]
        budget_left = self._user_job_budget - taken_off_primary
        return max(min(partition_left, budget_left), 0)

    def pick(self, cpu_capable: bool = False) -> str | None:
        """Partition the next training or agent job should go to, or None to hold off launching.

        The first candidate with open slots wins:
        the primary GPU partition, then the GPU spillover partitions for a GPU-only case,
        or the CPU partitions in order for a case that can also train on CPU.
        """
        fallbacks = cpu_partitions() if cpu_capable else list(config.spillover_partitions)
        for partition in [config.partition, *fallbacks]:
            if self.available(partition, cpu_capable) > 0:
                return partition
        return None

    def take(self, partition: str) -> None:
        """Debit one job SLURM accepted on partition."""
        self._taken[partition] += 1


@_cache_resolved
def gpu_type_exclude_nodes(partition: str, allowed_gpu_types: tuple[str, ...]) -> str | None:
    """Comma-separated nodes in the partition whose GPUs are not all in allowed_gpu_types (lowercase).

    Node features on this cluster do not tag GPU models, so sbatch
    --constraint cannot select card types and gres cannot request an OR of
    them. An explicit --exclude list is the only way to keep a job on the
    allowed cards. Returns None (exclude nothing) if sinfo fails:
    a job that lands on a disallowed card still trains, just slowly.
    A node listing only untyped GPUs names no card, so it is never excluded.
    """
    stdout = _slurm_stdout(["sinfo", "-p", partition, "-N", "--noheader", "-o", "%N %G"], level="WARNING")
    if stdout is None:
        return None
    nodes = set()
    for line in stdout.splitlines():
        node, _, gres = line.strip().partition(" ")
        node_gpu_types = [gpu_type.lower() for gpu_type, _ in re.findall(GPU_GRES_ENTRY_PATTERN, gres) if gpu_type]
        if node_gpu_types and not all(gpu_type in allowed_gpu_types for gpu_type in node_gpu_types):
            nodes.add(node)
    return ",".join(sorted(nodes))


def _gpu_exclude_nodes(partition: str, typed: bool = False) -> str | None:
    """The comma-separated nodes a GPU job must not land on, None for none.

    Combines the config.gpu_types card allowlist (untyped requests only, a
    typed request pins the card type by itself) with config.exclude_nodes,
    which applies to every request: a known-bad node must stay excluded even
    after the gres fallback switches to a typed request.
    """
    nodes: set[str] = set(config.exclude_nodes)
    if not typed and config.gpu_types:
        type_nodes = gpu_type_exclude_nodes(partition, config.gpu_types)
        if type_nodes:
            nodes.update(type_nodes.split(","))
    if not nodes:
        return None
    return ",".join(sorted(nodes))


GRES_UNAVAILABLE_ERROR = "Requested node configuration is not available"


@_cache_resolved
def partition_gpu_type_counts(partition: str) -> tuple[tuple[str, int], ...] | None:
    """(gpu_type, total GPU count) pairs in the partition, most plentiful first, None if sinfo fails."""
    stdout = _slurm_stdout(["sinfo", "-p", partition, "-N", "--noheader", "-o", "%G"], level="WARNING")
    if stdout is None:
        return None
    counts: dict[str, int] = {}
    for line in stdout.splitlines():
        for gpu_type, count in re.findall(GPU_GRES_ENTRY_PATTERN, line):
            # An untyped entry names no card a typed request could ask for
            if not gpu_type:
                continue
            gpu_type_lower = gpu_type.lower()
            counts[gpu_type_lower] = counts.get(gpu_type_lower, 0) + int(count)
    return tuple(sorted(counts.items(), key=lambda item: -item[1]))


def _gres_requests(partition: str) -> list[tuple[str, str | None]]:
    """Candidate (gres, exclude) pairs for a one-GPU job, in order.

    First the untyped request plus the exclude list keeping the job on
    config.gpu_types cards. Some partitions (mit_preemptable, mit_normal_gpu)
    reject untyped gpu requests aimed at a100/h100/h200 nodes
    with "Requested node configuration is not available" while typed requests
    for the same cards still work, so each allowed type present in the
    partition follows as a typed fallback, most plentiful first.
    A typed request pins the card type by itself,
    so its exclude list only carries config.exclude_nodes.
    """
    requests = [("gpu:1", _gpu_exclude_nodes(partition))]
    if config.gpu_types:
        gpu_type_counts = partition_gpu_type_counts(partition) or ()
        typed_exclude = _gpu_exclude_nodes(partition, typed=True)
        for gpu_type, _ in gpu_type_counts:
            if gpu_type in config.gpu_types:
                requests.append((f"gpu:{gpu_type}:1", typed_exclude))
    return requests


def _sbatch_gpu_job(build_script: Callable[[str, str | None], str], job_name: str, partition: str, kind: str) -> int | None:
    """Submit build_script(gres, exclude) via sbatch, falling back to typed gres, returns the accepted job id or None.

    Only the "Requested node configuration is not available" rejection moves
    on to the next gres request, any other sbatch failure is final.
    """
    result = None
    for gres, exclude in _gres_requests(partition):
        script = build_script(gres, exclude)
        result = _run_slurm(["sbatch"], stdin=script)
        if result is None:
            return None
        if result.returncode == 0:
            logger.info(f"Submitted {kind} job {job_name}: {result.stdout.strip()}")
            return _submitted_job_id(result.stdout, job_name)
        if GRES_UNAVAILABLE_ERROR not in result.stderr:
            break
        logger.info(f"{partition} rejected gres request {gres!r} for {job_name}, trying next GPU type")
    if result is not None:
        logger.error(f"sbatch failed for {kind} job {job_name}: {result.stderr}")
    return None


def _submit_training_job(
    job_name: str, partition: str, time_limit: str, log_path: Path | str, script_path: Path | str, job_setup: str, kind: str
) -> int | None:
    """Submit a training or agent job, returns the job id SLURM accepted it under, None when it did not.

    On a CPU partition it is a CPU job of cpu_train_cpus CPUs, anywhere else a one-GPU job.
    job_setup holds the job's own shell lines, the device banner goes in front and the device environment after.
    """
    if partition in cpu_partitions():
        setup = CPU_JOB_BANNER.format(job_name=job_name) + job_setup + CPU_JOB_ENV
        script = sbatch_script(
            job_name, partition, time_limit, log_path, script_path, setup, mem=config.train_mem, requeue=True, cpus=config.cpu_train_cpus
        )
        return submit_sbatch(script, job_name, kind)
    setup = GPU_JOB_BANNER.format(job_name=job_name) + job_setup + GPU_JOB_ENV

    def build_script(gres: str, exclude: str | None) -> str:
        return sbatch_script(
            job_name, partition, time_limit, log_path, script_path, setup, mem=config.train_mem, gres=gres, exclude=exclude, requeue=True
        )

    return _sbatch_gpu_job(build_script, job_name, partition, kind)


def clamp_time_limit(partition: str, time_limit: str) -> str:
    """sbatch --time of a job, cut to the partition's MaxTime when that is shorter.

    Some partitions cap walltime below the configured limits and reject over-limit requests outright.
    """
    requested_s = parse_slurm_time_s(time_limit)
    limit_s = partition_time_limit_s(partition)
    if requested_s is None or limit_s is None or limit_s >= requested_s:
        return time_limit
    clamped_time_limit = format_slurm_time(limit_s)
    logger.info(f"Partition {partition} MaxTime {clamped_time_limit} is below the requested {time_limit}, clamping the job time")
    return clamped_time_limit


def clamp_time_for_partition(partition: str, train_config: TrainConfig) -> tuple[str, TrainConfig]:
    """sbatch --time and in-job wall budget of a training or agent job, fitted to the partition's MaxTime.

    The in-job budget shrinks by the same amount as the job time,
    preserving the margin the trainer needs to checkpoint and exit before SLURM kills the job.
    """
    time_limit = clamp_time_limit(partition, config.train_time_limit)
    requested_s = parse_slurm_time_s(config.train_time_limit)
    clamped_s = parse_slurm_time_s(time_limit)
    if requested_s is None or clamped_s is None or clamped_s == requested_s:
        return time_limit, train_config
    checkpoint_margin_s = max(requested_s - config.train_wall_budget_s, 0)
    clamped_budget_s = float(max(clamped_s - checkpoint_margin_s, 600))
    if train_config.max_wall_seconds is not None:
        clamped_budget_s = min(clamped_budget_s, train_config.max_wall_seconds)
    logger.info(f"Clamping the in-job wall budget to {clamped_budget_s:.0f} s")
    return time_limit, train_config.model_copy(update={"max_wall_seconds": clamped_budget_s})


def launch_train_parallel(
    train_config: TrainConfig,
    job_name: str,
    result_path: Path | str,
    log_path: Path | str,
    partition: str | None = None,
) -> int | None:
    """Submit a SLURM job that runs training serially and saves results to result_path.

    Returns the job id SLURM accepted it under, None when it did not.

    Args:
        train_config: The training configuration.
        job_name: The SLURM job name.
        result_path: Path where the seed result (seed_results.seed_result) will be saved as NetCDF.
        log_path: The job's SLURM stdout/stderr log, its directory also holds the job's script and configs.
        partition: The SLURM partition to submit to.
    """
    if partition is None:
        partition = config.partition
    time_limit, train_config = clamp_time_for_partition(partition, train_config)
    # All temp files must live on the shared filesystem (not /tmp which is
    # node-local), so that compute nodes can read them.
    log_path = Path(log_path)
    log_dir = log_path.parent
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
from transport_study.orchestration.study import write_netcdf_atomic
from transport_study.orchestration.seed_results import seed_result
{_config_reload_script(study_config_path)}
with open({config_path!r}) as f:
    train_config = TrainConfig(**yaml.safe_load(f))

trainer, _, _, _, result_dict = launch_train(train_config)
if result_dict is None:
    # Trainer hit its wall-clock budget and saved the latest checkpoint. Exit
    # cleanly with no result file, the orchestrator resubmits and the next job
    # resumes from that checkpoint.
    logger.info("Training stopped at the wall-clock budget before finishing, resubmitted job will resume.")
else:
    ds = seed_result(trainer, result_dict)
    write_netcdf_atomic(ds, {str(result_path)!r})
Path({str(config_path)!r}).unlink()
Path({str(study_config_path)!r}).unlink()
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix=f"{job_name}_script_", dir=log_dir) as f:
        f.write(py_script)
        script_path = f.name

    # Save results to netcdf only, no need to sync wandb runs online from batch jobs
    return _submit_training_job(job_name, partition, time_limit, log_path, script_path, "\nexport WANDB_MODE=offline\n\n", "training")


def launch_agent_parallel(
    train_config: TrainConfig,
    sweep_id: str,
    kwargs_agent: dict,
    job_name: str,
    log_dir: Path | str,
    partition: str | None = None,
) -> int | None:
    """Submit a SLURM job that launches a W&B agent for an existing sweep, returns the accepted job id or None.

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

    return _submit_training_job(job_name, partition, time_limit, log_path, script_path, "\n", "agent")


def launch_case_analysis_parallel(study, case) -> int | None:
    """Submit a CPU SLURM job computing one case's stage metrics and case report, returns the accepted job id or None.

    Analysis (metrics aggregation, PDF pages, GIF frames) is matplotlib and
    numpy bound with no GPU work, so it goes to config.analysis_partition when
    set (config.partition otherwise) and requests no GPU.

    The job runs orchestration.case_metrics and orchestration.case_reports,
    which pick up the study's own pieces from its ANALYSIS_METRICS_MODULE / ANALYSIS_REPORTS_MODULE ClassVars.
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
from {importable_module(study_cls)} import {study_cls.__name__}
from transport_study.orchestration.case_metrics import compute_and_save_case_metrics
from transport_study.orchestration.case_reports import generate_case_report

study = {study_cls.__name__}(Path({study_config_path!r}))
case = study.case_by_name({case_str!r})

case_metrics, ts_metrics = compute_and_save_case_metrics(study, case)
if case_metrics.data_vars:
    generate_case_report(study, case, study.figure_dir, ts_metrics)
Path({study_config_path!r}).unlink()
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix=f"{job_name}_", dir=log_dir) as f:
        f.write(py_script)
        script_path = f.name
    log_path = log_dir / f"{job_name}.log"

    # Analysis is cpu-only.
    # Pinning jax keeps behavior identical regardless of the driver env and skips the cuda plugin init on cpu partitions.
    setup = f"""\
echo "=== $(date) job $SLURM_JOB_ID ({job_name}) start on $SLURMD_NODENAME ==="
export MPLBACKEND=Agg
{CPU_JOB_ENV}"""
    time_limit = clamp_time_limit(partition, config.analysis_time_limit)
    script = sbatch_script(job_name, partition, time_limit, log_path, script_path, setup, mem=config.analysis_mem)
    return submit_sbatch(script, job_name, "analysis")
