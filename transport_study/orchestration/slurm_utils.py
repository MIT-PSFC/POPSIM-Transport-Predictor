import functools
import getpass
import inspect
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import yaml
from loguru import logger
from popsim.ml import TrainConfig

from transport_study.config import config

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

# One preallocated XLA pool at 80% of VRAM.
# Growth-mode allocation fragments the pool,
# and a batch-2048 TORAX transport grad step needs one ~36 GB contiguous buffer.
# The other 20% holds the CUDA context and kernel images (~0.7 GB measured).
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
"""
)

# Start banner of a GPU job.
# SLURM_JOB_GPUS is the physical index on the node, the uuid names the exact card for bad-hardware reports.
# Not an f-string, job_name is filled in with str.format.
GPU_JOB_BANNER = """\
echo "=== $(date) job $SLURM_JOB_ID ({job_name}) start ==="
echo "=== node $SLURMD_NODENAME gpu ${{SLURM_JOB_GPUS:-$CUDA_VISIBLE_DEVICES}} $(nvidia-smi --query-gpu=name,uuid --format=csv,noheader 2>/dev/null || echo nvidia-smi unavailable) ==="
"""


def _run_slurm(cmd: list[str], stdin: str | None = None) -> subprocess.CompletedProcess | None:
    """The completed SLURM command, None (logged) when it timed out."""
    try:
        return subprocess.run(cmd, input=stdin, check=False, capture_output=True, text=True, timeout=SLURM_COMMAND_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        logger.critical(f"{cmd[0]} timed out after {SLURM_COMMAND_TIMEOUT_S} s")
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

    The partition inventory and limits are static for a study run,
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
    requeue: bool = False,
) -> str:
    """sbatch script that runs script_path with this interpreter after the setup shell lines, then deletes it.

    Resubmitted attempts append to the same log_path.
    A time_limit of None leaves the partition default.
    """
    time_line = f"\n#SBATCH --time={time_limit}" if time_limit else ""
    gres_line = f"\n#SBATCH --gres={gres}" if gres else ""
    requeue_line = "\n#SBATCH --requeue" if requeue else ""
    return f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}{time_line}{gres_line}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=4
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


def submit_sbatch(script: str, job_name: str, kind: str) -> bool:
    """Submit an sbatch script, returns whether SLURM accepted it."""
    result = _run_slurm(["sbatch"], stdin=script)
    if result is None:
        return False
    if result.returncode != 0:
        logger.error(f"sbatch failed for {kind} job {job_name}: {result.stderr}")
        return False
    logger.info(f"Submitted {kind} job {job_name}: {result.stdout.strip()}")
    return True


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
    stdout = _slurm_stdout(["squeue", "-p", partition, "-n", job_name, "--state=RUNNING,PENDING", "--noheader"])
    if stdout is None:
        return 999999  # Return a large number to prevent launching more jobs if squeue fails
    return _count_lines(stdout)


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
    # Default %j truncates long names, and case names run long
    stdout = _slurm_stdout(["squeue", "-p", partition, "-u", getpass.getuser(), "--state=RUNNING,PENDING", "--noheader", "--format=%512j"])
    if stdout is None:
        return None
    return {line.strip() for line in stdout.splitlines() if line.strip()}


def get_running_job_elapsed_s(partition: str | None = None) -> dict[str, int] | None:
    """Elapsed running time in seconds for this user's RUNNING jobs, keyed by job name.

    Pending jobs are excluded, they haven't started accumulating epochs yet.
    Returns None when squeue fails, so callers can tell "no running jobs" apart
    from "scheduler unreachable" (mirrors get_running_job_names).
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
            elapsed[name.strip()] = elapsed_s
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


# One GPU entry of a sinfo gres field, e.g. gpu:a100:4(S:1,3,5,7), the groups are the type and the count
GPU_GRES_PATTERN = r"gpu:([\w.]+):(\d+)"

# Node states that take new jobs.
# A trailing "-" (planned by backfill for a pending job) is stripped first, any other flag such as "*" excludes the node.
SCHEDULABLE_NODE_STATES = ("idle", "mix", "alloc", "comp")

# One per-node group of scontrol -d -o show job, the groups are the hostlist and the GPUs held on each of its nodes
JOB_NODE_GPUS_PATTERN = r" Nodes=(\S+) CPU_IDs=\S+ Mem=\S+ GRES=gpu:(?:[\w.]+:)?(\d+)"


def _gres_gpu_count(gres: str) -> int:
    """GPUs of every type in a sinfo gres or gresused field."""
    type_counts = re.findall(GPU_GRES_PATTERN, gres)
    return sum(int(count) for _, count in type_counts)


def _schedulable_node_free_gpus(partition: str) -> dict[str, int] | None:
    """Unallocated GPUs on each node of the partition that takes new jobs, None if sinfo fails."""
    stdout = _slurm_stdout(["sinfo", "-p", partition, "-N", "--noheader", "--Format=nodehost:64,gres:128,gresused:128,statecompact:16"])
    if stdout is None:
        return None
    node_free_gpus: dict[str, int] = {}
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        node, gres_total, gres_used, state = fields
        if state.removesuffix("-") not in SCHEDULABLE_NODE_STATES:
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


def _evictable_node_gpus(nodes: list[str], partitions: tuple[str, ...]) -> dict[str, int] | None:
    """GPUs held on each of these nodes by running jobs of these partitions, None if a query fails.

    squeue tres-alloc is the job total over all its nodes,
    so the per-node split comes from the job's scontrol detail groups.
    """
    if not nodes or not partitions:
        return {}
    node_list = ",".join(nodes)
    partition_list = ",".join(partitions)
    squeue_stdout = _slurm_stdout(
        ["squeue", "-w", node_list, "-p", partition_list, "--states=RUNNING", "--noheader", "-O", "JobID:20,tres-alloc:200"]
    )
    if squeue_stdout is None:
        return None
    gpu_job_ids = [line.split()[0] for line in squeue_stdout.splitlines() if "gres/gpu=" in line]
    node_gpus: dict[str, int] = {}
    for job_id in gpu_job_ids:
        job_stdout = _slurm_stdout(["scontrol", "-d", "-o", "show", "job", job_id], level="WARNING")
        if job_stdout is None:
            return None
        for group_hostlist, gpus_per_node in re.findall(JOB_NODE_GPUS_PATTERN, job_stdout):
            group_nodes = _expand_hostlist(group_hostlist)
            if group_nodes is None:
                return None
            for node in group_nodes:
                node_gpus[node] = node_gpus.get(node, 0) + int(gpus_per_node)
    # A multi-node job may also hold GPUs outside these nodes
    queried_nodes = set(nodes)
    return {node: gpus for node, gpus in node_gpus.items() if node in queried_nodes}


def count_idle_gpus(partition: str | None = None, buffer_gpus: int | None = None) -> int:
    """GPUs a job submitted to this partition could take now, minus buffer_gpus.

    A GPU counts when it sits on a node of the partition that takes new jobs,
    and either no job holds it or a job of a partition this one preempts holds it.
    sinfo gresused covers every job on a node, whatever its partition, GPU request style or node count.
    Any failed query counts as no idle GPUs, so nothing launches.
    """
    if partition is None:
        partition = config.partition
    if buffer_gpus is None:
        buffer_gpus = config.buffer_gpus
    node_free_gpus = _schedulable_node_free_gpus(partition)
    evictable_partitions = _preemptable_partitions(partition)
    if node_free_gpus is None or evictable_partitions is None:
        return 0
    schedulable_nodes = list(node_free_gpus)
    node_evictable_gpus = _evictable_node_gpus(schedulable_nodes, evictable_partitions)
    if node_evictable_gpus is None:
        return 0
    free_gpus = sum(node_free_gpus.values())
    evictable_gpus = sum(node_evictable_gpus.values())
    return max(free_gpus + evictable_gpus - buffer_gpus, 0)


def count_pending_jobs(partition: str | None = None) -> int:
    """Count this user's pending jobs on the partition."""
    if partition is None:
        partition = config.partition
    stdout = _slurm_stdout(["squeue", "-p", partition, "-u", getpass.getuser(), "--state=PENDING", "--noheader"])
    if stdout is None:
        return 999999  # Return a large number to prevent launching more jobs if squeue fails
    return _count_lines(stdout)


def open_gpu_slots(partition: str | None = None, buffer_gpus: int | None = None) -> int:
    """GPU jobs worth submitting to this partition now: idle GPUs beyond buffer_gpus less this user's pending jobs.

    Pending jobs will take idle GPUs once scheduled, so they claim slots already.
    A negative buffer_gpus keeps that many jobs pending on a full partition.
    """
    idle_gpus = count_idle_gpus(partition, buffer_gpus)
    pending_jobs = count_pending_jobs(partition)
    return max(idle_gpus - pending_jobs, 0)


def resources_available(partition: str | None = None, buffer_gpus: int | None = None) -> bool:
    return open_gpu_slots(partition, buffer_gpus) > 0


def count_user_jobs() -> int:
    """This user's running + pending jobs across all partitions.

    Every job counts toward the association/QOS MaxSubmit ceilings no matter
    which partition it went to, so the spillover budget is based on this total.
    """
    stdout = _slurm_stdout(["squeue", "-u", getpass.getuser(), "--state=RUNNING,PENDING", "--noheader"])
    if stdout is None:
        return 999999  # Return a large number to prevent launching more jobs if squeue fails
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


_partition_user_gpu_cap_cache: dict[str, int | None] = {}


def partition_user_gpu_cap(partition: str) -> int | None:
    """Per-user GPU cap on a partition (its QOS MaxTRESPU gres/gpu), None if uncapped.

    E.g. mit_preemptable's QOS allows 4 running GPUs per user, mit_normal_gpu's
    allows 2. Submitting more jobs than this just parks them pending on the QOS
    limit, so the spillover logic treats it as that partition's submission cap.
    Only resolved lookups are cached. On a transient scontrol/sacctmgr failure
    this returns None UNcached: None means "uncapped", which routes
    spillover_slots to the idle-minus-pending estimate (~0 on busy public
    partitions), so a cached failure would silently disable spillover.
    """
    if partition in _partition_user_gpu_cap_cache:
        return _partition_user_gpu_cap_cache[partition]
    info = _partition_info(partition)
    if not info:
        return None
    qos = info.get("QoS")
    if qos in (None, "N/A"):
        _partition_user_gpu_cap_cache[partition] = None
        return None
    stdout = _slurm_stdout(["sacctmgr", "-nP", "show", "qos", qos, "format=MaxTRESPU"], level="WARNING")
    if stdout is None:
        return None
    if not stdout.strip():
        logger.warning(f"sacctmgr show qos {qos} returned no output, not caching")
        return None
    match = re.search(r"gres/gpu=(\d+)", stdout)
    cap = int(match.group(1)) if match else None
    _partition_user_gpu_cap_cache[partition] = cap
    return cap


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


def count_user_gpus(partition: str) -> int:
    """This user's allocated + requested GPUs among running and pending jobs on a partition."""
    stdout = _slurm_stdout(
        ["squeue", "-p", partition, "-u", getpass.getuser(), "--state=RUNNING,PENDING", "--noheader", "-O", "tres-alloc:200"]
    )
    if stdout is None:
        return 999999  # Return a large number to prevent launching more jobs if squeue fails
    # Generic gres/gpu=N only, the typed gres/gpu:<type>=N entry would double count
    return sum(int(n) for n in re.findall(r"gres/gpu=(\d+)", stdout))


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
    return open_gpu_slots(partition, buffer_gpus=0)


def pick_partition() -> str | None:
    """Partition the next GPU job should go to, or None to hold off launching.

    The primary partition wins while it has open GPU slots (open_gpu_slots).
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


@_cache_resolved
def gpu_type_exclude_nodes(partition: str, allowed_gpu_types: tuple[str, ...]) -> str | None:
    """Comma-separated nodes in the partition whose GPUs are not all in allowed_gpu_types.

    Node features on this cluster do not tag GPU models, so sbatch
    --constraint cannot select card types and gres cannot request an OR of
    them. An explicit --exclude list is the only way to keep a job on the
    allowed cards. Returns None (exclude nothing) if sinfo fails:
    a job that lands on a disallowed card still trains, just slowly.
    """
    stdout = _slurm_stdout(["sinfo", "-p", partition, "-N", "--noheader", "-o", "%N %G"], level="WARNING")
    if stdout is None:
        return None
    nodes = set()
    for line in stdout.splitlines():
        node, _, gres = line.strip().partition(" ")
        gpu_types = re.findall(r"gpu:([A-Za-z0-9_]+):", gres)
        if gpu_types and not all(t.lower() in allowed_gpu_types for t in gpu_types):
            nodes.add(node)
    return ",".join(sorted(nodes))


def _gpu_exclude_directive(partition: str, typed: bool = False) -> str:
    """The #SBATCH --exclude line for a GPU job, or "".

    Combines the config.gpu_types card allowlist (untyped requests only, a
    typed request pins the card type by itself) with config.exclude_nodes,
    which applies to every request: a known-bad node must stay excluded even
    after the gres fallback switches to a typed request.
    """
    nodes: set[str] = set(config.exclude_nodes)
    if not typed and config.gpu_types:
        type_nodes = gpu_type_exclude_nodes(partition, tuple(config.gpu_types))
        if type_nodes:
            nodes.update(type_nodes.split(","))
    if not nodes:
        return ""
    return f"\n#SBATCH --exclude={','.join(sorted(nodes))}"


GRES_UNAVAILABLE_ERROR = "Requested node configuration is not available"


@_cache_resolved
def partition_gpu_type_counts(partition: str) -> tuple[tuple[str, int], ...] | None:
    """(gpu_type, total GPU count) pairs in the partition, most plentiful first, None if sinfo fails."""
    stdout = _slurm_stdout(["sinfo", "-p", partition, "-N", "--noheader", "-o", "%G"], level="WARNING")
    if stdout is None:
        return None
    counts: dict[str, int] = {}
    for line in stdout.splitlines():
        for gpu_type, count in re.findall(GPU_GRES_PATTERN, line):
            counts[gpu_type.lower()] = counts.get(gpu_type.lower(), 0) + int(count)
    return tuple(sorted(counts.items(), key=lambda item: -item[1]))


def _gres_fragments(partition: str) -> list[str]:
    """Candidate values for the #SBATCH --gres line of a one-GPU job, in order.

    First the untyped request plus the exclude list keeping the job on
    config.gpu_types cards. Some partitions (mit_preemptable, mit_normal_gpu)
    reject untyped gpu requests aimed at a100/h100/h200 nodes
    with "Requested node configuration is not available" while typed requests
    for the same cards still work, so each allowed type present in the
    partition follows as a typed fallback, most plentiful first.
    A typed request pins the card type by itself,
    so its exclude list only carries config.exclude_nodes.
    """
    fragments = [f"gpu:1{_gpu_exclude_directive(partition)}"]
    if config.gpu_types:
        allowed = {t.lower() for t in config.gpu_types}
        gpu_type_counts = partition_gpu_type_counts(partition) or ()
        for gpu_type, _ in gpu_type_counts:
            if gpu_type in allowed:
                fragments.append(f"gpu:{gpu_type}:1{_gpu_exclude_directive(partition, typed=True)}")
    return fragments


def _sbatch_gpu_job(build_script: Callable[[str], str], job_name: str, partition: str, kind: str) -> bool:
    """Submit build_script(gres_fragment) via sbatch, falling back to typed gres, returns whether SLURM accepted it.

    Only the "Requested node configuration is not available" rejection moves
    on to the next gres fragment, any other sbatch failure is final.
    """
    result = None
    for gres_fragment in _gres_fragments(partition):
        script = build_script(gres_fragment)
        result = _run_slurm(["sbatch"], stdin=script)
        if result is None:
            return False
        if result.returncode == 0:
            logger.info(f"Submitted {kind} job {job_name}: {result.stdout.strip()}")
            return True
        if GRES_UNAVAILABLE_ERROR not in result.stderr:
            break
        logger.info(f"{partition} rejected gres request {gres_fragment.splitlines()[0]!r} for {job_name}, trying next GPU type")
    if result is not None:
        logger.error(f"sbatch failed for {kind} job {job_name}: {result.stderr}")
    return False


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
) -> bool:
    """Submit a SLURM job that runs training serially and saves results to result_path, returns whether SLURM accepted it.

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
from transport_study.orchestration.study import write_netcdf_atomic
from transport_study.orchestration.topk_results import compute_topk_study_results
{_config_reload_script(study_config_path)}
with open({config_path!r}) as f:
    train_config = TrainConfig(**yaml.safe_load(f))

trainer, _, _, test_dl, result_dict = launch_train(train_config)
if result_dict is None:
    # Trainer hit its wall-clock budget and saved the latest checkpoint. Exit
    # cleanly with no result file, the orchestrator resubmits and the next job
    # resumes from that checkpoint.
    logger.info("Training stopped at the wall-clock budget before finishing, resubmitted job will resume.")
else:
    ds = compute_topk_study_results(trainer, test_dl, train_config, result_dict)
    write_netcdf_atomic(ds, {str(result_path)!r})
Path({str(config_path)!r}).unlink()
Path({str(study_config_path)!r}).unlink()
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix=f"{job_name}_script_", dir=log_dir) as f:
        f.write(py_script)
        script_path = f.name
    log_path = log_dir / f"{job_name}.log"

    # Save results to netcdf only, no need to sync wandb runs online from batch jobs
    setup = GPU_JOB_BANNER.format(job_name=job_name) + "\nexport WANDB_MODE=offline\n\n" + GPU_JOB_ENV

    def build_script(gres_fragment: str) -> str:
        return sbatch_script(job_name, partition, time_limit, log_path, script_path, setup, gres=gres_fragment, requeue=True)

    return _sbatch_gpu_job(build_script, job_name, partition, "training")


def launch_agent_parallel(
    train_config: TrainConfig,
    sweep_id: str,
    kwargs_agent: dict,
    job_name: str,
    log_dir: Path | str,
    partition: str | None = None,
) -> bool:
    """Submit a SLURM job that launches a W&B agent for an existing sweep, returns whether SLURM accepted it.

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

    setup = GPU_JOB_BANNER.format(job_name=job_name) + "\n" + GPU_JOB_ENV

    def build_script(gres_fragment: str) -> str:
        return sbatch_script(job_name, partition, time_limit, log_path, script_path, setup, gres=gres_fragment, requeue=True)

    return _sbatch_gpu_job(build_script, job_name, partition, "agent")


def launch_case_analysis_parallel(study, case) -> None:
    """Submit a CPU SLURM job computing one case's stage metrics and case report.

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

case_metrics = compute_and_save_case_metrics(study, case)
if case_metrics.data_vars:
    generate_case_report(study, case, study.figure_dir)
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
export JAX_PLATFORMS=cpu
{SINGLE_THREAD_BLAS_ENV}"""
    script = sbatch_script(job_name, partition, config.analysis_time_limit, log_path, script_path, setup, mem="32G")
    submit_sbatch(script, job_name, "analysis")
