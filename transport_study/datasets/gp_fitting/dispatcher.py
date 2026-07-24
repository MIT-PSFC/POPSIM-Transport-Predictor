"""Dispatch GP profile fitting batches to a SLURM cluster.

The dispatcher uploads batch input files (one npz per batch of shots, to avoid
many-small-file transfers on clusters like Engaging) plus the standalone
fit_worker.py, submits one CPU job per batch, polls until completion, and pulls
the result files back.

Two backends:
- "ssh": submit to a remote cluster through an srunx SSH profile (set up once
  with `srunx ssh profile add <name> --ssh-host <host>`). Used for C-Mod,
  where the cluster has no access to the source data.
- "local": running on the cluster itself (e.g. MAST fitting on Engaging);
  files are copied on the shared filesystem and sbatch runs locally.

Jobs get deterministic names gpfit-{device}-{batch_id}-a{attempt} (batch_id is
a hash of the shot list), so a restarted workflow finds in-flight jobs instead
of resubmitting them. Killed jobs (TIMEOUT, PREEMPTED, OOM, ...) are retried up
to max_retries times, and both retries and jobs stuck PENDING past
pending_timeout_s move to the next partition in the preference list (wrapping
around).
"""

import hashlib
import shlex
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from transport_study.datasets.gp_fitting import fit_worker
from transport_study.datasets.gp_fitting.fit_worker import (
    ShotFitInput,
    ShotFitOutput,
    pack_fit_batch,
    read_batch_shots,
    unpack_fit_results,
)

WORKER_FILENAME = "fit_worker.py"
WORKER_SOURCE = Path(fit_worker.__file__)

# rsync transfers go through the cluster login node, which sometimes drops
# connections (exit 255), so transfers are retried before giving up
_TRANSFER_ATTEMPTS = 3
_TRANSFER_RETRY_DELAY_S = 10.0
# rsync exit codes that mean the source file does not exist (not a
# connection problem), so retrying the transfer is pointless
_RSYNC_SOURCE_MISSING_CODES = {23, 24}
# polls to wait for a COMPLETED job's output to become pullable before
# declaring the batch failed
_MAX_OUTPUT_PULL_POLLS = 3
# polls to tolerate a job in an unrecognized/unknown state (e.g. it vanished
# from the queue) before treating the attempt as failed
_MAX_UNKNOWN_POLLS = 5
# how long clean() waits for cancelled jobs to actually leave the queue before
# it deletes their batch files (see _wait_for_jobs_to_drain)
_CLEAN_DRAIN_TIMEOUT_S = 120.0
_CLEAN_DRAIN_POLL_S = 5.0

# SLURM states that mean the job will never produce output
_TERMINAL_FAILURE_STATES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}


@dataclass
class PartitionSpec:
    """One partition to try, with its own time limit and optional constraint.

    time_limit must not exceed the partition's MaxTime; query it with:
        scontrol show partition <name> | grep MaxTime
    or: sinfo -p <name> -O partitionname,time
    """

    name: str
    time_limit: str
    constraint: str | None = None


def parse_partition_specs(value) -> list["PartitionSpec"]:
    """Parse a partition spec string into PartitionSpecs.

    Format: comma-separated entries of name@time_limit or
    name@time_limit@constraint, e.g.
    "sched_psfc_mit_r8@8:00:00,mit_preemptable@8:00:00@rocky8".
    Also accepts a tuple/list of entry strings (Python Fire may pre-split
    comma-separated arguments).
    """
    if isinstance(value, str):
        entries = value.split(",")
    else:
        entries = list(value)
    specs = []
    for entry in entries:
        fields = entry.strip().split("@")
        if len(fields) == 2:
            specs.append(PartitionSpec(name=fields[0], time_limit=fields[1]))
        elif len(fields) == 3:
            specs.append(PartitionSpec(name=fields[0], time_limit=fields[1], constraint=fields[2]))
        else:
            raise ValueError(f"Bad partition spec '{entry}': expected name@time_limit or name@time_limit@constraint")
    if not specs:
        raise ValueError("Empty partition spec")
    return specs


@dataclass
class ClusterFitConfig:
    """Launch options for cluster-based GP fitting.

    Parameters
    ----------
    profile : str
        srunx SSH profile name, or "local" when already running on the
        target cluster (shared filesystem, local sbatch).
    partitions : list[PartitionSpec] | str | tuple
        Ordered partition preference list, each with its own time limit and
        optional constraint (see parse_partition_specs for the string
        format). mkgp is CPU-only, so these should be CPU partitions. A
        batch is submitted to the first partition; it falls back to the
        next (wrapping around) when its job is killed or sits PENDING
        longer than pending_timeout_s.
    remote_workdir : str
        Scratch directory on the cluster where batch files, the worker
        script, and job logs are placed. Cluster-specific.
    venv_path : str
        Path to a pre-built venv on the cluster (see bootstrap_remote.sh).
    max_concurrent_jobs : int
        Cap on simultaneously queued/running fitting jobs, to be a good
        cluster citizen.
    shots_per_batch : int
        Shots packed into one npz / one job. At ~40 core-minutes per shot,
        10 shots on 32 CPUs is ~15 minutes wall time. Small batches keep
        jobs running concurrently and bound the work lost to a killed job.
    cpus_per_job : int
        cpus-per-task for each fitting job; the worker runs this many
        slice-fit processes.
    max_retries : int
        Resubmissions allowed per batch after a terminal job failure
        (TIMEOUT, PREEMPTED, OOM, ...). Each retry moves to the next
        partition in the list, wrapping around.
    pending_timeout_s : float
        Cancel a PENDING job and resubmit it on the next partition after
        this long in the queue. Stops once every partition has been tried.
    """

    profile: str
    partitions: list[PartitionSpec] | str | tuple
    remote_workdir: str
    venv_path: str
    max_concurrent_jobs: int = 8
    shots_per_batch: int = 10
    cpus_per_job: int = 32
    memory_per_node: str | None = None
    poll_interval_s: float = 60.0
    job_name_prefix: str = "gpfit"
    max_retries: int = 2
    pending_timeout_s: float = 1800.0

    def __post_init__(self):
        if not isinstance(self.partitions, list) or not all(isinstance(p, PartitionSpec) for p in self.partitions):
            self.partitions = parse_partition_specs(self.partitions)


def batch_id(device: str, shots: list[int]) -> str:
    """Deterministic short id for a batch, stable across restarts."""
    digest = hashlib.sha1(f"{device}:{','.join(str(s) for s in sorted(shots))}".encode())
    return digest.hexdigest()[:10]


def plan_batches(
    device: str,
    pending_shots: list[int],
    batches_dir: Path,
    shots_per_batch: int,
) -> dict[str, list[int]]:
    """Assign pending shots to batches, reusing batch files from earlier runs.

    Existing batch input npz files in batches_dir keep their membership (and
    therefore their batch id and job name), so a restarted workflow lines up
    with jobs already in the cluster queue. Shots not covered by an existing
    batch are chunked into new batches of shots_per_batch.

    Returns a mapping of batch_id -> shots from pending_shots in that batch.
    """
    pending = set(pending_shots)
    batches: dict[str, list[int]] = {}

    for batch_path in sorted(batches_dir.glob("batch_*.npz")):
        try:
            batch_shots = read_batch_shots(batch_path)
        except Exception as e:
            logger.warning(f"Could not read existing batch file {batch_path}: {e}")
            continue
        bid = batch_path.stem.removeprefix("batch_")
        claimed = [s for s in batch_shots if s in pending]
        if claimed:
            batches[bid] = claimed
            pending -= set(claimed)

    remaining = sorted(pending)
    for i in range(0, len(remaining), shots_per_batch):
        chunk = remaining[i : i + shots_per_batch]
        batches[batch_id(device, chunk)] = chunk

    return batches


@contextmanager
def _srunx_rsync_logs_disabled():
    """Silence srunx's per-call rsync warning.

    Pulls regularly probe for output files that may not exist yet, and
    srunx logs a warning for every miss. The dispatcher logs the failures
    it actually cares about itself, with retry context and stderr.
    """
    logger.disable("srunx.sync.rsync")
    try:
        yield
    finally:
        logger.enable("srunx.sync.rsync")


class _SlurmJobControl:
    """Queue inspection and cancellation, shared by both backends.

    Both wrap an srunx client (SSH or local), so only the file transfer differs.
    """

    def queued_jobs(self) -> list[tuple[str, int]]:
        """(name, job id) for every queued/running job of this user.

        A list, not a name-keyed dict: two jobs can carry the same name (two
        runs each submitting attempt 1 of the same batch), and clean() has to
        cancel both.
        """
        return [(j.name, j.job_id) for j in self._client.queue(user=self._username) if j.job_id]

    def queued_job_names(self) -> dict[str, int]:
        """Names of this user's queued/running jobs -> job id."""
        return dict(self.queued_jobs())

    def job_states(self, job_ids: list[int]) -> dict[int, str]:
        return {jid: snap.status for jid, snap in self._client.queue_by_ids(job_ids).items()}

    def cancel(self, job_id: int) -> None:
        self._client.cancel(job_id)


class _SSHBackend(_SlurmJobControl):
    """File transfer and job control on a remote cluster via srunx."""

    def __init__(self, config: ClusterFitConfig):
        from srunx.slurm.clients.ssh import SlurmSSHClient
        from srunx.ssh.core.config import ConfigManager
        from srunx.sync.mount_helpers import build_rsync_client

        profile = ConfigManager().get_profile(config.profile)
        if profile is None:
            raise ValueError(
                f"srunx SSH profile '{config.profile}' not found. Create it with: srunx ssh profile add {config.profile} --ssh-host <host>"
            )
        self._client = SlurmSSHClient(profile_name=config.profile)
        # connection_spec carries the username resolved from ~/.ssh/config,
        # which profile.username lacks for --ssh-host profiles
        self._username = self._client.connection_spec.username
        if not self._username:
            logger.warning(
                f"No username resolved for profile '{config.profile}' (no User line in ~/.ssh/config?); "
                "job adoption will scan all users' queued jobs"
            )
        # build_rsync_client delegates to ~/.ssh/config for --ssh-host
        # profiles, where profile.hostname/username are empty
        self._rsync = build_rsync_client(profile)
        # srunx detects --mkpath from the local rsync only, but on push the
        # flag reaches the remote rsync, which may be too old for it
        # (e.g. Engaging has 3.1.3). Disable it to force srunx's ssh mkdir -p
        # fallback, which works regardless of remote rsync version.
        self._rsync._supports_mkpath = False

    def push_file(self, local: Path, remote_dir: str) -> None:
        for attempt in range(1, _TRANSFER_ATTEMPTS + 1):
            result = self._rsync.push(str(local), f"{remote_dir}/", delete=False)
            if result.returncode == 0:
                return
            if attempt < _TRANSFER_ATTEMPTS:
                logger.warning(
                    f"rsync push of {local} failed (attempt {attempt}/{_TRANSFER_ATTEMPTS}), "
                    f"retrying in {_TRANSFER_RETRY_DELAY_S:.0f}s: {result.stderr.strip()}"
                )
                time.sleep(_TRANSFER_RETRY_DELAY_S)
        raise RuntimeError(f"rsync push of {local} failed: {result.stderr}")

    def pull_file(self, remote_path: str, local_dir: Path) -> bool:
        """Pull a single remote file into local_dir. Returns False if unavailable."""
        local_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, _TRANSFER_ATTEMPTS + 1):
            with _srunx_rsync_logs_disabled():
                result = self._rsync.pull(remote_path, f"{local_dir}/")
            if result.returncode == 0:
                return (local_dir / Path(remote_path).name).exists()
            if result.returncode in _RSYNC_SOURCE_MISSING_CODES:
                return False
            if attempt < _TRANSFER_ATTEMPTS:
                logger.warning(
                    f"rsync pull of {remote_path} failed (attempt {attempt}/{_TRANSFER_ATTEMPTS}), "
                    f"retrying in {_TRANSFER_RETRY_DELAY_S:.0f}s: {result.stderr.strip()}"
                )
                time.sleep(_TRANSFER_RETRY_DELAY_S)
        return False

    def submit_script(self, script: str, job_name: str) -> int:
        return int(self._client.submit_job(script, job_name=job_name)["job_id"])

    def ensure_dir(self, path: str) -> None:
        result = self._rsync._ssh_run(f"mkdir -p {shlex.quote(path)}")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create remote dir {path}: {result.stderr.strip()}")

    def remove_glob(self, remote_dir: str, pattern: str) -> None:
        """Delete remote files matching pattern via ssh (the remote shell expands the glob)."""
        result = self._rsync._ssh_run(f"rm -f -- {shlex.quote(remote_dir)}/{pattern}")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to remove remote files {remote_dir}/{pattern}: {result.stderr.strip()}")


class _LocalBackend(_SlurmJobControl):
    """Job control when already running on the target cluster (shared FS)."""

    def __init__(self, config: ClusterFitConfig):
        import getpass

        from srunx import Slurm

        self._username = getpass.getuser()
        self._client = Slurm()

    def push_file(self, local: Path, remote_dir: str) -> None:
        dest_dir = Path(remote_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / local.name
        if not (dest.exists() and dest.stat().st_mtime >= local.stat().st_mtime):
            shutil.copy2(local, dest)

    def pull_file(self, remote_path: str, local_dir: Path) -> bool:
        src = Path(remote_path)
        if not src.exists():
            return False
        local_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_dir / src.name)
        return True

    def submit_script(self, script: str, job_name: str) -> int:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write(script)
            script_path = f.name
        try:
            result = subprocess.run(["sbatch", "--parsable", script_path], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"sbatch failed for {job_name}: {result.stderr.strip()}")
            return int(result.stdout.strip().split(";")[0])
        finally:
            Path(script_path).unlink(missing_ok=True)

    def ensure_dir(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def remove_glob(self, remote_dir: str, pattern: str) -> None:
        for p in Path(remote_dir).glob(pattern):
            p.unlink(missing_ok=True)


@dataclass
class BatchState:
    bid: str
    shots: list[int]
    input_path: Path
    output_path: Path
    job_base_name: str
    job_id: int | None = None
    attempt: int = 0  # submissions so far; job names carry -a{attempt}
    failures: int = 0  # terminal failures so far, vs config.max_retries
    partition_idx: int = 0
    pending_since: float | None = None
    pending_hops: int = 0
    fail_reason: str | None = None
    done: bool = False
    failed: bool = False
    output_pull_polls: int = 0
    unknown_polls: int = 0

    @property
    def job_name(self) -> str:
        return f"{self.job_base_name}-a{self.attempt}"

    def reset_for_resubmit(self) -> None:
        self.job_id = None
        self.pending_since = None
        self.output_pull_polls = 0
        self.unknown_polls = 0


class ClusterFitDispatcher:
    """Run GP fitting batches on a SLURM cluster and collect the results.

    Parameters
    ----------
    config : ClusterFitConfig
        Cluster launch options.
    device : str
        Dataset/device name ('cmod', 'mast'), used in job names and ids.
    staging_dir : Path
        Local directory holding batch input/output npz files between runs.
    """

    def __init__(self, config: ClusterFitConfig, device: str, staging_dir: Path):
        self.config = config
        self.device = device
        self.batches_dir = Path(staging_dir) / "batches"
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        self.backend = self._create_backend(config)

    @staticmethod
    def _create_backend(config: ClusterFitConfig):
        if config.profile == "local":
            return _LocalBackend(config)
        return _SSHBackend(config)

    def job_name(self, bid: str) -> str:
        return f"{self.config.job_name_prefix}-{self.device}-{bid}"

    def input_path(self, bid: str) -> Path:
        return self.batches_dir / f"batch_{bid}.npz"

    def output_path(self, bid: str) -> Path:
        return self.batches_dir / f"batch_{bid}_out.npz"

    def run(
        self,
        shot_inputs: dict[int, ShotFitInput],
        x_star,
        min_points: int,
        scale_per_slice: bool,
    ) -> dict[int, ShotFitOutput | None]:
        """Fit all shots on the cluster. Returns per-shot outputs (None = batch failed).

        Blocks until every batch has either produced results or failed.
        Idempotent: existing batch files, completed outputs, and queued jobs
        from a previous run are reused rather than redone.
        """
        plan = plan_batches(
            self.device,
            sorted(shot_inputs),
            self.batches_dir,
            self.config.shots_per_batch,
        )

        batches: list[BatchState] = []
        for bid, shots in plan.items():
            state = BatchState(
                bid=bid,
                shots=shots,
                input_path=self.input_path(bid),
                output_path=self.output_path(bid),
                job_base_name=self.job_name(bid),
            )
            if not state.input_path.exists():
                pack_fit_batch(
                    state.input_path,
                    {s: shot_inputs[s] for s in shots},
                    x_star,
                    min_points=min_points,
                    scale_per_slice=scale_per_slice,
                )
                logger.info(f"Packed batch {bid} with {len(shots)} shots at {state.input_path}")
            if state.output_path.exists():
                state.done = True
                logger.info(f"Batch {bid}: output already present locally, skipping job")
            batches.append(state)

        self._run_jobs(batches)
        results = self._collect_results(batches, shot_inputs)
        summary = self._run_summary(batches, results)
        if any(r is None for r in results.values()):
            logger.error(summary)
        else:
            logger.info(summary)
        return results

    def clean(self) -> None:
        """Cancel this device's queued/running jobs and remove its batch files, local and remote.

        Call before run() for a from-scratch fit (CLI --clean): otherwise
        plan_batches/_run_jobs would adopt the cancelled jobs or reuse
        leftover batch outputs on the cluster.
        """
        prefix = f"{self.config.job_name_prefix}-{self.device}-"
        for name, job_id in self.backend.queued_jobs():
            if not name.startswith(prefix):
                continue
            logger.info(f"Clean: cancelling job {name} (id {job_id})")
            try:
                self.backend.cancel(job_id)
            except Exception as e:
                logger.warning(f"Clean: failed to cancel job {name} (id {job_id}): {e}")

        self._wait_for_jobs_to_drain(prefix)

        # Remove by remote glob, not by mirroring the local batch listing:
        # remote files with no local counterpart (e.g. outputs from a run
        # whose staging was already cleaned) would otherwise survive and be
        # adopted as pre-existing results by the next run. The trailing * also
        # takes the .npz.tmp a killed worker leaves behind mid-write. Raises on
        # failure so a clean that did not actually clean stops the run.
        self.backend.remove_glob(self.config.remote_workdir, "batch_*.npz*")
        if self.batches_dir.exists():
            shutil.rmtree(self.batches_dir)
        self.batches_dir.mkdir(parents=True, exist_ok=True)

    def _wait_for_jobs_to_drain(self, prefix: str) -> None:
        """Block until no job named with prefix is left in the queue.

        scancel returns as soon as it is issued, but SLURM only SIGTERMs (then
        SIGKILLs) the job some time later. Deleting the batch files before the
        job is really gone lets it write batch_<bid>_out.npz AFTER the delete,
        and the next run pulls that file back and adopts the OLD fit - silently,
        which is the exact failure clean exists to prevent. Raises rather than
        deleting anyway: a stuck job (e.g. wedged in COMPLETING) needs a human,
        and proceeding would quietly reuse stale fits.
        """
        deadline = time.monotonic() + _CLEAN_DRAIN_TIMEOUT_S
        while True:
            remaining = [(name, job_id) for name, job_id in self.backend.queued_jobs() if name.startswith(prefix)]
            if not remaining:
                return
            if time.monotonic() >= deadline:
                listed = ", ".join(f"{name} (id {job_id})" for name, job_id in remaining)
                raise RuntimeError(
                    f"Clean: {len(remaining)} {prefix}* jobs still queued {_CLEAN_DRAIN_TIMEOUT_S:.0f}s after cancelling: {listed}. "
                    "Not removing batch files - a job that outlives the delete would leave a stale output for the next run to adopt. "
                    "Wait for the queue to clear (or scancel them by hand) and rerun."
                )
            logger.info(f"Clean: waiting for {len(remaining)} cancelled jobs to leave the queue")
            time.sleep(_CLEAN_DRAIN_POLL_S)

    # ------------------------------------------------------------------
    def _run_jobs(self, batches: list[BatchState]) -> None:
        todo = [b for b in batches if not b.done]
        if not todo:
            return

        # A previous run's job may have produced output that never made it
        # back (e.g. the pull failed transiently), so check the cluster
        # before submitting anything.
        for state in todo:
            remote_out = f"{self.config.remote_workdir}/{state.output_path.name}"
            if self.backend.pull_file(remote_out, self.batches_dir):
                state.done = True
                logger.info(f"Batch {state.bid}: pulled existing results from cluster, skipping job")
        todo = [b for b in todo if not b.done]
        if not todo:
            return

        logger.info(f"Uploading worker script to {self.config.remote_workdir}")
        self.backend.push_file(WORKER_SOURCE, self.config.remote_workdir)
        # sbatch does not create --output directories
        self.backend.ensure_dir(f"{self.config.remote_workdir}/logs")

        # Adopt jobs already in the queue from a previous run
        queued = self.backend.queued_job_names()
        for state in todo:
            self._adopt_queued_job(state, queued)

        while True:
            self._poll_finished(todo)
            self._submit_ready(todo)
            remaining = [b for b in todo if not b.done and not b.failed]
            if not remaining:
                break
            n_active = len([b for b in remaining if b.job_id is not None])
            logger.info(
                f"Waiting on {len(remaining)} batches ({n_active} jobs active), polling again in {self.config.poll_interval_s:.0f}s"
            )
            time.sleep(self.config.poll_interval_s)

    def _submit_ready(self, todo: list[BatchState]) -> None:
        active = [b for b in todo if b.job_id is not None and not b.done and not b.failed]
        budget = self.config.max_concurrent_jobs - len(active)
        for state in todo:
            if budget <= 0:
                break
            if state.done or state.failed or state.job_id is not None:
                continue
            state.job_id = self._submit_batch(state)
            budget -= 1

    def _adopt_queued_job(self, state: BatchState, queued: dict[str, int]) -> None:
        """Adopt a queued job from a previous run instead of resubmitting.

        Job names carry an attempt suffix (-a{n})
        The highest attempt wins and lower-attempt stragglers are cancelled.
        """
        candidates: list[tuple[int, int]] = []  # (attempt, job_id)
        for name, job_id in queued.items():
            if name.startswith(f"{state.job_base_name}-a"):
                suffix = name.removeprefix(f"{state.job_base_name}-a")
                if suffix.isdigit():
                    candidates.append((int(suffix), job_id))
        if not candidates:
            return
        candidates.sort()
        state.attempt, state.job_id = candidates[-1]
        state.pending_since = time.monotonic()
        logger.info(f"Batch {state.bid}: found existing job {state.job_id} (attempt {state.attempt}) in queue, not resubmitting")
        for _, stale_id in candidates[:-1]:
            logger.info(f"Batch {state.bid}: cancelling stale lower-attempt job {stale_id}")
            try:
                self.backend.cancel(stale_id)
            except Exception as e:
                logger.warning(f"Batch {state.bid}: failed to cancel stale job {stale_id}: {e}")

    def _render_script(self, state: BatchState, part: PartitionSpec) -> str:
        workdir = self.config.remote_workdir
        lines = [
            "#!/bin/bash",
            "",
            f"#SBATCH --job-name={state.job_name}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            f"#SBATCH --cpus-per-task={self.config.cpus_per_job}",
        ]
        if self.config.memory_per_node:
            lines.append(f"#SBATCH --mem={self.config.memory_per_node}")
        lines.append(f"#SBATCH --time={part.time_limit}")
        lines.append(f"#SBATCH --partition={part.name}")
        if part.constraint:
            lines.append(f"#SBATCH --constraint={part.constraint}")
        lines += [
            f"#SBATCH --output={workdir}/logs/%x_%j.log",
            f"#SBATCH --error={workdir}/logs/%x_%j.log",
            f"#SBATCH --chdir={workdir}",
            "#SBATCH --wait-all-nodes=1",
            "",
            "set -euxo pipefail",
            "",
            f"source '{self.config.venv_path}/bin/activate'",
            "",
            f"srun python {workdir}/{WORKER_FILENAME} {workdir}/{state.input_path.name} "
            f"{workdir}/{state.output_path.name} --num-workers {self.config.cpus_per_job}",
            "",
        ]
        return "\n".join(lines)

    def _submit_batch(self, state: BatchState) -> int:
        workdir = self.config.remote_workdir
        self.backend.push_file(state.input_path, workdir)

        part = self.config.partitions[state.partition_idx]
        state.attempt += 1
        script = self._render_script(state, part)
        job_id = self.backend.submit_script(script, state.job_name)
        state.pending_since = time.monotonic()
        logger.info(
            f"Batch {state.bid}: submitted job {state.job_name} (id {job_id}, {len(state.shots)} shots, "
            f"partition {part.name}, attempt {state.attempt})"
        )
        return job_id

    def _poll_finished(self, todo: list[BatchState]) -> None:
        active = [b for b in todo if b.job_id is not None and not b.done and not b.failed]
        if not active:
            return
        states = self.backend.job_states([b.job_id for b in active])
        for state in active:
            slurm_state = states.get(state.job_id, "UNKNOWN")
            if slurm_state == "PENDING":
                self._check_pending_timeout(state)
                continue
            if slurm_state in ("RUNNING", "COMPLETING", "CONFIGURING"):
                state.pending_since = None
                continue
            # Terminal or unknown: the output file is the source of truth
            remote_out = f"{self.config.remote_workdir}/{state.output_path.name}"
            if self.backend.pull_file(remote_out, self.batches_dir):
                state.done = True
                logger.info(f"Batch {state.bid}: job {state.job_id} finished, results pulled back")
            elif slurm_state in _TERMINAL_FAILURE_STATES:
                self._handle_failure(
                    state,
                    f"job {state.job_id} ended in state {slurm_state} without producing {remote_out}",
                )
            elif slurm_state == "COMPLETED":
                # The job claims success, so the output may exist but be
                # unreachable (login node dropping connections) or still
                # in flight. Keep trying for a few polls before giving up.
                state.output_pull_polls += 1
                if state.output_pull_polls >= _MAX_OUTPUT_PULL_POLLS:
                    self._handle_failure(
                        state,
                        f"job {state.job_id} COMPLETED but {remote_out} could not be pulled after {state.output_pull_polls} polls",
                    )
                else:
                    logger.warning(
                        f"Batch {state.bid}: job {state.job_id} COMPLETED but output not retrieved yet "
                        f"(poll {state.output_pull_polls}/{_MAX_OUTPUT_PULL_POLLS}), will retry"
                    )
            else:
                # UNKNOWN (e.g. the job vanished from the queue): tolerate a
                # few polls, then treat the attempt as failed
                state.unknown_polls += 1
                if state.unknown_polls >= _MAX_UNKNOWN_POLLS:
                    self._handle_failure(
                        state,
                        f"job {state.job_id} in state {slurm_state} for {state.unknown_polls} polls with no output",
                    )
                else:
                    logger.warning(f"Batch {state.bid}: job {state.job_id} state {slurm_state}, no output yet")

    def _check_pending_timeout(self, state: BatchState) -> None:
        """Cancel a job stuck PENDING too long and move it to the next partition.

        Stops hopping after one full cycle through the partition list:
        if every partition is congested, cancelling only resets the batch's
        queue position.
        """
        if state.pending_since is None:
            # Job returned to PENDING (e.g. preemption requeue): restart the clock
            state.pending_since = time.monotonic()
            return
        if len(self.config.partitions) < 2 or state.pending_hops >= len(self.config.partitions):
            return
        elapsed = time.monotonic() - state.pending_since
        if elapsed <= self.config.pending_timeout_s:
            return
        old_part = self.config.partitions[state.partition_idx].name
        state.partition_idx = (state.partition_idx + 1) % len(self.config.partitions)
        state.pending_hops += 1
        new_part = self.config.partitions[state.partition_idx].name
        logger.warning(
            f"Batch {state.bid}: job {state.job_id} PENDING for {elapsed:.0f}s on {old_part}, cancelling and falling back to {new_part}"
        )
        try:
            self.backend.cancel(state.job_id)
        except Exception as e:
            logger.warning(f"Batch {state.bid}: failed to cancel job {state.job_id}: {e}")
        state.reset_for_resubmit()
        if state.pending_hops >= len(self.config.partitions):
            logger.warning(f"Batch {state.bid}: tried every partition for pending fallback, will wait in queue from now on")

    def _handle_failure(self, state: BatchState, reason: str) -> None:
        """Retry a failed batch on the next partition, or give up past max_retries."""
        state.failures += 1
        part = self.config.partitions[state.partition_idx]
        if state.failures > self.config.max_retries:
            state.failed = True
            state.fail_reason = (
                f"{reason} (partition {part.name}, attempt {state.attempt}, {state.failures - 1}/{self.config.max_retries} retries used)"
            )
            logger.error(f"Batch {state.bid}: {state.fail_reason}; giving up, see logs in {self.config.remote_workdir}/logs")
            return
        state.partition_idx = (state.partition_idx + 1) % len(self.config.partitions)
        next_part = self.config.partitions[state.partition_idx].name
        logger.warning(f"Batch {state.bid}: {reason}; retry {state.failures}/{self.config.max_retries} on partition {next_part}")
        state.reset_for_resubmit()

    # ------------------------------------------------------------------
    def _collect_results(
        self,
        batches: list[BatchState],
        shot_inputs: dict[int, ShotFitInput],
    ) -> dict[int, ShotFitOutput | None]:
        results: dict[int, ShotFitOutput | None] = dict.fromkeys(shot_inputs)
        for state in batches:
            if not state.output_path.exists():
                logger.error(f"Batch {state.bid} produced no results; {len(state.shots)} shots will be skipped")
                continue
            try:
                outputs = unpack_fit_results(state.output_path)
            except Exception as e:
                logger.error(f"Failed to read results for batch {state.bid}: {e}")
                continue
            for shot in state.shots:
                if shot in outputs:
                    results[shot] = outputs[shot]
                else:
                    logger.error(f"Batch {state.bid} results missing shot {shot}")
        return results

    @staticmethod
    def _run_summary(
        batches: list[BatchState],
        results: dict[int, ShotFitOutput | None],
    ) -> str:
        """Reconciliation report: every requested shot is accounted for."""
        failed_shots = sorted(s for s, r in results.items() if r is None)
        lines = [
            f"GP fit reconciliation: {len(results)} shots requested, {len(results) - len(failed_shots)} fitted, {len(failed_shots)} FAILED"
        ]
        for state in batches:
            batch_failed = [s for s in state.shots if results.get(s) is None]
            if not batch_failed:
                continue
            reason = state.fail_reason or "output missing, unreadable, or incomplete"
            lines.append(f"  batch {state.bid} (attempts {state.attempt}): {reason}; shots: {', '.join(str(s) for s in batch_failed)}")
        if failed_shots:
            lines.append(f"  FAILED shots ({len(failed_shots)}): {', '.join(str(s) for s in failed_shots)}")
        return "\n".join(lines)
