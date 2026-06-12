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

Jobs get deterministic names gpfit-{device}-{batch_id} (batch_id is a hash of
the shot list), so a restarted workflow finds in-flight jobs instead of
resubmitting them.
"""

import hashlib
import shutil
import time
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
class ClusterFitConfig:
    """Launch options for cluster-based GP fitting.

    Parameters
    ----------
    profile : str
        srunx SSH profile name, or "local" when already running on the
        target cluster (shared filesystem, local sbatch).
    partition : str
        SLURM partition for the fitting jobs. gptools is CPU-only, so this
        must be a CPU partition.
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
        50 shots on 32 CPUs is ~1 hour wall time, safely under a 4 hour
        job limit.
    cpus_per_job : int
        cpus-per-task for each fitting job; the worker runs this many
        slice-fit processes.
    """

    profile: str
    partition: str
    remote_workdir: str
    venv_path: str
    max_concurrent_jobs: int = 8
    shots_per_batch: int = 50
    cpus_per_job: int = 32
    memory_per_node: str | None = None
    time_limit: str = "3:50:00"
    poll_interval_s: float = 60.0
    job_name_prefix: str = "gpfit"


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


class _SSHBackend:
    """File transfer and job control on a remote cluster via srunx."""

    def __init__(self, config: ClusterFitConfig):
        from srunx.slurm.clients.ssh import ConfigManager, SlurmSSHClient
        from srunx.sync import RsyncClient

        profile = ConfigManager().get_profile(config.profile)
        if profile is None:
            raise ValueError(
                f"srunx SSH profile '{config.profile}' not found. Create it with: srunx ssh profile add {config.profile} --ssh-host <host>"
            )
        self._username = profile.username
        self._client = SlurmSSHClient(profile_name=config.profile)
        self._rsync = RsyncClient(
            hostname=profile.hostname,
            username=profile.username,
            port=profile.port,
            key_filename=profile.key_filename,
            proxy_jump=profile.proxy_jump,
        )

    def push_file(self, local: Path, remote_dir: str) -> None:
        result = self._rsync.push(str(local), f"{remote_dir}/", delete=False)
        if result.returncode != 0:
            raise RuntimeError(f"rsync push of {local} failed: {result.stderr}")

    def pull_file(self, remote_path: str, local_dir: Path) -> bool:
        """Pull a single remote file into local_dir. Returns False if unavailable."""
        local_dir.mkdir(parents=True, exist_ok=True)
        result = self._rsync.pull(remote_path, f"{local_dir}/")
        return result.returncode == 0 and (local_dir / Path(remote_path).name).exists()

    def submit(self, job) -> int:
        return self._client.submit(job).job_id

    def queued_job_names(self) -> dict[str, int]:
        """Names of this user's queued/running jobs -> job id."""
        return {j.name: j.job_id for j in self._client.queue(user=self._username) if j.job_id}

    def job_states(self, job_ids: list[int]) -> dict[int, str]:
        return {jid: snap.status for jid, snap in self._client.queue_by_ids(job_ids).items()}


class _LocalBackend:
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

    def submit(self, job) -> int:
        return self._client.submit(job).job_id

    def queued_job_names(self) -> dict[str, int]:
        return {j.name: j.job_id for j in self._client.queue(user=self._username) if j.job_id}

    def job_states(self, job_ids: list[int]) -> dict[int, str]:
        return {jid: snap.status for jid, snap in self._client.queue_by_ids(job_ids).items()}


@dataclass
class _BatchState:
    bid: str
    shots: list[int]
    input_path: Path
    output_path: Path
    job_name: str
    job_id: int | None = None
    done: bool = False
    failed: bool = False


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

        batches: list[_BatchState] = []
        for bid, shots in plan.items():
            state = _BatchState(
                bid=bid,
                shots=shots,
                input_path=self.input_path(bid),
                output_path=self.output_path(bid),
                job_name=self.job_name(bid),
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
        return self._collect_results(batches, shot_inputs)

    # ------------------------------------------------------------------
    def _run_jobs(self, batches: list[_BatchState]) -> None:
        todo = [b for b in batches if not b.done]
        if not todo:
            return

        logger.info(f"Uploading worker script to {self.config.remote_workdir}")
        self.backend.push_file(WORKER_SOURCE, self.config.remote_workdir)

        # Adopt jobs already in the queue from a previous run
        queued = self.backend.queued_job_names()
        for state in todo:
            if state.job_name in queued:
                state.job_id = queued[state.job_name]
                logger.info(f"Batch {state.bid}: found existing job {state.job_id} in queue, not resubmitting")

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

    def _submit_ready(self, todo: list[_BatchState]) -> None:
        active = [b for b in todo if b.job_id is not None and not b.done and not b.failed]
        budget = self.config.max_concurrent_jobs - len(active)
        for state in todo:
            if budget <= 0:
                break
            if state.done or state.failed or state.job_id is not None:
                continue
            state.job_id = self._submit_batch(state)
            budget -= 1

    def _submit_batch(self, state: _BatchState) -> int:
        from srunx import Job, JobEnvironment, JobResource

        workdir = self.config.remote_workdir
        self.backend.push_file(state.input_path, workdir)

        job = Job(
            name=state.job_name,
            command=[
                "python",
                f"{workdir}/{WORKER_FILENAME}",
                f"{workdir}/{state.input_path.name}",
                f"{workdir}/{state.output_path.name}",
                "--num-workers",
                str(self.config.cpus_per_job),
            ],
            resources=JobResource(
                nodes=1,
                ntasks_per_node=1,
                cpus_per_task=self.config.cpus_per_job,
                partition=self.config.partition,
                time_limit=self.config.time_limit,
                memory_per_node=self.config.memory_per_node,
            ),
            environment=JobEnvironment(venv=self.config.venv_path),
            work_dir=workdir,
            log_dir=f"{workdir}/logs",
        )
        job_id = self.backend.submit(job)
        logger.info(f"Batch {state.bid}: submitted job {state.job_name} (id {job_id}, {len(state.shots)} shots)")
        return job_id

    def _poll_finished(self, todo: list[_BatchState]) -> None:
        active = [b for b in todo if b.job_id is not None and not b.done and not b.failed]
        if not active:
            return
        states = self.backend.job_states([b.job_id for b in active])
        for state in active:
            slurm_state = states.get(state.job_id, "UNKNOWN")
            if slurm_state in ("PENDING", "RUNNING", "COMPLETING", "CONFIGURING"):
                continue
            # Terminal or unknown: the output file is the source of truth
            remote_out = f"{self.config.remote_workdir}/{state.output_path.name}"
            if self.backend.pull_file(remote_out, self.batches_dir):
                state.done = True
                logger.info(f"Batch {state.bid}: job {state.job_id} finished, results pulled back")
            elif slurm_state in _TERMINAL_FAILURE_STATES or slurm_state == "COMPLETED":
                state.failed = True
                logger.error(
                    f"Batch {state.bid}: job {state.job_id} ended in state {slurm_state} without producing "
                    f"{remote_out}; see logs in {self.config.remote_workdir}/logs"
                )
            else:
                # UNKNOWN with no output yet: give it until the next poll
                logger.warning(f"Batch {state.bid}: job {state.job_id} state {slurm_state}, no output yet")

    # ------------------------------------------------------------------
    def _collect_results(
        self,
        batches: list[_BatchState],
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
