import os
import shutil
import sys
from datetime import datetime
from types import SimpleNamespace

import pytest

from transport_study.config import StudyConfig, load_config
from transport_study.orchestration import slurm_utils
from transport_study.orchestration.slurm_utils import (
    LaunchSlots,
    TrainingDevices,
    count_idle_gpus,
    importable_module,
    open_cpu_slots,
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


FAKE_PARTITION_TABLE = """\
PartitionName=fake_psfc_gpu PriorityTier=50 PreemptMode=OFF
PartitionName=fake_psfc_cpu PriorityTier=50 PreemptMode=OFF
PartitionName=fake_shared_gpu PriorityTier=50 PreemptMode=OFF
PartitionName=fake_normal_gpu PriorityTier=25 PreemptMode=OFF
PartitionName=fake_normal_cpu PriorityTier=25 PreemptMode=OFF
PartitionName=fake_quicktest PriorityTier=100 PreemptMode=REQUEUE
PartitionName=fake_preemptable PriorityTier=1 PreemptMode=REQUEUE
"""

# The preempt/partition_prio scheduler config and its partition table, by command
FAKE_PREEMPTION_STDOUT = {
    ("scontrol", "show", "config"): "PreemptMode             = REQUEUE\nPreemptType             = preempt/partition_prio\n",
    ("scontrol", "show", "partition", "-o"): FAKE_PARTITION_TABLE,
}

FAKE_NODE_STATES = {"node01": "mix-", "node02": "alloc", "node03": "alloc", "node04": "mix", "node05": "drain"}

# (job id, partition, hostlist, nodes, GPUs per node), node99 lies outside the partition
FAKE_RUNNING_JOBS = [
    ("101", "fake_preemptable", "node01", ("node01",), 1),
    ("102", "fake_preemptable", "node[01,99]", ("node01", "node99"), 1),
    ("103", "fake_preemptable", "node01", ("node01",), 0),
    ("104", "fake_psfc_gpu", "node[02-03]", ("node02", "node03"), 4),
    ("105", "fake_shared_gpu", "node04", ("node04",), 1),
    ("106", "fake_quicktest", "node04", ("node04",), 1),
    ("107", "fake_psfc_gpu", "node01", ("node01",), 2),
    ("108", "fake_preemptable", "node05", ("node05",), 1),
    ("109", "fake_normal_gpu", "node04", ("node04",), 1),
]

FAKE_HOSTNAMES = {"node[01,99]": "node01\nnode99\n", "node[02-03]": "node02\nnode03\n"}


def _fake_slurm_stdout(cmd: list[str]) -> str:
    """A scheduler holding FAKE_RUNNING_JOBS on the FAKE_NODE_STATES nodes, 4 a100s each."""
    if tuple(cmd) in FAKE_PREEMPTION_STDOUT:
        return FAKE_PREEMPTION_STDOUT[tuple(cmd)]
    if cmd[:3] == ["scontrol", "show", "hostnames"]:
        return FAKE_HOSTNAMES[cmd[3]]
    if cmd[0] == "sinfo":
        lines = []
        for node, state in FAKE_NODE_STATES.items():
            gpus_used = sum(gpus for _, _, _, nodes, gpus in FAKE_RUNNING_JOBS if node in nodes)
            lines.append(f"{node}  gpu:a100:4(S:0-1)  gpu:a100:{gpus_used}(IDX:N/A)  {state}")
        return "\n".join(lines) + "\n"
    if cmd[0] == "squeue":
        # Only the user's pending-job query has no node list, and this user has no pending jobs
        queried_nodes = cmd[cmd.index("-w") + 1].split(",") if "-w" in cmd else []
        queried_partitions = cmd[cmd.index("-p") + 1].split(",")
        lines = []
        for job_id, partition, hostlist, nodes, gpus in FAKE_RUNNING_JOBS:
            if partition not in queried_partitions or not set(nodes) & set(queried_nodes):
                continue
            gpu_tres = f",gres/gpu={gpus * len(nodes)},gres/gpu:a100={gpus * len(nodes)}" if gpus else ""
            job_tres = f"cpu={4 * len(nodes)},mem={16 * len(nodes)}G,node={len(nodes)},billing=4{gpu_tres}"
            lines.append(f"{job_id}|2026-10-01T00:00:00|{job_tres}|{hostlist}")
        return "\n".join(lines) + "\n"
    if cmd[:5] == ["scontrol", "-d", "-o", "show", "job"]:
        job_id, partition, hostlist, nodes, gpus = next(job for job in FAKE_RUNNING_JOBS if job[0] == cmd[5])
        gres = f"gpu:a100:{gpus}(IDX:0)" if gpus else ""
        return f"JobId={job_id} Partition={partition} NodeList={hostlist} NumNodes={len(nodes)} Nodes={hostlist} CPU_IDs=0-3 Mem=16384 GRES={gres}\n"
    raise AssertionError(f"unexpected SLURM command {cmd}")


def _fake_config(**overrides):
    return load_config(StudyConfig(study_name="test_slurm_utils", dataset_paths={}, target_device="test_device", **overrides))


def _serve_fake_slurm(monkeypatch, fake_stdout):
    def run(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout=fake_stdout(cmd), stderr="")

    monkeypatch.setattr(slurm_utils.subprocess, "run", run)


@pytest.mark.parametrize(("exclude_nodes", "idle_gpus"), [((), 2), (("node04",), 1), (("node01",), 0)])
def test_count_idle_gpus_counts_free_and_preemptable_gpus_on_schedulable_nodes(monkeypatch, exclude_nodes, idle_gpus):
    """Free GPUs on nodes taking jobs plus those held there by lower-tier preemptable partitions.

    node01 holds 2 preemptable GPUs (job 101 and job 102's node01 share, its node99 share is outside).
    node04 has 1 free GPU, the shared, quicktest and normal jobs there are not preemptable from tier 50.
    node02 / node03 are full (the 2-node job holds 8), drained node05 counts nothing.
    2 + 1 less a buffer of 1 leaves 2, and an excluded node's GPUs count for nothing.
    """
    _fake_config(exclude_nodes=exclude_nodes)
    _serve_fake_slurm(monkeypatch, _fake_slurm_stdout)
    assert count_idle_gpus("fake_psfc_gpu", 1) == idle_gpus


def test_gpu_buffers_share_one_primary_count_read_once_per_pass(monkeypatch):
    """gpu_pending_jobs queues one GPU-only job past the 3 idle GPUs,
    while cpu_capable_buffer_gpus keeps the last one for them.

    Both read one count, read once, and every accepted job debits it.
    """
    _fake_config(partition="fake_psfc_gpu", buffer_gpus=0, gpu_pending_jobs=1, cpu_partition="fake_cpu", cpu_capable_buffer_gpus=1)
    slurm_commands: list[list[str]] = []

    def recorded_stdout(cmd):
        slurm_commands.append(cmd)
        return _fake_slurm_stdout(cmd)

    _serve_fake_slurm(monkeypatch, recorded_stdout)
    slots = LaunchSlots()
    assert slots.available("fake_psfc_gpu", TrainingDevices.GPU_ONLY) == 4
    n_commands_first_read = len(slurm_commands)

    assert slots.available("fake_psfc_gpu", TrainingDevices.CPU_CAPABLE) == 2
    slots.take("fake_psfc_gpu")
    slots.take("fake_psfc_gpu")
    assert slots.available("fake_psfc_gpu", TrainingDevices.GPU_ONLY) == 2
    assert slots.available("fake_psfc_gpu", TrainingDevices.CPU_CAPABLE) == 0
    assert slots.pick(TrainingDevices.GPU_ONLY) == "fake_psfc_gpu"
    assert len(slurm_commands) == n_commands_first_read


@pytest.mark.parametrize(
    ("buffer_gpus", "gpu_pending_jobs", "gpu_only_slots"),
    [(0, 0, 3), (1, 0, 2), (0, 2, 5), (5, 2, 2)],
)
def test_gpu_only_slots_are_the_gpus_past_the_buffer_plus_the_queue_allowance(monkeypatch, buffer_gpus, gpu_pending_jobs, gpu_only_slots):
    """Of 3 available GPUs the buffer leaves some idle, never fewer than none, and gpu_pending_jobs more jobs may wait in the queue."""
    _fake_config(partition="fake_psfc_gpu", buffer_gpus=buffer_gpus, gpu_pending_jobs=gpu_pending_jobs)
    _serve_fake_slurm(monkeypatch, _fake_slurm_stdout)
    assert LaunchSlots().available("fake_psfc_gpu", TrainingDevices.GPU_ONLY) == gpu_only_slots


@pytest.mark.parametrize(
    ("devices", "cpu_capable_buffer_gpus", "cpu_slots", "partition"),
    [
        (TrainingDevices.GPU_ONLY, 1, 5, "fake_gpu"),
        # More GPUs idle than its buffer
        (TrainingDevices.CPU_CAPABLE, 1, 5, "fake_gpu"),
        # No more idle GPUs than its buffer, so the CPU partition
        (TrainingDevices.CPU_CAPABLE, 3, 5, "fake_cpu"),
        # Never an idle GPU
        (TrainingDevices.CPU_ONLY, 1, 5, "fake_cpu"),
        # Held while the CPU partition is full
        (TrainingDevices.CPU_ONLY, 1, 0, None),
    ],
)
def test_each_device_kind_is_placed_on_its_own_partitions(monkeypatch, devices, cpu_capable_buffer_gpus, cpu_slots, partition):
    """3 GPUs idle on the primary partition, cpu_slots jobs fitting on the CPU partition."""
    _fake_config(partition="fake_gpu", cpu_partition="fake_cpu", buffer_gpus=0, cpu_capable_buffer_gpus=cpu_capable_buffer_gpus)
    monkeypatch.setattr(slurm_utils, "count_available_gpus", lambda partition_name: 3)
    monkeypatch.setattr(slurm_utils, "count_pending_jobs", lambda partition_name=None: 0)
    monkeypatch.setattr(slurm_utils, "open_cpu_slots", lambda partition_name: cpu_slots)
    monkeypatch.setattr(slurm_utils, "user_job_budget", lambda: 100)

    assert LaunchSlots().pick(devices) == partition


# (node, allocated/idle/other/total CPUs, memory MB, allocated memory MB, state) of a CPU partition
FAKE_CPU_NODES = [
    # 100000 MB free fits 2 jobs of 48G although 40 CPUs are idle
    ("cpu01", "24/40/0/64", 249087, 149087, "mix"),
    # 40 idle CPUs fit 10 jobs of 4, 400000 MB only 8
    ("cpu02", "24/40/0/64", 507002, 107002, "idle"),
    ("cpu03", "0/64/0/64", 507002, 0, "drain"),
    # Planned by backfill but still taking jobs, room for exactly one
    ("cpu04", "60/4/0/64", 249087, 199935, "mix-"),
]
# This user's TRES on the CPU partition, 2 running jobs and 1 pending one
FAKE_USER_CPU_TRES = ["cpu=4,mem=48G,node=1,billing=4"] * 3


def _fake_cpu_slurm_stdout(qos_caps: str | None):
    """qos_caps is the MaxTRESPU of the partition's QOS, None for a partition without a QOS.

    The scheduler preempts nothing (see test_open_cpu_slots_counts_preemptable_jobs_as_free).
    """

    def fake_stdout(cmd: list[str]) -> str:
        if cmd == ["scontrol", "show", "config"]:
            return "PreemptType             = preempt/none\n"
        if cmd[:3] == ["scontrol", "show", "partition"]:
            qos = "N/A" if qos_caps is None else "fake_qos"
            return f"PartitionName={cmd[3]} QoS={qos} MaxTime=12:00:00\n"
        if cmd[0] == "sacctmgr":
            return f"fake_qos|{qos_caps}\n"
        if cmd[0] == "sinfo":
            return "".join(f"{node}  {cpus}  {memory}  {alloc}  {state}\n" for node, cpus, memory, alloc, state in FAKE_CPU_NODES)
        if cmd[0] == "squeue" and "tres-alloc:200" in cmd:
            return "\n".join(FAKE_USER_CPU_TRES) + "\n"
        if cmd[0] == "squeue":
            # This user's 1 pending job on the partition, or every job of this user anywhere for the user-wide job budget
            job_count = 1 if "--state=PENDING" in cmd else 7
            return "".join(f"{job_id}\n" for job_id in range(job_count))
        raise AssertionError(f"unexpected SLURM command {cmd}")

    return fake_stdout


# The partition QOS lookups are cached per partition name, so every scenario gets its own partition
@pytest.mark.parametrize(
    ("partition", "qos_caps", "cpu_pending_jobs", "slots"),
    [
        # 2 + 8 + 1 node slots, less the 1 pending job
        ("fake_cpu_no_qos", None, 0, 10),
        # A QOS without MaxTRESPU caps nothing either
        ("fake_cpu_uncapped", "", 0, 10),
        # (386G - 3 x 48G) // 48G = 5 jobs of memory left under the QOS cap, CPUs would allow 21
        ("fake_cpu_capped", "cpu=96,mem=386G", 0, 5),
        # 2 more jobs may pend beyond what fits
        ("fake_cpu_pending_allowance", None, 2, 12),
        # but never past the QOS cap
        ("fake_cpu_capped_allowance", "cpu=96,mem=386G", 2, 5),
    ],
)
def test_open_cpu_slots_fits_jobs_by_cpus_memory_and_qos(monkeypatch, partition, qos_caps, cpu_pending_jobs, slots):
    _fake_config(cpu_partition=partition, cpu_train_cpus=4, train_mem="48G", cpu_pending_jobs=cpu_pending_jobs)
    _serve_fake_slurm(monkeypatch, _fake_cpu_slurm_stdout(qos_caps))
    assert open_cpu_slots(partition) == slots


# (node, allocated/idle/other/total CPUs, memory MB, allocated memory MB, state) of fake_psfc_cpu
FAKE_EVICTION_CPU_NODES = [
    ("cpu21", "64/0/0/64", 262144, 262144, "alloc"),
    ("cpu22", "24/40/0/64", 1048576, 163840, "mix"),
    ("cpu23", "64/0/0/64", 262144, 262144, "alloc"),
    ("cpu24", "64/0/0/64", 262144, 262144, "drain"),
]

# (job id, partition, hostlist, tres-alloc CPUs, (node, CPU IDs, memory MB) per node), cpu99 lies outside the partition
FAKE_CPU_RUNNING_JOBS = [
    ("201", "fake_preemptable", "cpu21", 16, (("cpu21", "0-15", 98304),)),
    ("202", "fake_normal_cpu", "cpu21", 48, (("cpu21", "16-63", 163840),)),
    ("203", "fake_preemptable", "cpu[22,99]", 40, (("cpu22", "0-47", 163840), ("cpu99", "0-31", 65536))),
    ("204", "fake_quicktest", "cpu23", 64, (("cpu23", "0-63", 262144),)),
    ("205", "fake_preemptable", "cpu24", 64, (("cpu24", "0-63", 262144),)),
]


def _fake_cpu_eviction_slurm_stdout(job_detail_queries: list[str]):
    """A scheduler holding FAKE_CPU_RUNNING_JOBS on the FAKE_EVICTION_CPU_NODES, each job detail query appended to job_detail_queries."""

    def fake_stdout(cmd: list[str]) -> str:
        if tuple(cmd) in FAKE_PREEMPTION_STDOUT:
            return FAKE_PREEMPTION_STDOUT[tuple(cmd)]
        if cmd[:3] == ["scontrol", "show", "partition"]:
            return f"PartitionName={cmd[3]} QoS=N/A MaxTime=12:00:00\n"
        if cmd[0] == "sinfo":
            return "".join(f"{node}  {cpus}  {memory}  {alloc}  {state}\n" for node, cpus, memory, alloc, state in FAKE_EVICTION_CPU_NODES)
        if cmd[0] == "squeue" and "--state=PENDING" in cmd:
            return ""
        if cmd[0] == "squeue":
            queried_nodes = set(cmd[cmd.index("-w") + 1].split(","))
            queried_partitions = cmd[cmd.index("-p") + 1].split(",")
            lines = []
            for job_id, partition, hostlist, job_cpus, node_details in FAKE_CPU_RUNNING_JOBS:
                job_nodes = {node for node, _, _ in node_details}
                if partition not in queried_partitions or not job_nodes & queried_nodes:
                    continue
                job_mem_MB = sum(mem_MB for _, _, mem_MB in node_details)
                job_tres = f"cpu={job_cpus},mem={job_mem_MB}M,node={len(node_details)},billing={job_cpus}"
                lines.append(f"{job_id}|2026-10-01T00:00:00|{job_tres}|{hostlist}")
            return "\n".join(lines) + "\n"
        if cmd[:5] == ["scontrol", "-d", "-o", "show", "job"]:
            job_detail_queries.append(cmd[5])
            node_details = next(job[4] for job in FAKE_CPU_RUNNING_JOBS if job[0] == cmd[5])
            groups = "".join(f" Nodes={node} CPU_IDs={cpu_ids} Mem={mem_MB} GRES=" for node, cpu_ids, mem_MB in node_details)
            return f"JobId={cmd[5]} NumNodes={len(node_details)}{groups}\n"
        raise AssertionError(f"unexpected SLURM command {cmd}")

    return fake_stdout


def test_open_cpu_slots_counts_preemptable_jobs_as_free(monkeypatch):
    """CPUs and memory held by lower-tier preemptable jobs on nodes taking jobs count as free.

    cpu21: job 201's 16 CPUs and 96G fit 2 jobs of 4 CPUs and 48G, the tier 25 job 202 is not preemptable.
    cpu22: job 203's 40 CPUs split 24 / 16 by its 48 and 32 CPU IDs, so 40 idle + 24 CPUs fit 16 jobs, memory 21.
    cpu23: the tier 100 job 204 is not preemptable, and drained cpu24 counts nothing.
    Only the two-node job 203 needs its scontrol detail, a one-node job's squeue tres-alloc is its node's share.
    """
    _fake_config(cpu_partition="fake_psfc_cpu", cpu_train_cpus=4, train_mem="48G")
    job_detail_queries: list[str] = []
    _serve_fake_slurm(monkeypatch, _fake_cpu_eviction_slurm_stdout(job_detail_queries))
    assert open_cpu_slots("fake_psfc_cpu") == 18
    assert job_detail_queries == ["203"]


def test_cpu_partitions_keep_the_user_job_budget_without_spillover(monkeypatch):
    """With no GPU spillover configured the user-wide job ceiling still bounds a CPU partition: 10 - 2 - 7 jobs leaves 1."""
    _fake_config(cpu_partition="fake_cpu_budget", max_user_jobs=10, spillover_job_headroom=2, spillover_partitions=(), train_mem="48G")
    _serve_fake_slurm(monkeypatch, _fake_cpu_slurm_stdout(""))
    slots = LaunchSlots()
    assert slots.available("fake_cpu_budget", TrainingDevices.CPU_ONLY) == 1
    slots.take("fake_cpu_budget")
    assert slots.available("fake_cpu_budget", TrainingDevices.CPU_ONLY) == 0


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


def test_submit_sbatch_returns_accepted_job_id(monkeypatch):
    _serve_fake_slurm(monkeypatch, lambda cmd: "Submitted batch job 25741062\n")

    assert slurm_utils.submit_sbatch("#!/bin/bash\n", "train_case.a.seed_0.study", "training") == 25741062


def test_job_end_states_reads_terminal_ends_and_skips_unrecorded_jobs(monkeypatch):
    """A terminal state carries its end as wall-clock seconds, a running job none, CANCELLED keeps only its state word."""
    sacct_stdout = "7|COMPLETED|2026-10-13T19:20:31\n8|RUNNING|Unknown\n9|CANCELLED by 1234|2026-10-13T19:21:00\n"
    _serve_fake_slurm(monkeypatch, lambda cmd: sacct_stdout)

    end_states = slurm_utils.job_end_states([7, 8, 9, 10])

    assert end_states[7] == slurm_utils.JobEnd("COMPLETED", datetime(2026, 10, 13, 19, 20, 31).timestamp())
    assert end_states[8] == slurm_utils.JobEnd("RUNNING", None)
    assert end_states[9] == slurm_utils.JobEnd("CANCELLED", datetime(2026, 10, 13, 19, 21, 0).timestamp())
    assert 10 not in end_states


def test_recent_submitted_jobs_keeps_the_newest_job_of_each_name(monkeypatch):
    """Only names of the given prefix and suffix, the training jobs of one study."""
    sacct_stdout = "7|train_case.a.seed_0.study\n9|train_case.a.seed_0.study\n8|agent_case.a.study\n10|train_case.b.seed_0.other\n"
    _serve_fake_slurm(monkeypatch, lambda cmd: sacct_stdout)

    recent_jobs = slurm_utils.recent_submitted_jobs("train_", ".study")

    assert {job_name: job.job_id for job_name, job in recent_jobs.items()} == {"train_case.a.seed_0.study": 9}


@pytest.mark.parametrize(
    ("end_states", "now", "is_held"),
    [
        ({}, 100.0, True),
        ({}, 400.0, False),
        ({7: slurm_utils.JobEnd("RUNNING", None)}, 1000.0, True),
        ({7: slurm_utils.JobEnd("COMPLETED", 50.0)}, 200.0, True),
        ({7: slurm_utils.JobEnd("COMPLETED", 50.0)}, 240.0, False),
        (None, 1000.0, True),
    ],
    ids=["unrecorded_within_lag", "unrecorded_past_lag", "sacct_still_running", "inside_grace", "past_grace", "sacct_failed"],
)
def test_held_job_names_holds_a_gone_job_until_it_is_resolved(monkeypatch, end_states, now, is_held):
    """A job submitted at t = 0 and gone from the queue, beside one still queued."""
    submitted_jobs = {"job.gone": slurm_utils.SubmittedJob(7, 0.0), "job.queued": slurm_utils.SubmittedJob(8, 0.0)}
    monkeypatch.setattr(slurm_utils, "job_end_states", lambda job_ids: end_states)
    monkeypatch.setattr(slurm_utils.time, "time", lambda: now)

    held = slurm_utils.held_job_names(submitted_jobs, running_job_names={"job.queued"})

    assert held == ({"job.gone"} if is_held else set())
    # A released job is no longer tracked, a queued one always is
    assert ("job.gone" in submitted_jobs) == is_held
    assert "job.queued" in submitted_jobs
