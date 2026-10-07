import os
import shutil
import sys
from types import SimpleNamespace

import pytest

from transport_study.config import StudyConfig, load_config
from transport_study.orchestration import slurm_utils
from transport_study.orchestration.slurm_utils import (
    count_idle_gpus,
    importable_module,
    open_cpu_slots,
    open_slots,
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
PartitionName=fake_shared_gpu PriorityTier=50 PreemptMode=OFF
PartitionName=fake_normal_gpu PriorityTier=25 PreemptMode=OFF
PartitionName=fake_quicktest PriorityTier=100 PreemptMode=REQUEUE
PartitionName=fake_preemptable PriorityTier=1 PreemptMode=REQUEUE
"""

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
    if cmd == ["scontrol", "show", "config"]:
        return "PreemptMode             = REQUEUE\nPreemptType             = preempt/partition_prio\n"
    if cmd == ["scontrol", "show", "partition", "-o"]:
        return FAKE_PARTITION_TABLE
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
        for job_id, partition, _, nodes, gpus in FAKE_RUNNING_JOBS:
            if partition not in queried_partitions or not set(nodes) & set(queried_nodes):
                continue
            gpu_tres = f",gres/gpu={gpus * len(nodes)},gres/gpu:a100={gpus * len(nodes)}" if gpus else ""
            lines.append(f"{job_id}  cpu=4,mem=16G,node={len(nodes)},billing=4{gpu_tres}")
        return "\n".join(lines) + "\n"
    if cmd[:5] == ["scontrol", "-d", "-o", "show", "job"]:
        job_id, partition, hostlist, nodes, gpus = next(job for job in FAKE_RUNNING_JOBS if job[0] == cmd[5])
        return f"JobId={job_id} Partition={partition} NodeList={hostlist} NumNodes={len(nodes)} Nodes={hostlist} CPU_IDs=0-3 Mem=16384 GRES=gpu:a100:{gpus}(IDX:0)\n"
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


def test_cpu_capable_cases_keep_their_own_gpu_buffer(monkeypatch):
    """A negative buffer_gpus queues GPU-only jobs past the 3 idle GPUs,
    while cpu_capable_buffer_gpus keeps the last 3 for them."""
    _fake_config(partition="fake_psfc_gpu", buffer_gpus=-1, cpu_partition="fake_cpu", cpu_capable_buffer_gpus=3)
    _serve_fake_slurm(monkeypatch, _fake_slurm_stdout)
    assert open_slots("fake_psfc_gpu") == 4
    assert open_slots("fake_psfc_gpu", cpu_capable=True) == 0


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


def _fake_cpu_slurm_stdout(qos_caps: str):
    def fake_stdout(cmd: list[str]) -> str:
        if cmd[:3] == ["scontrol", "show", "partition"]:
            qos = "fake_qos" if qos_caps else "N/A"
            return f"PartitionName={cmd[3]} QoS={qos} MaxTime=12:00:00\n"
        if cmd[0] == "sacctmgr":
            return qos_caps + "\n"
        if cmd[0] == "sinfo":
            return "".join(f"{node}  {cpus}  {memory}  {alloc}  {state}\n" for node, cpus, memory, alloc, state in FAKE_CPU_NODES)
        if cmd[0] == "squeue" and "--state=PENDING" in cmd:
            return "1001\n"
        if cmd[0] == "squeue" and "tres-alloc:200" in cmd:
            return "\n".join(FAKE_USER_CPU_TRES) + "\n"
        if cmd[0] == "squeue":
            # Every job of this user anywhere, for the user-wide job budget
            return "".join(f"{job_id}\n" for job_id in range(7))
        raise AssertionError(f"unexpected SLURM command {cmd}")

    return fake_stdout


# The partition QOS lookups are cached per partition name, so every scenario gets its own partition
@pytest.mark.parametrize(
    ("partition", "qos_caps", "slots"),
    [
        # 2 + 8 + 1 node slots, less the 1 pending job
        ("fake_cpu_uncapped", "", 10),
        # (386G - 3 x 48G) // 48G = 5 jobs of memory left under the QOS cap, CPUs would allow 21
        ("fake_cpu_capped", "cpu=96,mem=386G", 5),
    ],
)
def test_open_cpu_slots_fits_jobs_by_cpus_memory_and_qos(monkeypatch, partition, qos_caps, slots):
    _fake_config(cpu_partition=partition, cpu_train_cpus=4, train_mem="48G")
    _serve_fake_slurm(monkeypatch, _fake_cpu_slurm_stdout(qos_caps))
    assert open_cpu_slots(partition) == slots


def test_cpu_partitions_keep_the_user_job_budget_without_spillover(monkeypatch):
    """With no GPU spillover configured the user-wide job ceiling still bounds a CPU partition: 10 - 2 - 7 jobs leaves 1."""
    _fake_config(cpu_partition="fake_cpu_budget", max_user_jobs=10, spillover_job_headroom=2, spillover_partitions=(), train_mem="48G")
    _serve_fake_slurm(monkeypatch, _fake_cpu_slurm_stdout(""))
    assert open_slots("fake_cpu_budget") == 1


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
