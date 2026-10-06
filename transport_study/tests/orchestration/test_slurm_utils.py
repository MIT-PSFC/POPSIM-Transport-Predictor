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
        queried_nodes = cmd[cmd.index("-w") + 1].split(",")
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


def test_count_idle_gpus_counts_free_and_preemptable_gpus_on_schedulable_nodes(monkeypatch):
    """Free GPUs on nodes taking jobs plus those held there by lower-tier preemptable partitions.

    node01 holds 2 preemptable GPUs (job 101 and job 102's node01 share, its node99 share is outside).
    node04 has 1 free GPU, the shared, quicktest and normal jobs there are not preemptable from tier 50.
    node02 / node03 are full (the 2-node job holds 8), drained node05 counts nothing.
    2 + 1 less a buffer of 1 leaves 2.
    """

    def run(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout=_fake_slurm_stdout(cmd), stderr="")

    monkeypatch.setattr(slurm_utils.subprocess, "run", run)
    assert count_idle_gpus("fake_psfc_gpu", 1) == 2


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
