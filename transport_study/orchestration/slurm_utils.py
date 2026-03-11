import re
import subprocess

from loguru import logger

from transport_study.config import config


def count_running_jobs(job_name: str, partition: str = config.partition) -> int:
    """Run squeue to list running jobs on the partition with the specific name"""
    result = subprocess.run(
        ["squeue", "-p", partition, "-n", job_name, "--state=RUNNING", "--noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.critical(f"squeue failed: {result.stderr}")
        return 999999  # Return a large number to prevent launching more jobs if squeue fails
    # Count lines (each line is a job)
    return len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0


def count_idle_gpus(partition: str = config.partition) -> int:
    """Count the number of idle GPUs on this partition."""
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
    return total - used


def resources_available(
    partition: str = config.partition, buffer_gpus: int = config.buffer_gpus
) -> bool:
    idle_gpus = count_idle_gpus(partition)
    return idle_gpus >= buffer_gpus
