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
