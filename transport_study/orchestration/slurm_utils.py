import os
import re
import subprocess
import sys
import tempfile

import yaml
from loguru import logger
from popsim.ml import TrainConfig

from transport_study.config import config


def count_running_jobs(job_name: str, partition: str = config.partition) -> int:
    """Run squeue to list running jobs on the partition with the specific name"""
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


def launch_train_parallel(
    train_config: TrainConfig,
    job_name: str,
    result_path: str,
    log_dir: str,
    partition: str = config.partition,
) -> None:
    """Submit a SLURM job that runs training serially and saves results to result_path.

    Args:
        train_config: The training configuration.
        job_name: The SLURM job name.
        result_path: Path where the result xarray Dataset will be saved as NetCDF.
        log_dir: Directory for SLURM stdout/stderr logs.
        partition: The SLURM partition to submit to.
    """
    # All temp files must live on the shared filesystem (not /tmp which is
    # node-local), so that compute nodes can read them.
    os.makedirs(log_dir, exist_ok=True)

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

    # Write the serial training logic as a small Python script
    py_script = f"""\
import os
import yaml
from popsim.ml import TrainConfig
from popsim.ml.launch import launch_train

with open({config_path!r}) as f:
    train_config = TrainConfig(**yaml.safe_load(f))

_, _, _, _, result_dict = launch_train(train_config)
ds = result_dict["test/study_results"]
os.makedirs(os.path.dirname({result_path!r}) or ".", exist_ok=True)
ds.to_netcdf({result_path!r})
os.remove({config_path!r})
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix=f"{job_name}_script_", dir=log_dir
    ) as f:
        f.write(py_script)
        script_path = f.name
    log_path = os.path.join(log_dir, f"{job_name}.log")

    sbatch_script = f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --exclude=node2301,node2101
#SBATCH --output={log_path}
#SBATCH --error={log_path}

# Save results to netcdf only; no need to sync wandb runs online from batch jobs
export WANDB_MODE=offline

{sys.executable} {script_path}
rm -f {script_path}
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
    log_dir: str,
    partition: str = config.partition,
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
    os.makedirs(log_dir, exist_ok=True)

    run_dir = tempfile.mkdtemp(prefix=f"{job_name}_", dir=log_dir)
    config_path = os.path.join(run_dir, "config.yaml")
    script_path = os.path.join(run_dir, "run_agent.py")
    log_path = os.path.join(run_dir, "slurm.log")

    with open(config_path, "w") as f:
        yaml.dump(train_config.model_dump(), f, indent=4)

    py_script = f"""\
import os
import yaml
from popsim.ml import TrainConfig
from popsim.ml.launch import launch_agent

with open({config_path!r}) as f:
    train_config = TrainConfig(**yaml.safe_load(f))

launch_agent(
    train_config,
    {sweep_id!r},
    kwargs_agent={kwargs_agent!r},
)
os.remove({config_path!r})
"""

    with open(script_path, "w") as f:
        f.write(py_script)

    sbatch_script = f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --exclude=node2301,node2101
#SBATCH --output={log_path}
#SBATCH --error={log_path}

{sys.executable} {script_path}
rm -f {script_path}
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
