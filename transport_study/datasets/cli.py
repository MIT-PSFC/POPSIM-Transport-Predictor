import os
import shutil
from datetime import datetime
from pathlib import Path

import fire
from loguru import logger

from transport_study.datasets.cmod.cmod_dataset import CModDataWorkflow
from transport_study.datasets.d3d.d3d_dataset import D3DDataWorkflow
from transport_study.datasets.mast.mast_dataset import MASTDataWorkflow
from transport_study.datasets.tcv.tcv_dataset import TCVDataWorkflow


def _build_cluster_config(
    cluster_profile: str | None,
    cluster_partitions: str | None,
    cluster_remote_workdir: str | None,
    cluster_venv: str | None,
    cluster_max_jobs: int,
    cluster_shots_per_batch: int,
    cluster_cpus_per_job: int,
    cluster_mem: str | None,
    cluster_max_retries: int,
    cluster_pending_timeout_s: float,
):
    """Build a ClusterFitConfig from CLI options, or None when no profile is given."""
    if cluster_profile is None:
        return None
    if cluster_partitions is None or cluster_remote_workdir is None or cluster_venv is None:
        raise ValueError("--cluster_partitions, --cluster_remote_workdir, and --cluster_venv are required with --cluster_profile")

    from transport_study.datasets.gp_fitting.dispatcher import ClusterFitConfig

    return ClusterFitConfig(
        profile=cluster_profile,
        partitions=cluster_partitions,
        remote_workdir=cluster_remote_workdir,
        venv_path=cluster_venv,
        max_concurrent_jobs=cluster_max_jobs,
        shots_per_batch=cluster_shots_per_batch,
        cpus_per_job=cluster_cpus_per_job,
        memory_per_node=cluster_mem,
        max_retries=cluster_max_retries,
        pending_timeout_s=cluster_pending_timeout_s,
    )


class DatasetCLI:
    """Command line interface for dataset workflows

    The cmod and mast workflows can dispatch GP profile fitting to a SLURM
    cluster. Set up once per cluster:
        srunx ssh profile add <profile> --ssh-host <host>
        bash transport_study/datasets/gp_fitting/bootstrap_remote.sh <host> <scratch-dir>
    Then pass --cluster_profile <profile> (or "local" when already running on
    the cluster), --cluster_partitions <spec>,
    --cluster_remote_workdir <scratch-dir>, and --cluster_venv <scratch-dir>/.venv.

    --cluster_partitions is an ordered preference list of comma-separated
    name@time_limit or name@time_limit@constraint entries, e.g.
        --cluster_partitions "sched_psfc_mit_r8@8:00:00,mit_preemptable@8:00:00@rocky8"
    Killed jobs are retried up to --cluster_max_retries times, and both
    retries and jobs stuck PENDING past --cluster_pending_timeout_s move to
    the next partition in the list (wrapping around).
    """

    def cmod(
        self,
        ds_name: str = "cmod",
        shotlist_file: Path | str | None = None,
        data_assembly_dir: Path | str | None = None,
        max_num_shots: int | None = None,
        mode: str | None = "raw",
        clean: bool | None = False,
        fit_workers: int = 1,
        cluster_profile: str | None = None,
        cluster_partitions: str | None = None,
        cluster_remote_workdir: str | None = None,
        cluster_venv: str | None = None,
        cluster_max_jobs: int = 8,
        cluster_shots_per_batch: int = 10,
        cluster_cpus_per_job: int = 32,
        cluster_mem: str | None = None,
        cluster_max_retries: int = 2,
        cluster_pending_timeout_s: float = 1800.0,
    ):
        data_assembly_dir = Path(data_assembly_dir)
        cluster_config = _build_cluster_config(
            cluster_profile,
            cluster_partitions,
            cluster_remote_workdir,
            cluster_venv,
            cluster_max_jobs,
            cluster_shots_per_batch,
            cluster_cpus_per_job,
            cluster_mem,
            cluster_max_retries,
            cluster_pending_timeout_s,
        )
        workflow = CModDataWorkflow(
            ds_name=ds_name,
            shotlist_file=shotlist_file,
            data_assembly_dir=data_assembly_dir,
            max_num_shots=max_num_shots,
            cluster_config=cluster_config,
            fit_workers=fit_workers,
        )

        self._execute(workflow, mode, clean)

    def d3d(
        self,
        ds_name: str = "d3d",
        shotlist_file: Path | str | None = None,
        data_assembly_dir: Path | str | None = None,
        max_num_shots: int | None = None,
        mode: str | None = "raw",
        clean: bool | None = False,
    ):
        data_assembly_dir = Path(data_assembly_dir)
        workflow = D3DDataWorkflow(
            ds_name=ds_name,
            shotlist_file=shotlist_file,
            data_assembly_dir=data_assembly_dir,
            max_num_shots=max_num_shots,
        )

        self._execute(workflow, mode, clean)

    def mast(
        self,
        ds_name: str = "mast",
        shotlist_file: Path | str | None = None,
        data_assembly_dir: Path | str | None = None,
        max_num_shots: int | None = None,
        mode: str | None = "raw",
        clean: bool | None = False,
        fit_workers: int = 1,
        cluster_profile: str | None = None,
        cluster_partitions: str | None = None,
        cluster_remote_workdir: str | None = None,
        cluster_venv: str | None = None,
        cluster_max_jobs: int = 8,
        cluster_shots_per_batch: int = 10,
        cluster_cpus_per_job: int = 32,
        cluster_mem: str | None = None,
        cluster_max_retries: int = 2,
        cluster_pending_timeout_s: float = 1800.0,
    ):
        data_assembly_dir = Path(data_assembly_dir)
        cluster_config = _build_cluster_config(
            cluster_profile,
            cluster_partitions,
            cluster_remote_workdir,
            cluster_venv,
            cluster_max_jobs,
            cluster_shots_per_batch,
            cluster_cpus_per_job,
            cluster_mem,
            cluster_max_retries,
            cluster_pending_timeout_s,
        )
        workflow = MASTDataWorkflow(
            ds_name=ds_name,
            shotlist_file=shotlist_file,
            data_assembly_dir=data_assembly_dir,
            max_num_shots=max_num_shots,
            cluster_config=cluster_config,
            fit_workers=fit_workers,
        )

        self._execute(workflow, mode, clean)

    def tcv(
        self,
        ds_name: str = "tcv",
        shotlist_file: Path | str | None = None,
        data_assembly_dir: Path | str | None = None,
        source_dataset_path: Path | str | None = None,
        max_num_shots: int | None = None,
        mode: str | None = "raw",
        clean: bool | None = False,
    ):
        data_assembly_dir = Path(data_assembly_dir)
        workflow = TCVDataWorkflow(
            ds_name=ds_name,
            shotlist_file=shotlist_file,
            data_assembly_dir=data_assembly_dir,
            source_dataset_path=source_dataset_path,
            max_num_shots=max_num_shots,
        )

        self._execute(workflow, mode, clean)

    def _execute(
        self,
        workflow,
        mode: str,
        clean: bool,
    ):
        if mode == "raw":
            if clean:
                workflow.clean_cluster_state()
                if Path(workflow.raw_data_dir).exists():
                    shutil.rmtree(workflow.raw_data_dir)
                if Path(workflow.fit_staging_dir).exists():
                    shutil.rmtree(workflow.fit_staging_dir)
                ts_fit_plots_dir = workflow.data_assembly_dir / workflow.ds_name / "ts_fit_plots"
                if ts_fit_plots_dir.exists():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ts_fit_plots_dir.rename(ts_fit_plots_dir.with_name(f"ts_fit_plots_{timestamp}"))
            Path(workflow.raw_data_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(workflow.raw_data_dir) / f"raw_data_{os.getpid()}.log"
            logger.add(log_path)
            workflow.make_raw_data_files()
        elif mode == "process":
            if clean and Path(workflow.final_ds_dir).exists():
                shutil.rmtree(workflow.final_ds_dir)
            Path(workflow.final_ds_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(workflow.final_ds_dir) / f"processed_data_{os.getpid()}.log"
            logger.add(log_path)
            workflow.run_processed_data_workflow()
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'raw' or 'process'.")


if __name__ == "__main__":
    fire.Fire(DatasetCLI)
