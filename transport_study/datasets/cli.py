import os
import shutil
from pathlib import Path

import fire
from loguru import logger

from transport_study.datasets.cmod.cmod_dataset import CModDataWorkflow
from transport_study.datasets.d3d.d3d_dataset import D3DDataWorkflow
from transport_study.datasets.mast.mast_dataset import MASTDataWorkflow
from transport_study.datasets.tcv.tcv_dataset import TCVDataWorkflow
from transport_study.datasets.workflow import DataWorkflow, RawFileWorkflow


class DatasetCLI:
    """Command line interface for dataset workflows

    cmod and mast build their stores from external published datasets,
    so they only need to process.
    d3d and tcv first pull raw per-shot files from source (--mode raw), then process them (--mode process).
    """

    def cmod(
        self,
        data_assembly_dir: Path | str,
        published_store_path: Path | str,
        ds_name: str = "cmod",
        max_num_shots: int | None = None,
        clean: bool = False,
    ):
        workflow = CModDataWorkflow(
            ds_name=ds_name,
            data_assembly_dir=data_assembly_dir,
            published_store_path=published_store_path,
            max_num_shots=max_num_shots,
        )
        self._process(workflow, clean)

    def mast(
        self,
        data_assembly_dir: Path | str,
        published_store_path: Path | str,
        ds_name: str = "mast",
        max_num_shots: int | None = None,
        clean: bool = False,
    ):
        workflow = MASTDataWorkflow(
            ds_name=ds_name,
            data_assembly_dir=data_assembly_dir,
            published_store_path=published_store_path,
            max_num_shots=max_num_shots,
        )
        self._process(workflow, clean)

    def d3d(
        self,
        data_assembly_dir: Path | str,
        ds_name: str = "d3d",
        shotlist_file: Path | str | None = None,
        max_num_shots: int | None = None,
        mode: str = "raw",
        clean: bool = False,
    ):
        workflow = D3DDataWorkflow(
            ds_name=ds_name,
            shotlist_file=shotlist_file,
            data_assembly_dir=data_assembly_dir,
            max_num_shots=max_num_shots,
        )
        self._execute(workflow, mode, clean)

    def tcv(
        self,
        data_assembly_dir: Path | str,
        ds_name: str = "tcv",
        shotlist_file: Path | str | None = None,
        source_dataset_path: Path | str | None = None,
        max_num_shots: int | None = None,
        mode: str = "raw",
        clean: bool = False,
    ):
        workflow = TCVDataWorkflow(
            ds_name=ds_name,
            shotlist_file=shotlist_file,
            data_assembly_dir=data_assembly_dir,
            source_dataset_path=source_dataset_path,
            max_num_shots=max_num_shots,
        )
        self._execute(workflow, mode, clean)

    def _execute(self, workflow: RawFileWorkflow, mode: str, clean: bool):
        if mode == "raw":
            if clean and workflow.raw_data_dir.exists():
                shutil.rmtree(workflow.raw_data_dir)
            workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
            log_path = workflow.raw_data_dir / f"raw_data_{os.getpid()}.log"
            logger.add(log_path)
            workflow.make_raw_data_files()
        elif mode == "process":
            self._process(workflow, clean)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'raw' or 'process'.")

    def _process(self, workflow: DataWorkflow, clean: bool):
        if clean and workflow.final_ds_dir.exists():
            shutil.rmtree(workflow.final_ds_dir)
        workflow.final_ds_dir.mkdir(parents=True, exist_ok=True)
        log_path = workflow.final_ds_dir / f"processed_data_{os.getpid()}.log"
        logger.add(log_path)
        workflow.run_processed_data_workflow()


if __name__ == "__main__":
    fire.Fire(DatasetCLI)
