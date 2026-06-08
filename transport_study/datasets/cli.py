import os
import shutil
from pathlib import Path

import fire
from loguru import logger

from transport_study.datasets.cmod.cmod_dataset import CModDataWorkflow
from transport_study.datasets.d3d.d3d_dataset import D3DDataWorkflow
from transport_study.datasets.mast.mast_dataset import MASTDataWorkflow
from transport_study.datasets.tcv.tcv_dataset import TCVDataWorkflow


class DatasetCLI:
    """Command line interface for dataset workflows"""

    def cmod(
        self,
        ds_name: str = "cmod",
        shotlist_file: Path | str | None = None,
        data_assembly_dir: Path | str | None = None,
        max_num_shots: int | None = None,
        mode: str | None = "raw",
        clean: bool | None = False,
        skip_profiles: bool | None = False,
    ):
        data_assembly_dir = Path(data_assembly_dir)
        workflow = CModDataWorkflow(
            ds_name=ds_name,
            shotlist_file=shotlist_file,
            data_assembly_dir=data_assembly_dir,
            max_num_shots=max_num_shots,
            skip_profiles=skip_profiles,
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
        use_ida: bool | None = True,
    ):
        data_assembly_dir = Path(data_assembly_dir)
        workflow = D3DDataWorkflow(
            ds_name=ds_name,
            shotlist_file=shotlist_file,
            data_assembly_dir=data_assembly_dir,
            max_num_shots=max_num_shots,
            use_ida=use_ida,
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
    ):
        data_assembly_dir = Path(data_assembly_dir)
        workflow = MASTDataWorkflow(
            ds_name=ds_name,
            shotlist_file=shotlist_file,
            data_assembly_dir=data_assembly_dir,
            max_num_shots=max_num_shots,
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
            if clean and Path(workflow.raw_data_dir).exists():
                shutil.rmtree(workflow.raw_data_dir)
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
