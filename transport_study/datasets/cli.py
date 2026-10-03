import os
import shutil
from pathlib import Path

import fire
from loguru import logger

from transport_study.datasets.cmod.cmod_dataset import CModDataWorkflow
from transport_study.datasets.d3d.d3d_dataset import D3DDataWorkflow
from transport_study.datasets.mast.mast_dataset import MASTDataWorkflow
from transport_study.datasets.tcv.tcv_dataset import TCVDataWorkflow
from transport_study.datasets.workflow import StoreWorkflow


class DatasetCLI:
    """Build one device's POPSIM store from its transport-validation-datasets store.

        python -m transport_study.datasets.cli <device> <data_assembly_dir> <source_store_path>

    transport-validation-datasets builds every device:
    C-Mod and MAST up to a published store (<ds>_published.zarr),
    TCV and DIII-D up to an internal one (<ds>_internal.zarr), since their data has no release permission.
    """

    def cmod(
        self,
        data_assembly_dir: Path | str,
        source_store_path: Path | str,
        ds_name: str = "cmod",
        max_num_shots: int | None = None,
        clean: bool = False,
    ):
        _build(CModDataWorkflow, data_assembly_dir, source_store_path, ds_name, max_num_shots, clean)

    def mast(
        self,
        data_assembly_dir: Path | str,
        source_store_path: Path | str,
        ds_name: str = "mast",
        max_num_shots: int | None = None,
        clean: bool = False,
    ):
        _build(MASTDataWorkflow, data_assembly_dir, source_store_path, ds_name, max_num_shots, clean)

    def tcv(
        self,
        data_assembly_dir: Path | str,
        source_store_path: Path | str,
        ds_name: str = "tcv",
        max_num_shots: int | None = None,
        clean: bool = False,
    ):
        _build(TCVDataWorkflow, data_assembly_dir, source_store_path, ds_name, max_num_shots, clean)

    def d3d(
        self,
        data_assembly_dir: Path | str,
        source_store_path: Path | str,
        ds_name: str = "d3d",
        max_num_shots: int | None = None,
        clean: bool = False,
    ):
        _build(D3DDataWorkflow, data_assembly_dir, source_store_path, ds_name, max_num_shots, clean)


def _build(
    workflow_cls: type[StoreWorkflow],
    data_assembly_dir: Path | str,
    source_store_path: Path | str,
    ds_name: str,
    max_num_shots: int | None,
    clean: bool,
):
    """Build one device's store, logging next to it. clean deletes an earlier build first."""
    workflow = workflow_cls(
        ds_name=ds_name,
        data_assembly_dir=data_assembly_dir,
        source_store_path=source_store_path,
        max_num_shots=max_num_shots,
    )
    if clean and workflow.final_ds_dir.exists():
        shutil.rmtree(workflow.final_ds_dir)
    workflow.final_ds_dir.mkdir(parents=True, exist_ok=True)
    log_path = workflow.final_ds_dir / f"processed_data_{os.getpid()}.log"
    logger.add(log_path)
    workflow.run_processed_data_workflow()


if __name__ == "__main__":
    fire.Fire(DatasetCLI)
