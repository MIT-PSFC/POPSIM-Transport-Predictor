import os
import shutil
from pathlib import Path

import fire
from loguru import logger

from transport_study.datasets.workflow import StoreWorkflow

# The devices transport-validation-datasets builds, the default ds_name of each
DEVICES = ("cmod", "mast", "tcv", "d3d")


class DatasetCLI:
    """Build one device's study store from its transport-validation-datasets store.

        python -m transport_study.datasets.cli build --device cmod --store <cmod_published.zarr> --data_assembly_dir <dir>

    transport-validation-datasets builds every device:
    C-Mod and MAST up to a published store (<ds>_published.zarr),
    TCV and DIII-D up to an internal one (<ds>_internal.zarr), since their data has no release permission.
    """

    def build(
        self,
        device: str,
        store: Path | str,
        data_assembly_dir: Path | str,
        ds_name: str | None = None,
        max_num_shots: int | None = None,
        clean: bool = False,
    ):
        """Build one device's store, logging next to it.

        device: one of DEVICES, also the default ds_name
        store: the TVD store to read
        data_assembly_dir: the store is written to <data_assembly_dir>/<ds_name>/dataset_full/ds.zarr
        max_num_shots: build only the first shots of the store, into dataset_<max_num_shots> instead
        clean: delete an earlier build first
        """
        if device not in DEVICES:
            raise ValueError(f"Unknown device {device!r}, transport-validation-datasets builds {DEVICES}")
        workflow = StoreWorkflow(
            ds_name=device if ds_name is None else ds_name,
            data_assembly_dir=data_assembly_dir,
            source_store_path=store,
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
