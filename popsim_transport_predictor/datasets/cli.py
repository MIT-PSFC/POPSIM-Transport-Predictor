import os
import shutil

import fire
from loguru import logger

from popsim_transport_predictor.config import DATA_DUMP_DIR
from popsim_transport_predictor.datasets.cmod.cmod_dataset import CModDataWorkflow
from popsim_transport_predictor.datasets.d3d.d3d_dataset import (
    DEFAULT_SHOTLIST_FILE,
    D3DDataWorkflow,
)
from popsim_transport_predictor.datasets.plotting import ds_time_plot


class DatasetCLI:
    """Command line interface for dataset workflows"""

    def cmod(
        self,
        ds_name: str = "cmod",
        raw_data_dir: str = DATA_DUMP_DIR,
        final_ds_dir: str = DATA_DUMP_DIR,
        max_num_shots: int | None = None,
        mode: str | None = "raw",
        clean: bool | None = False,
    ):
        workflow = CModDataWorkflow(
            ds_name=ds_name,
            raw_data_dir=raw_data_dir,
            final_ds_dir=final_ds_dir,
            max_num_shots=max_num_shots,
        )

        self._execute(workflow, mode, clean)

    def d3d(
        self,
        ds_name: str = "d3d",
        shotlist_file: str = DEFAULT_SHOTLIST_FILE,
        raw_data_dir: str = DATA_DUMP_DIR,
        final_ds_dir: str = DATA_DUMP_DIR,
        max_num_shots: int | None = None,
        mode: str | None = "raw",
        clean: bool | None = False,
        use_ida: bool | None = True,
    ):
        workflow = D3DDataWorkflow(
            ds_name=ds_name,
            shotlist_file=shotlist_file,
            raw_data_dir=raw_data_dir,
            final_ds_dir=final_ds_dir,
            max_num_shots=max_num_shots,
            use_ida=use_ida,
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
                shutil.rmtree(workflow.raw_data_dir)
            os.makedirs(workflow.raw_data_dir, exist_ok=True)
            log_path = os.path.join(
                workflow.raw_data_dir, f"raw_data_{os.getpid()}.log"
            )
            logger.add(log_path)
            workflow.make_raw_data_files()
        elif mode == "process":
            if clean:
                shutil.rmtree(workflow.final_ds_dir)
            os.makedirs(workflow.final_ds_dir, exist_ok=True)
            log_path = os.path.join(
                workflow.final_ds_dir, f"processed_data_{os.getpid()}.log"
            )
            logger.add(log_path)
            workflow.run_processed_data_workflow()
            ds_time_plot(
                os.path.join(workflow.final_ds_dir, f"{workflow.ds_name}.zarr"),
                os.path.join(workflow.final_ds_dir, "time_traces"),
                title=f"{workflow.ds_name.upper()} Dataset Time Traces",
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'raw' or 'process'.")


if __name__ == "__main__":
    fire.Fire(DatasetCLI)
