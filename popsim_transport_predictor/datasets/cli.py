import os
import shutil

import fire
from loguru import logger

from popsim_transport_predictor.datasets.d3d.d3d_dataset import (
    DEFAULT_SHOTLIST_FILE,
    D3DDataWorkflow,
)


class DatasetCLI:
    """Command line interface for dataset workflows"""

    def d3d(
        self,
        ds_name: str = "d3d",
        shotlist_file: str = DEFAULT_SHOTLIST_FILE,
        raw_data_dir: str = "/fusion/projects/disruption_warning/data/popsim/tpt_d3d_raw",
        final_ds_dir: str = "/fusion/projects/disruption_warning/data/popsim/tpt_d3d_final",
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
        if mode == "raw":
            if clean:
                shutil.rmtree(raw_data_dir)
            os.makedirs(raw_data_dir, exist_ok=True)
            log_path = os.path.join(raw_data_dir, "raw_data.log")
            logger.add(log_path)
            workflow.make_raw_data_files()
        elif mode == "process":
            if clean:
                shutil.rmtree(final_ds_dir)
            os.makedirs(final_ds_dir, exist_ok=True)
            log_path = os.path.join(final_ds_dir, "processed_data.log")
            logger.add(log_path)
            workflow.run_processed_data_workflow()
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'raw' or 'process'.")


if __name__ == "__main__":
    fire.Fire(DatasetCLI)
