"""
Comparing different signals from DIII-D to ensure the dataset for profile prediction is being generated correctly.

Profile predictor inputs:
Ip
B0
beta
ne20_ped
R0
a_minor
kappa
delta_top
delta_bot
"""

import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import tempfile
from transport_study import PACKAGE_ROOT
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings.output_setting import DatasetOutputSetting
from disruption_py.settings import (
    LogSettings,
    RetrievalSettings,
    TimeSetting,
    TimeSettingParams,
)
from transport_study.datasets.d3d.d3d_dataset import (
    D3DDataWorkflow,
    DEFAULT_SHOTLIST_FILE,
    Uniform1kHzTimeSetting,
)

from disruption_py.workflow import get_shots_data
from disruption_py.settings.time_setting import _postprocess

from loguru import logger


def test_trajopt_input_mapping():
    """The profile predictor gets trained on C-Mod and TCV but we want to apply it to DIII-D.
    The DIII-D PCS has a unique way to input shapes (gapin, R0, X points).
    We need to have a mapping from these shape parameters to the profile predictor inputs.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the workflow and run it on a test shot
        workflow = D3DDataWorkflow(
            ds_name="d3d_dataset_prof_test_trajopt_input_mapping",
            shotlist_file=None,
            data_assembly_dir=tmpdir,
        )
        workflow._1kHz_efit(201927)


class TimeCheckSetting(TimeSetting):
    def _get_times(self, params: TimeSettingParams) -> np.ndarray:
        """
        Parameters
        ----------
        params : TimeSettingParams
            Parameters needed to retrieve the timebase.

        Returns
        -------
        np.ndarray
            Array of times in the timebase.
        """
        (efit_time,) = params.mds_conn.get_dims(
            r"\efit_aeqdsk:ali", tree_name="_efit_tree"
        )
        # If timebase is much slower than 1 kHz, log a warning
        result_dir = os.path.join(
            PACKAGE_ROOT, "tests", "test_outputs", "d3d_1kHz_discrepancies"
        )
        good_shots_dir = os.path.join(result_dir, "good_shots")
        bad_shots_dir = os.path.join(result_dir, "bad_shots")
        typical_delta = np.median(np.diff(efit_time))
        if typical_delta > 2:
            logger.critical(
                f"EFIT timebase is much slower than 1 kHz (typical delta: {typical_delta:.3f} ms). This may cause issues with interpolation and data quality."
            )
            with open(os.path.join(bad_shots_dir, f"{params.shot_id}.txt"), "w") as f:
                f.write(
                    f"EFIT timebase is much slower than 1 kHz (typical delta: {typical_delta:.3f} ms). This may cause issues with interpolation and data quality."
                )
        else:
            with open(os.path.join(good_shots_dir, f"{params.shot_id}.txt"), "w") as f:
                f.write(
                    f"EFIT timebase is good (typical delta: {typical_delta:.3f} ms)."
                )

        max_time = np.max(efit_time)
        if params.tokamak == Tokamak.CMOD:
            times = np.round(np.arange(0, max_time + 1e-3, 1e-3), 3)
            efit_time_unit = "s"
        if params.tokamak == Tokamak.D3D:
            times = np.round(np.arange(0, max_time + 1, 1), 0)
            efit_time_unit = "ms"
        return _postprocess(times=times, units=efit_time_unit)


def find_1kHz_discrepancies():
    shotlist_file = os.path.join(
        PACKAGE_ROOT, "transport_study", "datasets", "d3d", DEFAULT_SHOTLIST_FILE
    )
    shotlist_initial = np.loadtxt(shotlist_file, dtype=int)

    shotlist_ida = []
    for shot in shotlist_initial:
        ida_path = f"/fusion/projects/results/ida-results/HBP_database/IDA_{shot}_.cdf"
        if os.path.exists(ida_path):
            shotlist_ida.append(shot)

    result_dir = os.path.join(
        PACKAGE_ROOT, "tests", "test_outputs", "d3d_1kHz_discrepancies"
    )
    good_shots_dir = os.path.join(result_dir, "good_shots")
    bad_shots_dir = os.path.join(result_dir, "bad_shots")

    retrieval_settings = RetrievalSettings(
        run_methods=["get_efit_parameters"],
        time_setting=TimeCheckSetting(),
        only_requested_columns=False,
    )

    # shutil.rmtree(result_dir, ignore_errors=True)
    # os.makedirs(good_shots_dir, exist_ok=True)
    # os.makedirs(bad_shots_dir, exist_ok=True)
    # for shot in shotlist_ida:
    #     efit_result = get_shots_data(
    #         tokamak=Tokamak.D3D,
    #         shotlist_setting=shot,
    #         retrieval_settings=retrieval_settings,
    #         output_setting=DatasetOutputSetting(path=False),
    #         log_settings=LogSettings(file_path=None),
    #         num_processes=1,
    #     )

    shotlist_good = []
    shotlist_bad = []
    with open(os.path.join(result_dir, "results_good.txt"), "w") as f:
        for filename in os.listdir(good_shots_dir):
            f.write(f"{filename[:-4]}\n")
            shotlist_good.append(int(filename[:-4]))
    with open(os.path.join(result_dir, "results_bad.txt"), "w") as f:
        for filename in os.listdir(bad_shots_dir):
            f.write(f"{filename[:-4]}\n")
            shotlist_bad.append(int(filename[:-4]))

    disruption_efit_dir = "/fusion/projects/disruption_warning/data/disruption-efit"
    for shot in shotlist_good:
        if not os.path.exists(os.path.join(disruption_efit_dir, f"{shot}.tgz")):
            logger.warning(
                f"Shot {shot} is in the good list but does not have a disruption efit file."
            )

    for shot in shotlist_bad:
        if os.path.exists(os.path.join(disruption_efit_dir, f"{shot}.tgz")):
            logger.warning(
                f"Shot {shot} is in the bad list but has a disruption efit file."
            )


if __name__ == "__main__":
    # test_trajopt_input_mapping()
    find_1kHz_discrepancies()
