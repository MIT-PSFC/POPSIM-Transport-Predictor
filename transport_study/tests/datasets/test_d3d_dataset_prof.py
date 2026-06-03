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

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

_DS_PATH = Path(
    "/fusion/projects/disruption_warning/data/popsim/popsim_studies/profopt/hbp_ida/raw_data/199056.nc"
)
if not _DS_PATH.exists():
    pytest.skip("ds_path not found, skipping module", allow_module_level=True)

import matplotlib.pyplot as plt
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import (
    LogSettings,
    RetrievalSettings,
    TimeSetting,
    TimeSettingParams,
)
from disruption_py.settings.output_setting import DatasetOutputSetting
from disruption_py.settings.time_setting import _postprocess
from disruption_py.workflow import get_shots_data
from loguru import logger

from transport_study import PACKAGE_ROOT
from transport_study.datasets.d3d.d3d_dataset import D3DDataWorkflow
from transport_study.datasets.d3d.utils import (
    DEFAULT_SHOTLIST_FILE,
    Uniform1kHzTimeSetting,
)


def test_trajopt_input_mapping():
    """The profile predictor gets trained on C-Mod and TCV but we want to apply it to DIII-D.
    The DIII-D PCS has a unique way to input shapes (gapin, R0, X points).
    We need to have a mapping from these shape parameters to the profile predictor inputs.
    """
    ds = xr.open_dataset(_DS_PATH)
    ds = ds.isel(shot=0)

    a_minor = ds["a_minor"].values
    kappa = ds["kappa"].values
    delta_top = ds["delta_top"].values
    delta_bot = ds["delta_bot"].values

    valid_point_mask = (ds["rxtop"] > 0) & (ds["rxbot"] > 0)
    for var in ["rxtop", "rxbot", "zxtop", "zxbot"]:
        ds[var] = ds[var].where(valid_point_mask, np.nan)

    # RW = (ds["R0"] - ds["gapin"] - ds["a_minor"]).mean()
    RW = 1.046  # Location of the inner wall

    # Need to remake these things from our input parameters
    a_minor_reconst = ds["R0"].values - ds["gapin"].values - RW
    kappa_reconst = (ds["zxtop"].values - ds["zxbot"].values) / (a_minor_reconst * 2)
    delta_top_reconst = (ds["R0"].values - ds["rxtop"].values) / a_minor_reconst
    delta_bot_reconst = (ds["R0"].values - ds["rxbot"].values) / a_minor_reconst

    fig, axes = plt.subplots(4, 1, figsize=(10, 15))
    axes[0].plot(ds["time"], a_minor, label="a_minor")
    axes[0].plot(
        ds["time"], a_minor_reconst, label="a_minor_reconst", linestyle="dashed"
    )
    axes[0].set_title("a_minor")
    axes[0].legend()
    axes[1].plot(ds["time"], kappa, label="kappa")
    axes[1].plot(ds["time"], kappa_reconst, label="kappa_reconst", linestyle="dashed")
    axes[1].set_title("kappa")
    axes[1].legend()
    axes[2].plot(ds["time"], delta_top, label="delta_top")
    axes[2].plot(
        ds["time"], delta_top_reconst, label="delta_top_reconst", linestyle="dashed"
    )
    axes[2].set_title("delta_top")
    axes[2].legend()
    axes[3].plot(ds["time"], delta_bot, label="delta_bot")
    axes[3].plot(
        ds["time"], delta_bot_reconst, label="delta_bot_reconst", linestyle="dashed"
    )
    axes[3].set_title("delta_bot")
    axes[3].legend()
    fig.tight_layout()

    fig_dir = PACKAGE_ROOT / "tests" / "test_outputs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "input_mapping_reconstruction.png")


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
        result_dir = PACKAGE_ROOT / "tests" / "test_outputs" / "d3d_1kHz_discrepancies"
        good_shots_dir = result_dir / "good_shots"
        bad_shots_dir = result_dir / "bad_shots"
        typical_delta = np.median(np.diff(efit_time))
        if typical_delta > 2:
            logger.critical(
                f"EFIT timebase is much slower than 1 kHz (typical delta: {typical_delta:.3f} ms). This may cause issues with interpolation and data quality."
            )
            with open(bad_shots_dir / f"{params.shot_id}.txt", "w") as f:
                f.write(
                    f"EFIT timebase is much slower than 1 kHz (typical delta: {typical_delta:.3f} ms). This may cause issues with interpolation and data quality."
                )
        else:
            with open(good_shots_dir / f"{params.shot_id}.txt", "w") as f:
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
    shotlist_file = (
        PACKAGE_ROOT / "transport_study" / "datasets" / "d3d" / DEFAULT_SHOTLIST_FILE
    )
    shotlist_initial = np.loadtxt(shotlist_file, dtype=int)

    shotlist_ida = []
    for shot in shotlist_initial:
        ida_path = Path(
            f"/fusion/projects/results/ida-results/HBP_database/IDA_{shot}_.cdf"
        )
        if ida_path.exists():
            shotlist_ida.append(shot)

    result_dir = PACKAGE_ROOT / "tests" / "test_outputs" / "d3d_1kHz_discrepancies"
    good_shots_dir = result_dir / "good_shots"
    bad_shots_dir = result_dir / "bad_shots"

    retrieval_settings = RetrievalSettings(
        run_methods=["get_efit_parameters"],
        time_setting=TimeCheckSetting(),
        only_requested_columns=False,
    )

    shutil.rmtree(result_dir, ignore_errors=True)
    good_shots_dir.mkdir(parents=True, exist_ok=True)
    bad_shots_dir.mkdir(parents=True, exist_ok=True)
    for shot in shotlist_ida:
        get_shots_data(
            tokamak=Tokamak.D3D,
            shotlist_setting=shot,
            retrieval_settings=retrieval_settings,
            output_setting=DatasetOutputSetting(path=False),
            log_settings=LogSettings(file_path=None),
            num_processes=1,
        )

    shotlist_good = []
    shotlist_bad = []
    with open(result_dir / "results_good.txt", "w") as f:
        for p in good_shots_dir.iterdir():
            f.write(f"{p.stem}\n")
            shotlist_good.append(int(p.stem))
    with open(result_dir / "results_bad.txt", "w") as f:
        for p in bad_shots_dir.iterdir():
            f.write(f"{p.stem}\n")
            shotlist_bad.append(int(p.stem))

    disruption_efit_dir = Path(
        "/fusion/projects/disruption_warning/data/disruption-efit"
    )
    for shot in shotlist_good:
        if not (disruption_efit_dir / f"{shot}.tgz").exists():
            logger.warning(
                f"Shot {shot} is in the good list but does not have a disruption efit file."
            )

    for shot in shotlist_bad:
        if (disruption_efit_dir / f"{shot}.tgz").exists():
            logger.warning(
                f"Shot {shot} is in the bad list but has a disruption efit file."
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the workflow and run it on a test shot
        workflow = D3DDataWorkflow(
            ds_name="d3d_dataset_prof_test_trajopt_input_mapping",
            shotlist_file=None,
            data_assembly_dir=tmpdir,
        )
    ds_fast = workflow._1kHz_efit(199072)

    ds_slow_path = "/fusion/projects/disruption_warning/data/popsim/popsim_studies/orchestration_test/hbp_trajopt/dataset_full/ds.zarr"
    ds_slow = xr.open_zarr(ds_slow_path)
    ds_slow_shot = ds_slow.where(ds_slow["shot"] == 199072, drop=True).load()

    max_time = 0.2
    ds_slow_shot = ds_slow_shot.where(ds_slow["time"] <= max_time, drop=True)
    ds_fast = ds_fast.where(ds_fast["time"] <= max_time, drop=True)

    # Plot a few signals to see if they look reasonably similar
    fig, axs = plt.subplots(4, 1, figsize=(10, 15))
    axs[0].plot(ds_fast["time"], ds_fast["gapin"], label="gapin (fast)")
    axs[0].plot(
        ds_slow_shot["time"],
        ds_slow_shot["gapin"],
        label="gapin (slow)",
        linestyle="dashed",
    )
    axs[0].set_title("gapin")
    axs[0].legend()
    axs[1].plot(ds_fast["time"], ds_fast["rsurf"], label="R0 (fast)")
    axs[1].plot(
        ds_slow_shot["time"], ds_slow_shot["R0"], label="R0 (slow)", linestyle="dashed"
    )
    axs[1].set_title("R0")
    axs[1].legend()
    axs[2].plot(ds_fast["time"], ds_fast["rxpt1"], label="rxpt1 (fast)")
    axs[2].plot(
        ds_slow_shot["time"],
        ds_slow_shot["rxpt1"],
        label="rxpt1 (slow)",
        linestyle="dashed",
    )
    axs[2].set_title("rxpt1")
    axs[2].legend()
    axs[3].plot(ds_fast["time"], ds_fast["zxpt1"], label="zxpt1 (fast)")
    axs[3].plot(
        ds_slow_shot["time"],
        ds_slow_shot["zxpt1"],
        label="zxpt1 (slow)",
        linestyle="dashed",
    )
    axs[3].set_title("zxpt1")
    axs[3].legend()
    fig.tight_layout()
    fig.savefig(
        PACKAGE_ROOT
        / "tests"
        / "test_outputs"
        / "d3d_1kHz_discrepancies"
        / "input_mapping_comparison.png"
    )
    plt.close(fig)


def correct_betan_source():
    """When betanf is missing, we need a fallback"""

    # Find a shot where these three things all exist and compare them
    # 1. betanf from pedestal
    # 2. betan from fast EFIT
    # 3. betan from slow EFIT
    # 4. programmed betan
    raw_path = Path(
        "/fusion/projects/disruption_warning/data/popsim/popsim_studies/profopt/hbp_ida/raw_data/201927.nc"
    )
    ds_raw = xr.open_dataset(raw_path).isel(shot=0)

    betanf = ds_raw["betan"].values
    beta_n = ds_raw["beta_n"].values
    betan_prog = ds_raw["betan_prog"].values

    betat = ds_raw["betat"].values
    ip_norm = ds_raw["Ip_MA"].values / (ds_raw["a_minor"].values * ds_raw["B0"].values)
    betan_reconstructed = betat / ip_norm

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ds_raw["time"], betanf, label="betanf (pedestal)")
    ax.plot(ds_raw["time"], beta_n, label="betan (EFIT)", linestyle="dashed")
    ax.plot(
        ds_raw["time"], betan_prog, label="betan_prog (programmed)", linestyle="dotted"
    )
    ax.plot(ds_raw["time"], betat, label="betat (EFIT)", linestyle="dashdot")
    ax.plot(
        ds_raw["time"],
        betan_reconstructed,
        label="betan_reconstructed (from betat)",
        linestyle="dotted",
    )
    ax.set_title("betan comparison")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("betan")
    ax.set_ylim(-0.01, 5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        PACKAGE_ROOT
        / "tests"
        / "test_outputs"
        / "d3d_1kHz_discrepancies"
        / "betan_comparison.png"
    )
    plt.close(fig)


if __name__ == "__main__":
    # test_trajopt_input_mapping()
    # find_1kHz_discrepancies()
    correct_betan_source()
