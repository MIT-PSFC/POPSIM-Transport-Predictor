"""Tests for the quality culls of the raw-file workflows.

radiated_fraction_cull catches a dead bolometer,
and density_ratio_cull catches profiles that disagree with the interferometer.
Both run in RawFileWorkflow.cull_shot on every device, with per-device thresholds.
"""

import numpy as np
import pytest

from transport_study.signals import PREDICTION_STORE_NAME
from transport_study.tests.datasets.synthetic_shot import N_T, SHOT, raw_shot


@pytest.fixture
def tcv_workflow(tmp_path):
    from transport_study.datasets.tcv.tcv_dataset import TCVDataWorkflow

    shotlist = tmp_path / "shotlist"
    shotlist.write_text(f"{SHOT}\n")
    workflow = TCVDataWorkflow(ds_name="tcv_test", shotlist_file=shotlist, data_assembly_dir=tmp_path)
    workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
    return workflow


def _processed(workflow, ds):
    """Write ds as the shot's raw file and run it through the prediction store's processing."""
    ds.to_netcdf(workflow.raw_data_dir / f"{SHOT}.nc")
    return workflow.process_fn(SHOT, workflow.STORE_VARIABLES[PREDICTION_STORE_NAME])


def test_dead_bolometer_culled_and_live_one_kept(tcv_workflow, logged):
    """1 kW radiated on 300 kW of ohmic input is a dead bolometer, 100 kW is a live one."""
    assert tcv_workflow.radiated_fraction_cull(raw_shot()) is False

    shot_dead_bolometer = raw_shot(power_radiated=np.full((1, N_T), 1e3))
    assert tcv_workflow.radiated_fraction_cull(shot_dead_bolometer) is True
    assert any("bolometer record is broken or missing" in msg for msg in logged), logged


def test_profiles_disagreeing_with_interferometer_culled(tcv_workflow, logged):
    """The synthetic profiles sit at ~1.05 of the line average and are kept,
    against a line average 2.5x higher they read ~0.42 of it and the shot goes."""
    assert _processed(tcv_workflow, raw_shot()) is not None

    shot_low_profiles = raw_shot(n_e_line_average=np.full((1, N_T), 1e20))
    assert _processed(tcv_workflow, shot_low_profiles) is None
    assert any("the profiles disagree with the interferometer" in msg for msg in logged), logged
