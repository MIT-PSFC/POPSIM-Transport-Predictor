"""Tests for the energy sanity check on the raw-file workflows.

energy_sanity_cull runs in RawFileWorkflow.cull_shot for every raw-file device (TCV, DIII-D),
so shots with broken input power records get excluded.

The check compares the stored energy RISE from the start of the filtered window
to its peak against the input energy (ohmic + heating) delivered over that
same interval, with 5% of leeway.
"""

import numpy as np
import pytest

from transport_study.signals import PREDICTION_STORE_NAME
from transport_study.tests.datasets.synthetic_shot import N_T, SHOT, raw_shot


def _broken_power_shot(**overrides):
    """0.45 MJ of stored energy rise on 15 kJ of input energy."""
    signals = {"energy_mhd": np.linspace(4e4, 4.9e5, N_T)[None, :], "power_ohm": np.full((1, N_T), 1e4)}
    return raw_shot(**(signals | overrides))


def _healthy_shot(**overrides):
    """0.2 MJ of rise on 1.5 MJ of ohmic input."""
    signals = {"energy_mhd": np.linspace(5e4, 2.5e5, N_T)[None, :], "power_ohm": np.full((1, N_T), 1e6)}
    return raw_shot(**(signals | overrides))


@pytest.fixture
def tcv_workflow(tmp_path):
    from transport_study.datasets.tcv.tcv_dataset import TCVDataWorkflow

    shotlist = tmp_path / "shotlist"
    shotlist.write_text(f"{SHOT}\n")
    return TCVDataWorkflow(ds_name="tcv_test", shotlist_file=shotlist, data_assembly_dir=tmp_path)


def test_broken_power_record_culled(tcv_workflow, logged):
    """A shot whose stored energy rise exceeds the integrated input power
    (ohmic + heating) up to that peak is excluded."""
    assert tcv_workflow.energy_sanity_cull(_broken_power_shot()) is True
    assert any("input power record is broken or missing" in msg for msg in logged), logged


def test_healthy_shot_kept(tcv_workflow):
    """A shot with input energy comfortably above the stored energy rise is not culled."""
    assert tcv_workflow.energy_sanity_cull(_healthy_shot()) is False


def test_missing_power_samples_count_as_zero(tcv_workflow):
    """NaN samples in power_ohm or any heating power lower the input energy
    estimate but never crash the check."""
    power_ohm = np.full((1, N_T), 1e6)
    power_ohm[0, ::10] = np.nan
    ds = _healthy_shot(power_ohm=power_ohm, power_nbi=np.full((1, N_T), np.nan))

    # 10% of the ohmic samples gone still leaves ~1.35 MJ against 0.2 MJ of rise
    assert tcv_workflow.energy_sanity_cull(ds) is False


def test_fewer_than_two_valid_slices_skip_check(tcv_workflow):
    """A shot with fewer than 2 slices where both time and energy_mhd are finite
    skips the check instead of integrating a degenerate trace."""
    two_valid = np.full((1, N_T), np.nan)
    two_valid[0, 0] = 4e4
    two_valid[0, -1] = 4.9e5
    assert tcv_workflow.energy_sanity_cull(_broken_power_shot(energy_mhd=two_valid)) is True

    one_valid = two_valid.copy()
    one_valid[0, 0] = np.nan
    assert tcv_workflow.energy_sanity_cull(_broken_power_shot(energy_mhd=one_valid)) is False

    # No valid slice at all: without the guard the peak search reduces over an
    # empty array and raises rather than returning a verdict
    no_valid = np.full((1, N_T), np.nan)
    assert tcv_workflow.energy_sanity_cull(_broken_power_shot(energy_mhd=no_valid)) is False


def test_energy_check_runs_after_device_culling(tcv_workflow):
    """The check runs in cull_shot after device_specific_culling, which passes this shot."""
    ds = tcv_workflow.device_specific_processing(_broken_power_shot())
    filtered = tcv_workflow.filter_ds(ds)

    assert tcv_workflow.device_specific_culling(filtered) is False, "the profile cull should pass this shot"
    assert tcv_workflow.energy_sanity_cull(filtered) is True

    tcv_workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
    _broken_power_shot().to_netcdf(tcv_workflow.raw_data_dir / f"{SHOT}.nc")
    assert tcv_workflow.process_fn(SHOT, tcv_workflow.STORE_VARIABLES[PREDICTION_STORE_NAME]) is None
