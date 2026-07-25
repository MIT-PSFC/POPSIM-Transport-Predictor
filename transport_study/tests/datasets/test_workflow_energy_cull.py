"""Tests for the device-independent energy sanity check.

energy_sanity_cull lives on the DataWorkflow base and runs for every device in
process_fn's common culling, so C-Mod, DIII-D and TCV shots with broken input
power records get excluded too - not just the MAST shots it was written for.

The check compares the stored energy RISE from the start of the filtered window
to its peak against the input energy (ohmic + auxiliary) delivered over that
same interval, with 5% of leeway.
"""

import numpy as np
import pytest

from transport_study.tests.datasets.synthetic_shot import N_T, RHO, SHOT, raw_shot

MAST_SHOT = 30284


def _broken_power_shot(**overrides):
    """1.45 MJ of stored energy rise on 0.15 MJ of input energy."""
    signals = {"Wtot_MJ": np.linspace(0.05, 1.5, N_T)[None, :], "P_oh_MW": np.full((1, N_T), 0.1)}
    return raw_shot(**(signals | overrides))


def _healthy_shot(**overrides):
    """0.25 MJ of rise on 1.5 MJ of ohmic input."""
    signals = {"Wtot_MJ": np.linspace(0.05, 0.30, N_T)[None, :], "P_oh_MW": np.full((1, N_T), 1.0)}
    return raw_shot(**(signals | overrides))


@pytest.fixture
def cmod_workflow(tmp_path):
    from transport_study.datasets.cmod.cmod_dataset import CModDataWorkflow

    shotlist = tmp_path / "shotlist"
    shotlist.write_text(f"{SHOT}\n")
    return CModDataWorkflow(ds_name="cmod_test", shotlist_file=shotlist, data_assembly_dir=tmp_path, gp_fit_rho=RHO)


@pytest.fixture
def mast_workflow(tmp_path):
    from transport_study.datasets.mast.mast_dataset import MASTDataWorkflow

    shotlist = tmp_path / "shotlist"
    shotlist.write_text(f"{MAST_SHOT}\n")
    return MASTDataWorkflow(ds_name="mast_test", shotlist_file=shotlist, data_assembly_dir=tmp_path)


def test_broken_power_record_culled(cmod_workflow, logged):
    """A shot whose peak stored energy exceeds the integrated input power
    (ohmic + auxiliary) up to that peak is excluded."""
    assert cmod_workflow.energy_sanity_cull(_broken_power_shot()) is True
    assert any("input power record is broken or missing" in msg for msg in logged), logged


def test_healthy_shot_kept(cmod_workflow):
    """A shot with input energy comfortably above the stored energy rise is not culled."""
    assert cmod_workflow.energy_sanity_cull(_healthy_shot()) is False


def test_missing_power_samples_count_as_zero(cmod_workflow):
    """NaN samples in P_oh_MW or any AUX_POWER_SIGNALS lower the input energy
    estimate but never crash the check."""
    p_oh = np.full((1, N_T), 1.0)
    p_oh[0, ::10] = np.nan
    ds = _healthy_shot(P_oh_MW=p_oh, P_NBI_MW=np.full((1, N_T), np.nan))

    # 10% of the ohmic samples gone still leaves ~1.35 MJ against 0.25 MJ of rise
    assert cmod_workflow.energy_sanity_cull(ds) is False


@pytest.mark.parametrize("missing", ["Wtot_MJ", "P_oh_MW"])
def test_missing_signals_skip_check(cmod_workflow, missing):
    """A shot without Wtot_MJ or without P_oh_MW skips the check, even though
    it would be culled if the check ran."""
    ds = _broken_power_shot()
    assert cmod_workflow.energy_sanity_cull(ds) is True  # the same shot, un-dropped

    assert cmod_workflow.energy_sanity_cull(ds.drop_vars(missing)) is False


def test_fewer_than_two_valid_slices_skip_check(cmod_workflow):
    """A shot with fewer than 2 slices where both time and Wtot_MJ are finite
    skips the check instead of integrating a degenerate trace."""
    two_valid = np.full((1, N_T), np.nan)
    two_valid[0, 0] = 0.05
    two_valid[0, -1] = 1.5
    assert cmod_workflow.energy_sanity_cull(_broken_power_shot(Wtot_MJ=two_valid)) is True

    one_valid = two_valid.copy()
    one_valid[0, 0] = np.nan
    assert cmod_workflow.energy_sanity_cull(_broken_power_shot(Wtot_MJ=one_valid)) is False

    # No valid slice at all: without the guard the peak search reduces over an
    # empty array and raises rather than returning a verdict
    no_valid = np.full((1, N_T), np.nan)
    assert cmod_workflow.energy_sanity_cull(_broken_power_shot(Wtot_MJ=no_valid)) is False


def test_energy_check_runs_despite_device_culling_override(mast_workflow):
    """The check runs in process_fn common culling, so devices overriding
    device_specific_culling (MAST, TCV) still get it."""
    ds = mast_workflow.device_specific_processing(_broken_power_shot(shot=MAST_SHOT))
    filtered = mast_workflow.filter_ds(ds)

    assert mast_workflow.device_specific_culling(filtered) is False, "MAST's own cull should pass this shot"
    assert mast_workflow.energy_sanity_cull(filtered) is True

    mast_workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
    _broken_power_shot(shot=MAST_SHOT).to_netcdf(mast_workflow.raw_data_dir / f"{MAST_SHOT}.nc")
    assert mast_workflow.process_fn(MAST_SHOT) is None
