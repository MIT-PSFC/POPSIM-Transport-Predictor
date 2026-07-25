"""Test stubs for the device-independent energy sanity check.

energy_sanity_cull moved from the MAST workflow to the DataWorkflow base and
now runs for every device in process_fn's common culling, so C-Mod, DIII-D and
TCV shots with broken input power records get excluded too.
"""


def test_broken_power_record_culled():
    """A shot whose peak stored energy exceeds the integrated input power
    (ohmic + auxiliary) up to that peak is excluded with the 'input power
    record is broken or missing' message."""


def test_healthy_shot_kept():
    """A shot with input energy comfortably above the peak stored energy is
    not culled."""


def test_missing_power_samples_count_as_zero():
    """NaN samples in P_oh_MW or any AUX_POWER_SIGNALS lower the input energy
    estimate but never crash the check, and an otherwise healthy shot with a
    few NaN power samples is kept."""


def test_missing_signals_skip_check():
    """A shot without Wtot_MJ or without P_oh_MW skips the energy check and is
    not culled by it."""


def test_fewer_than_two_valid_slices_skip_check():
    """A shot with fewer than 2 slices where both time and Wtot_MJ are finite
    skips the energy check instead of integrating a degenerate trace."""


def test_energy_check_runs_despite_device_culling_override():
    """The check runs in process_fn common culling, so devices overriding
    device_specific_culling (MAST, TCV) still get it: a broken-power shot that
    passes the device override is excluded anyway."""
