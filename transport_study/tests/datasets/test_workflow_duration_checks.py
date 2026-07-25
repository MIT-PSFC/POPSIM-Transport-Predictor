"""Test stubs for the shot duration culling in DataWorkflow.process_fn.

The span check (time.max - time.min) cannot see mid-shot holes, so process_fn
also requires the accumulated valid time (valid slice count times the 1 kHz
dt) to reach min_shot_duration. These stubs cover both checks.
"""


def test_dense_shot_shorter_than_min_duration_excluded():
    """A shot with contiguous valid slices whose span is below
    min_shot_duration is excluded by the span check with the
    'duration after processing' warning."""


def test_sparse_shot_with_wide_span_excluded():
    """A shot surviving filtering as a few isolated timeslices (e.g. 3 slices
    spread over 0.28 s, like MAST shot 29447) passes the span check but is
    excluded by the accumulated valid time check with the 'seconds of valid
    data' warning."""


def test_dense_shot_meeting_min_duration_kept():
    """A shot with accumulated valid time at or above min_shot_duration is
    returned unchanged by process_fn."""


def test_accumulated_time_exactly_at_threshold_kept():
    """A shot whose valid slice count times UNIFORM_TIMEBASE_DT_S equals
    min_shot_duration exactly is kept (check is strict less-than)."""


def test_all_nan_shot_excluded():
    """A shot whose slices are all NaN after filtering (cleaned size 0) is
    excluded and does not crash either duration computation."""
