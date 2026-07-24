import numpy as np

from transport_study.orchestration.stages import (
    AUX_SIGNIFICANT_MW,
    STAGE_AGG_NAMES,
    segment_stages,
)


def test_segment_stages_trapezoid():
    """Trapezoid Ip trace splits into rampup / flattop / rampdown at the 90 percent level."""
    ip = np.concatenate([np.linspace(0.0, 1.0, 20), np.full(60, 1.0), np.linspace(1.0, 0.0, 20)])
    p_aux = np.zeros_like(ip)
    p_aux[40:60] = 5.0

    stage, aux_heated = segment_stages(ip, p_aux)

    assert set(stage) == {"rampup", "flattop", "rampdown"}
    # The flattop span is contiguous and covers the constant segment
    flattop_idxs = np.flatnonzero(stage == "flattop")
    assert (np.diff(flattop_idxs) == 1).all()
    assert stage[20] == "flattop"
    assert stage[79] == "flattop"
    assert stage[0] == "rampup"
    assert stage[-1] == "rampdown"
    # Stages are ordered rampup < flattop < rampdown
    assert np.flatnonzero(stage == "rampup").max() < flattop_idxs.min()
    assert flattop_idxs.max() < np.flatnonzero(stage == "rampdown").min()

    assert aux_heated[40:60].all()
    assert not aux_heated[:40].any()
    assert not aux_heated[60:].any()


def test_segment_stages_nan_handling():
    """NaN Ip timeslices get an empty label, NaN aux power counts as unheated."""
    ip = np.concatenate([np.linspace(0.0, 1.0, 10), np.full(20, 1.0), np.linspace(1.0, 0.0, 10)])
    ip[3] = np.nan
    p_aux = np.full_like(ip, np.nan)
    p_aux[15] = 2 * AUX_SIGNIFICANT_MW

    stage, aux_heated = segment_stages(ip, p_aux)

    assert stage[3] == ""
    assert aux_heated[15]
    assert not aux_heated[16]

    all_nan_stage, all_nan_aux = segment_stages(np.full(5, np.nan), np.zeros(5))
    assert (all_nan_stage == "").all()
    assert not all_nan_aux.any()


def test_stage_agg_names_cover_flattop_split():
    assert "flattop_ohmic" in STAGE_AGG_NAMES
    assert "flattop_aux" in STAGE_AGG_NAMES
    assert STAGE_AGG_NAMES[0] == "all"
