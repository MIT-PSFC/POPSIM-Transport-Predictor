"""Shot-stage segmentation shared by the study analysis pipelines.

Labels each timeslice of a shot as rampup / flattop / rampdown from the
plasma current trace, with an auxiliary-heating flag that subdivides the
flattop into ohmic and aux-heated timeslices.
Every study aggregates its per-timeslice metrics over these stages.
"""

import numpy as np

# Flattop is the contiguous span where |Ip| is at least this fraction of the
# shot's 95th percentile |Ip|. Rampup is everything before, rampdown after
FLATTOP_IP_FRACTION = 0.9

# Auxiliary heating (NBI/LH/ECRH/ICRF) above this total power counts as
# significant, splitting the flattop into ohmic and aux-heated timeslices
AUX_SIGNIFICANT_MW = 0.1

# The per-stage aggregation buckets every study reports
STAGE_AGG_NAMES = ("all", "rampup", "flattop", "flattop_ohmic", "flattop_aux", "rampdown")


def segment_stages(ip_ma: np.ndarray, p_aux_mw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label each timeslice of one shot as rampup / flattop / rampdown.

    Flattop is the contiguous index span between the first and last timeslice
    where |Ip| >= FLATTOP_IP_FRACTION * p95(|Ip|). Timeslices with NaN Ip get
    an empty stage label and are excluded from the stage aggregations.

    Returns:
        (stage, aux_heated): stage is an array of "rampup" / "flattop" /
        "rampdown" / "" labels, aux_heated a boolean array marking timeslices
        where the total auxiliary heating power exceeds AUX_SIGNIFICANT_MW.
    """
    abs_ip = np.abs(ip_ma)
    stage = np.full(abs_ip.shape, "", dtype=object)
    valid = np.isfinite(abs_ip)
    aux_heated = np.nan_to_num(p_aux_mw, nan=0.0) > AUX_SIGNIFICANT_MW

    if not valid.any():
        return stage.astype(str), aux_heated

    ip_p95 = np.nanpercentile(abs_ip, 95)
    at_flattop = valid & (abs_ip >= FLATTOP_IP_FRACTION * ip_p95)
    if not at_flattop.any():
        # Degenerate Ip trace, call every valid timeslice rampup
        stage[valid] = "rampup"
        return stage.astype(str), aux_heated

    flattop_idxs = np.flatnonzero(at_flattop)
    start, end = flattop_idxs[0], flattop_idxs[-1]
    idxs = np.arange(abs_ip.shape[0])
    stage[valid & (idxs < start)] = "rampup"
    stage[valid & (idxs >= start) & (idxs <= end)] = "flattop"
    stage[valid & (idxs > end)] = "rampdown"
    return stage.astype(str), aux_heated
