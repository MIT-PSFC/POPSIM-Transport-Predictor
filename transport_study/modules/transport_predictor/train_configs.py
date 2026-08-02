"""TORAX config construction for the torax-backed transport predictors."""

import copy

from transport_study.datasets import UNIFORM_TIMEBASE_DT_S
from transport_study.modules.profile_predictor.train_configs import (
    TORAX_CONFIG_BASE,
    TORAX_TRANSPORT_BLOCKS,
)


def make_transport_torax_config(transport_model: str) -> dict:
    """One-step TORAX config for the transport predictor.

    Reuses the profile predictor's TORAX skeleton and transport blocks, with
    the numerics window collapsed to a single solver step of the dataset
    timebase: TransportPredictorToraxBase advances exactly one step per
    __call__ and enforces t_final - t_initial == fixed_dt == sim_dt.
    """
    if transport_model not in TORAX_TRANSPORT_BLOCKS:
        raise ValueError(f"Unknown transport model '{transport_model}', valid: {sorted(TORAX_TRANSPORT_BLOCKS)}")
    torax_config = copy.deepcopy(TORAX_CONFIG_BASE)
    torax_config["transport"] = copy.deepcopy(TORAX_TRANSPORT_BLOCKS[transport_model])
    # Cyclotron radiation is dropped HERE and not from TORAX_CONFIG_BASE, so
    # the profile study (which never diverged) keeps its physics unchanged.
    #
    # TORAX models it with something where the fit is singular
    # on a FLAT profile (p_axis -> p_edge zeroes the denominator) and NaN on a
    # non-monotonic one (log of a negative). This study seeds every rollout
    # from measured profiles floored at TE_SEED_FLOOR_KEV / NE_SEED_FLOOR_20,
    # so plasma-initiation timeslices arrive largely flat at those floors and
    # hit the first case, while record-pressure timeslices hit the second.
    #
    # Physically cheap to lose: cyclotron losses scale ~ B^2 and matter for
    # high-field reactor plasmas, not for MAST at ~0.66 T (utterly negligible)
    # or C-Mod at a few keV (small).
    torax_config["sources"].pop("cyclotron_radiation", None)
    torax_config["numerics"].update(
        {
            "t_initial": 0.0,
            "t_final": UNIFORM_TIMEBASE_DT_S,
            "fixed_dt": UNIFORM_TIMEBASE_DT_S,
            "min_dt": UNIFORM_TIMEBASE_DT_S / 10,
            "adaptive_dt": False,
        }
    )
    # The base already uses the minimum 1 Picard corrector iteration
    # (benchmarked: corrector count does not change converged loss), and a
    # 1 ms step is far less stiff than the base's 20 ms relaxation anyway
    return torax_config
