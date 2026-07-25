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
