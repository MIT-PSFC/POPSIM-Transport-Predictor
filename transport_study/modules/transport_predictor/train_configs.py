"""TORAX config construction for the torax-backed transport predictors."""

import copy

from transport_validation_datasets.machine.generic import UNIFORM_TIMEBASE_DT

from transport_study.modules.profile_predictor.train_configs import (
    TORAX_CONFIG_BASE,
    TORAX_TRANSPORT_BLOCKS,
)

# Solver per transport model, replacing the profile relaxation's solver.
# Without a corrector the transient n_e is taken at the old state,
# so the step solves n dT/dt instead of d(nT)/dt and loses -T dn/dt at any dt.
# One corrector iteration puts the 1 ms step within 2 percent of a converged reference on all four devices.
# Pereverzev at the TORAX defaults damps shape changes by ~dt chi / a^2,
# up to 40 percent in ne over 100 ms on C-Mod and TCV, so the smooth models run without it.
# QLKNN is stiff enough that its corrector iterations need Pereverzev to stay bounded.
# Each corrector iteration costs about one more full step, and a second one barely helps QLKNN.
TRANSPORT_SOLVER_BLOCKS = {
    "constant": {"use_pereverzev": False, "use_predictor_corrector": True, "n_corrector_steps": 1},
    "gyrobohm": {"use_pereverzev": False, "use_predictor_corrector": True, "n_corrector_steps": 1},
    "qlknn": {"use_pereverzev": True, "use_predictor_corrector": True, "n_corrector_steps": 1},
}


def make_transport_torax_config(transport_model: str) -> dict:
    """One-step TORAX config for the transport predictor.

    Reuses the profile predictor's TORAX skeleton and transport blocks, with
    the numerics window collapsed to a single solver step of the dataset
    timebase: TransportPredictorToraxBase advances exactly one step per
    __call__ and enforces t_final - t_initial == fixed_dt == sim_dt.
    The solver comes from TRANSPORT_SOLVER_BLOCKS, never from the shared base.
    """
    if transport_model not in TORAX_TRANSPORT_BLOCKS:
        raise ValueError(f"Unknown transport model '{transport_model}', valid: {sorted(TORAX_TRANSPORT_BLOCKS)}")
    torax_config = copy.deepcopy(TORAX_CONFIG_BASE)
    torax_config["transport"] = copy.deepcopy(TORAX_TRANSPORT_BLOCKS[transport_model])
    torax_config["solver"].update(TRANSPORT_SOLVER_BLOCKS[transport_model])
    torax_config["numerics"].update(
        {
            "t_initial": 0.0,
            "t_final": UNIFORM_TIMEBASE_DT,
            "fixed_dt": UNIFORM_TIMEBASE_DT,
            "min_dt": UNIFORM_TIMEBASE_DT / 10,
            "adaptive_dt": False,
            # The rebuild re-derives psi from Ip every step, so evolving it within the step changes nothing
            # (measured identical within 0.01) and costs 15-40 percent.
            # A carried TORAX state would hold psi at its seed instead
            "evolve_current": False,
        }
    )
    return torax_config
