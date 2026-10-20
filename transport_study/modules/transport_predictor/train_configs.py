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

# How the TORAX-backed transport predictors carry state between steps:
# "rebuild" (TransportPredictorTorax) re-seeds TORAX from the stored ne / te every step,
# "carry" (TransportPredictorToraxCarry) also keeps T_i and psi, so the ion channel and the current evolve
VALID_TORAX_STATES = ("rebuild", "carry")


def make_transport_torax_config(transport_model: str, torax_state: str = "rebuild") -> dict:
    """One-step TORAX config for the transport predictor.

    Reuses the profile predictor's TORAX skeleton and transport blocks, with
    the numerics window collapsed to a single solver step of the dataset
    timebase: TransportPredictorToraxBase advances exactly one step per
    __call__ and enforces t_final - t_initial == fixed_dt == sim_dt.
    The solver comes from TRANSPORT_SOLVER_BLOCKS, never from the shared base.
    Angioni-Sauter neoclassical transport runs on top of the core transport model, unlike the profile relaxation.
    torax_state picks the module the config is for, one of VALID_TORAX_STATES.
    """
    if transport_model not in TORAX_TRANSPORT_BLOCKS:
        raise ValueError(f"Unknown transport model '{transport_model}', valid: {sorted(TORAX_TRANSPORT_BLOCKS)}")
    if torax_state not in VALID_TORAX_STATES:
        raise ValueError(f"Unknown torax state '{torax_state}', valid: {VALID_TORAX_STATES}")
    torax_config = copy.deepcopy(TORAX_CONFIG_BASE)
    torax_config["transport"] = copy.deepcopy(TORAX_TRANSPORT_BLOCKS[transport_model])
    torax_config["solver"].update(TRANSPORT_SOLVER_BLOCKS[transport_model])
    # Neoclassical chi, D and V including the Ware pinch, at 1.05 - 1.10x the cost of a rollout step.
    # On top of BGB it cut the DIII-D zero-shot seed spread 2.5x, and the network moved heat transport from gyroBohm to it
    torax_config["neoclassical"]["transport"] = {"model_name": "angioni_sauter"}
    torax_config["numerics"].update(
        {
            "t_initial": 0.0,
            "t_final": UNIFORM_TIMEBASE_DT,
            "fixed_dt": UNIFORM_TIMEBASE_DT,
            "min_dt": UNIFORM_TIMEBASE_DT / 10,
            "adaptive_dt": False,
            # The rebuild re-derives psi from Ip every step, so evolving it within the step changes nothing
            # (measured identical within 0.01) and costs 15-40 percent
            "evolve_current": False,
        }
    )
    if torax_state == "carry":
        # The carried psi persists between steps, so the current evolves over the rollout.
        # Every step re-enters it through profile_conditions.psi, a placeholder here overridden per step,
        # the segment start takes it from the current-profile formula instead (TransportPredictorToraxCarry.seed_step_fn)
        torax_config["numerics"]["evolve_current"] = True
        torax_config["profile_conditions"]["initial_psi_mode"] = "profile_conditions"
        torax_config["profile_conditions"]["psi"] = {0: {0: 0.0, 1: 1.0}}
    return torax_config
