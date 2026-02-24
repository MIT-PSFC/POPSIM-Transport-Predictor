from popsim.simulate import StepperType

from popsim_transport_predictor.modules.profile_trajectory.trb import (
    ProfileTrajectoryOptimizerTRB,
)

PROFILE_TRAJECTORY_OPTIMIZER_CONFIG = {
    "project": "traj_opt_d3d_hbp",
    "train_run_builder": ProfileTrajectoryOptimizerTRB,
    "max_epochs": 4,
    "epochs_per_val": 2,
    "checkpoint_dir": None,
    "dataloader_config": {
        "ds_path": None,
        "debug": True,
        "state_vars": [
            "time"
        ],  # TODO(ZanderKeith): Trajectory optimization is time-dependent but stateless... might need a dummy state var
        "input_vars": ["Ip_MA", "B0", "ne20_edge", "beta"],
    },
    "model_init_config": {
        "stepper": StepperType.SIMPLE_EULER,
        "shape_ranges": {
            "R0": {"min": 0.0, "max": 1.0},
            "a_minor": {"min": 0.0, "max": 1.0},
            "kappa": {"min": 0.0, "max": 1.0},
            "delta_top": {"min": 0.0, "max": 1.0},
            "delta_bottom": {"min": 0.0, "max": 1.0},
        },
    },
    "loss_config": {
        "huber_delta": 0.1,
        "time_penalty": 1.0,  # Penalty to avoid trajectory shapes changing too much between timesteps
    },
    "optimizer_config": {
        "lr0": 5e-4,
        "lr_peak": 1e-2,
        "warmup_steps": 100,
        "decay_steps": 20000,
        "lrf": 5e-3,
        "weight_decay": 5e-3,
    },
}
