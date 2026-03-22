import os
import shutil

import fire
from chex import dataclass
from loguru import logger
from popsim.ml.launch import launch_train
from popsim.ml.train_config import TrainConfig
from popsim.modules.transport_predictor.train_configs import update_submodule_configs

from transport_study.config import config
from transport_study.modules.profile_trajectory.train_configs import (
    PROFILE_TRAJECTORY_OPTIMIZER_CONFIG,
)
from transport_study.profile_transfer.run_study import ProfileStudy

# These times are informed by our reference shot 201927
# At most 8 trajectory points to plug in by hand
TRAJ_TIMES = [
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
]


class TrajectoryOptimization:
    @dataclass
    class Case:
        """
        Which profile predictor we're using
        How many times we allow the trajectory to change
        Whether to optimize the density or leave it unchanged
        """

        profile_predictor: str
        num_traj_times: int
        optimize_density: bool = True

        def __str__(self):
            return f"trajopt.{self.num_traj_times}.od_{self.optimize_density}.{self.profile_predictor}"

    def _path_to_profile_case(self, path: str) -> ProfileStudy.Case:
        # Extract the case name from the path, assuming it's the name of the last directory in the path
        case_name = path.split("/")[-1]
        case_pieces = case_name.split(".")
        model_type = case_pieces[1]
        training_data = case_pieces[2][3:]  # remove "td_" prefix
        data_normalization = "physics"  # Always using this for profile predictor

        if len(case_pieces) == 4:
            raise NotImplementedError(
                "Don't have a case with 4 pieces, need to update the parsing logic if we want to add one"
            )
        elif len(case_pieces) == 5:
            raise NotImplementedError(
                "Don't have a case with 5 pieces, need to update the parsing logic if we want to add one"
            )
        elif len(case_pieces) == 6:
            domain_adaptation = case_pieces[3][3:]  # remove "da_" prefix
            freeze_shapes = (
                case_pieces[4][7:] == "True"
            )  # remove "freeze_" prefix and convert to bool
            num_hp_shots = int(
                case_pieces[5][3:]
            )  # remove "hp_" prefix and convert to int
        else:
            raise ValueError(f"Unexpected case name format: {case_name}")

        return ProfileStudy.Case(
            model_type=model_type,
            training_data=training_data,
            data_normalization=data_normalization,
            domain_adaptation=domain_adaptation,
            freeze_shapes=freeze_shapes,
            num_hp_shots=num_hp_shots,
        )

    def _make_cases(self):
        cases = []
        for num_traj_times in range(1, len(TRAJ_TIMES) + 1):
            cases.extend(
                self.Case(
                    profile_predictor=str(self.predictor_case),
                    num_traj_times=num_traj_times,
                    optimize_density=optimize_density,
                )
                for optimize_density in [True, False]
            )
        return cases

    def __init__(
        self,
        name: str,
        working_dir_base: str,
        profile_module_checkpoint_dir: str,
        ds_path: str,
        traj_times: list[float],
        max_num_traj_times: int,
    ):
        if max_num_traj_times > len(traj_times):
            raise ValueError(
                f"max_num_traj_times {max_num_traj_times} cannot be greater than the number of traj_times {len(traj_times)}"
            )

        self.name = name
        self.profile_module_checkpoint_dir = profile_module_checkpoint_dir
        self.ds_path = ds_path
        self.traj_times = traj_times
        self.max_num_traj_times = max_num_traj_times

        self.working_dir = os.path.join(working_dir_base, name)
        os.makedirs(self.working_dir, exist_ok=True)
        self.predictor_case = self._path_to_profile_case(profile_module_checkpoint_dir)
        self.cases = self._make_cases()

    ###########################
    # Directories and Pathing #
    ###########################
    def checkpoint_dir(self, case: Case) -> str:
        return os.path.join(
            self.working_dir,
            "checkpoints",
            str(case),
        )

    ######################
    # Set up the configs #
    ######################
    def setup_optimization_config(
        self,
        case: Case,
    ) -> TrainConfig:
        """Set up the training config for trajectory optimization.
        Based on the dataset, we determine the allowable ranges for optimization variables and update the config accordingly.

        Args:
            ds_path (str): Path to the dataset.
            shape_times (list[float] | None, optional): List of shape times to use for the trajectory optimization.

        Returns:
            TrainConfig: The training config for trajectory optimization.
        """
        # Informed by the dataset characterization and Jayson Barr
        control_input_ranges = {
            "ne20_edge": (0.3, 0.4),  # Pedestal density [10^20 m^-3]
            "R0": (1.63, 1.95),  # Major radius [m]
            "gapin": (0.01, 0.12),  # Inner gap [m]
            "rxpt1": (1.09, 1.29),  # Lower X-point R [m]
            "zxpt1": (-1.25, -0.85),  # Lower X-point Z [m] (From Jayson Barr)
            "rxpt2": (1.08, 1.25),  # Upper X-point R [m]
            "zxpt2": (0.85, 1.25),  # Upper X-point Z [m]  (From Jayson Barr)
        }

        base_trajopt_config = TrainConfig.load(PROFILE_TRAJECTORY_OPTIMIZER_CONFIG)

        def _make_profile_predictor_config(
            case: ProfileStudy.Case, checkpoint_dir: str
        ) -> TrainConfig:
            profile_working_dir = os.path.dirname(checkpoint_dir)
            hyperparam_case = case.get_hyperparam_prereq()
            tuned_config_path = os.path.join(
                profile_working_dir, str(hyperparam_case), "tuned_config.yaml"
            )
            if not os.path.exists(tuned_config_path):
                raise FileNotFoundError(
                    f"Tuned config not found for profile predictor case {case} at path {tuned_config_path}"
                )

            profile_predictor_config_base = TrainConfig.load(tuned_config_path)
            if case.model_type != "unstructured_nn":
                # TODO(ZanderKeith) for the shape_init models, need to update their dataloader config
                # so the Te_shapes and ne_shapes get created properly
                # For now, unstructured_nn just needs the model_init_config
                raise NotImplementedError(
                    f"Don't have logic to update the profile predictor config for model type {case.model_type}, need to implement that if we want to use something other than unstructured_nn"
                )
            else:
                profile_predictor_config = profile_predictor_config_base

            return profile_predictor_config

        profile_predictor_config = _make_profile_predictor_config(
            self.predictor_case, self.profile_module_checkpoint_dir
        )

        # Special case because if we aren't changing density I want it to be exactly the same as the original shot,
        # and dstdenp is fixed after t = 2.5s
        if case.num_traj_times == 1 and self.traj_times == TRAJ_TIMES:
            traj_times = [self.traj_times[1]]
        else:
            traj_times = self.traj_times[: case.num_traj_times]

        trajopt_config = base_trajopt_config.model_copy(
            update={
                "max_epochs": config.max_epochs,
                "epochs_per_val": config.epochs_per_val,
                "patience": config.patience,
                "checkpoint_dir": self.checkpoint_dir(case),
                "dataloader_config": {
                    **base_trajopt_config.dataloader_config,
                    "ds_path": self.ds_path,
                    "debug": config.debug,
                },
                "model_init_config": {
                    **base_trajopt_config.model_init_config,
                    "input_ranges": control_input_ranges,
                    "traj_times": traj_times,
                    "submodules": {
                        "profile_predictor": profile_predictor_config.model_dump(),
                    },
                },
            }
        )

        # Update all submodule configs to use the same dataloader as the base module
        trajopt_config = update_submodule_configs(
            trajopt_config.model_dump(),
            [
                "profile_predictor",
            ],
        )

        return trajopt_config

    ############################
    # Run an optimization case #
    ############################
    def run_case(self, case: Case):
        """Run a trajectory optimization case."""
        config = self.setup_optimization_config(case)
        launch_train(config)

    #######################################################################
    # Output the control signal dataset encoding the optimized trajectory #
    #######################################################################
    def output_optimized_trajectory(self, case: Case):
        """Output the optimized trajectory as a dataset and as an instruction set to give to DIII-D physics operator

        Taken directly from 201927:
        iptipp (plasma current [A])
        bttbt (toroidal magnetic field [T])
        bmtpwrtar (normalized plasma beta) <- MAKE IT CLEAR THIS IS BETAN

        Things we may be optimizing over:
        dstdenp (pedestal density) <- ENSURE THE UNITS ARE CLEAR ON THIS
        idtrp   (geometric major radius) <- MAKE IT CLEAR THIS IS GEOMETRIC MAJOR RADIUS
        rxbot   (lower X-point R)
        zxbot   (lower X-point Z)

        Things that we're optimizing over but need to vibe out, not a clear mapping from DIII-D inputs:
        a_minor_desired  (desired minor radius, related to gapin and ieeseg06/ieeseg07 but not the same)
        kappa_desired (desired elongation, related to zxpt1 but disconnected from zxpt2)
        delta_top_desired (desired upper triangularity, related to zxpt2 and rxpt2 but not the same)
        delta_bot_desired (desired lower triangularity, we should be able to do this directly as long as we have gapin)

        My suggestion as to what the physics operator should plug in to recreate the optimized trajectory:
        gapin_suggested (my suggestion as to what the inner gap should be, but the other shapes are more important)
        """


###############################################
# Train the model and optimize the trajectory #
###############################################


def run_trajectory_optimization(
    trajopt_name: str,
    working_dir_base: str,
    profile_module_checkpoint_dir: str,
    ds_path: str | None = config.d3d_hp_dataset_path,
    traj_times: list[float] | None = TRAJ_TIMES,
    max_num_traj_times: int | None = 1,
    clean: bool | None = False,
):
    """Run trajectory optimization.

    Args:
        trajopt_name (str): Name of the trajectory optimization run, used for naming the checkpoint directory.
        working_dir_base (str): Base directory for checkpoints and logs. The actual checkpoint directory will be working_dir_base/trajopt_name.
        profile_module_checkpoint_dir (str): Checkpoint directory for the profile predictor submodule, taken from the ProfileStudy runs.
        ds_path (str): Path to the dataset.
        traj_times (list[float] | None): The times at which to optimize the trajectory.
        max_num_traj_times (int | None): The maximum number of times to change the shape during the trajectory.
        clean (bool, optional): Whether to clean the checkpoint directory before training.
    """

    if config.debug:
        trajopt_name = f"{trajopt_name}_debug"

    trajopt = TrajectoryOptimization(
        name=trajopt_name,
        working_dir_base=working_dir_base,
        profile_module_checkpoint_dir=profile_module_checkpoint_dir,
        ds_path=ds_path,
        traj_times=traj_times,
        max_num_traj_times=max_num_traj_times,
    )

    for case in trajopt.cases:
        if not os.path.exists(trajopt.checkpoint_dir(case)) or clean:
            shutil.rmtree(trajopt.checkpoint_dir(case), ignore_errors=True)
            trajopt.run_case(case)
        else:
            logger.info(
                f"Checkpoint directory already exists, skipping trajectory optimization... for case \n{case}"
            )

    #########################
    # Collect results, etc. #
    #########################


if __name__ == "__main__":
    # Usage: python popsim_transport_predictor/trajectory_optimization/optimize.py <command> [--options]
    fire.Fire(
        {
            "run_trajectory_optimization": run_trajectory_optimization,
        }
    )
