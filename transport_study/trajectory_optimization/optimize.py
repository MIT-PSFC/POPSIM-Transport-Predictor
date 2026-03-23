import getpass
import os
import shutil
import time
from datetime import datetime

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
from transport_study.orchestration.slurm_utils import (
    count_running_jobs,
    launch_trajopt_case_parallel,
    resources_available,
)
from transport_study.profile_transfer.run_study import ProfileStudy

# These times are informed by our reference shot 201927
# At most 8 trajectory points to plug in by hand
TRAJ_TIMES = [
    2.1,
    2.6,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
]
import xarray as xr

from transport_study.datasets.d3d.d3d_dataset import INNER_WALL


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
        for num_traj_times in range(1, self.max_num_traj_times + 1):
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
        traj_times: list[float],
        max_num_traj_times: int,
    ):
        if max_num_traj_times > len(traj_times):
            raise ValueError(
                f"max_num_traj_times {max_num_traj_times} cannot be greater than the number of traj_times {len(traj_times)}"
            )

        self.name = name
        self.profile_module_checkpoint_dir = profile_module_checkpoint_dir
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

    def result_path(self, case: Case) -> str:
        """Path to the predicted profiles dataset (test eval results)."""
        return os.path.join(
            self.working_dir, "outputs", str(case), "predicted_profiles.nc"
        )

    def output_path(self, case: Case) -> str:
        """Path to the done-marker output dataset for a case. Analogous to result_path in ProfileStudy."""
        return os.path.join(
            self.working_dir, "outputs", str(case), "optimized_trajectory.nc"
        )

    def train_job_name(self, case: Case) -> str:
        return f"trajopt_{case}"

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
        case (Case): The trajectory optimization case to set up the config for.

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
                    "ds_path": config.d3d_hp_dataset_path,
                    "debug": config.debug,
                },
                "model_init_config": {
                    **base_trajopt_config.model_init_config,
                    "input_ranges": control_input_ranges,
                    "traj_times": traj_times,
                    "optimize_density": case.optimize_density,
                    "submodules": {
                        "profile_predictor": profile_predictor_config.model_dump(),
                    },
                },
                "test_eval_suite_config": {"enabled": True},
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
        _, _, _, _, test_results = launch_train(config)
        if test_results is not None:
            ds = test_results.get("test/predicted_profiles")
            if ds is not None:
                result_path = self.result_path(case)
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                ds.to_netcdf(result_path)

    #######################################################################
    # Output the control signal dataset encoding the optimized trajectory #
    #######################################################################
    def output_optimized_trajectory(self, case: Case):  # noqa: PLR0915
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

        Output a text file with a human readable instruction set for the physics operator,

        The format should be something like a big paragraph that says
        "Reproducing shot 201927.
        Ensure the control system is in the following mode:
        betan control
        (I'll fill this in later)
        "

        Then there is a table of time points with updates, if they're given, as output by the optimized trajectory

        Also plot the signals on 5x3 grid comparing the original trajectory with the modified trajectory (use dark mode like found in the other plotting files)
        column 1: Ip, B0, betan, ne20_edge, blank
        column 2: R0, a_minor (derived), kappa (derived) delta_top (derived), delta_bot (derived)
        column 3: gapin (suggested), rxbot, zxbot, rxtop, zxtop
        """

        import matplotlib
        import numpy as np

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from popsim.ml.checkpointing import (
            create_default_checkpoint_manager,
            restore_model,
        )

        from transport_study.modules.profile_trajectory.trb import (
            ProfileTrajectoryOptimizerTRB,
        )

        ds_orig = xr.open_dataset(config.d3d_hp_dataset_path)
        ds_shot = ds_orig.sel(shot=201927)
        shot_time = ds_shot["time"].values  # 1-D time array [s] indexed by time_idx

        def _get_optimized_trajectory():
            """Load trajectory from checkpoint if available, otherwise fall back to
            the reference shot's programmed values as an unoptimized baseline."""
            checkpoint_dir = self.checkpoint_dir(case)

            # Replicate the traj_times selection logic from setup_optimization_config
            if case.num_traj_times == 1 and self.traj_times == TRAJ_TIMES:
                traj_times = np.array([self.traj_times[1]])
            else:
                traj_times = np.array(self.traj_times[: case.num_traj_times])

            if os.path.exists(checkpoint_dir):
                trajopt_config = self.setup_optimization_config(case)
                _, train_dl, _, _ = ProfileTrajectoryOptimizerTRB.get_dataloaders(
                    trajopt_config.dataloader_config
                )
                env = ProfileTrajectoryOptimizerTRB.model_init(
                    train_dl, trajopt_config.model_init_config
                )
                manager = create_default_checkpoint_manager(checkpoint_dir)
                env = restore_model(manager, env)
                module = env.module
                return {
                    "time": np.array(module.config.traj_times),
                    "R0": np.array(module.R0),
                    "gapin": np.array(module.gapin),
                    "rxbot": np.array(
                        module.rxpt1
                    ),  # lower X-pt R (rxpt1 in module → rxbot in dataset)
                    "zxbot": np.array(
                        module.zxpt1
                    ),  # lower X-pt Z (zxpt1 in module → zxbot in dataset)
                    "rxtop": np.array(
                        module.rxpt2
                    ),  # upper X-pt R (rxpt2 in module → rxtop in dataset)
                    "zxtop": np.array(
                        module.zxpt2
                    ),  # upper X-pt Z (zxpt2 in module → zxtop in dataset)
                    "ne20_edge": np.array(module.ne20_edge),
                }
            else:
                logger.critical(
                    f"No checkpoint found at {checkpoint_dir}. "
                    "Falling back to sample for debugging"
                )
                return {
                    "time": traj_times,
                    "R0": [1.665, 1.670],
                    "gapin": [0.023, 0.018],
                    "rxbot": [1.15, 1.12],
                    "zxbot": [-1.22, -1.19],
                    "rxtop": [1.150, 1.12],
                    "zxtop": [1.16, 1.14],
                    "ne20_edge": [0.55, 0.45],
                }

        optimized_trajectory = _get_optimized_trajectory()
        traj_times = np.array(optimized_trajectory["time"])

        # Derived shape quantities — replicates PCSInputMapper logic
        R0 = np.array(optimized_trajectory["R0"])
        gapin = np.array(optimized_trajectory["gapin"])
        rxpt1 = np.array(optimized_trajectory["rxbot"])
        zxpt1 = np.array(optimized_trajectory["zxbot"])
        rxpt2 = np.array(optimized_trajectory["rxtop"])
        zxpt2 = np.array(optimized_trajectory["zxtop"])
        a_minor = R0 - gapin - INNER_WALL
        kappa = np.abs(zxpt2 - zxpt1) / (a_minor * 2)
        delta_bot = (R0 - rxpt1) / a_minor
        delta_top = (R0 - rxpt2) / a_minor

        def _build_opt_waveform(prog_sig, opt_vals):
            """Return a full waveform over shot_time: follows ds_shot[prog_sig] up to
            traj_times[0], then forward-fills opt_vals from that point onward."""
            base = ds_shot[prog_sig].values.copy().astype(float)
            mask = shot_time >= traj_times[0]
            if mask.any():
                idx = np.clip(
                    np.searchsorted(traj_times, shot_time[mask], side="right") - 1,
                    0,
                    len(opt_vals) - 1,
                )
                base[mask] = opt_vals[idx]
            return base

        # Full optimized waveforms over shot_time
        R0_wave = _build_opt_waveform("R0_prog", R0)
        gapin_wave = _build_opt_waveform(
            "gapin", gapin
        )  # no gapin_prog, use measured as base
        rxbot_wave = _build_opt_waveform("rxbot_prog", rxpt1)
        zxbot_wave = _build_opt_waveform("zxbot_prog", zxpt1)
        rxtop_wave = _build_opt_waveform("rxtop_prog", rxpt2)
        zxtop_wave = _build_opt_waveform("zxtop_prog", zxpt2)
        ne20_wave = _build_opt_waveform(
            "ne20_edge_prog", np.array(optimized_trajectory["ne20_edge"])
        )

        def _var(arr):
            return ("time_idx", arr)

        # Save the optimized trajectory as full waveforms over shot_time.
        # Signal names and units match DIII-D PCS conventions.
        ds_traj = xr.Dataset(
            {
                "iptipp": _var(
                    ds_shot["Ip_MA_prog"].values * 1e6
                ),  # [A]            — unchanged from 201927
                "bttbt": _var(
                    ds_shot["B0_prog"].values
                ),  # [T]            — unchanged from 201927
                "bmtpwrtar": _var(
                    ds_shot["betan_prog"].values
                ),  #                — unchanged from 201927
                "dstdenp": _var(ne20_wave * 10),  # [10^19 m^-3]
                "idtrp": _var(R0_wave),  # [m]
                "gapin_opt": _var(
                    gapin_wave
                ),  # [m] — no clean PCS mapping (ieeseg06/07 not equivalent)
                "idtrxbot": _var(rxbot_wave),  # [m]
                "idtzxbot": _var(zxbot_wave),  # [m]
                "idtrxtop": _var(rxtop_wave),  # [m]
                "idtzxtop": _var(zxtop_wave),  # [m]
            },
            coords={
                "time_idx": np.arange(len(shot_time)),
                "time": ("time_idx", shot_time),
            },
            attrs={
                "description": "Optimized trajectory for DIII-D reference shot 201927",
                "case": str(case),
            },
        )

        output_dir = os.path.join(self.working_dir, "outputs", str(case))
        os.makedirs(output_dir, exist_ok=True)

        ds_traj_path = os.path.join(output_dir, "optimized_trajectory.nc")
        ds_traj.to_netcdf(ds_traj_path)
        logger.info(f"Saved optimized trajectory dataset to {ds_traj_path}")

        ###########################################################################
        # Human-readable instruction file for DIII-D physics operator            #
        ###########################################################################
        instruction_path = os.path.join(output_dir, "instructions.txt")
        with open(instruction_path, "w") as f:
            f.write("TRAJECTORY OPTIMIZATION PHYSICS OPERATOR INSTRUCTIONS\n")
            f.write("Reproducing shot 201927\n\n")
            f.write("Ensure the control system is in the following mode:\n")
            f.write("  - Betan control (bmtpwrtar waveform follows betanf)\n")
            f.write(
                "  - Pedestal density feedback (dstdenp waveform follows dssneped)\n"
            )
            f.write(
                "  - X-point shape control (idtrxbot, idtzxbot, idtrxtop, idtzxtop)\n\n"
            )
            f.write(
                "The following signals are UNCHANGED from shot 201927\n"
                "and should be programmed identically:\n"
                "  iptipp    — programmed plasma current [A]\n"
                "  bttbt     — programmed toroidal field [T]\n"
                "  bmtpwrtar — programmed normalized beta\n\n"
            )
            f.write(
                "Optimized control waveform updates.\n"
                "At each time point below, update the PCS waveform as either\n"
                "a linearly interpolated function or a step function\n\n"
            )
            # Rows: signals, columns: time points
            sig_labels = [
                "dstdenp [1e19/m3]",
                "idtrp [m]",
                "gapin [m]*",
                "idtrxbot [m]",
                "idtzxbot [m]",
                "idtrxtop [m]",
                "idtzxtop [m]",
                "--- desired shape ---",
                "a_minor [m]",
                "kappa",
                "delta_top",
                "delta_bot",
            ]
            sig_values = [
                [
                    f"{optimized_trajectory['ne20_edge'][i] * 10:.3f}"
                    for i in range(len(traj_times))
                ],
                [f"{R0[i]:.3f}" for i in range(len(traj_times))],
                [f"{gapin[i]:.3f}" for i in range(len(traj_times))],
                [f"{rxpt1[i]:.3f}" for i in range(len(traj_times))],
                [f"{zxpt1[i]:.3f}" for i in range(len(traj_times))],
                [f"{rxpt2[i]:.3f}" for i in range(len(traj_times))],
                [f"{zxpt2[i]:.3f}" for i in range(len(traj_times))],
                [""] * len(traj_times),
                [f"{a_minor[i]:.3f}" for i in range(len(traj_times))],
                [f"{kappa[i]:.3f}" for i in range(len(traj_times))],
                [f"{delta_top[i]:.3f}" for i in range(len(traj_times))],
                [f"{delta_bot[i]:.3f}" for i in range(len(traj_times))],
            ]
            time_headers = [f"t={t:.3f}s" for t in traj_times]
            label_w = max(len(s) for s in sig_labels) + 2
            col_w = 12
            # Header row
            f.write(
                " " * label_w + "  ".join(h.ljust(col_w) for h in time_headers) + "\n"
            )
            f.write(
                "-" * (label_w + col_w * len(traj_times) + 2 * (len(traj_times) - 1))
                + "\n"
            )
            for label, values in zip(sig_labels, sig_values, strict=True):
                f.write(
                    label.ljust(label_w)
                    + "  ".join(v.ljust(col_w) for v in values)
                    + "\n"
                )
            f.write(
                "\n* gapin: there is no clean mapping from gapin to a PCS signal\n"
                "  (ieeseg06/07 are related but not equivalent).\n"
                "  The physics operator must adjust the shape to achieve this inner gap. \n"
                "\nNote: a_minor, kappa, delta_top, and delta_bot are the actual values\n"
                "the trajectory is optimized for, derived from the X-point targets.\n"
                "If the desired shape cannot be achieved exactly due to the imperfect mapping to PCS signals, prioritize\n"
                "achieving the derived shape values over matching the exact idtrx/zx values\n\n"
            )
            f.write(
                f"\nGenerated by {getpass.getuser()} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
        logger.info(f"Saved instruction text to {instruction_path}")

        ###########################################################################
        # Comparison plot: 5x3 dark-mode grid                                    #
        ###########################################################################
        BACKGROUND_COLOR = "#2F2F2F"
        FACE_COLOR = "#1A1A1A"
        TEXT_COLOR = "white"
        ORIG_COLOR = "#4FC3F7"  # light blue — original programmed 201927
        MEAS_COLOR = "#A5D6A7"  # light green — measured 201927 (reference)
        OPT_COLOR = "#FFB74D"  # orange — optimized

        # Derived shape quantities from the original shot's PROGRAMMED X-point waveforms.
        # gapin has no programmed equivalent in the dataset, so use measured throughout.
        R0_prog_full = ds_shot["R0_prog"].values
        gapin_full = ds_shot["gapin"].values
        rxbot_prog_full = ds_shot["rxbot_prog"].values
        zxbot_prog_full = ds_shot["zxbot_prog"].values
        rxtop_prog_full = ds_shot["rxtop_prog"].values
        zxtop_prog_full = ds_shot["zxtop_prog"].values
        a_minor_prog = R0_prog_full - gapin_full - INNER_WALL
        kappa_prog = np.abs(zxtop_prog_full - zxbot_prog_full) / (a_minor_prog * 2)
        delta_bot_prog = (R0_prog_full - rxbot_prog_full) / a_minor_prog
        delta_top_prog = (R0_prog_full - rxtop_prog_full) / a_minor_prog

        fig, axes = plt.subplots(5, 3, figsize=(18, 20))
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        fig.suptitle(
            f"Trajectory: 201927 original vs optimized\n{case}",
            color=TEXT_COLOR,
            fontsize=13,
            y=0.98,
        )

        def _style(ax, title, ylabel=""):
            ax.set_facecolor(FACE_COLOR)
            ax.set_title(title, color=TEXT_COLOR, fontsize=9)
            ax.set_xlabel("Time [s]", color=TEXT_COLOR, fontsize=8)
            ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=8)
            ax.tick_params(colors=TEXT_COLOR, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(TEXT_COLOR)

        def _plot_signal(
            ax,
            orig_sig,
            opt_vals,
            title,
            ylabel,
            pcs_name="",
            measured_sig=None,
            prog_vals=None,
        ):
            """Plot original programmed, optional measured, and optional optimized traces.

            When prog_vals is provided it is used as the "programmed" reference line and
            ds_shot[orig_sig] is treated as the measured reference instead (measured_sig
            is then ignored).  When prog_vals is None, ds_shot[orig_sig] is the programmed
            line and ds_shot[measured_sig] (if given) is the measured line.
            """
            if prog_vals is not None:
                # Pre-computed programmed array (e.g. derived shape quantities)
                prog_valid = ~np.isnan(shot_time) & ~np.isnan(prog_vals)
                ax.plot(
                    shot_time[prog_valid],
                    prog_vals[prog_valid],
                    color=ORIG_COLOR,
                    linewidth=1.2,
                    label="201927 prog",
                    alpha=0.85,
                )
                meas = ds_shot[orig_sig].values
                meas_valid = ~np.isnan(shot_time) & ~np.isnan(meas)
                ax.plot(
                    shot_time[meas_valid],
                    meas[meas_valid],
                    color=MEAS_COLOR,
                    linewidth=1.0,
                    linestyle=":",
                    label="201927 meas",
                    alpha=0.8,
                )
                base = prog_vals
            else:
                orig = ds_shot[orig_sig].values
                valid = ~np.isnan(shot_time) & ~np.isnan(orig)
                ax.plot(
                    shot_time[valid],
                    orig[valid],
                    color=ORIG_COLOR,
                    linewidth=1.2,
                    label="201927 prog",
                    alpha=0.85,
                )
                if measured_sig is not None:
                    meas = ds_shot[measured_sig].values
                    meas_valid = ~np.isnan(shot_time) & ~np.isnan(meas)
                    ax.plot(
                        shot_time[meas_valid],
                        meas[meas_valid],
                        color=MEAS_COLOR,
                        linewidth=1.0,
                        linestyle=":",
                        label="201927 meas",
                        alpha=0.8,
                    )
                base = orig

            if opt_vals is not None:
                # Follow the programmed trace up to the first traj_time, then
                # forward-fill the optimized values from that point onward.
                opt_full = base.copy().astype(float)
                mask = shot_time >= traj_times[0]
                if mask.any():
                    idx = np.clip(
                        np.searchsorted(traj_times, shot_time[mask], side="right") - 1,
                        0,
                        len(opt_vals) - 1,
                    )
                    opt_full[mask] = opt_vals[idx]
                opt_valid = ~np.isnan(shot_time) & ~np.isnan(opt_full)
                ax.plot(
                    shot_time[opt_valid],
                    opt_full[opt_valid],
                    color=OPT_COLOR,
                    linewidth=1.5,
                    linestyle="--",
                    label="optimized",
                )

            full_title = title + (f"\n[{pcs_name}]" if pcs_name else "")
            _style(ax, full_title, ylabel)
            ax.legend(
                fontsize=7,
                labelcolor=TEXT_COLOR,
                facecolor=BACKGROUND_COLOR,
                edgecolor=TEXT_COLOR,
                loc="best",
            )

        # column 1: Ip, B0, betan, ne20_edge, blank
        _plot_signal(
            axes[0, 0],
            "Ip_MA_prog",
            None,
            "Plasma Current (prog)",
            r"$I_p$ [MA]",
            "iptipp",
            measured_sig="Ip_MA",
        )
        _plot_signal(
            axes[1, 0],
            "B0_prog",
            None,
            "Toroidal Field (prog)",
            r"$B_0$ [T]",
            "bttbt",
            measured_sig="B0",
        )
        _plot_signal(
            axes[2, 0],
            "betan_prog",
            None,
            "Betan (prog)",
            r"$\beta_N$",
            "bmtpwrtar",
            measured_sig="betan",
        )
        _plot_signal(
            axes[3, 0],
            "ne20_edge_prog",
            np.array(optimized_trajectory["ne20_edge"]),
            "Pedestal Density (prog)",
            r"$n_{e,edge}$ [$10^{20}$ m$^{-3}$]",
            "dstdenp÷10",
            measured_sig="ne20_edge",
        )
        axes[4, 0].set_visible(False)

        # column 2: R0, a_minor (derived), kappa (derived), delta_top (derived), delta_bot (derived)
        # Derived signals: prog_vals = derived from original programmed X-points; orig_sig = measured
        _plot_signal(
            axes[0, 1],
            "R0_prog",
            R0,
            "Major Radius (prog)",
            r"$R_0$ [m]",
            "idtrp",
            measured_sig="R0",
        )
        _plot_signal(
            axes[1, 1],
            "a_minor",
            a_minor,
            "Minor Radius (derived)",
            r"$a$ [m]",
            "derived",
            prog_vals=a_minor_prog,
        )
        _plot_signal(
            axes[2, 1],
            "kappa",
            kappa,
            "Elongation (derived)",
            r"$\kappa$",
            "derived",
            prog_vals=kappa_prog,
        )
        _plot_signal(
            axes[3, 1],
            "delta_top",
            delta_top,
            "Upper Triang. (derived)",
            r"$\delta_{top}$",
            "derived",
            prog_vals=delta_top_prog,
        )
        _plot_signal(
            axes[4, 1],
            "delta_bot",
            delta_bot,
            "Lower Triang. (derived)",
            r"$\delta_{bot}$",
            "derived",
            prog_vals=delta_bot_prog,
        )

        # column 3: gapin (suggested), rxbot, zxbot, rxtop, zxtop
        # X-point cols show programmed values as primary reference and measured as secondary reference
        _plot_signal(
            axes[0, 2], "gapin", gapin, "Inner Gap (suggested)", r"$g_{in}$ [m]"
        )
        _plot_signal(
            axes[1, 2],
            "rxbot_prog",
            rxpt1,
            "Lower X-pt R",
            r"$R_{x,bot}$ [m]",
            "idtrxbot",
            measured_sig="rxbot",
        )
        _plot_signal(
            axes[2, 2],
            "zxbot_prog",
            zxpt1,
            "Lower X-pt Z",
            r"$Z_{x,bot}$ [m]",
            "idtzxbot",
            measured_sig="zxbot",
        )
        _plot_signal(
            axes[3, 2],
            "rxtop_prog",
            rxpt2,
            "Upper X-pt R",
            r"$R_{x,top}$ [m]",
            "idtrxtop",
            measured_sig="rxtop",
        )
        _plot_signal(
            axes[4, 2],
            "zxtop_prog",
            zxpt2,
            "Upper X-pt Z",
            r"$Z_{x,top}$ [m]",
            "idtzxtop",
            measured_sig="zxtop",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        plot_path = os.path.join(output_dir, "trajectory_comparison.png")
        fig.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"Saved trajectory comparison plot to {plot_path}")


###############################################
# Train the model and optimize the trajectory #
###############################################


def run_trajectory_optimization(  # noqa: PLR0912
    trajopt_name: str,
    working_dir_base: str,
    profile_module_checkpoint_dir: str,
    traj_times: list[float] | None = TRAJ_TIMES,
    max_num_traj_times: int | None = 1,
    clean: bool | None = False,
    enable_parallelism: bool | None = False,
):
    """Run trajectory optimization.

    Args:
        trajopt_name (str): Name of the trajectory optimization run, used for naming the checkpoint directory.
        working_dir_base (str): Base directory for checkpoints and logs. The actual checkpoint directory will be working_dir_base/trajopt_name.
        profile_module_checkpoint_dir (str): Checkpoint directory for the profile predictor submodule, taken from the ProfileStudy runs.
        traj_times (list[float] | None): The times at which to optimize the trajectory.
        max_num_traj_times (int | None): The maximum number of times to change the shape during the trajectory.
        clean (bool, optional): Whether to clean the checkpoint and output directories before training.
        enable_parallelism (bool, optional): If True, submit each case as a SLURM job and loop until all outputs exist.
    """

    if config.debug:
        trajopt_name = f"{trajopt_name}_debug"

    trajopt = TrajectoryOptimization(
        name=trajopt_name,
        working_dir_base=working_dir_base,
        profile_module_checkpoint_dir=profile_module_checkpoint_dir,
        traj_times=traj_times,
        max_num_traj_times=max_num_traj_times,
    )

    if clean:
        if enable_parallelism:
            raise ValueError(
                "Cleaning is not supported when parallelism is enabled, to avoid accidentally deleting in-progress jobs. Please clean manually if desired."
            )
        for case in trajopt.cases:
            shutil.rmtree(trajopt.checkpoint_dir(case), ignore_errors=True)
            shutil.rmtree(
                os.path.dirname(trajopt.output_path(case)), ignore_errors=True
            )

    unfinished_cases = [
        case for case in trajopt.cases if not os.path.exists(trajopt.output_path(case))
    ]

    while len(unfinished_cases) > 0:
        logger.opt(colors=True).info(
            f"<bold><green>{len(unfinished_cases)} cases remaining</green></bold>"
        )
        for case in unfinished_cases:
            if os.path.exists(trajopt.output_path(case)):
                continue
            if enable_parallelism:
                running = count_running_jobs(
                    trajopt.train_job_name(case), config.partition
                )
                if running > 0:
                    logger.info(f"Job already running for {case}, skipping")
                    continue
                if not resources_available():
                    logger.info("No resources available, waiting...")
                    continue
                launch_trajopt_case_parallel(trajopt, case)
            else:
                if not os.path.exists(trajopt.checkpoint_dir(case)):
                    trajopt.run_case(case)
                if not os.path.exists(trajopt.output_path(case)):
                    trajopt.output_optimized_trajectory(case)

        unfinished_cases = [
            case
            for case in unfinished_cases
            if not os.path.exists(trajopt.output_path(case))
        ]
        if len(unfinished_cases) > 0:
            time.sleep(8)


if __name__ == "__main__":
    # Usage: python popsim_transport_predictor/trajectory_optimization/optimize.py <command> [--options]
    fire.Fire(
        {
            "run_trajectory_optimization": run_trajectory_optimization,
        }
    )
