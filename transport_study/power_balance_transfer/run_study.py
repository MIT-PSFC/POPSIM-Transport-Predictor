import json
import os
import shutil
from dataclasses import dataclass
from itertools import product

import fire
import netCDF4  # noqa: F401
from loguru import logger
from popsim.ml import TrainConfig
from popsim.ml.launch import launch_agent, launch_sweep, launch_train

from transport_study import PACKAGE_ROOT
from transport_study.config import config
from transport_study.orchestration.slurm_utils import count_running_jobs
from transport_study.orchestration.study import Study
from transport_study.orchestration.wandb_utils import (
    get_completed_runs,
    get_sweep_id,
    run_clean_sweeps,
)


class PowerBalanceStudy(Study):
    HYPERPARAM_TRAINING_DATA = "cmod_tcv"
    HYPERPARAM_DATA_NORMALIZATION = "coral"
    HYPERPARAM_DOMAIN_ADAPTATION = None
    HYPERPARAM_FREEZE_SUBMODULES = True
    HYPERPARAM_NUM_HP_SHOTS = None

    ##################
    # INITIALIZATION #
    ##################
    @dataclass
    class Case(Study.Case):
        """
        model_type: The type of power_balance model to use.
        - scaling_law: H89, H98, and P_LH scaling laws to predict tau_e
        - sciml: neural network predicts tau_e, and we do the power balance calculation
        - unstructured_nn: a single neural network directly predicts stored energy evolution

        training_data: The dataset(s) used for training
        - cmod: C-Mod only
        - tcv: TCV only
        - cmod_tcv: C-Mod + TCV
        - exnihilo: No historic training data

        data_normalization: The method for normalizing the input data.
        - raw: No normalization, Ip, Wtot, etc. are in their original units
        - physics: Convert to typical dimensionless parameters like beta, q95, f_G, etc.
        - z_score: Within each device, normalize each variable to zero mean and unit variance.
        - coral: Use the CORAL method to align covariances of source and target domains (https://arxiv.org/abs/1612.01939)

        domain_adaptation: The method for domain adaptation between source and target devices.
        - none: No domain adaptation, train and test on the same device(s). This is used for hyperparameter tuning and as a baseline for comparison, answering the question "what is the best possible performance we could expect if we had a bunch of data?"
        - mixing: Add a small amount of highly-weighted target data during training
        - transfer: Train on source data, freeze all but the last layers of the model, and fine-tune on a small amount of target data

        freeze_submodules: Whether to freeze certain submodules of the model during training.
        The P_oh and P_rad signals are hard to quantify, we might want to let them drift from the original targets to better match Wtot_MJ

        num_hp_shots: The number of high-performance shots included in the training data, or None to include all high-performance shots (including all shots in training is cheating, but again answers the question of what is the best possible performance).
        """

        model_type: str  # scaling_law, sciml, unstructured_nn
        training_data: str  # cmod, tcv, cmod_tcv, exnihilo
        data_normalization: str  # raw, physics, z_score, coral
        domain_adaptation: str  # none, mixing, transfer
        freeze_submodules: bool
        num_hp_shots: (
            int | None
        )  # Number of high-performance shots included in training, or None for all (should be None if domain_adaptation is None)
        # So that's 3 (model type) x 4 (training data) x 4 (normalization) x 3 (domain adaptation) x 2 (freeze or not) x 6 (hp shots included) = 1728 results
        # Even less since the hyperparameter tuning is only done for a subset of cases
        prereqs: (
            list[Study.Case] | None
        )  # If not None, this case depends on the results of another case, and should only be run after that case has been run

        def is_hyperparam_case(self) -> bool:
            if (
                self.training_data == PowerBalanceStudy.HYPERPARAM_TRAINING_DATA
                and self.data_normalization
                == PowerBalanceStudy.HYPERPARAM_DATA_NORMALIZATION
                and self.domain_adaptation
                == PowerBalanceStudy.HYPERPARAM_DOMAIN_ADAPTATION
                and self.freeze_submodules
                == PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES
                and self.num_hp_shots == PowerBalanceStudy.HYPERPARAM_NUM_HP_SHOTS
            ):
                return True
            else:
                return False

        def get_hyperparam_prereq(self) -> Study.Case:
            if self.is_hyperparam_case():
                return self
            else:
                return Study.Case(
                    model_type=self.model_type,
                    training_data=PowerBalanceStudy.HYPERPARAM_TRAINING_DATA,
                    data_normalization=PowerBalanceStudy.HYPERPARAM_DATA_NORMALIZATION,
                    domain_adaptation=PowerBalanceStudy.HYPERPARAM_DOMAIN_ADAPTATION,
                    freeze_submodules=PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES,
                    num_hp_shots=PowerBalanceStudy.HYPERPARAM_NUM_HP_SHOTS,
                )

        def __init__(
            self,
            model_type: str,
            training_data: str,
            data_normalization: str,
            domain_adaptation: str,
            freeze_submodules: bool,
            num_hp_shots: int | None,
        ):
            self.model_type = model_type
            self.training_data = training_data
            self.data_normalization = data_normalization
            self.domain_adaptation = domain_adaptation
            self.freeze_submodules = freeze_submodules
            self.num_hp_shots = num_hp_shots

            # Recursively add prereqs based on the logic of which cases depend on which other cases
            if model_type not in [
                "scaling_law",
                "sciml",
                "unstructured_nn",
                "p_oh",
                "p_rad",
            ]:
                raise ValueError(f"Unknown model type: {model_type}")
            if domain_adaptation is None and num_hp_shots is not None:
                raise ValueError(
                    "If domain_adaptation is None, num_hp_shots must also be None since this means we're training and testing on the same dataset and no high-performance data is being used"
                )
            if (
                model_type in ["p_oh", "p_rad"]
                and freeze_submodules != PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES
            ):
                raise ValueError(
                    f"freeze_submodules should be a dummy value ({PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES}) for submodule {model_type}"
                )

            prereqs = []

            # Add prereqs based on whether this is a hyperparameter tuning case or not.
            if not self.is_hyperparam_case():
                prereqs += [
                    PowerBalanceStudy.Case(
                        model_type=model_type,
                        training_data=PowerBalanceStudy.HYPERPARAM_TRAINING_DATA,
                        data_normalization=PowerBalanceStudy.HYPERPARAM_DATA_NORMALIZATION,
                        domain_adaptation=PowerBalanceStudy.HYPERPARAM_DOMAIN_ADAPTATION,
                        freeze_submodules=PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES,
                        num_hp_shots=PowerBalanceStudy.HYPERPARAM_NUM_HP_SHOTS,
                    )
                ]

            # Set prereqs based on model type
            if model_type in ["sciml", "scaling_law"]:
                prereqs += [
                    PowerBalanceStudy.Case(
                        model_type="p_oh",
                        training_data=training_data,
                        data_normalization=data_normalization,
                        domain_adaptation=domain_adaptation,
                        freeze_submodules=PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES,
                        num_hp_shots=num_hp_shots,
                    ),
                    PowerBalanceStudy.Case(
                        model_type="p_rad",
                        training_data=training_data,
                        data_normalization=data_normalization,
                        domain_adaptation=domain_adaptation,
                        freeze_submodules=PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES,
                        num_hp_shots=num_hp_shots,
                    ),
                ]

            # Set prereqs based on domain adaptation
            if domain_adaptation == "transfer":
                prereqs += [
                    PowerBalanceStudy.Case(
                        model_type=model_type,
                        training_data=training_data,
                        data_normalization=data_normalization,
                        domain_adaptation=None,
                        freeze_submodules=freeze_submodules,
                        num_hp_shots=None,
                    )
                ]

            if len(prereqs) > 0:
                self.prereqs = prereqs
            else:
                self.prereqs = None

        def __str__(self):
            if self.domain_adaptation:
                return f"case.{self.model_type}.td_{self.training_data}.dn_{self.data_normalization}.da_{self.domain_adaptation}.freezesub_{self.freeze_submodules}.hp_{self.num_hp_shots}"
            else:
                return f"case.{self.model_type}.td_{self.training_data}.dn_{self.data_normalization}.freezesub_{self.freeze_submodules}"

        def __hash__(self):
            if self.domain_adaptation:
                return hash(
                    (
                        self.model_type,
                        self.training_data,
                        self.data_normalization,
                        self.domain_adaptation,
                        self.freeze_submodules,
                        self.num_hp_shots,
                    )
                )
            else:
                return hash(
                    (
                        self.model_type,
                        self.training_data,
                        self.data_normalization,
                        self.domain_adaptation,
                        self.freeze_submodules,
                    )
                )

    def make_cases(
        self,
        model_types,
        training_datasets,
        data_normalization_methods,
        domain_adaptation_methods,
        freeze_submodules_options,
        num_hp_shots_options,
    ):
        cases = []
        # Make every case we're interested in for this study
        for (
            model_type,
            training_dataset,
            data_normalization,
            domain_adaptation,
            freeze_submodules,
            num_hp_shots,
        ) in product(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_submodules_options,
            num_hp_shots_options,
        ):
            if domain_adaptation is None and num_hp_shots is not None:
                continue  # Invalid case, skip

            case = self.Case(
                model_type=model_type,
                training_data=training_dataset,
                data_normalization=data_normalization,
                domain_adaptation=domain_adaptation,
                freeze_submodules=freeze_submodules,
                num_hp_shots=num_hp_shots,
            )

            cases.append(case)

        unwrapped_cases = []
        for case in cases:

            def _unwrap_prereqs(case):
                unwrapped_cases.append(case)
                if case.prereqs is not None:
                    for prereq in case.prereqs:
                        _unwrap_prereqs(prereq)

            _unwrap_prereqs(case)

        unique_cases = list(set(unwrapped_cases))  # Remove duplicates
        return unique_cases

    def __init__(
        self,
        name: str,
        working_dir_base: str,
        dataset_paths: dict[str, str],
        model_types: list[str],
        training_datasets: list[str],
        data_normalization_methods: list[str],
        domain_adaptation_methods: list[str],
        freeze_submodules_options: list[bool],
        num_hp_shots_options: list[int | None],
        hp_test_set_size: int,
    ):
        cases = self.make_cases(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_submodules_options,
            num_hp_shots_options,
        )
        super().__init__(name, working_dir_base, dataset_paths, cases, hp_test_set_size)

        logger.info(f"C-Mod dataset path: {dataset_paths.get('cmod', 'Not provided')}")
        logger.info(f"TCV dataset path: {dataset_paths.get('tcv', 'Not provided')}")
        logger.info(
            f"DIII-D high-performance dataset path: {dataset_paths.get('d3d_hp', 'Not provided')}"
        )

        logger.info(f"Model types: {model_types}")
        logger.info(f"Training datasets: {training_datasets}")
        logger.info(f"Data normalization methods: {data_normalization_methods}")
        logger.info(f"Domain adaptation methods: {domain_adaptation_methods}")
        logger.info(f"Freeze submodules options: {freeze_submodules_options}")
        logger.info(f"Number of high-performance shots options: {num_hp_shots_options}")

    ####################
    # PATHING / NAMING #
    ####################
    def trained_model_dir(self, case: Case) -> str:
        """Given a case, return the path where the trained model checkpoints for that case should be stored."""
        return os.path.join(self.model_dir, str(case))

    def result_path(self, case: Case) -> str:
        """Given a case, return the path where the results for that case should be stored."""
        return os.path.join(self.result_dir, str(case), "result_data.nc")

    def tuned_config_path(self, case: Case) -> str:
        """Given a case, return the path where the tuned hyperparameters for that case should be stored"""
        hyperparam_case = case.get_hyperparam_prereq()
        return os.path.join(self.model_dir, str(hyperparam_case), "tuned_config.yaml")

    def wandb_project_name(self, case: Case) -> str:
        """Given a case, return the wandb project name to use for that case"""
        return f"{self.name}.{case}"

    def sweep_job_name(self, case: Case) -> str:
        return f"sweep_{case}"

    def train_job_name(self, case: Case) -> str:
        return f"train_{case}"

    #############
    # EXECUTION #
    #############
    def check_data_requirements(self, case: Case) -> bool:
        """Given a case, check if the required data for that case is available. If not, return False and print a message indicating what data is missing."""
        required_datasets = set()

        if case.training_data in ["cmod", "cmod_tcv"]:
            required_datasets.add("cmod")
        if case.training_data in ["tcv", "cmod_tcv"]:
            required_datasets.add("tcv")
        if case.training_data == "exnihilo" or case.domain_adaptation in [
            "mixing",
            "transfer",
        ]:
            required_datasets.add("d3d_hp")

        missing_datasets = [
            ds for ds in required_datasets if ds not in self.dataset_paths.keys()
        ]
        if len(missing_datasets) > 0:
            logger.warning(
                f"Case {case} is missing required datasets: {missing_datasets}. Skipping this case."
            )
            return False

        return True

    def run_case(  # noqa: PLR0912
        self,
        case: Study.Case,
        skip_tuning: bool,
        enable_parallelism: bool,
    ):
        """Run a single case of the study, including hyperparameter tuning, training, and evaluation as needed.

        If case or a prereq is in progress, simply return and let orchestration loop try again later.
        """
        if not self.check_data_requirements(case):
            raise ValueError(
                f"Case {case} does not have the required data to run. This should have been caught earlier!"
            )

        if self.check_prereq_satisfied(case):
            logger.info(f"RUNNING CASE:\n{case}")
            # Prereq is satisfied, can run this case.
            if case.is_hyperparam_case():
                if skip_tuning:
                    logger.info("Skipping hyperparameter tuning")
                    # Copy default config for this module and put it in the trained model dir so the rest of the workflow can find it
                    default_config = self.make_train_config(case)
                    tuned_config_path = self.tuned_config_path(case)
                    os.makedirs(os.path.dirname(tuned_config_path), exist_ok=True)
                    with open(tuned_config_path, "w") as f:
                        json.dump(default_config.model_dump(), f, indent=4)
                else:
                    logger.info("Checking if hyperparameter tuning is already done")
                    tuned_config_path = self.tuned_config_path(case)
                    if os.path.exists(tuned_config_path):
                        logger.info(
                            f"Hyperparameter tuning completed, tuned config found at {tuned_config_path}"
                        )
                    else:
                        completed_runs = get_completed_runs(
                            self.wandb_project_name(case)
                        )
                        if len(completed_runs) > config.hyperparam_sweeps:
                            logger.info(
                                f"Hyperparameter sweeps completed with {len(completed_runs)} runs"
                            )
                            # Check if there are any running jobs for this case
                            if enable_parallelism:
                                running_jobs = count_running_jobs(
                                    self.sweep_job_name(case), config.partition
                                )
                                if len(running_jobs) > 0:
                                    logger.info(
                                        f"Found {len(running_jobs)} running jobs, waiting for them to complete before proceeding"
                                    )
                                    return
                        else:
                            logger.info(
                                f"Hyperparameter sweeps incomplete, {len(completed_runs)} out of {config.hyperparam_sweeps} runs"
                            )
                            logger.info("Launching hyperparameter sweep")
                            self.launch_sweep(case)
                            return

                # At this point, we know the tuned config is available at tuned_config_path, so we can proceed to training
                if enable_parallelism:
                    running_jobs = count_running_jobs(
                        self.train_job_name(case), config.partition
                    )
                    if len(running_jobs) > 0:
                        logger.info(
                            f"Found {len(running_jobs)} running training jobs, waiting for them to complete before proceeding"
                        )
                        return

                logger.info("Launching training")
                self.launch_train(case)
        else:
            for prereq in case.prereqs:
                if not os.path.exists(self.result_path(prereq)):
                    logger.debug(
                        f"Prereq not satisfied yet, running that first.\nCase: {case}\nPrereq: {prereq}"
                    )
                    self.run_case(
                        prereq,
                        skip_tuning=skip_tuning,
                        enable_parallelism=enable_parallelism,
                    )
                    return

    def _input_vars(self, case: Case) -> list[str]:
        input_vars_base = [
            "Ip_MA",
            "B0",
            "R0",
            "a_minor",
            "kappa",
            "ne20_line_avg",
            "P_aux_MW",
        ]
        if case.data_normalization == "raw":
            input_vars = input_vars_base
        elif case.data_normalization == "physics":
            input_vars = [
                *input_vars_base,
                "epsilon",
                "q_star",
                "f_G",
                "aB0",
                "surface_power_density",
            ]
        elif case.data_normalization == "z_score":
            input_vars = [*input_vars_base, *(f"{var}_z" for var in input_vars_base)]
        elif case.data_normalization == "coral":
            input_vars = [
                *input_vars_base,
                *(f"{var}_coral" for var in input_vars_base),
            ]
        else:
            raise ValueError(f"Unknown normalization method: {case.data_normalization}")

        return input_vars

    def make_train_config(self, case: Case) -> TrainConfig:
        """Make the TrainConfig for a given case.
        If a hyperparameter tuned config is available, fills in the hyperparameters from that, otherwise uses default config.
        """
        optimizer_config_base = {
            "lr0": 1e-4,
            "transition_steps": 500,
            "decay_rate": 0.5,
            "lrf": 5e-4,
            "weight_decay": 2e-4,
        }
        loss_config_base = {
            "huber_delta": 0.5,
        }
        dataloader_config_base = {
            "training_data": case.training_data,
            "data_normalization": case.data_normalization,
            "domain_adaptation": case.domain_adaptation,
            "num_hp_shots": case.num_hp_shots,
            "hp_test_set_size": self.hp_test_set_size,
            "prng_seed": 42,
            "debug": config.debug,
            # Hyperparameters
            "segment_length_train": 100,
            "segment_overlap_train": 50,
            "batch_size": 8192,
            # Part of validation, should be left alone during hyperparameter tuning
            "segment_length_val": None,
            "segment_overlap_val": 0,
        }
        test_eval_suite_config_base = {
            "result_path": self.result_path(case),
        }

        if case.model_type == "p_oh":
            input_vars = self._input_vars(case)
            train_config_base = TrainConfig(
                project=self.wandb_project_name(case),
                train_run_builder="transport_study.modules.power_balance.p_oh.trb.OhmicPowerTRB",
                max_epochs=config.max_epochs,
                epochs_per_val=config.epochs_per_val,
                checkpoint_dir=self.trained_model_dir(
                    case
                ),  # When doing hyperparameter tuning, this gets overwritten by the wandb agent
                dataloader_config={
                    "target_vars": ["P_oh_MW"],
                    "input_vars": input_vars,
                    "data_train_run_builder": "transport_study.modules.power_balance.trb.PowerBalanceTRB",  # Needed for submodules
                    **dataloader_config_base,
                },
                model_init_config={
                    "nn_depth": 2,
                    "nn_width": 16,
                    "min_val": 0,  # Minimum ohmic power in MW
                    "max_val": None,  # Get max from training data
                    "prng_seed": 42,
                    "in_size": 7,
                    "out_size": 1,
                    "data_normalization": case.data_normalization,
                },
                loss_config=loss_config_base,
                optimizer_config=optimizer_config_base,
                test_eval_suite_config=test_eval_suite_config_base,
            )
        elif case.model_type == "p_rad":
            input_vars = self._input_vars(case)
            train_config_base = TrainConfig(
                project=self.wandb_project_name(case),
                train_run_builder="transport_study.modules.power_balance.p_rad.trb.RadiatedPowerTRB",
                max_epochs=config.max_epochs,
                epochs_per_val=config.epochs_per_val,
                checkpoint_dir=self.trained_model_dir(
                    case
                ),  # When doing hyperparameter tuning, this gets overwritten by the wandb agent
                dataloader_config={
                    "target_vars": ["P_rad_MW"],
                    "input_vars": input_vars,
                    "data_train_run_builder": "transport_study.modules.power_balance.trb.PowerBalanceTRB",
                    **dataloader_config_base,
                },
                model_init_config={
                    "nn_depth": 2,
                    "nn_width": 16,
                    "min_val": 0,  # Minimum radiated power in MW, probably doesn't need to be enforced but just in case
                    "max_val": None,  # Get max from training data
                    "prng_seed": 42,
                    "in_size": 7,
                    "out_size": 1,
                    "data_normalization": case.data_normalization,
                },
                loss_config=loss_config_base,
                optimizer_config=optimizer_config_base,
                test_eval_suite_config=test_eval_suite_config_base,
            )
        elif case.model_type == "scaling_law":
            p_oh_config = self.make_train_config(
                self.Case(
                    model_type="p_oh",
                    training_data=case.training_data,
                    data_normalization=case.data_normalization,
                    domain_adaptation=case.domain_adaptation,
                    freeze_submodules=case.freeze_submodules,
                    num_hp_shots=case.num_hp_shots,
                )
            )
            p_rad_config = self.make_train_config(
                self.Case(
                    model_type="p_rad",
                    training_data=case.training_data,
                    data_normalization=case.data_normalization,
                    domain_adaptation=case.domain_adaptation,
                    freeze_submodules=case.freeze_submodules,
                    num_hp_shots=case.num_hp_shots,
                )
            )
            train_config_base = TrainConfig(
                project=self.wandb_project_name(case),
                train_run_builder="transport_study.modules.power_balance.trb.PowerBalanceTRB",
                max_epochs=config.max_epochs,
                epochs_per_val=config.epochs_per_val,
                checkpoint_dir=self.trained_model_dir(
                    case
                ),  # When doing hyperparameter tuning, this gets overwritten by the wandb agent
                dataloader_config={
                    "target_vars": ["Wtot_MJ"],
                    "input_vars": self._input_vars(case),
                    **dataloader_config_base,
                },
                model_init_config={
                    "model_type": case.model_type,
                    "data_normalization": case.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    "freeze_submodules": case.freeze_submodules,
                    "nn_depth": 2,
                    "nn_width": 16,
                    "in_size": 7,  # B0, Ip, R0, a_minor, kappa, ne20_line_avg, P_aux_MW
                    "out_size": 1,
                    "prng_seed": 42,
                    "submodules": {
                        "p_oh_predictor": p_oh_config,
                        "p_rad_predictor": p_rad_config,
                    },
                    "restore_submodules": True,  # Always restoring pre-trained submodules in this study
                },
                loss_config=loss_config_base,
                optimizer_config=optimizer_config_base,
                test_eval_suite_config=test_eval_suite_config_base,
            )
        elif case.model_type == "sciml":
            p_oh_config = self.make_train_config(
                self.Case(
                    model_type="p_oh",
                    training_data=case.training_data,
                    data_normalization=case.data_normalization,
                    domain_adaptation=case.domain_adaptation,
                    freeze_submodules=case.freeze_submodules,
                    num_hp_shots=case.num_hp_shots,
                )
            )
            p_rad_config = self.make_train_config(
                self.Case(
                    model_type="p_rad",
                    training_data=case.training_data,
                    data_normalization=case.data_normalization,
                    domain_adaptation=case.domain_adaptation,
                    freeze_submodules=case.freeze_submodules,
                    num_hp_shots=case.num_hp_shots,
                )
            )
            train_config_base = TrainConfig(
                project=self.wandb_project_name(case),
                train_run_builder="transport_study.modules.power_balance.trb.PowerBalanceTRB",
                max_epochs=config.max_epochs,
                epochs_per_val=config.epochs_per_val,
                checkpoint_dir=self.trained_model_dir(
                    case
                ),  # When doing hyperparameter tuning, this gets overwritten by the wandb agent
                dataloader_config={
                    "target_vars": ["Wtot_MJ"],
                    "input_vars": self._input_vars(case),
                    **dataloader_config_base,
                },
                model_init_config={
                    "model_case": case.model_type,
                    "data_normalization": case.data_normalization,
                    "freeze_submodules": case.freeze_submodules,
                    "nn_depth": 2,
                    "nn_width": 16,
                    "in_size": 7,  # B0, Ip, R0, a_minor, kappa, ne20_line_avg, P_aux_MW
                    "out_size": 1,
                    "prng_seed": 42,
                    "submodules": {
                        "p_oh_predictor": p_oh_config,
                        "p_rad_predictor": p_rad_config,
                    },
                    "restore_submodules": True,  # Always restoring pre-trained submodules in this study
                },
                loss_config=loss_config_base,
                optimizer_config=optimizer_config_base,
                test_eval_suite_config=test_eval_suite_config_base,
            )
        elif case.model_type == "unstructured_nn":
            train_config_base = TrainConfig(
                project=self.wandb_project_name(case),
                train_run_builder="transport_study.modules.power_balance.trb.PowerBalanceTRB",
                max_epochs=config.max_epochs,
                epochs_per_val=config.epochs_per_val,
                checkpoint_dir=self.trained_model_dir(
                    case
                ),  # When doing hyperparameter tuning, this gets overwritten by the wandb agent
                dataloader_config={
                    "target_vars": ["Wtot_MJ"],
                    "input_vars": self._input_vars(case),
                    **dataloader_config_base,
                },
                model_init_config={
                    "model_case": case.model_type,
                    "data_normalization": case.data_normalization,
                    "freeze_submodules": case.freeze_submodules,
                    "nn_depth": 2,
                    "nn_width": 16,
                    "in_size": 7,  # B0, Ip, R0, a_minor, kappa, ne20_line_avg, P_aux_MW
                    "out_size": 1,
                    "prng_seed": 42,
                },
                loss_config=loss_config_base,
                optimizer_config=optimizer_config_base,
                test_eval_suite_config=test_eval_suite_config_base,
            )
        else:
            raise ValueError(f"Unknown model type: {case.model_type}")

        tuned_config_path = self.tuned_config_path(case)
        if os.path.exists(tuned_config_path):
            tuned_config = TrainConfig.load(tuned_config_path)
            # Restore hyperparameters from the tuned config, but keep the rest of the settings the same

            # Hyperparameters swept for all modules
            train_config = train_config_base.model_copy(
                update={
                    "dataloader_config": {
                        **tuned_config.dataloader_config,
                        "segment_length_train": tuned_config.dataloader_config[
                            "segment_length_train"
                        ],
                        "segment_overlap_train": tuned_config.dataloader_config[
                            "segment_overlap_train"
                        ],
                        "batch_size": tuned_config.dataloader_config["batch_size"],
                    },
                    "optimizer_config": tuned_config.optimizer_config,
                }
            )

            # Hyperparameters swept for only certain modules
            if case.model_type in ["p_oh", "p_rad", "unstructured_nn", "sciml"]:
                train_config = train_config.model_copy(
                    update={
                        "model_init_config": {
                            **train_config.model_init_config,
                            "nn_depth": tuned_config.model_init_config["nn_depth"],
                            "nn_width": tuned_config.model_init_config["nn_width"],
                        }
                    }
                )

            return train_config
        else:
            return train_config_base

    def launch_sweep(self, case: Case):
        """Launch a wandb hyperparameter sweep for the given case."""
        train_config = self.make_train_config(case)
        sweep_id = get_sweep_id(self.wandb_project_name(case))

        if not sweep_id:
            logger.info(
                f"No existing sweep found for case {case}, creating a new sweep"
            )
            sweep_config_path = os.path.join(
                PACKAGE_ROOT,
                "transport_study",
                "power_balance_transfer",
                "sweep_configs",
                f"{case.model_type}.yaml",
            )
            launch_sweep(train_config, sweep_config_path)
        else:
            launch_agent(train_config, sweep_id)

    def launch_train(self, case: Case):
        """Launch a training job for the given case."""
        train_config = self.make_train_config(case)
        _, _, _, _, result_dict = launch_train(train_config)
        ds = result_dict["test/study_results"]
        result_path = self.result_path(case)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        ds.to_netcdf(result_path)


def run_study(  # noqa: PLR0915
    project_name: str,
    working_dir_base: str | None,
    model_types: list[str] | None = None,
    training_datasets: list[str] | None = None,
    data_normalization_methods: list[str] | None = None,
    domain_adaptation_methods: list[str] | None = None,
    freeze_submodules_options: list[bool] | None = None,
    num_hp_shots_options: list[int | None] | None = None,
    hp_test_set_size: int | None = None,
    enable_parallelism: bool | None = False,
    skip_tuning: bool | None = True,
    skip_visualization: bool | None = False,
    clean_sweeps: bool | None = False,
    clean_models: bool | None = False,
    clean_results: bool | None = False,
    clean_figures: bool | None = False,
):
    """
    Go from datasets to all figures in one command.
    See `popsim_transport_predictor/scripts/reproduce_figures.sh` for an example usage.

    Requires specifying paths to the source datasets in environment variables.
    Due to data sharing restrictions, the only dataset included in this repository is for C-Mod.
    If you have access to data from other tokamaks (e.g. DIII-D), create a source dataset using the scripts in `popsim_transport_predictor/datasets/`
    and provide the path when running this script.
    If a dataset is not provided for a tokamak, figures which require that data will be skipped.

    "I hardly lifted a finger" - Engi B

    Parameters
    ----------
    project_name : str | None
        Name of the project. Used to separate different runs within the working and figure directories.
    working_dir_base : str | None
        Base directory for working data. Trained models and intermediate data files will be placed in `{working_dir_base}/{project_name}`.
    figure_dir_base : str | None
        Base directory for figures. Figures will be placed in `{figure_dir_base}/{project_name}`.
    enable_parallelism : bool | None
        If false, runs the entire study sequentially in one process.
        If true, submits independent training steps with SLURM up to configurable resource limits.
        The idea is you would periodically call this 'run_study' function, and it checks what models still need to be trained and submit jobs for those, until eventually all models are trained and all results are computed.
        Not the cleanest solution, but it works.
    skip_tuning : bool | None
        If True, skip hyperparameter tuning steps.
    skip_visualization : bool | None
        If True, skip data visualization steps.
    clean_sweeps : bool | None
        If True, delete any existing wandb sweeps for this project before running.
    clean_models : bool | None
        If True, delete any existing trained models in the working directory before running.
    clean_results : bool | None
        If True, delete any existing intermediate results in the working directory before running.
    clean_figures : bool | None
        If True, delete any existing figures in the figure directory before running.
    """

    def _validate_args(
        model_types,
        training_datasets,
        data_normalization_methods,
        domain_adaptation_methods,
        freeze_submodules_options,
        num_hp_shots_options,
        hp_test_set_size,
    ):
        def _assign_args(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_submodules_options,
            num_hp_shots_options,
            hp_test_set_size,
        ):
            if model_types is None:
                model_types = ["scaling_law", "sciml", "unstructured_nn"]
            if training_datasets is None:
                training_datasets = ["cmod", "tcv", "cmod_tcv", "exnihilo"]
            if data_normalization_methods is None:
                data_normalization_methods = ["raw", "physics", "z_score", "coral"]
            if domain_adaptation_methods is None:
                domain_adaptation_methods = [None, "mixing", "transfer"]
            if freeze_submodules_options is None:
                freeze_submodules_options = [True, False]
            if num_hp_shots_options is None:
                num_hp_shots_options = [0, 1, 3, 10, 30, None]
            if hp_test_set_size is None:
                hp_test_set_size = 60

            return (
                model_types,
                training_datasets,
                data_normalization_methods,
                domain_adaptation_methods,
                freeze_submodules_options,
                num_hp_shots_options,
                hp_test_set_size,
            )

        (
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_submodules_options,
            num_hp_shots_options,
            hp_test_set_size,
        ) = _assign_args(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_submodules_options,
            num_hp_shots_options,
            hp_test_set_size,
        )

        def _check_args(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
        ):
            for model_type in model_types:
                if model_type not in ["scaling_law", "sciml", "unstructured_nn"]:
                    raise ValueError(
                        f"Invalid model type: {model_type}. Must be one of 'scaling_law', 'sciml', or 'unstructured_nn'."
                    )

            for training_dataset in training_datasets:
                if training_dataset not in ["cmod", "tcv", "cmod_tcv", "exnihilo"]:
                    raise ValueError(
                        f"Invalid training dataset: {training_dataset}. Must be one of 'cmod', 'tcv', 'cmod_tcv', or 'exnihilo'."
                    )

            for data_normalization in data_normalization_methods:
                if data_normalization not in ["raw", "physics", "z_score", "coral"]:
                    raise ValueError(
                        f"Invalid data normalization method: {data_normalization}. Must be one of 'raw', 'physics', 'z_score', or 'coral'."
                    )

            for domain_adaptation in domain_adaptation_methods:
                if domain_adaptation not in [None, "mixing", "transfer"]:
                    raise ValueError(
                        f"Invalid domain adaptation method: {domain_adaptation}. Must be one of None, 'mixing', or 'transfer'."
                    )

        _check_args(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
        )

        return (
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_submodules_options,
            num_hp_shots_options,
            hp_test_set_size,
        )

    (
        model_types,
        training_datasets,
        data_normalization_methods,
        domain_adaptation_methods,
        freeze_submodules_options,
        num_hp_shots_options,
        hp_test_set_size,
    ) = _validate_args(
        model_types,
        training_datasets,
        data_normalization_methods,
        domain_adaptation_methods,
        freeze_submodules_options,
        num_hp_shots_options,
        hp_test_set_size,
    )

    ###########################################
    # Initialize study and Set up directories #
    ###########################################
    if working_dir_base is None:
        working_dir_base = os.path.join(PACKAGE_ROOT, "popsim_studies", "working_dir")

    study = PowerBalanceStudy(
        name=project_name,
        working_dir_base=working_dir_base,
        dataset_paths={
            "cmod": config.cmod_dataset_path,
            "tcv": config.tcv_dataset_path,
            "d3d_hp": config.d3d_hp_dataset_path,
        },
        model_types=model_types,
        training_datasets=training_datasets,
        data_normalization_methods=data_normalization_methods,
        domain_adaptation_methods=domain_adaptation_methods,
        freeze_submodules_options=freeze_submodules_options,
        num_hp_shots_options=num_hp_shots_options,
        hp_test_set_size=hp_test_set_size,
    )

    def _setup_directories(study: PowerBalanceStudy):
        logger.info("SETTING UP DIRECTORIES")
        logger.info(f"Enable parallelism: {enable_parallelism}")
        logger.info(f"Skip hyperparameter tuning: {skip_tuning}")
        logger.info(f"Skip visualization: {skip_visualization}")
        logger.info(f"Clean sweeps: {clean_sweeps}")
        logger.info(f"Clean models: {clean_models}")
        logger.info(f"Clean results: {clean_results}")
        logger.info(f"Clean figures: {clean_figures}")

        if (
            clean_sweeps or clean_models or clean_results or clean_figures
        ) and enable_parallelism:
            raise ValueError(
                "Cannot clean models, results, or figures when parallelism is enabled, as this would interfere with jobs currently running or queued."
            )

        if (not skip_tuning) and (not enable_parallelism):
            logger.critical(
                "Hyperparameter tuning without parallelism enabled is probably gonna take a long time, are you sure you want to do this?"
            )

        if clean_sweeps:
            project_names = {study.wandb_project_name(case) for case in study.cases}
            run_clean_sweeps(project_names)
        if clean_models:
            shutil.rmtree(study.model_dir, ignore_errors=True)
        if clean_results:
            shutil.rmtree(study.result_dir, ignore_errors=True)
        if clean_figures:
            shutil.rmtree(study.figure_dir, ignore_errors=True)

        for directory in [study.model_dir, study.result_dir, study.figure_dir]:
            os.makedirs(directory, exist_ok=True)

    _setup_directories(study)

    ######################
    # Data Visualization #
    ######################
    if not skip_visualization:
        logger.info("DATA VISUALIZATION")

    ########################
    # Launch Orchestration #
    ########################
    logger.info("ORCHESTRATION")

    # Unfinished cases are those we have data to run but haven't gotten results for yet
    unfinished_cases = [
        case
        for case in study.cases
        if not os.path.exists(study.result_path(case))
        and study.check_data_requirements(case)
    ]
    while len(unfinished_cases) > 0:
        for case in unfinished_cases:
            study.run_case(
                case, skip_tuning=skip_tuning, enable_parallelism=enable_parallelism
            )

        # Check which cases are still unfinished
        unfinished_cases = [
            case for case in unfinished_cases if not study.check_prereq_satisfied(case)
        ]
        logger.info(f"{len(unfinished_cases)} cases remaining.")

    ############################
    # Training Data Comparison #
    ############################
    logger.info("TRAINING DATA COMPARISON")

    ####################
    # Model Comparison #
    ####################
    logger.info("MODEL COMPARISON")


if __name__ == "__main__":
    fire.Fire(
        {
            "run_study": run_study,
        }
    )
