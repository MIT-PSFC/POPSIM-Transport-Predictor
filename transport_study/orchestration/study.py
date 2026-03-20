import os
import time
from dataclasses import dataclass

import yaml
from loguru import logger
from popsim.ml import DataLoader, TrainConfig, Trainer
from popsim.ml.launch import _get_train_run_builder_class

from transport_study.config import config
from transport_study.orchestration.slurm_utils import (
    count_running_jobs,
    resources_available,
)
from transport_study.orchestration.wandb_utils import (
    get_completed_runs,
)


class Study:
    """A class for organizing various components of a study, essentially outlining everything that needs to be done
    to go from raw data to comparison figures.
    - Paths to source data
    - Model checkpoints
    - Results
    """

    @dataclass
    class Case:
        """A class for organizing the different cases we want to compare in this study.
        For example, different treatments of the transport predictor module, different training datasets, different normalization methods, etc.
        Each case should have all the information needed to train and evaluate a model for that case, and to compare it to other cases.
        """

        def __hash__(self):
            return hash(
                tuple(
                    v
                    for k, v in self.__dict__.items()
                    if k not in ("prereq", "weight_submodules")
                )
            )

    ####################
    # PATHING / NAMING #
    ####################
    def trained_model_dir(self, case: Case) -> str:
        """Given a case, return the path where the trained model checkpoints for that case should be stored."""
        return os.path.join(self.model_dir, str(case))

    def result_path(self, case: Case) -> str:
        """Given a case, return the path where the results for that case should be stored."""
        return os.path.join(self.result_dir, str(case), "result_data.nc")

    def collected_results_path(self) -> str:
        """Return the path where the collected results for all cases should be stored."""
        return os.path.join(self.result_dir, "collected_results.nc")

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

    def check_prereq_satisfied(self, case: Case) -> bool:
        """Check if the prerequisites for this case have been satisfied by looking for the existence of the result path"""
        if case.prereqs is None:
            return True
        for prereq in case.prereqs:
            prereq_result_path = self.result_path(prereq)
            if not os.path.exists(prereq_result_path):
                return False
        return True

    def __init__(
        self,
        name: str,
        working_dir_base: str,
        dataset_paths: dict[str, str],
        cases: list[Case],
        hp_test_set_size: int,
    ):
        """
        Initialize this study with the given name, dataset paths, and cases.
        """
        self.name = name
        self.dataset_paths = dataset_paths
        self.cases = cases
        self.hp_test_set_size = hp_test_set_size

        self.working_dir = os.path.join(working_dir_base, name)
        self.model_dir = os.path.join(self.working_dir, "models")
        self.result_dir = os.path.join(self.working_dir, "results")
        self.figure_dir = os.path.join(self.working_dir, "figures")

        log_path = os.path.join(
            self.working_dir, "logs", f"{os.getpid()}_run_study.log"
        )
        logger.add(log_path)

        logger.info("INITIALIZING STUDY")
        logger.info(f"Study name: {name}")
        logger.info(f"Working directory base: {working_dir_base}")
        logger.info(f"Total number of cases: {len(cases)}")
        logger.info(f"High-performance test set size: {hp_test_set_size}")

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
        case: Case,
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

        if os.path.exists(self.result_path(case)):
            logger.warning(f"Case {case} already has results, skipping.")
            return

        if self.check_prereq_satisfied(case):
            logger.opt(colors=True).info(f"<bold>RUNNING CASE:</bold>\n{case}")
            # Prereq is satisfied, can run this case.
            if case.is_hyperparam_case():
                if skip_tuning:
                    logger.info("Skipping hyperparameter tuning")
                    # Copy default config for this module and put it in the trained model dir so the rest of the workflow can find it
                    default_config = self.make_train_config(case)
                    tuned_config_path = self.tuned_config_path(case)
                    os.makedirs(os.path.dirname(tuned_config_path), exist_ok=True)
                    with open(tuned_config_path, "w") as f:
                        yaml.dump(default_config.model_dump(), f, indent=4)
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
                if running_jobs > 0:
                    logger.info(
                        f"Found {running_jobs} running training jobs, waiting for them to complete before proceeding"
                    )
                    return
                if not resources_available():
                    logger.info(
                        "No resources currently available, waiting before trying again..."
                    )
                    time.sleep(10)
                    return

            self.launch_train(case, enable_parallelism=enable_parallelism)

        else:
            for prereq in case.prereqs:
                if not os.path.exists(self.result_path(prereq)):
                    logger.debug(
                        f"Prereq not satisfied yet, running that first.\nCase:\t{case}\nPrereq:\t{prereq}"
                    )
                    self.run_case(
                        prereq,
                        skip_tuning=skip_tuning,
                        enable_parallelism=enable_parallelism,
                    )
                    return

    ##############
    # COLLECTION #
    ##############
    def restore_trainer(
        self, case: Case, restore_best_checkpoint: bool = True
    ) -> tuple[Trainer, DataLoader]:
        """Restore a given case's trainer and the test dataloader"""
        if not os.path.exists(self.trained_model_dir(case)):
            raise ValueError(
                f"Trained model directory for case\n{case}\nnot found at\n{self.trained_model_dir(case)}"
            )
        if not os.path.exists(self.result_path(case)):
            logger.warning(
                f"Result file for case\n{case}\nnot found at\n{self.result_path(case)}\nTraining may be incomplete!"
            )
        training_config = self.make_train_config(case)
        train_run_builder = _get_train_run_builder_class(
            training_config.train_run_builder
        )
        # If this is a submodule, use the dataloader construction logic from the main module. Fallback to using the submodule's own logic otherwise.
        if training_config.dataloader_config.get("data_train_run_builder"):
            data_train_run_builder = _get_train_run_builder_class(
                training_config.dataloader_config["data_train_run_builder"]
            )
            _, train_dl, _val_dl, test_dl = data_train_run_builder.get_dataloaders(
                training_config.dataloader_config
            )
        else:
            _, train_dl, _val_dl, test_dl = train_run_builder.get_dataloaders(
                training_config.dataloader_config
            )
        model = train_run_builder.model_init(
            train_dl, training_config.model_init_config
        )
        loss_fn = train_run_builder.get_loss_fn(training_config.loss_config)
        opt = train_run_builder.get_optimizer(training_config.optimizer_config)
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=opt,
            checkpoint_dir=training_config.checkpoint_dir,
            trainable_getter=train_run_builder.get_trainable_getter(
                training_config.model_init_config
            ),
        )
        if restore_best_checkpoint:
            trainer.restore_best_checkpoint(path=self.trained_model_dir(case))

        return trainer, test_dl


def update_submodule_configs(main_config: dict, submodules: list[str]) -> TrainConfig:
    new_submodule_configs = {}
    for submodule in submodules:
        submodule_config = main_config["model_init_config"]["submodules"][submodule]
        if not isinstance(submodule_config, dict):
            submodule_config = submodule_config.model_dump()

        # Set the data_train_run_builder for the submodules to match the main module's train_run_builder.
        submodule_config["dataloader_config"]["data_train_run_builder"] = main_config[
            "train_run_builder"
        ]

        # Ensure there is a perfect match between the dataloader configs of the main module and the submodules,
        # excepting the state_vars, input_vars, target_vars, and extra_vars which are specific to each submodule.
        for key in main_config["dataloader_config"].keys():
            if key not in ["state_vars", "input_vars", "target_vars", "extra_vars"]:
                submodule_config["dataloader_config"][key] = main_config[
                    "dataloader_config"
                ][key]

        new_submodule_configs[submodule] = submodule_config

    main_config["model_init_config"]["submodules"] = new_submodule_configs

    return TrainConfig(**main_config)
