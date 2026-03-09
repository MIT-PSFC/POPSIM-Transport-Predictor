import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from loguru import logger
from popsim.ml import TrainConfig


class Study(ABC):
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

    @abstractmethod
    def result_path(self, case: Case) -> str:
        """Given a case, return the path where the results for that case should be stored"""
        raise NotImplementedError

    def check_prereq_satisfied(self, case: Case) -> bool:
        """Check if the prerequisites for this case have been satisfied by looking for the existence of the result path"""
        if case.prereqs is None:
            return True
        for prereq in case.prereqs:
            prereq_result_path = self.result_path(prereq)
            if not os.path.exists(prereq_result_path):
                return False
        return True

    @abstractmethod
    def trained_model_dir(self, case: Case) -> str:
        """Given a case, return the path where the trained model checkpoints for that case should be stored"""
        raise NotImplementedError

    @abstractmethod
    def check_data_requirements(self, case: Case) -> bool:
        """Given a case, check if the required data for that case is available. If not, return False and print a message indicating what data is missing."""
        raise NotImplementedError

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
