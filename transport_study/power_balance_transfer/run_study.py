import os
import shutil
from dataclasses import dataclass
from itertools import product

import fire
from loguru import logger

from transport_study import PACKAGE_ROOT
from transport_study.config import config
from transport_study.orchestration.study import Study


class PowerBalanceStudy(Study):
    HYPERPARAM_TRAINING_DATASET = "cmod_tcv"
    HYPERPARAM_FREEZE_SUBMODULES = True
    HYPERPARAM_DATA_NORMALIZATION = "coral"

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

        def __str__(self):
            return f"case.{self.model_type}.td_{self.training_data}.dn_{self.data_normalization}.da_{self.domain_adaptation}.freezesub_{self.freeze_submodules}.hp_{self.num_hp_shots})"

        model_type: str  # scaling_law, sciml, unstructured_nn
        training_data: str  # cmod, tcv, cmod_tcv, exnihilo
        data_normalization: str  # raw, physics, z_score, coral
        domain_adaptation: str  # none, mixing, transfer
        freeze_submodules: bool
        weight_submodules: dict[str, float] | None = (
            None  # If not None, the submodule predictions get weighted according to this value
        )
        num_hp_shots: int | None = (
            None  # Number of high-performance shots included in training, or None for all (also None if domain_adaptation is None)
        )
        # So that's 3 (model type) x 4 (training data) x 4 (normalization) x 3 (domain adaptation) x 2 (freeze or not) x 6 (hp shots included) = 1728 results
        # Even less since the hyperparameter tuning is only done for a subset of cases
        prereq: Study.Case | None = (
            None  # If not None, this case depends on the results of another case, and should only be run after that case has been run
        )

        def __hash__(self):
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

    def _get_hyperparam_prereq(self, model_type: str) -> Case:
        """Get the hyperparameter tuning case that this case depends on,
        which is the case with the same model type (same architecture) but with
        values set by the HYPERPARAM_ constants
        """
        hyperparam_case = self.Case(
            model_type=model_type,
            training_data=self.HYPERPARAM_TRAINING_DATASET,
            data_normalization=self.HYPERPARAM_DATA_NORMALIZATION,
            domain_adaptation=None,
            freeze_submodules=self.HYPERPARAM_FREEZE_SUBMODULES,
        )
        return hyperparam_case

    def _get_transfer_prereq(
        self,
        model_type: str,
        training_data: str,
        data_normalization: str,
        freeze_submodules: bool,
    ) -> Case:
        """Get the transfer learning case that this case depends on,
        which is the case with the same model type, training data, data normalization, and freeze_submodules setting, but with no domain adaptation yet (train and test on same device)
        """
        is_hyperparam = (
            training_data == self.HYPERPARAM_TRAINING_DATASET
            and data_normalization == self.HYPERPARAM_DATA_NORMALIZATION
            and freeze_submodules == self.HYPERPARAM_FREEZE_SUBMODULES
        )
        transfer_case = self.Case(
            model_type=model_type,
            training_data=training_data,
            data_normalization=data_normalization,
            domain_adaptation=None,
            freeze_submodules=freeze_submodules,
            prereq=None if is_hyperparam else self._get_hyperparam_prereq(model_type),
        )
        return transfer_case

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
            # A few special cases to consider:
            # 1. Minimize the number of hyperparameter tuning runs
            # - We want to investigate the differences between power balance architectures, not between P_oh and P_rad architectures
            # - We want to investigate the differences between data preprocessing and domain adaptation, so they should have the same hyperparameters
            # - Things of that nature
            if domain_adaptation is None and num_hp_shots is None:
                # This is the case where we train and test on the same device with all available data,
                # so we aren't using high-performance data anyway and only need to add these cases to the list once
                if (
                    training_dataset == self.HYPERPARAM_TRAINING_DATASET
                    and freeze_submodules == self.HYPERPARAM_FREEZE_SUBMODULES
                    and data_normalization == self.HYPERPARAM_DATA_NORMALIZATION
                ):
                    # So, we only do hyperparameter tuning for a specific subset of cases, and then use those hyperparameters for all others.
                    case = self.Case(
                        model_type=model_type,
                        training_data=training_dataset,
                        data_normalization=data_normalization,
                        domain_adaptation=None,
                        freeze_submodules=freeze_submodules,
                    )
                else:
                    case = self.Case(
                        model_type=model_type,
                        training_data=training_dataset,
                        data_normalization=data_normalization,
                        domain_adaptation=domain_adaptation,
                        freeze_submodules=freeze_submodules,
                        num_hp_shots=num_hp_shots,
                        prereq=self._get_hyperparam_prereq(model_type),
                    )
            # 2. For transfer learning, we need to already have a trained model first
            elif domain_adaptation == "transfer":
                case = self.Case(
                    model_type=model_type,
                    training_data=training_dataset,
                    data_normalization=data_normalization,
                    domain_adaptation=domain_adaptation,
                    freeze_submodules=freeze_submodules,
                    num_hp_shots=num_hp_shots,
                    prereq=self._get_transfer_prereq(
                        model_type,
                        training_dataset,
                        data_normalization,
                        freeze_submodules,
                    ),
                )
            # 3. For mixing, we only need the hyperparameter tuning to be done already
            elif domain_adaptation == "mixing":
                case = self.Case(
                    model_type=model_type,
                    training_data=training_dataset,
                    data_normalization=data_normalization,
                    domain_adaptation=domain_adaptation,
                    freeze_submodules=freeze_submodules,
                    num_hp_shots=num_hp_shots,
                    prereq=self._get_hyperparam_prereq(model_type),
                )
            else:
                continue

            cases.append(case)

        return cases

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
        debug: bool = False,
    ):
        cases = self.make_cases(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_submodules_options,
            num_hp_shots_options,
        )
        super().__init__(name, working_dir_base, dataset_paths, cases, debug)

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

    ###########
    # PATHING #
    ###########
    def result_path(self, case: Case) -> str:
        """Given a case, return the path where the results for that case should be stored."""
        return os.path.join(self.result_dir, str(case), "eval_data.nc")

    def trained_model_dir(self, case: Case) -> str:
        """Given a case, return the path where the trained model checkpoints for that case should be stored."""
        return os.path.join(self.model_dir, str(case))

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

    def check_prereq_satisfied(self, case: Case) -> bool:
        """Check if the prerequisites for this case have been satisfied by looking for the existence of the result path"""
        if case.prereq is None:
            return True
        prereq_result_path = self.result_path(case.prereq)
        return os.path.exists(prereq_result_path)

    def run_case(
        self,
        case: Study.Case,
    ):
        """Run a single case of the study, including hyperparameter tuning, training, and evaluation as needed.

        If case or a prereq is in progress, simply return and let orchestration loop try again later.
        """
        if not self.check_data_requirements(case):
            return

        if self.check_prereq_satisfied(case):
            if case.prereq is None:
                # No prerequisite case, so we know it's a hyperparameter tuning case
                logger.info(f"Running hyperparameter tuning for case: {case}")
                # do some necessary stuff and return if we're not done yet

            logger.info(f"Running case: {case}")
            if config.dry_run:
                result_path = self.result_path(case)
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                with open(result_path, "w") as f:
                    f.write("This is a dummy result file for dry run.")
        else:
            logger.debug(
                f"Prerequisite for case {case} not satisfied yet,\nrunning prerequisite case {case.prereq} first."
            )
            self.run_case(case.prereq)


def run_study(  # noqa: PLR0915
    project_name: str,
    working_dir_base: str | None,
    enable_parallelism: bool | None = False,
    model_types: list[str] | None = None,
    training_datasets: list[str] | None = None,
    data_normalization_methods: list[str] | None = None,
    domain_adaptation_methods: list[str] | None = None,
    freeze_submodules_options: list[bool] | None = None,
    num_hp_shots_options: list[int | None] | None = None,
    clean_models: bool | None = False,
    clean_results: bool | None = False,
    clean_figures: bool | None = False,
    skip_visualization: bool | None = False,
    skip_tuning: bool | None = False,
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
        I would *like* to develop a better way of doing this, but for now it's straightforward for me to set up and execute and we have an experiment scheduled in 2 weeks so I gotta move fast.
    clean_models : bool | None
        If True, delete any existing trained models in the working directory before running.
    clean_results : bool | None
        If True, delete any existing intermediate results in the working directory before running.
    clean_figures : bool | None
        If True, delete any existing figures in the figure directory before running.
    skip_visualization : bool | None
        If True, skip data visualization steps.
    skip_tuning : bool | None
        If True, skip hyperparameter tuning steps.
    """

    def _validate_args(
        model_types,
        training_datasets,
        data_normalization_methods,
        domain_adaptation_methods,
        freeze_submodules_options,
        num_hp_shots_options,
    ):
        def _assign_args(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_submodules_options,
            num_hp_shots_options,
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

            return (
                model_types,
                training_datasets,
                data_normalization_methods,
                domain_adaptation_methods,
                freeze_submodules_options,
                num_hp_shots_options,
            )

        (
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_submodules_options,
            num_hp_shots_options,
        ) = _assign_args(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_submodules_options,
            num_hp_shots_options,
        )

        def _check_args(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_submodules_options,
            num_hp_shots_options,
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
            freeze_submodules_options,
            num_hp_shots_options,
        )

        return (
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_submodules_options,
            num_hp_shots_options,
        )

    (
        model_types,
        training_datasets,
        data_normalization_methods,
        domain_adaptation_methods,
        freeze_submodules_options,
        num_hp_shots_options,
    ) = _validate_args(
        model_types,
        training_datasets,
        data_normalization_methods,
        domain_adaptation_methods,
        freeze_submodules_options,
        num_hp_shots_options,
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
    )

    def _setup_directories(study: PowerBalanceStudy):
        logger.info("SETTING UP DIRECTORIES")
        logger.info(f"Enable parallelism: {enable_parallelism}")
        logger.info(f"Clean models: {clean_models}")
        logger.info(f"Clean results: {clean_results}")
        logger.info(f"Clean figures: {clean_figures}")
        logger.info(f"Skip visualization: {skip_visualization}")

        if (clean_models or clean_results or clean_figures) and enable_parallelism:
            raise ValueError(
                "Cannot clean models, results, or figures when parallelism is enabled, as this could interfere with jobs currently running or queued."
            )

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
            study.run_case(case)

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
