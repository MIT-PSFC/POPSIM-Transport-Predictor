import os
import shutil
import time
from dataclasses import dataclass
from itertools import product

import fire
import netCDF4  # noqa: F401
from loguru import logger

from transport_study import PACKAGE_ROOT
from transport_study.config import config
from transport_study.orchestration.study import Study
from transport_study.orchestration.wandb_utils import (
    run_clean_sweeps,
)


class ProfileStudy(Study):
    HYPERPARAM_TRAINING_DATA = "cmod_tcv"
    HYPERPARAM_DATA_NORMALIZATION = "raw"
    HYPERPARAM_DOMAIN_ADAPTATION = None
    HYPERPARAM_FREEZE_SHAPES = True
    HYPERPARAM_NUM_HP_SHOTS = -1

    ##################
    # INITIALIZATION #
    ##################
    @dataclass
    class Case(Study.Case):
        """
        model_type: The type of profile_predictor model to use
        - shape_init: Use principal component analysis to determine dominant shapes
        - unstructured_nn: a single neural network directly predicts profiles at certain points

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

        freeze_shapes:
        - some profile predictors first use PCA to identify dominant shapes. these shapes may be frozen or modified during module training

        num_hp_shots: The number of high-performance shots included in the training data, or -1 to include all high-performance shots (including all shots in training is cheating, but again answers the question of what is the best possible performance).
        """

        model_type: str
        training_data: str  # cmod, tcv, cmod_tcv, exnihilo
        data_normalization: str  # raw, physics, z_score, coral
        domain_adaptation: str  # none, mixing, transfer
        freeze_shapes: bool
        num_hp_shots: int  # Number of high-performance shots included in training, or -1 for all (should be -1 if domain_adaptation is None)
        prereqs: (
            list[Study.Case] | None
        )  # If not None, this case depends on the results of another case, and should only be run after that case has been run

        def is_hyperparam_case(self) -> bool:
            if (
                self.training_data == ProfileStudy.HYPERPARAM_TRAINING_DATA
                and self.data_normalization
                == ProfileStudy.HYPERPARAM_DATA_NORMALIZATION
                and self.domain_adaptation == ProfileStudy.HYPERPARAM_DOMAIN_ADAPTATION
                and self.freeze_shapes == ProfileStudy.HYPERPARAM_FREEZE_SHAPES
                and self.num_hp_shots == ProfileStudy.HYPERPARAM_NUM_HP_SHOTS
            ):
                return True
            else:
                return False

        def is_impossible(self) -> bool:
            """Some cases don't make sense to run. Mark those cases as impossible and raise an error if we try to run them."""
            # Can't do transfer learning or training from nothing with 0 high-performance shots.
            if (
                self.domain_adaptation == "transfer" or self.training_data == "exnihilo"
            ) and self.num_hp_shots == 0:
                return True

            return False

        def get_hyperparam_prereq(self) -> Study.Case:
            if self.is_hyperparam_case():
                return self
            else:
                return ProfileStudy.Case(
                    model_type=self.model_type,
                    training_data=ProfileStudy.HYPERPARAM_TRAINING_DATA,
                    data_normalization=ProfileStudy.HYPERPARAM_DATA_NORMALIZATION,
                    domain_adaptation=ProfileStudy.HYPERPARAM_DOMAIN_ADAPTATION,
                    freeze_shapes=ProfileStudy.HYPERPARAM_FREEZE_SHAPES,
                    num_hp_shots=ProfileStudy.HYPERPARAM_NUM_HP_SHOTS,
                )

        def __init__(
            self,
            model_type: str,
            training_data: str,
            data_normalization: str,
            domain_adaptation: str,
            freeze_shapes: bool,
            num_hp_shots: int,
        ):
            self.model_type = model_type
            self.training_data = training_data
            self.data_normalization = data_normalization
            self.domain_adaptation = domain_adaptation
            self.freeze_shapes = freeze_shapes
            self.num_hp_shots = num_hp_shots

            # Recursively add prereqs based on the logic of which cases depend on which other cases
            if model_type not in [
                "shape_init",
                "unstructured_nn",
            ]:
                raise ValueError(f"Unknown model type: {model_type}")
            if domain_adaptation is None and num_hp_shots != -1:
                raise ValueError(
                    "If domain_adaptation is None, num_hp_shots must be -1 since this means we're training and testing on the same dataset and no high-performance data is being used"
                )

            prereqs = []

            # Add prereqs based on whether this is a hyperparameter tuning case or not.
            if not self.is_hyperparam_case():
                prereqs += [
                    ProfileStudy.Case(
                        model_type=model_type,
                        training_data=ProfileStudy.HYPERPARAM_TRAINING_DATA,
                        data_normalization=ProfileStudy.HYPERPARAM_DATA_NORMALIZATION,
                        domain_adaptation=ProfileStudy.HYPERPARAM_DOMAIN_ADAPTATION,
                        freeze_shapes=ProfileStudy.HYPERPARAM_FREEZE_SHAPES,
                        num_hp_shots=ProfileStudy.HYPERPARAM_NUM_HP_SHOTS,
                    )
                ]

            # Set prereqs based on model type
            # This shouldn't need to happen, since all the profile predictors don't have submodules

            # Set prereqs based on domain adaptation
            if domain_adaptation == "transfer":
                prereqs += [
                    ProfileStudy.Case(
                        model_type=model_type,
                        training_data=training_data,
                        data_normalization=data_normalization,
                        domain_adaptation=None,
                        freeze_shapes=freeze_shapes,
                        num_hp_shots=-1,
                    )
                ]

            if len(prereqs) > 0:
                self.prereqs = prereqs
            else:
                self.prereqs = None

        def __str__(self):
            if self.domain_adaptation:
                return f"case.{self.model_type}.td_{self.training_data}.dn_{self.data_normalization}.da_{self.domain_adaptation}.freeze_{self.freeze_shapes}.hp_{self.num_hp_shots}"
            else:
                return f"case.{self.model_type}.td_{self.training_data}.dn_{self.data_normalization}.freeze_{self.freeze_shapes}"

        def __hash__(self):
            if self.domain_adaptation:
                return hash(
                    (
                        self.model_type,
                        self.training_data,
                        self.data_normalization,
                        self.domain_adaptation,
                        self.freeze_shapes,
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
                        self.freeze_shapes,
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
            if domain_adaptation is None and num_hp_shots != -1:
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
        possible_cases = [
            case for case in unique_cases if not case.is_impossible()
        ]  # Remove impossible cases

        return possible_cases


def run_study(  # noqa: PLR0915
    project_name: str,
    working_dir_base: str | None,
    model_types: list[str] | None = None,
    training_datasets: list[str] | None = None,
    data_normalization_methods: list[str] | None = None,
    domain_adaptation_methods: list[str] | None = None,
    freeze_shapes_options: list[bool] | None = None,
    num_hp_shots_options: list[int] | None = None,
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
    See `transport_study/scripts/reproduce_figures.sh` for an example usage.

    Requires specifying paths to the source datasets in environment variables.
    Due to data sharing restrictions, the only dataset included in this repository is for C-Mod.
    If you have access to data from other tokamaks (e.g. DIII-D), create a source dataset using the scripts in `transport_study/datasets/`
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
        freeze_shapes_options,
        num_hp_shots_options,
        hp_test_set_size,
    ):
        def _assign_args(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_shapes_options,
            num_hp_shots_options,
            hp_test_set_size,
        ):
            if model_types is None:
                model_types = ["shape_init", "unstructured_nn"]
            if training_datasets is None:
                training_datasets = ["cmod", "tcv", "cmod_tcv", "exnihilo"]
            if data_normalization_methods is None:
                data_normalization_methods = ["raw", "physics", "z_score", "coral"]
            if domain_adaptation_methods is None:
                domain_adaptation_methods = [None, "mixing", "transfer"]
            if freeze_shapes_options is None:
                freeze_shapes_options = [True, False]
            if num_hp_shots_options is None:
                num_hp_shots_options = [0, 1, 3, 10, 32, -1]
            if hp_test_set_size is None:
                hp_test_set_size = 65

            return (
                model_types,
                training_datasets,
                data_normalization_methods,
                domain_adaptation_methods,
                freeze_shapes_options,
                num_hp_shots_options,
                hp_test_set_size,
            )

        (
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_shapes_options,
            num_hp_shots_options,
            hp_test_set_size,
        ) = _assign_args(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_shapes_options,
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
            freeze_shapes_options,
            num_hp_shots_options,
            hp_test_set_size,
        )

    (
        model_types,
        training_datasets,
        data_normalization_methods,
        domain_adaptation_methods,
        freeze_shapes_options,
        num_hp_shots_options,
        hp_test_set_size,
    ) = _validate_args(
        model_types,
        training_datasets,
        data_normalization_methods,
        domain_adaptation_methods,
        freeze_shapes_options,
        num_hp_shots_options,
        hp_test_set_size,
    )

    ###########################################
    # Initialize study and Set up directories #
    ###########################################
    if working_dir_base is None:
        working_dir_base = os.path.join(PACKAGE_ROOT, "popsim_studies", "working_dir")

    study = ProfileStudy(
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
        freeze_shapes_options=freeze_shapes_options,
        num_hp_shots_options=num_hp_shots_options,
        hp_test_set_size=hp_test_set_size,
    )

    def _setup_directories(study: ProfileStudy):
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

    def _move_data(study):
        logger.info("Moving data to cluster scratch for faster training")

    _move_data(study)

    ######################
    # Data Visualization #
    ######################
    if not skip_visualization:
        logger.opt(colors=True).info(
            "<bold><magenta>DATA VISUALIZATION</magenta></bold>"
        )

    ########################
    # Launch Orchestration #
    ########################
    if os.path.exists(study.collected_results_path()):
        logger.info(
            f"Collected results file found at\n{study.collected_results_path()}\nSkipping orchestration and going straight to analysis and visualization"
        )
    else:
        logger.opt(colors=True).info("<bold><magenta>ORCHESTRATION</magenta></bold>")

        # Unfinished cases are those we have data to run but haven't gotten results for yet
        unfinished_cases = [
            case
            for case in study.cases
            if not os.path.exists(study.result_path(case))
            and study.check_data_requirements(case)
        ]
        while len(unfinished_cases) > 0:
            logger.opt(colors=True).info(
                f"<<bold><green>{len(unfinished_cases)} cases remain</green></bold>>"
            )
            for case in unfinished_cases:
                # TODO(ZanderKeith): Duplicates are happening somehow, but going fast
                if not os.path.exists(study.result_path(case)):
                    study.run_case(
                        case,
                        skip_tuning=skip_tuning,
                        enable_parallelism=enable_parallelism,
                    )

            # Check which cases are still unfinished
            unfinished_cases = [
                case
                for case in unfinished_cases
                if not os.path.exists(study.result_path(case))
            ]
            # Sleep for a bit before checking again to avoid spamming slurm
            time.sleep(8)

        ds_final = study.collect_results()
        ds_final.to_netcdf(study.collected_results_path())

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
