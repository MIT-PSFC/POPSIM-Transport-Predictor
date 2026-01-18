import os

import fire
from loguru import logger

from popsim_transport_predictor import PACKAGE_ROOT
from popsim_transport_predictor.popsim_study.config import config
from popsim_transport_predictor.popsim_study.figures.data_visualization import (
    performance_extrapolation_plot,
)
from popsim_transport_predictor.popsim_study.orchestration import (
    HP_SHOTS_INCLUDED,
    MODEL_CASES,
    TRAINING_DATA_CASES,
)
from popsim_transport_predictor.popsim_study.orchestration.organize_data import (
    get_train_test_datasets_transfer,
    get_train_val_test_datasets,
)


class DataVisualization:
    """
    Visualizations of the datasets used in training and testing.
    """

    @staticmethod
    def performance_extrapolation(  # noqa: PLR0912
        figure_dir: str,
    ):
        """
        Performance is ip**2 + beta**2 <- need to formalize this metric by looking at the distribution of ip and beta separately.

        With all data present this creates the following figures:
        1. Performance extrapolation for C-Mod
        2. Performance extrapolation for TCV
        3. Performance extrapolation for DIII-D low-performance shots
        4. Performance extrapolation for C-Mod + TCV
        5. Performance extrapolation for C-Mod + TCV + DIII-D low-performance shots
        6. Showing there is no overlap in parameter space between the DIII-D low-performance shots, high-performance shots used in training, and high-performance shots used in testing
        7. Put DIII-D high-performance shots in context of all training data
        """

        save_dir = os.path.join(
            figure_dir, "data_visualization", "performance_extrapolation"
        )
        logger.info(f"Creating performance extrapolation figures in {save_dir}")

        # C-Mod
        if config.cmod_dataset_path:
            # TODO(ZanderKeith) Use the exact dataloader from the study to ensure consistency
            fig_path = os.path.join(save_dir, "cmod_performance_extrapolation.png")
            if not os.path.exists(fig_path):
                train_ds, val_ds, test_ds = get_train_val_test_datasets(
                    training_data_case="cmod",
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, val_ds, test_ds],
                    labels=["Train", "Validation", "Test"],
                    performance_metric="performance",
                    x_var="Ip_MA",
                    y_var="Wtot_MJ",
                )
        else:
            logger.warning("C-Mod dataset path not provided, skipping C-Mod figures.")

        # TCV
        if config.tcv_dataset_path:
            fig_path = os.path.join(save_dir, "tcv_performance_extrapolation.png")
            if not os.path.exists(fig_path):
                train_ds, val_ds, test_ds = get_train_val_test_datasets(
                    training_data_case="tcv",
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, val_ds, test_ds],
                    labels=["Train", "Validation", "Test"],
                    performance_metric="performance",
                    x_var="Ip_MA",
                    y_var="Wtot_MJ",
                )
        else:
            logger.warning("TCV dataset path not provided, skipping TCV figures.")

        return

        # C-Mod + TCV
        if config.tcv_dataset_path and config.cmod_dataset_path:
            fig_path = os.path.join(save_dir, "cmod_tcv_performance_extrapolation.png")
            if not os.path.exists(fig_path):
                train_ds, val_ds, test_ds = get_train_val_test_datasets(
                    training_data_case="cmod_tcv",
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, val_ds, test_ds],
                    labels=["Train", "Validation", "Test"],
                    performance_metric="performance",
                    x_var="Ip_MA",
                    y_var="Wtot_MJ",
                )
        else:
            logger.warning(
                "TCV or C-Mod dataset path not provided, skipping combined C-Mod + TCV figures."
            )

        # DIII-D low-performance
        if config.d3d_lp_dataset_path:
            fig_path = os.path.join(save_dir, "d3d_lp_performance_extrapolation.png")
            if not os.path.exists(fig_path):
                train_ds, val_ds, test_ds = get_train_val_test_datasets(
                    training_data_case="d3d_lp",
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, val_ds, test_ds],
                    labels=["Train", "Validation", "Test"],
                    performance_metric="performance",
                    x_var="Ip_MA",
                    y_var="Wtot_MJ",
                )
        else:
            logger.warning(
                "DIII-D low-performance dataset path not provided, skipping DIII-D low-performance figures."
            )

        # C-Mod + TCV + DIII-D low-performance
        if (
            config.cmod_dataset_path
            and config.tcv_dataset_path
            and config.d3d_lp_dataset_path
        ):
            fig_path = os.path.join(
                save_dir, "cmod_tcv_d3d_lp_performance_extrapolation.png"
            )
            if not os.path.exists(fig_path):
                train_ds, val_ds, test_ds = get_train_val_test_datasets(
                    training_data_case="cmod_tcv_d3d_lp",
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, val_ds, test_ds],
                    labels=["Train", "Validation", "Test"],
                    performance_metric="performance",
                    x_var="Ip_MA",
                    y_var="Wtot_MJ",
                )
        else:
            logger.warning(
                "C-Mod, TCV, or DIII-D low-performance dataset path not provided, skipping combined C-Mod + TCV + DIII-D low-performance figures."
            )

        # DIII-D performance overlap
        if config.d3d_hp_dataset_path and config.d3d_lp_dataset_path:
            fig_path = os.path.join(save_dir, "d3d_performance_overlap.png")
            if not os.path.exists(fig_path):
                train_ds, test_ds = get_train_test_datasets_transfer(
                    training_data_case="d3d_lp",
                    num_hp_shots=max(HP_SHOTS_INCLUDED),
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, test_ds],
                    labels=["Training Data", "High-Performance Test Data"],
                    performance_metric="performance",
                    x_var="Ip_MA",
                    y_var="Wtot_MJ",
                )

        # DIII-D high-performance in context of all training data
        if (
            config.d3d_hp_dataset_path
            and config.cmod_dataset_path
            and config.tcv_dataset_path
            and config.d3d_lp_dataset_path
        ):
            fig_path = os.path.join(
                save_dir, "d3d_hp_in_context_performance_extrapolation.png"
            )
            if not os.path.exists(fig_path):
                train_ds, test_ds = get_train_test_datasets_transfer(
                    training_data_case="cmod_tcv_d3d_lp",
                    num_hp_shots=max(HP_SHOTS_INCLUDED),
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, test_ds],
                    labels=["Training Data", "High-Performance Test Data"],
                    performance_metric="performance",
                    x_var="Ip_MA",
                    y_var="Wtot_MJ",
                )


class ComputeResults:
    """
    Compute results for all trained models and datasets.
    If results already exist, skip computation.
    If a model is not trained, go train it.
    """

    @staticmethod
    def _result_path(
        result_dir: str,
        training_data_case: str,
        model_case: str,
        transfer_learning: bool = False,
        num_hp_shots: int | None = None,
    ) -> str:
        if transfer_learning:
            if num_hp_shots is None:
                raise ValueError(
                    "num_hp_shots must be provided for transfer learning results."
                )
            return os.path.join(
                result_dir,
                "transfer_learning",
                training_data_case,
                model_case,
                f"hp_shots_{num_hp_shots}.nc",
            )
        else:
            return os.path.join(
                result_dir,
                "standard_learning",
                training_data_case,
                model_case,
            )

    @staticmethod
    def _trained_model_dir(
        model_dir: str,
        training_data_case: str,
        model_case: str,
    ) -> str:
        return os.path.join(model_dir, training_data_case, model_case)

    @staticmethod
    def _compute_standard_learning_result(
        trained_model_dir: str,
        result_path: str,
    ):
        """
        Compute results for standard learning.
        """

    @staticmethod
    def _check_data_requirements(
        training_data_case: str,
        transfer_learning: bool,
    ):
        """
        Check if results can be computed given the available datasets and trained models.
        """
        if transfer_learning and (config.d3d_hp_dataset_path is None):
            logger.warning(
                "DIII-D high-performance dataset path not provided, cannot compute transfer learning results."
            )
            return False

        if training_data_case in ["cmod", "cmod_tcv", "cmod_tcv_d3d_lp"] and (
            config.cmod_dataset_path is None
        ):
            logger.warning(
                "C-Mod dataset path not provided, cannot compute results for training data case "
                f"'{training_data_case}'."
            )
            return False

        if training_data_case in ["tcv", "cmod_tcv"] and (
            config.tcv_dataset_path is None
        ):
            logger.warning(
                "TCV dataset path not provided, cannot compute results for training data case "
                f"'{training_data_case}'."
            )
            return False

        if training_data_case in ["d3d_lp", "cmod_tcv_d3d_lp"] and (
            config.d3d_lp_dataset_path is None
        ):
            logger.warning(
                "DIII-D low-performance dataset path not provided, cannot compute results for training data case "
                f"'{training_data_case}'."
            )
            return False

    @staticmethod
    def standard_learning_results(
        model_dir: str,
        result_dir: str,
        training_data_cases: list[str] = TRAINING_DATA_CASES,
        model_cases: list[str] = MODEL_CASES,
    ):
        """
        Compute results with standard learning, training and testing on similar datasets.
        For this study, that means the following:
        C-Mod -> C-Mod
        TCV -> TCV
        C-Mod + TCV -> C-Mod + TCV
        DIII-D low-performance -> DIII-D low-performance
        C-Mod + TCV + DIII-D low-performance -> C-Mod + TCV + DIII-D low-performance
        """

        for training_data_case in training_data_cases:
            if training_data_case == "exnihilo":
                continue  # No training data, skip
            for model_case in model_cases:
                result_path = ComputeResults._result_path(
                    result_dir=result_dir,
                    training_data_case=training_data_case,
                    model_case=model_case,
                )
                if not os.path.exists(result_path):
                    if not ComputeResults._check_data_requirements(
                        training_data_case=training_data_case,
                        transfer_learning=False,
                    ):
                        continue

                    # Results not found, need to compute them
                    trained_model_dir = ComputeResults._trained_model_dir(
                        model_dir=model_dir,
                        training_data_case=training_data_case,
                        model_case=model_case,
                    )
                    if not os.path.exists(trained_model_dir):
                        # Model not trained, need to go do that first
                        pass

    @staticmethod
    def transfer_learning_results(
        working_dir: str,
        training_data_cases: list[str] = TRAINING_DATA_CASES,
        model_cases: list[str] = MODEL_CASES,
        hp_shots_included: list[int] = HP_SHOTS_INCLUDED,
    ):
        """
        Compute results with transfer learning, training on historic data and testing on high-performance data.
        For this study, that means the following:
        C-Mod -> DIII-D high-performance
        TCV -> DIII-D high-performance
        C-Mod + TCV -> DIII-D high-performance
        DIII-D low-performance -> DIII-D high-performance
        C-Mod + TCV + DIII-D low-performance -> DIII-D high-performance

        The number of high-performance shots included in training is varied as specified in `hp_shots_included`.
        """

        for _training_data_case in training_data_cases:
            for _model_case in model_cases:
                for _num_hp_shots in hp_shots_included:
                    pass
                    # Compute results for this combination of training data, model architecture, and number of high-performance shots
                    # If the model is not trained, train it first


class TrainingDataComparison:
    """
    Performance vs Training Datasets
    """


class ModelComparison:
    """
    Performance vs Model Architectures
    """


@staticmethod
def run_study(
    project_name: str = "popsim_transport_predictor",
    working_dir_base: str = PACKAGE_ROOT,
    figure_dir_base: str = PACKAGE_ROOT,
    clean_models: bool = False,
    clean_results: bool = False,
    clean_figures: bool = False,
):
    """
    Go from datasets to all figures in one command.
    See `popsim_transport_predictor/scripts/reproduce_figures.sh` for an example usage.

    Requires specifying paths to the source datasets.
    Due to data sharing restrictions, the only dataset included in this repository is for C-Mod.
    If you have access to data from other tokamaks (e.g. DIII-D), create a source dataset using the script at (TODO(ZanderKeith))
    and provide the path when running this script.
    If a dataset is not provided for a tokamak, figures which require that data will be skipped.

    *I hardly lifted a finger*

    Parameters
    ----------
    project_name : str
        Name of the project. Used to separate different runs within the working and figure directories.
    working_dir_base : str
        Base directory for working data. Trained models and intermediate data files will be placed in `{working_dir_base}/{project_name}`.
    figure_dir_base : str
        Base directory for figures. Figures will be placed in `{figure_dir_base}/{project_name}`.
    cmod_dataset_path : str
        Path to the C-Mod dataset file. If not provided, C-Mod figures will be skipped.
    tcv_dataset_path : str
        Path to the TCV dataset file. If not provided, TCV figures will be skipped.
    d3d_lp_dataset_path : str
        Path to the DIII-D low-performance dataset file. If not provided, DIII-D figures will be skipped.
    d3d_hp_dataset_path : str
        Path to the DIII-D high-performance dataset file. If not provided, DIII-D figures will be skipped.
    clean_models : bool
        If True, delete any existing trained models in the working directory before running.
    clean_results : bool
        If True, delete any existing intermediate results in the working directory before running.
    clean_figures : bool
        If True, delete any existing figures in the figure directory before running.
    """

    ######################
    # Set up directories #
    ######################

    working_dir = os.path.join(working_dir_base, project_name)
    model_dir = os.path.join(working_dir, "models")
    result_dir = os.path.join(working_dir, "results")
    figure_dir = os.path.join(figure_dir_base, project_name)

    if clean_models:
        os.removedirs(model_dir)
    if clean_results:
        os.removedirs(result_dir)
    if clean_figures:
        os.removedirs(figure_dir)

    for directory in [model_dir, result_dir, figure_dir]:
        os.makedirs(directory, exist_ok=True)

    ######################
    # Data Visualization #
    ######################
    logger.info("DATA VISUALIZATION")

    DataVisualization.performance_extrapolation(
        figure_dir=figure_dir,
    )

    ########################
    # Launch Orchestration #
    ########################
    logger.info("ORCHESTRATION")

    # Standard Learning Results
    ComputeResults.standard_learning_results(
        model_dir=model_dir,
        result_dir=result_dir,
    )

    # Transfer Learning Results
    ComputeResults.transfer_learning_results(
        working_dir=working_dir,
    )

    ############################
    # Training Data Comparison #
    ############################
    logger.info("TRAINING DATA COMPARISON")

    ####################
    # Model Comparison #
    ####################
    logger.info("MODEL COMPARISON")


if __name__ == "__main__":
    fire.Fire(run_study)
