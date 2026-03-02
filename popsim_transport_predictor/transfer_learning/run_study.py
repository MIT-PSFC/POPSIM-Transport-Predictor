import os
import shutil
from itertools import product

import fire
import xarray as xr
from loguru import logger

from popsim_transport_predictor import PACKAGE_ROOT
from popsim_transport_predictor.transfer_learning.config import config
from popsim_transport_predictor.transfer_learning.figures.data_visualization import (
    domain_plot,
    performance_extrapolation_plot,
    transfer_learning_losses,
)
from popsim_transport_predictor.transfer_learning.orchestration import (
    DOMAIN_NORMALIZATION_METHODS,
    HP_SHOTS_INCLUDED,
    MODEL_CASES,
    TRAINING_DATA_CASES,
)
from popsim_transport_predictor.transfer_learning.orchestration.organize_data import (
    get_train_test_datasets_transfer,
    get_train_val_datasets,
)
from popsim_transport_predictor.transfer_learning.orchestration.train import (
    train_model_standard,
    train_model_transfer,
)


class DataVisualization:
    """
    Visualizations of the datasets used in training and testing.
    """

    def _get_largest_dataset_case():
        # Determine the biggest dataset we can use so that we only need to make one plot
        # for the domain overlap visualization. Want to only do this once since it's expensive.
        if (
            config.cmod_dataset_path
            and config.tcv_dataset_path
            and config.d3d_lp_dataset_path
        ):
            training_data_case = "cmod_tcv_d3d_lp"
        elif (
            config.cmod_dataset_path
            and config.tcv_dataset_path
            and config.d3d_lp_dataset_path
        ):
            training_data_case = "cmod_tcv_d3d_lp"
        elif config.cmod_dataset_path and config.tcv_dataset_path:
            training_data_case = "cmod_tcv"
        elif config.cmod_dataset_path:
            training_data_case = "cmod"
        elif config.tcv_dataset_path:
            training_data_case = "tcv"
        elif config.d3d_lp_dataset_path:
            training_data_case = "d3d_lp"
        else:
            raise ValueError(
                "No dataset paths provided in config, cannot determine largest dataset case for domain overlap plot."
            )

        return training_data_case

    @staticmethod
    def performance_extrapolation(  # noqa: PLR0912
        figure_dir: str,
    ):
        """
        Performance is ip**2 + Wtot_MJ**2

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

        # C-Mod
        if config.cmod_dataset_path:
            fig_path = os.path.join(save_dir, "cmod_performance_extrapolation.png")
            if not os.path.exists(fig_path):
                train_ds, val_ds = get_train_val_datasets(
                    training_data_case="cmod",
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, val_ds],
                    ds_type_list=["train", "val"],
                    x_var="Ip_MA",
                    y_var="Wtot_MJ",
                )
        else:
            logger.warning("C-Mod dataset path not provided, skipping C-Mod figures.")

        # TCV
        if config.tcv_dataset_path:
            fig_path = os.path.join(save_dir, "tcv_performance_extrapolation.png")
            if not os.path.exists(fig_path):
                train_ds, val_ds = get_train_val_datasets(
                    training_data_case="tcv",
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, val_ds],
                    ds_type_list=["train", "val"],
                    x_var="Ip_MA",
                    y_var="Wtot_MJ",
                )
        else:
            logger.warning("TCV dataset path not provided, skipping TCV figures.")

        # C-Mod + TCV
        if config.tcv_dataset_path and config.cmod_dataset_path:
            fig_path = os.path.join(save_dir, "cmod_tcv_performance_extrapolation.png")
            if not os.path.exists(fig_path):
                train_ds, val_ds = get_train_val_datasets(
                    training_data_case="cmod_tcv",
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, val_ds],
                    ds_type_list=["train", "val"],
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
                train_ds, val_ds = get_train_val_datasets(
                    training_data_case="d3d_lp",
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, val_ds],
                    ds_type_list=["train", "val"],
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
                train_ds, val_ds = get_train_val_datasets(
                    training_data_case="cmod_tcv_d3d_lp",
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, val_ds],
                    ds_type_list=["train", "val"],
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
                    ds_type_list=["train", "test"],
                    x_var="Ip_MA",
                    y_var="Wtot_MJ",
                )

        # DIII-D high-performance in context of available training data
        context_dict = {
            "cmod": {
                "condition": config.cmod_dataset_path,
                "fig_name": "d3d_hp_in_context_cmod_performance_extrapolation.png",
            },
            "tcv": {
                "condition": config.tcv_dataset_path,
                "fig_name": "d3d_hp_in_context_tcv_performance_extrapolation.png",
            },
            "d3d_lp": {
                "condition": config.d3d_lp_dataset_path,
                "fig_name": "d3d_hp_in_context_d3d_lp_performance_extrapolation.png",
            },
            "cmod_tcv": {
                "condition": config.cmod_dataset_path and config.tcv_dataset_path,
                "fig_name": "d3d_hp_in_context_cmod_tcv_performance_extrapolation.png",
            },
            "cmod_tcv_d3d_lp": {
                "condition": config.cmod_dataset_path
                and config.tcv_dataset_path
                and config.d3d_lp_dataset_path,
                "fig_name": "d3d_hp_in_context_cmod_tcv_d3d_lp_performance_extrapolation.png",
            },
        }

        for context_case, context_info in context_dict.items():
            if context_info["condition"] and config.d3d_hp_dataset_path:
                fig_path = os.path.join(save_dir, context_info["fig_name"])
                if not os.path.exists(fig_path):
                    train_ds, test_ds = get_train_test_datasets_transfer(
                        training_data_case=context_case,
                        num_hp_shots=max(HP_SHOTS_INCLUDED),
                    )
                    performance_extrapolation_plot(
                        save_path=fig_path,
                        ds_list=[train_ds, test_ds],
                        ds_type_list=["train", "test"],
                        x_var="Ip_MA",
                        y_var="Wtot_MJ",
                    )

    @staticmethod
    def domain_overlap(
        figure_dir: str,
    ):
        """
        Compare different data preparation cases to how the parameter space overlaps.
        This is different from the performance extrapolation plots because here it is desirable to have a lot of overlap.
        While we are ALWAYS extrapolating in real units (Ip and Wtot, things that WILL break the device)
        first normalizing the data should help with transfer learning.

        Basically, this normalization doesn't impact the transfer learning, because the dataset is being split into train and val/test beforehand.
        """

        training_data_case = DataVisualization._get_largest_dataset_case()

        for method in DOMAIN_NORMALIZATION_METHODS:
            if method == "raw":
                var_groups = [
                    ["Ip_MA", "Wtot_MJ"],
                    ["R0", "a_minor"],
                    ["ne20_line_avg", "B0"],
                    ["P_aux_MW", "kappa"],
                ]
            elif method == "physics":
                var_groups = [
                    ["Ip_MA", "beta"],
                    ["q_star", "epsilon"],
                    ["f_G", "aB0"],
                    ["surface_power_density", "kappa"],
                ]
            elif method == "z_score":
                var_groups = [
                    ["Ip_MA_z", "Wtot_MJ_z"],
                    ["R0_z", "a_minor_z"],
                    ["ne20_line_avg_z", "B0_z"],
                    ["P_aux_MW_z", "kappa_z"],
                ]
            elif method == "coral":
                var_groups = [
                    ["Ip_MA_coral", "Wtot_MJ_coral"],
                    ["R0_coral", "a_minor_coral"],
                    ["ne20_line_avg_coral", "B0_coral"],
                    ["P_aux_MW_coral", "kappa_coral"],
                ]
            else:
                raise ValueError(f"Unknown normalization method '{method}' specified.")

            if config.d3d_hp_dataset_path:
                ds, _ = get_train_test_datasets_transfer(
                    training_data_case=training_data_case,
                    num_hp_shots=max(HP_SHOTS_INCLUDED),
                    normalization_method=method,
                )
            else:
                ds, _ = get_train_val_datasets(
                    training_data_case=training_data_case, normalization_method=method
                )

            fig_path = os.path.join(
                figure_dir,
                "domain_overlap",
                f"{training_data_case}_domain_overlap_{method}.png",
            )
            domain_plot(
                ds=ds,
                var_groups=var_groups,
                title=f"{training_data_case} domain overlap {method}",
                save_path=fig_path,
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
        normalization_method: str,
        model_case: str,
        transfer_learning: bool,
        num_hp_shots: int | None,
    ) -> str:
        """Get the path to save evaluation results for a given combination of training data, normalization, and model architecture"""
        if transfer_learning:
            if num_hp_shots is None:
                num_hp_shots = "all"
            return os.path.join(
                result_dir,
                "transfer_learning",
                training_data_case,
                normalization_method,
                f"{model_case}_{num_hp_shots}.nc",
            )
        else:
            return os.path.join(
                result_dir,
                "standard_learning",
                training_data_case,
                normalization_method,
                f"{model_case}.nc",
            )

    @staticmethod
    def _trained_model_dir(
        model_dir: str,
        training_data_case: str,
        normalization_method: str,
        model_case: str,
        transfer_learning: bool,
        num_hp_shots: int | None,
    ) -> str:
        """Get the directory to save a trained model for a given combination of training data, normalization, and model architecture"""
        if transfer_learning:
            if num_hp_shots is None:
                num_hp_shots = "all"
            return os.path.join(
                model_dir,
                training_data_case,
                normalization_method,
                model_case,
                f"transfer_learning_{num_hp_shots}",
            )
        else:
            return os.path.join(
                model_dir,
                training_data_case,
                normalization_method,
                model_case,
                "standard_learning",
            )

    @staticmethod
    def _compute_standard_learning_result(
        trained_model_dir: str,
        result_path: str,
    ):
        """
        Compute results for standard learning.

        1. Restore trained model checkpoint
        2. Construct dataloaders
        3. Evaluate model on validation set and save results
        """

    @staticmethod
    def _compute_transfer_learning_result(
        trained_model_dir: str,
        result_path: str,
    ):
        """
        Compute results for transfer learning.
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

        if transfer_learning and training_data_case == "exnihilo":
            return True  # Doesn't need historic data

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

        return True

    @staticmethod
    def standard_learning_results(
        model_dir: str,
        result_dir: str,
        training_data_cases: list[str] = TRAINING_DATA_CASES,
        domain_normalization_methods: list[str] = DOMAIN_NORMALIZATION_METHODS,
        model_cases: list[str] = MODEL_CASES["power_balance"],
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
        logger.info("Computing standard learning results...")

        for training_data_case in training_data_cases:
            if training_data_case == "exnihilo":
                continue  # No training data, skip
            if not ComputeResults._check_data_requirements(
                training_data_case=training_data_case,
                transfer_learning=False,
            ):
                logger.warning(
                    f"Data requirements not met to compute results for training data case '{training_data_case}', skipping."
                )
                continue
            for normalization_method, model_case in product(
                domain_normalization_methods, model_cases
            ):
                result_path = ComputeResults._result_path(
                    result_dir=result_dir,
                    training_data_case=training_data_case,
                    normalization_method=normalization_method,
                    model_case=model_case,
                )
                if not os.path.exists(result_path):
                    trained_model_dir = ComputeResults._trained_model_dir(
                        model_dir=model_dir,
                        training_data_case=training_data_case,
                        normalization_method=normalization_method,
                        model_case=model_case,
                    )
                    if not os.path.exists(trained_model_dir):
                        logger.info(
                            f"Trained model not found for standard learning with training data case '{training_data_case}', model case '{model_case}', "
                            f"normalization method '{normalization_method}'. Training model now."
                        )
                        train_model_standard(
                            model_dir=trained_model_dir,
                            training_data_case=training_data_case,
                            normalization_method=normalization_method,
                            model_case=model_case,
                        )
                    else:
                        logger.info(
                            f"Trained model found for standard learning with training data case '{training_data_case}', model case '{model_case}', "
                            f"normalization method '{normalization_method}'. Computing results now."
                        )
                else:
                    logger.info(
                        f"Results already exist for training data case '{training_data_case}', model case '{model_case}', "
                        f"and normalization method '{normalization_method}', skipping computation."
                    )

    @staticmethod
    def transfer_learning_results(
        model_dir: str,
        result_dir: str,
        training_data_cases: list[str] = TRAINING_DATA_CASES,
        domain_normalization_methods: list[str] = DOMAIN_NORMALIZATION_METHODS,
        model_cases: list[str] = MODEL_CASES["power_balance"],
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
        logger.info("Computing transfer learning results...")

        for (
            training_data_case,
            normalization_method,
            model_case,
            num_hp_shots,
        ) in product(
            training_data_cases,
            domain_normalization_methods,
            model_cases,
            hp_shots_included,
        ):
            # Compute results for this combination of training data, model architecture, and number of high-performance shots
            # If the model is not trained, train it first
            if not ComputeResults._check_data_requirements(
                training_data_case=training_data_case,
                transfer_learning=True,
            ):
                logger.warning(
                    f"Data requirements not met to compute transfer learning results for training data case '{training_data_case}', skipping."
                )
                continue

            result_path = ComputeResults._result_path(
                result_dir=result_dir,
                training_data_case=training_data_case,
                normalization_method=normalization_method,
                model_case=model_case,
                transfer_learning=True,
                num_hp_shots=num_hp_shots,
            )

            if not os.path.exists(result_path):
                # Results not found, need to compute them
                trained_model_dir = ComputeResults._trained_model_dir(
                    model_dir=model_dir,
                    training_data_case=training_data_case,
                    normalization_method=normalization_method,
                    model_case=model_case,
                    transfer_learning=True,
                    num_hp_shots=num_hp_shots,
                )
                if not os.path.exists(trained_model_dir):
                    # Model not trained, need to go do that first
                    logger.info(
                        f"Trained model not found for transfer learning with training data case '{training_data_case}', model case '{model_case}', "
                        f"normalization method '{normalization_method}', and {num_hp_shots} high-performance shots included. Training model now."
                    )
                    # TODO(ZanderKeith) Call training function, and if parallelism is enabled, return to finish this step.
                    trainer, eval_dl = train_model_transfer(
                        model_dir=trained_model_dir,
                        training_data_case=training_data_case,
                        normalization_method=normalization_method,
                        model_case=model_case,
                        num_hp_shots=num_hp_shots,
                    )
                else:
                    logger.info(
                        f"Trained model found for transfer learning with training data case '{training_data_case}', model case '{model_case}', "
                        f"normalization method '{normalization_method}', and {num_hp_shots} high-performance shots included. Computing results now."
                    )
                    # make trainer and eval_dl objects using the trained model directory so we can evaluate and save results
                    trainer = None
                    eval_dl = None

                trainer.restore_best_checkpoint()
                eval_data = trainer.run_evals(eval_dl)
                input_ds = eval_data.input_ds.reset_index("sample")
                output_ds = eval_data.output_ds.reset_index("sample")
                eval_ds = xr.merge([input_ds, output_ds], compat="override")
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                eval_ds.to_netcdf(result_path)
            else:
                logger.info(
                    f"Results already exist for transfer learning with training data case '{training_data_case}', model case '{model_case}', "
                    f"normalization method '{normalization_method}', and {num_hp_shots} high-performance shots included, skipping computation."
                )


class ModelComparison:
    """
    Performance vs Model Architectures
    """

    @staticmethod
    def plot_losses(
        model_dir: str,
        result_dir: str,
        figure_dir: str,
        training_data_cases: list[str] = TRAINING_DATA_CASES,
        domain_normalization_methods: list[str] = DOMAIN_NORMALIZATION_METHODS,
        model_cases: list[str] = MODEL_CASES["power_balance"],
        hp_shots_included: list[int] = HP_SHOTS_INCLUDED,
    ):
        """
        Plot transfer learning losses comparing different model architectures.

        For each combination of training data case and normalization method, creates
        a figure with two panels:
        1. Integrated loss (mean +/- std over test samples) vs number of HP shots
           for each model architecture.
        2. Per-timestep loss profile for each model architecture at the highest
           HP shot count.

        Reads the evaluation result netCDF files produced by
        ``ComputeResults.transfer_learning_results`` and computes MSE between
        ``Wtot_MJ_pred`` and ``Wtot_MJ``.
        """
        import numpy as np

        for training_data_case, normalization_method in product(
            training_data_cases, domain_normalization_methods
        ):
            loss_ds_list = []
            label_list = []

            for model_case in model_cases:
                integrated_means: list[float] = []
                integrated_stds: list[float] = []
                timestep_means_list: list[np.ndarray] = []
                timestep_stds_list: list[np.ndarray] = []
                valid_hp_coords: list[int] = []

                for num_hp_shots in hp_shots_included:
                    result_path = ComputeResults._result_path(
                        result_dir=result_dir,
                        training_data_case=training_data_case,
                        normalization_method=normalization_method,
                        model_case=model_case,
                        transfer_learning=True,
                        num_hp_shots=num_hp_shots,
                    )

                    if not os.path.exists(result_path):
                        logger.warning(
                            f"Result file not found: {result_path}, skipping."
                        )
                        continue

                    eval_ds = xr.open_dataset(result_path)

                    if "Wtot_MJ_pred" not in eval_ds or "Wtot_MJ" not in eval_ds:
                        logger.warning(
                            f"Missing Wtot_MJ prediction variables in {result_path}, skipping."
                        )
                        eval_ds.close()
                        continue

                    # MSE between predicted and target stored energy
                    sq_error = (eval_ds["Wtot_MJ_pred"] - eval_ds["Wtot_MJ"]) ** 2

                    # Integrated loss: mean over time steps per sample, then stats across samples
                    integrated = sq_error.mean(dim="time_idx")
                    integrated_means.append(float(integrated.mean()))
                    integrated_stds.append(float(integrated.std()))

                    # Per-timestep loss: stats across samples at each time step
                    timestep_means_list.append(sq_error.mean(dim="sample").values)
                    timestep_stds_list.append(sq_error.std(dim="sample").values)

                    valid_hp_coords.append(
                        num_hp_shots if num_hp_shots is not None else -1
                    )
                    eval_ds.close()

                if not valid_hp_coords:
                    continue

                # Pad timestep arrays to uniform length (datasets may differ in time_idx size)
                max_time = max(len(t) for t in timestep_means_list)
                padded_ts_means = np.full((len(valid_hp_coords), max_time), np.nan)
                padded_ts_stds = np.full((len(valid_hp_coords), max_time), np.nan)
                for j, (tm, ts) in enumerate(
                    zip(timestep_means_list, timestep_stds_list, strict=True)
                ):
                    padded_ts_means[j, : len(tm)] = tm
                    padded_ts_stds[j, : len(ts)] = ts

                loss_ds = xr.Dataset(
                    {
                        "integrated_loss_mean": ("num_hp_shots", integrated_means),
                        "integrated_loss_std": ("num_hp_shots", integrated_stds),
                        "timestep_loss_mean": (
                            ["num_hp_shots", "time_idx"],
                            padded_ts_means,
                        ),
                        "timestep_loss_std": (
                            ["num_hp_shots", "time_idx"],
                            padded_ts_stds,
                        ),
                    },
                    coords={"num_hp_shots": valid_hp_coords},
                )

                loss_ds_list.append(loss_ds)
                label_list.append(model_case)

            if not loss_ds_list:
                logger.warning(
                    f"No transfer learning results found for "
                    f"{training_data_case}/{normalization_method}, skipping loss plot."
                )
                continue

            save_path = os.path.join(
                figure_dir,
                "model_comparison",
                f"{training_data_case}_{normalization_method}_transfer_learning_losses.png",
            )

            transfer_learning_losses(
                loss_ds_list=loss_ds_list,
                label_list=label_list,
                hp_shots_included=hp_shots_included,
                save_path=save_path,
            )


class DataComparison:
    """
    Performance vs Data Normalization
    """


@staticmethod
def run_study(
    project_name: str,
    working_dir_base: str | None,
    figure_dir_base: str | None,
    cmod_dataset_path: str | None = config.cmod_dataset_path,
    tcv_dataset_path: str | None = config.tcv_dataset_path,
    d3d_lp_dataset_path: str | None = config.d3d_lp_dataset_path,
    d3d_hp_dataset_path: str | None = config.d3d_hp_dataset_path,
    enable_parallelism: bool | None = False,
    clean_models: bool | None = False,
    clean_results: bool | None = False,
    clean_figures: bool | None = False,
    skip_visualization: bool | None = False,
    skip_standard_learning_results: bool | None = False,
):
    """
    Go from datasets to all figures in one command.
    See `popsim_transport_predictor/scripts/reproduce_figures.sh` for an example usage.

    Requires specifying paths to the source datasets.
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
    cmod_dataset_path : str | None
        Path to the C-Mod dataset file. If not provided, figures which require C-Mod data will be skipped.
    tcv_dataset_path : str | None
        Path to the TCV dataset file. If not provided, figures which require TCV data will be skipped.
    d3d_lp_dataset_path : str | None
        Path to the DIII-D low-performance dataset file. If not provided, figures which require DIII-D low-performance data will be skipped.
    d3d_hp_dataset_path : str | None
        Path to the DIII-D high-performance dataset file. If not provided, figures which require DIII-D high-performance data will be skipped.
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
    """

    ######################
    # Set up directories #
    ######################
    if working_dir_base is None:
        working_dir_base = os.path.join(PACKAGE_ROOT, "popsim_studies", "working_dir")
    if figure_dir_base is None:
        figure_dir_base = working_dir_base

    def _setup_directories():
        working_dir = os.path.join(working_dir_base, project_name)
        model_dir = os.path.join(working_dir, "models")
        result_dir = os.path.join(working_dir, "results")
        figure_dir = os.path.join(figure_dir_base, project_name)

        log_path = os.path.join(working_dir, "logs", f"{os.getpid()}_run_study.log")
        logger.add(log_path)

        logger.info("STARTING STUDY")
        logger.info(f"Project name: {project_name}")
        logger.info(f"Working directory base: {working_dir_base}")
        logger.info(f"Figure directory base: {figure_dir_base}")
        logger.info(f"C-Mod dataset path: {cmod_dataset_path}")
        logger.info(f"TCV dataset path: {tcv_dataset_path}")
        logger.info(f"DIII-D low-performance dataset path: {d3d_lp_dataset_path}")
        logger.info(f"DIII-D high-performance dataset path: {d3d_hp_dataset_path}")
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
            shutil.rmtree(model_dir, ignore_errors=True)
        if clean_results:
            shutil.rmtree(result_dir, ignore_errors=True)
        if clean_figures:
            shutil.rmtree(figure_dir, ignore_errors=True)

        for directory in [model_dir, result_dir, figure_dir]:
            os.makedirs(directory, exist_ok=True)

        return working_dir, model_dir, result_dir, figure_dir

    _working_dir, model_dir, result_dir, figure_dir = _setup_directories()

    ######################
    # Data Visualization #
    ######################
    if not skip_visualization:
        logger.info("DATA VISUALIZATION")

        DataVisualization.domain_overlap(
            figure_dir=figure_dir,
        )

        DataVisualization.performance_extrapolation(
            figure_dir=figure_dir,
        )

    ########################
    # Launch Orchestration #
    ########################
    logger.info("ORCHESTRATION")

    # Standard Learning Results
    if not skip_standard_learning_results:
        ComputeResults.standard_learning_results(
            model_dir=model_dir,
            result_dir=result_dir,
        )

    # Transfer Learning Results
    ComputeResults.transfer_learning_results(
        model_dir=model_dir,
        result_dir=result_dir,
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
