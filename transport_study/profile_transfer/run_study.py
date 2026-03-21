import os
import shutil
import time
from dataclasses import dataclass
from itertools import product

import fire
import netCDF4  # noqa: F401
import xarray as xr
from loguru import logger
from popsim.ml import TrainConfig
from popsim.ml.launch import (
    launch_agent,
    launch_sweep,
    launch_train,
)

from transport_study import PACKAGE_ROOT
from transport_study.config import config
from transport_study.orchestration.slurm_utils import (
    launch_train_parallel,
)
from transport_study.orchestration.study import Study
from transport_study.orchestration.wandb_utils import (
    get_sweep_id,
    run_clean_sweeps,
)


class ProfileStudy(Study):
    HYPERPARAM_TRAINING_DATA = "cmod_tcv"
    HYPERPARAM_DATA_NORMALIZATION = "physics"
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
        - physics: Convert to typical dimensionless parameters like beta, q95, f_G, etc.
        - Since we found that physics transfers best for power balance, only studying it here

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

            # The whole point of exnihilo is training from nothing, so it doesn't make sense to have domain adaptation in that case since there's no source domain to adapt from
            if self.training_data == "exnihilo" and self.domain_adaptation is not None:
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
                "shape_init_pca",
                "shape_init_kmeans",
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
                return f"case.{self.model_type}.td_{self.training_data}.da_{self.domain_adaptation}.freeze_{self.freeze_shapes}.hp_{self.num_hp_shots}"
            else:
                return f"case.{self.model_type}.td_{self.training_data}.freeze_{self.freeze_shapes}"

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
        freeze_shapes_options,
        num_hp_shots_options,
    ):
        cases = []
        # Make every case we're interested in for this study
        for (
            model_type,
            training_dataset,
            data_normalization,
            domain_adaptation,
            freeze_shapes,
            num_hp_shots,
        ) in product(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_shapes_options,
            num_hp_shots_options,
        ):
            if domain_adaptation is None and num_hp_shots != -1:
                continue  # Invalid case, skip

            case = self.Case(
                model_type=model_type,
                training_data=training_dataset,
                data_normalization=data_normalization,
                domain_adaptation=domain_adaptation,
                freeze_shapes=freeze_shapes,
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

    def __init__(
        self,
        name: str,
        working_dir_base: str,
        dataset_paths: dict[str, str],
        model_types: list[str],
        training_datasets: list[str],
        data_normalization_methods: list[str],
        domain_adaptation_methods: list[str],
        freeze_shapes_options: list[bool],
        num_hp_shots_options: list[int],
    ):
        cases = self.make_cases(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_shapes_options,
            num_hp_shots_options,
        )
        super().__init__(name, working_dir_base, dataset_paths, cases)

        logger.info(f"C-Mod dataset path: {dataset_paths.get('cmod', 'Not provided')}")
        logger.info(f"TCV dataset path: {dataset_paths.get('tcv', 'Not provided')}")
        logger.info(
            f"DIII-D high-performance dataset path: {dataset_paths.get('d3d_hp', 'Not provided')}"
        )

        logger.info(f"Model types: {model_types}")
        logger.info(f"Training datasets: {training_datasets}")
        logger.info(f"Data normalization methods: {data_normalization_methods}")
        logger.info(f"Domain adaptation methods: {domain_adaptation_methods}")
        logger.info(f"Freeze shapes options: {freeze_shapes_options}")
        logger.info(f"Number of high-performance shots options: {num_hp_shots_options}")

    #############
    # EXECUTION #
    #############
    def _input_vars(self, case: Case) -> list[str]:
        input_vars_base = [
            "Ip_MA",
            "B0",
            "betan",
            "ne20_edge",
            "R0",
            "a_minor",
            "kappa",
            "delta_top",
            "delta_bot",
        ]
        if case.data_normalization == "physics":
            input_vars = [
                *input_vars_base,
                "epsilon",
                "q_star",
                "f_G",
                "aB0",
            ]
        else:
            raise ValueError(
                f"Profile study only uses physics normalization, but got {case.data_normalization}"
            )

        return input_vars

    def make_train_config(self, case: Case) -> TrainConfig:
        """Make the TrainConfig for a given case
        If a hyperparameter tuned config is available, fills in the hyperparameters from that, otherwise uses default config.
        """
        dataloader_config_base = {
            "target_vars": ["Te_keV_psi", "ne20_psi", "ds_source_idx"],
            "training_data": case.training_data,
            "data_normalization": case.data_normalization,
            "domain_adaptation": case.domain_adaptation,
            "num_hp_shots": case.num_hp_shots,
            "hp_test_set_size": config.hp_test_set_size,
            "prng_seed": 42,
            "debug": config.debug,
            # Hyperparameters
            "batch_size": 8192,
        }
        loss_config_base = {
            "huber_delta": 0.5,
        }
        optimizer_config_base = {
            "lr0": 1e-4,
            "transition_steps": 500,
            "decay_rate": 0.5,
            "lrf": 5e-4,
            "weight_decay": 2e-4,
        }
        test_eval_suite_config_base = {
            "result_path": self.result_path(case),
        }
        if case.domain_adaptation == "mixing":
            # Special logic for loss weighting when doing mixing domain adaptation
            # Assuming ~1000 shots of historic data for C-Mod and TCV and DIII-D low-performance, and num_hp_shots of DIII-D high-performance
            # we want the high-performance data to be consistently heavily weighted
            # Weights are chosen so that each device's effective contribution F_x = W_x * N_x
            # (where N_x is the shot count) sums to 200, with d3d_hp carrying ~50% of that total.
            # So the C-Mod and TCV data each make up 40 out of 200,
            # the DIII-D low-performance data makes up 60 out of 200,
            # and the DIII-D high-performance data makes up 100 out of 200
            W_c = 20 / 1000
            W_t = 20 / 1000
            W_dlp = 60 / 1000
            if case.num_hp_shots in [-1, 0]:
                # If -1, all 97 high-performance shots in the DIII-D dataset
                # If 0, weights aren't being used anyway
                N_dhp = 97
            else:
                N_dhp = case.num_hp_shots
            W_dhp = 100 / N_dhp
            # Multiply all by 100 to get back to a value ~1
            dataloader_config_base["device_weights"] = {
                "cmod": W_c * 100,
                "tcv": W_t * 100,
                "d3d_lp": W_dlp * 100,
                "d3d_hp": W_dhp * 100,
            }

        def _make_train_config_base(case: ProfileStudy.Case) -> TrainConfig:
            if case.model_type in ["shape_init_pca", "shape_init_kmeans"]:
                train_config_base = TrainConfig(
                    project=self.wandb_project_name(case),
                    train_run_builder="transport_study.modules.profile_predictor.trb.ProfilePredictorTRB",
                    max_epochs=config.max_epochs,
                    epochs_per_val=config.epochs_per_val,
                    checkpoint_dir=self.trained_model_dir(
                        case
                    ),  # When doing hyperparameter tuning, this gets overwritten by the wandb agent
                    dataloader_config={
                        "input_vars": self._input_vars(case),
                        "extra_vars": ["Te_shape", "ne_shape"],
                        **dataloader_config_base,
                    },
                    model_init_config={
                        "model_type": case.model_type,
                        "data_normalization": case.data_normalization,
                        "domain_adaptation": case.domain_adaptation,
                        "freeze_shapes": case.freeze_shapes,
                        "te_shape_var": "Te_shape",
                        "ne_shape_var": "ne_shape",
                        "n_shapes": 3,
                        "nn_depth": 2,
                        "nn_width": 16,
                        "min_val": 0,  # Minimum profile value
                        "max_val": 6,  # Maximum profile value
                        "in_size": 9,  # Ip_MA, B0, betan, ne20_edge, R0, a_minor, kappa, delta_top, delta_bot
                        "softmax_temp": 1,
                        "prng_seed": 42,
                    },
                    loss_config=loss_config_base,
                    optimizer_config=optimizer_config_base,
                    test_eval_suite_config=test_eval_suite_config_base,
                )
            elif case.model_type == "unstructured_nn":
                train_config_base = TrainConfig(
                    project=self.wandb_project_name(case),
                    train_run_builder="transport_study.modules.profile_predictor.trb.ProfilePredictorTRB",
                    max_epochs=config.max_epochs,
                    epochs_per_val=config.epochs_per_val,
                    checkpoint_dir=self.trained_model_dir(
                        case
                    ),  # When doing hyperparameter tuning, this gets overwritten by the wandb agent
                    dataloader_config={
                        "input_vars": self._input_vars(case),
                        **dataloader_config_base,
                    },
                    model_init_config={
                        "model_type": case.model_type,
                        "data_normalization": case.data_normalization,
                        "domain_adaptation": case.domain_adaptation,
                        "freeze_shapes": case.freeze_shapes,
                        "n_points": 21,  # Number of points along the profile to predict for the unstructured NN
                        "nn_depth": 2,
                        "nn_width": 16,
                        "in_size": 9,  # Ip_MA, B0, betan, ne20_edge, R0, a_minor, kappa, delta_top, delta_bot
                        "prng_seed": 42,
                    },
                    loss_config=loss_config_base,
                    optimizer_config=optimizer_config_base,
                    test_eval_suite_config=test_eval_suite_config_base,
                )
            else:
                raise ValueError(f"Unknown model type: {case.model_type}")

            return train_config_base

        train_config_base = _make_train_config_base(case)

        # For transfer learning, point to where this case's pretrained checkpoint
        if case.domain_adaptation == "transfer":
            # Pretrained model has same model_type, training_data, and data_normalization,
            # but no domain adaptation and no high-performance shots
            transfer_case = ProfileStudy.Case(
                model_type=case.model_type,
                training_data=case.training_data,
                data_normalization=case.data_normalization,
                domain_adaptation=None,
                freeze_shapes=case.freeze_shapes,
                num_hp_shots=-1,
            )
            transfer_case_model_dir = self.trained_model_dir(transfer_case)

            train_config_base = train_config_base.model_copy(
                update={
                    "model_init_config": {
                        **train_config_base.model_init_config,
                        "transfer_checkpoint": transfer_case_model_dir,
                    }
                }
            )

        tuned_config_path = self.tuned_config_path(case)
        if os.path.exists(tuned_config_path):
            tuned_config = TrainConfig.load(tuned_config_path)
            # Restore hyperparameters from the tuned config, but keep the rest of the settings the same

            # Hyperparameters swept for all modules
            train_config = train_config_base.model_copy(
                update={
                    "model_init_config": {
                        **train_config_base.model_init_config,
                        "nn_depth": tuned_config.model_init_config["nn_depth"],
                        "nn_width": tuned_config.model_init_config["nn_width"],
                    },
                    "dataloader_config": {
                        **train_config_base.dataloader_config,
                        "batch_size": tuned_config.dataloader_config["batch_size"],
                    },
                    "optimizer_config": tuned_config.optimizer_config,
                }
            )

            # Hyperparameters swept for only certain modules
            if case.model_type in ["shape_init_pca", "shape_init_kmeans"]:
                train_config = train_config.model_copy(
                    update={
                        "model_init_config": {
                            **train_config.model_init_config,
                            "n_shapes": tuned_config.model_init_config["n_shapes"],
                            "softmax_temp": tuned_config.model_init_config[
                                "softmax_temp"
                            ],
                        }
                    }
                )
            elif case.model_type in ["unstructured_nn"]:
                train_config = train_config.model_copy(
                    update={
                        "model_init_config": {
                            **train_config.model_init_config,
                            "n_points": tuned_config.model_init_config["n_points"],
                        }
                    }
                )
        else:
            train_config = train_config_base

        return train_config

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
                "profile_transfer",
                "sweep_configs",
                f"{case.model_type}.yaml",
            )
            launch_sweep(train_config, sweep_config_path)
        else:
            launch_agent(train_config, sweep_id)

    def launch_train(self, case: Case, enable_parallelism: bool = False):
        """Launch a training job for the given case."""
        if case.is_impossible():
            raise ValueError(
                f"Case {case} is not a possible case to run, check the logic in the Case dataclass to see why this is. This should have been caught earlier!"
            )

        logger.opt(colors=True).info(
            f"<bold><red>LAUNCHING TRAINING for case\n{case}</red></bold>"
        )

        train_config = self.make_train_config(case)
        result_path = self.result_path(case)
        train_job_name = self.train_job_name(case)
        if enable_parallelism:
            logger.info(f"Launching training job {train_job_name} for case\n{case}")
            launch_train_parallel(
                train_config,
                train_job_name,
                result_path,
                os.path.join(self.result_dir, "logs"),
            )
        else:
            logger.info("Launching training serially")
            _, _, _, _, result_dict = launch_train(train_config)
            ds = result_dict["test/study_results"]
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            ds.to_netcdf(result_path)

    ##############
    # COLLECTION #
    ##############

    def collect_results(self):
        """Collect results from all cases and combine them into a single xarray dataset for analysis and visualization.

        Dims: case_idx, shot_idx
        Coords:
        - shot(case_idx, shot_idx)
        - ds_source(case_idx, shot_idx)
        - model_type(case_idx)
        - training_data(case_idx)
        - data_normalization(case_idx)
        - domain_adaptation(case_idx)
        - freeze_shapes(case_idx)
        - num_hp_shots(case_idx)
        Data variables: (E is either relative 'rel' or absolute 'abs', and D is dimension either 'shot' or per-timeslice 'ts')
        - error_E_D_mean(case_idx)
        - error_E_D_std(case_idx)
        - error_E_D_med(case_idx)
        - error_E_D_p25(case_idx)
        - error_E_D_p75(case_idx)
        - error_E_D_min(case_idx)
        - error_E_D_max(case_idx)
        """
        results = []
        for case in self.cases:
            result_path = self.result_path(case)
            if not os.path.exists(result_path):
                continue

            ds = xr.load_dataset(result_path)

            err_abs_shot = ds["error_abs_shot"]
            err_rel_shot = ds["error_rel_shot"]
            err_abs_ts = ds["error_abs_ts"]
            err_rel_ts = ds["error_rel_ts"]

            result = xr.Dataset(
                {
                    "err_abs_shot_mean": err_abs_shot.mean(),
                    "err_abs_shot_std": err_abs_shot.std(),
                    "err_abs_shot_med": err_abs_shot.median(),
                    "err_abs_shot_p25": err_abs_shot.quantile(0.25).drop_vars(
                        "quantile"
                    ),
                    "err_abs_shot_p75": err_abs_shot.quantile(0.75).drop_vars(
                        "quantile"
                    ),
                    "err_abs_shot_min": err_abs_shot.min(),
                    "err_abs_shot_max": err_abs_shot.max(),
                    "err_rel_shot_mean": err_rel_shot.mean(),
                    "err_rel_shot_std": err_rel_shot.std(),
                    "err_rel_shot_med": err_rel_shot.median(),
                    "err_rel_shot_p25": err_rel_shot.quantile(0.25).drop_vars(
                        "quantile"
                    ),
                    "err_rel_shot_p75": err_rel_shot.quantile(0.75).drop_vars(
                        "quantile"
                    ),
                    "err_rel_shot_min": err_rel_shot.min(),
                    "err_rel_shot_max": err_rel_shot.max(),
                    "err_abs_ts_mean": err_abs_ts.mean(),
                    "err_abs_ts_std": err_abs_ts.std(),
                    "err_abs_ts_med": err_abs_ts.median(),
                    "err_abs_ts_p25": err_abs_ts.quantile(0.25).drop_vars("quantile"),
                    "err_abs_ts_p75": err_abs_ts.quantile(0.75).drop_vars("quantile"),
                    "err_abs_ts_min": err_abs_ts.min(),
                    "err_abs_ts_max": err_abs_ts.max(),
                    "err_rel_ts_mean": err_rel_ts.mean(),
                    "err_rel_ts_std": err_rel_ts.std(),
                    "err_rel_ts_med": err_rel_ts.median(),
                    "err_rel_ts_p25": err_rel_ts.quantile(0.25).drop_vars("quantile"),
                    "err_rel_ts_p75": err_rel_ts.quantile(0.75).drop_vars("quantile"),
                    "err_rel_ts_min": err_rel_ts.min(),
                    "err_rel_ts_max": err_rel_ts.max(),
                }
            ).assign_coords(
                {
                    "model_type": case.model_type,
                    "training_data": case.training_data,
                    "data_normalization": case.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    "freeze_shapes": case.freeze_shapes,
                    "num_hp_shots": case.num_hp_shots,
                }
            )
            results.append(result)

        ds_merged = xr.concat(results, dim="case_idx")
        return ds_merged


def run_study(  # noqa: PLR0915
    project_name: str,
    working_dir_base: str | None,
    model_types: list[str] | None = None,
    training_datasets: list[str] | None = None,
    data_normalization_methods: list[str] | None = None,
    domain_adaptation_methods: list[str] | None = None,
    freeze_shapes_options: list[bool] | None = None,
    num_hp_shots_options: list[int] | None = None,
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
    ):
        def _assign_args(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_shapes_options,
            num_hp_shots_options,
        ):
            if model_types is None:
                model_types = ["shape_init_pca", "shape_init_kmeans", "unstructured_nn"]
            if training_datasets is None:
                training_datasets = ["cmod", "tcv", "cmod_tcv", "exnihilo"]
            if data_normalization_methods is None:
                data_normalization_methods = ["physics"]
            if domain_adaptation_methods is None:
                domain_adaptation_methods = [None, "mixing", "transfer"]
            if freeze_shapes_options is None:
                freeze_shapes_options = [True, False]
            if num_hp_shots_options is None:
                num_hp_shots_options = [0, 1, 3, 10, 32, -1]

            return (
                model_types,
                training_datasets,
                data_normalization_methods,
                domain_adaptation_methods,
                freeze_shapes_options,
                num_hp_shots_options,
            )

        (
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_shapes_options,
            num_hp_shots_options,
        ) = _assign_args(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_shapes_options,
            num_hp_shots_options,
        )

        def _check_args(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
        ):
            for model_type in model_types:
                if model_type not in [
                    "shape_init_pca",
                    "shape_init_kmeans",
                    "unstructured_nn",
                ]:
                    raise ValueError(
                        f"Invalid model type: {model_type}. Must be one of 'shape_init_pca', 'shape_init_kmeans', or 'unstructured_nn'."
                    )

            for training_dataset in training_datasets:
                if training_dataset not in ["cmod", "tcv", "cmod_tcv", "exnihilo"]:
                    raise ValueError(
                        f"Invalid training dataset: {training_dataset}. Must be one of 'cmod', 'tcv', 'cmod_tcv', or 'exnihilo'."
                    )

            for data_normalization in data_normalization_methods:
                if data_normalization not in ["physics"]:
                    raise ValueError(
                        f"Invalid data normalization method: {data_normalization}. Only 'physics' is implemented for profile transfer."
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
        )

    (
        model_types,
        training_datasets,
        data_normalization_methods,
        domain_adaptation_methods,
        freeze_shapes_options,
        num_hp_shots_options,
    ) = _validate_args(
        model_types,
        training_datasets,
        data_normalization_methods,
        domain_adaptation_methods,
        freeze_shapes_options,
        num_hp_shots_options,
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

    def _move_data(study: Study):
        logger.info("Moving data to cluster scratch for faster training")
        for ds_path in [
            config.cmod_dataset_path,
            config.tcv_dataset_path,
            config.d3d_lp_dataset_path,
            config.d3d_hp_dataset_path,
        ]:
            if ds_path:
                _, file_name = os.path.split(ds_path)
                scratch_dir = os.path.join(config.scratch_dir, study.name)
                os.makedirs(scratch_dir, exist_ok=True)
                scratch_path = os.path.join(scratch_dir, file_name)
                if not os.path.exists(scratch_path):
                    ds = xr.open_dataset(ds_path)
                    # For training, only need fresh profiles
                    ds = ds.where(ds.fresh_profiles == 1, drop=True)
                    # Rechunk to be ~50 MB per chunk
                    raise NotImplementedError

    if config.scratch_dir:
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
