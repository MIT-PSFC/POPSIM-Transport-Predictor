from __future__ import annotations

import shutil
import time
import tomllib
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import fire
import netCDF4  # noqa: F401
import toml
import xarray as xr
from loguru import logger
from popsim.ml import TrainConfig
from pydantic import Field, field_validator, model_validator

from transport_study import PACKAGE_ROOT
from transport_study.config import StudyConfig, config, load_config
from transport_study.orchestration.organize_data import TrainingData
from transport_study.orchestration.study import Study
from transport_study.orchestration.wandb_utils import (
    run_clean_sweeps,
)

# When domain adaptation is None, we aren't using any target data during training anyway so this is unused
# During domain adaptation, we aren't doing hyperparameter tuning
HYPERPARAM_TARGET_SHOTS = 0


def _parse_training_data(s: str, dataset_paths: dict, target_device: str | None) -> TrainingData:
    """Convert a string like 'cmod_tcv' or 'exnihilo' to a TrainingData object."""
    if s == "exnihilo":
        non_target = frozenset(dataset_paths.keys()) - ({target_device} if target_device else set())
        return TrainingData(sources_unsorted=non_target, exnihilo=True)
    return TrainingData(sources_unsorted=frozenset(s.split("_")))


class ProfileStudy(Study):
    ##################
    # INITIALIZATION #
    ##################
    class Config(StudyConfig):
        # Organization for datasets and wandb projects
        working_dir_base: Path = PACKAGE_ROOT / "popsim_studies" / "working_dir"
        # The different cases being compared in this study
        model_types: tuple[str, ...] = Field(default_factory=lambda: ("shape_init_pca", "shape_init_kmeans", "unstructured_nn"))
        training_datasets: tuple[TrainingData, ...]
        data_normalization_methods: tuple[str, ...] = Field(default_factory=lambda: ("physics",))
        domain_adaptation_methods: tuple[str | None, ...] = Field(default_factory=lambda: (None, "mixing", "transfer"))
        freeze_shapes_options: tuple[bool, ...] = Field(default_factory=lambda: (True,))
        num_target_shots_options: tuple[int, ...] = Field(default_factory=lambda: (0, 1, 10, -1))
        target_test_set_size: int
        # configurations for the hyperparameter tuning case
        hyperparam_data_normalization: str = "physics"
        hyperparam_domain_adaptation: str | None = None
        hyperparam_freeze_shapes: bool = True
        hyperparam_num_target_shots: int = HYPERPARAM_TARGET_SHOTS
        # Misc configurations
        dataset_sizes: dict[str, int] = Field(
            default_factory=dict
        )  # Optional dict of dataset sizes to use for weighting during domain adaptation, only used if domain_adaptation includes "mixing"
        dataset_fractions: dict[str, float] = Field(
            default_factory=dict
        )  # Optional dict of dataset fractions to use during domain adaptation, only used if domain_adaptation includes "mixing".

        @field_validator("model_types")
        @classmethod
        def _validate_model_types(cls, v: tuple[str, ...]) -> tuple[str, ...]:
            valid = {"shape_init_pca", "shape_init_kmeans", "unstructured_nn"}
            for mt in v:
                if mt not in valid:
                    raise ValueError(f"Invalid model type: {mt}. Must be one of {sorted(valid)}.")
            return v

        @field_validator("data_normalization_methods")
        @classmethod
        def _validate_data_normalization(cls, v: tuple[str, ...]) -> tuple[str, ...]:
            for dn in v:
                if dn not in ["physics"]:
                    raise ValueError(f"Invalid data normalization: {dn}. Only 'physics' is implemented for profile transfer.")
            return v

        @field_validator("domain_adaptation_methods")
        @classmethod
        def _validate_domain_adaptation(cls, v: tuple[str | None, ...]) -> tuple[str | None, ...]:
            valid = {None, "mixing", "transfer"}
            converted = tuple(None if da == "none" else da for da in v)
            for da in converted:
                if da not in valid:
                    raise ValueError(f"Invalid domain adaptation: {da}. Must be one of {valid}.")
            return converted

        @model_validator(mode="before")
        @classmethod
        def _coerce_training_datasets(cls, data) -> dict:
            if "training_datasets" not in data:
                return data
            dataset_paths = dict(data.get("dataset_paths", {}))
            target_device = data.get("target_device")
            data["training_datasets"] = tuple(
                _parse_training_data(s, dataset_paths, target_device) if isinstance(s, str) else s for s in data["training_datasets"]
            )
            return data

        @classmethod
        def from_toml(cls, path: Path) -> ProfileStudy.Config:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            datasets = data.pop("datasets", {})
            target = datasets.pop("target", None)
            study_cases = data.pop("study_cases", {})
            return cls(
                **data,
                **study_cases,
                dataset_paths={k: Path(v) for k, v in datasets.items()},
                target_device=target,
            )

        def is_compatible(self, cfg: ProfileStudy.Config) -> bool:
            """Check if two configs are compatible for running the same study
            For the ProfileStudy, that means the following:
            1: study name must match (used in WandB sweeps, etc.)
            2: dataset paths must be identical
            3: target device must be the same
            4: target test set size must be the same
            5: hyperparameter tuning configs must match
            6: Dataset sizes and weights must match
            """
            return (
                self.study_name == cfg.study_name
                and self.dataset_paths == cfg.dataset_paths
                and self.target_device == cfg.target_device
                and self.target_test_set_size == cfg.target_test_set_size
                and self.hyperparam_data_normalization == cfg.hyperparam_data_normalization
                and self.hyperparam_domain_adaptation == cfg.hyperparam_domain_adaptation
                and self.hyperparam_freeze_shapes == cfg.hyperparam_freeze_shapes
                and self.hyperparam_num_target_shots == cfg.hyperparam_num_target_shots
                and self.dataset_sizes == cfg.dataset_sizes
                and self.dataset_fractions == cfg.dataset_fractions
            )

        def save(self, path: Path):
            raw = self.model_dump()
            datasets = {k: str(v) for k, v in self.dataset_paths.items()}
            if self.target_device is not None:
                datasets["target"] = self.target_device
            skip = {"dataset_paths", "target_device", "training_datasets", "domain_adaptation_methods"}
            data = {}
            for k, v in raw.items():
                if k in skip:
                    continue
                if v is None:
                    continue
                if isinstance(v, Path):
                    data[k] = str(v)
                else:
                    data[k] = v
            data["training_datasets"] = [str(td) for td in self.training_datasets]
            data["domain_adaptation_methods"] = [da if da is not None else "none" for da in self.domain_adaptation_methods]
            data["datasets"] = datasets
            with open(path, "w") as f:
                toml.dump(data, f)

    def __init__(
        self,
        cfg: str | Path | ProfileStudy.Config,
    ):
        if not config.initialized:
            load_config(cfg)

        cases = self.make_cases(
            config.model_types,
            config.training_datasets,
            config.data_normalization_methods,
            config.domain_adaptation_methods,
            config.freeze_shapes_options,
            config.num_target_shots_options,
        )
        super().__init__(config.study_name, config.working_dir_base, cases)

        logger.info(f"Model types: {config.model_types}")
        logger.info(f"Training datasets: {config.training_datasets}")
        logger.info(f"Data normalization methods: {config.data_normalization_methods}")
        logger.info(f"Domain adaptation methods: {config.domain_adaptation_methods}")
        logger.info(f"Freeze shapes options: {config.freeze_shapes_options}")
        logger.info(f"Number of target shots included in training options: {config.num_target_shots_options}")

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

        num_target_shots: The number of shots included in the training data from the target dataset, or -1 to include all shots (including all shots in training is cheating, but again answers the question of what is the best possible performance).
        """

        model_type: str
        training_data: TrainingData
        data_normalization: str  # raw, physics, z_score, coral
        domain_adaptation: str  # none, mixing, transfer
        freeze_shapes: bool
        num_target_shots: int  # Number of target shots included in training, or -1 for all (should be HYPERPARAM_TARGET_SHOTS if domain_adaptation is None)
        prereqs: (
            list[Study.Case] | None
        )  # If not None, this case depends on the results of another case, and should only be run after that case has been run

        def is_hyperparam_case(self) -> bool:
            if (
                self.training_data == ProfileStudy._hyperparam_training_data()
                and self.data_normalization == config.hyperparam_data_normalization
                and self.domain_adaptation == config.hyperparam_domain_adaptation
                and self.freeze_shapes == config.hyperparam_freeze_shapes
                and self.num_target_shots == config.hyperparam_num_target_shots
            ):
                return True
            else:
                return False

        def is_impossible(self) -> bool:
            """Some cases don't make sense to run. Mark those cases as impossible and raise an error if we try to run them."""
            # Can't do transfer learning or training from nothing with 0 target shots.
            if (self.domain_adaptation == "transfer" or self.training_data.exnihilo) and self.num_target_shots == 0:
                return True

            # exnihilo means training from nothing - no source domain to adapt from
            if self.training_data.exnihilo and self.domain_adaptation is not None:
                return True

            return False

        def get_hyperparam_prereq(self) -> Study.Case:
            if self.is_hyperparam_case():
                return self
            else:
                return ProfileStudy.Case(
                    model_type=self.model_type,
                    training_data=ProfileStudy._hyperparam_training_data(),
                    data_normalization=config.hyperparam_data_normalization,
                    domain_adaptation=config.hyperparam_domain_adaptation,
                    freeze_shapes=config.hyperparam_freeze_shapes,
                    num_target_shots=config.hyperparam_num_target_shots,
                )

        def __init__(
            self,
            model_type: str,
            training_data: TrainingData | str,
            data_normalization: str,
            domain_adaptation: str,
            freeze_shapes: bool,
            num_target_shots: int,
        ):
            if isinstance(training_data, str):
                training_data = _parse_training_data(training_data, dict(config.dataset_paths), config.target_device)
            self.model_type = model_type
            self.training_data = training_data
            self.data_normalization = data_normalization
            self.domain_adaptation = domain_adaptation
            self.freeze_shapes = freeze_shapes
            self.num_target_shots = num_target_shots

            # Recursively add prereqs based on the logic of which cases depend on which other cases
            if model_type not in [
                "shape_init_pca",
                "shape_init_kmeans",
                "unstructured_nn",
            ]:
                raise ValueError(f"Unknown model type: {model_type}")
            if domain_adaptation is None:
                if not training_data.exnihilo and num_target_shots != HYPERPARAM_TARGET_SHOTS:
                    raise ValueError(
                        "If domain_adaptation is None and training data is not 'exnihilo', num_target_shots must be HYPERPARAM_TARGET_SHOTS since this means we're training and testing on the same dataset"
                    )

            prereqs = []

            # Add prereqs based on whether this is a hyperparameter tuning case or not.
            if not self.is_hyperparam_case():
                prereqs += [
                    ProfileStudy.Case(
                        model_type=model_type,
                        training_data=ProfileStudy._hyperparam_training_data(),
                        data_normalization=config.hyperparam_data_normalization,
                        domain_adaptation=config.hyperparam_domain_adaptation,
                        freeze_shapes=config.hyperparam_freeze_shapes,
                        num_target_shots=config.hyperparam_num_target_shots,
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
                        num_target_shots=HYPERPARAM_TARGET_SHOTS,
                    )
                ]

            prereqs = list(dict.fromkeys(prereqs))
            if len(prereqs) > 0:
                self.prereqs = prereqs
            else:
                self.prereqs = None

        def __str__(self):
            if self.domain_adaptation:
                return f"case.{self.model_type}.td_{self.training_data}.freeze_{self.freeze_shapes}.targ_{self.num_target_shots}.da_{self.domain_adaptation}"
            elif self.training_data.exnihilo:
                return f"case.{self.model_type}.td_{self.training_data}.freeze_{self.freeze_shapes}.targ_{self.num_target_shots}"
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
                        self.num_target_shots,
                    )
                )
            elif self.training_data.exnihilo:
                return hash(
                    (
                        self.model_type,
                        self.training_data,
                        self.data_normalization,
                        self.freeze_shapes,
                        self.num_target_shots,
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
        num_target_shots_options,
    ):
        cases = []
        # Make every case we're interested in for this study
        for (
            model_type,
            training_dataset,
            data_normalization,
            domain_adaptation,
            freeze_shapes,
            num_target_shots,
        ) in product(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_shapes_options,
            num_target_shots_options,
        ):
            if domain_adaptation is None:
                if training_dataset.exnihilo:
                    if num_target_shots == 0:
                        continue  # Can't train from nothing with 0 target shots
                elif num_target_shots != HYPERPARAM_TARGET_SHOTS:
                    continue  # Invalid case, skip
            if model_type == "unstructured_nn" and not freeze_shapes:
                continue  # No shapes to freeze, just do one of the two

            case = self.Case(
                model_type=model_type,
                training_data=training_dataset,
                data_normalization=data_normalization,
                domain_adaptation=domain_adaptation,
                freeze_shapes=freeze_shapes,
                num_target_shots=num_target_shots,
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
        possible_cases = [case for case in unique_cases if not case.is_impossible()]  # Remove impossible cases

        return possible_cases

    @classmethod
    def _hyperparam_training_data(cls) -> TrainingData:
        """All configured non-target source devices - the canonical hyperparam case."""
        target = config.target_device
        sources = frozenset(config.dataset_paths.keys()) - ({target} if target else set())
        return TrainingData(sources_unsorted=sources)

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
            raise ValueError(f"Profile study only uses physics normalization, but got {case.data_normalization}")

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
            "num_target_shots": case.num_target_shots,
            "target_test_set_size": config.target_test_set_size,
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
        val_eval_suite_config_base = {
            "loss_config": loss_config_base,
        }
        test_eval_suite_config_base = {
            "result_path": str(self.result_path(case)),
        }
        if case.domain_adaptation == "mixing":
            # Special logic for loss weighting when doing mixing domain adaptation
            # Weights are chosen so that each device's effective contribution F_x = W_x * N_x
            # (where N_x is the shot count)
            # Typically, the target device is weighted most heavily
            if config.dataset_sizes is None:
                logger.warning(
                    "Dataset sizes not provided in config, reading from disk. This will be slow, consider adding dataset sizes to the config."
                )
                dataset_sizes = {}
                for device, path in config.dataset_paths.items():
                    ds = xr.open_dataset(path)
                    dataset_sizes[device] = len(ds.shot)
                    ds.close()
            else:
                dataset_sizes = config.dataset_sizes

            if config.dataset_fractions is None:
                logger.info(
                    "Dataset fractions not provided in config. Using 50% for target and dividing remaining 50% evenly among sources."
                )
                dataset_fractions = {}
                num_sources = len(config.dataset_paths) - 1
                dataset_fractions[config.target_device] = 0.5
                for device in config.dataset_paths:
                    if device != config.target_device:
                        dataset_fractions[device] = 0.5 / num_sources
            else:
                dataset_fractions = config.dataset_fractions

            if case.num_target_shots in [-1, 0]:
                # If -1, all target shots are being included
                # If 0, weights aren't being used anyway
                N_target = dataset_sizes[config.target_device]
            else:
                N_target = case.num_target_shots

            avg_size = sum(dataset_sizes.values()) / len(dataset_sizes)
            dataset_weights = {}
            for device in config.dataset_paths.keys():
                if device == config.target_device:
                    N_x = N_target
                else:
                    N_x = dataset_sizes[device]
                F_x = dataset_fractions[device]
                W_x = F_x / N_x
                # Dividing by number of shots can make the weight very small, problematic for loss function
                # Multiply by avg_size so weights go back to around 1
                dataset_weights[device] = W_x * avg_size

            dataloader_config_base["dataset_weights"] = dataset_weights

        def _make_train_config_base(case: ProfileStudy.Case) -> TrainConfig:
            if case.model_type in ["shape_init_pca", "shape_init_kmeans"]:
                train_config_base = TrainConfig(
                    project=self.wandb_project_name(case),
                    train_run_builder="transport_study.modules.profile_predictor.trb.ProfilePredictorTRB",
                    max_epochs=config.max_epochs,
                    epochs_per_val=config.epochs_per_val,
                    patience=config.patience,
                    # When doing hyperparameter tuning, this gets overwritten by the wandb agent
                    checkpoint_dir=str(self.trained_model_dir(case)),
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
                        "in_size": 9,  # Ip_MA, B0, betan, ne20_edge, R0, a_minor, kappa, delta_top, delta_bot
                        "softmax_temp": 1,
                        "prng_seed": 42,
                    },
                    loss_config=loss_config_base,
                    optimizer_config=optimizer_config_base,
                    val_eval_suite_config=val_eval_suite_config_base,
                    test_eval_suite_config=test_eval_suite_config_base,
                )
            elif case.model_type == "unstructured_nn":
                train_config_base = TrainConfig(
                    project=self.wandb_project_name(case),
                    train_run_builder="transport_study.modules.profile_predictor.trb.ProfilePredictorTRB",
                    max_epochs=config.max_epochs,
                    epochs_per_val=config.epochs_per_val,
                    patience=config.patience,
                    # When doing hyperparameter tuning, this gets overwritten by the wandb agent
                    checkpoint_dir=str(self.trained_model_dir(case)),
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
                    val_eval_suite_config=val_eval_suite_config_base,
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
                num_target_shots=HYPERPARAM_TARGET_SHOTS,
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
        if tuned_config_path.exists():
            tuned_config = TrainConfig.load(str(tuned_config_path))
            logger.info(f"Found tuned hyperparameter config for case {case}, using hyperparameters from that config")
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
                            "softmax_temp": tuned_config.model_init_config["softmax_temp"],
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
        - num_target_shots(case_idx)
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
            if not result_path.exists():
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
                    "err_abs_shot_p25": err_abs_shot.quantile(0.25).drop_vars("quantile"),
                    "err_abs_shot_p75": err_abs_shot.quantile(0.75).drop_vars("quantile"),
                    "err_abs_shot_min": err_abs_shot.min(),
                    "err_abs_shot_max": err_abs_shot.max(),
                    "err_rel_shot_mean": err_rel_shot.mean(),
                    "err_rel_shot_std": err_rel_shot.std(),
                    "err_rel_shot_med": err_rel_shot.median(),
                    "err_rel_shot_p25": err_rel_shot.quantile(0.25).drop_vars("quantile"),
                    "err_rel_shot_p75": err_rel_shot.quantile(0.75).drop_vars("quantile"),
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
                    "training_data": str(case.training_data),
                    "data_normalization": case.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    "freeze_shapes": case.freeze_shapes,
                    "num_target_shots": case.num_target_shots,
                }
            )
            results.append(result)

        ds_merged = xr.concat(results, dim="case_idx")
        return ds_merged


def run_study(
    config: ProfileStudy.Config | str | Path,
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

    ###########################################
    # Initialize study and Set up directories #
    ###########################################

    study = ProfileStudy(config)

    def _setup_directories(study: ProfileStudy):
        logger.info("SETTING UP DIRECTORIES")
        logger.info(f"Enable parallelism: {enable_parallelism}")
        logger.info(f"Skip hyperparameter tuning: {skip_tuning}")
        logger.info(f"Skip visualization: {skip_visualization}")
        logger.info(f"Clean sweeps: {clean_sweeps}")
        logger.info(f"Clean models: {clean_models}")
        logger.info(f"Clean results: {clean_results}")
        logger.info(f"Clean figures: {clean_figures}")

        if (clean_sweeps or clean_models or clean_results or clean_figures) and enable_parallelism:
            raise ValueError(
                "Cannot clean models, results, or figures when parallelism is enabled, as this would interfere with jobs currently running or queued."
            )

        if (not skip_tuning) and (not enable_parallelism):
            logger.critical(
                "Hyperparameter tuning without parallelism enabled is probably gonna take a long time, are you sure you want to do this?"
            )

        if clean_sweeps:
            project_names = {study.wandb_project_name(case) for case in study.cases if case.is_hyperparam_case()}
            run_clean_sweeps(project_names)
        if clean_models:
            shutil.rmtree(study.model_dir, ignore_errors=True)
            shutil.rmtree(study.working_dir / "wandb", ignore_errors=True)
        if clean_results:
            shutil.rmtree(study.result_dir, ignore_errors=True)
        if clean_figures:
            shutil.rmtree(study.figure_dir, ignore_errors=True)

        for directory in [study.model_dir, study.result_dir, study.figure_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    _setup_directories(study)

    def _move_data(study: Study):
        logger.info("Moving data to cluster scratch for faster training")
        for ds_path in config.dataset_paths.values():
            if ds_path:
                file_name = Path(ds_path).name
                scratch_dir = Path(config.scratch_dir) / study.name
                scratch_dir.mkdir(parents=True, exist_ok=True)
                scratch_path = scratch_dir / file_name
                if not scratch_path.exists():
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
        logger.opt(colors=True).info("<bold><magenta>DATA VISUALIZATION</magenta></bold>")

    ########################
    # Launch Orchestration #
    ########################
    if study.collected_results_path().exists():
        logger.info(
            f"Collected results file found at\n{study.collected_results_path()}\nSkipping orchestration and going straight to analysis and visualization"
        )
    else:
        logger.opt(colors=True).info("<bold><magenta>ORCHESTRATION</magenta></bold>")

        # Unfinished cases are those we have data to run but haven't gotten results for yet
        unfinished_cases = [case for case in study.cases if not study.result_path(case).exists() and study.check_data_requirements(case)]
        while len(unfinished_cases) > 0:
            logger.opt(colors=True).info(f"<<bold><green>{len(unfinished_cases)} cases remain</green></bold>>")
            for case in unfinished_cases:
                if not study.result_path(case).exists():
                    study.run_case(
                        case,
                        skip_tuning=skip_tuning,
                        enable_parallelism=enable_parallelism,
                    )

            # Check which cases are still unfinished
            unfinished_cases = [case for case in unfinished_cases if not study.result_path(case).exists()]
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
