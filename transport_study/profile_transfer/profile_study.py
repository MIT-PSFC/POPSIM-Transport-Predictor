from __future__ import annotations

import os
import time
import tomllib
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import fire
import netCDF4  # noqa: F401
import numpy as np
import toml
import xarray as xr
from loguru import logger
from popsim.ml import TrainConfig
from pydantic import Field, field_validator, model_validator

from transport_study import PACKAGE_ROOT, TIME_DIM
from transport_study.config import StudyConfig, _env_dataset_paths, config, load_config
from transport_study.modules.profile_predictor.train_configs import (
    PROFILE_PREDICTOR_TORAX_CONFIGS,
)
from transport_study.orchestration.organize_data import (
    PROFILE_TARGET_VARS,
    TrainingData,
)
from transport_study.orchestration.study import Study
from transport_study.profile_transfer.data_visualization import DataVisualization

# When domain adaptation is None, we aren't using any target data during training anyway so this is unused
# During domain adaptation, we aren't doing hyperparameter tuning
HYPERPARAM_TARGET_SHOTS = 0

# Transfer cases fine-tune from a pretrained checkpoint, and the tuned learning
# rate they inherit was swept for training from scratch - too hot for fine-tuning
# (torax-constant transfer hit NaN loss with it). Scale the schedule down
TRANSFER_LR_FACTOR = 0.1


def _parse_training_data(s: str, dataset_paths: dict, target_device: str | None) -> TrainingData:
    """Convert a string like 'cmod_tcv' or 'exnihilo' to a TrainingData object."""
    if s == "exnihilo":
        non_target = set(dataset_paths.keys()) - ({target_device} if target_device else set())
        return TrainingData(sources_unsorted=non_target, exnihilo=True)
    return TrainingData(sources_unsorted=s.split("_"))


class ProfileStudy(Study):
    ##################
    # INITIALIZATION #
    ##################
    class Config(StudyConfig):
        # Organization for datasets and wandb projects
        working_dir_base: Path = Field(
            default_factory=lambda: Path(os.environ.get("PTPS_WORKING_DIR_BASE", str(PACKAGE_ROOT / "popsim_studies" / "working_dir")))
        )
        # The different cases being compared in this study
        model_types: tuple[str, ...] = Field(default_factory=lambda: ("shape_init_pca", "shape_init_kmeans", "unstructured_nn"))
        training_datasets: tuple[TrainingData, ...]
        domain_adaptation_methods: tuple[str | None, ...] = Field(default_factory=lambda: (None, "mixing", "transfer"))
        freeze_shapes_options: tuple[bool, ...] = Field(default_factory=lambda: (True,))
        num_target_shots_options: tuple[int, ...] = Field(default_factory=lambda: (0, 1, 10, -1))
        target_test_set_size: int
        # configurations for the hyperparameter tuning case
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
            valid = {"shape_init_pca", "shape_init_kmeans", "unstructured_nn", "reservoir", "torax-constant", "torax-cgm", "torax-gyrobohm"}
            for mt in v:
                if mt not in valid:
                    raise ValueError(f"Invalid model type: {mt}. Must be one of {sorted(valid)}.")
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
            # Field defaults aren't applied yet in a before-validator, so when
            # dataset_paths isn't passed explicitly, mirror its default_factory
            dataset_paths = dict(data.get("dataset_paths") or _env_dataset_paths())
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
            # PTPS_DATASET_PATHS env var provides defaults, explicit TOML paths win
            return cls(
                **data,
                **study_cases,
                dataset_paths=_env_dataset_paths() | {k: Path(v) for k, v in datasets.items()},
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
            # load_config(Path) only knows how to build the base StudyConfig,
            # which forbids this study's extra fields, so parse with the subclass
            if isinstance(cfg, (str, Path)):
                cfg = self.Config.from_toml(Path(cfg))
            load_config(cfg)

        cases = self.make_cases(
            config.model_types,
            config.training_datasets,
            config.domain_adaptation_methods,
            config.freeze_shapes_options,
            config.num_target_shots_options,
        )
        super().__init__(config.study_name, config.working_dir_base, cases)

        logger.info(f"Model types: {config.model_types}")
        logger.info(f"Training datasets: {config.training_datasets}")
        logger.info(f"Domain adaptation methods: {config.domain_adaptation_methods}")
        logger.info(f"Freeze shapes options: {config.freeze_shapes_options}")
        logger.info(f"Number of target shots included in training options: {config.num_target_shots_options}")

    @dataclass
    class Case(Study.Case):
        """
        model_type: The type of profile_predictor model to use
        - shape_init: Use principal component analysis to determine dominant shapes
        - unstructured_nn: a single neural network directly predicts profiles at certain points
        - torax-constant / torax-cgm / torax-gyrobohm: TORAX simulation with NN-predicted
          parameters for the constant, critical gradient, or Bohm-GyroBohm transport model

        training_data: The dataset(s) used for training
        - cmod: C-Mod only
        - tcv: TCV only
        - cmod_tcv: C-Mod + TCV
        - exnihilo: No historic training data

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
        domain_adaptation: str  # none, mixing, transfer
        freeze_shapes: bool
        num_target_shots: int  # Number of target shots included in training, or -1 for all (should be HYPERPARAM_TARGET_SHOTS if domain_adaptation is None)
        prereqs: (
            list[Study.Case] | None
        )  # If not None, this case depends on the results of another case, and should only be run after that case has been run

        def is_hyperparam_case(self) -> bool:
            if (
                self.training_data == ProfileStudy._hyperparam_training_data()
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
                    domain_adaptation=config.hyperparam_domain_adaptation,
                    freeze_shapes=config.hyperparam_freeze_shapes,
                    num_target_shots=config.hyperparam_num_target_shots,
                )

        def __init__(
            self,
            model_type: str,
            training_data: TrainingData | str,
            domain_adaptation: str,
            freeze_shapes: bool,
            num_target_shots: int,
        ):
            if isinstance(training_data, str):
                training_data = _parse_training_data(training_data, dict(config.dataset_paths), config.target_device)
            self.model_type = model_type
            self.training_data = training_data
            self.domain_adaptation = domain_adaptation
            self.freeze_shapes = freeze_shapes
            self.num_target_shots = num_target_shots

            # Recursively add prereqs based on the logic of which cases depend on which other cases
            if model_type not in [
                "shape_init_pca",
                "shape_init_kmeans",
                "unstructured_nn",
                "reservoir",
                "torax-constant",
                "torax-cgm",
                "torax-gyrobohm",
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
                        self.freeze_shapes,
                        self.num_target_shots,
                    )
                )
            else:
                return hash(
                    (
                        self.model_type,
                        self.training_data,
                        self.domain_adaptation,
                        self.freeze_shapes,
                    )
                )

    def make_cases(
        self,
        model_types,
        training_datasets,
        domain_adaptation_methods,
        freeze_shapes_options,
        num_target_shots_options,
    ):
        cases = []
        # Make every case we're interested in for this study
        for (
            model_type,
            training_dataset,
            domain_adaptation,
            freeze_shapes,
            num_target_shots,
        ) in product(
            model_types,
            training_datasets,
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
            if model_type in ("unstructured_nn", "reservoir") and not freeze_shapes:
                continue  # No shapes to freeze, just do one of the two

            case = self.Case(
                model_type=model_type,
                training_data=training_dataset,
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

        unique_cases = sorted(set(unwrapped_cases), key=str)
        possible_cases = [case for case in unique_cases if not case.is_impossible()]  # Remove impossible cases

        return possible_cases

    @classmethod
    def _hyperparam_training_data(cls) -> TrainingData:
        """All configured non-target source devices - the canonical hyperparam case."""
        target = config.target_device
        sources = sorted(set(config.dataset_paths.keys()) - ({target} if target else set()))
        return TrainingData(sources_unsorted=sources)

    #############
    # EXECUTION #
    #############

    @staticmethod
    def _make_mixing_device_weights(case: Case) -> dict[str, float]:
        """Loss weights per device for mixing domain adaptation.

        Weights are chosen so that each device's effective contribution
        F_x = W_x * N_x (where N_x is the shot count). Typically the target
        device is weighted most heavily.
        """
        if not config.dataset_sizes:
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

        if not config.dataset_fractions:
            logger.info("Dataset fractions not provided in config. Using 50% for target and dividing remaining 50% evenly among sources.")
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

        return dataset_weights

    def make_train_config(self, case: Case) -> TrainConfig:
        """Make the TrainConfig for a given case
        If a hyperparameter tuned config is available, fills in the hyperparameters from that, otherwise uses default config.
        """
        dataloader_config_base = {
            # Profiles plus their gradient / error-bar companions, the loss
            # uses the error bars as an epsilon-insensitive deadband
            "target_vars": [*PROFILE_TARGET_VARS, "ds_source_idx"],
            "training_data": case.training_data,
            "domain_adaptation": case.domain_adaptation,
            "num_target_shots": case.num_target_shots,
            "target_test_set_size": config.target_test_set_size,
            "prng_seed": 42,
            "debug": config.debug,
            # Hyperparameters
            "batch_size": 8192,
        }
        loss_config_base = {
            # The loss runs on peak-normalized profiles (target scaled to max 1),
            # so both deltas read as fractional errors. Both are swept
            # hyperparameters (see sweep_configs/*.yaml), these values are the
            # fallback for cases run without a tuned config. Validation loss is
            # delta-free (get_val_loss_fn), so the sweep metric cannot be gamed
            # by shrinking the deltas
            "huber_delta": 0.1,
            # Penalize profile gradient mismatch too, since stability predictions
            # depend on dTe/drho and dne/drho. Normalized gradients are ~10x the
            # normalized value scale over rho in [0, 1], so they get their own delta.
            "gradient_weight": 0.1,
            "huber_delta_grad": 1.0,
            # Residual inside the GP-fit error bar is down-weighted by this
            # factor: predictions are still pulled toward the fit mean, but
            # landing within the error bars costs much less than missing them
            "within_error_weight": 0.25,
        }
        optimizer_config_base = {
            "lr0": 1e-4,
            "transition_steps": 500,
            "decay_rate": 0.5,
            "lrf": 5e-4,
            "weight_decay": 2e-4,
            # Cap on global L2 gradient norm per update, guards against rare
            # gradient spikes from the differentiated TORAX solve NaN-ing a run
            "grad_clip_max_norm": 1.0,
        }
        val_eval_suite_config_base = {
            "loss_config": loss_config_base,
        }
        test_eval_suite_config_base = {
            "result_path": str(self.result_path(case)),
        }
        if case.domain_adaptation == "mixing":
            # Loss function reads these from loss_config as "device_weights".
            # val_eval_suite_config_base references the same dict, so validation
            # loss is weighted consistently with training
            loss_config_base["device_weights"] = self._make_mixing_device_weights(case)

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
                        "input_vars": [
                            "Ip_MA",
                            "B0",
                            "betan",
                            "ne20_line_avg",
                            "R0",
                            "a_minor",
                            "kappa",
                            "delta_top",
                            "delta_bot",
                        ],
                        "extra_vars": ["Te_shape", "ne_shape"],
                        **dataloader_config_base,
                    },
                    model_init_config={
                        "model_type": case.model_type,
                        "domain_adaptation": case.domain_adaptation,
                        "freeze_shapes": case.freeze_shapes,
                        "te_shape_var": "Te_shape",
                        "ne_shape_var": "ne_shape",
                        "n_shapes": 3,
                        "nn_depth": 2,
                        "nn_width": 16,
                        "in_size": 9,  # Ip_MA, B0, betan, ne20_line_avg, R0, a_minor, kappa, delta_top, delta_bot
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
                        "input_vars": [
                            "Ip_MA",
                            "B0",
                            "betan",
                            "ne20_line_avg",
                            "R0",
                            "a_minor",
                            "kappa",
                            "delta_top",
                            "delta_bot",
                        ],
                        **dataloader_config_base,
                    },
                    model_init_config={
                        "model_type": case.model_type,
                        "domain_adaptation": case.domain_adaptation,
                        "n_points": 21,  # Number of points along the profile to predict for the unstructured NN
                        "nn_depth": 2,
                        "nn_width": 16,
                        "in_size": 9,  # Ip_MA, B0, betan, ne20_line_avg, R0, a_minor, kappa, delta_top, delta_bot
                        "prng_seed": 42,
                    },
                    loss_config=loss_config_base,
                    optimizer_config=optimizer_config_base,
                    val_eval_suite_config=val_eval_suite_config_base,
                    test_eval_suite_config=test_eval_suite_config_base,
                )
            elif case.model_type == "reservoir":
                train_config_base = TrainConfig(
                    project=self.wandb_project_name(case),
                    train_run_builder="transport_study.modules.profile_predictor.trb.ProfilePredictorTRB",
                    max_epochs=config.max_epochs,
                    epochs_per_val=config.epochs_per_val,
                    patience=config.patience,
                    # When doing hyperparameter tuning, this gets overwritten by the wandb agent
                    checkpoint_dir=str(self.trained_model_dir(case)),
                    dataloader_config={
                        "input_vars": [
                            "Ip_MA",
                            "B0",
                            "betan",
                            "ne20_line_avg",
                            "R0",
                            "a_minor",
                            "kappa",
                            "delta_top",
                            "delta_bot",
                        ],
                        **dataloader_config_base,
                    },
                    model_init_config={
                        "model_type": case.model_type,
                        "domain_adaptation": case.domain_adaptation,
                        "n_points": 21,  # Number of points along the profile predicted by the readout
                        "reservoir_size": 128,  # Fixed random reservoir state dimension
                        "spectral_radius": 0.9,  # Contraction factor of the recurrent weights
                        "input_scaling": 0.5,  # Scale of the random input weights and bias
                        "leak_rate": 1.0,  # Leaky integration rate of the state update
                        "n_steps": 20,  # Reservoir iterations before readout
                        "in_size": 9,  # Ip_MA, B0, betan, ne20_line_avg, R0, a_minor, kappa, delta_top, delta_bot
                        "prng_seed": 42,
                    },
                    loss_config=loss_config_base,
                    optimizer_config=optimizer_config_base,
                    val_eval_suite_config=val_eval_suite_config_base,
                    test_eval_suite_config=test_eval_suite_config_base,
                )
            elif case.model_type.startswith("torax-"):
                transport_model = case.model_type.removeprefix("torax-")
                train_config_base = TrainConfig(
                    project=self.wandb_project_name(case),
                    train_run_builder="transport_study.modules.profile_predictor.trb.ProfilePredictorTRB",
                    max_epochs=config.max_epochs,
                    epochs_per_val=config.epochs_per_val,
                    patience=config.patience,
                    # When doing hyperparameter tuning, this gets overwritten by the wandb agent
                    checkpoint_dir=str(self.trained_model_dir(case)),
                    dataloader_config={
                        "input_vars": [
                            "Ip_MA",
                            "B0",
                            "betan",
                            "ne20_line_avg",
                            "R0",
                            "a_minor",
                            "kappa",
                            "delta_top",
                            "delta_bot",
                        ],
                        **dataloader_config_base,
                        # Each sample in a batch runs a differentiated multi-step
                        # TORAX simulation under vmap, so activation memory scales
                        # linearly with batch size (~11 MB/sample observed)
                        # 1024 keeps it near ~11 GB with headroom on an A100
                        "batch_size": 1024,
                    },
                    model_init_config={
                        "model_type": case.model_type,
                        "domain_adaptation": case.domain_adaptation,
                        "freeze_shapes": case.freeze_shapes,
                        "nn_depth": 2,
                        "nn_width": 16,
                        "in_size": 9,  # Ip_MA, B0, betan, ne20_line_avg, R0, a_minor, kappa, delta_top, delta_bot
                        "torax_config": PROFILE_PREDICTOR_TORAX_CONFIGS[transport_model]["model_init_config"]["torax_config"],
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
            # Pretrained model has same model_type and training_data,
            # but no domain adaptation and no high-performance shots
            transfer_case = ProfileStudy.Case(
                model_type=case.model_type,
                training_data=case.training_data,
                domain_adaptation=None,
                freeze_shapes=case.freeze_shapes,
                num_target_shots=HYPERPARAM_TARGET_SHOTS,
            )
            transfer_case_model_dir = str(self.trained_model_dir(transfer_case))

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
            # Only the deltas come from the tuned loss_config: device_weights and
            # gradient_weight stay case-specific (mixing weights differ per case)
            train_config = train_config_base.model_copy(
                update={
                    "dataloader_config": {
                        **train_config_base.dataloader_config,
                        "batch_size": tuned_config.dataloader_config["batch_size"],
                    },
                    "optimizer_config": tuned_config.optimizer_config,
                    "loss_config": {
                        **train_config_base.loss_config,
                        "huber_delta": tuned_config.loss_config.get("huber_delta", train_config_base.loss_config["huber_delta"]),
                        "huber_delta_grad": tuned_config.loss_config.get(
                            "huber_delta_grad", train_config_base.loss_config["huber_delta_grad"]
                        ),
                    },
                }
            )

            # Hyperparameters swept for only certain modules
            # The reservoir has no MLP depth/width, everything else sweeps them
            def _tuned_model_init_updates() -> dict:
                updates = {}
                if case.model_type != "reservoir":
                    updates["nn_depth"] = tuned_config.model_init_config["nn_depth"]
                    updates["nn_width"] = tuned_config.model_init_config["nn_width"]
                if case.model_type in ["shape_init_pca", "shape_init_kmeans"]:
                    updates["n_shapes"] = tuned_config.model_init_config["n_shapes"]
                    updates["softmax_temp"] = tuned_config.model_init_config["softmax_temp"]
                elif case.model_type in ["unstructured_nn"]:
                    updates["n_points"] = tuned_config.model_init_config["n_points"]
                elif case.model_type == "reservoir":
                    updates["n_points"] = tuned_config.model_init_config["n_points"]
                    updates["reservoir_size"] = tuned_config.model_init_config["reservoir_size"]
                    updates["spectral_radius"] = tuned_config.model_init_config["spectral_radius"]
                    updates["input_scaling"] = tuned_config.model_init_config["input_scaling"]
                    updates["leak_rate"] = tuned_config.model_init_config["leak_rate"]
                    updates["n_steps"] = tuned_config.model_init_config["n_steps"]
                return updates

            train_config = train_config.model_copy(
                update={
                    "model_init_config": {
                        **train_config.model_init_config,
                        **_tuned_model_init_updates(),
                    }
                }
            )
        else:
            train_config = train_config_base

        # Fine-tuning from a pretrained checkpoint needs a cooler learning rate
        # than training from scratch (see TRANSFER_LR_FACTOR). Applied after the
        # tuned-config merge so the swept optimizer_config cannot overwrite it
        if case.domain_adaptation == "transfer":
            train_config = train_config.model_copy(
                update={
                    "optimizer_config": {
                        **train_config.optimizer_config,
                        "lr0": train_config.optimizer_config["lr0"] * TRANSFER_LR_FACTOR,
                        "lrf": train_config.optimizer_config["lrf"] * TRANSFER_LR_FACTOR,
                    }
                }
            )

        return train_config

    ##############
    # COLLECTION #
    ##############

    # Coords describing which case a record belongs to (broadcast over every shot of that case).
    _CASE_COORD_NAMES = (
        "case_idx",
        "model_type",
        "training_data",
        "domain_adaptation",
        "freeze_shapes",
        "num_target_shots",
    )

    def _case_coords(self, case_idx: int, case) -> dict:
        """Build the per-case coordinate values shared by every shot of a case."""
        return {
            "case_idx": case_idx,
            "model_type": case.model_type,
            "training_data": str(case.training_data),
            # Normalize None -> "none" so the coord stays string-typed
            "domain_adaptation": case.domain_adaptation if case.domain_adaptation is not None else "none",
            "freeze_shapes": case.freeze_shapes,
            "num_target_shots": case.num_target_shots,
        }

    def collect_results(self):
        """Collect per-shot test errors from all finished cases into one tidy (long-form) dataset.

        TODO(ZanderKeith): Cristina really wants more fine-grained statistics for specific situations
        - rampup vs flattop vs rampdown
        - H mode vs L mode
        - disruptive vs non-disruptive shots

        Each row is one (case, shot) pair so individual shots where a model struggles can be
        inspected directly (e.g. sort by ``err_abs_shot`` within a ``model_type`` group).

        Dims: record (flat index over all case x shot pairs)
        Coords (along record):
        - case_idx, model_type, training_data, domain_adaptation,
          freeze_shapes, num_target_shots (identify the case)
        - shot (device shot id), ds_source (which dataset the shot came from)
        Data variables (along record):
        - err_abs_shot / err_rel_shot: time-integrated combined (ne+Te) error for the shot
        - ne_err_abs_shot / te_err_abs_shot / ne_err_rel_shot / te_err_rel_shot: per-channel
          time-integrated errors (see which channel drives a bad shot)
        - err_abs_ts_max / err_rel_ts_max: worst single timeslice in the shot
        - err_abs_ts_mean / err_rel_ts_mean: mean over the shot's timeslices
        - n_valid_ts: number of non-NaN timeslices contributing to the shot

        Use ``collect_case_summary`` for the older scalar-per-case aggregate view.
        """
        data_var_names = [
            "err_abs_shot",
            "err_rel_shot",
            "ne_err_abs_shot",
            "te_err_abs_shot",
            "ne_err_rel_shot",
            "te_err_rel_shot",
            "err_abs_ts_max",
            "err_rel_ts_max",
            "err_abs_ts_mean",
            "err_rel_ts_mean",
            "n_valid_ts",
        ]

        records: list[dict] = []
        for case_idx, case in enumerate(self.cases):
            result_path = self.result_path(case)
            if not result_path.exists():
                continue

            ds = xr.load_dataset(result_path)

            # Reduce per-timeslice errors to per-shot summaries (worst and mean timeslice)
            err_abs_ts = ds["error_abs_ts"]
            err_rel_ts = ds["error_rel_ts"]
            per_shot = {
                "err_abs_shot": ds["error_abs_shot"].values,
                "err_rel_shot": ds["error_rel_shot"].values,
                "ne_err_abs_shot": ds["ne_error_abs_shot"].values,
                "te_err_abs_shot": ds["te_error_abs_shot"].values,
                "ne_err_rel_shot": ds["ne_error_rel_shot"].values,
                "te_err_rel_shot": ds["te_error_rel_shot"].values,
                "err_abs_ts_max": err_abs_ts.max(TIME_DIM, skipna=True).values,
                "err_rel_ts_max": err_rel_ts.max(TIME_DIM, skipna=True).values,
                "err_abs_ts_mean": err_abs_ts.mean(TIME_DIM, skipna=True).values,
                "err_rel_ts_mean": err_rel_ts.mean(TIME_DIM, skipna=True).values,
                "n_valid_ts": err_abs_ts.notnull().sum(TIME_DIM).values,
            }

            shot_ids = ds["shot"].values
            n_shots = len(shot_ids)
            ds_source = ds["ds_source"].values if "ds_source" in ds.coords else np.array([""] * n_shots)
            case_coords = self._case_coords(case_idx, case)

            for i in range(n_shots):
                record = {name: per_shot[name][i] for name in data_var_names}
                record.update(case_coords)
                record["shot"] = shot_ids[i]
                record["ds_source"] = ds_source[i]
                records.append(record)

        if not records:
            return xr.Dataset()

        coord_names = [*self._CASE_COORD_NAMES, "shot", "ds_source"]
        data_vars = {name: ("record", np.array([r[name] for r in records])) for name in data_var_names}
        coords = {name: ("record", np.array([r[name] for r in records])) for name in coord_names}
        return xr.Dataset(data_vars=data_vars, coords=coords)

    def collect_case_summary(self):
        """Collect scalar summary statistics per case (one row per case).

        Reduces each case's per-shot and per-timeslice error distributions to scalar
        statistics. Use ``collect_results`` for the per-shot view.

        Dims: case_idx
        Coords (along case_idx): model_type, training_data,
        domain_adaptation, freeze_shapes, num_target_shots
        Data variables (along case_idx): err_E_D_S where E is 'abs' or 'rel', D is
        'shot' (time-integrated per shot) or 'ts' (per timeslice), and S is one of
        mean, std, med, p25, p75, min, max. E.g. err_abs_shot_mean, err_rel_ts_p75.
        """
        results = []
        for case_idx, case in enumerate(self.cases):
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
            ).assign_coords(self._case_coords(case_idx, case))
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

    # Parse with the concrete Config subclass so the local `config` name holds a
    # real config object (it shadows the module-level proxy below)
    if isinstance(config, (str, Path)):
        config = ProfileStudy.Config.from_toml(Path(config))
    study = ProfileStudy(config)
    study.setup_directories(
        enable_parallelism=enable_parallelism,
        skip_tuning=skip_tuning,
        skip_visualization=skip_visualization,
        clean_sweeps=clean_sweeps,
        clean_models=clean_models,
        clean_results=clean_results,
        clean_figures=clean_figures,
    )

    if enable_parallelism and not config.partition:
        raise ValueError("enable_parallelism is True but no SLURM partition is specified in the config")

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
        DataVisualization.performance_extrapolation(study.figure_dir)
        DataVisualization.domain_overlap(study.figure_dir)

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
