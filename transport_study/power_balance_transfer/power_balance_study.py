from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import fire
import netCDF4  # noqa: F401
import xarray as xr
from loguru import logger
from popsim.ml import TrainConfig
from pydantic import Field, field_validator

from transport_study import PACKAGE_ROOT
from transport_study.config import config, load_config
from transport_study.orchestration.organize_data import (
    TrainingData,
    parse_training_data,
)
from transport_study.orchestration.study import CaseGridConfig, Study
from transport_study.power_balance_transfer.data_visualization import DataVisualization

# When domain adaptation is None, we aren't using any target data during training anyway so this is unused
# During domain adaptation, we aren't doing hyperparameter tuning
HYPERPARAM_TARGET_SHOTS = 0

# The 7 physical inputs every power-balance model consumes. Normalization is
# done inside the modules (transport_study.modules.normalization), so the
# dataloader always pulls exactly these (the TRB adds ds_source_idx itself)
POWER_BALANCE_INPUT_VARS = [
    "Ip_MA",
    "B0",
    "R0",
    "a_minor",
    "kappa",
    "ne20_line_avg",
    "P_aux_MW",
]

# Model types with p_oh/p_rad submodules (the SciML-style structured models)
MODEL_TYPES_WITH_SUBMODULES = ("scaling_law", "sciml")
# Purely data-driven model types, no submodules so nothing to freeze
MODEL_TYPES_WITHOUT_SUBMODULES = ("unstructured_nn", "transformer")
# Submodule pseudo-model-types, they appear as prereq cases of the structured models
SUBMODULE_MODEL_TYPES = ("p_oh", "p_rad")


class PowerBalanceStudy(Study):
    SWEEP_CONFIG_DIR = Path(PACKAGE_ROOT) / "power_balance_transfer" / "sweep_configs"
    STUDY_TYPE = "power_balance_transfer"

    ##################
    # INITIALIZATION #
    ##################
    class Config(CaseGridConfig):
        # The different cases being compared in this study
        model_types: tuple[str, ...] = Field(default_factory=lambda: ("scaling_law", "sciml", "unstructured_nn", "transformer"))
        data_normalization_methods: tuple[str, ...] = Field(default_factory=lambda: ("raw", "physics", "z_score", "coral"))
        freeze_submodules_options: tuple[bool, ...] = Field(default_factory=lambda: (True,))
        num_target_shots_options: tuple[int, ...] = Field(default_factory=lambda: (0, 1, 3, 10, 32, -1))
        # configurations for the hyperparameter tuning case
        hyperparam_data_normalization: str = "coral"
        hyperparam_domain_adaptation: str | None = None
        hyperparam_freeze_submodules: bool = True
        hyperparam_num_target_shots: int = HYPERPARAM_TARGET_SHOTS
        # Misc configurations
        dataset_fractions: dict[str, float] = Field(
            default_factory=dict
        )  # Optional dict of dataset fractions to use during domain adaptation, only used if domain_adaptation includes "mixing".

        @field_validator("model_types")
        @classmethod
        def _validate_model_types(cls, v: tuple[str, ...]) -> tuple[str, ...]:
            valid = {*MODEL_TYPES_WITH_SUBMODULES, *MODEL_TYPES_WITHOUT_SUBMODULES, *SUBMODULE_MODEL_TYPES}
            for mt in v:
                if mt not in valid:
                    raise ValueError(f"Invalid model type: {mt}. Must be one of {sorted(valid)}.")
            return v

        @field_validator("data_normalization_methods", "hyperparam_data_normalization")
        @classmethod
        def _validate_data_normalization(cls, v):
            valid = {"raw", "physics", "z_score", "coral"}
            methods = (v,) if isinstance(v, str) else v
            for dn in methods:
                if dn not in valid:
                    raise ValueError(f"Invalid data normalization method: {dn}. Must be one of {sorted(valid)}.")
            return v

        def is_compatible(self, cfg: PowerBalanceStudy.Config) -> bool:
            """Check if two configs are compatible for running the same study
            For the PowerBalanceStudy, that means the following:
            1: study name must match (used in WandB sweeps, etc.)
            2: dataset paths must be identical
            3: target device must be the same
            4: target test set size must be the same
            5: hyperparameter tuning configs must match
            6: Dataset fractions must match
            """
            return (
                self.study_name == cfg.study_name
                and self.dataset_paths == cfg.dataset_paths
                and self.target_device == cfg.target_device
                and self.target_test_set_size == cfg.target_test_set_size
                and self.hyperparam_data_normalization == cfg.hyperparam_data_normalization
                and self.hyperparam_domain_adaptation == cfg.hyperparam_domain_adaptation
                and self.hyperparam_freeze_submodules == cfg.hyperparam_freeze_submodules
                and self.hyperparam_num_target_shots == cfg.hyperparam_num_target_shots
                and self.dataset_fractions == cfg.dataset_fractions
            )

    def __init__(
        self,
        cfg: str | Path | PowerBalanceStudy.Config,
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
            config.data_normalization_methods,
            config.domain_adaptation_methods,
            config.freeze_submodules_options,
            config.num_target_shots_options,
        )
        super().__init__(config.study_name, config.working_dir_base, cases)

        logger.info(f"Model types: {config.model_types}")
        logger.info(f"Training datasets: {config.training_datasets}")
        logger.info(f"Data normalization methods: {config.data_normalization_methods}")
        logger.info(f"Domain adaptation methods: {config.domain_adaptation_methods}")
        logger.info(f"Freeze submodules options: {config.freeze_submodules_options}")
        logger.info(f"Number of target shots included in training options: {config.num_target_shots_options}")

    @dataclass
    class Case(Study.Case):
        """
        model_type: The type of power_balance model to use.
        - scaling_law: H89, H98, and P_LH scaling laws to predict tau_e
        - sciml: neural network predicts tau_e, and we do the power balance calculation
        - unstructured_nn: a simple MLP directly predicts stored energy evolution
        - transformer: recurrent causal attention over past inputs directly predicts stored energy evolution
        - p_oh / p_rad: submodule predictors, appear only as prereq cases of scaling_law and sciml

        training_data: The dataset(s) used for training
        - cmod: C-Mod only
        - tcv: TCV only
        - cmod_tcv: C-Mod + TCV
        - exnihilo: No historic training data

        data_normalization: The method for normalizing the model's NN inputs, implemented
        as a POPSIM module configured from the training data only (transport_study/modules/normalization.py).
        - raw: No normalization, Ip, Wtot, etc. are in their original units
        - physics: Convert to typical dimensionless parameters like q_star, f_G, etc.
        - z_score: Within each device, normalize each variable to zero mean and unit variance.
        - coral: Use the CORAL method to align covariances of source and target domains (https://arxiv.org/abs/1612.01939)

        domain_adaptation: The method for domain adaptation between source and target devices.
        - none: No domain adaptation, train and test on the same device(s). This is used for hyperparameter tuning and as a baseline for comparison, answering the question "what is the best possible performance we could expect if we had a bunch of data?"
        - mixing: Add a small amount of highly-weighted target data during training
        - transfer: Train on source data, freeze all but the last layers of the model, and fine-tune on a small amount of target data

        freeze_submodules: Whether to freeze the p_oh/p_rad submodules of the model during training.
        The P_oh and P_rad signals are hard to quantify, we might want to let them drift from the original targets to better match Wtot_MJ

        num_target_shots: The number of shots included in the training data from the target dataset, or -1 to include all shots (including all shots in training is cheating, but again answers the question of what is the best possible performance).
        """

        model_type: str
        training_data: TrainingData
        data_normalization: str  # raw, physics, z_score, coral
        domain_adaptation: str  # none, mixing, transfer
        freeze_submodules: bool
        num_target_shots: int  # Number of target shots included in training, or -1 for all (should be HYPERPARAM_TARGET_SHOTS if domain_adaptation is None)
        # If not None, this case depends on the results of another case, and should only be run after that case has been run
        prereqs: list[Study.Case] | None

        def is_hyperparam_case(self) -> bool:
            if (
                self.training_data == PowerBalanceStudy._hyperparam_training_data()
                and self.data_normalization == config.hyperparam_data_normalization
                and self.domain_adaptation == config.hyperparam_domain_adaptation
                and self.freeze_submodules == config.hyperparam_freeze_submodules
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
                return PowerBalanceStudy.Case(
                    model_type=self.model_type,
                    training_data=PowerBalanceStudy._hyperparam_training_data(),
                    data_normalization=config.hyperparam_data_normalization,
                    domain_adaptation=config.hyperparam_domain_adaptation,
                    freeze_submodules=config.hyperparam_freeze_submodules,
                    num_target_shots=config.hyperparam_num_target_shots,
                )

        def __init__(
            self,
            model_type: str,
            training_data: TrainingData | str,
            data_normalization: str,
            domain_adaptation: str,
            freeze_submodules: bool,
            num_target_shots: int,
        ):
            if isinstance(training_data, str):
                training_data = parse_training_data(training_data, dict(config.dataset_paths), config.target_device)
            self.model_type = model_type
            self.training_data = training_data
            self.data_normalization = data_normalization
            self.domain_adaptation = domain_adaptation
            self.freeze_submodules = freeze_submodules
            self.num_target_shots = num_target_shots

            if model_type not in [*MODEL_TYPES_WITH_SUBMODULES, *MODEL_TYPES_WITHOUT_SUBMODULES, *SUBMODULE_MODEL_TYPES]:
                raise ValueError(f"Unknown model type: {model_type}")
            if data_normalization not in ["raw", "physics", "z_score", "coral"]:
                raise ValueError(f"Unknown data normalization method: {data_normalization}")
            if domain_adaptation is None:
                if not training_data.exnihilo and num_target_shots != HYPERPARAM_TARGET_SHOTS:
                    raise ValueError(
                        "If domain_adaptation is None and training data is not 'exnihilo', num_target_shots must be HYPERPARAM_TARGET_SHOTS since this means we're training and testing on the same dataset"
                    )
            if model_type in SUBMODULE_MODEL_TYPES and freeze_submodules != config.hyperparam_freeze_submodules:
                raise ValueError(
                    f"freeze_submodules should be a dummy value ({config.hyperparam_freeze_submodules}) for submodule {model_type}"
                )

            # Recursively add prereqs based on the logic of which cases depend on which other cases
            prereqs = []

            # Add prereqs based on whether this is a hyperparameter tuning case or not.
            if not self.is_hyperparam_case():
                prereqs += [
                    PowerBalanceStudy.Case(
                        model_type=model_type,
                        training_data=PowerBalanceStudy._hyperparam_training_data(),
                        data_normalization=config.hyperparam_data_normalization,
                        domain_adaptation=config.hyperparam_domain_adaptation,
                        freeze_submodules=config.hyperparam_freeze_submodules,
                        num_target_shots=config.hyperparam_num_target_shots,
                    )
                ]

            # Set prereqs based on model type: the structured models restore
            # pre-trained p_oh/p_rad submodules
            if model_type in MODEL_TYPES_WITH_SUBMODULES:
                for submodule_type in SUBMODULE_MODEL_TYPES:
                    prereqs += [
                        PowerBalanceStudy.Case(
                            model_type=submodule_type,
                            training_data=training_data,
                            data_normalization=data_normalization,
                            domain_adaptation=domain_adaptation,
                            freeze_submodules=config.hyperparam_freeze_submodules,
                            num_target_shots=num_target_shots,
                        )
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
                return f"case.{self.model_type}.td_{self.training_data}.norm_{self.data_normalization}.freeze_{self.freeze_submodules}.targ_{self.num_target_shots}.da_{self.domain_adaptation}"
            elif self.training_data.exnihilo:
                return f"case.{self.model_type}.td_{self.training_data}.norm_{self.data_normalization}.freeze_{self.freeze_submodules}.targ_{self.num_target_shots}"
            else:
                return f"case.{self.model_type}.td_{self.training_data}.norm_{self.data_normalization}.freeze_{self.freeze_submodules}"

        def __hash__(self):
            if self.domain_adaptation:
                return hash(
                    (
                        self.model_type,
                        self.training_data,
                        self.data_normalization,
                        self.domain_adaptation,
                        self.freeze_submodules,
                        self.num_target_shots,
                    )
                )
            elif self.training_data.exnihilo:
                return hash(
                    (
                        self.model_type,
                        self.training_data,
                        self.data_normalization,
                        self.freeze_submodules,
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
        num_target_shots_options,
    ):
        cases = []
        # Make every case we're interested in for this study
        for (
            model_type,
            training_dataset,
            data_normalization,
            domain_adaptation,
            freeze_submodules,
            num_target_shots,
        ) in product(
            model_types,
            training_datasets,
            data_normalization_methods,
            domain_adaptation_methods,
            freeze_submodules_options,
            num_target_shots_options,
        ):
            if domain_adaptation is None:
                if training_dataset.exnihilo:
                    if num_target_shots == 0:
                        continue  # Can't train from nothing with 0 target shots
                elif num_target_shots != HYPERPARAM_TARGET_SHOTS:
                    continue  # Invalid case, skip
            if model_type in MODEL_TYPES_WITHOUT_SUBMODULES and freeze_submodules != config.hyperparam_freeze_submodules:
                continue  # No submodules to freeze, just do one of the two

            case = self.Case(
                model_type=model_type,
                training_data=training_dataset,
                data_normalization=data_normalization,
                domain_adaptation=domain_adaptation,
                freeze_submodules=freeze_submodules,
                num_target_shots=num_target_shots,
            )

            cases.append(case)

        return self.finalize_cases(cases)

    #############
    # EXECUTION #
    #############

    def make_train_config(self, case: Case) -> TrainConfig:
        """Make the TrainConfig for a given case.
        If a hyperparameter tuned config is available, fills in the hyperparameters from that, otherwise uses default config.
        """
        dataloader_config_base = {
            "training_data": case.training_data,
            "domain_adaptation": case.domain_adaptation,
            "num_target_shots": case.num_target_shots,
            "target_test_set_size": config.target_test_set_size,
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
            # Loss function reads these from loss_config as "device_weights".
            # val_eval_suite_config_base references the same dict, so validation
            # loss is weighted consistently with training
            loss_config_base["device_weights"] = self._make_mixing_device_weights(case)

        def _make_submodule_config(submodule_type: str) -> TrainConfig:
            return self.make_train_config(
                self.Case(
                    model_type=submodule_type,
                    training_data=case.training_data,
                    data_normalization=case.data_normalization,
                    domain_adaptation=case.domain_adaptation,
                    freeze_submodules=config.hyperparam_freeze_submodules,
                    num_target_shots=case.num_target_shots,
                )
            )

        def _make_train_config_base(case: PowerBalanceStudy.Case) -> TrainConfig:
            if case.model_type in SUBMODULE_MODEL_TYPES:
                submodule_settings = {
                    "p_oh": {
                        "train_run_builder": "transport_study.modules.power_balance.p_oh.trb.OhmicPowerTRB",
                        "target_vars": ["P_oh_MW", "ds_source_idx"],
                        "max_val": 16,  # Maximum ohmic power in MW
                    },
                    "p_rad": {
                        "train_run_builder": "transport_study.modules.power_balance.p_rad.trb.RadiatedPowerTRB",
                        "target_vars": ["P_rad_MW", "ds_source_idx"],
                        "max_val": 16,  # Maximum radiated power in MW
                    },
                }[case.model_type]
                train_config_base = TrainConfig(
                    project=self.wandb_project_name(case),
                    train_run_builder=submodule_settings["train_run_builder"],
                    max_epochs=config.max_epochs,
                    epochs_per_val=config.epochs_per_val,
                    patience=config.patience,
                    # When doing hyperparameter tuning, this gets overwritten by the wandb agent
                    checkpoint_dir=str(self.trained_model_dir(case)),
                    dataloader_config={
                        "target_vars": submodule_settings["target_vars"],
                        "input_vars": POWER_BALANCE_INPUT_VARS,
                        "data_train_run_builder": "transport_study.modules.power_balance.trb.PowerBalanceTRB",  # Needed for submodules
                        **dataloader_config_base,
                    },
                    model_init_config={
                        "nn_depth": 2,
                        "nn_width": 16,
                        "min_val": 0,  # Minimum power in MW
                        "max_val": submodule_settings["max_val"],
                        "prng_seed": 42,
                        "in_size": 7,
                        "out_size": 1,
                        "data_normalization": case.data_normalization,
                        "domain_adaptation": case.domain_adaptation,
                    },
                    loss_config=loss_config_base,
                    optimizer_config=optimizer_config_base,
                    test_eval_suite_config=test_eval_suite_config_base,
                )
            elif case.model_type in MODEL_TYPES_WITH_SUBMODULES:
                train_config_base = TrainConfig(
                    project=self.wandb_project_name(case),
                    train_run_builder="transport_study.modules.power_balance.trb.PowerBalanceTRB",
                    max_epochs=config.max_epochs,
                    epochs_per_val=config.epochs_per_val,
                    patience=config.patience,
                    # When doing hyperparameter tuning, this gets overwritten by the wandb agent
                    checkpoint_dir=str(self.trained_model_dir(case)),
                    dataloader_config={
                        "input_vars": POWER_BALANCE_INPUT_VARS,
                        "target_vars": ["Wtot_MJ", "ds_source_idx"],
                        "state_vars": ["Wtot_MJ"],
                        "extra_vars": [
                            "P_oh_MW",
                            "P_rad_MW",
                        ],  # Bring these along for comparison / device weighting
                        **dataloader_config_base,
                    },
                    model_init_config={
                        "model_type": case.model_type,
                        "data_normalization": case.data_normalization,
                        "domain_adaptation": case.domain_adaptation,
                        "freeze_submodules": case.freeze_submodules,
                        "nn_depth": 2,
                        "nn_width": 16,
                        "in_size": 7,  # Ip, B0, R0, a_minor, kappa, ne20_line_avg, P_aux_MW
                        "out_size": 1,
                        "prng_seed": 42,
                        "submodules": {
                            "p_oh_predictor": _make_submodule_config("p_oh"),
                            "p_rad_predictor": _make_submodule_config("p_rad"),
                        },
                        "restore_submodules": True,  # Always restoring pre-trained submodules in this study
                    },
                    loss_config=loss_config_base,
                    optimizer_config=optimizer_config_base,
                    val_eval_suite_config=val_eval_suite_config_base,
                    test_eval_suite_config=test_eval_suite_config_base,
                )
            elif case.model_type == "unstructured_nn":
                train_config_base = TrainConfig(
                    project=self.wandb_project_name(case),
                    train_run_builder="transport_study.modules.power_balance.trb.PowerBalanceTRB",
                    max_epochs=config.max_epochs,
                    epochs_per_val=config.epochs_per_val,
                    patience=config.patience,
                    # When doing hyperparameter tuning, this gets overwritten by the wandb agent
                    checkpoint_dir=str(self.trained_model_dir(case)),
                    dataloader_config={
                        "input_vars": POWER_BALANCE_INPUT_VARS,
                        "target_vars": ["Wtot_MJ", "ds_source_idx"],
                        "state_vars": ["Wtot_MJ"],
                        "extra_vars": ["P_oh_MW", "P_rad_MW"],
                        **dataloader_config_base,
                    },
                    model_init_config={
                        "model_type": case.model_type,
                        "data_normalization": case.data_normalization,
                        "domain_adaptation": case.domain_adaptation,
                        "nn_depth": 2,
                        "nn_width": 16,
                        "in_size": 7,  # Ip, B0, R0, a_minor, kappa, ne20_line_avg, P_aux_MW
                        "out_size": 1,
                        "prng_seed": 42,
                    },
                    loss_config=loss_config_base,
                    optimizer_config=optimizer_config_base,
                    val_eval_suite_config=val_eval_suite_config_base,
                    test_eval_suite_config=test_eval_suite_config_base,
                )
            elif case.model_type == "transformer":
                train_config_base = TrainConfig(
                    project=self.wandb_project_name(case),
                    train_run_builder="transport_study.modules.power_balance.trb.PowerBalanceTRB",
                    max_epochs=config.max_epochs,
                    epochs_per_val=config.epochs_per_val,
                    patience=config.patience,
                    # When doing hyperparameter tuning, this gets overwritten by the wandb agent
                    checkpoint_dir=str(self.trained_model_dir(case)),
                    dataloader_config={
                        "input_vars": POWER_BALANCE_INPUT_VARS,
                        "target_vars": ["Wtot_MJ", "ds_source_idx"],
                        "state_vars": ["Wtot_MJ"],
                        "extra_vars": ["P_oh_MW", "P_rad_MW"],
                        **dataloader_config_base,
                    },
                    model_init_config={
                        "model_type": case.model_type,
                        "data_normalization": case.data_normalization,
                        "domain_adaptation": case.domain_adaptation,
                        "d_model": 16,  # Token embedding width
                        "num_heads": 2,
                        "history_len": 20,  # Attention window over past timesteps
                        "nn_depth": 2,  # MLP head after attention
                        "nn_width": 16,
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
            # but no domain adaptation and no target shots
            transfer_case = PowerBalanceStudy.Case(
                model_type=case.model_type,
                training_data=case.training_data,
                data_normalization=case.data_normalization,
                domain_adaptation=None,
                freeze_submodules=case.freeze_submodules,
                num_target_shots=HYPERPARAM_TARGET_SHOTS,
            )
            train_config_base = self._set_transfer_checkpoint(train_config_base, transfer_case)

        tuned_config_path = self.tuned_config_path(case)
        if tuned_config_path.exists():
            tuned_config = TrainConfig.load(str(tuned_config_path))
            logger.info(f"Found tuned hyperparameter config for case {case}, using hyperparameters from that config")
            # Restore hyperparameters from the tuned config, but keep the rest of the settings the same

            # Hyperparameters swept for all modules
            # Only the delta comes from the tuned loss_config: device_weights stay
            # case-specific (mixing weights differ per case)
            train_config = train_config_base.model_copy(
                update={
                    "dataloader_config": {
                        **train_config_base.dataloader_config,
                        "segment_length_train": tuned_config.dataloader_config["segment_length_train"],
                        "segment_overlap_train": tuned_config.dataloader_config["segment_overlap_train"],
                        "batch_size": tuned_config.dataloader_config["batch_size"],
                    },
                    "optimizer_config": tuned_config.optimizer_config,
                    "loss_config": {
                        **train_config_base.loss_config,
                        "huber_delta": tuned_config.loss_config.get("huber_delta", train_config_base.loss_config["huber_delta"]),
                    },
                }
            )

            # Hyperparameters swept for only certain modules
            # The scaling law has no NN of its own (its submodules carry their own tuned configs)
            def _tuned_model_init_updates() -> dict:
                updates = {}
                if case.model_type in ["p_oh", "p_rad", "unstructured_nn", "sciml", "transformer"]:
                    updates["nn_depth"] = tuned_config.model_init_config["nn_depth"]
                    updates["nn_width"] = tuned_config.model_init_config["nn_width"]
                if case.model_type == "transformer":
                    updates["d_model"] = tuned_config.model_init_config["d_model"]
                    updates["num_heads"] = tuned_config.model_init_config["num_heads"]
                    updates["history_len"] = tuned_config.model_init_config["history_len"]
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
            train_config = self._scale_transfer_lr(train_config)

        return train_config

    ##############
    # COLLECTION #
    ##############

    # Coords describing which case a record belongs to
    _CASE_COORD_NAMES = (
        "case_idx",
        "model_type",
        "training_data",
        "data_normalization",
        "domain_adaptation",
        "freeze_submodules",
        "num_target_shots",
    )

    def _case_coords(self, case_idx: int, case) -> dict:
        """Build the per-case coordinate values."""
        return {
            "case_idx": case_idx,
            "model_type": case.model_type,
            "training_data": str(case.training_data),
            # Normalize None -> "none" so the coord stays string-typed
            "domain_adaptation": case.domain_adaptation if case.domain_adaptation is not None else "none",
            "data_normalization": case.data_normalization,
            "freeze_submodules": case.freeze_submodules,
            "num_target_shots": case.num_target_shots,
        }

    def collect_results(self):
        """Collect scalar summary statistics per case (one row per case).

        Dims: case_idx
        Coords (along case_idx): model_type, training_data, data_normalization,
        domain_adaptation, freeze_submodules, num_target_shots
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

            result = self._summarize_case_errors(ds).assign_coords(self._case_coords(case_idx, case))
            results.append(result)

        # coords="different" stacks the per-case scalar coords along case_idx
        ds_merged = xr.concat(results, dim="case_idx", coords="different")
        return ds_merged


def run_study(
    config: PowerBalanceStudy.Config | str | Path,
    enable_parallelism: bool | None = False,
    skip_tuning: bool | None = True,
    skip_visualization: bool | None = False,
    clean_sweeps: bool | None = False,
    clean_models: bool | None = False,
    clean_results: bool | None = False,
    clean_figures: bool | None = False,
):
    """
    Go from datasets to collected results in one command.

    Requires specifying paths to the source datasets in the config TOML or environment variables.
    Due to data sharing restrictions, the only dataset included in this repository is for C-Mod.
    If you have access to data from other tokamaks (e.g. DIII-D), create a source dataset using the scripts in `transport_study/datasets/`
    and provide the path when running this script.
    If a dataset is not provided for a tokamak, figures which require that data will be skipped.

    Parameters
    ----------
    config : PowerBalanceStudy.Config | str | Path
        The study config, or a path to its TOML file.
    enable_parallelism : bool | None
        If false, runs the entire study sequentially in one process.
        If true, submits independent training steps with SLURM up to configurable resource limits.
        The idea is you would periodically call this 'run_study' function, and it checks what models still need to be trained and submit jobs for those, until eventually all models are trained and all results are computed.
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
        config = PowerBalanceStudy.Config.from_toml(Path(config))
    study = PowerBalanceStudy(config)
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
            f"Collected results file found at\n{study.collected_results_path()}\nSkipping orchestration and going straight to analysis"
        )
    else:
        logger.opt(colors=True).info("<bold><magenta>ORCHESTRATION</magenta></bold>")

        study.run_unfinished_cases(skip_tuning=skip_tuning, enable_parallelism=enable_parallelism)

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
