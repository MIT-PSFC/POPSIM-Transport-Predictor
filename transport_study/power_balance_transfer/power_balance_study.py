from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar

if TYPE_CHECKING:
    from popsim.ml import TrainConfig

import fire
import netCDF4  # noqa: F401
from pydantic import Field

from transport_study import PACKAGE_ROOT
from transport_study.config import CaseAxis, config
from transport_study.modules.normalization import INPUT_NORMALIZATIONS, NORM_INPUT_VARS
from transport_study.modules.power_balance.module import (
    MODEL_TYPES_WITH_SUBMODULES,
    MULTIOBJECTIVE_MODEL_TYPES,
    SUBMODULE_MODEL_TYPES,
)
from transport_study.orchestration.case_analysis import run_summary_analysis
from transport_study.orchestration.study import (
    CaseGridConfig,
    ModelTrainSpec,
    Study,
)
from transport_study.orchestration.target_shots import BASE_TARGET_SHOT_ORDER
from transport_study.power_balance_transfer.data_visualization import DataVisualization
from transport_study.power_balance_transfer.plotting import COMPARISON_FAMILIES, LAYOUT
from transport_study.power_balance_transfer.tables import SPEC as TABLE_SPEC

# The 7 physical inputs every power-balance model consumes. Normalization is
# done inside the modules (transport_study.modules.normalization), so the
# dataloader always pulls exactly these (the TRB adds ds_source_idx itself)
POWER_BALANCE_INPUT_VARS = list(NORM_INPUT_VARS)

# Purely data-driven model types, no submodules so nothing to freeze
MODEL_TYPES_WITHOUT_SUBMODULES = ("mlp", "transformer")
# The measured powers, targets of the anchor terms in the training loss
POWER_TARGET_VARS = ["power_ohm_MW", "power_radiated_MW"]


# Per-submodule train settings shared with the transport study, which trains
# the same p_oh/p_rad prereq cases for its power balance submodule
SCALAR_SUBMODULE_SETTINGS = {
    "p_oh": {
        "train_run_builder": "transport_study.modules.power_balance.p_oh.trb.OhmicPowerTRB",
        "target_vars": ["power_ohm_MW", "ds_source_idx"],
    },
    "p_rad": {
        "train_run_builder": "transport_study.modules.power_balance.p_rad.trb.RadiatedPowerTRB",
        "target_vars": ["power_radiated_MW", "ds_source_idx"],
    },
}


class PowerBalanceStudy(Study):
    SWEEP_CONFIG_DIR = Path(PACKAGE_ROOT) / "power_balance_transfer" / "sweep_configs"
    STUDY_TYPE = "power_balance_transfer"
    DATA_VISUALIZATION = DataVisualization
    ANALYSIS_METRICS_MODULE = "transport_study.power_balance_transfer.study_metrics"
    ANALYSIS_REPORTS_MODULE = "transport_study.power_balance_transfer.case_reports"
    TUNED_DATALOADER_KEYS = ("segment_length_train", "segment_overlap_train", "batch_size")
    TUNED_LOSS_KEYS = ("huber_delta",)

    ##################
    # INITIALIZATION #
    ##################
    class Config(CaseGridConfig):
        # The different cases being compared in this study
        model_types: Annotated[tuple[str, ...], CaseAxis("model_type")] = Field(
            default_factory=lambda: ("sciml-taue-scalinglaw", "sciml-taue-nn", "mlp", "transformer")
        )
        data_normalization_methods: Annotated[tuple[str, ...], CaseAxis("data_normalization")] = Field(
            default_factory=lambda: INPUT_NORMALIZATIONS
        )
        freeze_submodules_options: Annotated[tuple[bool, ...], CaseAxis("freeze_submodules")] = Field(default_factory=lambda: (False,))
        multiobjective_options: Annotated[tuple[bool, ...], CaseAxis("multiobjective")] = Field(default_factory=lambda: (False,))
        num_target_shots_options: Annotated[tuple[int, ...], CaseAxis("num_target_shots")] = Field(
            default_factory=lambda: (0, 1, 3, 10, 32)
        )
        # Hyperparameter tuning case configuration
        # (hyperparam_domain_adaptation and hyperparam_num_target_shots live on CaseGridConfig)
        hyperparam_data_normalization: str = "physics"
        hyperparam_freeze_submodules: bool = False
        hyperparam_multiobjective: bool = False

        FIELD_CHOICES: ClassVar[dict[str, tuple]] = {
            "model_types": (*MODEL_TYPES_WITH_SUBMODULES, *MODEL_TYPES_WITHOUT_SUBMODULES, *SUBMODULE_MODEL_TYPES),
            "data_normalization_methods": INPUT_NORMALIZATIONS,
            "hyperparam_data_normalization": INPUT_NORMALIZATIONS,
        }

    @dataclass
    class Case(Study.Case):
        """
        model_type: The type of power_balance model to use.
        - sciml-taue-scalinglaw: H89, H98, and P_LH scaling laws to predict tau_e
        - sciml-taue-nn: neural network predicts tau_e, and we do the power balance calculation
        - mlp: an MLP predicts the stored-energy evolution from the inputs and its own predicted stored energy
        - transformer: the current inputs attend over a buffer of past predicted stored energies to predict its evolution
        - p_oh / p_rad: submodule predictors, appear only as prereq cases of sciml-taue-scalinglaw and sciml-taue-nn

        training_data: The historic dataset(s) used for training
        - a device name from config.dataset_paths, e.g. cmod
        - device names joined by _ for a combination, e.g. cmod_tcv
        - exnihilo: No historic training data

        data_normalization: The method for normalizing the model's NN inputs, implemented
        as a POPSIM module configured from the training data only (transport_study/modules/normalization.py).
        - raw: No normalization, Ip, Wtot, etc. are in their original units
        - physics: Convert to typical dimensionless parameters like q_star, f_G, etc.
        - zscore: Within each device, normalize each variable to zero mean and unit variance.
        - coral: Use the CORAL method to align covariances of source and target domains (https://arxiv.org/abs/1612.01939)
        - physics-coral: The physics parameters followed by CORAL alignment fitted on them.
        - physics-zscore: The physics parameters followed by a per-device z-score fitted on them.

        domain_adaptation: The method for domain adaptation between source and target devices.
        - none: No domain adaptation, train and test on the same device(s). This is used for hyperparameter tuning and as a baseline for comparison, answering the question "what is the best possible performance we could expect if we had a bunch of data?"
        - weighted: Add a small amount of highly-weighted target data during training
        - addition: Add target shots to the training set as normal samples, no weighting
        - transfer: Train on source data, freeze all but the last layers of the model, and fine-tune on a small amount of target data
        - transfer_pretrain: The pretrain half of a transfer case, never a case-grid axis value (see Study.Case.transfer_pretrain_case). Trains on historic data only with checkpoint selection on the target test set. Stat normalizations fit the normalizer on historic + the transfer case's target shots so the fine-tune case inherits target-aware statistics through the checkpoint restore, stateless ones (raw / physics) share one twin at 0 target shots

        freeze_submodules: Whether to freeze the p_oh/p_rad submodules of the model during training.
        The P_oh and P_rad signals are hard to quantify, we might want to let them drift from the original targets to better match energy_mhd_MJ

        multiobjective: Whether the transformer's head also predicts P_oh and P_rad,
        anchored to the measured powers in the training loss like the sciml submodules (MULTIOBJECTIVE_MODEL_TYPES only).
        Shares the plain case's sweep, only multiobjective cases show it in the name

        num_target_shots: The number of shots included in the training data from the target dataset, never any of the held-out test shots.

        target_shot_order: The order the target training shots are added in as num_target_shots grows, see orchestration/target_shots.py
        - ascending: the base extrapolation, lowest hazard first, suppressed from the case name
        - descending: the highest-hazard non-test shots first, the ones closest to the test regime
        - spanning: the shots whose timeslice footprints span the power balance input and output space
        """

        data_normalization: str
        freeze_submodules: bool
        multiobjective: bool

        VALID_MODEL_TYPES = (*MODEL_TYPES_WITH_SUBMODULES, *MODEL_TYPES_WITHOUT_SUBMODULES, *SUBMODULE_MODEL_TYPES)
        # Unfrozen submodules and single-objective training are the defaults, so only the others show in the name
        STR_TOKEN_FIELDS = (
            ("norm_", "data_normalization"),
            ("freeze_", "freeze_submodules", False),
            ("mo_", "multiobjective", False),
        )
        HYPERPARAM_FIELDS = ("data_normalization", "domain_adaptation", "freeze_submodules", "multiobjective", "num_target_shots")

        # The dataclass decorator would null an inherited __hash__
        __hash__ = Study.Case.__hash__

        def __init__(
            self,
            model_type: str,
            training_data,
            data_normalization: str,
            domain_adaptation: str | None,
            freeze_submodules: bool,
            num_target_shots: int,
            target_shot_order: str = BASE_TARGET_SHOT_ORDER,
            multiobjective: bool = False,
        ):
            self.data_normalization = data_normalization
            self.freeze_submodules = freeze_submodules
            self.multiobjective = multiobjective
            self.init_common(model_type, training_data, domain_adaptation, num_target_shots, target_shot_order)

        def normalization_method(self) -> str | None:
            return self.data_normalization

        def validate(self):
            super().validate()
            if self.data_normalization not in INPUT_NORMALIZATIONS:
                raise ValueError(f"Unknown data normalization method: {self.data_normalization}")
            if self.model_type in SUBMODULE_MODEL_TYPES and self.freeze_submodules != config.hyperparam_freeze_submodules:
                raise ValueError(
                    f"freeze_submodules should be a dummy value ({config.hyperparam_freeze_submodules}) for submodule {self.model_type}"
                )
            if self.model_type not in MULTIOBJECTIVE_MODEL_TYPES and self.multiobjective != config.hyperparam_multiobjective:
                raise ValueError(
                    f"multiobjective should be a dummy value ({config.hyperparam_multiobjective}) for {self.model_type}, "
                    f"only {MULTIOBJECTIVE_MODEL_TYPES} have multiobjective heads"
                )

        @classmethod
        def pin_inapplicable_axes(cls, fields: dict) -> dict:
            pinned = super().pin_inapplicable_axes(fields)
            # Only the structured models have submodules to freeze
            if fields["model_type"] not in MODEL_TYPES_WITH_SUBMODULES:
                pinned |= {"freeze_submodules": config.hyperparam_freeze_submodules}
            if fields["model_type"] not in MULTIOBJECTIVE_MODEL_TYPES:
                pinned |= {"multiobjective": config.hyperparam_multiobjective}
            return pinned

        def model_type_prereqs(self) -> list[Study.Case]:
            # The structured models restore pre-trained p_oh/p_rad submodules
            if self.model_type not in MODEL_TYPES_WITH_SUBMODULES:
                return []
            return [
                self.replace(model_type=submodule_type, freeze_submodules=config.hyperparam_freeze_submodules)
                for submodule_type in SUBMODULE_MODEL_TYPES
            ]

    #############
    # EXECUTION #
    #############

    def base_dataloader_config(self, case: Case) -> dict:
        return {
            **self.target_split_config(case),
            # Hyperparameters
            "segment_length_train": 100,
            "segment_overlap_train": 50,
            # 4096 measured 42 GB on the worst case
            # (transformer at sweep max, d_model 64, history_len 50, these 100-step segments)
            # Memory scales linearly with batch and segment_length_train so revisit this cap if segment_length_train grows.
            "batch_size": 4096,
            # Part of validation, should be left alone during hyperparameter tuning
            "segment_length_val": None,
            "segment_overlap_val": 0,
        }

    def base_optimizer_config(self) -> dict:
        return {
            **super().base_optimizer_config(),
            # The p_oh/p_rad submodules train at a reduced rate relative to
            # the taue network so joint training does not pull them far from
            # their pretrained behavior. In transfer finetunes this stacks
            # with the step-budgeted transfer LR on purpose: the submodules
            # already ran their own pretrain + finetune prereq chain, so the
            # joint finetune lets their last layers drift only at 0.1x the
            # transfer LR.
            # No-op for model types without submodules (no matching pytree paths)
            "submodule_lr_factors": {"p_oh_predictor": 0.1, "p_rad_predictor": 0.1},
        }

    def base_loss_config(self) -> dict:
        return {
            # [MJ] on the scale of a Wtot error, C-Mod and MAST medians are 0.03-0.05 MJ
            "huber_delta": 0.05,
            # Anchor terms keeping the predicted P_oh / P_rad (the sciml submodules, a multiobjective transformer's heads)
            # close to the measured signals while the whole module trains on Wtot.
            # Training loss only, and a no-op for model types whose target_vars carry no power_ohm_MW / power_radiated_MW.
            # Sized to the Wtot term in the trained state,
            # at 0.1 the anchors outweighed it ~100x and unfrozen submodules barely moved toward Wtot
            "anchor_weight_power_ohm": 2e-3,
            "anchor_weight_power_radiated": 2e-3,
        }

    def _make_submodule_config(self, case: Case, submodule_type: str) -> TrainConfig:
        """Full TrainConfig for a p_oh/p_rad submodule, recursing through make_train_config
        so submodule cases get their own tuned merge and transfer wiring."""
        return self.make_train_config(case.replace(model_type=submodule_type, freeze_submodules=config.hyperparam_freeze_submodules))

    def model_train_spec(self, case: Case, dataloader_config_base: dict) -> ModelTrainSpec:
        trb = "transport_study.modules.power_balance.trb.PowerBalanceTRB"
        if case.model_type in SUBMODULE_MODEL_TYPES:
            submodule_settings = SCALAR_SUBMODULE_SETTINGS[case.model_type]
            return ModelTrainSpec(
                train_run_builder=submodule_settings["train_run_builder"],
                dataloader_config={
                    "target_vars": submodule_settings["target_vars"],
                    "input_vars": POWER_BALANCE_INPUT_VARS,
                    "data_train_run_builder": trb,  # Needed for submodules
                    **dataloader_config_base,
                },
                model_init_config={
                    "nn_depth": 2,
                    "nn_width": 16,
                    "prng_seed": 42,
                    "data_normalization": case.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                },
            )
        elif case.model_type in MODEL_TYPES_WITH_SUBMODULES:
            return ModelTrainSpec(
                train_run_builder=trb,
                dataloader_config={
                    "input_vars": POWER_BALANCE_INPUT_VARS,
                    # The measured powers are targets so the training loss can
                    # anchor the submodule predictions to them (anchor_weight_*
                    # in the loss config)
                    "target_vars": ["energy_mhd_MJ", *POWER_TARGET_VARS, "ds_source_idx"],
                    "state_vars": ["energy_mhd_MJ"],
                    **dataloader_config_base,
                },
                model_init_config={
                    "model_type": case.model_type,
                    "data_normalization": case.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    "freeze_submodules": case.freeze_submodules,
                    "nn_depth": 2,
                    "nn_width": 16,
                    "prng_seed": 42,
                    "submodules": {
                        "p_oh_predictor": self._make_submodule_config(case, "p_oh"),
                        "p_rad_predictor": self._make_submodule_config(case, "p_rad"),
                    },
                },
            )
        elif case.model_type == "mlp":
            return ModelTrainSpec(
                train_run_builder=trb,
                dataloader_config={
                    "input_vars": POWER_BALANCE_INPUT_VARS,
                    "target_vars": ["energy_mhd_MJ", "ds_source_idx"],
                    "state_vars": ["energy_mhd_MJ"],
                    # Unused by the model, but the dataloader drops the NaN slices of every loaded variable,
                    # so the measured powers keep the training segments identical to the structured models'
                    "extra_vars": POWER_TARGET_VARS,
                    **dataloader_config_base,
                },
                model_init_config={
                    "model_type": case.model_type,
                    "data_normalization": case.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    "nn_depth": 2,
                    "nn_width": 16,
                    "prng_seed": 42,
                },
            )
        elif case.model_type == "transformer":
            # The multiobjective head predicts the measured powers, so they are targets its training loss anchors.
            # For the plain transformer they only keep the training segments identical to the structured models', see mlp
            if case.multiobjective:
                power_vars = {"target_vars": ["energy_mhd_MJ", *POWER_TARGET_VARS, "ds_source_idx"]}
            else:
                power_vars = {"target_vars": ["energy_mhd_MJ", "ds_source_idx"], "extra_vars": POWER_TARGET_VARS}
            return ModelTrainSpec(
                train_run_builder=trb,
                dataloader_config={
                    "input_vars": POWER_BALANCE_INPUT_VARS,
                    "state_vars": ["energy_mhd_MJ"],
                    **power_vars,
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
                    "multiobjective": case.multiobjective,
                },
            )
        else:
            raise ValueError(f"Unknown model type: {case.model_type}")

    def tuned_model_init_updates(self, case: Case, tuned_config: TrainConfig) -> dict:
        # sciml-taue-scalinglaw has no NN of its own (its submodules carry their own tuned configs)
        updates = {}
        if case.model_type in [*SUBMODULE_MODEL_TYPES, "mlp", "sciml-taue-nn", "transformer"]:
            updates["nn_depth"] = tuned_config.model_init_config["nn_depth"]
            updates["nn_width"] = tuned_config.model_init_config["nn_width"]
        if case.model_type == "transformer":
            updates["d_model"] = tuned_config.model_init_config["d_model"]
            updates["num_heads"] = tuned_config.model_init_config["num_heads"]
            updates["history_len"] = tuned_config.model_init_config["history_len"]
        return updates

    ############
    # ANALYSIS #
    ############

    def run_analysis(self, enable_parallelism: bool) -> None:
        run_summary_analysis(self, enable_parallelism, LAYOUT, COMPARISON_FAMILIES, TABLE_SPEC)


run_study = PowerBalanceStudy.run_study


if __name__ == "__main__":
    fire.Fire(
        {
            "run_study": run_study,
        }
    )
