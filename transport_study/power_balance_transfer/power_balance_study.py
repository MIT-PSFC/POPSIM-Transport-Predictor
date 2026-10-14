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
    BASE_MODEL_SIZE,
    NN_INIT_DEFAULTS,
    CaseGridConfig,
    ModelTrainSpec,
    Study,
    model_type_field_choices,
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
# Every model type a case of this study can have, the prereq submodules included
STUDY_MODEL_TYPES = (*MODEL_TYPES_WITH_SUBMODULES, *MODEL_TYPES_WITHOUT_SUBMODULES, *SUBMODULE_MODEL_TYPES)
# The measured powers, targets of the anchor terms in the training loss
POWER_TARGET_VARS = ["power_ohm_MW", "power_radiated_MW"]


# Per-submodule train settings shared with the transport study, which trains
# the same p_oh/p_rad prereq cases for its power balance submodule.
# fresh_var keeps the training and validation samples to the timeslices where it is 1 (see trb_utils.get_time_dep_dataloaders):
# P_oh comes from the reconstruction, which the stores hold between reconstructions, P_rad from bolometry
SCALAR_SUBMODULE_SETTINGS = {
    "p_oh": {
        "train_run_builder": "transport_study.modules.power_balance.p_oh.trb.OhmicPowerTRB",
        "target_vars": ["power_ohm_MW", "ds_source_idx"],
        "fresh_var": "fresh_equilibrium",
    },
    "p_rad": {
        "train_run_builder": "transport_study.modules.power_balance.p_rad.trb.RadiatedPowerTRB",
        "target_vars": ["power_radiated_MW", "ds_source_idx"],
        "fresh_var": None,
    },
}

# Training batches, shared with the transport study's power balance pseudo-types.
# Each loader family's batch sits near the throughput knee of its most expensive model at its sweep maximum,
# the batch past which the time per sample stops falling.
# On an A100-80GB: the transformer's time per sample falls 25 percent from 2048 to 4096 and 11 percent more to 8192,
# where it peaks at 72.6 GB of the 77 GB XLA pool. 4096 keeps the headroom, 16 batches per epoch on the largest set at 36 GB.
POWER_BALANCE_BATCH_SIZE = 4096
# p_oh / p_rad: 18 and 15 percent per doubling from 98304, then 6 / 9 percent past 393216, 7 / 8 batches per epoch at 1.3 GB.
# They should train on CPU, where an epoch takes about as long at any batch, so this only sets their steps per epoch
SCALAR_SUBMODULE_BATCH_SIZE = 393216

POWER_BALANCE_TRB = "transport_study.modules.power_balance.trb.PowerBalanceTRB"
# The stored energy and its freshness flag the Wtot loss reads, with the device label
WTOT_TARGET_VARS = ["energy_mhd_MJ", "fresh_equilibrium", "ds_source_idx"]
# The structured models add the measured powers, so the training loss anchors their submodule predictions to them
STRUCTURED_TARGET_VARS = ["energy_mhd_MJ", "fresh_equilibrium", *POWER_TARGET_VARS, "ds_source_idx"]

# Fallback loss hyperparameters of every power balance case run without a tuned config,
# also those of the transport study's power balance pseudo-types (p_oh, p_rad, power_balance)
POWER_BALANCE_LOSS_DEFAULTS = {
    # The training residuals are divided by the device's typical value (trb_utils.residual_scales),
    # so the delta is a fraction of it, swept over 1 to 100 percent
    "huber_delta": 0.1,
    # Anchor terms keeping the predicted P_oh / P_rad (the sciml submodules, a multiobjective transformer's heads)
    # close to the measured signals while the whole module trains on Wtot.
    # Training loss only, and a no-op for model types whose target_vars carry no power_ohm_MW / power_radiated_MW.
    # A relative P_oh or P_rad miss weighs a tenth of the same relative Wtot miss, an untuned prior
    "anchor_weight_power_ohm": 0.1,
    "anchor_weight_power_radiated": 0.1,
}


def scalar_submodule_train_spec(
    submodule_type: str, dataloader_config_base: dict, data_normalization: str, domain_adaptation: str | None
) -> ModelTrainSpec:
    """The spec of a p_oh or p_rad prereq case, the power balance study's and the transport study's pseudo-type's alike."""
    settings = SCALAR_SUBMODULE_SETTINGS[submodule_type]
    return ModelTrainSpec(
        train_run_builder=settings["train_run_builder"],
        dataloader_config={
            "target_vars": settings["target_vars"],
            "fresh_var": settings["fresh_var"],
            "input_vars": POWER_BALANCE_INPUT_VARS,
            # The submodules load the power balance view
            "data_train_run_builder": POWER_BALANCE_TRB,
            **dataloader_config_base,
            "batch_size": SCALAR_SUBMODULE_BATCH_SIZE,
        },
        model_init_config={**NN_INIT_DEFAULTS, "data_normalization": data_normalization, "domain_adaptation": domain_adaptation},
    )


def structured_power_balance_train_spec(
    model_type: str,
    dataloader_config_base: dict,
    data_normalization: str,
    domain_adaptation: str | None,
    freeze_submodules: bool,
    submodule_configs: dict[str, TrainConfig],
) -> ModelTrainSpec:
    """The spec of a sciml-taue-nn or sciml-taue-scalinglaw case over its restored p_oh / p_rad configs, for both studies."""
    return ModelTrainSpec(
        train_run_builder=POWER_BALANCE_TRB,
        dataloader_config={
            "input_vars": POWER_BALANCE_INPUT_VARS,
            "target_vars": STRUCTURED_TARGET_VARS,
            "state_vars": ["energy_mhd_MJ"],
            **dataloader_config_base,
        },
        model_init_config={
            "model_type": model_type,
            "data_normalization": data_normalization,
            "domain_adaptation": domain_adaptation,
            "freeze_submodules": freeze_submodules,
            **NN_INIT_DEFAULTS,
            "submodules": submodule_configs,
        },
    )


class PowerBalanceStudy(Study):
    SWEEP_CONFIG_DIR = Path(PACKAGE_ROOT) / "power_balance_transfer" / "sweep_configs"
    STUDY_TYPE = "power_balance_transfer"
    RESIDUAL_SCALE_LOSS = True
    ANCHOR_DELTA_MODEL_TYPES: ClassVar[dict[str, str]] = {"power_ohm_MW": "p_oh", "power_radiated_MW": "p_rad"}
    DATA_VISUALIZATION = DataVisualization
    ANALYSIS_METRICS_MODULE = "transport_study.power_balance_transfer.study_metrics"
    ANALYSIS_REPORTS_MODULE = "transport_study.power_balance_transfer.case_reports"
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
            **model_type_field_choices(STUDY_MODEL_TYPES),
            "data_normalization_methods": INPUT_NORMALIZATIONS,
            "hyperparam_data_normalization": INPUT_NORMALIZATIONS,
        }

    @dataclass
    class Case(Study.Case):
        """The power balance case, see Study.Case for the shared axes.

        model_type:
        - sciml-taue-scalinglaw: H89, H98 and P_LH scaling laws predict tau_e
        - sciml-taue-nn: a neural network predicts tau_e, the power balance does the rest
        - mlp: an MLP predicts the stored-energy evolution from the inputs and its own predicted stored energy
        - transformer: the current inputs attend over a buffer of past predicted stored energies
        - p_oh / p_rad: the submodule predictors, prereq cases of the sciml types alone

        data_normalization: the input normalization of the model's NN,
        a POPSIM module fitted on the training data only (transport_study/modules/normalization.py)
        - raw: Ip, Wtot and the rest in their working units
        - physics: dimensionless parameters like q_star and f_G
        - zscore: per-device zero mean and unit variance
        - coral: per-device covariance alignment to the target device (https://arxiv.org/abs/1612.01939)
        - physics-coral / physics-zscore: the physics parameters followed by the per-device stage fitted on them

        freeze_submodules: whether the sciml types keep their p_oh / p_rad submodules fixed in joint training,
        the alternative letting them drift from the measured powers toward a better Wtot

        multiobjective: whether the transformer's head also predicts P_oh and P_rad,
        anchored to the measured powers like the sciml submodules (MULTIOBJECTIVE_MODEL_TYPES alone).
        Shares the plain case's sweep, named mo_True
        """

        data_normalization: str
        freeze_submodules: bool
        multiobjective: bool

        VALID_MODEL_TYPES = STUDY_MODEL_TYPES
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
            model_size: str = BASE_MODEL_SIZE,
        ):
            self.data_normalization = data_normalization
            self.freeze_submodules = freeze_submodules
            self.multiobjective = multiobjective
            self.init_common(model_type, training_data, domain_adaptation, num_target_shots, target_shot_order, model_size)

        def normalization_method(self) -> str | None:
            return self.data_normalization

        def validate(self):
            super().validate()
            if self.data_normalization not in INPUT_NORMALIZATIONS:
                raise ValueError(f"Unknown data normalization method: {self.data_normalization}")

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
            # The structured models restore pre-trained p_oh/p_rad submodules at the case's own model size,
            # so a sized case sizes the whole model
            if self.model_type in MODEL_TYPES_WITH_SUBMODULES:
                return [self.replace_pinned(model_type=submodule_type) for submodule_type in SUBMODULE_MODEL_TYPES]
            # A multiobjective case's anchors take the huber deltas the hyperparam p_oh / p_rad cases were tuned with.
            # The model type guard matters when the hyperparam value is True, which pins the submodules' own flag True
            if self.model_type in MULTIOBJECTIVE_MODEL_TYPES and self.multiobjective:
                return [self.hyperparam_case_of(submodule_type) for submodule_type in SUBMODULE_MODEL_TYPES]
            return []

    #############
    # EXECUTION #
    #############

    def base_dataloader_config(self, case: Case) -> dict:
        return {
            **self.target_split_config(case),
            # Fixed loader settings, no sweep changes them (test_sweep_configs)
            "segment_length_train": 100,
            "segment_overlap_train": 50,
            # Memory scales linearly with batch and segment_length_train, see POWER_BALANCE_BATCH_SIZE
            "batch_size": POWER_BALANCE_BATCH_SIZE,
            # Validation scores whole episodes
            "segment_length_val": None,
            "segment_overlap_val": 0,
        }

    def base_optimizer_config(self) -> dict:
        return {
            **super().base_optimizer_config(),
            # The p_oh/p_rad submodules train at a reduced rate relative to
            # the taue network so joint training does not pull them far from
            # their pretrained behavior. In transfer finetunes this stacks
            # with the target-only lr0 scale on purpose: the submodules
            # already ran their own pretrain + finetune prereq chain, so the
            # joint finetune lets their last layers drift only at 0.1x the
            # transfer LR.
            # No-op for model types without submodules (no matching pytree paths)
            "submodule_lr_factors": {"p_oh_predictor": 0.1, "p_rad_predictor": 0.1},
        }

    def base_loss_config(self, case: Case) -> dict:
        return dict(POWER_BALANCE_LOSS_DEFAULTS)

    def model_train_spec(self, case: Case, dataloader_config_base: dict) -> ModelTrainSpec:
        if case.model_type in SUBMODULE_MODEL_TYPES:
            return scalar_submodule_train_spec(case.model_type, dataloader_config_base, case.data_normalization, case.domain_adaptation)
        if case.model_type in MODEL_TYPES_WITH_SUBMODULES:
            submodule_configs = {
                "p_oh_predictor": self.make_submodule_config(case, "p_oh"),
                "p_rad_predictor": self.make_submodule_config(case, "p_rad"),
            }
            return structured_power_balance_train_spec(
                case.model_type,
                dataloader_config_base,
                case.data_normalization,
                case.domain_adaptation,
                case.freeze_submodules,
                submodule_configs,
            )
        if case.model_type == "mlp":
            return ModelTrainSpec(
                train_run_builder=POWER_BALANCE_TRB,
                dataloader_config={
                    "input_vars": POWER_BALANCE_INPUT_VARS,
                    "target_vars": WTOT_TARGET_VARS,
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
                    **NN_INIT_DEFAULTS,
                },
            )
        if case.model_type == "transformer":
            # The multiobjective head predicts the measured powers, so they are targets its training loss anchors.
            # For the plain transformer they only keep the training segments identical to the structured models', see mlp
            if case.multiobjective:
                power_vars = {"target_vars": STRUCTURED_TARGET_VARS}
            else:
                power_vars = {"target_vars": WTOT_TARGET_VARS, "extra_vars": POWER_TARGET_VARS}
            return ModelTrainSpec(
                train_run_builder=POWER_BALANCE_TRB,
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
                    **NN_INIT_DEFAULTS,
                    "multiobjective": case.multiobjective,
                },
            )
        raise ValueError(f"Unknown model type: {case.model_type}")

    ############
    # ANALYSIS #
    ############

    def run_analysis(self, enable_parallelism: bool) -> None:
        run_summary_analysis(self, enable_parallelism, LAYOUT, COMPARISON_FAMILIES, TABLE_SPEC)


if __name__ == "__main__":
    fire.Fire(
        {
            "run_study": PowerBalanceStudy.run_study,
            "reset_study": PowerBalanceStudy.reset_study,
        }
    )
