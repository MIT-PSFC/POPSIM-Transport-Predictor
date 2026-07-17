from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from popsim.ml import TrainConfig

import fire
import netCDF4  # noqa: F401
import xarray as xr
from loguru import logger
from pydantic import Field, field_validator

from transport_study import PACKAGE_ROOT
from transport_study.config import config
from transport_study.orchestration.case_analysis import run_case_analysis_parallel
from transport_study.orchestration.study import (
    HYPERPARAM_TARGET_SHOTS,
    CaseGridConfig,
    ModelTrainSpec,
    Study,
)
from transport_study.power_balance_transfer.case_reports import generate_case_reports
from transport_study.power_balance_transfer.data_visualization import DataVisualization
from transport_study.power_balance_transfer.plotting import (
    data_normalization_comparison,
    domain_adaptation_comparison,
    model_comparison,
    training_dataset_comparison,
)
from transport_study.power_balance_transfer.study_metrics import collect_metrics
from transport_study.power_balance_transfer.tables import write_comparison_tables

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

VALID_DATA_NORMALIZATIONS = ("raw", "physics", "z_score", "coral", "physics-coral")


class PowerBalanceStudy(Study):
    SWEEP_CONFIG_DIR = Path(PACKAGE_ROOT) / "power_balance_transfer" / "sweep_configs"
    STUDY_TYPE = "power_balance_transfer"
    DATA_VISUALIZATION = DataVisualization
    ANALYSIS_METRICS_MODULE = "transport_study.power_balance_transfer.study_metrics"
    ANALYSIS_REPORTS_MODULE = "transport_study.power_balance_transfer.case_reports"
    CASE_AXIS_FIELDS = (
        "model_types",
        "training_datasets",
        "data_normalization_methods",
        "domain_adaptation_methods",
        "freeze_submodules_options",
        "num_target_shots_options",
    )
    TUNED_DATALOADER_KEYS = ("segment_length_train", "segment_overlap_train", "batch_size")
    TUNED_LOSS_KEYS = ("huber_delta",)

    ##################
    # INITIALIZATION #
    ##################
    class Config(CaseGridConfig):
        # The different cases being compared in this study
        model_types: tuple[str, ...] = Field(default_factory=lambda: ("scaling_law", "sciml", "unstructured_nn", "transformer"))
        data_normalization_methods: tuple[str, ...] = Field(default_factory=lambda: VALID_DATA_NORMALIZATIONS)
        freeze_submodules_options: tuple[bool, ...] = Field(default_factory=lambda: (True,))
        num_target_shots_options: tuple[int, ...] = Field(default_factory=lambda: (0, 1, 3, 10, 32, -1))
        # Hyperparameter tuning case configuration
        # (hyperparam_domain_adaptation and hyperparam_num_target_shots live on CaseGridConfig)
        hyperparam_data_normalization: str = "physics-coral"
        hyperparam_freeze_submodules: bool = True

        COMPAT_HYPERPARAM_FIELDS = (
            "hyperparam_data_normalization",
            "hyperparam_domain_adaptation",
            "hyperparam_freeze_submodules",
            "hyperparam_num_target_shots",
        )

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
            methods = (v,) if isinstance(v, str) else v
            for dn in methods:
                if dn not in VALID_DATA_NORMALIZATIONS:
                    raise ValueError(f"Invalid data normalization method: {dn}. Must be one of {sorted(VALID_DATA_NORMALIZATIONS)}.")
            return v

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
        - physics-coral: The physics parameters followed by CORAL alignment fitted on them.

        domain_adaptation: The method for domain adaptation between source and target devices.
        - none: No domain adaptation, train and test on the same device(s). This is used for hyperparameter tuning and as a baseline for comparison, answering the question "what is the best possible performance we could expect if we had a bunch of data?"
        - mixing: Add a small amount of highly-weighted target data during training
        - transfer: Train on source data, freeze all but the last layers of the model, and fine-tune on a small amount of target data
        - transfer_pretrain: The pretrain half of a stat-normalized transfer case, never a case-grid axis value (see Study.Case.transfer_pretrain_case). Trains on historic data only, with the normalizer fitted on historic + the transfer case's target shots so the fine-tune case inherits target-aware statistics through the checkpoint restore

        freeze_submodules: Whether to freeze the p_oh/p_rad submodules of the model during training.
        The P_oh and P_rad signals are hard to quantify, we might want to let them drift from the original targets to better match Wtot_MJ

        num_target_shots: The number of shots included in the training data from the target dataset, or -1 to include all shots (including all shots in training is cheating, but again answers the question of what is the best possible performance).
        """

        data_normalization: str
        freeze_submodules: bool

        VALID_MODEL_TYPES = (*MODEL_TYPES_WITH_SUBMODULES, *MODEL_TYPES_WITHOUT_SUBMODULES, *SUBMODULE_MODEL_TYPES)
        STR_TOKEN_FIELDS = (("norm_", "data_normalization"), ("freeze_", "freeze_submodules"))
        HYPERPARAM_FIELDS = ("data_normalization", "domain_adaptation", "freeze_submodules", "num_target_shots")

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
        ):
            self.data_normalization = data_normalization
            self.freeze_submodules = freeze_submodules
            self._init_common(model_type, training_data, domain_adaptation, num_target_shots)

        def _normalization_method(self) -> str | None:
            return self.data_normalization

        def _validate(self):
            super()._validate()
            if self.data_normalization not in VALID_DATA_NORMALIZATIONS:
                raise ValueError(f"Unknown data normalization method: {self.data_normalization}")
            if self.model_type in SUBMODULE_MODEL_TYPES and self.freeze_submodules != config.hyperparam_freeze_submodules:
                raise ValueError(
                    f"freeze_submodules should be a dummy value ({config.hyperparam_freeze_submodules}) for submodule {self.model_type}"
                )

        def _model_type_prereqs(self) -> list[Study.Case]:
            # The structured models restore pre-trained p_oh/p_rad submodules
            if self.model_type not in MODEL_TYPES_WITH_SUBMODULES:
                return []
            return [
                self._replace(model_type=submodule_type, freeze_submodules=config.hyperparam_freeze_submodules)
                for submodule_type in SUBMODULE_MODEL_TYPES
            ]

    def make_cases(self):
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
            config.model_types,
            config.training_datasets,
            config.data_normalization_methods,
            config.domain_adaptation_methods,
            config.freeze_submodules_options,
            config.num_target_shots_options,
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

    def _base_dataloader_config(self, case: Case) -> dict:
        return {
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

    def _base_loss_config(self) -> dict:
        return {
            "huber_delta": 0.5,
        }

    def _make_submodule_config(self, case: Case, submodule_type: str) -> TrainConfig:
        """Full TrainConfig for a p_oh/p_rad submodule, recursing through make_train_config
        so submodule cases get their own tuned merge and transfer wiring."""
        return self.make_train_config(case._replace(model_type=submodule_type, freeze_submodules=config.hyperparam_freeze_submodules))

    def _model_train_spec(self, case: Case, dataloader_config_base: dict) -> ModelTrainSpec:
        trb = "transport_study.modules.power_balance.trb.PowerBalanceTRB"
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
                    "min_val": 0,  # Minimum power in MW
                    "max_val": submodule_settings["max_val"],
                    "prng_seed": 42,
                    "in_size": 7,
                    "out_size": 1,
                    "data_normalization": case.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                },
            )
        elif case.model_type in MODEL_TYPES_WITH_SUBMODULES:
            return ModelTrainSpec(
                train_run_builder=trb,
                dataloader_config={
                    "input_vars": POWER_BALANCE_INPUT_VARS,
                    "target_vars": ["Wtot_MJ", "ds_source_idx"],
                    "state_vars": ["Wtot_MJ"],
                    # Bring these along for comparison / device weighting
                    "extra_vars": ["P_oh_MW", "P_rad_MW"],
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
                        "p_oh_predictor": self._make_submodule_config(case, "p_oh"),
                        "p_rad_predictor": self._make_submodule_config(case, "p_rad"),
                    },
                    "restore_submodules": True,  # Always restoring pre-trained submodules in this study
                },
            )
        elif case.model_type == "unstructured_nn":
            return ModelTrainSpec(
                train_run_builder=trb,
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
            )
        elif case.model_type == "transformer":
            return ModelTrainSpec(
                train_run_builder=trb,
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
            )
        else:
            raise ValueError(f"Unknown model type: {case.model_type}")

    def _tuned_model_init_updates(self, case: Case, tuned_config: TrainConfig) -> dict:
        # The scaling law has no NN of its own (its submodules carry their own tuned configs)
        updates = {}
        if case.model_type in [*SUBMODULE_MODEL_TYPES, "unstructured_nn", "sciml", "transformer"]:
            updates["nn_depth"] = tuned_config.model_init_config["nn_depth"]
            updates["nn_width"] = tuned_config.model_init_config["nn_width"]
        if case.model_type == "transformer":
            updates["d_model"] = tuned_config.model_init_config["d_model"]
            updates["num_heads"] = tuned_config.model_init_config["num_heads"]
            updates["history_len"] = tuned_config.model_init_config["history_len"]
        return updates

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

        if not results:
            logger.warning("No case results found to collect!")
            return xr.Dataset()

        # coords="different" stacks the per-case scalar coords along case_idx
        ds_merged = xr.concat(results, dim="case_idx", coords="different")
        return ds_merged

    ############
    # ANALYSIS #
    ############

    def _run_analysis(self, enable_parallelism: bool) -> None:
        # Per-case analysis (stage metrics + best/worst shot PDFs) is CPU-bound
        # matplotlib and numpy work, so we can fan it out over SLURM to speed things up
        if enable_parallelism:
            run_case_analysis_parallel(self)

        # Stage-resolved (rampup / flattop ohmic / flattop aux / rampdown)
        # time-averaged errors for every finished case, cached to
        # collected_metrics.nc alongside collected_results.nc
        metrics_ds = collect_metrics(self)

        results_ds = xr.load_dataset(self.collected_results_path())

        logger.opt(colors=True).info("<bold><magenta>TRAINING DATA COMPARISON</magenta></bold>")
        training_dataset_comparison(results_ds, self.figure_dir)

        logger.opt(colors=True).info("<bold><magenta>MODEL COMPARISON</magenta></bold>")
        model_comparison(results_ds, self.figure_dir)

        logger.opt(colors=True).info("<bold><magenta>DATA NORMALIZATION COMPARISON</magenta></bold>")
        data_normalization_comparison(results_ds, self.figure_dir)

        logger.opt(colors=True).info("<bold><magenta>DOMAIN ADAPTATION COMPARISON</magenta></bold>")
        domain_adaptation_comparison(results_ds, self.figure_dir)

        # Per-case deep dives: best/worst holdout shot PDFs by time-averaged error
        logger.opt(colors=True).info("<bold><magenta>CASE REPORTS</magenta></bold>")
        generate_case_reports(self, self.figure_dir)

        # One markdown table per case axis and combination of the other axes
        logger.opt(colors=True).info("<bold><magenta>COMPARISON TABLES</magenta></bold>")
        write_comparison_tables(results_ds, metrics_ds, self.figure_dir)


run_study = PowerBalanceStudy.run_study


if __name__ == "__main__":
    import os

    # Orchestrator does no GPU compute. Default jax to cpu so it can run on cpu
    # nodes. Submitted GPU jobs override this in their sbatch scripts.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    fire.Fire(
        {
            "run_study": run_study,
        }
    )
