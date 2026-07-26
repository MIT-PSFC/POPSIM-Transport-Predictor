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
from transport_study.datasets import UNIFORM_TIMEBASE_DT_S
from transport_study.modules.transport_predictor.train_configs import (
    make_transport_torax_config,
)
from transport_study.orchestration.case_analysis import run_case_analysis_parallel
from transport_study.orchestration.organize_data import PROFILE_TARGET_VARS
from transport_study.orchestration.study import (
    HYPERPARAM_TARGET_SHOTS,
    CaseGridConfig,
    ModelTrainSpec,
    Study,
)
from transport_study.power_balance_transfer.power_balance_study import (
    POWER_BALANCE_INPUT_VARS,
    SCALAR_SUBMODULE_SETTINGS,
)
from transport_study.profile_transfer.profile_study import (
    PROFILE_INPUT_VARS,
    VALID_GEOMETRY_BUILDERS,
)
from transport_study.transport_transfer.case_reports import generate_case_reports
from transport_study.transport_transfer.data_visualization import DataVisualization
from transport_study.transport_transfer.plotting import (
    domain_adaptation_comparison,
    model_comparison,
    training_dataset_comparison,
)
from transport_study.transport_transfer.study_metrics import collect_metrics
from transport_study.transport_transfer.tables import write_comparison_tables

# The physical inputs every transport predictor model consumes (the TRB adds
# ds_source_idx itself). Normalization happens inside the modules; there is
# deliberately no betan: beta quantities come from the evolving state Wtot
TRANSPORT_INPUT_VARS = [
    "Ip_MA",
    "B0",
    "ne20_line_avg",
    "R0",
    "a_minor",
    "kappa",
    "delta_top",
    "delta_bot",
    "P_aux_MW",
]

# Predicted profile channels (the loss and test eval compare against these)
TRANSPORT_PROFILE_TARGETS = ["ne20_rho", "Te_keV_rho"]

# Everything the transport loss reads from the target side: the profiles,
# their error-bar companions (softening the validation loss residual), the
# freshness flag masking both losses to timeslices with a fresh profile
# measurement, and the device label for per-device weighting
TRANSPORT_TARGET_VARS = [
    *TRANSPORT_PROFILE_TARGETS,
    "ne20_rho_error",
    "Te_keV_rho_error",
    "fresh_profiles",
    "ds_source_idx",
]

# Everything the env may need to seed a state at the segment start: the
# measured profiles (transformer buffer / torax initial condition), the stored
# energy (sciml power balance state), and every scalar input plus the device
# index (the torax sim-state variant builds a full TORAX initial state, which
# needs the t0 inputs, see TransportPredictorEnv.create_state)
TRANSPORT_STATE_VARS = ["Wtot_MJ", *TRANSPORT_PROFILE_TARGETS, *TRANSPORT_INPUT_VARS, "ds_source_idx"]

# The TORAX-backed model types (one per TORAX transport model)
TORAX_MODEL_TYPES = ("torax-constant", "torax-cgm", "torax-gyrobohm", "torax-qlknn")
# Model types that appear on the study's case grid
TOP_LEVEL_MODEL_TYPES = ("transformer", "sciml", *TORAX_MODEL_TYPES)
# Submodule pseudo-model-types, they appear as prereq cases of sciml:
# sciml -> power_balance + profile, power_balance -> p_oh + p_rad
SUBMODULE_MODEL_TYPES = ("power_balance", "profile", "p_oh", "p_rad")

# How the TORAX-backed models carry state between steps (see
# modules/transport_predictor/module.py): "rebuild" re-seeds a TORAX initial
# state from the stored ne/te each step, "carry" keeps the full ToraxSimState
VALID_TORAX_STATES = ("rebuild", "carry")

# Power balance variants allowed as the sciml stored-energy submodule (the
# structured ones, so profile-loss gradients flow into physical parameters)
VALID_POWER_BALANCE_MODEL_TYPES = ("sciml-taue-nn", "sciml-taue-scalinglaw")
# Profile predictor variants allowed as the sciml profile submodule
VALID_PROFILE_MODEL_TYPES = ("shape-init-pca", "shape-init-kmeans", "mlp")


class TransportStudy(Study):
    SWEEP_CONFIG_DIR = Path(PACKAGE_ROOT) / "transport_transfer" / "sweep_configs"
    STUDY_TYPE = "transport_transfer"
    DATA_VISUALIZATION = DataVisualization
    ANALYSIS_METRICS_MODULE = "transport_study.transport_transfer.study_metrics"
    ANALYSIS_REPORTS_MODULE = "transport_study.transport_transfer.case_reports"
    CASE_AXIS_FIELDS = (
        "model_types",
        "training_datasets",
        "domain_adaptation_methods",
        "freeze_submodules_options",
        "geometry_builders",
        "torax_state_options",
        "num_target_shots_options",
    )
    TUNED_DATALOADER_KEYS = ("segment_length_train", "segment_overlap_train", "batch_size")
    # huber_delta_grad only matters for the profile submodule cases (their
    # sweep tunes it); .get-fallback merge keeps it harmless everywhere else
    TUNED_LOSS_KEYS = ("huber_delta", "huber_delta_grad")

    ##################
    # INITIALIZATION #
    ##################
    class Config(CaseGridConfig):
        # The different cases being compared in this study
        model_types: tuple[str, ...] = Field(default_factory=lambda: TOP_LEVEL_MODEL_TYPES)
        freeze_submodules_options: tuple[bool, ...] = Field(default_factory=lambda: (True,))
        # Per-sample geometry builders to compare for torax-* model types,
        # ignored by every other model type (see VALID_GEOMETRY_BUILDERS)
        geometry_builders: tuple[str, ...] = Field(default_factory=lambda: ("circular",))
        # TORAX state carry variants to compare for torax-* model types,
        # ignored by every other model type (see VALID_TORAX_STATES)
        torax_state_options: tuple[str, ...] = Field(default_factory=lambda: ("rebuild",))
        num_target_shots_options: tuple[int, ...] = Field(default_factory=lambda: (0, 1, 3, 10, 32, -1))
        # Input normalization applied to the 11 dimensionless transport
        # features, one setting for the whole study run (not a case axis, so
        # it never appears in case names):
        # - physics: use the dimensionless features as-is
        # - physics-coral: per-device CORAL alignment fitted on them
        # - physics-zscore: per-device z-score fitted on them (mean/std only,
        #   no covariance alignment)
        data_normalization: str = "physics-coral"
        # Hyperparameter tuning case configuration
        # (hyperparam_domain_adaptation and hyperparam_num_target_shots live on CaseGridConfig)
        hyperparam_freeze_submodules: bool = True
        # Which variants back the sciml prereq submodules. Study-wide settings
        # rather than case axes; they change model semantics under unchanged
        # case names, so they are part of the config lock below
        power_balance_model_type: str = "sciml-taue-nn"
        power_balance_data_normalization: str = "physics"
        # Whether the power balance prereq case freezes ITS p_oh/p_rad
        # submodules during training (the power balance study default)
        power_balance_freeze_submodules: bool = True
        profile_model_type: str = "shape-init-pca"

        COMPAT_HYPERPARAM_FIELDS = (
            "data_normalization",
            "hyperparam_domain_adaptation",
            "hyperparam_freeze_submodules",
            "hyperparam_num_target_shots",
            "power_balance_data_normalization",
            "power_balance_freeze_submodules",
            "power_balance_model_type",
            "profile_model_type",
        )

        @field_validator("model_types")
        @classmethod
        def _validate_model_types(cls, v: tuple[str, ...]) -> tuple[str, ...]:
            valid = {*TOP_LEVEL_MODEL_TYPES, *SUBMODULE_MODEL_TYPES}
            for mt in v:
                if mt not in valid:
                    raise ValueError(f"Invalid model type: {mt}. Must be one of {sorted(valid)}.")
            return v

        @field_validator("geometry_builders")
        @classmethod
        def _validate_geometry_builders(cls, v: tuple[str, ...]) -> tuple[str, ...]:
            for gb in v:
                if gb not in VALID_GEOMETRY_BUILDERS:
                    raise ValueError(f"Invalid geometry builder: {gb}. Must be one of {VALID_GEOMETRY_BUILDERS}.")
            return v

        @field_validator("torax_state_options")
        @classmethod
        def _validate_torax_states(cls, v: tuple[str, ...]) -> tuple[str, ...]:
            for ts in v:
                if ts not in VALID_TORAX_STATES:
                    raise ValueError(f"Invalid torax state option: {ts}. Must be one of {VALID_TORAX_STATES}.")
            return v

        @field_validator("data_normalization")
        @classmethod
        def _validate_data_normalization(cls, v: str) -> str:
            valid = ("physics", "physics-coral", "physics-zscore")
            if v not in valid:
                raise ValueError(f"Invalid data normalization method: {v}. Must be one of {valid}.")
            return v

        @field_validator("power_balance_model_type")
        @classmethod
        def _validate_power_balance_model_type(cls, v: str) -> str:
            if v not in VALID_POWER_BALANCE_MODEL_TYPES:
                raise ValueError(f"Invalid power balance model type: {v}. Must be one of {VALID_POWER_BALANCE_MODEL_TYPES}.")
            return v

        @field_validator("profile_model_type")
        @classmethod
        def _validate_profile_model_type(cls, v: str) -> str:
            if v not in VALID_PROFILE_MODEL_TYPES:
                raise ValueError(f"Invalid profile model type: {v}. Must be one of {VALID_PROFILE_MODEL_TYPES}.")
            return v

    @dataclass
    class Case(Study.Case):
        """
        model_type: The type of transport predictor model to use.
        - transformer: recurrent causal attention over a rolling buffer of past profiles
        - sciml: a time-dependent power balance evolves the stored energy, a
          time-independent profile predictor maps the state-implied betan to profiles
        - torax-constant / torax-cgm / torax-gyrobohm / torax-qlknn: one-step
          differentiable TORAX simulation with NN-predicted transport, source
          shape, and edge parameters
        - power_balance / profile / p_oh / p_rad: submodule predictors, appear
          only as prereq cases of sciml (power_balance itself chains p_oh and p_rad)

        training_data: The dataset(s) used for training
        - cmod: C-Mod only
        - tcv: TCV only
        - cmod_tcv: C-Mod + TCV
        - exnihilo: No historic training data

        domain_adaptation: The method for domain adaptation between source and target devices.
        - none: No domain adaptation, train and test on the same device(s). This is used for hyperparameter tuning and as a baseline for comparison, answering the question "what is the best possible performance we could expect if we had a bunch of data?"
        - weighted: Add a small amount of highly-weighted target data during training
        - addition: Add target shots to the training set as normal samples, no weighting
        - transfer: Train on source data, freeze all but the last layers of the model, and fine-tune on a small amount of target data
        - transfer_pretrain: The pretrain half of a transfer case, never a case-grid axis value (see Study.Case.transfer_pretrain_case). Trains on historic data only with checkpoint selection on the target test set. Stat normalizations (physics-coral / physics-zscore) fit the stat stage on historic + the transfer case's target shots, stateless ones share one twin at 0 target shots

        freeze_submodules: Whether to freeze the power_balance / profile
        submodules of the sciml model during training. Only meaningful for
        sciml, every other model type is pinned to the hyperparam default.

        geometry_builder: Per-sample TORAX geometry construction, only meaningful for torax-* model
        types (every other model type is pinned to "circular").
        - circular: large-aspect-ratio analytic geometry (delta = 0 everywhere)
        - miller: shaped Miller geometry driven by delta_top/delta_bot

        torax_state: How the torax-* model types carry state between steps,
        only meaningful for torax-* (every other model type is pinned to "rebuild").
        - rebuild: carry only ne/te and rebuild a TORAX initial state each step
        - carry: carry the full ToraxSimState pytree between steps

        num_target_shots: The number of shots included in the training data from the target dataset, or -1 to include all shots (including all shots in training is cheating, but again answers the question of what is the best possible performance).
        """

        freeze_submodules: bool
        geometry_builder: str
        torax_state: str

        VALID_MODEL_TYPES = (*TOP_LEVEL_MODEL_TYPES, *SUBMODULE_MODEL_TYPES)
        # geometry_builder and torax_state defaults are suppressed from the
        # case name, so the default grid keeps clean case names and new axes
        # never rename pre-existing cases
        STR_TOKEN_FIELDS = (
            ("freeze_", "freeze_submodules"),
            ("geom_", "geometry_builder", "circular"),
            ("tstate_", "torax_state", "rebuild"),
        )
        # geometry_builder and torax_state are deliberately NOT hyperparam
        # fields (like model_type): they stay untouched by
        # _hyperparam_field_values, so miller / carry cases get their own
        # hyperparameter sweep and tuned config instead of inheriting the
        # circular / rebuild tuned hyperparameters
        HYPERPARAM_FIELDS = ("domain_adaptation", "freeze_submodules", "num_target_shots")

        # The dataclass decorator would null an inherited __hash__
        __hash__ = Study.Case.__hash__

        def __init__(
            self,
            model_type: str,
            training_data,
            domain_adaptation: str | None,
            freeze_submodules: bool,
            num_target_shots: int,
            geometry_builder: str = "circular",
            torax_state: str = "rebuild",
        ):
            self.freeze_submodules = freeze_submodules
            self.geometry_builder = geometry_builder
            self.torax_state = torax_state
            self._init_common(model_type, training_data, domain_adaptation, num_target_shots)

        def _normalization_method(self) -> str | None:
            # One study-wide setting, not a case axis
            return config.data_normalization

        def _validate(self):
            super()._validate()
            if self.geometry_builder not in VALID_GEOMETRY_BUILDERS:
                raise ValueError(f"Unknown geometry builder: {self.geometry_builder}")
            if self.torax_state not in VALID_TORAX_STATES:
                raise ValueError(f"Unknown torax state: {self.torax_state}")
            if not self.model_type.startswith("torax-"):
                if self.geometry_builder != "circular":
                    raise ValueError(f"geometry_builder only applies to torax model types, not {self.model_type}")
                if self.torax_state != "rebuild":
                    raise ValueError(f"torax_state only applies to torax model types, not {self.model_type}")
            if self.model_type in SUBMODULE_MODEL_TYPES and self.freeze_submodules != config.hyperparam_freeze_submodules:
                raise ValueError(
                    f"freeze_submodules should be a dummy value ({config.hyperparam_freeze_submodules}) for submodule {self.model_type}"
                )

        def _model_type_prereqs(self) -> list[Study.Case]:
            # sciml restores a pre-trained power balance and profile predictor;
            # the power balance itself restores pre-trained p_oh/p_rad
            if self.model_type == "sciml":
                submodule_types = ("power_balance", "profile")
            elif self.model_type == "power_balance":
                submodule_types = ("p_oh", "p_rad")
            else:
                return []
            return [
                self.replace(model_type=submodule_type, freeze_submodules=config.hyperparam_freeze_submodules)
                for submodule_type in submodule_types
            ]

    def make_cases(self):
        cases = []
        # Make every case we're interested in for this study
        for (
            model_type,
            training_dataset,
            domain_adaptation,
            freeze_submodules,
            geometry_builder,
            torax_state,
            num_target_shots,
        ) in product(
            config.model_types,
            config.training_datasets,
            config.domain_adaptation_methods,
            config.freeze_submodules_options,
            config.geometry_builders,
            config.torax_state_options,
            config.num_target_shots_options,
        ):
            if domain_adaptation is None:
                if training_dataset.exnihilo:
                    if num_target_shots == 0:
                        continue  # Can't train from nothing with 0 target shots
                elif num_target_shots != HYPERPARAM_TARGET_SHOTS:
                    continue  # Invalid case, skip
            if model_type != "sciml" and freeze_submodules != config.hyperparam_freeze_submodules:
                continue  # No submodules to freeze, just do one of the two
            if not model_type.startswith("torax-") and geometry_builder != "circular":
                continue  # geometry_builder only applies to torax model types
            if not model_type.startswith("torax-") and torax_state != "rebuild":
                continue  # torax_state only applies to torax model types

            case = self.Case(
                model_type=model_type,
                training_data=training_dataset,
                domain_adaptation=domain_adaptation,
                freeze_submodules=freeze_submodules,
                num_target_shots=num_target_shots,
                geometry_builder=geometry_builder,
                torax_state=torax_state,
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
            # Far below the power balance study's 4096: every torax sample
            # runs a 100-step differentiated TORAX rollout under vmap (with
            # jax.checkpoint remat, see _advance_one_step). Measure before
            # raising; memory and time scale linearly with batch and
            # segment_length_train
            "batch_size": 64,
            # Part of validation, should be left alone during hyperparameter tuning
            "segment_length_val": None,
            "segment_overlap_val": 0,
        }

    def _base_loss_config(self) -> dict:
        return {
            # The loss runs on peak-normalized profiles (target scaled to max
            # 1), so the delta reads as a fractional error. Fallback for cases
            # run without a tuned config; validation loss is delta-free
            "huber_delta": 0.1,
            # Only read by the profile submodule cases (ProfilePredictorTRB),
            # matching the profile study's loss configuration; the transport
            # and power balance losses ignore these keys
            "gradient_weight": 0.1,
            "huber_delta_grad": 1.0,
            # Down-weighting of the residual inside the GP-fit error bars,
            # validation loss only. Read by both the transport loss and the
            # profile submodule cases
            "within_error_weight": 0.01,
            # Anchor terms keeping the sciml submodule predictions close to
            # the measured signals while the whole module trains on the
            # profiles: the power balance's Wtot plus its own p_oh/p_rad
            # submodules. Training loss only, and a no-op for model types
            # whose target_vars lack the measured signals (only the sciml
            # case carries them). The power balance submodule prereq case
            # also reads anchor_weight_p_oh/_p_rad through PowerBalanceTRB
            "anchor_weight_wtot": 0.1,
            "anchor_weight_p_oh": 0.1,
            "anchor_weight_p_rad": 0.1,
        }

    def _base_optimizer_config(self) -> dict:
        return {
            **super()._base_optimizer_config(),
            # Cap on global L2 gradient norm per update, guards against rare
            # gradient spikes from the differentiated TORAX solve NaN-ing a run
            "grad_clip_max_norm": 1.0,
            # During joint sciml training the restored power balance submodule
            # (including its own p_oh/p_rad submodules underneath) trains at a
            # reduced rate so profile-loss gradients do not pull the
            # stored-energy dynamics far from their pretrained behavior. The
            # profile predictor is the part being adapted, it keeps the full
            # schedule. Labeling is by pytree path, so this also applies to
            # the transfer-mode last-layer partition and is a no-op for model
            # types without a power_balance attribute. The power_balance
            # prereq case reuses this config through PowerBalanceTRB, where
            # only its p_oh_predictor/p_rad_predictor paths match
            "submodule_lr_factors": {
                "power_balance": 0.1,
                "p_oh_predictor": 0.1,
                "p_rad_predictor": 0.1,
            },
        }

    def _make_submodule_config(self, case: Case, submodule_type: str) -> TrainConfig:
        """Full TrainConfig for a submodule prereq case, recursing through make_train_config
        so submodule cases get their own tuned merge and transfer wiring."""
        return self.make_train_config(case.replace(model_type=submodule_type, freeze_submodules=config.hyperparam_freeze_submodules))

    def _transport_dataloader_config(self, dataloader_config_base: dict) -> dict:
        """Dataloader config shared by every top-level transport model type."""
        return {
            "input_vars": TRANSPORT_INPUT_VARS,
            "target_vars": TRANSPORT_TARGET_VARS,
            "state_vars": TRANSPORT_STATE_VARS,
            **dataloader_config_base,
        }

    def _model_train_spec(self, case: Case, dataloader_config_base: dict) -> ModelTrainSpec:
        trb = "transport_study.modules.transport_predictor.trb.TransportPredictorTRB"
        pb_trb = "transport_study.modules.power_balance.trb.PowerBalanceTRB"
        if case.model_type in ("p_oh", "p_rad"):
            # Identical to the power balance study's scalar submodule cases
            # (they load power-balance-prepped data through PowerBalanceTRB)
            submodule_settings = SCALAR_SUBMODULE_SETTINGS[case.model_type]
            return ModelTrainSpec(
                train_run_builder=submodule_settings["train_run_builder"],
                dataloader_config={
                    "target_vars": submodule_settings["target_vars"],
                    "input_vars": POWER_BALANCE_INPUT_VARS,
                    "data_train_run_builder": pb_trb,  # Needed for submodules
                    **dataloader_config_base,
                    # Scalar-signal training is cheap, match the power balance study
                    "batch_size": 4096,
                },
                model_init_config={
                    "nn_depth": 2,
                    "nn_width": 16,
                    "prng_seed": 42,
                    "in_size": 7,
                    "out_size": 1,
                    "data_normalization": config.power_balance_data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                },
            )
        elif case.model_type == "power_balance":
            # The stored-energy submodule of sciml, trained exactly like the
            # power balance study's structured cases
            return ModelTrainSpec(
                train_run_builder=pb_trb,
                dataloader_config={
                    "input_vars": POWER_BALANCE_INPUT_VARS,
                    # The measured powers are targets so the training loss can
                    # anchor the p_oh/p_rad submodule predictions to them
                    # (anchor_weight_* in the loss config), matching the power
                    # balance study's structured cases
                    "target_vars": ["Wtot_MJ", "P_oh_MW", "P_rad_MW", "ds_source_idx"],
                    "state_vars": ["Wtot_MJ"],
                    **dataloader_config_base,
                    # Scalar-signal training is cheap, match the power balance study
                    "batch_size": 4096,
                },
                model_init_config={
                    "model_type": config.power_balance_model_type,
                    "data_normalization": config.power_balance_data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    "freeze_submodules": config.power_balance_freeze_submodules,
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
        elif case.model_type == "profile":
            # The profile submodule of sciml, trained exactly like the profile
            # study's time-independent cases (profile-prepped data has betan)
            return ModelTrainSpec(
                train_run_builder="transport_study.modules.profile_predictor.trb.ProfilePredictorTRB",
                dataloader_config={
                    "input_vars": PROFILE_INPUT_VARS,
                    # Profiles plus their gradient / error-bar companions, the
                    # validation loss softens the residual inside the error bars
                    "target_vars": [*PROFILE_TARGET_VARS, "ds_source_idx"],
                    "extra_vars": ["Te_shape", "ne_shape"],
                    **dataloader_config_base,
                    # Timeslice samples are cheap, match the profile study
                    "batch_size": 2048,
                },
                model_init_config={
                    "model_type": config.profile_model_type,
                    "data_normalization": config.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    # Profile study hyperparam default; the shapes are an
                    # initial-guess basis, not physics to fine-tune here
                    "freeze_shapes": True,
                    "te_shape_var": "Te_shape",
                    "ne_shape_var": "ne_shape",
                    "n_shapes": 3,
                    "nn_depth": 2,
                    "nn_width": 16,
                    "in_size": 10,  # Dimensionless nn_inputs derived from the raw input_vars, includes log(nu_star)
                    "softmax_temp": 1,
                    "prng_seed": 42,
                },
            )
        elif case.model_type == "sciml":
            return ModelTrainSpec(
                train_run_builder=trb,
                dataloader_config={
                    **self._transport_dataloader_config(dataloader_config_base),
                    # The measured stored energy and powers are extra targets
                    # so the training loss can anchor the submodule
                    # predictions to them (anchor_weight_* in the loss config)
                    "target_vars": [
                        *TRANSPORT_TARGET_VARS,
                        "Wtot_MJ",
                        "P_oh_MW",
                        "P_rad_MW",
                    ],
                },
                model_init_config={
                    "model_type": case.model_type,
                    "data_normalization": config.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    "freeze_submodules": case.freeze_submodules,
                    "prng_seed": 42,
                    "submodules": {
                        "power_balance": self._make_submodule_config(case, "power_balance"),
                        "profile_predictor": self._make_submodule_config(case, "profile"),
                    },
                    "restore_submodules": True,  # Always restoring pre-trained submodules in this study
                },
            )
        elif case.model_type == "transformer":
            return ModelTrainSpec(
                train_run_builder=trb,
                dataloader_config=self._transport_dataloader_config(dataloader_config_base),
                model_init_config={
                    "model_type": case.model_type,
                    "data_normalization": config.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    "d_model": 16,  # Token embedding width
                    "num_heads": 2,
                    "history_len": 20,  # Attention window over past profiles
                    "nn_depth": 2,  # MLP head after attention
                    "nn_width": 16,
                    "prng_seed": 42,
                },
            )
        elif case.model_type.startswith("torax-"):
            transport_model = case.model_type.removeprefix("torax-")
            return ModelTrainSpec(
                train_run_builder=trb,
                dataloader_config=self._transport_dataloader_config(dataloader_config_base),
                model_init_config={
                    "model_type": case.model_type,
                    "data_normalization": config.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    "nn_depth": 2,
                    "nn_width": 16,
                    "torax_config": make_transport_torax_config(transport_model),
                    "torax_state": case.torax_state,
                    "geometry_builder": case.geometry_builder,
                    "delta_exponent": 2.0,
                    # One TORAX solver step per dataset timestep
                    "sim_dt": UNIFORM_TIMEBASE_DT_S,
                    "prng_seed": 42,
                },
            )
        else:
            raise ValueError(f"Unknown model type: {case.model_type}")

    def _tuned_model_init_updates(self, case: Case, tuned_config: TrainConfig) -> dict:
        # sciml has no NN of its own (its submodules carry their own tuned configs)
        updates = {}
        if case.model_type != "sciml":
            updates["nn_depth"] = tuned_config.model_init_config["nn_depth"]
            updates["nn_width"] = tuned_config.model_init_config["nn_width"]
        if case.model_type == "transformer":
            updates["d_model"] = tuned_config.model_init_config["d_model"]
            updates["num_heads"] = tuned_config.model_init_config["num_heads"]
            updates["history_len"] = tuned_config.model_init_config["history_len"]
        if case.model_type == "profile" and config.profile_model_type in ("shape-init-pca", "shape-init-kmeans"):
            updates["n_shapes"] = tuned_config.model_init_config["n_shapes"]
            updates["softmax_temp"] = tuned_config.model_init_config["softmax_temp"]
        return updates

    ##############
    # COLLECTION #
    ##############

    # Coords describing which case a record belongs to
    _CASE_COORD_NAMES = (
        "case_idx",
        "model_type",
        "training_data",
        "domain_adaptation",
        "freeze_submodules",
        "geometry_builder",
        "torax_state",
        "num_target_shots",
    )

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

        logger.opt(colors=True).info("<bold><magenta>DOMAIN ADAPTATION COMPARISON</magenta></bold>")
        domain_adaptation_comparison(results_ds, self.figure_dir)

        # Per-case deep dives: best/worst holdout shot PDFs by time-averaged error
        logger.opt(colors=True).info("<bold><magenta>CASE REPORTS</magenta></bold>")
        generate_case_reports(self, self.figure_dir)

        # One markdown table per case axis and combination of the other axes
        logger.opt(colors=True).info("<bold><magenta>COMPARISON TABLES</magenta></bold>")
        write_comparison_tables(results_ds, metrics_ds, self.figure_dir)


run_study = TransportStudy.run_study


if __name__ == "__main__":
    fire.Fire(
        {
            "run_study": run_study,
        }
    )
