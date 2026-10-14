from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar

import fire
import netCDF4  # noqa: F401
from pydantic import Field
from transport_validation_datasets.machine.generic import UNIFORM_TIMEBASE_DT

from transport_study import PACKAGE_ROOT
from transport_study.config import CaseAxis, config
from transport_study.modules.normalization import (
    FEATURE_NORMALIZATIONS,
    INPUT_NORMALIZATIONS,
)
from transport_study.modules.profile_predictor.module import MODEL_TYPES_WITH_SHAPES
from transport_study.modules.profile_predictor.torax_module import (
    TORAX_MODEL_TYPES,
    VALID_GEOMETRY_BUILDERS,
)
from transport_study.modules.transport_predictor.module import (
    MULTIOBJECTIVE_MODEL_TYPES,
    SUBMODULE_MODEL_TYPES,
)
from transport_study.modules.transport_predictor.train_configs import (
    VALID_TORAX_STATES,
    make_transport_torax_config,
)
from transport_study.orchestration.case_analysis import run_summary_analysis
from transport_study.orchestration.organize_data import PROFILE_TARGET_VARS
from transport_study.orchestration.study import (
    BASE_MODEL_SIZE,
    NN_INIT_DEFAULTS,
    CaseGridConfig,
    ModelTrainSpec,
    Study,
    model_type_field_choices,
)
from transport_study.orchestration.target_shots import BASE_TARGET_SHOT_ORDER
from transport_study.power_balance_transfer.power_balance_study import (
    POWER_BALANCE_LOSS_DEFAULTS,
    SCALAR_SUBMODULE_SETTINGS,
    PowerBalanceStudy,
    scalar_submodule_train_spec,
    structured_power_balance_train_spec,
)
from transport_study.profile_transfer.profile_study import (
    ProfileStudy,
    profile_train_spec,
)
from transport_study.transport_transfer.data_visualization import DataVisualization
from transport_study.transport_transfer.plotting import COMPARISON_FAMILIES, LAYOUT
from transport_study.transport_transfer.tables import SPEC as TABLE_SPEC

# The physical inputs every transport predictor model consumes (the TRB adds ds_source_idx itself).
# Normalization happens inside the modules.
# There is deliberately no beta_tor_norm, beta quantities come from the evolving state Wtot.
TRANSPORT_INPUT_VARS = [
    "ip_MA",
    "b0",
    "b_geo",
    "n_e_line_average_1e20",
    "geometric_axis_r",
    "minor_radius",
    "elongation",
    "triangularity_upper",
    "triangularity_lower",
    "power_additional_MW",
]

# Predicted profile channels (the loss and test eval compare against these)
TRANSPORT_PROFILE_TARGETS = ["n_e_1e20", "t_e_keV"]

# Everything the transport loss reads from the target side:
# the profiles, their GP-fit gradients and every error bar (the chi validation loss divides by them),
# the freshness flag masking both losses to timeslices with a fresh profile measurement,
# the one masking the stored-energy and ohmic power anchors to fresh reconstructions (trb_utils.freshness_weight),
# and the device label for per-device weighting
TRANSPORT_TARGET_VARS = [*PROFILE_TARGET_VARS, "fresh_profile", "fresh_equilibrium", "ds_source_idx"]

# The measured powers every top-level case loads (the stored energy is a state var of every case),
# so the training segments are the same whatever a case anchors (a NaN in any loaded variable drops a segment)
MEASURED_POWER_VARS = ["power_ohm_MW", "power_radiated_MW"]
# The signals a multiobjective case anchors, the ones TORAX infers from its state (no P_rad, see TransportPredictorToraxBase)
MULTIOBJECTIVE_TARGET_VARS = ["energy_mhd_MJ", "power_ohm_MW"]

# Segment batch of the top-level cases, the largest that every transport model type fits at its sweep maximum,
# 32 batches per epoch on the largest set. Transformer's memory caps it (TORAX could go faster at larger batch size).
# On an A100-80GB:
# - torax-gyrobohm 155 s and torax-qlknn 221 s per epoch at 15 and 17 GB, against 268 and 347 s at 1024.
# - The transformer peaks at 72 GB, inside the 90 percent XLA pool of slurm_utils.GPU_JOB_ENV,
#   and runs out of memory at 3072 (one 100 GB buffer).
TRANSPORT_BATCH_SIZE = 2048

# Everything the env may need to seed a state at the segment start: the
# measured profiles (transformer buffer / torax initial condition), the stored
# energy (sciml power balance state), and every scalar input plus the device
# index (the torax sim-state variant builds a full TORAX initial state, which
# needs the t0 inputs, see TransportPredictorEnv.create_state)
TRANSPORT_STATE_VARS = ["energy_mhd_MJ", *TRANSPORT_PROFILE_TARGETS, *TRANSPORT_INPUT_VARS, "ds_source_idx"]

# Model types that appear on the study's case grid
TOP_LEVEL_MODEL_TYPES = ("transformer", "sciml", *TORAX_MODEL_TYPES)
# Every model type a case of this study can have, the prereq pseudo types included
STUDY_MODEL_TYPES = (*TOP_LEVEL_MODEL_TYPES, *SUBMODULE_MODEL_TYPES)
# The submodule cases that train with power_balance_data_normalization instead of the study-wide data_normalization
POWER_BALANCE_SUBMODULE_TYPES = ("power_balance", "p_oh", "p_rad")

# Power balance variants allowed as the sciml stored-energy submodule (the
# structured ones, so profile-loss gradients flow into physical parameters)
VALID_POWER_BALANCE_MODEL_TYPES = ("sciml-taue-nn", "sciml-taue-scalinglaw")
# Profile predictor variants allowed as the sciml profile submodule
VALID_PROFILE_MODEL_TYPES = (*MODEL_TYPES_WITH_SHAPES, "mlp")
# The shapes of a shape-init profile submodule are an initial-guess basis, not physics the sciml fine-tunes, so they stay frozen
PROFILE_SUBMODULE_FREEZE_SHAPES = True


class TransportStudy(Study):
    SWEEP_CONFIG_DIR = Path(PACKAGE_ROOT) / "transport_transfer" / "sweep_configs"
    STUDY_TYPE = "transport_transfer"
    DATA_VISUALIZATION = DataVisualization
    # Scored with the validation chi, plus the power balance error metrics
    ANALYSIS_METRICS_MODULE = "transport_study.transport_transfer.study_metrics"
    ANALYSIS_REPORTS_MODULE = "transport_study.transport_transfer.case_reports"
    CHI_VALIDATION_LOSS = True
    RESIDUAL_SCALE_LOSS = True
    ANCHOR_DELTA_MODEL_TYPES: ClassVar[dict[str, str]] = {
        "energy_mhd_MJ": "power_balance",
        "power_ohm_MW": "p_oh",
        "power_radiated_MW": "p_rad",
    }
    # huber_delta_grad belongs to the profile chi-huber losses, a power balance pseudo-type's config has none
    TUNED_LOSS_KEYS = ("huber_delta", "huber_delta_grad")

    ##################
    # INITIALIZATION #
    ##################
    class Config(CaseGridConfig):
        # The different cases being compared in this study
        model_types: Annotated[tuple[str, ...], CaseAxis("model_type")] = Field(default_factory=lambda: TOP_LEVEL_MODEL_TYPES)
        freeze_submodules_options: Annotated[tuple[bool, ...], CaseAxis("freeze_submodules")] = Field(default_factory=lambda: (False,))
        # Per-sample geometry builders to compare for torax-* model types,
        # ignored by every other model type (see VALID_GEOMETRY_BUILDERS)
        geometry_builders: Annotated[tuple[str, ...], CaseAxis("geometry_builder")] = Field(default_factory=lambda: ("circular",))
        # TORAX state carry variants to compare for torax-* model types,
        # ignored by every other model type (see VALID_TORAX_STATES)
        torax_state_options: Annotated[tuple[str, ...], CaseAxis("torax_state")] = Field(default_factory=lambda: ("rebuild",))
        # Whether the MULTIOBJECTIVE_MODEL_TYPES also anchor their stored energy and ohmic power, ignored by every other model type
        multiobjective_options: Annotated[tuple[bool, ...], CaseAxis("multiobjective")] = Field(default_factory=lambda: (False,))
        num_target_shots_options: Annotated[tuple[int, ...], CaseAxis("num_target_shots")] = Field(
            default_factory=lambda: (0, 1, 3, 10, 32)
        )
        # Input normalization applied to the 11 dimensionless transport
        # features, one setting for the whole study run (not a case axis, so
        # it never appears in case names):
        # - physics: use the dimensionless features as-is
        # - physics-coral: per-device CORAL alignment fitted on them
        # - physics-zscore: per-device z-score fitted on them (mean/std only,
        #   no covariance alignment)
        data_normalization: str = "physics"
        # Hyperparameter tuning case configuration
        # (hyperparam_domain_adaptation and hyperparam_num_target_shots live on CaseGridConfig)
        hyperparam_freeze_submodules: bool = False
        hyperparam_multiobjective: bool = False
        # Which variants back the sciml prereq submodules.
        # Study-wide settings rather than case axes.
        # They change model semantics under unchanged case names, so they are locked.
        power_balance_model_type: str = "sciml-taue-nn"
        power_balance_data_normalization: str = "physics"
        # Whether the power balance prereq case freezes ITS p_oh/p_rad
        # submodules during training (the power balance study default)
        power_balance_freeze_submodules: bool = False
        profile_model_type: str = "shape-init-pca"

        FIELD_CHOICES: ClassVar[dict[str, tuple]] = {
            **CaseGridConfig.FIELD_CHOICES,
            **model_type_field_choices(STUDY_MODEL_TYPES),
            "geometry_builders": VALID_GEOMETRY_BUILDERS,
            "torax_state_options": VALID_TORAX_STATES,
            "data_normalization": FEATURE_NORMALIZATIONS,
            "power_balance_data_normalization": INPUT_NORMALIZATIONS,
            "power_balance_model_type": VALID_POWER_BALANCE_MODEL_TYPES,
            "profile_model_type": VALID_PROFILE_MODEL_TYPES,
        }

    @dataclass
    class Case(Study.Case):
        """The transport case, see Study.Case for the shared axes. Normalization is study-wide, never an axis.

        model_type:
        - transformer: recurrent causal attention over a rolling buffer of past predicted profiles
        - sciml: a time-dependent power balance evolves the stored energy,
          a time-independent profile predictor maps the state-implied beta_tor_norm to profiles
        - torax-constant / torax-gyrobohm / torax-qlknn: one differentiable TORAX step per call
          with NN-predicted transport, source shape and edge parameters
        - power_balance / profile / p_oh / p_rad: the submodule predictors, prereq cases of sciml alone
          (power_balance itself chains p_oh and p_rad)

        freeze_submodules: whether the sciml keeps the p_oh / p_rad predictors of its power balance fixed,
        as the power balance study's freeze does, its tau_E network and the profile predictor always train.
        Pinned to the hyperparam value for every other model type

        geometry_builder: the per-sample TORAX geometry of the torax-* types, pinned to circular for the rest
        - circular: TORAX's large-aspect-ratio analytic geometry
        - miller: the shaped Miller geometry driven by the triangularities

        torax_state: how the torax-* types carry state between steps, pinned to rebuild for the rest
        - rebuild: carry ne / te only and rebuild the TORAX state each step (T_i := T_e, psi from Ip)
        - carry: also carry T_i and psi, so the ion channel and the current evolve over the rollout

        multiobjective: whether the training loss also anchors the stored energy and the ohmic power to the measured
        signals (MULTIOBJECTIVE_MODEL_TYPES alone, sciml always anchors its submodules). The transformer predicts them with
        two more head outputs, TORAX infers them from its own state. Shares the plain case's sweep, named mo_True
        """

        freeze_submodules: bool
        geometry_builder: str
        torax_state: str
        multiobjective: bool

        VALID_MODEL_TYPES = STUDY_MODEL_TYPES
        # Unfrozen submodules and the geometry_builder and torax_state defaults are suppressed from the case name,
        # so the default grid keeps clean case names and new axes never rename pre-existing cases
        STR_TOKEN_FIELDS = (
            ("freeze_", "freeze_submodules", False),
            ("geom_", "geometry_builder", "circular"),
            ("tstate_", "torax_state", "rebuild"),
            ("mo_", "multiobjective", False),
        )
        # geometry_builder and torax_state are deliberately NOT hyperparam fields (like model_type):
        # miller and carry cases get their own hyperparameter sweep and tuned config
        # instead of inheriting the circular and rebuild ones.
        # multiobjective is one, a multiobjective case takes the plain case's tuned config
        HYPERPARAM_FIELDS = ("domain_adaptation", "freeze_submodules", "multiobjective", "num_target_shots")

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
            target_shot_order: str = BASE_TARGET_SHOT_ORDER,
            multiobjective: bool = False,
            model_size: str = BASE_MODEL_SIZE,
        ):
            self.freeze_submodules = freeze_submodules
            self.geometry_builder = geometry_builder
            self.torax_state = torax_state
            self.multiobjective = multiobjective
            self.init_common(model_type, training_data, domain_adaptation, num_target_shots, target_shot_order, model_size)

        def normalization_method(self) -> str | None:
            # Study-wide settings, not case axes
            if self.model_type in POWER_BALANCE_SUBMODULE_TYPES:
                return config.power_balance_data_normalization
            return config.data_normalization

        def validate(self):
            super().validate()
            if self.geometry_builder not in VALID_GEOMETRY_BUILDERS:
                raise ValueError(f"Unknown geometry builder: {self.geometry_builder}")
            if self.torax_state not in VALID_TORAX_STATES:
                raise ValueError(f"Unknown torax state: {self.torax_state}")

        @classmethod
        def pin_inapplicable_axes(cls, fields: dict) -> dict:
            pinned = super().pin_inapplicable_axes(fields)
            # Only sciml has submodules to freeze
            if fields["model_type"] != "sciml":
                pinned["freeze_submodules"] = config.hyperparam_freeze_submodules
            # The geometry and the carried TORAX state only enter the TORAX families
            if fields["model_type"] not in TORAX_MODEL_TYPES:
                pinned["geometry_builder"] = "circular"
                pinned["torax_state"] = "rebuild"
            if fields["model_type"] not in MULTIOBJECTIVE_MODEL_TYPES:
                pinned["multiobjective"] = config.hyperparam_multiobjective
            return pinned

        def model_type_prereqs(self) -> list[Study.Case]:
            # sciml restores a pre-trained power balance and profile predictor.
            # The power balance itself restores pre-trained p_oh/p_rad.
            # Every prereq is built at the case's own model size, so a sized case sizes the whole model.
            if self.model_type == "sciml":
                submodule_types = ("power_balance", "profile")
            elif self.model_type == "power_balance":
                submodule_types = ("p_oh", "p_rad")
            elif self.model_type in MULTIOBJECTIVE_MODEL_TYPES and self.multiobjective:
                # The Wtot and P_oh anchors take the huber deltas the hyperparam power_balance and p_oh cases were tuned with.
                # The model type guard matters when the hyperparam value is True, which pins the prereqs' own flag True
                return [self.hyperparam_case_of(model_type) for model_type in ("power_balance", "p_oh")]
            else:
                return []
            return [self.replace_pinned(model_type=submodule_type) for submodule_type in submodule_types]

    #############
    # EXECUTION #
    #############

    @classmethod
    def sweep_config(cls, model_type: str) -> dict:
        """A pseudo-type sweeps with the owning study's yaml of its configured variant, so it tunes as that study's case would.

        p_oh, p_rad and power_balance (as config.power_balance_model_type) take the power balance study's,
        profile (as config.profile_model_type) the profile study's, every top-level type this study's own.
        """
        if model_type in POWER_BALANCE_SUBMODULE_TYPES:
            variant = config.power_balance_model_type if model_type == "power_balance" else model_type
            return PowerBalanceStudy.sweep_config(variant)
        if model_type == "profile":
            return ProfileStudy.sweep_config(config.profile_model_type)
        return super().sweep_config(model_type)

    def base_dataloader_config(self, case: Case) -> dict:
        return {
            **self.target_split_config(case),
            # Fixed loader settings, no sweep changes them (test_sweep_configs)
            "segment_length_train": 100,
            "segment_overlap_train": 50,
            "batch_size": TRANSPORT_BATCH_SIZE,
            # Validation scores whole episodes
            "segment_length_val": None,
            "segment_overlap_val": 0,
        }

    def base_loss_config(self, case: Case) -> dict:
        # The power balance pseudo-types train with the power balance study's losses and fallbacks
        if case.model_type in POWER_BALANCE_SUBMODULE_TYPES:
            return dict(POWER_BALANCE_LOSS_DEFAULTS)
        return {
            # The training loss is the huber of chi (trb_utils.huber_profile_loss), so both deltas are in GP-fit error bars,
            # with the profile study's fallbacks for cases run without a tuned config.
            # The chi validation loss reads no delta
            "huber_delta": 1.0,
            "gradient_weight": 0.1,
            "huber_delta_grad": 3.0,
            # Charged per diverged (non-finite) timeslice, sized to swamp ordinary loss differences
            # so a diverged trial cannot win the sweep.
            # One per loss: the training penalty carries no gradient and only marks the logged loss,
            # the validation chi is ~1 per fresh timeslice, so one diverged timeslice in a thousand adds ~10 to it
            "divergence_penalty": 10.0,
            "divergence_penalty_val": 1e4,
            # Anchor terms keeping the sciml submodule predictions close to
            # the measured signals while the whole module trains on the
            # profiles: the power balance's Wtot plus its own p_oh/p_rad
            # submodules. A multiobjective case anchors its own Wtot and P_oh with the same weights.
            # Each residual is relative to the device's typical value (trb_utils.residual_scales), an untuned prior of 0.1 each.
            # Training loss only, and a no-op for cases whose target_vars lack the measured signals.
            # The power balance pseudo-types take the power balance study's own weights (POWER_BALANCE_LOSS_DEFAULTS)
            "anchor_weight_energy_mhd": 0.1,
            "anchor_weight_power_ohm": 0.1,
            "anchor_weight_power_radiated": 0.1,
        }

    def base_optimizer_config(self) -> dict:
        return {
            **super().base_optimizer_config(),
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

    def _transport_dataloader_config(self, case: Case, dataloader_config_base: dict) -> dict:
        """Dataloader config shared by every top-level transport model type.

        A multiobjective case's anchored signals are targets, so the training loss anchors them.
        The other measured powers ride along as extra_vars only to keep the segments of every case identical.
        """
        target_vars = [*TRANSPORT_TARGET_VARS, *(MULTIOBJECTIVE_TARGET_VARS if case.multiobjective else [])]
        return {
            "input_vars": TRANSPORT_INPUT_VARS,
            "target_vars": target_vars,
            "state_vars": TRANSPORT_STATE_VARS,
            "extra_vars": [name for name in MEASURED_POWER_VARS if name not in target_vars],
            **dataloader_config_base,
        }

    def model_train_spec(self, case: Case, dataloader_config_base: dict) -> ModelTrainSpec:
        trb = "transport_study.modules.transport_predictor.trb.TransportPredictorTRB"
        # The pseudo-types are built by the owning studies' builders, so they train exactly as those studies' cases do
        if case.model_type in SCALAR_SUBMODULE_SETTINGS:
            return scalar_submodule_train_spec(
                case.model_type, dataloader_config_base, config.power_balance_data_normalization, case.domain_adaptation
            )
        if case.model_type == "power_balance":
            submodule_configs = {
                "p_oh_predictor": self._make_submodule_config(case, "p_oh"),
                "p_rad_predictor": self._make_submodule_config(case, "p_rad"),
            }
            return structured_power_balance_train_spec(
                config.power_balance_model_type,
                dataloader_config_base,
                config.power_balance_data_normalization,
                case.domain_adaptation,
                config.power_balance_freeze_submodules,
                submodule_configs,
            )
        if case.model_type == "profile":
            return profile_train_spec(
                config.profile_model_type,
                dataloader_config_base,
                config.data_normalization,
                case.domain_adaptation,
                freeze_shapes=PROFILE_SUBMODULE_FREEZE_SHAPES,
                geometry_builder="circular",
            )
        if case.model_type == "sciml":
            return ModelTrainSpec(
                train_run_builder=trb,
                dataloader_config={
                    **self._transport_dataloader_config(case, dataloader_config_base),
                    # The measured stored energy and powers are extra targets
                    # so the training loss can anchor the submodule
                    # predictions to them (anchor_weight_* in the loss config)
                    "target_vars": [*TRANSPORT_TARGET_VARS, "energy_mhd_MJ", *MEASURED_POWER_VARS],
                    "extra_vars": [],
                },
                model_init_config={
                    "model_type": case.model_type,
                    "data_normalization": config.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    "freeze_submodules": case.freeze_submodules,
                    "prng_seed": NN_INIT_DEFAULTS["prng_seed"],
                    "submodules": {
                        "power_balance": self._make_submodule_config(case, "power_balance"),
                        "profile_predictor": self._make_submodule_config(case, "profile"),
                    },
                },
            )
        if case.model_type == "transformer":
            return ModelTrainSpec(
                train_run_builder=trb,
                dataloader_config=self._transport_dataloader_config(case, dataloader_config_base),
                model_init_config={
                    "model_type": case.model_type,
                    "data_normalization": config.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    "d_model": 16,  # Token embedding width
                    "num_heads": 2,
                    "history_len": 20,  # Attention window over past profiles
                    **NN_INIT_DEFAULTS,
                    "multiobjective": case.multiobjective,
                },
            )
        if case.model_type.startswith("torax-"):
            transport_model = case.model_type.removeprefix("torax-")
            return ModelTrainSpec(
                train_run_builder=trb,
                dataloader_config=self._transport_dataloader_config(case, dataloader_config_base),
                model_init_config={
                    "model_type": case.model_type,
                    "data_normalization": config.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    **NN_INIT_DEFAULTS,
                    "torax_config": make_transport_torax_config(transport_model, case.torax_state),
                    "torax_state": case.torax_state,
                    "geometry_builder": case.geometry_builder,
                    "delta_exponent": 2.0,
                    # One TORAX solver step per dataset timestep
                    "sim_dt": UNIFORM_TIMEBASE_DT,
                },
            )
        raise ValueError(f"Unknown model type: {case.model_type}")

    ############
    # ANALYSIS #
    ############

    def run_analysis(self, enable_parallelism: bool) -> None:
        # The figures are the stage grid of the collected metrics, the headline being the chi it was selected on
        run_summary_analysis(self, enable_parallelism, LAYOUT, COMPARISON_FAMILIES, TABLE_SPEC, figures_from_metrics=True)


if __name__ == "__main__":
    fire.Fire(
        {
            "run_study": TransportStudy.run_study,
            "reset_study": TransportStudy.reset_study,
        }
    )
