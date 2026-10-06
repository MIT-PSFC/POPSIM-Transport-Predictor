from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar

if TYPE_CHECKING:
    from popsim.ml import TrainConfig

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
from transport_study.modules.transport_predictor.module import SUBMODULE_MODEL_TYPES
from transport_study.modules.transport_predictor.train_configs import (
    VALID_TORAX_STATES,
    make_transport_torax_config,
)
from transport_study.orchestration.case_analysis import run_summary_analysis
from transport_study.orchestration.organize_data import PROFILE_TARGET_VARS
from transport_study.orchestration.study import (
    CaseGridConfig,
    ModelTrainSpec,
    Study,
)
from transport_study.orchestration.target_shots import BASE_TARGET_SHOT_ORDER
from transport_study.power_balance_transfer.power_balance_study import (
    POWER_BALANCE_INPUT_VARS,
    SCALAR_SUBMODULE_SETTINGS,
)
from transport_study.profile_transfer.profile_study import PROFILE_INPUT_VARS
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
# the profiles, their error bars (the chi validation loss divides by them),
# the freshness flag masking both losses to timeslices with a fresh profile measurement,
# and the device label for per-device weighting
TRANSPORT_TARGET_VARS = [
    *TRANSPORT_PROFILE_TARGETS,
    "n_e_1e20_error",
    "t_e_keV_error",
    "fresh_profile",
    "ds_source_idx",
]

# Everything the env may need to seed a state at the segment start: the
# measured profiles (transformer buffer / torax initial condition), the stored
# energy (sciml power balance state), and every scalar input plus the device
# index (the torax sim-state variant builds a full TORAX initial state, which
# needs the t0 inputs, see TransportPredictorEnv.create_state)
TRANSPORT_STATE_VARS = ["energy_mhd_MJ", *TRANSPORT_PROFILE_TARGETS, *TRANSPORT_INPUT_VARS, "ds_source_idx"]

# Model types that appear on the study's case grid
TOP_LEVEL_MODEL_TYPES = ("transformer", "sciml", *TORAX_MODEL_TYPES)
# The submodule cases that train with power_balance_data_normalization instead of the study-wide data_normalization
POWER_BALANCE_SUBMODULE_TYPES = ("power_balance", "p_oh", "p_rad")

# Power balance variants allowed as the sciml stored-energy submodule (the
# structured ones, so profile-loss gradients flow into physical parameters)
VALID_POWER_BALANCE_MODEL_TYPES = ("sciml-taue-nn", "sciml-taue-scalinglaw")
# Profile predictor variants allowed as the sciml profile submodule
VALID_PROFILE_MODEL_TYPES = (*MODEL_TYPES_WITH_SHAPES, "mlp")


class TransportStudy(Study):
    SWEEP_CONFIG_DIR = Path(PACKAGE_ROOT) / "transport_transfer" / "sweep_configs"
    STUDY_TYPE = "transport_transfer"
    DATA_VISUALIZATION = DataVisualization
    # The result files carry the power balance error variables, scored by its metrics module
    ANALYSIS_METRICS_MODULE = "transport_study.power_balance_transfer.study_metrics"
    ANALYSIS_REPORTS_MODULE = "transport_study.transport_transfer.case_reports"
    CHI_VALIDATION_LOSS = True
    TUNED_DATALOADER_KEYS = ("segment_length_train", "segment_overlap_train", "batch_size")
    # huber_delta_grad only matters for the profile submodule cases, whose sweep tunes it
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
        data_normalization: str = "physics-coral"
        # Hyperparameter tuning case configuration
        # (hyperparam_domain_adaptation and hyperparam_num_target_shots live on CaseGridConfig)
        hyperparam_freeze_submodules: bool = False
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
            "model_types": (*TOP_LEVEL_MODEL_TYPES, *SUBMODULE_MODEL_TYPES),
            "geometry_builders": VALID_GEOMETRY_BUILDERS,
            "torax_state_options": VALID_TORAX_STATES,
            "data_normalization": FEATURE_NORMALIZATIONS,
            "power_balance_data_normalization": INPUT_NORMALIZATIONS,
            "power_balance_model_type": VALID_POWER_BALANCE_MODEL_TYPES,
            "profile_model_type": VALID_PROFILE_MODEL_TYPES,
        }

    @dataclass
    class Case(Study.Case):
        """
        model_type: The type of transport predictor model to use.
        - transformer: recurrent causal attention over a rolling buffer of past profiles
        - sciml: a time-dependent power balance evolves the stored energy, a
          time-independent profile predictor maps the state-implied beta_tor_norm to profiles
        - torax-constant / torax-gyrobohm / torax-qlknn: one-step
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
        - miller: shaped Miller geometry driven by triangularity_upper/triangularity_lower

        torax_state: How the torax-* model types carry state between steps,
        only meaningful for torax-* (every other model type is pinned to "rebuild").
        - rebuild: carry only ne/te and rebuild a TORAX initial state each step (T_i := T_e, psi from Ip)
        - carry: also carry T_i and psi, so the ion channel and the current evolve over the rollout

        num_target_shots: The number of shots included in the training data from the target dataset, never any of the held-out test shots.

        target_shot_order: The order the target training shots are added in as num_target_shots grows, see orchestration/target_shots.py
        - ascending: the base extrapolation, lowest hazard first, suppressed from the case name
        - descending: the highest-hazard non-test shots first, the ones closest to the test regime
        - spanning: the shots whose timeslice footprints span the power balance input and output space
        """

        freeze_submodules: bool
        geometry_builder: str
        torax_state: str

        VALID_MODEL_TYPES = (*TOP_LEVEL_MODEL_TYPES, *SUBMODULE_MODEL_TYPES)
        # Unfrozen submodules and the geometry_builder and torax_state defaults are suppressed from the case name,
        # so the default grid keeps clean case names and new axes never rename pre-existing cases
        STR_TOKEN_FIELDS = (
            ("freeze_", "freeze_submodules", False),
            ("geom_", "geometry_builder", "circular"),
            ("tstate_", "torax_state", "rebuild"),
        )
        # geometry_builder and torax_state are deliberately NOT hyperparam fields (like model_type):
        # miller and carry cases get their own hyperparameter sweep and tuned config
        # instead of inheriting the circular and rebuild ones.
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
            target_shot_order: str = BASE_TARGET_SHOT_ORDER,
        ):
            self.freeze_submodules = freeze_submodules
            self.geometry_builder = geometry_builder
            self.torax_state = torax_state
            self.init_common(model_type, training_data, domain_adaptation, num_target_shots, target_shot_order)

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
            if not self.model_type.startswith("torax-"):
                if self.geometry_builder != "circular":
                    raise ValueError(f"geometry_builder only applies to torax model types, not {self.model_type}")
                if self.torax_state != "rebuild":
                    raise ValueError(f"torax_state only applies to torax model types, not {self.model_type}")
            if self.model_type in SUBMODULE_MODEL_TYPES and self.freeze_submodules != config.hyperparam_freeze_submodules:
                raise ValueError(
                    f"freeze_submodules should be a dummy value ({config.hyperparam_freeze_submodules}) for submodule {self.model_type}"
                )

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
            return pinned

        def model_type_prereqs(self) -> list[Study.Case]:
            # sciml restores a pre-trained power balance and profile predictor.
            # The power balance itself restores pre-trained p_oh/p_rad.
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

    #############
    # EXECUTION #
    #############

    def base_dataloader_config(self, case: Case) -> dict:
        return {
            **self.target_split_config(case),
            # Hyperparameters
            "segment_length_train": 100,
            "segment_overlap_train": 50,
            # Below the power balance study's 4096:
            # every torax sample runs a 100-step differentiated TORAX rollout (jax.checkpoint remat, see TransportPredictorToraxBase).
            # Measured on an A100-80GB with gyrobohm, VRAM is not the limit (~5 MB/sample, 8192 fits in 38 GB).
            # Epoch time stops falling past 512, so a larger batch only gives up optimizer steps.
            "batch_size": 512,
            # Part of validation, should be left alone during hyperparameter tuning
            "segment_length_val": None,
            "segment_overlap_val": 0,
        }

    def base_loss_config(self) -> dict:
        return {
            # The training loss runs on peak-normalized profiles (target scaled to max 1),
            # so the delta reads as a fractional error. Fallback for cases run without a tuned config.
            # The chi validation loss reads no delta
            "huber_delta": 0.1,
            # Only read by the profile submodule cases (ProfilePredictorTRB),
            # matching the profile study's loss configuration
            "gradient_weight": 0.1,
            "huber_delta_grad": 1.0,
            # Charged per diverged (non-finite) timeslice, sized to swamp ordinary loss differences
            # so a diverged trial cannot win the sweep.
            # One per loss scale: converged peak-normalized training losses are ~5e-3,
            # while chi averaged over the mostly stale timeslices is ~0.1-1.
            # At these values one diverged timeslice in a thousand adds ~1e-2 to training and ~10 to validation
            "divergence_penalty": 10.0,
            "divergence_penalty_val": 1e4,
            # Anchor terms keeping the sciml submodule predictions close to
            # the measured signals while the whole module trains on the
            # profiles: the power balance's Wtot plus its own p_oh/p_rad
            # submodules. Training loss only, and a no-op for model types
            # whose target_vars lack the measured signals (only the sciml
            # case carries them). The power balance submodule prereq case
            # also reads anchor_weight_power_ohm/_power_radiated through PowerBalanceTRB,
            # sized like the power balance study's to its Wtot term
            "anchor_weight_energy_mhd": 0.1,
            "anchor_weight_power_ohm": 2e-3,
            "anchor_weight_power_radiated": 2e-3,
        }

    def base_optimizer_config(self) -> dict:
        return {
            **super().base_optimizer_config(),
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

    def model_train_spec(self, case: Case, dataloader_config_base: dict) -> ModelTrainSpec:
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
                    "target_vars": ["energy_mhd_MJ", "power_ohm_MW", "power_radiated_MW", "ds_source_idx"],
                    "state_vars": ["energy_mhd_MJ"],
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
                    "prng_seed": 42,
                    "submodules": {
                        "p_oh_predictor": self._make_submodule_config(case, "p_oh"),
                        "p_rad_predictor": self._make_submodule_config(case, "p_rad"),
                    },
                },
            )
        elif case.model_type == "profile":
            # The profile submodule of sciml, trained exactly like the profile
            # study's time-independent cases (profile-prepped data has beta_tor_norm)
            return ModelTrainSpec(
                train_run_builder="transport_study.modules.profile_predictor.trb.ProfilePredictorTRB",
                dataloader_config={
                    "input_vars": PROFILE_INPUT_VARS,
                    # Profiles plus their gradient and error-bar companions, the chi validation loss divides by the error bars
                    "target_vars": [*PROFILE_TARGET_VARS, "ds_source_idx"],
                    "extra_vars": ["t_e_shape", "n_e_shape"],
                    **dataloader_config_base,
                    # Timeslice samples are cheap, match the profile study
                    "batch_size": 2048,
                },
                model_init_config={
                    "model_type": config.profile_model_type,
                    "data_normalization": config.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    # Profile study hyperparam default.
                    # The shapes are an initial-guess basis, not physics to fine-tune here.
                    "freeze_shapes": True,
                    "te_shape_var": "t_e_shape",
                    "ne_shape_var": "n_e_shape",
                    "n_shapes": 3,
                    "nn_depth": 2,
                    "nn_width": 16,
                    "in_size": 10,  # Dimensionless nn_inputs derived from the raw input_vars, includes log(nu_star)
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
                        "energy_mhd_MJ",
                        "power_ohm_MW",
                        "power_radiated_MW",
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
                    "torax_config": make_transport_torax_config(transport_model, case.torax_state),
                    "torax_state": case.torax_state,
                    "geometry_builder": case.geometry_builder,
                    "delta_exponent": 2.0,
                    # One TORAX solver step per dataset timestep
                    "sim_dt": UNIFORM_TIMEBASE_DT,
                    "prng_seed": 42,
                },
            )
        else:
            raise ValueError(f"Unknown model type: {case.model_type}")

    def tuned_model_init_updates(self, case: Case, tuned_config: TrainConfig) -> dict:
        # sciml has no NN of its own (its submodules carry their own tuned configs)
        updates = {}
        if case.model_type != "sciml":
            updates["nn_depth"] = tuned_config.model_init_config["nn_depth"]
            updates["nn_width"] = tuned_config.model_init_config["nn_width"]
        if case.model_type == "transformer":
            updates["d_model"] = tuned_config.model_init_config["d_model"]
            updates["num_heads"] = tuned_config.model_init_config["num_heads"]
            updates["history_len"] = tuned_config.model_init_config["history_len"]
        if case.model_type == "profile" and config.profile_model_type in MODEL_TYPES_WITH_SHAPES:
            updates["n_shapes"] = tuned_config.model_init_config["n_shapes"]
        return updates

    ############
    # ANALYSIS #
    ############

    def run_analysis(self, enable_parallelism: bool) -> None:
        run_summary_analysis(self, enable_parallelism, LAYOUT, COMPARISON_FAMILIES, TABLE_SPEC)


run_study = TransportStudy.run_study


if __name__ == "__main__":
    fire.Fire(
        {
            "run_study": run_study,
        }
    )
