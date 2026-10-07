from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar

if TYPE_CHECKING:
    from popsim.ml import TrainConfig

import fire
import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from pydantic import Field

from transport_study import PACKAGE_ROOT, TIME_DIM
from transport_study.config import CaseAxis, config
from transport_study.modules.normalization import FEATURE_NORMALIZATIONS
from transport_study.modules.profile_predictor.module import (
    MODEL_TYPES_WITH_SHAPES,
    NN_INPUT_SOURCE_VARS,
)
from transport_study.modules.profile_predictor.torax_module import (
    TORAX_MODEL_TYPES,
    VALID_GEOMETRY_BUILDERS,
)
from transport_study.modules.profile_predictor.train_configs import (
    PROFILE_PREDICTOR_TORAX_CONFIGS,
)
from transport_study.orchestration.case_analysis import (
    log_section,
    run_case_analysis_parallel,
)
from transport_study.orchestration.case_metrics import collect_metrics
from transport_study.orchestration.case_reports import generate_case_reports
from transport_study.orchestration.comparison_figures import comparison_figures
from transport_study.orchestration.organize_data import PROFILE_TARGET_VARS
from transport_study.orchestration.study import (
    CaseGridConfig,
    ModelTrainSpec,
    Study,
)
from transport_study.orchestration.target_shots import BASE_TARGET_SHOT_ORDER
from transport_study.orchestration.topk_results import BEST_EPOCH_ATTR, CKPT_DIM
from transport_study.profile_transfer.case_reports import torax_relaxation_report
from transport_study.profile_transfer.data_visualization import DataVisualization
from transport_study.profile_transfer.plotting import (
    COMPARISON_FAMILIES,
    LAYOUT,
    freeze_shapes_comparison,
)
from transport_study.profile_transfer.tables import write_comparison_tables

# Model families without freezable shape bases (the shape-init ones are MODEL_TYPES_WITH_SHAPES)
MODEL_TYPES_WITHOUT_SHAPES = ("mlp", "reservoir", *TORAX_MODEL_TYPES)
STUDY_MODEL_TYPES = (*MODEL_TYPES_WITH_SHAPES, *MODEL_TYPES_WITHOUT_SHAPES)

# Input normalization applied to the 10 dimensionless nn_inputs. The physics
# transform is built into the feature set itself, so unlike power balance
# there is no raw / zscore of physical units, only stat stages on top:
# - physics: use the dimensionless parameters as-is
# - physics-coral: per-device CORAL alignment fitted on them
# - physics-zscore: per-device z-score fitted on them (mean/std only, no
#   covariance alignment)

# The physical inputs every profile-predictor model consumes
PROFILE_INPUT_VARS = list(NN_INPUT_SOURCE_VARS)


class ProfileStudy(Study):
    SWEEP_CONFIG_DIR = Path(PACKAGE_ROOT) / "profile_transfer" / "sweep_configs"
    STUDY_TYPE = "profile_transfer"
    DATA_VISUALIZATION = DataVisualization
    ANALYSIS_METRICS_MODULE = "transport_study.profile_transfer.study_metrics"
    ANALYSIS_REPORTS_MODULE = "transport_study.profile_transfer.case_reports"
    CHI_VALIDATION_LOSS = True
    TUNED_DATALOADER_KEYS = ("batch_size",)
    TUNED_LOSS_KEYS = ("huber_delta", "huber_delta_grad")

    ##################
    # INITIALIZATION #
    ##################
    class Config(CaseGridConfig):
        # The different cases being compared in this study
        model_types: Annotated[tuple[str, ...], CaseAxis("model_type")] = Field(
            default_factory=lambda: ("shape-init-pca", "shape-init-kmeans", "mlp")
        )
        freeze_shapes_options: Annotated[tuple[bool, ...], CaseAxis("freeze_shapes")] = Field(default_factory=lambda: (True,))
        # Per-sample geometry builders to compare for torax-* model types, ignored
        # by every other model type (see VALID_GEOMETRY_BUILDERS)
        geometry_builders: Annotated[tuple[str, ...], CaseAxis("geometry_builder")] = Field(default_factory=lambda: ("circular",))
        num_target_shots_options: Annotated[tuple[int, ...], CaseAxis("num_target_shots")] = Field(default_factory=lambda: (0, 1, 10))
        # Input normalization case axis over the 10 dimensionless nn_inputs (see FEATURE_NORMALIZATIONS)
        data_normalization_methods: Annotated[tuple[str, ...], CaseAxis("data_normalization")] = Field(default_factory=lambda: ("physics",))
        # Hyperparameter tuning case configuration
        # (hyperparam_domain_adaptation and hyperparam_num_target_shots live on CaseGridConfig)
        hyperparam_data_normalization: str = "physics"
        hyperparam_freeze_shapes: bool = True

        FIELD_CHOICES: ClassVar[dict[str, tuple]] = {
            "model_types": STUDY_MODEL_TYPES,
            "cpu_model_types": STUDY_MODEL_TYPES,
            "geometry_builders": VALID_GEOMETRY_BUILDERS,
            "data_normalization_methods": FEATURE_NORMALIZATIONS,
            "hyperparam_data_normalization": FEATURE_NORMALIZATIONS,
        }

    @dataclass
    class Case(Study.Case):
        """
        model_type: The type of profile_predictor model to use
        - shape-init-pca / shape-init-kmeans: B-spline shape bases (PCA or k-means initialized) weighted by an NN
        - mlp: a single neural network directly predicts profiles at certain points
        - torax-constant / torax-gyrobohm / torax-qlknn: TORAX simulation with
          NN-predicted parameters for the constant, Bohm-GyroBohm,
          or QLKNN surrogate transport model

        training_data: The dataset(s) used for training
        - cmod: C-Mod only
        - tcv: TCV only
        - cmod_tcv: C-Mod + TCV
        - exnihilo: No historic training data

        data_normalization: The stat stage applied to the 10 dimensionless nn_inputs,
        implemented as a frozen POPSIM module fitted from training data only
        (transport_study/modules/normalization.py, see FEATURE_NORMALIZATIONS).
        - physics: the dimensionless parameters as-is
        - physics-coral: per-device CORAL alignment fitted on them
        - physics-zscore: per-device z-score fitted on them

        domain_adaptation: The method for domain adaptation between source and target devices.
        - none: No domain adaptation, train and test on the same device(s). This is used for hyperparameter tuning and as a baseline for comparison, answering the question "what is the best possible performance we could expect if we had a bunch of data?"
        - weighted: Add a small amount of highly-weighted target data during training
        - addition: Add target shots to the training set as normal samples, no weighting
        - transfer: Train on source data, freeze all but the last layer of every network, and fine-tune on a small amount of target data (ProfilePredictorTRB.get_trainable_getter)
        - transfer_pretrain: The pretrain half of a transfer case, never a case-grid axis value (see Study.Case.transfer_pretrain_case). Trains on historic data only with checkpoint selection on the target test set. Stat normalizations (physics-coral / physics-zscore) fit the stat stage on historic + the transfer case's target shots, stateless ones share one twin at 0 target shots

        freeze_shapes:
        - some profile predictors first use PCA to identify dominant shapes. these shapes may be frozen or modified during module training

        geometry_builder: Per-sample TORAX geometry construction, only meaningful for torax-* model
        types (every other model type is pinned to "circular").
        - circular: large-aspect-ratio analytic geometry (delta = 0 everywhere)
        - miller: shaped Miller geometry driven by triangularity_upper/triangularity_lower

        num_target_shots: The number of shots included in the training data from the target dataset, never any of the held-out test shots.

        target_shot_order: The order the target training shots are added in as num_target_shots grows, see orchestration/target_shots.py
        - ascending: the base extrapolation, lowest hazard first, suppressed from the case name
        - descending: the highest-hazard non-test shots first, the ones closest to the test regime
        - spanning: the shots whose timeslice footprints span the power balance input and output space
        """

        data_normalization: str
        freeze_shapes: bool
        geometry_builder: str

        VALID_MODEL_TYPES = STUDY_MODEL_TYPES
        # Only frozen shapes show in the name, unfrozen is the default
        STR_TOKEN_FIELDS = (
            ("norm_", "data_normalization"),
            ("freeze_", "freeze_shapes", False),
            ("geom_", "geometry_builder"),
        )
        # geometry_builder is deliberately NOT a hyperparam field (like model_type):
        # miller cases get their own hyperparameter sweep and tuned config,
        # keyed by their own geom_miller case string, instead of inheriting circular's.
        # data_normalization IS one (like power balance):
        # every method inherits the tuned config from the hyperparam_data_normalization sweep.
        HYPERPARAM_FIELDS = ("data_normalization", "domain_adaptation", "freeze_shapes", "num_target_shots")

        # The dataclass decorator would null an inherited __hash__
        __hash__ = Study.Case.__hash__

        def normalization_method(self) -> str | None:
            return self.data_normalization

        @classmethod
        def pin_inapplicable_axes(cls, fields: dict) -> dict:
            pinned = super().pin_inapplicable_axes(fields)
            # Only the shape-init families have shapes to freeze
            if fields["model_type"] in MODEL_TYPES_WITHOUT_SHAPES:
                pinned["freeze_shapes"] = config.hyperparam_freeze_shapes
            # The geometry only enters the TORAX families
            if fields["model_type"] not in TORAX_MODEL_TYPES:
                pinned["geometry_builder"] = "circular"
            return pinned

        def validate(self):
            super().validate()
            if self.data_normalization not in FEATURE_NORMALIZATIONS:
                raise ValueError(f"Unknown data normalization method: {self.data_normalization}")

        def __init__(
            self,
            model_type: str,
            training_data,
            domain_adaptation: str | None,
            freeze_shapes: bool,
            num_target_shots: int,
            geometry_builder: str = "circular",
            data_normalization: str = "physics",
            target_shot_order: str = BASE_TARGET_SHOT_ORDER,
        ):
            self.data_normalization = data_normalization
            self.freeze_shapes = freeze_shapes
            self.geometry_builder = geometry_builder
            self.init_common(model_type, training_data, domain_adaptation, num_target_shots, target_shot_order)

    #############
    # EXECUTION #
    #############

    def base_dataloader_config(self, case: Case) -> dict:
        return {
            # Profiles plus their gradient and error-bar companions, the chi validation loss divides by the error bars
            "target_vars": [*PROFILE_TARGET_VARS, "ds_source_idx"],
            **self.target_split_config(case),
            # Hyperparameters
            # 2048 measured 34.7 GB on the worst torax case
            # (qlknn, nn 64x4, 8 solver steps),
            # Throughput saturates here, 4096 gains nothing (1.8 ms/sample at both).
            "batch_size": 2048,
        }

    def base_loss_config(self) -> dict:
        return {
            # The training loss runs on peak-normalized profiles (target scaled to max 1),
            # so both deltas read as fractional errors.
            # Both are swept (see sweep_configs/*.yaml), these values are the fallback without a tuned config.
            # The chi validation loss reads no delta, so the sweep metric cannot be gamed by shrinking them
            "huber_delta": 0.1,
            # Profile gradient mismatch counts too, stability predictions depend on dTe/drho and dne/drho.
            # Weighs the gradient term in training and in the chi validation loss alike
            "gradient_weight": 0.1,
            # Normalized gradients are ~10x the normalized value scale, so they get their own delta
            "huber_delta_grad": 1.0,
        }

    def base_optimizer_config(self) -> dict:
        return {
            **super().base_optimizer_config(),
            # Cap on global L2 gradient norm per update, guards against rare
            # gradient spikes from the differentiated TORAX solve NaN-ing a run
            "grad_clip_max_norm": 1.0,
        }

    def model_train_spec(self, case: Case, dataloader_config_base: dict) -> ModelTrainSpec:
        trb = "transport_study.modules.profile_predictor.trb.ProfilePredictorTRB"
        if case.model_type in MODEL_TYPES_WITH_SHAPES:
            return ModelTrainSpec(
                train_run_builder=trb,
                dataloader_config={
                    "input_vars": PROFILE_INPUT_VARS,
                    "extra_vars": ["t_e_shape", "n_e_shape"],
                    **dataloader_config_base,
                },
                model_init_config={
                    "model_type": case.model_type,
                    "data_normalization": case.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    "freeze_shapes": case.freeze_shapes,
                    "te_shape_var": "t_e_shape",
                    "ne_shape_var": "n_e_shape",
                    "n_shapes": 3,
                    "nn_depth": 2,
                    "nn_width": 16,
                    "in_size": 10,  # Dimensionless nn_inputs derived from the raw input_vars, includes log(nu_star)
                    "prng_seed": 42,
                },
            )
        elif case.model_type == "mlp":
            return ModelTrainSpec(
                train_run_builder=trb,
                dataloader_config={"input_vars": PROFILE_INPUT_VARS, **dataloader_config_base},
                model_init_config={
                    "model_type": case.model_type,
                    "data_normalization": case.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    "nn_depth": 2,
                    "nn_width": 16,
                    "in_size": 10,  # Dimensionless nn_inputs derived from the raw input_vars, includes log(nu_star)
                    "prng_seed": 42,
                },
            )
        elif case.model_type == "reservoir":
            return ModelTrainSpec(
                train_run_builder=trb,
                dataloader_config={"input_vars": PROFILE_INPUT_VARS, **dataloader_config_base},
                model_init_config={
                    "model_type": case.model_type,
                    "data_normalization": case.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    "reservoir_size": 128,  # Fixed random reservoir state dimension
                    "spectral_radius": 0.9,  # Contraction factor of the recurrent weights
                    "input_scaling": 0.5,  # Scale of the random input weights and bias
                    "leak_rate": 1.0,  # Leaky integration rate of the state update
                    "n_steps": 20,  # Reservoir iterations before readout
                    "in_size": 10,  # Dimensionless nn_inputs derived from the raw input_vars, includes log(nu_star)
                    "prng_seed": 42,
                },
            )
        elif case.model_type.startswith("torax-"):
            transport_model = case.model_type.removeprefix("torax-")
            torax_defaults = PROFILE_PREDICTOR_TORAX_CONFIGS[transport_model]["model_init_config"]
            return ModelTrainSpec(
                train_run_builder=trb,
                dataloader_config={"input_vars": PROFILE_INPUT_VARS, **dataloader_config_base},
                model_init_config={
                    "model_type": case.model_type,
                    "data_normalization": case.data_normalization,
                    "domain_adaptation": case.domain_adaptation,
                    "nn_depth": 2,
                    "nn_width": 16,
                    "in_size": 10,  # Dimensionless nn_inputs derived from the raw input_vars, includes log(nu_star)
                    "torax_config": torax_defaults["torax_config"],
                    "geometry_builder": case.geometry_builder,
                    "delta_exponent": torax_defaults["delta_exponent"],
                    "n_solver_steps": torax_defaults["n_solver_steps"],
                    "prng_seed": 42,
                },
            )
        else:
            raise ValueError(f"Unknown model type: {case.model_type}")

    def tuned_model_init_updates(self, case: Case, tuned_config: TrainConfig) -> dict:
        # The reservoir has no MLP depth/width, everything else sweeps them
        updates = {}
        if case.model_type != "reservoir":
            updates["nn_depth"] = tuned_config.model_init_config["nn_depth"]
            updates["nn_width"] = tuned_config.model_init_config["nn_width"]
        if case.model_type in MODEL_TYPES_WITH_SHAPES:
            updates["n_shapes"] = tuned_config.model_init_config["n_shapes"]
        elif case.model_type == "reservoir":
            updates["reservoir_size"] = tuned_config.model_init_config["reservoir_size"]
            updates["spectral_radius"] = tuned_config.model_init_config["spectral_radius"]
            updates["input_scaling"] = tuned_config.model_init_config["input_scaling"]
            updates["leak_rate"] = tuned_config.model_init_config["leak_rate"]
            updates["n_steps"] = tuned_config.model_init_config["n_steps"]
        elif case.model_type.startswith("torax"):
            updates["n_solver_steps"] = tuned_config.model_init_config["n_solver_steps"]
        return updates

    ##############
    # COLLECTION #
    ##############

    # Coords describing which case a record belongs to (broadcast over every shot of that case).
    def collect_results(self):
        """Collect per-shot test errors from all finished cases into one tidy (long-form) dataset.

        Each row is one (case, shot) pair so individual shots where a model struggles can be
        inspected directly (e.g. sort by ``err_abs_ts_mean`` within a ``model_type`` group).
        Rank shots by the ``_ts_mean`` variables, not the ``_shot`` integrals: the integrals
        scale with shot duration, so sorting on them mostly sorts by shot length.

        Dims: record (flat index over all case x shot pairs)
        Coords (along record):
        - case_idx, model_type, training_data, data_normalization,
          domain_adaptation, freeze_shapes, geometry_builder,
          num_target_shots (identify the case)
        - shot (device shot id), ds_source (which dataset the shot came from)
        Data variables (along record), top-K means over the retained checkpoints:
        - err_abs_shot / err_rel_shot: time-integrated combined (ne+Te) error for the shot
          (duration-weighted, longer shots score larger at equal instantaneous error)
        - ne_err_abs_shot / te_err_abs_shot / ne_err_rel_shot / te_err_rel_shot: per-channel
          time-integrated errors (see which channel drives a bad shot)
        - err_abs_ts_max / err_rel_ts_max: worst single timeslice in the shot
        - err_abs_ts_mean / err_rel_ts_mean: mean over the shot's timeslices
        - n_valid_ts: number of non-NaN timeslices contributing to the shot, at the best checkpoint
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

            # Reduce per-timeslice errors to per-shot summaries (worst and mean timeslice) of each checkpoint,
            # then take their top-K mean
            err_abs_ts = ds["error_abs_ts"]
            err_rel_ts = ds["error_rel_ts"]
            per_shot_per_ckpt = {
                "err_abs_shot": ds["error_abs_shot"],
                "err_rel_shot": ds["error_rel_shot"],
                "ne_err_abs_shot": ds["ne_error_abs_shot"],
                "te_err_abs_shot": ds["te_error_abs_shot"],
                "ne_err_rel_shot": ds["ne_error_rel_shot"],
                "te_err_rel_shot": ds["te_error_rel_shot"],
                "err_abs_ts_max": err_abs_ts.max(TIME_DIM, skipna=True),
                "err_rel_ts_max": err_rel_ts.max(TIME_DIM, skipna=True),
                "err_abs_ts_mean": err_abs_ts.mean(TIME_DIM, skipna=True),
                "err_rel_ts_mean": err_rel_ts.mean(TIME_DIM, skipna=True),
            }
            per_shot = {name: per_ckpt.mean(CKPT_DIM).values for name, per_ckpt in per_shot_per_ckpt.items()}
            best_err_abs_ts = err_abs_ts.sel({CKPT_DIM: int(ds.attrs[BEST_EPOCH_ATTR])})
            per_shot["n_valid_ts"] = best_err_abs_ts.notnull().sum(TIME_DIM).values

            shot_ids = ds["shot"].values
            n_shots = len(shot_ids)
            ds_source = ds["ds_source"].values
            case_coords = self.case_coords(case_idx, case)

            for i in range(n_shots):
                record = {name: per_shot[name][i] for name in data_var_names}
                record.update(case_coords)
                record["shot"] = shot_ids[i]
                record["ds_source"] = ds_source[i]
                records.append(record)

        if not records:
            return xr.Dataset()

        coord_names = [*self.case_coord_names(), "shot", "ds_source"]
        data_vars = {name: ("record", np.array([r[name] for r in records])) for name in data_var_names}
        coords = {name: ("record", np.array([r[name] for r in records])) for name in coord_names}
        return xr.Dataset(data_vars=data_vars, coords=coords)

    ############
    # ANALYSIS #
    ############

    def run_analysis(self, enable_parallelism: bool) -> None:
        # The per-case stage metrics and reports fan out over SLURM with parallelism,
        # the serial paths after it skip the completed cases
        if enable_parallelism:
            run_case_analysis_parallel(self)

        # Stage-resolved value / gradient / combined chi metrics of every finished case, cached to collected_metrics.nc
        metrics_ds = collect_metrics(self)
        for family in COMPARISON_FAMILIES:
            log_section(family.title)
            comparison_figures(metrics_ds, LAYOUT, family, self.figure_dir)
        log_section("Shape freezing comparison")
        freeze_shapes_comparison(metrics_ds, self.figure_dir)

        # Best/worst timeslice PDFs and profile evolution GIFs
        log_section("Case reports")
        generate_case_reports(self, self.figure_dir)

        log_section("TORAX relaxation")
        torax_relaxation_report(self, metrics_ds, self.figure_dir)

        log_section("Comparison tables")
        results_ds = xr.load_dataset(self.collected_results_path())
        write_comparison_tables(results_ds, metrics_ds, self.figure_dir)


if __name__ == "__main__":
    fire.Fire(
        {
            "run_study": ProfileStudy.run_study,
            "reset_study": ProfileStudy.reset_study,
        }
    )
