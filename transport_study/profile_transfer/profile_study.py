from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from popsim.ml import TrainConfig

import fire
import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from loguru import logger
from pydantic import Field, field_validator

from transport_study import PACKAGE_ROOT, TIME_DIM
from transport_study.config import config
from transport_study.modules.profile_predictor.train_configs import (
    PROFILE_PREDICTOR_TORAX_CONFIGS,
)
from transport_study.orchestration.case_analysis import run_case_analysis_parallel
from transport_study.orchestration.organize_data import PROFILE_TARGET_VARS
from transport_study.orchestration.study import (
    HYPERPARAM_TARGET_SHOTS,
    CaseGridConfig,
    ModelTrainSpec,
    Study,
)
from transport_study.profile_transfer.case_reports import (
    generate_case_reports,
    torax_relaxation_report,
)
from transport_study.profile_transfer.data_visualization import DataVisualization
from transport_study.profile_transfer.plotting import (
    domain_adaptation_comparison,
    freeze_shapes_comparison,
    model_comparison,
    training_dataset_comparison,
)
from transport_study.profile_transfer.study_metrics import collect_metrics
from transport_study.profile_transfer.tables import write_comparison_tables

# Model families: the shape-init predictors have freezable shape bases, the rest do not
MODEL_TYPES_WITH_SHAPES = ("shape_init_pca", "shape_init_kmeans")
MODEL_TYPES_WITHOUT_SHAPES = ("unstructured_nn", "reservoir", "torax-constant", "torax-cgm", "torax-gyrobohm", "torax-qlknn")

# The physical inputs every profile-predictor model consumes
PROFILE_INPUT_VARS = [
    "Ip_MA",
    "B0",
    "betan",
    "ne20_line_avg",
    "R0",
    "a_minor",
    "kappa",
    "delta_top",
    "delta_bot",
]


class ProfileStudy(Study):
    SWEEP_CONFIG_DIR = Path(PACKAGE_ROOT) / "profile_transfer" / "sweep_configs"
    STUDY_TYPE = "profile_transfer"
    DATA_VISUALIZATION = DataVisualization
    ANALYSIS_METRICS_MODULE = "transport_study.profile_transfer.study_metrics"
    ANALYSIS_REPORTS_MODULE = "transport_study.profile_transfer.case_reports"
    CASE_AXIS_FIELDS = (
        "model_types",
        "training_datasets",
        "domain_adaptation_methods",
        "freeze_shapes_options",
        "num_target_shots_options",
    )
    TUNED_DATALOADER_KEYS = ("batch_size",)
    TUNED_LOSS_KEYS = ("huber_delta", "huber_delta_grad")

    ##################
    # INITIALIZATION #
    ##################
    class Config(CaseGridConfig):
        # The different cases being compared in this study
        model_types: tuple[str, ...] = Field(default_factory=lambda: ("shape_init_pca", "shape_init_kmeans", "unstructured_nn"))
        freeze_shapes_options: tuple[bool, ...] = Field(default_factory=lambda: (True,))
        num_target_shots_options: tuple[int, ...] = Field(default_factory=lambda: (0, 1, 10, -1))
        # Hyperparameter tuning case configuration
        # (hyperparam_domain_adaptation and hyperparam_num_target_shots live on CaseGridConfig)
        hyperparam_freeze_shapes: bool = True

        COMPAT_HYPERPARAM_FIELDS = (
            "hyperparam_domain_adaptation",
            "hyperparam_freeze_shapes",
            "hyperparam_num_target_shots",
        )

        @field_validator("model_types")
        @classmethod
        def _validate_model_types(cls, v: tuple[str, ...]) -> tuple[str, ...]:
            valid = {*MODEL_TYPES_WITH_SHAPES, *MODEL_TYPES_WITHOUT_SHAPES}
            for mt in v:
                if mt not in valid:
                    raise ValueError(f"Invalid model type: {mt}. Must be one of {sorted(valid)}.")
            return v

    @dataclass
    class Case(Study.Case):
        """
        model_type: The type of profile_predictor model to use
        - shape_init: Use principal component analysis to determine dominant shapes
        - unstructured_nn: a single neural network directly predicts profiles at certain points
        - torax-constant / torax-cgm / torax-gyrobohm / torax-qlknn: TORAX simulation with
          NN-predicted parameters for the constant, critical gradient, Bohm-GyroBohm,
          or QLKNN surrogate transport model

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

        freeze_shapes: bool

        VALID_MODEL_TYPES = (*MODEL_TYPES_WITH_SHAPES, *MODEL_TYPES_WITHOUT_SHAPES)
        STR_TOKEN_FIELDS = (("freeze_", "freeze_shapes"),)
        HYPERPARAM_FIELDS = ("domain_adaptation", "freeze_shapes", "num_target_shots")

        # The dataclass decorator would null an inherited __hash__
        __hash__ = Study.Case.__hash__

        def __init__(
            self,
            model_type: str,
            training_data,
            domain_adaptation: str | None,
            freeze_shapes: bool,
            num_target_shots: int,
        ):
            self.freeze_shapes = freeze_shapes
            self._init_common(model_type, training_data, domain_adaptation, num_target_shots)

    def make_cases(self):
        cases = []
        # Make every case we're interested in for this study
        for (
            model_type,
            training_dataset,
            domain_adaptation,
            freeze_shapes,
            num_target_shots,
        ) in product(
            config.model_types,
            config.training_datasets,
            config.domain_adaptation_methods,
            config.freeze_shapes_options,
            config.num_target_shots_options,
        ):
            if domain_adaptation is None:
                if training_dataset.exnihilo:
                    if num_target_shots == 0:
                        continue  # Can't train from nothing with 0 target shots
                elif num_target_shots != HYPERPARAM_TARGET_SHOTS:
                    continue  # Invalid case, skip
            if model_type in MODEL_TYPES_WITHOUT_SHAPES and not freeze_shapes:
                continue  # No shapes to freeze, just do one of the two

            case = self.Case(
                model_type=model_type,
                training_data=training_dataset,
                domain_adaptation=domain_adaptation,
                freeze_shapes=freeze_shapes,
                num_target_shots=num_target_shots,
            )

            cases.append(case)

        return self.finalize_cases(cases)

    #############
    # EXECUTION #
    #############

    def _base_dataloader_config(self, case: Case) -> dict:
        return {
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
            # 1024 keeps the torax train step under the ~36GB JAX pool on 48GB
            # L40S spillover nodes, batch 2048 needed 40.3GB and OOMed there
            "batch_size": 1024,
        }

    def _base_loss_config(self) -> dict:
        return {
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

    def _base_optimizer_config(self) -> dict:
        return {
            **super()._base_optimizer_config(),
            # Cap on global L2 gradient norm per update, guards against rare
            # gradient spikes from the differentiated TORAX solve NaN-ing a run
            "grad_clip_max_norm": 1.0,
        }

    def _model_train_spec(self, case: Case, dataloader_config_base: dict) -> ModelTrainSpec:
        trb = "transport_study.modules.profile_predictor.trb.ProfilePredictorTRB"
        if case.model_type in MODEL_TYPES_WITH_SHAPES:
            return ModelTrainSpec(
                train_run_builder=trb,
                dataloader_config={
                    "input_vars": PROFILE_INPUT_VARS,
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
                    "in_size": 10,  # Dimensionless nn_inputs derived from the raw input_vars, includes log(nu_star)
                    "softmax_temp": 1,
                    "prng_seed": 42,
                },
            )
        elif case.model_type == "unstructured_nn":
            return ModelTrainSpec(
                train_run_builder=trb,
                dataloader_config={"input_vars": PROFILE_INPUT_VARS, **dataloader_config_base},
                model_init_config={
                    "model_type": case.model_type,
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
                    "domain_adaptation": case.domain_adaptation,
                    "freeze_shapes": case.freeze_shapes,
                    "nn_depth": 2,
                    "nn_width": 16,
                    "in_size": 10,  # Dimensionless nn_inputs derived from the raw input_vars, includes log(nu_star)
                    "torax_config": torax_defaults["torax_config"],
                    "geometry_builder": torax_defaults.get("geometry_builder", "circular"),
                    "delta_exponent": torax_defaults.get("delta_exponent", 2.0),
                    "t_final": torax_defaults.get("t_final"),
                    "fixed_dt": torax_defaults.get("fixed_dt"),
                    "n_solver_steps": torax_defaults.get("n_solver_steps"),
                    "prng_seed": 42,
                },
            )
        else:
            raise ValueError(f"Unknown model type: {case.model_type}")

    def _tuned_model_init_updates(self, case: Case, tuned_config: TrainConfig) -> dict:
        # The reservoir has no MLP depth/width, everything else sweeps them
        updates = {}
        if case.model_type != "reservoir":
            updates["nn_depth"] = tuned_config.model_init_config["nn_depth"]
            updates["nn_width"] = tuned_config.model_init_config["nn_width"]
        if case.model_type in MODEL_TYPES_WITH_SHAPES:
            updates["n_shapes"] = tuned_config.model_init_config["n_shapes"]
            updates["softmax_temp"] = tuned_config.model_init_config["softmax_temp"]
        elif case.model_type == "reservoir":
            updates["reservoir_size"] = tuned_config.model_init_config["reservoir_size"]
            updates["spectral_radius"] = tuned_config.model_init_config["spectral_radius"]
            updates["input_scaling"] = tuned_config.model_init_config["input_scaling"]
            updates["leak_rate"] = tuned_config.model_init_config["leak_rate"]
            updates["n_steps"] = tuned_config.model_init_config["n_steps"]
        elif case.model_type.startswith("torax"):
            updates["t_final"] = tuned_config.model_init_config["t_final"]
            updates["fixed_dt"] = tuned_config.model_init_config["fixed_dt"]
            # .get: tuned configs written before the n_solver_steps sweep key lack it
            updates["n_solver_steps"] = tuned_config.model_init_config.get("n_solver_steps")
        return updates

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

    def collect_results(self):
        """Collect per-shot test errors from all finished cases into one tidy (long-form) dataset.

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

    ############
    # ANALYSIS #
    ############

    def _run_analysis(self, enable_parallelism: bool) -> None:
        # Per-case analysis (stage metrics + case reports) is CPU-bound matplotlib
        # and numpy work: with parallelism it fans out as one SLURM job per case on
        # the analysis partition, and anything unfinished falls back to the serial
        # paths below (collect_metrics / generate_case_reports skip completed cases)
        if enable_parallelism:
            run_case_analysis_parallel(self)

        # Stage-resolved value / gradient / combined metrics for every finished
        # case, cached to collected_metrics.nc alongside collected_results.nc
        metrics_ds = collect_metrics(self)

        logger.opt(colors=True).info("<bold><magenta>TRAINING DATASET COMPARISON</magenta></bold>")
        training_dataset_comparison(metrics_ds, self.figure_dir)

        logger.opt(colors=True).info("<bold><magenta>MODEL COMPARISON</magenta></bold>")
        model_comparison(metrics_ds, self.figure_dir)
        freeze_shapes_comparison(metrics_ds, self.figure_dir)
        # Per-case deep dives: best/worst timeslice PDFs and profile evolution GIFs
        generate_case_reports(self, self.figure_dir)

        logger.opt(colors=True).info("<bold><magenta>DOMAIN ADAPTATION COMPARISON</magenta></bold>")
        domain_adaptation_comparison(metrics_ds, self.figure_dir)

        logger.opt(colors=True).info("<bold><magenta>TORAX-SPECIFIC ANALYSIS</magenta></bold>")
        torax_relaxation_report(self, metrics_ds, self.figure_dir)

        # One markdown table per case axis and combination of the other axes
        logger.opt(colors=True).info("<bold><magenta>COMPARISON TABLES</magenta></bold>")
        results_ds = xr.load_dataset(self.collected_results_path())
        write_comparison_tables(results_ds, metrics_ds, self.figure_dir)


run_study = ProfileStudy.run_study


if __name__ == "__main__":
    fire.Fire(
        {
            "run_study": run_study,
        }
    )
