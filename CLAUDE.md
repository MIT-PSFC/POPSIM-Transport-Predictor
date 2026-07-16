# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tokamak plasma transport prediction study using POPSIM ML framework. Trains models to predict plasma profiles and power balance on three devices (C-Mod, DIII-D, TCV) with transfer learning between devices. Two active studies: `profile_transfer` (complete, the template for study structure) and `power_balance_transfer` (refactored to match it). A `transport_transfer` study is planned but not implemented. `trajectory_optimization` is out of scope (briefly attempted March 2026, inconclusive) - do not extend it.

## Commands

```bash
# Run tests
uv run pytest
uv run pytest transport_study/tests/power_balance_transfer/  # single study
uv run pytest transport_study/tests/modules/test_normalization.py

# Linting / formatting
uv run ruff format transport_study/
uv run ruff check --fix transport_study/
uv run isort transport_study/
uv run mypy transport_study/

# Pre-commit (runs ruff, isort, clears notebook outputs)
pre-commit run --all-files

# Entry points via Fire CLI (config is a TOML path or in-code Config object)
python -m transport_study.profile_transfer.profile_study run_study --config=path/to/study.toml [flags]
python -m transport_study.power_balance_transfer.power_balance_study run_study --config=path/to/study.toml [flags]
python -m transport_study.datasets.cli [args]
```

## Architecture

### Data Flow

```
Device raw data (MDSPlus/files)
    -> datasets/ workflows (per-device)
        - Gaussian process profile fitting (gptools submodule)
        - Standardize signal names, 1 kHz uniform timebase
        - Output: xarray Dataset (dims: shot, time_idx, psi_n)
    -> organize_data.py
        - get_ds(source, study_type) selects per-study signal prep
        - Concatenate shots, train/val/test split (80/20)
    -> POPSIM module training (submodules/popsim/)
        - Input normalization happens INSIDE the modules (see below)
    -> study.py orchestration (cases, W&B sweeps, SLURM)
    -> analysis / plotting
```

### Key Abstractions

**Study/Case pattern** (`orchestration/study.py`): A `Study` holds many `Case` objects. Each case is a unique combination of case-grid axes (e.g. `model_type x training_data x normalization x domain_adaptation x freeze x num_target_shots`). Cases support prerequisites (chained dependencies, e.g. transfer cases depend on their pretrained checkpoint case; sciml/scaling_law depend on p_oh/p_rad submodule cases). `ProfileStudy` (`profile_transfer/profile_study.py`) is the reference implementation; `PowerBalanceStudy` (`power_balance_transfer/power_balance_study.py`) mirrors it.

The `Study` base provides everything generic; subclasses only declare what differs:

- Base `Study.Case` (dataclass): common fields (`model_type`, `training_data`, `domain_adaptation`, `num_target_shots`, `prereqs`), validation, prereq building (`[hyperparam case, _model_type_prereqs(), transfer pretrain case]` deduped in order), `_replace` (rebuild through the real constructor), hyperparam-case logic driven by `HYPERPARAM_FIELDS` (each maps to a `config.hyperparam_<name>` attribute), token-driven `__str__` (`case.{mt}.td_{td}` + `STR_TOKEN_FIELDS` tokens + `targ_{n}` when da or exnihilo + `da_{da}`), and `__hash__` (all fields except prereqs).
- Subclass `Case` is thin: extra fields (`freeze_shapes` for profile; `data_normalization`, `freeze_submodules` for power balance), the three ClassVars (`VALID_MODEL_TYPES`, `STR_TOKEN_FIELDS`, `HYPERPARAM_FIELDS`), and an `__init__` that sets the extra fields then calls `_init_common`. TWO DATACLASS TRAPS: the subclass body must keep its own `__init__` (or `@dataclass` regenerates one, silently dropping validation and prereq building) and must alias `__hash__ = Study.Case.__hash__` (or `@dataclass` with eq=True sets `__hash__ = None`).
- `make_train_config` is a template method: base builds the shared scaffold (dataloader/loss/optimizer bases, mixing `device_weights` injected into the loss config which the val suite aliases, transfer checkpoint wiring via `_set_transfer_checkpoint`, tuned-config merge, then `_scale_transfer_lr` AFTER the merge so swept configs cannot overwrite it). Subclass hooks: `_base_dataloader_config(case)`, `_base_loss_config()`, `_base_optimizer_config()` (profile overrides to add grad clipping), `_model_train_spec(case, dataloader_config_base) -> ModelTrainSpec` (per-model-type train_run_builder + dataloader + model_init dicts), `_tuned_model_init_updates(case, tuned_config)`. Tuned-config merge semantics are deliberate: `TUNED_DATALOADER_KEYS` are indexed strictly (KeyError if missing), `TUNED_LOSS_KEYS` use `.get` fallback because tuned configs on disk may lack them; the optimizer_config is replaced wholesale. Power balance's `_make_submodule_config` recurses through `make_train_config`, so p_oh/p_rad submodule configs get their own tuned merge and transfer wiring.
- `run_study` is a base classmethod (Fire CLI reads its signature, so flags live in one place): config TOML parse via `cls.Config`, `setup_directories`, partition check, `_visualize_data()` (uses the `DATA_VISUALIZATION` ClassVar), orchestration loop + `collect_results` + write `collected_results.nc` (skipped if it exists), then `_run_analysis(enable_parallelism)`. Each study module keeps `run_study = <StudyClass>.run_study` at module level for the `fire.Fire` main and test imports.
- `collect_results` schemas intentionally differ: profile emits a per-shot long-form dataset (dims: record), power balance a per-case scalar summary (dims: case_idx) via the shared `_summarize_case_errors`; both use the base `_case_coords` driven by `_CASE_COORD_NAMES`.
- `_run_analysis` follows the same shape in both studies: with parallelism, per-case analysis (stage metrics + case report) fans out as one SLURM CPU job per case via `orchestration/case_analysis.run_case_analysis_parallel(study)`. The per-study pieces are declared as ClassVars `ANALYSIS_METRICS_MODULE` / `ANALYSIS_REPORTS_MODULE` (dotted paths; the metrics module exports `compute_and_save_case_metrics`, the reports module `generate_case_report` + `analysis_case_done`), which the driver and `slurm_utils.launch_case_analysis_parallel` resolve. Then the driver runs `collect_metrics` (per-case `case_metrics.nc` caches combined into `collected_metrics.nc`, stages from `orchestration/stages.segment_stages`), comparison figures, serial-fallback case reports, and (power balance) per-axis comparison tables. Power balance per-shot errors in result files are raw time integrals (long shots score worse); its study_metrics/case_reports rank shots by TIME-AVERAGED error instead.
- Other subclass ClassVars: `SWEEP_CONFIG_DIR`, `STUDY_TYPE`, `DATA_VISUALIZATION`, `CASE_AXIS_FIELDS` (config axis names logged at init).

CRITICAL: `str(case)` names checkpoint dirs, result files, tuned-config paths, wandb projects, and SLURM job names - byte-level drift orphans existing runs. Literal expectations are pinned in `tests/profile_transfer/test_profile_case_naming.py` and `tests/power_balance_transfer/test_power_balance_case_naming.py` (the profile one also round-trips `restore_predictor.checkpoint_to_profile_case`, which parses case-directory names back into Cases).

**TRB pattern and shared helpers** (`modules/trb_utils.py`, `modules/power_balance/scalar_power_trb.py`): TrainRunBuilders share `trapezoid_dropna` / `integrate_error_over_time` (per-shot NaN-safe time integrals of error), `make_exponential_adamw` (the standard optimizer), and `make_loss_eval_suite` (validation loss mean + a <=100-point subsampled loss vector for wandb; creates a fresh jitted closure per suite so eqx.filter_jit caches never collide across cases). `ScalarPowerTRB` is the shared base for the p_oh/p_rad predictors - `OhmicPowerTRB` / `RadiatedPowerTRB` are subclasses that only set `SIGNAL` and `MODULE_CLS` ClassVars; sweep configs reference them by their original dotted paths.

**DataWorkflow** (`datasets/workflow.py`): Abstract base for device-specific data acquisition. Each device (cmod/, d3d/, tcv/, mast/) implements this interface and outputs xarray Datasets in a standardized schema. The base provides `standardize_dim_names` (rename to POPSIM's `shot`/`time_idx`/`time` conventions), `has_all_nan_signal` (skip-shot check), and a default `device_specific_culling` (cull when either profile is all NaN; mast and tcv override it). Only the raw per-shot files are strictly uniform at 1 kHz - filtering drops interior timeslices, so processed datasets can have mid-shot dt gaps (`organize_data.reindex_to_uniform_timebase` puts them back on the canonical grid with NaNs).

**Input normalization as POPSIM modules** (`modules/normalization.py`): the power-balance models take the same 7 physical inputs (`Ip_MA, B0, R0, a_minor, kappa, ne20_line_avg, P_aux_MW`) plus `ds_source_idx`. Each model owns an `InputNormalizer` (`raw` identity / `physics` dimensionless / `z_score` per-device / `coral` per-device covariance alignment), fitted from TRAINING data only at model_init and stored as frozen buffers (they checkpoint with the model but are never trainable - trainable selectors pick `module.nn` leaves explicitly). `organize_data.normalize_domain` implements the same math on whole datasets for data visualization only. `organize_data.add_missing_profile_companions` fills gradient / error-bar companions for datasets that lack them (TCV produces no GP-fit gradients); it is live TCV support, not legacy handling.

**Data visualization** (`orchestration/data_visualization.py`): `DataVisualizationBase` holds the shared performance-extrapolation and domain-overlap plotting; each study's `data_visualization.py` is a thin subclass setting `STUDY_TYPE` and `VAR_GROUPS` (variable pairs per normalization method). Shared dark-theme figure styling (colors, `style_axis`) lives in `transport_study/plot_style.py`.

**Module pattern**: Predictors inherit from POPSIM's `TimeIndepModule`/`TimeDepModule`. Profile predictor (time-indep) predicts Te/ne on a uniform 51-point `rho` grid over `[0, 1]` (not psi_n); model families: `shape_init_pca` / `shape_init_kmeans` (B-spline shape bases weighted by an NN, shapes optionally frozen), `unstructured_nn`, `reservoir`, and `torax-constant` / `torax-cgm` / `torax-gyrobohm` / `torax-qlknn` (differentiable TORAX simulation with NN-predicted transport/edge/source parameters). Power balance (time-dep, simple-Euler stepper, `Wtot_MJ` state) predicts stored-energy evolution with four architectures: `scaling_law` (H89/H98 tau_e blend gated by a P_LH scaling; consumes PHYSICAL units), `sciml` (bounded NN tau_e, physics power balance dW/dt = P_aux + P_oh - P_rad - Wtot/tau_e), `unstructured_nn` (MLP directly predicts dW/dt), and `transformer` (recurrent causal attention: a rolling buffer of embedded past inputs is carried in the module State as a discrete `discrete_no_save_field`, the current token attends over it - see `popsim/modules/delay.py` for the discrete-state pattern). `scaling_law` and `sciml` hold `p_oh` / `p_rad` submodule predictors trained as their own prereq cases and restored from checkpoints.

**Domain adaptation / transfer modes** (`domain_adaptation`): `none` (train and test on the same device(s), baseline and hyperparam case), `mixing` (add a small amount of highly-weighted target data during training, weights from `Study._make_mixing_device_weights`), `transfer` (pretrain on source with da=none and num_target_shots=0, restore checkpoint, fine-tune last layers on target with lr scaled by `TRANSFER_LR_FACTOR`). The `exnihilo` training-data option trains from scratch on the target device only. The test set is the held-out highest-performance target shots; domain-adaptation cases reuse that test set for checkpoint selection (no separate val set, slightly optimistic but consistent).

### Module Directory Map

- `transport_study/datasets/` - Per-device data acquisition (cmod/, d3d/, tcv/, mast/), CLI, bundled sample datasets in `sample/`
- `transport_study/modules/normalization.py` - InputNormalizer POPSIM modules (raw/physics/z_score/coral)
- `transport_study/modules/trb_utils.py` - Helpers shared by every TrainRunBuilder (error integration, optimizer, val eval suite)
- `transport_study/modules/profile_predictor/` - Te/ne profiles over `rho` (shape-init, unstructured NN, reservoir, TORAX-backed model types in `torax_module.py`)
- `transport_study/modules/power_balance/` - Wtot evolution models incl. transformer (module.py), TRBs (trb.py), scalar_power_trb.py (shared p_oh/p_rad TRB base), submodules p_oh/ and p_rad/
- `transport_study/orchestration/` - study.py (Study/Case/CaseGridConfig/ModelTrainSpec), organize_data.py, data_visualization.py (DataVisualizationBase), stages.py (segment_stages rampup/flattop/rampdown labeling + STAGE_AGG_NAMES), case_analysis.py (generic SLURM fan-out driver for per-case analysis), slurm_utils.py, wandb_utils.py
- `transport_study/plot_style.py` - Shared dark-theme figure styling (colors, style_axis)
- `transport_study/profile_transfer/` - Profile prediction cross-device study (profile_study.py, sweep_configs/, metrics, reports, restore_predictor.py)
- `transport_study/power_balance_transfer/` - Power balance cross-device study (power_balance_study.py, sweep_configs/, data_visualization.py, study_metrics.py (stage-resolved time-averaged errors), case_reports.py (best/worst holdout-shot PDFs), tables.py (per-axis comparison tables))
- `transport_study/modules/profile_trajectory/`, `transport_study/trajectory_optimization/` - OUT OF SCOPE, do not extend
- `submodules/popsim-public/` - Core ML framework (TimeDep/TimeIndep modules, trainer, envs, checkpointing)
- `submodules/torax/` - Google TORAX transport model integration
- `submodules/disruption-py/` - C-Mod MDSPlus data access

### Config

- `transport_study/config.py`: `StudyConfig` (Pydantic, frozen, global one-shot proxy `config`), reads `PTPS_*` env vars
- `orchestration/study.py`: `CaseGridConfig(StudyConfig)` shared base for study configs (training_datasets, domain_adaptation_methods, target_test_set_size, hyperparam_domain_adaptation, hyperparam_num_target_shots, dataset_fractions, from_toml/save) plus the shared `is_compatible` config-lock check, which compares study identity and the hyperparam fields listed in the per-study `COMPAT_HYPERPARAM_FIELDS` ClassVar (case-grid axes like model_types may differ between runs); each study nests its own `Config(CaseGridConfig)` adding model_types and its study-specific hyperparam fields. `HYPERPARAM_TARGET_SHOTS` also lives in study.py (power_balance_study re-exports it)
- Per-device `config.toml` files in each dataset subfolder
- `pyproject.toml`: dependencies, build (hatchling), tooling config
- GPU group: `uv sync --group gpu-orcd` for JAX CUDA 13 support

### Data Schema

xarray Datasets with dims `(shot, time_idx, psi_n)`. Power balance signals: `Wtot_MJ`, `Ip_MA`, `B0`, `R0`, `kappa`, `a_minor`, `ne20_line_avg`, `P_oh_MW`, `P_rad_MW`, auxiliary power inputs (`P_NBI_MW` etc., summed to `P_aux_MW` in organize_data). Raw per-device profile data is on `psi_n`; the profile-transfer workflow resamples to a uniform `rho` grid, so profile signals there are `Te_keV_rho`, `ne20_rho`, `Te_shape`, `ne_shape` plus geometry. Saved as NetCDF.
