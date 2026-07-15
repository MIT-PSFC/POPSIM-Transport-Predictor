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

**Study/Case pattern** (`orchestration/study.py`): A `Study` holds many `Case` objects. Each case is a unique combination of case-grid axes (e.g. `model_type x training_data x normalization x domain_adaptation x freeze x num_target_shots`). Cases support prerequisites (chained dependencies, e.g. transfer cases depend on their pretrained checkpoint case; sciml/scaling_law depend on p_oh/p_rad submodule cases). The generic `Study` base provides orchestration (SLURM loop, W&B sweeps, checkpointing, config lock), plus shared helpers: `finalize_cases`, `_hyperparam_training_data`, `_make_mixing_device_weights`, `_set_transfer_checkpoint` / `_scale_transfer_lr`, `_summarize_case_errors`. Subclasses set `SWEEP_CONFIG_DIR` and `STUDY_TYPE`, and define `Config` (subclass of `CaseGridConfig`), `Case`, `make_cases`, `make_train_config`, `collect_results`. `ProfileStudy` (`profile_transfer/profile_study.py`) is the reference implementation; `PowerBalanceStudy` (`power_balance_transfer/power_balance_study.py`) mirrors it.

**DataWorkflow** (`datasets/workflow.py`): Abstract base for device-specific data acquisition. Each device (cmod/, d3d/, tcv/) implements this interface and outputs xarray Datasets in a standardized schema.

**Input normalization as POPSIM modules** (`modules/normalization.py`): the power-balance models take the same 7 physical inputs (`Ip_MA, B0, R0, a_minor, kappa, ne20_line_avg, P_aux_MW`) plus `ds_source_idx`. Each model owns an `InputNormalizer` (`raw` identity / `physics` dimensionless / `z_score` per-device / `coral` per-device covariance alignment), fitted from TRAINING data only at model_init and stored as frozen buffers (they checkpoint with the model but are never trainable - trainable selectors pick `module.nn` leaves explicitly). `organize_data.normalize_domain` implements the same math on whole datasets for data visualization only.

**Module pattern**: Predictors inherit from POPSIM's `TimeIndepModule`/`TimeDepModule`. Profile predictor (time-indep) predicts Te/ne on a uniform 51-point `rho` grid over `[0, 1]` (not psi_n); model families: `shape_init_pca` / `shape_init_kmeans` (B-spline shape bases weighted by an NN, shapes optionally frozen), `unstructured_nn`, `reservoir`, and `torax-constant` / `torax-cgm` / `torax-gyrobohm` / `torax-qlknn` (differentiable TORAX simulation with NN-predicted transport/edge/source parameters). Power balance (time-dep, simple-Euler stepper, `Wtot_MJ` state) predicts stored-energy evolution with four architectures: `scaling_law` (H89/H98 tau_e blend gated by a P_LH scaling; consumes PHYSICAL units), `sciml` (bounded NN tau_e, physics power balance dW/dt = P_aux + P_oh - P_rad - Wtot/tau_e), `unstructured_nn` (MLP directly predicts dW/dt), and `transformer` (recurrent causal attention: a rolling buffer of embedded past inputs is carried in the module State as a discrete `discrete_no_save_field`, the current token attends over it - see `popsim/modules/delay.py` for the discrete-state pattern). `scaling_law` and `sciml` hold `p_oh` / `p_rad` submodule predictors trained as their own prereq cases and restored from checkpoints.

**Domain adaptation / transfer modes** (`domain_adaptation`): `none` (train and test on the same device(s), baseline and hyperparam case), `mixing` (add a small amount of highly-weighted target data during training, weights from `Study._make_mixing_device_weights`), `transfer` (pretrain on source with da=none and num_target_shots=0, restore checkpoint, fine-tune last layers on target with lr scaled by `TRANSFER_LR_FACTOR`). The `exnihilo` training-data option trains from scratch on the target device only. The test set is the held-out highest-performance target shots; domain-adaptation cases reuse that test set for checkpoint selection (no separate val set, slightly optimistic but consistent).

### Module Directory Map

- `transport_study/datasets/` - Per-device data acquisition (cmod/, d3d/, tcv/, mast/), CLI, bundled sample datasets in `sample/`
- `transport_study/modules/normalization.py` - InputNormalizer POPSIM modules (raw/physics/z_score/coral)
- `transport_study/modules/profile_predictor/` - Te/ne profiles over `rho` (shape-init, unstructured NN, reservoir, TORAX-backed model types in `torax_module.py`)
- `transport_study/modules/power_balance/` - Wtot evolution models incl. transformer (module.py), TRBs (trb.py), submodules p_oh/ and p_rad/
- `transport_study/orchestration/` - study.py (Study/Case/CaseGridConfig), organize_data.py, slurm_utils.py, wandb_utils.py
- `transport_study/profile_transfer/` - Profile prediction cross-device study (profile_study.py, sweep_configs/, metrics, reports)
- `transport_study/power_balance_transfer/` - Power balance cross-device study (power_balance_study.py, sweep_configs/, data_visualization.py)
- `transport_study/modules/profile_trajectory/`, `transport_study/trajectory_optimization/` - OUT OF SCOPE, do not extend
- `submodules/popsim-public/` - Core ML framework (TimeDep/TimeIndep modules, trainer, envs, checkpointing)
- `submodules/torax/` - Google TORAX transport model integration
- `submodules/disruption-py/` - C-Mod MDSPlus data access

### Config

- `transport_study/config.py`: `StudyConfig` (Pydantic, frozen, global one-shot proxy `config`), reads `PTPS_*` env vars
- `orchestration/study.py`: `CaseGridConfig(StudyConfig)` shared base for study configs (training_datasets, domain_adaptation_methods, target_test_set_size, from_toml/save); each study nests its own `Config(CaseGridConfig)` with model_types, hyperparam selections, and `is_compatible` (config-lock semantics)
- Per-device `config.toml` files in each dataset subfolder
- `pyproject.toml`: dependencies, build (hatchling), tooling config
- GPU group: `uv sync --group gpu-orcd` for JAX CUDA 13 support

### Data Schema

xarray Datasets with dims `(shot, time_idx, psi_n)`. Power balance signals: `Wtot_MJ`, `Ip_MA`, `B0`, `R0`, `kappa`, `a_minor`, `ne20_line_avg`, `P_oh_MW`, `P_rad_MW`, auxiliary power inputs (`P_NBI_MW` etc., summed to `P_aux_MW` in organize_data). Raw per-device profile data is on `psi_n`; the profile-transfer workflow resamples to a uniform `rho` grid, so profile signals there are `Te_keV_rho`, `ne20_rho`, `Te_shape`, `ne_shape` plus geometry. Saved as NetCDF.
