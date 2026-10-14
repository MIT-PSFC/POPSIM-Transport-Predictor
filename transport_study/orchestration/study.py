from __future__ import annotations

import functools
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import time
import tomllib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, replace
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar

import wandb
import xarray as xr
import yaml
from loguru import logger
from popsim.ml import DataLoader, TrainConfig, Trainer
from popsim.ml.launch import (
    build_train_run,
    get_data_train_run_builder_class,
    get_train_run_builder_class,
    launch_agent,
    launch_train,
)
from popsim.ml.train_config import load_dict
from pydantic import Field, field_serializer, field_validator, model_validator

from transport_study import PACKAGE_ROOT, TIME_DIM
from transport_study.config import (
    DEBUG_NUM_SEEDS,
    ROLE_TABLES,
    CaseAxis,
    FieldRole,
    Identity,
    Orchestration,
    StudyConfig,
    config,
    env_dataset_paths,
    load_config,
)
from transport_study.modules.normalization import (
    CORAL_NORMALIZATIONS,
    STAT_NORMALIZATIONS,
)
from transport_study.modules.trb_utils import (
    ADAMW_DEFAULTS,
    chi_sigma_floors,
    residual_scales,
    resolve_case_datasets,
    training_chi_sigma_floors,
)
from transport_study.orchestration.config_lock import (
    CONFIG_LOCK_FILENAME,
    ConfigLock,
    new_stamp,
    read_config_lock,
    write_config_lock,
    write_toml_atomic,
)
from transport_study.orchestration.lineage import StudyArchive
from transport_study.orchestration.organize_data import (
    TrainingData,
    get_loaded_shot_count,
    parse_training_data,
)
from transport_study.orchestration.seed_results import (
    BEST_SEED_ATTR,
    SEED_BEST_EPOCH_ATTR,
    SEED_DIM,
    aggregate_seed_results,
    seed_result,
    seed_statistics,
)
from transport_study.orchestration.slurm_utils import (
    LaunchSlots,
    SubmittedJob,
    TrainingDevices,
    cancel_job,
    count_running_jobs,
    cpu_partitions,
    get_pending_job_pending_s,
    get_running_job_elapsed_s,
    get_running_job_names,
    held_job_names,
    launch_agent_parallel,
    launch_collect_results_job,
    launch_train_parallel,
    partition_exists,
    recent_submitted_jobs,
)
from transport_study.orchestration.target_shots import (
    BASE_TARGET_SHOT_ORDER,
    TARGET_SHOT_ORDERS,
    TargetSplit,
    configured_target_split,
)
from transport_study.orchestration.wandb_utils import (
    WANDB_READ_ERRORS,
    SweepReadError,
    get_best_train_config,
    get_completed_runs,
    get_sweep_id,
    has_live_agent_run,
    is_trained_to_completion,
    run_clean_sweeps,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Target-shot count for hyperparameter tuning cases. When domain adaptation is
# None no target data is used during training, and during domain adaptation no
# hyperparameter tuning is done, so 0 is the canonical value
HYPERPARAM_TARGET_SHOTS = 0

# The domain adaptation methods of the case grid, None trains and tests on the same devices (a TOML spells it "none").
# A case can also be a transfer_pretrain twin, which no grid lists (see Study.Case.transfer_pretrain_case)
DOMAIN_ADAPTATION_METHODS = (None, "weighted", "addition", "transfer")

# The architecture sizes of the model_size case axis (see model_size_updates).
# BASE_MODEL_SIZE trains the tuned config, every other size swaps only its architecture size.
MODEL_SIZES = ("sweep", "min", "max", "half", "quarter", "eighth")
BASE_MODEL_SIZE = "sweep"
# The hyperparameters min / max move to the ends of their sweep grids
SIZE_HYPERPARAMETERS = ("nn_width", "nn_depth", "d_model", "reservoir_size")
# The ones the width sizes divide, depth stays tuned
WIDTH_HYPERPARAMETERS = ("nn_width", "d_model", "reservoir_size")
# What each width size divides the tuned widths by
WIDTH_DIVISORS = {"half": 2, "quarter": 4, "eighth": 8}

# Abort the run once a single seed of a case has been launched this many times without a result file,
# guards against deterministically failing jobs being resubmitted forever
MAX_TRAIN_ATTEMPTS = 3

# A sweep is done once the trial target is met AND this fraction of it trained to completion,
# since the tuned config is picked from those runs only (see wandb_utils.is_trained_to_completion)
# Hyperband with eta 3 lets only a few percent of trials run to completion, so demanding much more than that forces extra trials far past the count target
MIN_FINISHED_FRACTION = 0.05

# Agents kept running per sweep while it is unfinished, even when fewer trials are outstanding
MIN_RUNNING_AGENTS = 6

# How long the orchestration loop sleeps between passes over the unfinished cases
ORCHESTRATION_POLL_INTERVAL_S = 20

# The order a pass places the device kinds in: the GPU-only cases first, then the CPU-only ones, which never take a GPU,
# then the CPU-capable ones, which take the GPUs and CPU slots the others leave
DEVICE_LAUNCH_ORDER = (TrainingDevices.GPU_ONLY, TrainingDevices.CPU_ONLY, TrainingDevices.CPU_CAPABLE)

# The training shuffle of every config before launch_train seeds it, the one sweep trials train with
BASE_SHUFFLE_SEED = 0

# Processes reading the result files of collect_results (map_result_files).
# A cold result read over NFS takes about 3 s and is I/O bound, 16 processes cut it about 14x (2026-10-13 benchmark).
# Threads cannot: netCDF4 is not thread-safe (it segfaults), and h5netcdf serializes its reads under one lock
COLLECT_PROCESSES = 16
# collect_results logs its progress every this many cases
COLLECT_PROGRESS_EVERY = 20
# How often the orchestrator checks on the collection job
COLLECT_POLL_INTERVAL_S = 30

# What the base collect_results reads of a result file, never the predicted and target profiles that make up most of it
SUMMARY_RESULT_VARS = ["error_abs_shot", "error_rel_shot", "error_abs_ts", "error_rel_ts", "time"]

# Measured batches per epoch of each loader config, kept between runs in the study's working dir.
# A fresh config lock starts it over (see Study.batches_per_epoch)
BATCHES_PER_EPOCH_FILENAME = "batches_per_epoch.toml"

# A training job is considered stuck once it has run at least this long with no progress (e.g. OpenBLAS or XLA compile-pool deadlocks)
WATCHDOG_MIN_AGE_S = 80 * 60

# A running job writes its train log every epoch (popsim's Epochs bar) and checkpoints at the validation cadence.
# A job that wrote neither for this long counts as stalled.
WATCHDOG_STALL_S = 60 * 60

# wandb heartbeats every ~30s during a live run, so 10 min of silence on a "running" run is already well past any legitimate gap
AGENT_HEARTBEAT_STALL_S = 10 * 60

# A job parked PENDING on a busy spillover or CPU fallback partition can sit in the queue indefinitely.
# After this long, cancel it so the orchestration loop re-queues it (the next pass's LaunchSlots re-decide placement).
# Jobs pending on a primary partition keep their queue position,
# except the jobs of a case that can also train on CPU pending for a GPU
WATCHDOG_PENDING_S = 30 * 60

# A CPU job runs its epochs several times slower, so one validation interval can outlast the GPU stall threshold.
# The stuck-job watchdog scales both of its thresholds by this for jobs on the CPU partitions.
CPU_WATCHDOG_SCALE = 4

# The time-dep rollout batches pad every shot to a common length by repeating its
# final timeslice with a clamped time value (not NaN). A real timeslice advances the
# shot clock by the 1 kHz sample period, a padded repeat by at most float jitter
# (~1e-13 s observed), so anything below this threshold is padding
PAD_TIME_STEP_S = 1e-6


def real_timeslice_mask(time_2d: xr.DataArray, time_dim: str = TIME_DIM) -> xr.DataArray:
    """True where a timeslice advances its shot's clock, False on the padded tail.

    The first timeslice of each shot is always real. Timeslices whose own time
    is NaN are masked out.
    """
    prev = time_2d.shift({time_dim: 1})
    return time_2d.notnull() & (prev.isnull() | ((time_2d - prev) > PAD_TIME_STEP_S))


def recorded_batches_per_epoch(working_dir: Path) -> dict[str, dict]:
    """The batches per epoch a study recorded in its working dir, by loader config hash (see Study.batches_per_epoch)."""
    path = working_dir / BATCHES_PER_EPOCH_FILENAME
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def model_size_updates(model_size: str, sweep_model_init_parameters: dict, model_init_config: dict) -> dict:
    """The model_init_config entries a model size overrides, only among the size hyperparameters the model type's sweep lists.

    min / max take the smallest / largest value of each SIZE_HYPERPARAMETERS grid.
    half / quarter / eighth divide each tuned WIDTH_HYPERPARAMETERS value by 2 / 4 / 8 (WIDTH_DIVISORS) and keep the depth.
    A divided width keeps at least one unit (4 // 8 is 0 on the transformer nn_width grid),
    and a divided d_model stays a multiple of num_heads and at least num_heads, so every attention head keeps a dimension.
    Everything else keeps its tuned value.
    """
    size_grids = {name: spec["values"] for name, spec in sweep_model_init_parameters.items() if name in SIZE_HYPERPARAMETERS}
    if model_size == BASE_MODEL_SIZE:
        return {}
    if model_size == "min":
        return {name: min(values) for name, values in size_grids.items()}
    if model_size == "max":
        return {name: max(values) for name, values in size_grids.items()}
    if model_size in WIDTH_DIVISORS:
        divisor = WIDTH_DIVISORS[model_size]
        updates = {name: max(model_init_config[name] // divisor, 1) for name in size_grids if name in WIDTH_HYPERPARAMETERS}
        if "d_model" in updates:
            num_heads = model_init_config["num_heads"]
            d_model_head_multiple = updates["d_model"] // num_heads * num_heads
            updates["d_model"] = max(num_heads, d_model_head_multiple)
        return updates
    raise ValueError(f"Unknown model size {model_size!r}, must be one of {MODEL_SIZES}")


def map_result_files[CaseRead](read_case: Callable[[Path], CaseRead], result_paths: list[Path]) -> list[CaseRead]:
    """read_case of every result file, in order, on up to COLLECT_PROCESSES processes.

    read_case must be a module-level function of the result path alone, so a fresh process can import it.
    The processes are spawned, never forked, since forking a process that imported JAX can deadlock.
    A single file is read in this process.
    Logs a progress line every COLLECT_PROGRESS_EVERY cases.
    """
    n_processes = min(COLLECT_PROCESSES, len(result_paths))
    if n_processes <= 1:
        return [read_case(result_path) for result_path in result_paths]
    spawn_context = multiprocessing.get_context("spawn")
    case_reads = []
    with ProcessPoolExecutor(n_processes, mp_context=spawn_context) as pool:
        for n_read, case_read in enumerate(pool.map(read_case, result_paths), start=1):
            case_reads.append(case_read)
            if n_read % COLLECT_PROGRESS_EVERY == 0 or n_read == len(result_paths):
                logger.info(f"Collected {n_read} of {len(result_paths)} case results")
    return case_reads


def summarize_result_file(result_path: Path) -> xr.Dataset:
    """The scalar error statistics of one case result file (Study._summarize_case_errors), a map_result_files reader."""
    with xr.open_dataset(result_path) as ds_result:
        ds = ds_result[SUMMARY_RESULT_VARS].load()
    return Study._summarize_case_errors(ds)


def write_netcdf_atomic(ds: xr.Dataset, path: Path | str) -> None:
    """Write ds to path through a temporary file, so a partial file is never visible at path.

    Result and metric files mark work as done by existing, and other processes poll for them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    ds.to_netcdf(tmp_path)
    tmp_path.replace(path)


def _load_study_config(study_config: CaseGridConfig) -> None:
    """Load study_config as the global config, which a process loads once.

    A process that already loaded an equal config passes, a different one raises.
    """
    if not config.initialized:
        load_config(study_config)
    elif study_config != config.get_instance():
        raise RuntimeError(
            f"Study {study_config.study_name} got a different config than the one this process already loaded (study {config.study_name}), "
            "the global config is loaded once per process"
        )


class CaseGridConfig(StudyConfig):
    """Config base for studies built on a case grid (model_type x training_data x ...).

    Holds the case-grid axes every such study shares.
    Study-specific axes (model_types, freeze options, ...) and hyperparam selections stay on the study's own Config subclass.
    Field roles (identity, locked, case axis, orchestration) are explained in transport_study/config.py.
    """

    # A study of the same type under the same working_dir_base whose cases this study borrows (see orchestration/lineage.py)
    parent_study: Identity[str | None] = None
    # Organization for datasets and wandb projects
    working_dir_base: Orchestration[Path] = Field(
        default_factory=lambda: Path(os.environ.get("PTPS_WORKING_DIR_BASE", str(PACKAGE_ROOT / "popsim_studies" / "working_dir")))
    )
    # Case-grid axes shared by every study of this shape
    training_datasets: Annotated[tuple[TrainingData, ...], CaseAxis("training_data")]
    domain_adaptation_methods: Annotated[tuple[str | None, ...], CaseAxis("domain_adaptation")] = DOMAIN_ADAPTATION_METHODS
    # The orders a case adds its target training shots in, see orchestration/target_shots.py
    target_shot_orders: Annotated[tuple[str, ...], CaseAxis("target_shot_order")] = (BASE_TARGET_SHOT_ORDER,)
    # The architecture sizes each case trains at, see model_size_updates
    model_sizes: Annotated[tuple[str, ...], CaseAxis("model_size")] = (BASE_MODEL_SIZE,)
    # The held-out target test set, never trained on, defined by exactly one of the two:
    # the target_test_set_size highest-hazard target shots,
    # or every target shot at or above the hazard of the lowest named target_test_shots shot
    target_test_set_size: int | None = Field(default=None, ge=1)
    target_test_shots: tuple[int, ...] = ()
    # Hyperparameter tuning case axes shared by every study of this shape
    hyperparam_domain_adaptation: str | None = None
    hyperparam_num_target_shots: int = HYPERPARAM_TARGET_SHOTS
    # Optional dict of dataset fractions to use during domain adaptation, only used if domain_adaptation includes "weighted"
    dataset_fractions: dict[str, float] = Field(default_factory=dict)
    # How many independently seeded training runs every production case trains, each its own job (orchestration/seed_results.py).
    # Every case score is the mean over them, with their std as the run-to-run spread, and the best seed's checkpoint is the one kept
    num_seeds: int = Field(default=5, ge=1)

    # Field name -> its valid values, set per subclass.
    # A tuple field must hold valid values only, a scalar field must be one.
    FIELD_CHOICES: ClassVar[dict[str, tuple]] = {}

    @model_validator(mode="after")
    def _validate_field_choices(self):
        for field_name, valid in self.FIELD_CHOICES.items():
            value = getattr(self, field_name)
            values = value if isinstance(value, tuple) else (value,)
            invalid = [item for item in values if item not in valid]
            if invalid:
                raise ValueError(f"Invalid {field_name}: {invalid}. Must be among {valid}.")
        return self

    @model_validator(mode="after")
    def _apply_debug_seed_limit(self):
        """Cap the seeds of a debug run, see DEBUG_NUM_SEEDS."""
        if self.debug:
            object.__setattr__(self, "num_seeds", min(self.num_seeds, DEBUG_NUM_SEEDS))
        return self

    @model_validator(mode="after")
    def _validate_test_set_source(self):
        if (self.target_test_set_size is None) == (not self.target_test_shots):
            raise ValueError(
                "Set exactly one of target_test_set_size and target_test_shots, the two ways to define the held-out target test set"
            )
        return self

    @field_validator("target_shot_orders")
    @classmethod
    def _validate_target_shot_orders(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        invalid = [order for order in v if order not in TARGET_SHOT_ORDERS]
        if invalid:
            raise ValueError(f"Invalid target shot orders: {invalid}. Must be among {TARGET_SHOT_ORDERS}.")
        return v

    @field_validator("model_sizes")
    @classmethod
    def _validate_model_sizes(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        invalid = [model_size for model_size in v if model_size not in MODEL_SIZES]
        if invalid:
            raise ValueError(f"Invalid model sizes: {invalid}. Must be among {MODEL_SIZES}.")
        return v

    @field_validator("domain_adaptation_methods", "hyperparam_domain_adaptation")
    @classmethod
    def _validate_domain_adaptation(cls, v: tuple[str | None, ...] | str | None) -> tuple[str | None, ...] | str | None:
        """The TOML spelling "none" becomes None, anything outside DOMAIN_ADAPTATION_METHODS raises."""
        methods = v if isinstance(v, tuple) else (v,)
        converted = tuple(None if da == "none" else da for da in methods)
        invalid = [da for da in converted if da not in DOMAIN_ADAPTATION_METHODS]
        if invalid:
            raise ValueError(f"Invalid domain adaptation: {invalid}. Must be among {DOMAIN_ADAPTATION_METHODS}.")
        return converted if isinstance(v, tuple) else converted[0]

    @model_validator(mode="before")
    @classmethod
    def _coerce_training_datasets(cls, data) -> dict:
        if "training_datasets" not in data:
            return data
        # Field defaults aren't applied yet in a before-validator, so when
        # dataset_paths isn't passed explicitly, mirror its default_factory
        dataset_paths = dict(data.get("dataset_paths") or env_dataset_paths())
        target_device = data.get("target_device")
        data["training_datasets"] = tuple(
            parse_training_data(s, dataset_paths, target_device) if isinstance(s, str) else s for s in data["training_datasets"]
        )
        return data

    # The TOML forms of the axes, which the validators above parse back
    @field_serializer("training_datasets")
    def _serialize_training_datasets(self, v):
        return [str(td) for td in v]

    @field_serializer("domain_adaptation_methods")
    def _serialize_domain_adaptation_methods(self, v):
        return ["none" if da is None else da for da in v]


@dataclass
class ModelTrainSpec:
    """The per-model-type pieces of a TrainConfig, returned by Study.model_train_spec."""

    train_run_builder: str
    dataloader_config: dict
    model_init_config: dict


class Study:
    """A class for organizing various components of a study, essentially outlining everything that needs to be done
    to go from raw data to comparison figures.
    - Paths to source data
    - Model checkpoints
    - Results

    Subclasses define a nested Config (CaseGridConfig subclass) and Case
    (Study.Case subclass), the make_train_config hooks
    (base_dataloader_config / base_loss_config / model_train_spec /
    tuned_model_init_updates), collect_results, and run_analysis.
    The case grid comes from the Config's case axes (make_cases).
    """

    # Stuff set by subclasses:
    # study's nested Config class (CaseGridConfig subclass)
    Config: ClassVar[type[CaseGridConfig]]
    # directory holding the per-model-type wandb sweep YAMLs
    SWEEP_CONFIG_DIR: ClassVar[Path]
    # study_type passed to organize_data (selects get_ds branch)
    STUDY_TYPE: ClassVar[str]
    # the study's DataVisualization class
    DATA_VISUALIZATION: ClassVar[type]
    # dotted paths of the per-case analysis modules dispatched over SLURM (see orchestration.case_analysis)
    # The metrics module exports METRIC_NAMES and case_timeslice_metrics (see orchestration.case_metrics)
    # The reports module case_report_done and render_case_report (see orchestration.case_reports)
    ANALYSIS_METRICS_MODULE: ClassVar[str]
    ANALYSIS_REPORTS_MODULE: ClassVar[str]

    # Whether the validation loss is chi, which needs the per-device error-bar floors in the loss config
    CHI_VALIDATION_LOSS: ClassVar[bool] = False

    # Whether the training losses divide power balance residuals by each device's typical value (trb_utils.residual_scales)
    RESIDUAL_SCALE_LOSS: ClassVar[bool] = False

    # The model type predicting each signal a training loss anchors (a TRB's ANCHOR_SIGNALS),
    # whose hyperparam case's tuned huber_delta the anchor term uses (see anchor_huber_deltas)
    ANCHOR_DELTA_MODEL_TYPES: ClassVar[dict[str, str]] = {}

    # Tuned-config dataloader and loss keys, merged by strict indexing (KeyError when a tuned config lacks one).
    # A loss key merges only into a case whose own base loss config holds it (a transport power balance pseudo-type has no huber_delta_grad)
    TUNED_DATALOADER_KEYS: ClassVar[tuple[str, ...]] = ()
    TUNED_LOSS_KEYS: ClassVar[tuple[str, ...]] = ()

    @dataclass
    class Case:
        """A unique combination of case-grid axes to compare in this study.

        Each case holds everything needed to train and evaluate a model and to
        compare it to other cases. Subclasses declare their extra fields plus
        the VALID_MODEL_TYPES / STR_TOKEN_FIELDS / HYPERPARAM_FIELDS ClassVars,
        keep a thin __init__ in the class body that sets the extra fields and
        then calls init_common, and alias __hash__ = Study.Case.__hash__
        (a dataclass body without its own __init__ or __hash__ would have them
        regenerated or nulled by the dataclass decorator).
        """

        model_type: str
        training_data: TrainingData
        # None, weighted, addition, transfer, or transfer_pretrain
        # (transfer_pretrain is never a case-grid axis value, it arises only as the
        # pretrain prereq of a transfer case, see transfer_pretrain_case)
        domain_adaptation: str | None
        num_target_shots: int  # Target shots included in training (HYPERPARAM_TARGET_SHOTS when domain_adaptation is None)
        # The order the target training shots are added in (see orchestration/target_shots.py),
        # always the base order without target shots in training
        target_shot_order: str
        # The architecture size, BASE_MODEL_SIZE trains the tuned config (see model_size_updates)
        model_size: str
        # Cases this one depends on, run first (None when independent)
        prereqs: list[Study.Case] | None

        # Model types accepted by validate
        VALID_MODEL_TYPES: ClassVar[tuple[str, ...]] = ()
        # (prefix, field name) or (prefix, field name, suppress_value) tokens
        # between the td_ and targ_ tokens of str(case). The 3-tuple form
        # omits the token entirely when the field holds suppress_value, so a
        # new axis whose default reproduces old behavior can be added without
        # renaming every existing case (see ProfileStudy.Case.geometry_builder)
        STR_TOKEN_FIELDS: ClassVar[tuple[tuple[str, str] | tuple[str, str, object], ...]] = ()
        # Per-case fields with a config.hyperparam_<name> counterpart
        HYPERPARAM_FIELDS: ClassVar[tuple[str, ...]] = ()

        def init_common(self, model_type, training_data, domain_adaptation, num_target_shots, target_shot_order, model_size):
            """Shared between every subclass __init__: parse, validate, build prereqs."""
            if isinstance(training_data, str):
                training_data = parse_training_data(training_data, dict(config.dataset_paths), config.target_device)
            self.model_type = model_type
            self.training_data = training_data
            self.domain_adaptation = domain_adaptation
            self.num_target_shots = num_target_shots
            self.target_shot_order = target_shot_order
            self.model_size = model_size
            self.validate()
            prereqs = self._build_prereqs()
            self.prereqs = prereqs if prereqs else None

        def validate(self):
            if self.model_type not in self.VALID_MODEL_TYPES:
                raise ValueError(f"Unknown model type: {self.model_type}")
            if self.num_target_shots < 0:
                raise ValueError(f"num_target_shots must be >= 0, got {self.num_target_shots}")
            if self.target_shot_order not in TARGET_SHOT_ORDERS:
                raise ValueError(f"Unknown target shot order: {self.target_shot_order}")
            if self.domain_adaptation not in (*DOMAIN_ADAPTATION_METHODS, "transfer_pretrain"):
                raise ValueError(f"Unknown domain adaptation: {self.domain_adaptation!r}")
            if self.model_size not in MODEL_SIZES:
                raise ValueError(f"Unknown model size: {self.model_size!r}")
            if self.num_target_shots == 0 and self.target_shot_order != BASE_TARGET_SHOT_ORDER:
                raise ValueError(f"target_shot_order {self.target_shot_order} needs target shots in training, this case has none")
            if self.domain_adaptation is None:
                if not self.training_data.exnihilo and self.num_target_shots != HYPERPARAM_TARGET_SHOTS:
                    raise ValueError(
                        "If domain_adaptation is None and training data is not 'exnihilo', num_target_shots must be HYPERPARAM_TARGET_SHOTS since this means we're training and testing on the same dataset"
                    )

        def _build_prereqs(self) -> list[Study.Case]:
            """[hyperparam case, model-type prereqs, transfer pretrain case], deduped in order."""
            prereqs = []
            if not self.is_hyperparam_case():
                prereqs.append(self.replace(**self._hyperparam_field_values()))
            prereqs.extend(self.model_type_prereqs())
            if self.domain_adaptation == "transfer":
                prereqs.append(self.transfer_pretrain_case())
            return list(dict.fromkeys(prereqs))

        def model_type_prereqs(self) -> list[Study.Case]:
            """Extra prereq cases implied by the model type (e.g. submodule predictors)."""
            return []

        def normalization_method(self) -> str | None:
            """The input normalization method this case trains with, None when the study has none."""
            return None

        def transfer_pretrain_case(self) -> Study.Case:
            """The pretrain prereq case this transfer case fine-tunes from.

            Every transfer case pretrains through a dedicated transfer_pretrain
            twin: it trains on historic data only, with checkpoint selection on
            the target test set like every other domain-adaptation case.

            Stat-based normalizations (zscore, coral, physics-coral, physics-zscore) must fit their per-device
            statistics on the combined historic + target data of THIS case
            (a shared source-only pretrain would leave the target device's stats at identity),
            so their twin keeps this case's num_target_shots and fits the
            normalizer on historic + target shots. The stateless
            normalizations (raw, physics) have nothing to fit, so all their
            transfer cases share one twin at HYPERPARAM_TARGET_SHOTS.
            """
            if self.normalization_method() in STAT_NORMALIZATIONS:
                return self.replace(domain_adaptation="transfer_pretrain")
            return self.replace(
                domain_adaptation="transfer_pretrain",
                num_target_shots=HYPERPARAM_TARGET_SHOTS,
                target_shot_order=BASE_TARGET_SHOT_ORDER,
            )

        def replace(self, **changes) -> Study.Case:
            """Rebuild through the real constructor with some fields changed, so validation and prereqs stay consistent."""
            kwargs = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "prereqs"}
            kwargs.update(changes)
            return type(self)(**kwargs)

        @classmethod
        def _hyperparam_field_values(cls) -> dict:
            # Every target shot order shares the base order's hyperparam case and sweep,
            # and every model size the tuned config of the plain case
            values = {
                "training_data": Study._hyperparam_training_data(),
                "target_shot_order": BASE_TARGET_SHOT_ORDER,
                "model_size": BASE_MODEL_SIZE,
            }
            for name in cls.HYPERPARAM_FIELDS:
                values[name] = getattr(config, f"hyperparam_{name}")
            return values

        def is_hyperparam_case(self) -> bool:
            return all(getattr(self, name) == value for name, value in self._hyperparam_field_values().items())

        def get_hyperparam_prereq(self) -> Study.Case:
            if self.is_hyperparam_case():
                return self
            return self.replace(**self._hyperparam_field_values())

        def hyperparam_case_of(self, model_type: str) -> Study.Case:
            """The study's hyperparam case of another model type, whose tuned config every case of that model type trains with."""
            case_fields = {field.name: getattr(self, field.name) for field in fields(self) if field.name != "prereqs"}
            case_fields.update(model_type=model_type, **self._hyperparam_field_values())
            return type(self)(**type(self).pin_inapplicable_axes(case_fields))

        @classmethod
        def pin_inapplicable_axes(cls, fields: dict) -> dict:
            """A case-grid point's Case fields with every axis it ignores pinned to the one value it always takes.

            The pinned duplicates of a grid point collapse into one case in finalize_cases.
            Subclass overrides pin their own axes on top of this.
            """
            # Without target shots in training there is nothing to order
            if fields["num_target_shots"] == 0:
                return fields | {"target_shot_order": BASE_TARGET_SHOT_ORDER}
            return fields

        def is_impossible(self) -> bool:
            """Some cases don't make sense to run. Mark those cases as impossible and raise an error if we try to run them."""
            # Can't do transfer learning or training from nothing with 0 target shots.
            if (self.domain_adaptation == "transfer" or self.training_data.exnihilo) and self.num_target_shots == 0:
                return True

            # exnihilo means training from nothing - no source domain to adapt from
            if self.training_data.exnihilo and self.domain_adaptation is not None:
                return True

            # CORAL aligns every other device to the target, which keeps the identity,
            # so on the target alone it is a duplicate of its stateless method
            if self.training_data.exnihilo and self.normalization_method() in CORAL_NORMALIZATIONS:
                return True

            # A stat normalization fits nothing of a target it never saw:
            # a z-score leaves the target unstandardized beside standardized sources,
            # and CORAL falls back to the identity on every device, a duplicate of its stateless method.
            # That covers the stat pretrain twins too, a stateless twin runs at HYPERPARAM_TARGET_SHOTS (see transfer_pretrain_case).
            # The da None cases train and score on the sources alone, so their z-scores are well-posed and they stay
            if self.domain_adaptation is not None and self.num_target_shots == 0 and self.normalization_method() in STAT_NORMALIZATIONS:
                return True

            # A case built on an impossible prereq can never run either
            return any(prereq.is_impossible() for prereq in self.prereqs or ())

        def __str__(self):
            parts = [f"case.{self.model_type}", f"td_{self.training_data}"]
            for token in self.STR_TOKEN_FIELDS:
                prefix, field_name = token[0], token[1]
                value = getattr(self, field_name)
                if len(token) == 3 and value == token[2]:
                    continue
                parts.append(f"{prefix}{value}")
            # Omitted at the base size, so the names from before the axis existed stay unchanged
            if self.model_size != BASE_MODEL_SIZE:
                parts.append(f"size_{self.model_size}")
            if self.domain_adaptation or self.training_data.exnihilo:
                parts.append(f"targ_{self.num_target_shots}")
                if self.target_shot_order != BASE_TARGET_SHOT_ORDER:
                    parts.append(f"order_{self.target_shot_order}")
            if self.domain_adaptation:
                parts.append(f"da_{self.domain_adaptation}")
            return ".".join(parts)

        def __hash__(self):
            return hash(tuple(v for k, v in self.__dict__.items() if k != "prereqs"))

    ####################
    # PATHING / NAMING #
    ####################
    # The three paths every read of a case's artifacts goes through, routed to the study the case lives in (case_home)

    def trained_model_dir(self, case: Case) -> Path:
        """The case's checkpoint dir, in the working dir of the study it lives in.

        Once its seeds are merged it holds the best seed's best checkpoint, the one every restore of the case reads.
        """
        return self.case_home(case).model_dir / str(case)

    def result_path(self, case: Case) -> Path:
        """The case's result file, in the working dir of the study it lives in, written once its seeds are merged."""
        return self.case_home(case).result_path(str(case))

    def seed_model_dir(self, case: Case, seed: int) -> Path:
        """Where one seed of the case trains, its best checkpoint and the latest one it resumes from."""
        return self.trained_model_dir(case) / f"seed_{seed}"

    def seed_result_path(self, case: Case, seed: int) -> Path:
        """The test results of one seed's best checkpoint, until merge_seed_results folds them into the case result."""
        return self.result_path(case).with_name(f"seed_{seed}.nc")

    def pending_seeds(self, case: Case) -> list[int]:
        """The seeds of the case without a seed result yet."""
        return [seed for seed in range(config.num_seeds) if not self.seed_result_path(case, seed).exists()]

    def _latest_checkpoint_dir_info(self, case: Case, seed: int) -> tuple[int, float] | None:
        """(epoch, mtime) of the newest checkpoint of one seed of the case,
        or None if there is none yet.

        A seed's checkpoint dir keeps its best checkpoint plus the latest one, which training resumes from.
        Orbax names each checkpoint directory after its step (here the epoch)
        and renames it into place atomically once fully written,
        so the newest dir's mtime marks the moment that epoch's checkpoint became visible.
        In-progress saves get a non-numeric tmp suffix and are skipped.
        """
        checkpoint_dir = self.seed_model_dir(case, seed)
        if not checkpoint_dir.exists():
            return None
        epoch_dirs = [p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        if not epoch_dirs:
            return None
        newest = max(epoch_dirs, key=lambda p: int(p.name))
        return int(newest.name), newest.stat().st_mtime

    def train_log_path(self, case: Case, seed: int) -> Path:
        """The SLURM log of one seed's training jobs, every launch attempt appends to it."""
        return self.log_dir / "logs_train" / f"{self.train_job_name(case, seed)}.log"

    def latest_checkpoint_epoch(self, case: Case, seed: int) -> int | None:
        """Epoch of one seed's newest resume checkpoint, or None if empty."""
        info = self._latest_checkpoint_dir_info(case, seed)
        return info[0] if info else None

    def collected_results_path(self) -> Path:
        """The collected results of every finished case, rebuilt by every run_study."""
        return self.result_dir / "collected_results.nc"

    def collected_metrics_path(self) -> Path:
        """The stage-resolved metrics of every finished case, rebuilt by every analysis (case_metrics.collect_metrics)."""
        return self.result_dir / "collected_metrics.nc"

    def study_complete_path(self) -> Path:
        """Written once run_study finished its analysis, the orchestrator wrappers stop resubmitting on it."""
        return self.result_dir / "study_complete"

    def sweep_checkpoint_dir(self, case: Case) -> Path:
        """Where the sweep trials of a hyperparam case keep their checkpoints, one subdirectory per wandb run id."""
        return Path(self.log_dir) / "sweep_checkpoints" / str(case)

    def tuned_config_path(self, case: Case) -> Path:
        """The tuned hyperparameters of the case's hyperparam case, in the working dir of the study that case lives in."""
        return self.trained_model_dir(case.get_hyperparam_prereq()) / "tuned_config.yaml"

    def wandb_project_name(self, case: Case) -> str:
        """Given a case, return the wandb project name to use for that case"""
        return f"{self.name}.{case}"

    # Job names end with the study name so concurrently running studies with
    # identical case grids never collide in squeue name matching
    # (in-flight checks, watchdog cancels)

    def sweep_job_name(self, case: Case) -> str:
        return f"sweep_{case}.{self.name}"

    def agent_job_name(self, case: Case) -> str:
        return f"agent_{case}.{self.name}"

    def train_job_name(self, case: Case, seed: int) -> str:
        return f"train_{case}.seed_{seed}.{self.name}"

    def analysis_job_name(self, case: Case) -> str:
        return f"analysis_{case}.{self.name}"

    def collect_job_name(self) -> str:
        return f"collect_results.{self.name}"

    def check_prereq_satisfied(self, case: Case) -> bool:
        """Check if the prerequisites for this case have been satisfied by looking for the existence of the result path"""
        if case.prereqs is None:
            return True
        for prereq in case.prereqs:
            prereq_result_path = self.result_path(prereq)
            if not prereq_result_path.exists():
                return False
        return True

    @classmethod
    def clean_working_dir(
        cls,
        cfg: CaseGridConfig,
        enable_parallelism: bool = False,
        clean_models: bool = False,
        clean_results: bool = False,
        clean_figures: bool = False,
    ):
        """Delete the chosen parts of a study's working dir, before the Study is built.

        It runs before Study.__init__ checks the config lock,
        so a study whose lock refuses the current config can still be reset.
        Results are scored from the models, so cleaning models cleans results too.
        It also deletes the lock, since the models it corresponded to are gone,
        and the next init writes a fresh one with a new stamp.
        Cleaning results alone keeps the models and the lock, so the next run recomputes every result from the kept checkpoints.
        """
        if clean_models and not clean_results:
            logger.info("Cleaning models cleans results too, results are scored from the models")
            clean_results = True
        logger.info(f"Clean models: {clean_models}")
        logger.info(f"Clean results: {clean_results}")
        logger.info(f"Clean figures: {clean_figures}")
        if (clean_models or clean_results or clean_figures) and enable_parallelism:
            raise ValueError(
                "Cannot clean models, results, or figures when parallelism is enabled, as this would interfere with jobs currently running or queued."
            )
        working_dir = Path(cfg.working_dir_base) / cfg.study_name
        if clean_models:
            shutil.rmtree(working_dir / "models", ignore_errors=True)
        if clean_results:
            shutil.rmtree(working_dir / "results", ignore_errors=True)
        if clean_figures:
            shutil.rmtree(working_dir / "figures", ignore_errors=True)
        if clean_models:
            (working_dir / CONFIG_LOCK_FILENAME).unlink(missing_ok=True)

    def setup_directories(
        self,
        enable_parallelism: bool = False,
        skip_tuning: bool = True,
        skip_visualization: bool = False,
        clean_sweeps: bool = False,
    ):
        logger.info("SETTING UP DIRECTORIES")
        logger.info(f"Enable parallelism: {enable_parallelism}")
        logger.info(f"Skip hyperparameter tuning: {skip_tuning}")
        logger.info(f"Skip visualization: {skip_visualization}")
        logger.info(f"Clean sweeps: {clean_sweeps}")

        if clean_sweeps and enable_parallelism:
            raise ValueError(
                "Cannot clean sweeps when parallelism is enabled, as this would interfere with agents currently running or queued."
            )

        if (not skip_tuning) and (not enable_parallelism):
            logger.critical(
                "Hyperparameter tuning without parallelism enabled is probably gonna take a long time, are you sure you want to do this?"
            )

        if clean_sweeps:
            # Only this study's own sweeps, a borrowed hyperparam case's sweep belongs to its home study
            own_hyperparam_cases = [case for case in self.cases if case.is_hyperparam_case() and not self.is_borrowed(case)]
            if config.wandb_entity is None:
                logger.warning("No wandb entity is set (PTPS_WANDB_ENTITY), so the study's wandb sweeps are left in place")
            else:
                run_clean_sweeps({self.wandb_project_name(case) for case in own_hyperparam_cases})
            for case in own_hyperparam_cases:
                shutil.rmtree(self.sweep_checkpoint_dir(case), ignore_errors=True)

        for directory in [self.model_dir, self.result_dir, self.figure_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    #################
    # CASE BUILDING #
    #################
    @classmethod
    def _hyperparam_training_data(cls) -> TrainingData:
        """All configured non-target source devices, the canonical hyperparam case."""
        target = config.target_device
        sources = sorted(set(config.dataset_paths.keys()) - ({target} if target else set()))
        return TrainingData(sources=sources)

    def make_cases(self, cfg: StudyConfig) -> list[Case]:
        """Every case of the grid spanned by cfg's case axes, plus their prereqs.

        cfg is an argument rather than the global config so another config's grid can be rebuilt.
        The Case construction still reads the global config, which only matters through locked fields.
        """
        case_axes = cfg.case_axes()
        cases = []
        for axis_values in product(*(getattr(cfg, name) for name in case_axes.values())):
            fields = dict(zip(case_axes, axis_values, strict=True))
            if fields["domain_adaptation"] is None:
                if fields["training_data"].exnihilo:
                    if fields["num_target_shots"] == 0:
                        continue  # Can't train from nothing with 0 target shots
                elif fields["num_target_shots"] != HYPERPARAM_TARGET_SHOTS:
                    continue  # Without adaptation a source-trained case sees no target shots
            cases.append(self.Case(**self.Case.pin_inapplicable_axes(fields)))
        return self.finalize_cases(cases)

    def finalize_cases(self, cases: list[Case]) -> list[Case]:
        """Unwrap prereq chains into the flat case list, dedupe, sort, drop impossible cases."""
        unwrapped_cases = []

        def _unwrap_prereqs(case):
            unwrapped_cases.append(case)
            if case.prereqs is not None:
                for prereq in case.prereqs:
                    _unwrap_prereqs(prereq)

        for case in cases:
            _unwrap_prereqs(case)

        unique_cases = sorted(set(unwrapped_cases), key=str)
        possible_cases = [case for case in unique_cases if not case.is_impossible()]  # Remove impossible cases

        return possible_cases

    #############
    # EXECUTION #
    #############
    def make_weighted_device_weights(self, case: Case, study_type: str) -> dict[str, float]:
        """Loss weights per device for weighted domain adaptation.

        Mirrors the actual training-set composition of get_train_test_datasets:
        every loaded shot of each source device in case.training_data (the
        historic train and val splits are both concatenated into the combined
        training set) plus case.num_target_shots target shots, whichever they are.
        Shot counts come from get_loaded_shot_count, so max_ds_size truncation
        and study-type filtering are accounted for.
        study_type is the prep the case's data train run builder loads with,
        which is not this study's for a prereq trained on another study's view (the transport profile pseudo-type).

        Weights are chosen so that each device's effective contribution to the
        loss is F_x = W_x * N_x (N_x the device's shot count in the training
        set, F_x its configured fraction). Typically the target device is
        weighted most heavily. Weights are scaled so the mean per-sample weight
        over the training set is 1, keeping the loss magnitude comparable across cases

        When config.dataset_fractions is not set, the target fraction defaults
        to min(0.5, sqrt(N_target / N_total)) with the remainder split evenly
        among the sources, so few-shot cases still get a strong boost above
        their natural share but the boost fades as target shots accumulate.
        """
        target = config.target_device

        # Shot counts as the training set actually sees them, each loaded once per study since it loads the whole device
        shot_counts = {}
        for source in case.training_data.sources:
            if (source, study_type) not in self._loaded_shot_counts:
                self._loaded_shot_counts[(source, study_type)] = get_loaded_shot_count(source, study_type=study_type)
            shot_counts[source] = self._loaded_shot_counts[(source, study_type)]
        if case.num_target_shots > 0:
            shot_counts[target] = case.num_target_shots
        # num_target_shots == 0: no target samples in training, so the target
        # device gets no weight entry and the sources split the full budget

        total_shots = sum(shot_counts.values())
        if config.dataset_fractions:
            dataset_fractions = {device: config.dataset_fractions[device] for device in shot_counts}
        else:
            num_sources = len(case.training_data.sources)
            if target in shot_counts:
                # A fixed 50 percent target budget overweights the few target
                # shots at intermediate N and drags the fit toward the low
                # end of the target hazard distribution, so scale the
                # boost down as target shots accumulate
                target_fraction = min(0.5, math.sqrt(shot_counts[target] / total_shots))
                dataset_fractions = {
                    device: target_fraction if device == target else (1 - target_fraction) / num_sources for device in shot_counts
                }
                logger.info(
                    f"Dataset fractions not provided in config. Target fraction {target_fraction:.3f} "
                    f"(sqrt-scaled, {shot_counts[target]} of {total_shots} training shots), remainder split evenly among sources."
                )
            else:
                logger.info("Dataset fractions not provided in config. No target shots in training, dividing budget evenly among sources.")
                dataset_fractions = dict.fromkeys(shot_counts, 1 / num_sources)

        # Renormalize over the devices actually present in this case's training set
        # (configured fractions may cover devices this case does not use)
        total_fraction = sum(dataset_fractions.values())
        dataset_weights = {}
        for device, N_x in shot_counts.items():
            F_x = dataset_fractions[device] / total_fraction
            # Scale by total_shots so sum(W_x * N_x) == total_shots, i.e. the
            # mean per-sample weight is exactly 1
            dataset_weights[device] = F_x / N_x * total_shots

        return dataset_weights

    @functools.cached_property
    def chi_sigma_floors_by_device(self) -> dict[str, dict[str, float]]:
        """trb_utils.chi_sigma_floors of every configured device, computed once per study since it loads each device's profiles."""
        return {device: chi_sigma_floors(device) for device in config.dataset_paths}

    def anchor_huber_deltas(self, case: Case, train_run_builder: str, target_vars: list[str]) -> dict[str, float]:
        """The huber delta of every anchor term in the case's training loss, the TRB's ANCHOR_SIGNALS that target_vars carry.

        Each is the tuned huber_delta of the study's hyperparam case of the model type predicting the signal (ANCHOR_DELTA_MODEL_TYPES),
        so an anchor is exactly as robust as that model's own training. That case is a prereq of every anchored case.
        A missing tuned config raises.
        """
        trb_class = get_train_run_builder_class(train_run_builder)
        anchored_signals = [signal for signal in getattr(trb_class, "ANCHOR_SIGNALS", {}) if signal in target_vars]
        deltas = {}
        for signal in anchored_signals:
            source_case = case.hyperparam_case_of(self.ANCHOR_DELTA_MODEL_TYPES[signal])
            tuned_config_path = self.tuned_config_path(source_case)
            if not tuned_config_path.exists():
                raise FileNotFoundError(
                    f"No tuned config of {source_case} at {tuned_config_path}, the {signal} anchor of {case} takes its huber delta from it"
                )
            tuned_config = TrainConfig.load(str(tuned_config_path))
            deltas[signal] = tuned_config.loss_config["huber_delta"]
        return deltas

    def training_loss_statistics(self, dataloader_config: dict) -> dict[str, dict]:
        """The loss config entries the training losses take from a case's own training data, never its held-out or test shots.

        residual_scales (RESIDUAL_SCALE_LOSS studies) from the power balance view,
        train_chi_sigma_floors (CHI_VALIDATION_LOSS studies) from the profile view.
        Both read the training parts trb_utils.resolve_case_datasets selects, the ones every TRB trains on,
        and every view holds the same shots.
        Cached by that selection (training data, domain adaptation and target split), which many cases share.
        """
        target_split = TargetSplit.from_dataloader_config(dataloader_config)
        selection = (str(dataloader_config["training_data"]), dataloader_config["domain_adaptation"], target_split)
        if selection not in self._training_statistics_cache:
            statistics = {}
            if self.RESIDUAL_SCALE_LOSS:
                power_balance_parts, _, _, _ = resolve_case_datasets(dataloader_config, "power_balance_transfer")
                statistics["residual_scales"] = residual_scales(power_balance_parts)
            if self.CHI_VALIDATION_LOSS:
                profile_parts, _, _, _ = resolve_case_datasets(dataloader_config, "profile_transfer")
                statistics["train_chi_sigma_floors"] = training_chi_sigma_floors(profile_parts)
            self._training_statistics_cache[selection] = statistics
        return self._training_statistics_cache[selection]

    def _set_transfer_checkpoint(self, train_config: TrainConfig, transfer_case: Case) -> TrainConfig:
        """Point model_init at the pretrained checkpoint the transfer case fine-tunes from."""
        return train_config.model_copy(
            update={
                "model_init_config": {
                    **train_config.model_init_config,
                    "transfer_checkpoint": str(self.trained_model_dir(transfer_case)),
                }
            }
        )

    def batches_per_epoch(self, train_config: TrainConfig) -> int:
        """The batches in one pass over the training set of this loader config, the case's optimizer steps per epoch.

        Only the transfer learning rate reads it (_scale_transfer_lr), on the historic training set, which loads every source device.
        A measurement builds the training loader, so it is recorded in the working dir (BATCHES_PER_EPOCH_FILENAME) for later runs.
        The root-most study of the lineage that recorded it wins, like case_home,
        so a child trains its own cases with the value its borrowed cases trained with.
        A fresh config lock deletes the record (_open_config_lock), so a reset study measures again.
        """
        data_trb = get_data_train_run_builder_class(train_config.train_run_builder, train_config.dataloader_config)
        config_json = json.dumps([str(data_trb), train_config.dataloader_config], sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()
        if config_hash not in self._batches_per_epoch_cache:
            recorded = [recorded_batches_per_epoch(archive.working_dir) for archive in reversed(self.archive.chain())]
            batches_per_epoch = next((record[config_hash]["batches_per_epoch"] for record in recorded if config_hash in record), None)
            if batches_per_epoch is None:
                _, train_dl, _, _ = data_trb.get_dataloaders(train_config.dataloader_config)
                batches_per_epoch = len(train_dl)
                self._record_batches_per_epoch(config_hash, data_trb.__name__, batches_per_epoch)
                logger.info(f"{batches_per_epoch} batches per epoch on {data_trb.__name__} loaders of this config")
            self._batches_per_epoch_cache[config_hash] = batches_per_epoch
        return self._batches_per_epoch_cache[config_hash]

    def _record_batches_per_epoch(self, config_hash: str, data_train_run_builder_name: str, batches_per_epoch: int) -> None:
        recorded = recorded_batches_per_epoch(self.working_dir)
        recorded[config_hash] = {"data_train_run_builder": data_train_run_builder_name, "batches_per_epoch": batches_per_epoch}
        write_toml_atomic(self.working_dir / BATCHES_PER_EPOCH_FILENAME, recorded)

    def _scale_transfer_lr(self, case: Case, train_config: TrainConfig) -> TrainConfig:
        """lr0 of a transfer finetune, the tuned lr0 / sqrt(the batches per epoch of the historic training set), whatever the finetune's size.

        The historic training set is the one the case's transfer_pretrain twin trains on,
        every source device's training and validation shots, so the scale is one value per loader family and training data.
        That set does not depend on the target split, so it is measured on the 0-shot split, one measurement for every rung.
        The tuned schedule is kept, only its scale changes.

        Applied after the tuned-config merge so the swept optimizer_config cannot overwrite it.
        """
        historic_split = configured_target_split(HYPERPARAM_TARGET_SHOTS, BASE_TARGET_SHOT_ORDER)
        historic_dataloader_config = {
            **train_config.dataloader_config,
            "training_data": case.training_data,
            "domain_adaptation": "transfer_pretrain",
            **historic_split.dataloader_config(),
        }
        historic_train_config = train_config.model_copy(update={"dataloader_config": historic_dataloader_config})
        batches_per_epoch = self.batches_per_epoch(historic_train_config)
        lr0_transfer = train_config.optimizer_config["lr0"] / math.sqrt(batches_per_epoch)
        logger.info(
            f"Transfer lr0 {lr0_transfer:.3g}, the tuned lr0 / sqrt({batches_per_epoch} batches per epoch of the historic training set)"
        )
        optimizer_config = {**train_config.optimizer_config, "lr0": lr0_transfer}
        return train_config.model_copy(update={"optimizer_config": optimizer_config})

    def make_train_config(self, case: Case) -> TrainConfig:
        """Make the TrainConfig for a given case.

        Builds the shared scaffold
        (dataloader/loss/optimizer bases, weighted device weights,
        transfer checkpoint wiring, tuned-config merge, transfer LR scaling)
        around the per-model-type pieces supplied by the model_train_spec hook.
        """
        dataloader_config_case = self.base_dataloader_config(case)
        # launch_train swaps in each seed's own training shuffle
        dataloader_config_base = {**dataloader_config_case, "shuffle_seed": BASE_SHUFFLE_SEED}
        loss_config = self.base_loss_config(case)
        optimizer_config = self.base_optimizer_config()
        spec = self.model_train_spec(case, dataloader_config_base)
        if case.domain_adaptation == "weighted":
            # Loss function reads these from loss_config as "device_weights".
            # val_eval_suite_config references the same dict, so validation
            # loss is weighted consistently with training.
            # "addition" adds the same target shots but as normal samples,
            # so it deliberately gets no device_weights entry
            data_trb = get_data_train_run_builder_class(spec.train_run_builder, spec.dataloader_config)
            loss_config["device_weights"] = self.make_weighted_device_weights(case, data_trb.STUDY_TYPE)
        if self.CHI_VALIDATION_LOSS:
            # Read by the chi validation loss, which val_eval_suite_config shares
            loss_config["chi_sigma_floors"] = self.chi_sigma_floors_by_device
        # The training losses' statistics of the case's own training data
        loss_config.update(self.training_loss_statistics(spec.dataloader_config))
        anchor_huber_deltas = self.anchor_huber_deltas(case, spec.train_run_builder, spec.dataloader_config["target_vars"])
        if anchor_huber_deltas:
            loss_config["anchor_huber_deltas"] = anchor_huber_deltas

        train_config_base = TrainConfig(
            project=self.wandb_project_name(case),
            train_run_builder=spec.train_run_builder,
            max_epochs=config.max_epochs,
            epochs_per_val=config.epochs_per_val,
            patience=config.patience,
            # Where the case's merged best checkpoint is restored from.
            # launch_train points each seed at its own dir, and the wandb agent each sweep trial
            checkpoint_dir=str(self.trained_model_dir(case)),
            dataloader_config=spec.dataloader_config,
            model_init_config=spec.model_init_config,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            # The test suites score the validation chi too, so they share the same loss_config
            val_eval_suite_config={"loss_config": loss_config},
            test_eval_suite_config={"result_path": str(self.result_path(case)), "loss_config": loss_config},
        )

        if case.domain_adaptation == "transfer":
            # Point model_init at the pretrained checkpoint it fine-tunes from
            train_config_base = self._set_transfer_checkpoint(train_config_base, case.transfer_pretrain_case())

        train_config = self._apply_tuned_config(case, train_config_base)
        # Outside _apply_tuned_config, which returns early without a tuned config
        if case.model_size != BASE_MODEL_SIZE:
            train_config = self._apply_model_size(case, train_config)

        # A transfer finetune gets a reduced lr0 (see _scale_transfer_lr), exnihilo keeps the tuned one.
        # Applied after the tuned-config merge so the swept optimizer_config cannot overwrite it
        if case.domain_adaptation == "transfer":
            train_config = self._scale_transfer_lr(case, train_config)

        return train_config

    def _apply_tuned_config(self, case: Case, train_config_base: TrainConfig) -> TrainConfig:
        """Merge swept hyperparameters from the case's tuned config, when one exists.

        Only the TUNED_* keys come from the tuned dataloader/loss configs, so
        case-specific entries (e.g. weighted device_weights) stay intact. The
        optimizer_config is replaced entirely.
        """
        tuned_config_path = self.tuned_config_path(case)
        if not tuned_config_path.exists():
            return train_config_base
        tuned_config = TrainConfig.load(str(tuned_config_path))
        logger.info(f"Found tuned hyperparameter config for case {case}, using hyperparameters from that config")

        tuned_dataloader = {key: tuned_config.dataloader_config[key] for key in self.TUNED_DATALOADER_KEYS}
        tuned_loss = {key: tuned_config.loss_config[key] for key in self.TUNED_LOSS_KEYS if key in train_config_base.loss_config}
        train_config = train_config_base.model_copy(
            update={
                "dataloader_config": {**train_config_base.dataloader_config, **tuned_dataloader},
                "optimizer_config": tuned_config.optimizer_config,
                "loss_config": {**train_config_base.loss_config, **tuned_loss},
            }
        )
        return train_config.model_copy(
            update={
                "model_init_config": {
                    **train_config.model_init_config,
                    **self.tuned_model_init_updates(case, tuned_config),
                }
            }
        )

    def _apply_model_size(self, case: Case, train_config: TrainConfig) -> TrainConfig:
        """Swap the tuned architecture size for the case's model size, every other hyperparameter stays tuned (see model_size_updates)."""
        sweep_model_init_parameters = self.sweep_model_init_parameters(case.model_type)
        size_updates = model_size_updates(case.model_size, sweep_model_init_parameters, train_config.model_init_config)
        logger.info(f"Model size {case.model_size} of case {case}: {size_updates}")
        return train_config.model_copy(update={"model_init_config": {**train_config.model_init_config, **size_updates}})

    @classmethod
    def sweep_config(cls, model_type: str) -> dict:
        """The wandb sweep config of a model type, its yaml in SWEEP_CONFIG_DIR."""
        sweep_config_path = Path(cls.SWEEP_CONFIG_DIR) / f"{model_type}.yaml"
        return load_dict(str(sweep_config_path))

    @classmethod
    def sweep_model_init_parameters(cls, model_type: str) -> dict:
        """The model_init_config parameters a model type's sweep varies, none for a model type without a swept net of its own.

        sciml-taue-scalinglaw and the transport sciml sweep their optimizer alone, their nets are their prereqs'.
        """
        sweep_config = cls.sweep_config(model_type)
        model_init_sweep = sweep_config["parameters"].get("model_init_config", {})
        return model_init_sweep.get("parameters", {})

    def base_optimizer_config(self) -> dict:
        """Fallback optimizer hyperparameters for cases run without a tuned config."""
        return {
            "lr0": 5e-4,
            "transition_steps": 1661,
            "decay_rate": 0.1,
            "lrf_frac": 0.2,
            "weight_decay": 2e-4,
            **ADAMW_DEFAULTS,
        }

    def base_dataloader_config(self, case: Case) -> dict:
        """Dataloader settings shared by every model type of this study, starting from target_split_config."""
        raise NotImplementedError

    def target_split_config(self, case: Case) -> dict:
        """The dataloader-config entries that pick a case's training data and its held-out target test set."""
        target_split = configured_target_split(case.num_target_shots, case.target_shot_order)
        return {
            "training_data": case.training_data,
            "domain_adaptation": case.domain_adaptation,
            **target_split.dataloader_config(),
        }

    def base_loss_config(self, case: Case) -> dict:
        """Fallback loss hyperparameters of a case run without a tuned config."""
        raise NotImplementedError

    def model_train_spec(self, case: Case, dataloader_config_base: dict) -> ModelTrainSpec:
        """Per-model-type train_run_builder, dataloader_config, and model_init_config."""
        raise NotImplementedError

    def tuned_model_init_updates(self, case: Case, tuned_config: TrainConfig) -> dict:
        """model_init_config entries swept only for certain model types."""
        raise NotImplementedError

    def check_data_requirements(self, case: Case) -> bool:
        """Given a case, check if the required data for that case is available."""
        required = set(case.training_data.sources)
        # Every domain adaptation method reads target shots, transfer_pretrain to fit its normalizer
        if case.training_data.exnihilo or case.domain_adaptation is not None:
            required.add(config.target_device)

        missing = [ds for ds in required if ds not in config.dataset_paths]
        if missing:
            logger.warning(f"Case {case} is missing required datasets: {missing}. Skipping this case.")
            return False

        return True

    def case_by_name(self, case_name: str) -> Case:
        """The study case whose str(case) is case_name."""
        for case in self.cases:
            if str(case) == case_name:
                return case
        raise ValueError(f"Study {self.name} has no case {case_name}")

    def get_unfinished_cases(self) -> list[Case]:
        unfinished = []
        for case in self.cases:
            if not self.result_path(case).exists():
                if not self.check_data_requirements(case):
                    logger.debug(f"Case {case} is missing required data, skipping.")
                    continue
                unfinished.append(case)
        return unfinished

    def kill_stuck_jobs(self, cases: list[Case]):
        """Kill training jobs that are running but making no progress.

        A job counts as deadlocked once it has run at least WATCHDOG_MIN_AGE_S
        with neither a checkpoint nor a train log write in the last WATCHDOG_STALL_S.
        The log advances every epoch, so a slow model whose validation interval outlasts the stall window survives.
        Both thresholds are CPU_WATCHDOG_SCALE times longer for a job on a CPU partition.
        Killing it frees the seed to be relaunched as a fresh process by the normal orchestration loop.
        launch_train's attempt counter only resets on checkpoint progress,
        so a job that keeps deadlocking still hits MAX_TRAIN_ATTEMPTS and aborts the study rather than looping forever.
        """
        elapsed = get_running_job_elapsed_s()
        if not elapsed:
            return
        cpu_job_names: set[str] = set()
        if cpu_partitions():
            # A failed query holds the pass, it must not judge a CPU job at the GPU thresholds
            cpu_partition_list = ",".join(cpu_partitions())
            cpu_elapsed = get_running_job_elapsed_s(partition=cpu_partition_list)
            if cpu_elapsed is None:
                return
            cpu_job_names = set(cpu_elapsed)
        now = time.time()
        for case, seed in product(cases, range(config.num_seeds)):
            job_name = self.train_job_name(case, seed)
            job_elapsed = elapsed.get(job_name)
            watchdog_scale = CPU_WATCHDOG_SCALE if job_name in cpu_job_names else 1
            min_age_s = WATCHDOG_MIN_AGE_S * watchdog_scale
            stall_s = WATCHDOG_STALL_S * watchdog_scale
            if job_elapsed is None or job_elapsed < min_age_s:
                continue
            progress_mtimes = []
            checkpoint_info = self._latest_checkpoint_dir_info(case, seed)
            if checkpoint_info is not None:
                progress_mtimes.append(checkpoint_info[1])
            train_log_path = self.train_log_path(case, seed)
            if train_log_path.exists():
                progress_mtimes.append(train_log_path.stat().st_mtime)
            last_progress = max(progress_mtimes, default=-math.inf)
            if now - last_progress < stall_s:
                continue
            logger.warning(
                f"Job {job_name} has run {job_elapsed}s with no checkpoint or train log progress in "
                f"the last {stall_s}s, likely deadlocked. Killing so it can be resubmitted.\n"
                f"Case:\t{case}, seed {seed}"
            )
            cancel_job(job_name)

    def _kill_stuck_agents(self, cases: list[Case]):
        """Cancel sweep agent jobs whose wandb run has gone stale.

        An agent runs exactly one trial then should exit.
        If the wandb agent wrapper angs after that trial finishes,
        the SLURM job holds a GPU until it hits the hard walltime limit.
        Kill agent jobs old enough to have plausibly finished their trial once
        wandb shows no actively heartbeating run left for the case's project
        """
        elapsed = get_running_job_elapsed_s()
        if not elapsed:
            return
        now = time.monotonic()
        for case in cases:
            if not case.is_hyperparam_case():
                continue
            job_name = self.agent_job_name(case)
            job_elapsed = elapsed.get(job_name)
            if job_elapsed is None or job_elapsed < WATCHDOG_MIN_AGE_S:
                continue
            if now < self._next_agent_check.get(job_name, -math.inf):
                continue
            self._next_agent_check[job_name] = now + config.sweep_poll_interval_s
            try:
                agent_live = has_live_agent_run(self.wandb_project_name(case), AGENT_HEARTBEAT_STALL_S)
            except WANDB_READ_ERRORS as e:
                logger.warning(f"wandb read failed for agent job {job_name}, checking it again at the next poll: {e}")
                continue
            if agent_live:
                continue
            logger.warning(
                f"Agent job {job_name} has run {job_elapsed}s with no actively heartbeating "
                f"wandb run, likely a zombied agent wrapper. Killing so launch_sweep can resubmit.\n"
                f"Case:\t{case}"
            )
            cancel_job(job_name)

    def _kill_long_pending_jobs(self, cases: list[Case]):
        """Cancel jobs stuck PENDING longer than WATCHDOG_PENDING_S where waiting on does not pay off.

        That is a GPU spillover or CPU fallback partition, any case's job can pend there
        indefinitely once other users grab the capacity it was placed on,
        and the primary GPU partition for a case that can also train on CPU.
        Cancelling hands the case back to the normal launch path, where the next pass's LaunchSlots re-decide placement.
        The primary partitions are otherwise exempt:
        jobs there keep their queue position (and accrued age priority)
        instead of cycling to the back every 30 min. A cancelled pending job
        never started training, so its launch attempt is refunded, otherwise a
        busy partition alone could burn MAX_TRAIN_ATTEMPTS and abort the study
        without a single actual training failure. Agent jobs need no refund,
        launch_sweep tops agents back up to the remaining trial count on its own.
        """
        fallback = ",".join(p for p in (*config.spillover_partitions, *config.cpu_fallback_partitions) if p and p != config.partition)
        watched = [(fallback, cases)] if fallback else []
        cpu_capable_cases = [case for case in cases if self.training_devices(case) is TrainingDevices.CPU_CAPABLE]
        if cpu_capable_cases and config.partition:
            watched.append((config.partition, cpu_capable_cases))
        for partition, partition_cases in watched:
            pending = get_pending_job_pending_s(partition=partition)
            if not pending:
                continue
            for case in partition_cases:
                train_job_names = [self.train_job_name(case, seed) for seed in range(config.num_seeds)]
                job_names = list(train_job_names)
                if case.is_hyperparam_case():
                    job_names.append(self.agent_job_name(case))
                for job_name in job_names:
                    pending_s = pending.get(job_name)
                    if pending_s is None or pending_s < WATCHDOG_PENDING_S:
                        continue
                    logger.warning(
                        f"Job {job_name} has been pending {pending_s}s on {partition}, "
                        f"cancelling so the orchestration loop can re-queue it.\nCase:\t{case}"
                    )
                    # Scoped to the watched partitions so a same-named job pending elsewhere can never be caught
                    cancel_job(job_name, partition=partition, state="PENDING")
                    # scancel --state=PENDING is a no-op if the job started since
                    # the squeue snapshot, so at worst this refund is one attempt
                    # too generous and the case gets one extra retry
                    if job_name in train_job_names:
                        attempts = self.train_attempts.get(job_name, 0)
                        self.train_attempts[job_name] = max(attempts - 1, 0)

    def training_devices(self, case: Case) -> TrainingDevices:
        """Where the case's training and sweep agents may run (gpu_only_model_types, cpu_only_model_types), GPU only without a cpu_partition."""
        if not config.cpu_partition or case.model_type in config.gpu_only_model_types:
            return TrainingDevices.GPU_ONLY
        if case.model_type in config.cpu_only_model_types:
            return TrainingDevices.CPU_ONLY
        return TrainingDevices.CPU_CAPABLE

    def run_unfinished_cases(self, skip_tuning: bool, enable_parallelism: bool):
        """Loop until every runnable case has a result file.

        Every case trains config.num_seeds seeds, each its own job and seed result (orchestration/seed_results.py).
        Each pass takes one squeue snapshot of this user's job names,
        merges the seeds of every case whose seeds have all finished (merge_seed_results),
        then only picks up the seeds that are actually able to run:
        prereq results on disk and no job for the seed already in flight.
        Blocked and in-flight cases are counted in a single per-pass summary line instead of being visited
        (and logged about) individually. Prereqs are themselves cases in
        self.cases, so a blocked case becomes runnable once its prereq case finishes.
        Nothing needs to recurse into prereq chains here.

        Every training job the study submits is recorded with its job id (submitted_train_jobs).
        A seed whose job has left the queue stays held until sacct shows that job ended RELAUNCH_GRACE_S ago (slurm_utils.held_job_names).
        Its seed result may already be written but not yet visible across nodes,
        and relaunching in that window submits a duplicate job for a finished seed.
        The hold needs no snapshot to have seen the job, so a seed submitted and finished between two passes is held too.
        A restarted orchestrator recalls the jobs its predecessor had in the queue (slurm_utils.recent_submitted_jobs).
        With parallelism a case still tuning is held until its sweep may be read again (config.sweep_poll_interval_s).

        Borrowed cases (see case_home) are only waited on, their home study trains them.
        The ancestor chain is reloaded every pass, so a reset ancestor stops the loop.
        """
        unfinished = self.get_unfinished_cases()
        if enable_parallelism:
            previous_train_jobs = recent_submitted_jobs("train_", f".{self.name}")
            self.submitted_train_jobs.update(previous_train_jobs)
        while unfinished:
            self.refresh_lineage()
            self._pass_train_configs = {}
            borrowed = [case for case in unfinished if self.is_borrowed(case)]
            local = [case for case in unfinished if not self.is_borrowed(case)]
            if enable_parallelism:
                running_job_names = get_running_job_names()
                if running_job_names is None:
                    logger.warning("Could not query SLURM job state, waiting before trying again...")
                    time.sleep(ORCHESTRATION_POLL_INTERVAL_S)
                    continue
                self.kill_stuck_jobs(local)
                self._kill_stuck_agents(local)
                self._kill_long_pending_jobs(local)
                held_train_job_names = held_job_names(self.submitted_train_jobs, running_job_names)
            else:
                running_job_names = set()
                held_train_job_names = set()

            runnable = [case for case in local if self.check_prereq_satisfied(case)]
            now = time.monotonic()
            n_in_flight, n_in_grace = 0, 0
            # Each ready case with the seeds of it to launch
            sweeping, to_launch = [], []
            for case in runnable:
                pending_seeds = self.pending_seeds(case)
                if not pending_seeds:
                    self.merge_seed_results(case)
                    continue
                if enable_parallelism and now < self._next_sweep_poll.get(str(case), -math.inf):
                    sweeping.append(case)
                    continue
                launch_seeds = []
                for seed in pending_seeds:
                    job_name = self.train_job_name(case, seed)
                    if job_name in running_job_names:
                        n_in_flight += 1
                    elif job_name in held_train_job_names:
                        n_in_grace += 1
                    else:
                        launch_seeds.append(seed)
                if launch_seeds:
                    to_launch.append((case, launch_seeds))
            n_blocked = len(local) - len(runnable)
            n_ready = sum(len(launch_seeds) for _, launch_seeds in to_launch)
            waiting = Counter(self.case_home(case).name for case in borrowed)
            waiting_summary = "".join(f", {n} waiting on {name}" for name, n in waiting.items())
            logger.opt(colors=True).info(
                f"<bold><green>{len(unfinished)} cases remain</green></bold> "
                f"({n_in_flight} seeds in flight, {n_in_grace} awaiting results, {len(sweeping)} cases tuning, "
                f"{n_ready} seeds of {len(to_launch)} cases ready to launch, {n_blocked} cases blocked on prereqs{waiting_summary})"
            )

            if enable_parallelism:
                self.launch_slots = LaunchSlots()
                self._launch_ready_cases(to_launch, skip_tuning)
                # Sleep for a bit before checking again to avoid spamming slurm
                time.sleep(ORCHESTRATION_POLL_INTERVAL_S)
            else:
                for case, launch_seeds in to_launch:
                    for seed in launch_seeds:
                        self.run_case(case, seed, skip_tuning=skip_tuning, enable_parallelism=False)
                # Nothing runs in the background of a serial run,
                # so only a pass that finished no seed waits (on an ancestor, or between sweep passes)
                finished_this_pass = [
                    case
                    for case, launch_seeds in to_launch
                    if self.result_path(case).exists() or any(self.seed_result_path(case, seed).exists() for seed in launch_seeds)
                ]
                if not finished_this_pass:
                    time.sleep(ORCHESTRATION_POLL_INTERVAL_S)
            unfinished = [case for case in unfinished if not self.result_path(case).exists()]

    def _launch_ready_cases(self, to_launch: list[tuple[Case, list[int]]], skip_tuning: bool):
        """Give each ready seed a partition from the pass's launch_slots and launch it, in DEVICE_LAUNCH_ORDER.

        A CPU-capable case only takes the primary GPUs the GPU-only cases of this pass leave beyond cpu_capable_buffer_gpus,
        and the CPU slots the CPU-only cases leave.
        So it still tries the primary after a GPU-only case found it full, since its GPU buffer may be the smaller one.
        A seed finding no partition holds the remaining cases of its kind until the next pass,
        and a CPU-capable one finding none holds the CPU-only cases too, since the CPU partitions are full.
        A case that cannot train yet (still tuning, or its prereqs unmet) holds its other seeds.
        The cases still tuning go last, each reads its wandb project (about 20 s) before any training launch could wait on it.
        """
        launch_order = sorted(to_launch, key=lambda case_seeds: self._launch_rank(case_seeds[0]))
        held_devices: set[TrainingDevices] = set()
        for case, seeds in launch_order:
            devices = self.training_devices(case)
            if devices in held_devices:
                continue
            for seed in seeds:
                partition = self.launch_slots.pick(devices)
                if partition is None:
                    logger.info(f"No open slots for the ready {devices.value} cases, holding them until the next pass")
                    held_devices.add(devices)
                    if devices is TrainingDevices.CPU_CAPABLE:
                        held_devices.add(TrainingDevices.CPU_ONLY)
                    break
                if not self.run_case(case, seed, skip_tuning=skip_tuning, enable_parallelism=True, partition=partition):
                    break

    def _launch_rank(self, case: Case) -> tuple[bool, int]:
        """Where a ready case goes in a pass's launch order, the cases still tuning last, each kind in DEVICE_LAUNCH_ORDER."""
        devices = self.training_devices(case)
        return self._reads_sweep(case), DEVICE_LAUNCH_ORDER.index(devices)

    def _reads_sweep(self, case: Case) -> bool:
        """Whether launching the case reads its wandb sweep first, a hyperparam case without its tuned config yet."""
        return case.is_hyperparam_case() and not self.tuned_config_path(case).exists()

    def run_case(
        self,
        case: Case,
        seed: int,
        skip_tuning: bool,
        enable_parallelism: bool,
        partition: str | None = None,
    ) -> bool:
        """Run one seed of a case of the study, tuning the case's hyperparameters first as needed.

        Returns whether the case is ready to train, so its other seeds may launch too.
        It is not while the case or a prereq is in progress, the orchestration loop tries again later.
        With parallelism the seed's job goes to partition, picked here when None.
        """
        if not self.check_data_requirements(case):
            raise ValueError(f"Case {case} does not have the required data to run. This should have been caught earlier!")
        self._require_local(case)
        # A merged case has no seed results left, its result file is the one to check
        if self.result_path(case).exists():
            logger.warning(f"Case {case} already has results, skipping.")
            return False
        if self.seed_result_path(case, seed).exists():
            logger.warning(f"Seed {seed} of case {case} already has results, skipping.")
            return True
        if not self.check_prereq_satisfied(case):
            logger.debug(f"Case blocked on unmet prereqs, skipping until they finish.\nCase:\t{case}")
            return False
        return self._run_ready_case(case, seed, skip_tuning, enable_parallelism, partition)

    def _run_ready_case(self, case: Case, seed: int, skip_tuning: bool, enable_parallelism: bool, partition: str | None) -> bool:
        """Launch one seed of a case whose prereqs are satisfied, returns whether the case was ready to train."""
        if enable_parallelism and partition is None:
            partition = self.launch_slots.pick(self.training_devices(case))
            if partition is None:
                logger.debug("No resources currently available, waiting before trying again...")
                return False
        if case.is_hyperparam_case() and not self._ensure_hyperparams_ready(case, skip_tuning, enable_parallelism, partition):
            return False
        logger.opt(colors=True).info(f"<bold><cyan>RUNNING CASE, seed {seed}:</cyan></bold>\n{case}")
        self.launch_train(case, seed, enable_parallelism=enable_parallelism, partition=partition)
        return True

    def _ensure_hyperparams_ready(self, case: Case, skip_tuning: bool, enable_parallelism: bool, partition: str | None) -> bool:
        """Ensure tuned config exists. Returns True if ready to proceed to training."""
        self._require_local(case)
        tuned_config_path = self.tuned_config_path(case)
        if tuned_config_path.exists():
            logger.debug(f"Tuned config found at {tuned_config_path}")
            return True
        if skip_tuning:
            logger.info(f"Skipping hyperparameter tuning, writing default config for {case}")
            self._write_tuned_config(case, self.make_train_config(case))
            return True
        # run_unfinished_cases holds the case until then, whether this read succeeds or not
        self._next_sweep_poll[str(case)] = time.monotonic() + config.sweep_poll_interval_s
        try:
            completed_runs = get_completed_runs(self.wandb_project_name(case))
            finished_runs = [r for r in completed_runs if is_trained_to_completion(r)]
            min_finished = math.ceil(MIN_FINISHED_FRACTION * config.hyperparam_sweeps)
            if len(completed_runs) < config.hyperparam_sweeps or len(finished_runs) < min_finished:
                logger.info(
                    f"Hyperparameter sweeps incomplete "
                    f"({len(completed_runs)}/{config.hyperparam_sweeps} runs, "
                    f"{len(finished_runs)}/{min_finished} finished)"
                )
                self.launch_sweep(
                    case,
                    enable_parallelism=enable_parallelism,
                    n_completed_runs=len(completed_runs),
                    n_finished_runs=len(finished_runs),
                    partition=partition,
                )
                return False
            return self._finalize_sweep(case, enable_parallelism, completed_runs)
        except WANDB_READ_ERRORS as e:
            # A wandb outage or an exhausted rate limit holds this sweep, the orchestrator keeps going
            logger.warning(f"wandb read failed, holding the sweep of case {case} until the next poll: {e}")
            return False

    def _finalize_sweep(self, case: Case, enable_parallelism: bool, completed_runs: list) -> bool:
        """Save best config once sweep runs are done. Returns True if ready."""
        logger.info(f"Hyperparameter sweeps completed with {len(completed_runs)}/{config.hyperparam_sweeps} runs")
        if enable_parallelism:
            # Sweep trials run inside agent jobs, so poll the agent job name.
            # Picking the best config while agents still run would ignore
            # their trials (and the last trials are often the best ones)
            running_jobs = count_running_jobs(self.agent_job_name(case))
            if running_jobs is None:
                logger.info("Could not query the agent jobs, holding the sweep finalization until the next pass")
                return False
            if running_jobs > 0:
                logger.info(f"Found {running_jobs} running agent jobs, waiting for them to complete before proceeding")
                return False
        # Read again now that no agent runs, an agent that exited since the first read would be missing from it
        final_runs = get_completed_runs(self.wandb_project_name(case))
        best_train_config = get_best_train_config(final_runs)
        if best_train_config is None:
            raise RuntimeError(
                f"Hyperparameter sweep for case {case} reports {len(completed_runs)} completed runs "
                f"but no best config could be recovered from wandb project {self.wandb_project_name(case)}. "
                "Only runs trained to completion with a finite val/loss.mean are eligible, "
                "check the project for runs stuck crashing, pruning or diverging."
            )
        self._write_tuned_config(case, best_train_config)
        # The case reads its sweep no more, so nothing holds its training launch
        self._next_sweep_poll.pop(str(case), None)
        # Every trial is over, so their resume checkpoints are no longer needed
        shutil.rmtree(self.sweep_checkpoint_dir(case), ignore_errors=True)
        logger.success(f"Saved best hyperparameter config for {case}")
        return True

    def _write_tuned_config(self, case: Case, train_config: TrainConfig):
        """Write a train config to the tuned config path for the given case."""
        self._require_local(case)
        tuned_config_path = self.tuned_config_path(case)
        tuned_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tuned_config_path, "w") as f:
            yaml.dump(train_config.model_dump(), f, indent=4)

    def launch_sweep(
        self,
        case: Case,
        enable_parallelism: bool = False,
        n_completed_runs: int = 0,
        n_finished_runs: int = 0,
        partition: str | None = None,
    ):
        """Launch a wandb hyperparameter sweep for the given case.

        With parallelism, launches at most as many agent jobs as sweep trials
        still outstanding, capped by the open GPU slots (idle GPUs beyond
        buffer_gpus less this user's pending jobs). Outstanding is the larger of the
        two remaining targets (both net of agents already running): trials to
        reach the completed-count target (hyperparam_sweeps - completed) and
        trials to reach the finished-run quota (min_finished - finished). Once
        the count target is met but too many trials have crashed to satisfy the
        finished quota, the second term keeps agents flowing. Without the
        outstanding cap the orchestration loop would submit another batch of
        agents every pass until the queue filled, far past what is needed.

        On any other partition the cap is that partition's open slots in the pass's launch_slots (LaunchSlots.available):
        the remaining per-user GPU allowance of a spillover partition,
        or the CPU jobs that fit on a CPU partition, bounded by the user-wide job ceiling.
        On every partition the sweep keeps at most config.max_running_agents agents running or pending,
        so the bayes search takes in finished trials before it suggests the next ones.
        """
        self._require_local(case)
        train_config = self.make_train_config(case)
        # Remove the test_eval_suite_config since that's for final results only.
        # A trial stopped by the wall-clock budget ends preempted, the sweep requeues it,
        # and the next agent resumes it from its own checkpoint subdirectory (popsim keys it by run id).
        # So every trial trains to max_epochs or patience, whatever device and job length it ran on.
        train_config = train_config.model_copy(
            update={
                "test_eval_suite_config": None,
                "max_epochs": min(config.max_epochs, config.hyperparam_max_epochs),
                "resume": True,
                "checkpoint_dir": str(self.sweep_checkpoint_dir(case)),
                "max_wall_seconds": float(config.train_wall_budget_s),
                # Sweep trials are compared on val loss only, one checkpoint like every production seed
                "checkpoint_max_to_keep": 1,
            }
        )
        wandb_project_name = self.wandb_project_name(case)
        try:
            sweep_id = get_sweep_id(wandb_project_name)
        except SweepReadError as e:
            logger.warning(f"{e}\nHolding the agent launches for case {case} until the next pass")
            return
        # One training run per agent
        kwargs_agent = {"count": 1, "entity": config.wandb_entity}

        if not sweep_id:
            logger.info(f"No existing sweep found for case {case}, creating a new sweep")
            sweep_config = self.sweep_config(case.model_type)
            sweep_id = wandb.sweep(sweep_config, entity=config.wandb_entity, project=wandb_project_name)

        if enable_parallelism:
            if partition is None:
                partition = config.partition
            agent_job_name = self.agent_job_name(case)
            running_agents = count_running_jobs(agent_job_name)
            if running_agents is None:
                logger.info(f"Could not query the agent jobs, holding the agent launches for case {case} until the next pass")
                return
            # Each agent runs exactly one trial (count=1), so outstanding trials
            # bound how many more agents are worth submitting. Take the larger of
            # the count target and the finished-quota target so a sweep whose
            # count target is met but whose trials keep crashing keeps launching
            # until enough runs finish (each in-flight agent may yet finish).
            min_finished = math.ceil(MIN_FINISHED_FRACTION * config.hyperparam_sweeps)
            outstanding_for_count = config.hyperparam_sweeps - n_completed_runs - running_agents
            outstanding_for_finished = min_finished - n_finished_runs - running_agents
            outstanding_trials = max(outstanding_for_count, outstanding_for_finished)
            capacity = self.launch_slots.available(partition, self.training_devices(case))
            # Keep MIN_RUNNING_AGENTS going (up to capacity) to finish out the sweep,
            # one agent at a time may get pruned.
            # A sweep smaller than that (debug) never runs more agents than it has trials
            min_running_agents = min(MIN_RUNNING_AGENTS, config.hyperparam_sweeps)
            agent_room = config.max_running_agents - running_agents
            sweep_jobs = min(capacity, agent_room, max(outstanding_trials, min_running_agents - running_agents))
            if sweep_jobs <= 0:
                logger.debug(f"No agent jobs needed for case {case} ({running_agents} agents already running)")
                return
            logger.info(f"Launching {sweep_jobs} agent job(s) {agent_job_name} on {partition} for case\n{case}")
            for _ in range(sweep_jobs):
                job_id = launch_agent_parallel(
                    train_config,
                    sweep_id,
                    kwargs_agent,
                    agent_job_name,
                    Path(self.log_dir) / "logs_sweep",
                    partition=partition,
                )
                if job_id is None:
                    break
                self.launch_slots.take(partition)
        else:
            logger.info("Launching agent serially")
            launch_agent(train_config, sweep_id, kwargs_agent=kwargs_agent)

    def launch_train(self, case: Case, seed: int, enable_parallelism: bool = False, partition: str | None = None):
        """Launch the training job of one seed of the given case."""
        if case.is_impossible():
            raise ValueError(
                f"Case {case} is not a possible case to run, check the logic in the Case dataclass to see why this is. This should have been caught earlier!"
            )
        self._require_local(case)

        # Only reached when no seed result exists and no job for this seed is
        # in flight or held (see run_unfinished_cases), so every call is a fresh (re)launch.
        # More than MAX_TRAIN_ATTEMPTS launches means the seed fails every time.
        # Exception: a relaunch whose latest checkpoint advanced since the last
        # launch is a resume of a wall-clock-limited job making real progress,
        # so the attempt counter resets rather than counting toward the cap
        train_job_name = self.train_job_name(case, seed)
        attempts = self.train_attempts.get(train_job_name, 0)
        latest_epoch = self.latest_checkpoint_epoch(case, seed)
        prev_epoch = self.train_attempt_epochs.get(train_job_name)
        if latest_epoch is not None and (prev_epoch is None or latest_epoch > prev_epoch):
            attempts = 0
        self.train_attempt_epochs[train_job_name] = latest_epoch
        if attempts >= MAX_TRAIN_ATTEMPTS:
            train_log_path = self.train_log_path(case, seed)
            summary = (
                f"ABORTING STUDY: seed {seed} of a case failed training {attempts} times without producing a result.\n"
                f"Case:      {case}\n"
                f"Job name:  {train_job_name}\n"
                f"Train log: {train_log_path}\n"
                f"(attempts append to the same log, separated by '=== ... job <id> start ===' lines)\n"
                f"Other unfinished cases were not attempted further. Fix the case or remove it, then rerun."
            )
            logger.critical(summary)
            raise RuntimeError(summary)
        self.train_attempts[train_job_name] = attempts + 1

        logger.opt(colors=True).info(f"<bold><red>LAUNCHING TRAINING for seed {seed} of case\n{case}</red></bold>")

        # Every seed of a case launched in one pass shares its train config
        case_train_config = self._pass_train_configs.get(str(case))
        if case_train_config is None:
            case_train_config = self.make_train_config(case)
            self._pass_train_configs[str(case)] = case_train_config
        train_config = self._seeded_train_config(case_train_config, case, seed)
        seed_result_path = self.seed_result_path(case, seed)
        if enable_parallelism:
            logger.info(f"Launching training job {train_job_name} for case\n{case}")
            job_id = launch_train_parallel(
                train_config,
                train_job_name,
                seed_result_path,
                self.train_log_path(case, seed),
                partition=partition,
            )
            if job_id is None:
                # Nothing ran, so a rejected submission does not count toward MAX_TRAIN_ATTEMPTS
                self.train_attempts[train_job_name] = attempts
            else:
                self.submitted_train_jobs[train_job_name] = SubmittedJob(job_id, time.time())
                self.launch_slots.take(partition)
        else:
            logger.info("Launching training serially")
            trainer, _, _, _, result_dict = launch_train(train_config)
            if result_dict is None:
                logger.info("Training stopped at the wall-clock budget before finishing, relaunch to resume from the latest checkpoint.")
                return
            seed_ds = seed_result(trainer, result_dict)
            write_netcdf_atomic(seed_ds, seed_result_path)

    def _seeded_train_config(self, train_config: TrainConfig, case: Case, seed: int) -> TrainConfig:
        """train_config as one seed of the case trains it.

        The seed sets the model initialization and the training shuffle.
        Each seed trains into its own checkpoint dir, keeping only its best checkpoint beside the latest one it resumes from.
        It stops cleanly at the wall-clock budget, so the next launch can continue.
        """
        model_init_config = {**train_config.model_init_config, "prng_seed": seed}
        dataloader_config = {**train_config.dataloader_config, "shuffle_seed": seed}
        return train_config.model_copy(
            update={
                "model_init_config": model_init_config,
                "dataloader_config": dataloader_config,
                "checkpoint_dir": str(self.seed_model_dir(case, seed)),
                "checkpoint_max_to_keep": 1,
                "resume": True,
                "max_wall_seconds": float(config.train_wall_budget_s),
            }
        )

    def merge_seed_results(self, case: Case) -> None:
        """Fold a case's finished seeds into its result file and keep only the best seed's best checkpoint.

        That checkpoint moves up into trained_model_dir(case), the dir every restore of the case reads
        (transfer finetunes, submodule prereqs, restore_trainer), and the seed dirs are deleted.
        Every step is skipped once done, so a merge short circuits reruns cleanly at the next pass.
        The result file is written last, it marks the case finished.
        """
        self._require_local(case)
        per_seed = {}
        for seed in range(config.num_seeds):
            seed_result_path = self.seed_result_path(case, seed)
            per_seed[seed] = xr.load_dataset(seed_result_path)
        case_ds = aggregate_seed_results(per_seed)
        best_seed = int(case_ds.attrs[BEST_SEED_ATTR])
        best_epoch = int(per_seed[best_seed].attrs[SEED_BEST_EPOCH_ATTR])
        best_checkpoint_dir = self.trained_model_dir(case) / str(best_epoch)
        if not best_checkpoint_dir.exists():
            seed_checkpoint_dir = self.seed_model_dir(case, best_seed) / str(best_epoch)
            os.rename(seed_checkpoint_dir, best_checkpoint_dir)
        for seed in per_seed:
            shutil.rmtree(self.seed_model_dir(case, seed), ignore_errors=True)
        write_netcdf_atomic(case_ds, self.result_path(case))
        for seed in per_seed:
            self.seed_result_path(case, seed).unlink(missing_ok=True)
        logger.info(f"Merged the {len(per_seed)} seeds of case {case}, seed {best_seed} has the lowest validation loss")

    ##############
    # COLLECTION #
    ##############
    @staticmethod
    def _summarize_case_errors(ds: xr.Dataset) -> xr.Dataset:
        """Reduce a case's result dataset to scalar error statistics.

        Reads the four error variables every study's test eval suite writes (error_{abs,rel}_{shot,ts})
        and emits err_E_D_S scalars where
        E is 'abs' or 'rel',
        D is 'shot' or 'ts',
        S is one of mean, std, med, p25, p75, min, max.
        Each statistic is computed per seed, one case score per seed,
        then reduced by seed_statistics to err_E_D_S (the seed mean), err_E_D_S_best and err_E_D_S_seed_std.

        The per-timeslice stats only count timeslices that advance the shot
        clock: the padded rollout tail repeats each shot's final timeslice
        (roughly 40 percent of the array entries in practice), which would
        otherwise weight the per-ts stats heavily toward shot-end error. The
        per-shot integrals need no masking because the padded repeats have
        near-zero dt and contribute nothing to the trapezoid.
        """
        real = real_timeslice_mask(ds["time"])
        errors = {
            "abs_shot": ds["error_abs_shot"],
            "rel_shot": ds["error_rel_shot"],
            "abs_ts": ds["error_abs_ts"].where(real),
            "rel_ts": ds["error_rel_ts"].where(real),
        }
        per_seed = {}
        for error_name, error in errors.items():
            # Every dim but the seeds is reduced, each seed keeps its own statistics
            error_flat = error.stack(point=[dim for dim in error.dims if dim != SEED_DIM])
            per_seed[f"err_{error_name}_mean"] = error_flat.mean("point")
            per_seed[f"err_{error_name}_std"] = error_flat.std("point")
            per_seed[f"err_{error_name}_med"] = error_flat.median("point")
            per_seed[f"err_{error_name}_p25"] = error_flat.quantile(0.25, dim="point").drop_vars("quantile")
            per_seed[f"err_{error_name}_p75"] = error_flat.quantile(0.75, dim="point").drop_vars("quantile")
            per_seed[f"err_{error_name}_min"] = error_flat.min("point")
            per_seed[f"err_{error_name}_max"] = error_flat.max("point")
        # Only the seed index survives as a coordinate
        per_seed_ds = xr.Dataset(per_seed).reset_coords(drop=True)
        return seed_statistics(per_seed_ds, int(ds.attrs[BEST_SEED_ATTR]))

    def write_collected_results(self, enable_parallelism: bool) -> None:
        """Collect the results of every finished case into collected_results.nc.

        With parallelism the collection runs as a CPU job of its own (slurm_utils.launch_collect_results_job),
        its COLLECT_PROCESSES processes reading the result files while the orchestrator waits for the file.
        A job SLURM refuses, or one that ends without writing the file, falls back to collecting here.
        """
        if enable_parallelism and self._collect_results_in_job():
            return
        ds_final = self.collect_results()
        if not ds_final.data_vars:
            raise RuntimeError(f"Study {self.name} has no case result to collect, check its [datasets] table and the logs above")
        write_netcdf_atomic(ds_final, self.collected_results_path())

    def _collect_results_in_job(self) -> bool:
        """Run collect_results in a job of its own and wait for collected_results.nc, returns whether the job wrote it.

        The job counts as failed once it has left the queue and sacct shows it ended RELAUNCH_GRACE_S ago (slurm_utils.held_job_names).
        """
        collected_results_path = self.collected_results_path()
        job_id = launch_collect_results_job(self, COLLECT_PROCESSES)
        if job_id is None:
            logger.warning("SLURM refused the collection job, collecting the results here instead")
            return False
        job_name = self.collect_job_name()
        submitted_jobs = {job_name: SubmittedJob(job_id, time.time())}
        logger.info(f"Collecting the case results in job {job_id}, waiting for {collected_results_path}")
        while not collected_results_path.exists():
            time.sleep(COLLECT_POLL_INTERVAL_S)
            running_job_names = get_running_job_names()
            if running_job_names is None or job_name in running_job_names:
                continue
            held_names = held_job_names(submitted_jobs, running_job_names)
            if job_name not in held_names and not collected_results_path.exists():
                logger.error(f"Collection job {job_id} ended without writing {collected_results_path}, collecting the results here instead")
                return False
        return True

    def restore_trainer(self, case: Case, restore_best_checkpoint: bool = True) -> tuple[Trainer, DataLoader]:
        """Restore a given case's trainer and the test dataloader"""
        if not self.trained_model_dir(case).exists():
            raise ValueError(f"Trained model directory for case\n{case}\nnot found at\n{self.trained_model_dir(case)}")
        if not self.result_path(case).exists():
            logger.warning(f"Result file for case\n{case}\nnot found at\n{self.result_path(case)}\nTraining may be incomplete!")
        training_config = self.make_train_config(case)
        trainer, _, _, test_dl = build_train_run(training_config)
        if restore_best_checkpoint:
            trainer.restore_best_checkpoint(path=self.trained_model_dir(case))

        return trainer, test_dl

    @classmethod
    def case_coord_names(cls) -> tuple[str, ...]:
        """Coords describing which case a record belongs to: case_idx and every case-grid axis."""
        return ("case_idx", *cls.Config.case_axes())

    def case_coords(self, case_idx: int, case: Case) -> dict:
        """Build the per-case coordinate values for collect_results."""
        coords = {}
        for name in self.case_coord_names():
            if name == "case_idx":
                coords[name] = case_idx
            elif name == "training_data":
                coords[name] = str(case.training_data)
            elif name == "domain_adaptation":
                # Normalize None -> "none" so the coord stays string-typed
                coords[name] = case.domain_adaptation if case.domain_adaptation is not None else "none"
            else:
                coords[name] = getattr(case, name)
        return coords

    def collect_results(self) -> xr.Dataset:
        """Collect scalar summary statistics per case (one row per case).

        Default implementation for the scalar-summary studies (power balance, transport).
        The profile study overrides it with a per-shot long form.

        Dims: case_idx
        Coords (along case_idx): the case_coord_names fields of each case
        Data variables (along case_idx): err_E_D_S where E is 'abs' or 'rel', D is
        'shot' (time-integrated per shot) or 'ts' (per timeslice), and S is one of
        mean, std, med, p25, p75, min, max. E.g. err_abs_shot_mean, err_rel_ts_p75.
        """
        finished_cases = [(case_idx, case) for case_idx, case in enumerate(self.cases) if self.result_path(case).exists()]
        result_paths = [self.result_path(case) for _, case in finished_cases]
        summaries = map_result_files(summarize_result_file, result_paths)
        results = []
        for (case_idx, case), summary in zip(finished_cases, summaries, strict=True):
            case_coords = self.case_coords(case_idx, case)
            results.append(summary.assign_coords(case_coords))

        if not results:
            logger.warning("No case results found to collect!")
            return xr.Dataset()

        # coords="different" stacks the per-case scalar coords along case_idx.
        # compat pinned explicitly, the xarray default is changing to
        # "override" which is incompatible with coords="different"
        return xr.concat(results, dim="case_idx", coords="different", compat="equals")

    def run_analysis(self, enable_parallelism: bool) -> None:
        """Post-orchestration analysis and plotting (study-specific)."""
        raise NotImplementedError

    def _visualize_data(self):
        self.DATA_VISUALIZATION.hazard_extrapolation(self.figure_dir)
        self.DATA_VISUALIZATION.target_shot_orders(self.figure_dir)
        self.DATA_VISUALIZATION.domain_overlap(self.figure_dir)

    @classmethod
    def run_study(
        cls,
        config: CaseGridConfig | str | Path,
        enable_parallelism: bool | None = False,
        skip_tuning: bool | None = True,
        skip_visualization: bool | None = False,
        clean_sweeps: bool | None = False,
        clean_models: bool | None = False,
        clean_results: bool | None = False,
        clean_figures: bool | None = False,
    ):
        """
        Go from datasets to collected results and figures in one command.

        Requires the dataset paths in the config TOML or environment variables.
        No dataset ships with this repository, build each device's store with `transport_study/datasets/` (see dataset_creation.md).

        *"I hardly lifted a finger" - Engi B*

        Parameters
        ----------
        config : CaseGridConfig | str | Path
            The study config, or a path to its TOML file.
        enable_parallelism : bool | None
            If false, runs the entire study sequentially in one process.
            If true, submits independent training steps with SLURM up to configurable resource limits.
            The idea is you would periodically call this 'run_study' function, and it checks what models still need to be trained and submit jobs for those, until eventually all models are trained and all results are computed.
        skip_tuning : bool | None
            If True, skip hyperparameter tuning steps.
        skip_visualization : bool | None
            If True, skip data visualization steps.
        clean_sweeps : bool | None
            If True, delete any existing wandb sweeps for this project before running.
        clean_models : bool | None
            If True, delete any existing trained models in the working directory before running.
            Also deletes the results and the config lock, so the study starts over under the current config.
        clean_results : bool | None
            If True, delete any existing results in the working directory before running.
            Keeps the models and the config lock, so every result is recomputed from the kept checkpoints.
        clean_figures : bool | None
            If True, delete any existing figures in the figure directory before running.
        """
        # Parse with the concrete Config subclass so the local `config` name holds a
        # real config object (it shadows the module-level proxy)
        if isinstance(config, (str, Path)):
            config = cls.Config.from_toml(Path(config))
        # One log file per orchestrator run, the processes that only build a Study (analysis jobs) log to SLURM alone
        run_log_dir = Path(config.working_dir_base) / config.study_name / "logs"
        run_log_dir.mkdir(parents=True, exist_ok=True)
        run_log_path = run_log_dir / f"{os.getpid()}_run_study.log"
        logger.add(run_log_path)
        # The partition checks read the global config, and run before anything is cleaned or written
        _load_study_config(config)
        if enable_parallelism:
            if not config.partition:
                raise ValueError("enable_parallelism is True but no SLURM partition is specified in the config")
            missing_cpu_partitions = [p for p in cpu_partitions() if not partition_exists(p)]
            if missing_cpu_partitions:
                raise ValueError(f"CPU partitions {missing_cpu_partitions} are unknown to scontrol")

        cls.clean_working_dir(
            config,
            enable_parallelism=bool(enable_parallelism),
            clean_models=bool(clean_models),
            clean_results=bool(clean_results),
            clean_figures=bool(clean_figures),
        )
        study = cls(config)
        study.record_case_grid()
        study.setup_directories(
            enable_parallelism=bool(enable_parallelism),
            skip_tuning=bool(skip_tuning),
            skip_visualization=bool(skip_visualization),
            clean_sweeps=bool(clean_sweeps),
        )

        # The files of an earlier run, possibly over another grid, never stand in for this run's
        for stale_path in (study.collected_results_path(), study.collected_metrics_path(), study.study_complete_path()):
            stale_path.unlink(missing_ok=True)

        if not skip_visualization:
            logger.opt(colors=True).info("<bold><magenta>DATA VISUALIZATION</magenta></bold>")
            study._visualize_data()

        # Returns at once when every case already has a result
        logger.opt(colors=True).info("<bold><magenta>ORCHESTRATION</magenta></bold>")
        study.run_unfinished_cases(skip_tuning=bool(skip_tuning), enable_parallelism=bool(enable_parallelism))
        study.write_collected_results(enable_parallelism=bool(enable_parallelism))

        study.run_analysis(enable_parallelism=bool(enable_parallelism))
        study.study_complete_path().touch()

    @classmethod
    def reset_study(cls, config: CaseGridConfig | str | Path) -> None:
        """Start a study over from a clean slate, ready for run_study.

        Deletes the whole working dir (models, results, figures, logs and the config lock)
        and the wandb sweeps of the study's own hyperparam cases, then writes a fresh config lock.
        Refuses while any job of the study is queued or running.
        Reset a lineage root-first, so each child's fresh lock records its parent's fresh stamp
        and no child orchestrator can open a parent lock that is about to be replaced.

        Parameters
        ----------
        config : CaseGridConfig | str | Path
            The study config, or a path to its TOML file.
        """
        if isinstance(config, (str, Path)):
            config = cls.Config.from_toml(Path(config))
        # A study without a partition never submitted a job
        if config.partition:
            job_names = get_running_job_names()
            if job_names is None:
                raise RuntimeError(f"squeue failed, so whether a job of study {config.study_name} is still queued or running is unknown")
            # Every job name of a study ends with the study name (see train_job_name)
            study_job_names = sorted(name for name in job_names if name.endswith(f".{config.study_name}"))
            if study_job_names:
                raise RuntimeError(
                    f"Study {config.study_name} has jobs queued or running, cancel them before resetting it:\n" + "\n".join(study_job_names)
                )
        working_dir = Path(config.working_dir_base) / config.study_name
        if working_dir.exists():
            shutil.rmtree(working_dir)
        study = cls(config)
        study.setup_directories(clean_sweeps=True)

    def __init__(self, cfg: str | Path | CaseGridConfig):
        """
        Initialize this study from its Config object or a path to its TOML file.
        Dataset paths come from the global config's dataset_paths
        (TOML [datasets] table, with the PTPS_DATASET_PATHS JSON env var as defaults).
        """
        # load_config(Path) only knows how to build the base StudyConfig,
        # which forbids this study's extra fields, so parse with the subclass
        if isinstance(cfg, (str, Path)):
            cfg = self.Config.from_toml(Path(cfg))
        _load_study_config(cfg)

        self.name = config.study_name
        # Launch counter per seed (train job name -> count) backing MAX_TRAIN_ATTEMPTS
        self.train_attempts: dict[str, int] = {}
        # Latest-checkpoint epoch per seed (train job name) as of its last launch.
        # A relaunch whose checkpoint advanced past this is a resume making progress, not a failure
        self.train_attempt_epochs: dict[str, int | None] = {}
        # Measured batches per epoch per loader config,
        # so repeated make_train_config calls do not reread the records (see batches_per_epoch)
        self._batches_per_epoch_cache: dict[str, int] = {}
        # get_loaded_shot_count per (device, study type), read by make_weighted_device_weights
        self._loaded_shot_counts: dict[tuple[str, str], int] = {}
        # training_loss_statistics per training-data selection
        self._training_statistics_cache: dict[tuple, dict[str, dict]] = {}
        # The next time each tuning case's sweep progress (str(case))
        # and each agent job's liveness (job name) may be read from wandb again
        self._next_sweep_poll: dict[str, float] = {}
        self._next_agent_check: dict[str, float] = {}
        # Every training job this orchestrator submitted (or recalled) by job name, until it is resolved (slurm_utils.held_job_names)
        self.submitted_train_jobs: dict[str, SubmittedJob] = {}
        # The open slots of the current orchestration pass, debited per accepted job (see _launch_ready_cases).
        # It queries nothing until a job needs a partition, and every parallel pass starts a fresh one
        self.launch_slots = LaunchSlots()
        # make_train_config of every case launched in the current pass, which all its seeds share
        self._pass_train_configs: dict[str, TrainConfig] = {}

        self.working_dir = Path(config.working_dir_base) / self.name
        self.model_dir = self.working_dir / "models"
        self.result_dir = self.working_dir / "results"
        self.figure_dir = self.working_dir / "figures"
        self.log_dir = self.working_dir / "logs"

        self.working_dir.mkdir(parents=True, exist_ok=True)
        # Ancestor grids are rebuilt only when their recorded axes change, see _load_archive
        self._ancestor_case_names: dict[tuple, frozenset[str]] = {}
        parent = self._load_parent_archive()
        self.lock = self._open_config_lock(parent)
        self.cases = self.make_cases(config.get_instance())
        self.archive = StudyArchive(
            name=self.name,
            working_dir=self.working_dir,
            stamp=self.lock.stamp,
            parent=parent,
            case_names=frozenset(str(case) for case in self.cases),
        )
        # str(case) -> the study it lives in, cleared whenever the ancestor chain is reloaded
        self._case_homes: dict[str, StudyArchive] = {}

        logger.info("INITIALIZING STUDY")
        logger.info(f"Study name: {self.name}")
        logger.info(f"Working directory base: {config.working_dir_base}")
        logger.info(f"Total number of cases: {len(self.cases)}")
        if parent is not None:
            homes = Counter(self.case_home(case).name for case in self.cases)
            logger.info(f"Case homes: {dict(homes)} ({homes[self.name]} train here, the rest are borrowed)")
        logger.info(f"Target test set size: {config.target_test_set_size}")
        logger.info(f"Target test shots: {config.target_test_shots}")
        logger.info(f"Dataset paths: {config.dataset_paths}")
        logger.info(f"Target device: {config.target_device}")
        for axis_field in config.fields_with_role(FieldRole.CASE_AXIS):
            logger.info(f"{axis_field}: {getattr(config, axis_field)}")

    def _open_config_lock(self, parent: StudyArchive | None) -> ConfigLock:
        """Check the current config against the study's config lock, or write the lock on the study's first run.

        The lock pins the identity and locked fields (see transport_study/config.py),
        so a run under a different setup raises instead of mixing its results with the existing ones.
        A lock is only ever created over an empty study, so deleting one by hand can never vouch for stale results.
        """
        lock_path = self.working_dir / CONFIG_LOCK_FILENAME
        if not lock_path.exists():
            stale_dirs = [directory for directory in (self.model_dir, self.result_dir) if directory.exists() and any(directory.iterdir())]
            if stale_dirs:
                raise RuntimeError(
                    f"Study {self.name} has no config lock, but {[str(d) for d in stale_dirs]} hold artifacts of an unknown setup.\n"
                    "Start the study over with clean_models, or with reset_study for a parallel study."
                )
            lock = ConfigLock(
                config=config.get_instance(),
                study_type=self.STUDY_TYPE,
                stamp=new_stamp(),
                parent_stamp=parent.stamp if parent is not None else None,
            )
            write_config_lock(lock_path, lock)
            # Batch counts recorded under an earlier setup or dataset do not carry over to a fresh lock
            (self.working_dir / BATCHES_PER_EPOCH_FILENAME).unlink(missing_ok=True)
            return lock

        lock = read_config_lock(lock_path, config.get_instance(), self.STUDY_TYPE)
        changed = config.differing_fields(lock.config, FieldRole.IDENTITY, FieldRole.LOCKED)
        if changed:
            details = "\n".join(f"  {name}: locked {getattr(lock.config, name)!r}, now {getattr(config, name)!r}" for name in changed)
            raise RuntimeError(
                f"Study {self.name} was locked under a different config, these fields changed:\n{details}\n"
                "Revert them, or start the study over under the new config with clean_models, or with reset_study for a parallel study."
            )
        if parent is not None:
            self._check_parent_stamp(self.name, lock.parent_stamp, parent)
        return lock

    def record_case_grid(self) -> None:
        """Rewrite the config lock when the case-grid axes changed, so it records the grid of the latest orchestrator run.

        Descendant studies read that record to know which cases this study will train.
        Only run_study calls this, analysis jobs and plotting entry points never change the record.
        """
        current = config.get_instance()
        if not current.differing_fields(self.lock.config, FieldRole.CASE_AXIS):
            return
        self.lock = ConfigLock(config=current, study_type=self.lock.study_type, stamp=self.lock.stamp, parent_stamp=self.lock.parent_stamp)
        write_config_lock(self.working_dir / CONFIG_LOCK_FILENAME, self.lock)

    ###########
    # LINEAGE #
    ###########
    @staticmethod
    def _check_parent_stamp(child_name: str, recorded_stamp: str | None, parent: StudyArchive) -> None:
        """Raise when the parent was reset after the child's lock recorded its stamp."""
        if recorded_stamp != parent.stamp:
            raise RuntimeError(
                f"Study {parent.name} was reset after study {child_name} was created on top of it, "
                f"so every case {child_name} borrowed or trained from it is stale.\n"
                f"Start {child_name} over on {parent.name}'s current results with clean_models, or with reset_study for a parallel study."
            )

    def _load_parent_archive(self) -> StudyArchive | None:
        """The read-only chain of studies above this one, None without a parent_study."""
        if config.parent_study is None:
            return None
        return self._load_archive(config.parent_study, descendants=(self.name,))

    def _load_archive(self, name: str, descendants: tuple[str, ...]) -> StudyArchive:
        """An ancestor's read-only view, from its config lock, with the chain above it.

        No Study object is built for an ancestor: the global config can only be loaded once per process,
        and Study.__init__ writes into its working dir.
        Its grid is rebuilt here under the global config, which is safe because every locked field is equal.
        """
        if name in descendants:
            raise RuntimeError(f"Study {name} is its own ancestor through {' -> '.join(descendants)}")
        working_dir = Path(config.working_dir_base) / name
        lock_path = working_dir / CONFIG_LOCK_FILENAME
        if not lock_path.exists():
            raise RuntimeError(f"Parent study {name} has no config lock at {lock_path}, run it before its descendants.")
        lock = read_config_lock(lock_path, config.get_instance(), self.STUDY_TYPE)
        changed = config.differing_fields(lock.config, FieldRole.LOCKED)
        if changed:
            details = "\n".join(
                f"  {field}: {name} has {getattr(lock.config, field)!r}, {self.name} has {getattr(config, field)!r}" for field in changed
            )
            raise RuntimeError(
                f"Study {self.name} can only borrow from {name} when every locked config field matches, these differ:\n{details}"
            )

        # read_config_lock built it with the running config's class, a CaseGridConfig
        ancestor_config = lock.config
        assert isinstance(ancestor_config, CaseGridConfig)
        parent = None
        if ancestor_config.parent_study is not None:
            parent = self._load_archive(ancestor_config.parent_study, descendants=(*descendants, name))
            self._check_parent_stamp(name, lock.parent_stamp, parent)

        axes = lock.config.toml_data((FieldRole.CASE_AXIS,))[ROLE_TABLES[FieldRole.CASE_AXIS]]
        grid_key = (name, lock.stamp, json.dumps(axes, sort_keys=True, default=str))
        if grid_key not in self._ancestor_case_names:
            self._ancestor_case_names[grid_key] = frozenset(str(case) for case in self.make_cases(lock.config))
        return StudyArchive(
            name=name, working_dir=working_dir, stamp=lock.stamp, parent=parent, case_names=self._ancestor_case_names[grid_key]
        )

    def refresh_lineage(self) -> None:
        """Reload the ancestor chain from the ancestors' config locks, every orchestration pass.

        A reset ancestor stops the run, and a changed ancestor grid moves cases between borrowed and local.
        """
        if self.archive.parent is None:
            return
        parent = self._load_parent_archive()
        assert parent is not None
        self._check_parent_stamp(self.name, self.lock.parent_stamp, parent)
        self.archive = replace(self.archive, parent=parent)
        self._case_homes.clear()

    def case_home(self, case: Case) -> StudyArchive:
        """The study whose working dir holds this case's checkpoints, tuned config and result.

        Root-first: the root-most ancestor that claims the case (has its result, or its current grid holds it),
        this study when no ancestor does.
        A study nearer than the home that holds its own copy means two studies of the chain disagree on the case,
        which raises rather than mixing their artifacts.
        """
        case_name = str(case)
        if case_name in self._case_homes:
            return self._case_homes[case_name]
        chain = self.archive.chain()
        home_idx = next((idx for idx in reversed(range(1, len(chain))) if chain[idx].claims(case_name)), 0)
        home = chain[home_idx]
        for study in chain[:home_idx]:
            own_copy = study.has_result(case_name) or (study is self.archive and (self.model_dir / case_name).exists())
            if own_copy:
                raise RuntimeError(
                    f"Study {study.name} holds its own copy of case {case_name}, but its ancestor {home.name} now claims it.\n"
                    f"Drop the case from {home.name}'s grid, or clean {study.name} so it borrows the case instead."
                )
        self._case_homes[case_name] = home
        return home

    def is_borrowed(self, case: Case) -> bool:
        return self.case_home(case) is not self.archive

    def _require_local(self, case: Case) -> None:
        """Guard on every path that writes a case's artifacts, a borrowed case's home is never written."""
        if self.is_borrowed(case):
            raise RuntimeError(f"Case {case} belongs to study {self.case_home(case).name}, study {self.name} must not write it.")
