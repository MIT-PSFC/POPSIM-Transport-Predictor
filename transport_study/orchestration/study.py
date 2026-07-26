from __future__ import annotations

import json
import math
import os
import shutil
import time
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar

import toml
import wandb
import xarray as xr
import yaml
from loguru import logger
from popsim.ml import DataLoader, TrainConfig, Trainer
from popsim.ml.launch import (
    get_train_run_builder_class,
    launch_agent,
    launch_train,
    resolve_transition_frac,
)
from popsim.ml.train_config import load_dict
from pydantic import Field, field_validator, model_validator

from transport_study import PACKAGE_ROOT, TIME_DIM
from transport_study.config import StudyConfig, config, env_dataset_paths, load_config
from transport_study.modules.normalization import STAT_NORMALIZATIONS
from transport_study.orchestration.organize_data import (
    TrainingData,
    get_loaded_shot_count,
    parse_training_data,
)
from transport_study.orchestration.slurm_utils import (
    cancel_job,
    count_idle_gpus,
    count_running_jobs,
    get_pending_job_pending_s,
    get_running_job_elapsed_s,
    get_running_job_names,
    launch_agent_parallel,
    launch_train_parallel,
    pick_partition,
    spillover_budget,
    spillover_slots,
)
from transport_study.orchestration.topk_results import compute_topk_study_results
from transport_study.orchestration.wandb_utils import (
    get_best_train_config,
    get_completed_runs,
    get_sweep_id,
    has_live_agent_run,
    run_clean_sweeps,
)

CONFIG_LOCK_FILENAME = "config_lock.toml"

# Target-shot count for hyperparameter tuning cases. When domain adaptation is
# None no target data is used during training, and during domain adaptation no
# hyperparameter tuning is done, so 0 is the canonical value
HYPERPARAM_TARGET_SHOTS = 0

# Abort the run once a single case has been launched this many times without a result file, guards against deterministically failing jobs being resubmitted forever
MAX_TRAIN_ATTEMPTS = 3

# A sweep is done once the trial target is met AND this fraction of it finished, since the tuned config is picked from finished runs only
# Hyperband with eta 3 lets only a few percent of trials run to completion, so demanding much more than that forces extra trials far past the count target
MIN_FINISHED_FRACTION = 0.05

# How long the orchestration loop sleeps between passes over the unfinished cases
ORCHESTRATION_POLL_INTERVAL_S = 20

# Result files can lag job exit by ~1 min on NFS, so wait this long after a case leaves the queue before relaunching it
RELAUNCH_GRACE_S = 180

# Tuned learning rates were swept for from-scratch training. Fine-tuning scales
# the schedule by a step budget instead of a fixed factor: keep lr x total_steps
# at roughly lr0 x max_epochs, so a step-starved finetune (batch_size larger
# than the finetune dataset, 1 optimizer step per epoch) runs at the full tuned
# LR while a step-rich one cools toward this floor. See _scale_transfer_lr
TRANSFER_LR_FLOOR = 0.1

# A training job is considered stuck once it has run at least this long with no progress (e.g. OpenBLAS or XLA compile-pool deadlocks)
WATCHDOG_MIN_AGE_S = 80 * 60

# Checkpoints are only written at the validation cadence, so stall detection is a coarse "no epochs completed recently" signal
WATCHDOG_STALL_S = 60 * 60

# wandb heartbeats every ~30s during a live run, so 10 min of silence on a "running" run is already well past any legitimate gap
AGENT_HEARTBEAT_STALL_S = 10 * 60

# A job parked PENDING on a busy spillover partition can sit in the queue indefinitely. After
# this long, cancel it so the orchestration loop re-queues it (pick_partition then re-decides
# placement). Jobs pending on the primary partition are exempt and keep their queue position
WATCHDOG_PENDING_S = 30 * 60

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


def configure_jax_platforms(enable_parallelism: bool) -> None:
    """Pick the jax backend for the orchestrator process before jax initializes it.

    An explicit JAX_PLATFORMS in the environment always wins.
    With parallelism the orchestrator does no GPU compute, so pin cpu
    (submitted jobs re-export their own value in their sbatch scripts, see slurm_utils)
    Without parallelism training runs in this process, so leave jax to its
    default backend selection: gpu when one exists, cpu fallback otherwise.
    """
    if "JAX_PLATFORMS" in os.environ:
        return
    if enable_parallelism:
        os.environ["JAX_PLATFORMS"] = "cpu"


class CaseGridConfig(StudyConfig):
    """Config base for studies built on a case grid (model_type x training_data x ...).

    Holds the case-grid axes every such study shares. Study-specific axes
    (model_types, freeze options, hyperparam selections) and is_compatible
    stay on the study's own Config subclass.
    """

    # Organization for datasets and wandb projects
    working_dir_base: Path = Field(
        default_factory=lambda: Path(os.environ.get("PTPS_WORKING_DIR_BASE", str(PACKAGE_ROOT / "popsim_studies" / "working_dir")))
    )
    # Case-grid axes shared by every study of this shape
    training_datasets: tuple[TrainingData, ...]
    domain_adaptation_methods: tuple[str | None, ...] = Field(default_factory=lambda: (None, "weighted", "addition", "transfer"))
    target_test_set_size: int
    # Hyperparameter tuning case axes shared by every study of this shape
    hyperparam_domain_adaptation: str | None = None
    hyperparam_num_target_shots: int = HYPERPARAM_TARGET_SHOTS
    # Optional dict of dataset fractions to use during domain adaptation, only used if domain_adaptation includes "weighted"
    dataset_fractions: dict[str, float] = Field(default_factory=dict)
    # How many best-by-val-loss checkpoints each production training run keeps.
    # Final results average the test metrics over all of them, so the report
    # is robust to the val-loss argmin flipping between near-tied epochs under
    # GPU float noise, and the per-checkpoint spread quantifies that noise.
    # Sweeps always keep 1. Must be >= 1 (1 only computes best-checkpoint test metrics)
    num_result_checkpoints: int = Field(default=10, ge=1)

    # Study-specific hyperparam field names checked by is_compatible, set per subclass
    COMPAT_HYPERPARAM_FIELDS: ClassVar[tuple[str, ...]] = ()

    def is_compatible(self, cfg: CaseGridConfig) -> bool:
        """Whether two configs can run the same study (the config-lock check).

        Compares study identity (name, datasets, target, test set size,
        dataset fractions) and the hyperparameter tuning configuration.
        Case-grid axes like model_types may differ between runs.
        """
        names = (
            "study_name",
            "dataset_paths",
            "target_device",
            "target_test_set_size",
            "dataset_fractions",
            "num_result_checkpoints",
            *self.COMPAT_HYPERPARAM_FIELDS,
        )
        return all(getattr(self, name) == getattr(cfg, name) for name in names)

    @field_validator("domain_adaptation_methods")
    @classmethod
    def _validate_domain_adaptation(cls, v: tuple[str | None, ...]) -> tuple[str | None, ...]:
        valid = {None, "weighted", "addition", "transfer"}
        converted = tuple(None if da == "none" else da for da in v)
        for da in converted:
            if da not in valid:
                raise ValueError(f"Invalid domain adaptation: {da}. Must be one of {valid}.")
        return converted

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

    @classmethod
    def from_toml(cls, path: Path) -> CaseGridConfig:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        datasets = data.pop("datasets", {})
        target = datasets.pop("target", None)
        study_cases = data.pop("study_cases", {})
        # PTPS_DATASET_PATHS env var provides defaults, explicit TOML paths win
        return cls(
            **data,
            **study_cases,
            dataset_paths=env_dataset_paths() | {k: Path(v) for k, v in datasets.items()},
            target_device=target,
        )

    def save(self, path: Path):
        raw = self.model_dump()
        datasets = {k: str(v) for k, v in self.dataset_paths.items()}
        if self.target_device is not None:
            datasets["target"] = self.target_device
        skip = {"dataset_paths", "target_device", "training_datasets", "domain_adaptation_methods"}
        data = {}
        for k, v in raw.items():
            if k in skip:
                continue
            if v is None:
                continue
            if isinstance(v, Path):
                data[k] = str(v)
            else:
                data[k] = v
        data["training_datasets"] = [str(td) for td in self.training_datasets]
        data["domain_adaptation_methods"] = [da if da is not None else "none" for da in self.domain_adaptation_methods]
        data["datasets"] = datasets
        with open(path, "w") as f:
            toml.dump(data, f)


@dataclass
class ModelTrainSpec:
    """The per-model-type pieces of a TrainConfig, returned by Study._model_train_spec."""

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
    (Study.Case subclass), make_cases, the make_train_config hooks
    (_base_dataloader_config / _base_loss_config / _model_train_spec /
    _tuned_model_init_updates), collect_results, and _run_analysis.
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
    # config attribute names of the case-grid axes (logged at init)
    CASE_AXIS_FIELDS: ClassVar[tuple[str, ...]] = ()
    # dotted paths of the per-case analysis modules dispatched over SLURM (see orchestration.case_analysis)
    # The metrics module must export a compute_and_save_case_metrics function
    # The reports module a generate_case_report and an analysis_case_done function
    ANALYSIS_METRICS_MODULE: ClassVar[str]
    ANALYSIS_REPORTS_MODULE: ClassVar[str]

    # Tuned-config dataloader keys merged by strict indexing (KeyError when a tuned config lacks one)
    TUNED_DATALOADER_KEYS: ClassVar[tuple[str, ...]] = ()
    # Tuned-config loss keys merged with .get fallback to the base value (tuned configs on disk may lack them)
    TUNED_LOSS_KEYS: ClassVar[tuple[str, ...]] = ()

    @dataclass
    class Case:
        """A unique combination of case-grid axes to compare in this study.

        Each case holds everything needed to train and evaluate a model and to
        compare it to other cases. Subclasses declare their extra fields plus
        the VALID_MODEL_TYPES / STR_TOKEN_FIELDS / HYPERPARAM_FIELDS ClassVars,
        keep a thin __init__ in the class body that sets the extra fields and
        then calls _init_common, and alias __hash__ = Study.Case.__hash__
        (a dataclass body without its own __init__ or __hash__ would have them
        regenerated or nulled by the dataclass decorator).
        """

        model_type: str
        training_data: TrainingData
        # None, weighted, addition, transfer, or transfer_pretrain
        # (transfer_pretrain is never a case-grid axis value, it arises only as the
        # pretrain prereq of a transfer case, see transfer_pretrain_case)
        domain_adaptation: str | None
        num_target_shots: int  # Target shots included in training, or -1 for all (HYPERPARAM_TARGET_SHOTS when domain_adaptation is None)
        # Cases this one depends on, run first (None when independent)
        prereqs: list[Study.Case] | None

        # Model types accepted by _validate
        VALID_MODEL_TYPES: ClassVar[tuple[str, ...]] = ()
        # (prefix, field name) or (prefix, field name, suppress_value) tokens
        # between the td_ and targ_ tokens of str(case). The 3-tuple form
        # omits the token entirely when the field holds suppress_value, so a
        # new axis whose default reproduces old behavior can be added without
        # renaming every existing case (see ProfileStudy.Case.geometry_builder)
        STR_TOKEN_FIELDS: ClassVar[tuple[tuple[str, str] | tuple[str, str, object], ...]] = ()
        # Per-case fields with a config.hyperparam_<name> counterpart
        HYPERPARAM_FIELDS: ClassVar[tuple[str, ...]] = ()

        def _init_common(self, model_type, training_data, domain_adaptation, num_target_shots):
            """Shared between every subclass __init__: parse, validate, build prereqs."""
            if isinstance(training_data, str):
                training_data = parse_training_data(training_data, dict(config.dataset_paths), config.target_device)
            self.model_type = model_type
            self.training_data = training_data
            self.domain_adaptation = domain_adaptation
            self.num_target_shots = num_target_shots
            self._validate()
            prereqs = self._build_prereqs()
            self.prereqs = prereqs if prereqs else None

        def _validate(self):
            if self.model_type not in self.VALID_MODEL_TYPES:
                raise ValueError(f"Unknown model type: {self.model_type}")
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
            prereqs.extend(self._model_type_prereqs())
            if self.domain_adaptation == "transfer":
                prereqs.append(self.transfer_pretrain_case())
            return list(dict.fromkeys(prereqs))

        def _model_type_prereqs(self) -> list[Study.Case]:
            """Extra prereq cases implied by the model type (e.g. submodule predictors)."""
            return []

        def _normalization_method(self) -> str | None:
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
            if self._normalization_method() in STAT_NORMALIZATIONS:
                return self.replace(domain_adaptation="transfer_pretrain")
            return self.replace(domain_adaptation="transfer_pretrain", num_target_shots=HYPERPARAM_TARGET_SHOTS)

        def replace(self, **changes) -> Study.Case:
            """Rebuild through the real constructor with some fields changed, so validation and prereqs stay consistent."""
            kwargs = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "prereqs"}
            kwargs.update(changes)
            return type(self)(**kwargs)

        @classmethod
        def _hyperparam_field_values(cls) -> dict:
            values = {"training_data": Study._hyperparam_training_data()}
            for name in cls.HYPERPARAM_FIELDS:
                values[name] = getattr(config, f"hyperparam_{name}")
            return values

        def is_hyperparam_case(self) -> bool:
            return all(getattr(self, name) == value for name, value in self._hyperparam_field_values().items())

        def get_hyperparam_prereq(self) -> Study.Case:
            if self.is_hyperparam_case():
                return self
            return self.replace(**self._hyperparam_field_values())

        def is_impossible(self) -> bool:
            """Some cases don't make sense to run. Mark those cases as impossible and raise an error if we try to run them."""
            # Can't do transfer learning or training from nothing with 0 target shots.
            if (self.domain_adaptation == "transfer" or self.training_data.exnihilo) and self.num_target_shots == 0:
                return True

            # A stat-normalized pretrain twin exists to fit target-aware statistics,
            # so with 0 target shots it is meaningless. Stateless-normalization twins
            # fit nothing and canonically run at HYPERPARAM_TARGET_SHOTS (see transfer_pretrain_case)
            if (
                self.domain_adaptation == "transfer_pretrain"
                and self.num_target_shots == 0
                and self._normalization_method() in STAT_NORMALIZATIONS
            ):
                return True

            # exnihilo means training from nothing - no source domain to adapt from
            if self.training_data.exnihilo and self.domain_adaptation is not None:
                return True

            return False

        def __str__(self):
            parts = [f"case.{self.model_type}", f"td_{self.training_data}"]
            for token in self.STR_TOKEN_FIELDS:
                prefix, field_name = token[0], token[1]
                value = getattr(self, field_name)
                if len(token) == 3 and value == token[2]:
                    continue
                parts.append(f"{prefix}{value}")
            if self.domain_adaptation:
                parts.append(f"targ_{self.num_target_shots}")
                parts.append(f"da_{self.domain_adaptation}")
            elif self.training_data.exnihilo:
                parts.append(f"targ_{self.num_target_shots}")
            return ".".join(parts)

        def __hash__(self):
            return hash(tuple(v for k, v in self.__dict__.items() if k != "prereqs"))

    ####################
    # PATHING / NAMING #
    ####################
    def trained_model_dir(self, case: Case) -> Path:
        """Given a case, return the path where the trained model checkpoints for that case should be stored."""
        return Path(self.model_dir) / str(case)

    def result_path(self, case: Case) -> Path:
        """Given a case, return the path where the results for that case should be stored."""
        return Path(self.result_dir) / str(case) / "result_data.nc"

    def _latest_checkpoint_dir_info(self, case: Case) -> tuple[int, float] | None:
        """(epoch, mtime) of the newest resume checkpoint of the case,
        or None if there is none yet.

        Orbax names each checkpoint directory after its step (here the epoch)
        and renames it into place atomically once fully written, so the dir's
        mtime marks the moment that epoch's checkpoint became visible.
        In-progress saves get a non-numeric tmp suffix and are skipped.
        """
        latest_dir = Path(f"{self.trained_model_dir(case)}_latest")
        if not latest_dir.exists():
            return None
        epoch_dirs = [p for p in latest_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        if not epoch_dirs:
            return None
        newest = max(epoch_dirs, key=lambda p: int(p.name))
        return int(newest.name), newest.stat().st_mtime

    def latest_checkpoint_epoch(self, case: Case) -> int | None:
        """Epoch of the newest resume checkpoint, or None if empty."""
        info = self._latest_checkpoint_dir_info(case)
        return info[0] if info else None

    def collected_results_path(self) -> Path:
        """Return the path where the collected results for all cases should be stored."""
        return Path(self.result_dir) / "collected_results.nc"

    def tuned_config_path(self, case: Case) -> Path:
        """Given a case, return the path where the tuned hyperparameters for that case should be stored"""
        hyperparam_case = case.get_hyperparam_prereq()
        return Path(self.model_dir) / str(hyperparam_case) / "tuned_config.yaml"

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

    def train_job_name(self, case: Case) -> str:
        return f"train_{case}.{self.name}"

    def analysis_job_name(self, case: Case) -> str:
        return f"analysis_{case}.{self.name}"

    def check_prereq_satisfied(self, case: Case) -> bool:
        """Check if the prerequisites for this case have been satisfied by looking for the existence of the result path"""
        if case.prereqs is None:
            return True
        for prereq in case.prereqs:
            prereq_result_path = self.result_path(prereq)
            if not prereq_result_path.exists():
                return False
        return True

    def setup_directories(
        self,
        enable_parallelism: bool = False,
        skip_tuning: bool = True,
        skip_visualization: bool = False,
        clean_sweeps: bool = False,
        clean_models: bool = False,
        clean_results: bool = False,
        clean_figures: bool = False,
    ):
        logger.info("SETTING UP DIRECTORIES")
        logger.info(f"Enable parallelism: {enable_parallelism}")
        logger.info(f"Skip hyperparameter tuning: {skip_tuning}")
        logger.info(f"Skip visualization: {skip_visualization}")
        logger.info(f"Clean sweeps: {clean_sweeps}")
        logger.info(f"Clean models: {clean_models}")
        logger.info(f"Clean results: {clean_results}")
        logger.info(f"Clean figures: {clean_figures}")

        if (clean_sweeps or clean_models or clean_results or clean_figures) and enable_parallelism:
            raise ValueError(
                "Cannot clean models, results, or figures when parallelism is enabled, as this would interfere with jobs currently running or queued."
            )

        if (not skip_tuning) and (not enable_parallelism):
            logger.critical(
                "Hyperparameter tuning without parallelism enabled is probably gonna take a long time, are you sure you want to do this?"
            )

        if clean_sweeps:
            project_names = {self.wandb_project_name(case) for case in self.cases if case.is_hyperparam_case()}
            run_clean_sweeps(project_names)
        if clean_models:
            shutil.rmtree(self.model_dir, ignore_errors=True)
            shutil.rmtree(self.working_dir / "wandb", ignore_errors=True)
        if clean_results:
            shutil.rmtree(self.result_dir, ignore_errors=True)
        if clean_figures:
            shutil.rmtree(self.figure_dir, ignore_errors=True)

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
        return TrainingData(sources_unsorted=sources)

    def make_cases(self) -> list[Case]:
        """Build every case of the study's case grid from the global config (study-specific)."""
        raise NotImplementedError

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
    def make_weighted_device_weights(self, case: Case) -> dict[str, float]:
        """Loss weights per device for weighted domain adaptation.

        Mirrors the actual training-set composition of get_train_test_datasets:
        every loaded shot of each source device in case.training_data (the
        historic train and val splits are both concatenated into the combined
        training set) plus case.num_target_shots target shots.
        Shot counts come from get_loaded_shot_count, so max_ds_size truncation
        and study-type filtering are accounted for.

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

        # Shot counts as the training set actually sees them
        shot_counts = {source: get_loaded_shot_count(source, study_type=self.STUDY_TYPE) for source in case.training_data.sources}
        if case.num_target_shots == -1:
            # All loaded target shots end up in training (CHEATING reference case)
            shot_counts[target] = get_loaded_shot_count(target, study_type=self.STUDY_TYPE)
        elif case.num_target_shots > 0:
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

    def _transfer_steps_per_epoch(self, train_config: TrainConfig) -> int:
        """Optimizer steps per epoch of the finetune train dataloader, measured from the data.

        Builds the train dataloader exactly as popsim launch will (same TRB
        resolution including the data_train_run_builder override), so the
        count reflects segmentation, NaN culling, and drop_last rather than
        an estimate from shot counts. Cached per dataloader config because
        building the dataloaders loads the datasets (relaunches and freeze
        twins share a config, so they share a cache entry).
        """
        dataloader_config = train_config.dataloader_config
        builder = dataloader_config.get("data_train_run_builder") or train_config.train_run_builder
        cache_key = json.dumps([str(builder), dataloader_config], sort_keys=True, default=str)
        if cache_key not in self._transfer_steps_cache:
            train_run_builder = get_train_run_builder_class(builder)
            _, train_dl, _, _ = train_run_builder.get_dataloaders(dataloader_config)
            self._transfer_steps_cache[cache_key] = max(1, len(train_dl))
        return self._transfer_steps_cache[cache_key]

    def _scale_transfer_lr(self, train_config: TrainConfig) -> TrainConfig:
        """Step-budget the learning-rate schedule for fine-tuning from a pretrained checkpoint.

        Tuned learning rates were swept for from-scratch training. A fixed
        cooling factor starves a step-poor finetune: with batch_size larger
        than the finetune dataset there is 1 optimizer step per epoch and
        max_epochs steps total, so a 0.1 factor leaves the pretrained model
        essentially unmoved. Budget rule: keep lr x total_steps at roughly
        lr0 x max_epochs, i.e. scale = max_epochs / total_steps with
        total_steps measured from the actual train dataloader, clipped to
        [TRANSFER_LR_FLOOR, 1.0].

        The schedule is also flattened (lrf = lr0, which optax
        exponential_decay clamps to a constant) so the whole step budget is
        spent at working LR - best-checkpoint selection and early stopping
        already guard against overshoot. Yes I know this is cheating since
        in a live case you wouldn't know when to stop, but it's a fair
        comparison to the other cases which also use early stopping.

        Applied after the tuned-config merge so the swept optimizer_config
        cannot overwrite it.
        """
        steps_per_epoch = self._transfer_steps_per_epoch(train_config)
        total_steps = steps_per_epoch * train_config.max_epochs
        scale = min(1.0, max(TRANSFER_LR_FLOOR, train_config.max_epochs / total_steps))
        logger.info(
            f"Transfer LR scale {scale:.3g} from step budget "
            f"({steps_per_epoch} steps/epoch x {train_config.max_epochs} epochs = {total_steps} steps)"
        )
        lr0_finetune = train_config.optimizer_config["lr0"] * scale
        return train_config.model_copy(
            update={
                "optimizer_config": {
                    **train_config.optimizer_config,
                    "lr0": lr0_finetune,
                    "lrf": lr0_finetune,
                }
            }
        )

    def make_train_config(self, case: Case) -> TrainConfig:
        """Make the TrainConfig for a given case.

        Builds the shared scaffold (dataloader/loss/optimizer bases, weighted
        device weights, transfer checkpoint wiring, tuned-config merge,
        transfer LR scaling) around the per-model-type pieces supplied by
        the _model_train_spec hook.
        """
        dataloader_config_base = self._base_dataloader_config(case)
        loss_config = self._base_loss_config()
        optimizer_config = self._base_optimizer_config()
        if case.domain_adaptation == "weighted":
            # Loss function reads these from loss_config as "device_weights".
            # val_eval_suite_config references the same dict, so validation
            # loss is weighted consistently with training.
            # "addition" adds the same target shots but as normal samples,
            # so it deliberately gets no device_weights entry
            loss_config["device_weights"] = self.make_weighted_device_weights(case)

        # Weighted / addition with no target shots runs to max_epochs, early stopping disabled
        patience = None if case.domain_adaptation in ("weighted", "addition") and case.num_target_shots == 0 else config.patience

        spec = self._model_train_spec(case, dataloader_config_base)
        train_config_base = TrainConfig(
            project=self.wandb_project_name(case),
            train_run_builder=spec.train_run_builder,
            max_epochs=config.max_epochs,
            epochs_per_val=config.epochs_per_val,
            patience=patience,
            # When doing hyperparameter tuning, this gets overwritten by the wandb agent
            checkpoint_dir=str(self.trained_model_dir(case)),
            dataloader_config=spec.dataloader_config,
            model_init_config=spec.model_init_config,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            val_eval_suite_config={"loss_config": loss_config},
            test_eval_suite_config={"result_path": str(self.result_path(case))},
        )

        if case.domain_adaptation == "transfer":
            # Point model_init at the pretrained checkpoint it fine-tunes from
            train_config_base = self._set_transfer_checkpoint(train_config_base, case.transfer_pretrain_case())

        train_config = self._apply_tuned_config(case, train_config_base)

        # Fine-tuning from a pretrained checkpoint gets a step-budgeted flat
        # learning rate instead of the swept schedule (see _scale_transfer_lr).
        # Applied after the tuned-config merge so the swept optimizer_config
        # cannot overwrite it
        if case.domain_adaptation == "transfer":
            train_config = self._scale_transfer_lr(train_config)

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
        tuned_loss = {key: tuned_config.loss_config.get(key, train_config_base.loss_config[key]) for key in self.TUNED_LOSS_KEYS}
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
                    **self._tuned_model_init_updates(case, tuned_config),
                }
            }
        )

    def _base_optimizer_config(self) -> dict:
        """Fallback optimizer hyperparameters for cases run without a tuned config."""
        return {
            "lr0": 5e-4,
            "transition_steps": 500,
            "decay_rate": 0.5,
            "lrf": 1e-4,
            "weight_decay": 2e-4,
        }

    def _base_dataloader_config(self, case: Case) -> dict:
        """Dataloader settings shared by every model type of this study."""
        raise NotImplementedError

    def _base_loss_config(self) -> dict:
        """Fallback loss hyperparameters for cases run without a tuned config."""
        raise NotImplementedError

    def _model_train_spec(self, case: Case, dataloader_config_base: dict) -> ModelTrainSpec:
        """Per-model-type train_run_builder, dataloader_config, and model_init_config."""
        raise NotImplementedError

    def _tuned_model_init_updates(self, case: Case, tuned_config: TrainConfig) -> dict:
        """model_init_config entries swept only for certain model types."""
        raise NotImplementedError

    def check_data_requirements(self, case: Case) -> bool:
        """Given a case, check if the required data for that case is available."""
        required = set(case.training_data.sources)
        if case.training_data.exnihilo or case.domain_adaptation in (
            "weighted",
            "addition",
            "transfer",
        ):
            required.add(config.target_device)

        missing = [ds for ds in required if ds not in config.dataset_paths]
        if missing:
            logger.warning(f"Case {case} is missing required datasets: {missing}. Skipping this case.")
            return False

        return True

    def get_unfinished_cases(self) -> list[Case]:
        unfinished = []
        for case in self.cases:
            if not self.result_path(case).exists():
                if not self.check_data_requirements(case):
                    logger.debug(f"Case {case} is missing required data, skipping.")
                    continue
                unfinished.append(case)
        return unfinished

    def case_in_flight(self, case: Case, running_job_names: set[str]) -> bool:
        """Whether a training job for this case is running or pending.

        Deliberately ignores sweep agent jobs: a hyperparam case with agents
        running may still need more agents launched (launch_sweep tops up to
        the remaining trial count), so it must stay eligible for pickup.
        """
        return self.train_job_name(case) in running_job_names

    def kill_stuck_jobs(self, cases: list[Case]):
        """Kill training jobs that are running but making no checkpoint progress.

        A job counts as deadlocked once it has run at least WATCHDOG_MIN_AGE_S
        with no checkpoint written in the last WATCHDOG_STALL_S (including one
        that never wrote a first checkpoint at all). Killing it frees the case
        to be relaunched as a fresh process by the normal orchestration loop;
        launch_train's attempt counter only resets on checkpoint progress, so a
        job that keeps deadlocking still hits MAX_TRAIN_ATTEMPTS and aborts the
        study rather than looping forever.
        """
        elapsed = get_running_job_elapsed_s()
        if not elapsed:
            return
        now = time.time()
        for case in cases:
            job_name = self.train_job_name(case)
            job_elapsed = elapsed.get(job_name)
            if job_elapsed is None or job_elapsed < WATCHDOG_MIN_AGE_S:
                continue
            info = self._latest_checkpoint_dir_info(case)
            last_progress = info[1] if info else None
            if last_progress is not None and now - last_progress < WATCHDOG_STALL_S:
                continue
            logger.warning(
                f"Job {job_name} has run {job_elapsed}s with no checkpoint progress in "
                f"the last {WATCHDOG_STALL_S}s, likely deadlocked. Killing so it can be resubmitted.\n"
                f"Case:\t{case}"
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
        for case in cases:
            if not case.is_hyperparam_case():
                continue
            job_name = self.agent_job_name(case)
            job_elapsed = elapsed.get(job_name)
            if job_elapsed is None or job_elapsed < WATCHDOG_MIN_AGE_S:
                continue
            if has_live_agent_run(self.wandb_project_name(case), AGENT_HEARTBEAT_STALL_S):
                continue
            logger.warning(
                f"Agent job {job_name} has run {job_elapsed}s with no actively heartbeating "
                f"wandb run, likely a zombied agent wrapper. Killing so launch_sweep can resubmit.\n"
                f"Case:\t{case}"
            )
            cancel_job(job_name)

    def _kill_long_pending_jobs(self, cases: list[Case]):
        """Cancel jobs stuck PENDING on a spillover partition longer than WATCHDOG_PENDING_S.

        A job submitted when a spillover partition looked free can pend
        indefinitely once other users grab the capacity. Cancelling it hands
        the case back to the normal launch path, where pick_partition
        re-decides placement. The primary partition is deliberately exempt:
        jobs there keep their queue position (and accrued age priority)
        instead of cycling to the back every 30 min. A cancelled pending job
        never started training, so its launch attempt is refunded, otherwise a
        busy partition alone could burn MAX_TRAIN_ATTEMPTS and abort the study
        without a single actual training failure. Agent jobs need no refund,
        launch_sweep tops agents back up to the remaining trial count on its own.
        """
        spillover = ",".join(p for p in config.spillover_partitions if p and p != config.partition)
        if not spillover:
            return
        pending = get_pending_job_pending_s(partition=spillover)
        if not pending:
            return
        for case in cases:
            job_names = [self.train_job_name(case)]
            if case.is_hyperparam_case():
                job_names.append(self.agent_job_name(case))
            for job_name in job_names:
                pending_s = pending.get(job_name)
                if pending_s is None or pending_s < WATCHDOG_PENDING_S:
                    continue
                logger.warning(
                    f"Job {job_name} has been pending {pending_s}s on a spillover partition, "
                    f"cancelling so the orchestration loop can re-queue it.\nCase:\t{case}"
                )
                # Scoped to the spillover partitions so a same-named job
                # pending on the primary partition can never be caught
                cancel_job(job_name, partition=spillover, state="PENDING")
                # scancel --state=PENDING is a no-op if the job started since
                # the squeue snapshot, so at worst this refund is one attempt
                # too generous and the case gets one extra retry
                if job_name == self.train_job_name(case):
                    attempts = self.train_attempts.get(str(case), 0)
                    self.train_attempts[str(case)] = max(attempts - 1, 0)

    def run_unfinished_cases(self, skip_tuning: bool, enable_parallelism: bool):
        """Loop until every runnable case has a result file.

        Each pass takes one squeue snapshot of this user's job names, then only
        picks up cases that are actually able to run: prereq results on disk
        and no job for the case already in flight. Blocked and in-flight cases
        are counted in a single per-pass summary line instead of being visited
        (and logged about) individually. Prereqs are themselves cases in
        self.cases, so a blocked case becomes runnable once its prereq case
        finishes; nothing needs to recurse into prereq chains here.

        A case whose job just left the queue is held for RELAUNCH_GRACE_S
        before it can be relaunched: its result file may already be written but
        not yet visible across nodes, and relaunching in that window submits a
        duplicate job for a finished case.
        """
        unfinished = self.get_unfinished_cases()
        # train job name -> monotonic time the job was last seen in the queue
        last_in_flight: dict[str, float] = {}
        while unfinished:
            if enable_parallelism:
                running_job_names = get_running_job_names()
                if running_job_names is None:
                    logger.warning("Could not query SLURM job state, waiting before trying again...")
                    time.sleep(ORCHESTRATION_POLL_INTERVAL_S)
                    continue
                self.kill_stuck_jobs(unfinished)
                self._kill_stuck_agents(unfinished)
                self._kill_long_pending_jobs(unfinished)
            else:
                running_job_names = set()

            runnable = [case for case in unfinished if self.check_prereq_satisfied(case)]
            now = time.monotonic()
            in_flight, in_grace, to_launch = [], [], []
            for case in runnable:
                if self.case_in_flight(case, running_job_names):
                    in_flight.append(case)
                    last_in_flight[self.train_job_name(case)] = now
                elif now - last_in_flight.get(self.train_job_name(case), -math.inf) < RELAUNCH_GRACE_S:
                    in_grace.append(case)
                else:
                    to_launch.append(case)
            n_blocked = len(unfinished) - len(runnable)
            logger.opt(colors=True).info(
                f"<bold><green>{len(unfinished)} cases remain</green></bold> "
                f"({len(in_flight)} in flight, {len(in_grace)} awaiting results, "
                f"{len(to_launch)} ready to launch, {n_blocked} blocked on prereqs)"
            )

            for case in to_launch:
                if enable_parallelism and pick_partition() is None:
                    logger.info(f"No idle resources or spillover budget, holding {len(to_launch)} ready cases until the next pass")
                    break
                self.run_case(case, skip_tuning=skip_tuning, enable_parallelism=enable_parallelism)

            # Sleep for a bit before checking again to avoid spamming slurm
            time.sleep(ORCHESTRATION_POLL_INTERVAL_S)
            unfinished = [case for case in unfinished if not self.result_path(case).exists()]

    def run_case(
        self,
        case: Case,
        skip_tuning: bool,
        enable_parallelism: bool,
    ):
        """Run a single case of the study, including hyperparameter tuning, training, and evaluation as needed.

        If case or a prereq is in progress, simply return and let orchestration loop try again later.
        """
        if not self.check_data_requirements(case):
            raise ValueError(f"Case {case} does not have the required data to run. This should have been caught earlier!")
        if self.result_path(case).exists():
            logger.warning(f"Case {case} already has results, skipping.")
            return
        if self.check_prereq_satisfied(case):
            self._run_ready_case(case, skip_tuning, enable_parallelism)
        else:
            logger.debug(f"Case blocked on unmet prereqs, skipping until they finish.\nCase:\t{case}")

    def _run_ready_case(self, case: Case, skip_tuning: bool, enable_parallelism: bool):
        """Execute a case whose prereqs are satisfied."""
        if case.is_hyperparam_case() and not self._ensure_hyperparams_ready(case, skip_tuning, enable_parallelism):
            return
        if not self._no_blocking_jobs(case, enable_parallelism):
            logger.debug(f"Jobs already in flight for case {case}, waiting before trying again...")
            return
        partition = pick_partition() if enable_parallelism else None
        if enable_parallelism and partition is None:
            logger.debug("No resources currently available, waiting before trying again...")
            return
        logger.opt(colors=True).info(f"<bold><cyan>RUNNING CASE:</cyan></bold>\n{case}")
        self.launch_train(case, enable_parallelism=enable_parallelism, partition=partition)

    def _ensure_hyperparams_ready(self, case: Case, skip_tuning: bool, enable_parallelism: bool) -> bool:
        """Ensure tuned config exists. Returns True if ready to proceed to training."""
        tuned_config_path = self.tuned_config_path(case)
        if tuned_config_path.exists():
            logger.debug(f"Tuned config found at {tuned_config_path}")
            return True
        if skip_tuning:
            logger.info(f"Skipping hyperparameter tuning, writing default config for {case}")
            self._write_tuned_config(case, self.make_train_config(case))
            return True
        completed_runs = get_completed_runs(self.wandb_project_name(case))
        finished_runs = [r for r in completed_runs if r.state == "finished"]
        min_finished = math.ceil(MIN_FINISHED_FRACTION * config.hyperparam_sweeps)
        if len(completed_runs) < config.hyperparam_sweeps or len(finished_runs) < min_finished:
            logger.info(
                f"Hyperparameter sweeps incomplete "
                f"({len(completed_runs)}/{config.hyperparam_sweeps} runs, "
                f"{len(finished_runs)}/{min_finished} finished)"
            )
            partition = pick_partition() if enable_parallelism else None
            if enable_parallelism and partition is None:
                logger.debug("No resources currently available for sweep agents, waiting before trying again...")
                return False
            self.launch_sweep(
                case,
                enable_parallelism=enable_parallelism,
                n_completed_runs=len(completed_runs),
                n_finished_runs=len(finished_runs),
                partition=partition,
            )
            return False
        return self._finalize_sweep(case, enable_parallelism, completed_runs)

    def _finalize_sweep(self, case: Case, enable_parallelism: bool, completed_runs: list) -> bool:
        """Save best config once sweep runs are done. Returns True if ready."""
        logger.info(f"Hyperparameter sweeps completed with {len(completed_runs)}/{config.hyperparam_sweeps} runs")
        if enable_parallelism:
            # Sweep trials run inside agent jobs, so poll the agent job name.
            # Picking the best config while agents still run would ignore
            # their trials (and the last trials are often the best ones)
            running_jobs = count_running_jobs(self.agent_job_name(case))
            if running_jobs > 0:
                logger.info(f"Found {running_jobs} running agent jobs, waiting for them to complete before proceeding")
                return False
        best_train_config = get_best_train_config(self.wandb_project_name(case))
        if best_train_config is None:
            raise RuntimeError(
                f"Hyperparameter sweep for case {case} reports {len(completed_runs)} completed runs "
                f"but no best config could be recovered from wandb project {self.wandb_project_name(case)}. "
                "Only 'finished' runs are eligible, check the project for runs stuck crashing/pruning "
                "before logging val/loss.mean."
            )
        self._write_tuned_config(case, best_train_config)
        logger.success(f"Saved best hyperparameter config for {case}")
        return True

    def _write_tuned_config(self, case: Case, train_config: TrainConfig):
        """Write a train config to the tuned config path for the given case."""
        tuned_config_path = self.tuned_config_path(case)
        tuned_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tuned_config_path, "w") as f:
            yaml.dump(train_config.model_dump(), f, indent=4)

    def _no_blocking_jobs(self, case: Case, enable_parallelism: bool) -> bool:
        """Returns True if no in-flight SLURM jobs are blocking training."""
        if not enable_parallelism:
            return True
        for job_name, label in [
            (self.agent_job_name(case), "agent"),
            (self.train_job_name(case), "training"),
        ]:
            running = count_running_jobs(job_name)
            if running > 0:
                logger.info(f"Found {running} running {label} jobs, waiting for them to complete before proceeding")
                return False
        return True

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
        still outstanding, capped by idle GPUs. Outstanding is the larger of the
        two remaining targets (both net of agents already running): trials to
        reach the completed-count target (hyperparam_sweeps - completed) and
        trials to reach the finished-run quota (min_finished - finished). Once
        the count target is met but too many trials have crashed to satisfy the
        finished quota, the second term keeps agents flowing. Without the
        outstanding cap the orchestration loop would submit another batch of
        agents every pass until the queue filled, far past what is needed.

        When partition is a spillover partition the idle-GPU cap is replaced by
        that partition's remaining per-user slots (its QOS GPU allowance minus
        jobs already there), bounded by the user-wide job ceiling.
        """
        train_config = self.make_train_config(case)
        # Remove the test_eval_suite_config since that's for final results only.
        # Trials get the same wall-clock budget as production jobs, making the
        # sweep an anytime comparison: best val loss reachable within one job.
        # Resume stays off, a trial is a fresh sample of its hyperparameters.
        train_config = train_config.model_copy(
            update={
                "test_eval_suite_config": None,
                "max_epochs": min(config.max_epochs, config.hyperparam_max_epochs),
                "resume": False,
                "max_wall_seconds": float(config.train_wall_budget_s),
                # Sweep trials are compared on val loss only, no need to keep
                # the top-K checkpoints production runs retain for results
                "checkpoint_max_to_keep": 1,
            }
        )
        wandb_project_name = self.wandb_project_name(case)
        sweep_id = get_sweep_id(wandb_project_name)
        kwargs_agent = {"count": 1}  # One training run per agent

        if not sweep_id:
            logger.info(f"No existing sweep found for case {case}, creating a new sweep")
            sweep_config_path = Path(self.SWEEP_CONFIG_DIR) / f"{case.model_type}.yaml"
            sweep_config = load_dict(str(sweep_config_path))
            sweep_id = wandb.sweep(sweep_config, project=wandb_project_name)

        if enable_parallelism:
            if partition is None:
                partition = config.partition
            agent_job_name = self.agent_job_name(case)
            running_agents = count_running_jobs(agent_job_name)
            # Each agent runs exactly one trial (count=1), so outstanding trials
            # bound how many more agents are worth submitting. Take the larger of
            # the count target and the finished-quota target so a sweep whose
            # count target is met but whose trials keep crashing keeps launching
            # until enough runs finish (each in-flight agent may yet finish).
            min_finished = math.ceil(MIN_FINISHED_FRACTION * config.hyperparam_sweeps)
            outstanding_for_count = config.hyperparam_sweeps - n_completed_runs - running_agents
            outstanding_for_finished = min_finished - n_finished_runs - running_agents
            outstanding_trials = max(outstanding_for_count, outstanding_for_finished)
            if partition == config.partition:
                capacity = count_idle_gpus(config.partition, config.buffer_gpus)
            else:
                capacity = min(spillover_budget(), spillover_slots(partition))
            # Always have at least a couple agents going (up to capacity) to finish out the sweep
            # Want to avoid launching only one at a time which may get pruned
            sweep_jobs = min(capacity, max(outstanding_trials, 6 - running_agents))
            if sweep_jobs <= 0:
                logger.debug(f"No agent jobs needed for case {case} ({running_agents} agents already running)")
                return
            logger.info(f"Launching {sweep_jobs} agent job(s) {agent_job_name} on {partition} for case\n{case}")
            for _ in range(sweep_jobs):
                launch_agent_parallel(
                    train_config,
                    sweep_id,
                    kwargs_agent,
                    agent_job_name,
                    Path(self.log_dir) / "logs_sweep",
                    partition=partition,
                )
        else:
            logger.info("Launching agent serially")
            launch_agent(train_config, sweep_id, kwargs_agent=kwargs_agent)

    def launch_train(self, case: Case, enable_parallelism: bool = False, partition: str | None = None):
        """Launch a training job for the given case."""
        if case.is_impossible():
            raise ValueError(
                f"Case {case} is not a possible case to run, check the logic in the Case dataclass to see why this is. This should have been caught earlier!"
            )

        # Only reached when no result file exists and no job for this case is
        # running (see _no_blocking_jobs), so every call is a fresh (re)launch.
        # More than MAX_TRAIN_ATTEMPTS launches means the case fails every time.
        # Exception: a relaunch whose latest checkpoint advanced since the last
        # launch is a resume of a wall-clock-limited job making real progress,
        # so the attempt counter resets rather than counting toward the cap
        attempts = self.train_attempts.get(str(case), 0)
        latest_epoch = self.latest_checkpoint_epoch(case)
        prev_epoch = self.train_attempt_epochs.get(str(case))
        if latest_epoch is not None and (prev_epoch is None or latest_epoch > prev_epoch):
            attempts = 0
        self.train_attempt_epochs[str(case)] = latest_epoch
        if attempts >= MAX_TRAIN_ATTEMPTS:
            train_log_path = Path(self.log_dir) / "logs_train" / f"{self.train_job_name(case)}.log"
            summary = (
                f"ABORTING STUDY: case failed training {attempts} times without producing a result.\n"
                f"Case:      {case}\n"
                f"Job name:  {self.train_job_name(case)}\n"
                f"Train log: {train_log_path}\n"
                f"(attempts append to the same log, separated by '=== ... job <id> start ===' lines)\n"
                f"Other unfinished cases were not attempted further. Fix the case or remove it, then rerun."
            )
            logger.critical(summary)
            raise RuntimeError(summary)
        self.train_attempts[str(case)] = attempts + 1

        logger.opt(colors=True).info(f"<bold><red>LAUNCHING TRAINING for case\n{case}</red></bold>")

        train_config = self.make_train_config(case)
        # Real training runs resume from the latest checkpoint if one exists and
        # stop cleanly at the wall-clock budget so the next launch can continue.
        # They keep the top num_result_checkpoints checkpoints, whose test
        # metrics the final result averages (see orchestration/topk_results.py)
        train_config = train_config.model_copy(
            update={
                "resume": True,
                "max_wall_seconds": float(config.train_wall_budget_s),
                "checkpoint_max_to_keep": config.num_result_checkpoints,
            }
        )
        result_path = self.result_path(case)
        if enable_parallelism:
            train_job_name = self.train_job_name(case)
            logger.info(f"Launching training job {train_job_name} for case\n{case}")
            launch_train_parallel(
                train_config,
                train_job_name,
                result_path,
                Path(self.log_dir) / "logs_train",
                partition=partition,
            )
        else:
            logger.info("Launching training serially")
            trainer, _, _, test_dl, result_dict = launch_train(train_config)
            if result_dict is None:
                logger.info("Training stopped at the wall-clock budget before finishing, relaunch to resume from the latest checkpoint.")
                return
            ds = compute_topk_study_results(trainer, test_dl, train_config, result_dict)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp name then rename so a partially written file is
            # never visible at the result path, whose existence marks the case done
            tmp_path = result_path.with_name(result_path.name + ".tmp")
            ds.to_netcdf(tmp_path)
            tmp_path.replace(result_path)

    ##############
    # COLLECTION #
    ##############
    @staticmethod
    def _summarize_case_errors(ds: xr.Dataset) -> xr.Dataset:
        """Reduce a case's result dataset to scalar error statistics.

        Reads the four error variables every study's test eval suite writes
        (error_{abs,rel}_{shot,ts}) and emits err_E_D_S scalars where E is
        'abs' or 'rel', D is 'shot' or 'ts', and S is one of mean, std, med,
        p25, p75, min, max.

        The per-timeslice stats only count timeslices that advance the shot
        clock: the padded rollout tail repeats each shot's final timeslice
        (roughly 40 percent of the array entries in practice), which would
        otherwise weight the per-ts stats heavily toward shot-end error. The
        per-shot integrals need no masking because the padded repeats have
        near-zero dt and contribute nothing to the trapezoid.
        """
        err_abs_shot = ds["error_abs_shot"]
        err_rel_shot = ds["error_rel_shot"]
        real = real_timeslice_mask(ds["time"])
        err_abs_ts = ds["error_abs_ts"].where(real)
        err_rel_ts = ds["error_rel_ts"].where(real)

        return xr.Dataset(
            {
                "err_abs_shot_mean": err_abs_shot.mean(),
                "err_abs_shot_std": err_abs_shot.std(),
                "err_abs_shot_med": err_abs_shot.median(),
                "err_abs_shot_p25": err_abs_shot.quantile(0.25).drop_vars("quantile"),
                "err_abs_shot_p75": err_abs_shot.quantile(0.75).drop_vars("quantile"),
                "err_abs_shot_min": err_abs_shot.min(),
                "err_abs_shot_max": err_abs_shot.max(),
                "err_rel_shot_mean": err_rel_shot.mean(),
                "err_rel_shot_std": err_rel_shot.std(),
                "err_rel_shot_med": err_rel_shot.median(),
                "err_rel_shot_p25": err_rel_shot.quantile(0.25).drop_vars("quantile"),
                "err_rel_shot_p75": err_rel_shot.quantile(0.75).drop_vars("quantile"),
                "err_rel_shot_min": err_rel_shot.min(),
                "err_rel_shot_max": err_rel_shot.max(),
                "err_abs_ts_mean": err_abs_ts.mean(),
                "err_abs_ts_std": err_abs_ts.std(),
                "err_abs_ts_med": err_abs_ts.median(),
                "err_abs_ts_p25": err_abs_ts.quantile(0.25).drop_vars("quantile"),
                "err_abs_ts_p75": err_abs_ts.quantile(0.75).drop_vars("quantile"),
                "err_abs_ts_min": err_abs_ts.min(),
                "err_abs_ts_max": err_abs_ts.max(),
                "err_rel_ts_mean": err_rel_ts.mean(),
                "err_rel_ts_std": err_rel_ts.std(),
                "err_rel_ts_med": err_rel_ts.median(),
                "err_rel_ts_p25": err_rel_ts.quantile(0.25).drop_vars("quantile"),
                "err_rel_ts_p75": err_rel_ts.quantile(0.75).drop_vars("quantile"),
                "err_rel_ts_min": err_rel_ts.min(),
                "err_rel_ts_max": err_rel_ts.max(),
            }
        )

    def restore_trainer(self, case: Case, restore_best_checkpoint: bool = True) -> tuple[Trainer, DataLoader]:
        """Restore a given case's trainer and the test dataloader"""
        if not self.trained_model_dir(case).exists():
            raise ValueError(f"Trained model directory for case\n{case}\nnot found at\n{self.trained_model_dir(case)}")
        if not self.result_path(case).exists():
            logger.warning(f"Result file for case\n{case}\nnot found at\n{self.result_path(case)}\nTraining may be incomplete!")
        training_config = self.make_train_config(case)
        train_run_builder = get_train_run_builder_class(training_config.train_run_builder)
        # If this is a submodule, use the dataloader construction logic from the main module. Fallback to using the submodule's own logic otherwise.
        if training_config.dataloader_config.get("data_train_run_builder"):
            data_train_run_builder = get_train_run_builder_class(training_config.dataloader_config["data_train_run_builder"])
            _, train_dl, _val_dl, test_dl = data_train_run_builder.get_dataloaders(training_config.dataloader_config)
        else:
            _, train_dl, _val_dl, test_dl = train_run_builder.get_dataloaders(training_config.dataloader_config)
        model = train_run_builder.model_init(train_dl, training_config.model_init_config)
        loss_fn = train_run_builder.get_loss_fn(training_config.loss_config)
        # Tuned configs sweep transition_frac, which launch_train resolves to
        # transition_steps before get_optimizer, so mirror that here
        optimizer_config = resolve_transition_frac(training_config.optimizer_config, len(train_dl), training_config.max_epochs)
        opt = train_run_builder.get_optimizer(optimizer_config)
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=opt,
            checkpoint_dir=training_config.checkpoint_dir,
            trainable_getter=train_run_builder.get_trainable_getter(training_config.model_init_config),
        )
        if restore_best_checkpoint:
            trainer.restore_best_checkpoint(path=self.trained_model_dir(case))

        return trainer, test_dl

    # Coords describing which case a record belongs to, set per subclass
    _CASE_COORD_NAMES: ClassVar[tuple[str, ...]] = ()

    def case_coords(self, case_idx: int, case: Case) -> dict:
        """Build the per-case coordinate values for collect_results."""
        coords = {}
        for name in self._CASE_COORD_NAMES:
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

        Default implementation for the scalar-summary studies (power balance,
        transport); the profile study overrides with a per-shot long form.

        Dims: case_idx
        Coords (along case_idx): the _CASE_COORD_NAMES fields of each case
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

            result = self._summarize_case_errors(ds).assign_coords(self.case_coords(case_idx, case))
            results.append(result)

        if not results:
            logger.warning("No case results found to collect!")
            return xr.Dataset()

        # coords="different" stacks the per-case scalar coords along case_idx.
        # compat pinned explicitly, the xarray default is changing to
        # "override" which is incompatible with coords="different"
        return xr.concat(results, dim="case_idx", coords="different", compat="equals")

    def _run_analysis(self, enable_parallelism: bool) -> None:
        """Post-orchestration analysis and plotting (study-specific)."""
        raise NotImplementedError

    def _visualize_data(self):
        self.DATA_VISUALIZATION.hazard_extrapolation(self.figure_dir)
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

        Requires specifying paths to the source datasets in the config TOML or environment variables.
        Due to data sharing restrictions, the only dataset included in this repository is for C-Mod.
        If you have access to data from other tokamaks (e.g. DIII-D), create a source dataset using the scripts in `transport_study/datasets/`
        and provide the path when running this script.
        If a dataset is not provided for a tokamak, figures which require that data will be skipped.

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
        clean_results : bool | None
            If True, delete any existing intermediate results in the working directory before running.
        clean_figures : bool | None
            If True, delete any existing figures in the figure directory before running.
        """
        # Must run before anything touches jax so the backend is still unset
        configure_jax_platforms(bool(enable_parallelism))

        # Parse with the concrete Config subclass so the local `config` name holds a
        # real config object (it shadows the module-level proxy)
        if isinstance(config, (str, Path)):
            config = cls.Config.from_toml(Path(config))
        study = cls(config)
        study.setup_directories(
            enable_parallelism=enable_parallelism,
            skip_tuning=skip_tuning,
            skip_visualization=skip_visualization,
            clean_sweeps=clean_sweeps,
            clean_models=clean_models,
            clean_results=clean_results,
            clean_figures=clean_figures,
        )

        if enable_parallelism and not config.partition:
            raise ValueError("enable_parallelism is True but no SLURM partition is specified in the config")

        if not skip_visualization:
            logger.opt(colors=True).info("<bold><magenta>DATA VISUALIZATION</magenta></bold>")
            study._visualize_data()

        if study.collected_results_path().exists():
            logger.info(
                f"Collected results file found at\n{study.collected_results_path()}\nSkipping orchestration and going straight to analysis"
            )
        else:
            logger.opt(colors=True).info("<bold><magenta>ORCHESTRATION</magenta></bold>")
            study.run_unfinished_cases(skip_tuning=skip_tuning, enable_parallelism=enable_parallelism)
            ds_final = study.collect_results()
            ds_final.to_netcdf(study.collected_results_path())

        study._run_analysis(enable_parallelism=bool(enable_parallelism))

    def __init__(self, cfg: str | Path | CaseGridConfig):
        """
        Initialize this study from its Config object or a path to its TOML file.
        Dataset paths come from the global config's dataset_paths
        (TOML [datasets] table, with the PTPS_DATASET_PATHS JSON env var as defaults).
        """
        if not config.initialized:
            # load_config(Path) only knows how to build the base StudyConfig,
            # which forbids this study's extra fields, so parse with the subclass
            if isinstance(cfg, (str, Path)):
                cfg = self.Config.from_toml(Path(cfg))
            load_config(cfg)

        self.name = config.study_name
        self.cases = self.make_cases()
        # Launch counter per case (str(case) -> count) backing MAX_TRAIN_ATTEMPTS
        self.train_attempts: dict[str, int] = {}
        # Latest-checkpoint epoch per case as of its last launch. A relaunch whose
        # checkpoint advanced past this is a resume making progress, not a failure
        self.train_attempt_epochs: dict[str, int | None] = {}
        # Measured steps-per-epoch per transfer dataloader config, so repeated
        # make_train_config calls do not rebuild dataloaders (see _transfer_steps_per_epoch)
        self._transfer_steps_cache: dict[str, int] = {}

        self.working_dir = Path(config.working_dir_base) / self.name
        self.model_dir = self.working_dir / "models"
        self.result_dir = self.working_dir / "results"
        self.figure_dir = self.working_dir / "figures"
        self.log_dir = self.working_dir / "logs"

        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Save config on first run, and on subsequent runs check for changes.
        # This locks in the state so on subsequent runs if the dataset paths or
        # target device changes, we'll get an error instead of silently wrong results
        config_path = self.working_dir / CONFIG_LOCK_FILENAME
        if config_path.exists():
            saved = config.from_toml(config_path)
            if not config.is_compatible(saved):
                raise RuntimeError(
                    f"Dataset config changed since study was created.\n"
                    f"Saved:   {saved}\n"
                    f"Current: {config}\n"
                    f"Delete {config_path} to reset (will invalidate existing results)."
                )
        else:
            config.save(config_path)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{os.getpid()}_run_study.log"
        logger.add(log_path)

        logger.info("INITIALIZING STUDY")
        logger.info(f"Study name: {self.name}")
        logger.info(f"Working directory base: {config.working_dir_base}")
        logger.info(f"Total number of cases: {len(self.cases)}")
        logger.info(f"Target test set size: {config.target_test_set_size}")
        logger.info(f"Dataset paths: {config.dataset_paths}")
        logger.info(f"Target device: {config.target_device}")
        for axis_field in self.CASE_AXIS_FIELDS:
            logger.info(f"{axis_field}: {getattr(config, axis_field)}")
