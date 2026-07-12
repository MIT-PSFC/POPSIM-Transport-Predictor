import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger
from popsim.ml import DataLoader, TrainConfig, Trainer
from popsim.ml.launch import (
    _get_train_run_builder_class,
    launch_agent,
    launch_train,
)
from popsim.ml.train_config import load_dict

import wandb
from transport_study import PACKAGE_ROOT
from transport_study.config import config
from transport_study.orchestration.slurm_utils import (
    count_idle_gpus,
    count_running_jobs,
    get_running_job_names,
    launch_agent_parallel,
    launch_train_parallel,
    pick_partition,
    spillover_budget,
    spillover_slots,
)
from transport_study.orchestration.wandb_utils import (
    get_best_train_config,
    get_completed_runs,
    get_sweep_id,
    run_clean_sweeps,
)

CONFIG_LOCK_FILENAME = "config_lock.toml"

# A training job that fails deterministically (e.g. NaN loss) would otherwise be
# resubmitted forever by the run_study orchestration loop, silently burning GPU
# time ~10 min at a go. Abort the whole run once a single case has been launched
# this many times without producing a result file
MAX_TRAIN_ATTEMPTS = 3

# A hyperparam sweep is only done once the trial-count target is met AND this
# fraction of the target has runs in the "finished" state. Crashed/pruned trials
# still count toward the trial target, but the tuned config is picked only from
# finished runs (get_best_train_config), so the sweep must produce enough of them
# or there is nothing safe to select. Guards against a sweep where most trials
# diverge to NaN and crash.
MIN_FINISHED_FRACTION = 0.1

# How long the orchestration loop sleeps between passes over the unfinished cases
ORCHESTRATION_POLL_INTERVAL_S = 8


class Study:
    """A class for organizing various components of a study, essentially outlining everything that needs to be done
    to go from raw data to comparison figures.
    - Paths to source data
    - Model checkpoints
    - Results
    """

    @dataclass
    class Case:
        """A class for organizing the different cases we want to compare in this study.
        For example, different treatments of the transport predictor module, different training datasets, different normalization methods, etc.
        Each case should have all the information needed to train and evaluate a model for that case, and to compare it to other cases.
        """

        def __hash__(self):
            return hash(tuple(v for k, v in self.__dict__.items() if k not in ("prereq", "weight_submodules")))

    ####################
    # PATHING / NAMING #
    ####################
    def trained_model_dir(self, case: Case) -> Path:
        """Given a case, return the path where the trained model checkpoints for that case should be stored."""
        return Path(self.model_dir) / str(case)

    def result_path(self, case: Case) -> Path:
        """Given a case, return the path where the results for that case should be stored."""
        return Path(self.result_dir) / str(case) / "result_data.nc"

    def latest_checkpoint_epoch(self, case: Case) -> int | None:
        """Highest epoch saved in the case's latest-checkpoint (resume) directory, or None if empty.

        Orbax names each checkpoint directory after its step (here the epoch).
        In-progress saves get a non-numeric tmp suffix and are skipped.
        """
        latest_dir = Path(f"{self.trained_model_dir(case)}_latest")
        if not latest_dir.exists():
            return None
        epochs = [int(p.name) for p in latest_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        return max(epochs, default=None)

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

    def sweep_job_name(self, case: Case) -> str:
        return f"sweep_{case}"

    def agent_job_name(self, case: Case) -> str:
        return f"agent_{case}"

    def train_job_name(self, case: Case) -> str:
        return f"train_{case}"

    def analysis_job_name(self, case: Case) -> str:
        return f"analysis_{case}"

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
        clean_figures: bool = True,
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

    #############
    # EXECUTION #
    #############
    def check_data_requirements(self, case: Case) -> bool:
        """Given a case, check if the required data for that case is available."""
        required = set(case.training_data.sources)
        if case.training_data.exnihilo or case.domain_adaptation in (
            "mixing",
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

    def run_unfinished_cases(self, skip_tuning: bool, enable_parallelism: bool):
        """Loop until every runnable case has a result file.

        Each pass takes one squeue snapshot of this user's job names, then only
        picks up cases that are actually able to run: prereq results on disk
        and no job for the case already in flight. Blocked and in-flight cases
        are counted in a single per-pass summary line instead of being visited
        (and logged about) individually. Prereqs are themselves cases in
        self.cases, so a blocked case becomes runnable once its prereq case
        finishes; nothing needs to recurse into prereq chains here.
        """
        unfinished = self.get_unfinished_cases()
        while unfinished:
            if enable_parallelism:
                running_job_names = get_running_job_names()
                if running_job_names is None:
                    logger.warning("Could not query SLURM job state, waiting before trying again...")
                    time.sleep(ORCHESTRATION_POLL_INTERVAL_S)
                    continue
            else:
                running_job_names = set()

            runnable = [case for case in unfinished if self.check_prereq_satisfied(case)]
            in_flight = [case for case in runnable if self.case_in_flight(case, running_job_names)]
            to_launch = [case for case in runnable if not self.case_in_flight(case, running_job_names)]
            n_blocked = len(unfinished) - len(runnable)
            logger.opt(colors=True).info(
                f"<bold><green>{len(unfinished)} cases remain</green></bold> "
                f"({len(in_flight)} in flight, {len(to_launch)} ready to launch, {n_blocked} blocked on prereqs)"
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
            self._run_first_unmet_prereq(case, skip_tuning, enable_parallelism)

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

    def _run_first_unmet_prereq(self, case: Case, skip_tuning: bool, enable_parallelism: bool):
        """Find the first unsatisfied prereq and recurse into it."""
        for prereq in case.prereqs:
            if not self.result_path(prereq).exists():
                logger.debug(f"Prereq not satisfied yet, running that first.\nCase:\t{case}\nPrereq:\t{prereq}")
                self.run_case(
                    prereq,
                    skip_tuning=skip_tuning,
                    enable_parallelism=enable_parallelism,
                )
                return

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
            }
        )
        wandb_project_name = self.wandb_project_name(case)
        sweep_id = get_sweep_id(wandb_project_name)
        kwargs_agent = {"count": 1}  # One training run per agent

        if not sweep_id:
            logger.info(f"No existing sweep found for case {case}, creating a new sweep")
            sweep_config_path = Path(PACKAGE_ROOT) / "profile_transfer" / "sweep_configs" / f"{case.model_type}.yaml"
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
            sweep_jobs = min(capacity, max(outstanding_trials, 0))
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
        # Sweep trials get neither (see launch_sweep)
        train_config = train_config.model_copy(
            update={
                "resume": True,
                "max_wall_seconds": float(config.train_wall_budget_s),
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
            _, _, _, _, result_dict = launch_train(train_config)
            if result_dict is None:
                logger.info("Training stopped at the wall-clock budget before finishing, relaunch to resume from the latest checkpoint.")
                return
            ds = result_dict["test/study_results"]
            result_path.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(result_path)

    ##############
    # COLLECTION #
    ##############
    def restore_trainer(self, case: Case, restore_best_checkpoint: bool = True) -> tuple[Trainer, DataLoader]:
        """Restore a given case's trainer and the test dataloader"""
        if not self.trained_model_dir(case).exists():
            raise ValueError(f"Trained model directory for case\n{case}\nnot found at\n{self.trained_model_dir(case)}")
        if not self.result_path(case).exists():
            logger.warning(f"Result file for case\n{case}\nnot found at\n{self.result_path(case)}\nTraining may be incomplete!")
        training_config = self.make_train_config(case)
        train_run_builder = _get_train_run_builder_class(training_config.train_run_builder)
        # If this is a submodule, use the dataloader construction logic from the main module. Fallback to using the submodule's own logic otherwise.
        if training_config.dataloader_config.get("data_train_run_builder"):
            data_train_run_builder = _get_train_run_builder_class(training_config.dataloader_config["data_train_run_builder"])
            _, train_dl, _val_dl, test_dl = data_train_run_builder.get_dataloaders(training_config.dataloader_config)
        else:
            _, train_dl, _val_dl, test_dl = train_run_builder.get_dataloaders(training_config.dataloader_config)
        model = train_run_builder.model_init(train_dl, training_config.model_init_config)
        loss_fn = train_run_builder.get_loss_fn(training_config.loss_config)
        opt = train_run_builder.get_optimizer(training_config.optimizer_config)
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

    def __init__(
        self,
        name: str,
        working_dir_base: Path | str,
        cases: list[Case],
    ):
        """
        Initialize this study with the given name and cases.
        Dataset paths come from the global config's dataset_paths
        (TOML [datasets] table, with the PTPS_DATASET_PATHS JSON env var as defaults).
        """
        self.name = name
        self.cases = cases
        # Launch counter per case (str(case) -> count) backing MAX_TRAIN_ATTEMPTS
        self.train_attempts: dict[str, int] = {}
        # Latest-checkpoint epoch per case as of its last launch. A relaunch whose
        # checkpoint advanced past this is a resume making progress, not a failure
        self.train_attempt_epochs: dict[str, int | None] = {}

        self.working_dir = Path(working_dir_base) / name
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
        logger.info(f"Study name: {name}")
        logger.info(f"Working directory base: {working_dir_base}")
        logger.info(f"Total number of cases: {len(cases)}")
        logger.info(f"Target test set size: {config.target_test_set_size}")
        logger.info(f"Dataset paths: {config.dataset_paths}")
        logger.info(f"Target device: {config.target_device}")


def update_submodule_configs(main_config: dict, submodules: list[str]) -> TrainConfig:
    new_submodule_configs = {}
    for submodule in submodules:
        submodule_config = main_config["model_init_config"]["submodules"][submodule]
        if not isinstance(submodule_config, dict):
            submodule_config = submodule_config.model_dump()

        # Set the data_train_run_builder for the submodules to match the main module's train_run_builder.
        submodule_config["dataloader_config"]["data_train_run_builder"] = main_config["train_run_builder"]

        # Ensure there is a perfect match between the dataloader configs of the main module and the submodules,
        # excepting the state_vars, input_vars, target_vars, and extra_vars which are specific to each submodule.
        for key in main_config["dataloader_config"].keys():
            if key not in ["state_vars", "input_vars", "target_vars", "extra_vars"]:
                submodule_config["dataloader_config"][key] = main_config["dataloader_config"][key]

        new_submodule_configs[submodule] = submodule_config

    main_config["model_init_config"]["submodules"] = new_submodule_configs

    return TrainConfig(**main_config)
