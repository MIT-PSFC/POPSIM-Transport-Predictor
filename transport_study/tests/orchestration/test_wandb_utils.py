"""The tuned config comes from the sweep trial with the lowest val loss it ever logged, among trials trained to completion."""

from popsim.ml.loggers import STOP_REQUESTED_SUMMARY_KEY

from transport_study.orchestration import wandb_utils
from transport_study.orchestration.wandb_utils import (
    SWEEP_METRIC,
    get_best_train_config,
)


class FakeRun:
    """A wandb run whose summary holds the last logged val loss, as wandb's does."""

    def __init__(self, run_id: str, history: list, state: str = "finished", pruned: bool = False):
        self.id = run_id
        self.name = run_id
        self.state = state
        self.config = {"run_id": run_id}
        self.summary = {SWEEP_METRIC: history[-1], STOP_REQUESTED_SUMMARY_KEY: pruned}
        self.history = history

    def scan_history(self, keys: list[str]) -> list[dict]:
        return [{SWEEP_METRIC: value} for value in self.history]


def test_best_trial_ranks_by_history_minimum(monkeypatch):
    # Early stopped past its best: last 0.4, minimum 0.2
    run_overfit = FakeRun("overfit", [0.5, 0.2, 0.3, 0.4])
    run_steady = FakeRun("steady", [0.35, 0.3])
    monkeypatch.setattr(wandb_utils.TrainConfig, "load", lambda run_config: run_config)
    best_config = get_best_train_config([run_steady, run_overfit])
    assert best_config["run_id"] == "overfit"


def test_only_finite_trials_trained_to_completion_rank(monkeypatch):
    run_pruned = FakeRun("pruned", [0.01], pruned=True)
    run_crashed = FakeRun("crashed", [0.01], state="crashed")
    run_diverged = FakeRun("diverged", ["NaN", float("nan")])
    # A NaN logged after the best epoch must not hide the minimum
    run_eligible = FakeRun("eligible", [0.6, 0.5, "NaN"])
    monkeypatch.setattr(wandb_utils.TrainConfig, "load", lambda run_config: run_config)
    best_config = get_best_train_config([run_pruned, run_crashed, run_diverged, run_eligible])
    assert best_config["run_id"] == "eligible"
    assert get_best_train_config([run_pruned, run_crashed, run_diverged]) is None
