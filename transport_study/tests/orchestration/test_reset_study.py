"""Study.reset_study: refused under a queued job of the study, and the old working dir survives a failed lineage check."""

import pytest

from transport_study.config import reset_config
from transport_study.orchestration import study as study_module
from transport_study.orchestration.config_lock import CONFIG_LOCK_FILENAME
from transport_study.orchestration.study import orchestrator_job_name
from transport_study.power_balance_transfer.power_balance_study import PowerBalanceStudy

DATASET_PATHS = {"cmod-low": "path/to/cmod_low.nc", "cmod-high": "path/to/cmod_high.nc"}


def _config(tmp_path, **overrides) -> PowerBalanceStudy.Config:
    defaults = dict(
        study_name="test_reset",
        dataset_paths=dict(DATASET_PATHS),
        target_device="cmod-high",
        working_dir_base=tmp_path / "working_dir_base",
        training_datasets=("cmod-low",),
        model_types=("mlp",),
        domain_adaptation_methods=("none",),
        num_target_shots_options=(0,),
        target_test_set_size=60,
        partition="fake_partition",
        # No wandb entity, so the reset skips the sweep cleanup
        wandb_entity=None,
    )
    defaults.update(overrides)
    return PowerBalanceStudy.Config(**defaults)


def _stale_working_dir(cfg):
    working_dir = cfg.working_dir_base / cfg.study_name
    (working_dir / "results").mkdir(parents=True)
    (working_dir / "results" / "stale.nc").touch()
    return working_dir


@pytest.mark.parametrize("queued", ["train_case.x.seed_0.test_reset", orchestrator_job_name("test_reset")])
def test_reset_refuses_queued_job_of_study(tmp_path, monkeypatch, queued):
    cfg = _config(tmp_path)
    working_dir = _stale_working_dir(cfg)
    monkeypatch.setattr(study_module, "get_running_job_names", lambda: {queued, "train_case.x.seed_0.other_study"})
    with pytest.raises(RuntimeError, match="queued or running"):
        PowerBalanceStudy.reset_study(cfg)
    assert (working_dir / "results" / "stale.nc").exists()


def test_reset_replaces_working_dir_and_writes_fresh_lock(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    working_dir = _stale_working_dir(cfg)
    monkeypatch.setattr(study_module, "get_running_job_names", lambda: {"train_case.x.seed_0.other_study"})
    reset_config()
    PowerBalanceStudy.reset_study(cfg)
    assert not (working_dir / "results" / "stale.nc").exists()
    assert (working_dir / CONFIG_LOCK_FILENAME).exists()
    assert not working_dir.with_name(f"{cfg.study_name}{study_module.PREVIOUS_WORKING_DIR_SUFFIX}").exists()


def test_reset_keeps_working_dir_when_parent_missing(tmp_path, monkeypatch):
    """A child reset before its parent would otherwise wipe itself and then fail the parent check."""
    cfg = _config(tmp_path, parent_study="never_run_parent")
    working_dir = _stale_working_dir(cfg)
    monkeypatch.setattr(study_module, "get_running_job_names", lambda: set())
    reset_config()
    with pytest.raises(RuntimeError, match="no config lock"):
        PowerBalanceStudy.reset_study(cfg)
    assert (working_dir / "results" / "stale.nc").exists()
    assert not (working_dir / CONFIG_LOCK_FILENAME).exists()
    assert not working_dir.with_name(f"{cfg.study_name}{study_module.PREVIOUS_WORKING_DIR_SUFFIX}").exists()


def test_reset_restores_an_interrupted_reset_before_checking_the_lineage(tmp_path, monkeypatch):
    """A set-aside dir without a working dir is a reset that was interrupted, it is the study again before anything is deleted."""
    cfg = _config(tmp_path, parent_study="never_run_parent")
    working_dir = _stale_working_dir(cfg)
    working_dir.rename(working_dir.with_name(f"{cfg.study_name}{study_module.PREVIOUS_WORKING_DIR_SUFFIX}"))
    monkeypatch.setattr(study_module, "get_running_job_names", lambda: set())
    reset_config()
    with pytest.raises(RuntimeError, match="no config lock"):
        PowerBalanceStudy.reset_study(cfg)
    assert (working_dir / "results" / "stale.nc").exists()
