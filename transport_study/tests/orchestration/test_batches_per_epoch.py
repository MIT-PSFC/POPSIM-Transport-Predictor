"""The historic training set's batches per epoch (Study.batches_per_epoch), which set the transfer learning rate,
are measured once and recorded in the working dir for later runs and child studies, until a reset."""

from transport_study.config import reset_config
from transport_study.modules.power_balance.trb import PowerBalanceTRB
from transport_study.orchestration.study import BATCHES_PER_EPOCH_FILENAME
from transport_study.power_balance_transfer.power_balance_study import PowerBalanceStudy
from transport_study.tests.tuned_configs import write_hyperparam_tuned_configs

# Small batches and short segments, so the synthetic historic set spans more than one batch
SHORT_LOADERS = {"batch_size": 2, "segment_length_train": 20, "segment_overlap_train": 10}


def _config(synthetic_device_stores, working_dir_base, **overrides) -> PowerBalanceStudy.Config:
    return PowerBalanceStudy.Config(
        **{
            "study_name": "test-batches",
            "working_dir_base": working_dir_base,
            "dataset_paths": synthetic_device_stores,
            "target_device": "mast",
            # Two of the three target shots are the test set, popsim rejects a single-shot whole-episode test set
            "target_test_set_size": 2,
            "training_datasets": ("cmod",),
            "model_types": ("mlp",),
            "data_normalization_methods": ("physics",),
            "domain_adaptation_methods": ("none", "transfer"),
            "num_target_shots_options": (0, 1),
            **overrides,
        }
    )


def _study(study_config: PowerBalanceStudy.Config, monkeypatch) -> PowerBalanceStudy:
    reset_config()
    study = PowerBalanceStudy(study_config)
    base_dataloader_config = study.base_dataloader_config
    monkeypatch.setattr(study, "base_dataloader_config", lambda case: {**base_dataloader_config(case), **SHORT_LOADERS})
    write_hyperparam_tuned_configs(study)
    return study


def _transfer_lr0(study: PowerBalanceStudy) -> float:
    case = next(case for case in study.cases if case.domain_adaptation == "transfer")
    return study.make_train_config(case).optimizer_config["lr0"]


def test_recorded_batches_per_epoch_serve_later_runs_and_children_until_a_reset(synthetic_device_stores, tmp_path, monkeypatch):
    """A later run and a child study read the recorded batch count back instead of rebuilding the transfer loader,
    and a clean, which writes a fresh config lock, drops the record."""
    parent_config = _config(synthetic_device_stores, tmp_path)
    lr0_measured = _transfer_lr0(_study(parent_config, monkeypatch))

    def refuse_to_measure(dataloader_config):
        raise AssertionError("the recorded batches per epoch should have been read back")

    with monkeypatch.context() as no_measurement:
        no_measurement.setattr(PowerBalanceTRB, "get_dataloaders", staticmethod(refuse_to_measure))
        rerun = _study(parent_config, monkeypatch)
        child = _study(
            _config(synthetic_device_stores, tmp_path, study_name="test-batches-child", parent_study="test-batches"), monkeypatch
        )
        assert _transfer_lr0(rerun) == lr0_measured
        assert _transfer_lr0(child) == lr0_measured
    # The child read its parent's record and wrote none of its own
    assert not (child.working_dir / BATCHES_PER_EPOCH_FILENAME).exists()

    PowerBalanceStudy.clean_working_dir(parent_config, clean_models=True, clean_results=True)
    reset_study = _study(parent_config, monkeypatch)
    assert not (reset_study.working_dir / BATCHES_PER_EPOCH_FILENAME).exists()
