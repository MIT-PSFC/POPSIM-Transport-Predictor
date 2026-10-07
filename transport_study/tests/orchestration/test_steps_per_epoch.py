"""Every case of a study trains the same steps per epoch (Study.steps_per_epoch),
the natural steps per epoch of the largest training set its locked fields allow.
"""

from transport_study.config import reset_config
from transport_study.modules.power_balance.trb import PowerBalanceTRB
from transport_study.orchestration.study import STEPS_PER_EPOCH_FILENAME
from transport_study.power_balance_transfer.power_balance_study import PowerBalanceStudy

# Small batches and short segments, so the synthetic training sets differ in their natural steps per epoch
SHORT_LOADERS = {"batch_size": 2, "segment_length_train": 20, "segment_overlap_train": 10}


def _config(synthetic_device_stores, working_dir_base, num_target_shots_options, **overrides) -> PowerBalanceStudy.Config:
    return PowerBalanceStudy.Config(
        **{
            "study_name": "test-steps",
            "working_dir_base": working_dir_base,
            "dataset_paths": synthetic_device_stores,
            "target_device": "mast",
            # Two of the three target shots are the test set, popsim rejects a single-shot whole-episode test set
            "target_test_set_size": 2,
            "training_datasets": ("cmod", "exnihilo"),
            "model_types": ("mlp",),
            "data_normalization_methods": ("physics",),
            "domain_adaptation_methods": ("none", "addition", "transfer"),
            "num_target_shots_options": num_target_shots_options,
            **overrides,
        }
    )


def _study(study_config: PowerBalanceStudy.Config, monkeypatch) -> PowerBalanceStudy:
    reset_config()
    study = PowerBalanceStudy(study_config)
    base_dataloader_config = study.base_dataloader_config
    monkeypatch.setattr(study, "base_dataloader_config", lambda case: {**base_dataloader_config(case), **SHORT_LOADERS})
    return study


def _train_steps(dataloader_config: dict) -> int:
    _, train_dl, _, _ = PowerBalanceTRB.get_dataloaders(dataloader_config)
    return len(train_dl)


def _addition_steps_per_epoch(study: PowerBalanceStudy) -> int:
    case = next(case for case in study.cases if case.domain_adaptation == "addition")
    return study.make_train_config(case).dataloader_config["steps_per_epoch"]


def test_every_case_trains_steps_of_largest_training_set(synthetic_device_stores, tmp_path, monkeypatch):
    """addition at the whole target pool is the largest set, the target-only exnihilo and transfer sets repeat up to it."""
    study = _study(_config(synthetic_device_stores, tmp_path, (0, 1)), monkeypatch)
    # A none case on the sources validates on their val split, one synthetic shot, which popsim rejects as a whole-episode set
    cases = [case for case in study.cases if case.domain_adaptation is not None or case.training_data.exnihilo]
    steps_per_epoch, natural_steps = {}, {}
    for case in cases:
        dataloader_config = study.make_train_config(case).dataloader_config
        steps_per_epoch[str(case)] = _train_steps(dataloader_config)
        natural_steps[str(case)] = _train_steps({**dataloader_config, "steps_per_epoch": None})

    assert set(steps_per_epoch.values()) == {max(natural_steps.values())}
    assert min(natural_steps.values()) < max(natural_steps.values())


def test_steps_per_epoch_do_not_follow_case_grid(synthetic_device_stores, tmp_path, monkeypatch):
    """A grid without the largest training set, like a lineage child's slice, still trains the same steps per epoch."""
    steps_per_epoch = {}
    for num_target_shots_options in ((0, 1), (0,)):
        working_dir_base = tmp_path / str(len(num_target_shots_options))
        study = _study(_config(synthetic_device_stores, working_dir_base, num_target_shots_options), monkeypatch)
        case = next(case for case in study.cases if case.domain_adaptation == "addition")
        dataloader_config = study.make_train_config(case).dataloader_config
        steps_per_epoch[num_target_shots_options] = dataloader_config["steps_per_epoch"]
        natural_steps = _train_steps({**dataloader_config, "steps_per_epoch": None})

    assert steps_per_epoch[(0, 1)] == steps_per_epoch[(0,)]
    # The largest set of the (0,) grid, addition without target shots, falls short of the study's steps per epoch
    assert natural_steps < steps_per_epoch[(0,)]


def test_recorded_steps_per_epoch_serve_later_runs_and_children_until_a_reset(synthetic_device_stores, tmp_path, monkeypatch):
    """A later run and a child study read the recorded steps per epoch back instead of rebuilding the largest training set,
    and a clean, which writes a fresh config lock, drops the record."""
    parent_config = _config(synthetic_device_stores, tmp_path, (0, 1))
    steps_measured = _addition_steps_per_epoch(_study(parent_config, monkeypatch))

    def refuse_to_measure(dataloader_config):
        raise AssertionError("the recorded steps per epoch should have been read back")

    with monkeypatch.context() as no_measurement:
        no_measurement.setattr(PowerBalanceTRB, "get_dataloaders", staticmethod(refuse_to_measure))
        rerun = _study(parent_config, monkeypatch)
        child = _study(
            _config(synthetic_device_stores, tmp_path, (0,), study_name="test-steps-child", parent_study="test-steps"), monkeypatch
        )
        assert _addition_steps_per_epoch(rerun) == steps_measured
        assert _addition_steps_per_epoch(child) == steps_measured
    # The child read its parent's record and wrote none of its own
    assert not (child.working_dir / STEPS_PER_EPOCH_FILENAME).exists()

    PowerBalanceStudy.clean_working_dir(parent_config, clean_models=True, clean_results=True)
    reset_study = _study(parent_config, monkeypatch)
    assert not (reset_study.working_dir / STEPS_PER_EPOCH_FILENAME).exists()
