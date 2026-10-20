"""Study._apply_tuned_config through make_train_config: what a tuned config contributes, and when a missing one raises."""

import re

import pytest
from popsim.ml import TrainConfig

from transport_study.power_balance_transfer.power_balance_study import (
    POWER_BALANCE_BATCH_SIZE,
    PowerBalanceStudy,
)
from transport_study.tests.tuned_configs import write_hyperparam_tuned_configs

# The synthetic shots are shorter than the study's training segments
SHORT_LOADERS = {"segment_length_train": 20, "segment_overlap_train": 10}


@pytest.fixture
def study(synthetic_device_stores, tmp_path, monkeypatch) -> PowerBalanceStudy:
    """mlp on cmod with mast as the target, the hyperparam case and an addition case that merges its tuned config."""
    study = PowerBalanceStudy(
        PowerBalanceStudy.Config(
            study_name="test-tuned-merge",
            working_dir_base=tmp_path,
            dataset_paths=synthetic_device_stores,
            target_device="mast",
            # Two target test shots, popsim rejects a single-shot whole-episode test set
            target_test_set_size=2,
            training_datasets=("cmod",),
            model_types=("mlp",),
            data_normalization_methods=("physics",),
            domain_adaptation_methods=("none", "addition"),
            num_target_shots_options=(0, 1),
        )
    )
    base_dataloader_config = study.base_dataloader_config
    monkeypatch.setattr(study, "base_dataloader_config", lambda case: {**base_dataloader_config(case), **SHORT_LOADERS})
    return study


def _cases(study: PowerBalanceStudy) -> tuple[PowerBalanceStudy.Case, PowerBalanceStudy.Case]:
    hyperparam_case = next(case for case in study.cases if case.is_hyperparam_case())
    addition_case = next(case for case in study.cases if case.domain_adaptation == "addition")
    return hyperparam_case, addition_case


def test_only_hyperparam_case_may_lack_its_tuned_config(study):
    hyperparam_case, addition_case = _cases(study)
    assert study.make_train_config(hyperparam_case).optimizer_config == study.base_optimizer_config()
    with pytest.raises(FileNotFoundError, match=re.escape(str(hyperparam_case))):
        study.make_train_config(addition_case)


def test_tuned_config_contributes_optimizer_swept_loss_keys_and_swept_init(study):
    """The dataloader config and the unswept loss and init entries stay the base ones."""
    hyperparam_case, addition_case = _cases(study)
    write_hyperparam_tuned_configs(study)
    skip_tuning_config = TrainConfig.load(str(study.tuned_config_path(hyperparam_case)))
    optimizer_config_tuned = {key: value for key, value in skip_tuning_config.optimizer_config.items() if key != "transition_steps"}
    optimizer_config_tuned |= {"lr0": 1e-3, "transition_frac": 0.5}
    tuned_config = skip_tuning_config.model_copy(
        update={
            "optimizer_config": optimizer_config_tuned,
            "loss_config": {**skip_tuning_config.loss_config, "huber_delta": 0.02, "anchor_weight_power_ohm": 99.0},
            "dataloader_config": {**skip_tuning_config.dataloader_config, "batch_size": 7, "segment_length_train": 5},
            "model_init_config": {**skip_tuning_config.model_init_config, "nn_width": 3, "prng_seed": 7},
        }
    )
    study.write_tuned_config(hyperparam_case, tuned_config)

    train_config = study.make_train_config(addition_case)

    assert train_config.optimizer_config == optimizer_config_tuned
    assert train_config.loss_config["huber_delta"] == 0.02
    assert train_config.loss_config["anchor_weight_power_ohm"] == skip_tuning_config.loss_config["anchor_weight_power_ohm"]
    assert train_config.dataloader_config["batch_size"] == POWER_BALANCE_BATCH_SIZE
    assert train_config.dataloader_config["segment_length_train"] == SHORT_LOADERS["segment_length_train"]
    assert train_config.model_init_config["nn_width"] == 3
    assert train_config.model_init_config["prng_seed"] == skip_tuning_config.model_init_config["prng_seed"]


def test_tuned_config_missing_swept_loss_key_raises(study):
    hyperparam_case, addition_case = _cases(study)
    write_hyperparam_tuned_configs(study)
    tuned_config = TrainConfig.load(str(study.tuned_config_path(hyperparam_case)))
    loss_config = {key: value for key, value in tuned_config.loss_config.items() if key != "huber_delta"}
    study.write_tuned_config(hyperparam_case, tuned_config.model_copy(update={"loss_config": loss_config}))
    with pytest.raises(KeyError, match="huber_delta"):
        study.make_train_config(addition_case)
