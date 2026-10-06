"""The transfer_pretrain twin every transfer case pretrains through (Study.Case.transfer_pretrain_case).

Stateless normalizations (raw, physics) share one twin at HYPERPARAM_TARGET_SHOTS.
Stat normalizations keep one twin per num_target_shots, their statistics are fitted on historic + n target shots.
"""

from collections import Counter
from typing import Any

import numpy as np
import pytest

from transport_study import EPISODE_DIM
from transport_study.config import config
from transport_study.orchestration.organize_data import (
    TrainingData,
    get_transfer_pretrain_datasets,
    merge_parts,
)
from transport_study.orchestration.study import HYPERPARAM_TARGET_SHOTS
from transport_study.orchestration.target_shots import TargetSplit
from transport_study.power_balance_transfer.power_balance_study import PowerBalanceStudy

STATELESS_NORMALIZATIONS = ("raw", "physics")


def _study(store_paths: dict, working_dir_base, target_test_set_size: int, num_target_shots_options: tuple[int, ...]) -> PowerBalanceStudy:
    """cmod the source and mast the target, of the 3 synthetic target shots target_test_set_size held out for testing."""
    return PowerBalanceStudy(
        PowerBalanceStudy.Config(
            study_name="test-transfer-pretrain-twin",
            working_dir_base=working_dir_base,
            dataset_paths=store_paths,
            target_device="mast",
            target_test_set_size=target_test_set_size,
            training_datasets=("cmod",),
            domain_adaptation_methods=("none", "transfer"),
            model_types=("mlp", "sciml-taue-nn"),
            data_normalization_methods=("physics", "coral"),
            num_target_shots_options=num_target_shots_options,
        )
    )


@pytest.fixture
def study(synthetic_device_stores, tmp_path) -> PowerBalanceStudy:
    """One target test shot, so 2 target shots remain for training."""
    return _study(synthetic_device_stores, tmp_path, target_test_set_size=1, num_target_shots_options=(1, 2))


def _case(**overrides) -> PowerBalanceStudy.Case:
    kwargs: dict[str, Any] = {
        "model_type": "mlp",
        "training_data": "cmod",
        "data_normalization": "physics",
        "domain_adaptation": "transfer",
        "freeze_submodules": True,
        "num_target_shots": 1,
    }
    kwargs.update(overrides)
    return PowerBalanceStudy.Case(**kwargs)


@pytest.mark.parametrize("data_normalization", ["raw", "physics", "coral", "physics-zscore"])
def test_twin_target_shots_follow_normalization(study, data_normalization):
    twins = [_case(data_normalization=data_normalization, num_target_shots=n).transfer_pretrain_case() for n in (1, 2)]

    assert all(twin.domain_adaptation == "transfer_pretrain" for twin in twins)
    if data_normalization in STATELESS_NORMALIZATIONS:
        assert twins[0] == twins[1]
        assert twins[0].num_target_shots == HYPERPARAM_TARGET_SHOTS
    else:
        assert [twin.num_target_shots for twin in twins] == [1, 2]


def test_case_grid_holds_one_stateless_twin_per_combination(study):
    twin_counts = Counter(
        (case.model_type, case.data_normalization) for case in study.cases if case.domain_adaptation == "transfer_pretrain"
    )

    # Both shot counts share the physics twin, coral has one per shot count
    assert twin_counts[("mlp", "physics")] == 1
    assert twin_counts[("mlp", "coral")] == 2


@pytest.mark.parametrize(
    ("domain_adaptation", "data_normalization", "impossible"),
    [
        # Nothing to finetune on
        ("transfer", "physics", True),
        # A stat twin exists to fit target-aware statistics
        ("transfer_pretrain", "coral", True),
        # The shared stateless twin
        ("transfer_pretrain", "physics", False),
    ],
)
def test_zero_target_shot_impossibility(study, domain_adaptation, data_normalization, impossible):
    case = _case(domain_adaptation=domain_adaptation, data_normalization=data_normalization, num_target_shots=0)
    assert case.is_impossible() == impossible


def test_finetune_restores_the_twin_and_twin_trains_from_scratch(synthetic_device_stores, tmp_path, monkeypatch):
    """The transfer case starts from its twin's checkpoint, the twin itself from scratch on the untouched schedule."""
    # The transfer lr budget builds the dataloaders, and popsim rejects a single-shot whole-episode test set
    study = _study(synthetic_device_stores, tmp_path, target_test_set_size=2, num_target_shots_options=(1,))
    # The synthetic shots are shorter than the study's training segments
    base_dataloader_config = study.base_dataloader_config
    monkeypatch.setattr(
        study,
        "base_dataloader_config",
        lambda case: {**base_dataloader_config(case), "segment_length_train": 20, "segment_overlap_train": 10},
    )
    case = _case()
    twin = case.transfer_pretrain_case()

    finetune_config = study.make_train_config(case)
    twin_config = study.make_train_config(twin)

    assert finetune_config.model_init_config["transfer_checkpoint"] == str(study.trained_model_dir(twin))
    assert not twin_config.model_init_config.get("transfer_checkpoint")
    assert twin_config.optimizer_config == study.base_optimizer_config()


def test_zero_shot_pretrain_datasets(study):
    """With no target shots the normalizer-fit set is the historic set, and the empty target selection concatenates cleanly."""
    target_split = TargetSplit(num_target_shots=0, target_shot_order="ascending", target_test_set_size=1, target_test_shots=())
    train_parts, normalizer_fit_ds, test_ds = get_transfer_pretrain_datasets(
        TrainingData(sources=["cmod"]), target_split=target_split, study_type="power_balance_transfer"
    )
    train_ds = merge_parts(train_parts)

    assert np.atleast_1d(train_ds["ds_source"].values).tolist() == ["cmod"]
    assert normalizer_fit_ds[EPISODE_DIM].values.tolist() == train_ds[EPISODE_DIM].values.tolist()
    assert test_ds.sizes[EPISODE_DIM] == 1
    assert np.atleast_1d(test_ds["ds_source"].values).tolist() == ["mast"]


def test_submodule_twins_follow_parent(study):
    """A sciml-taue-nn twin pretrains on p_oh / p_rad twins at the same shot count, pinned to the hyperparam freeze value."""
    twin = _case(model_type="sciml-taue-nn", num_target_shots=2).transfer_pretrain_case()

    submodule_twins = [prereq for prereq in twin.prereqs if prereq.model_type in ("p_oh", "p_rad")]

    assert sorted(prereq.model_type for prereq in submodule_twins) == ["p_oh", "p_rad"]
    for prereq in submodule_twins:
        assert prereq.domain_adaptation == "transfer_pretrain"
        assert prereq.num_target_shots == HYPERPARAM_TARGET_SHOTS
        assert prereq.freeze_submodules == config.hyperparam_freeze_submodules
