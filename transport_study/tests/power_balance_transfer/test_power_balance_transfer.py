import os
import shutil
from pathlib import Path

import chex
import numpy as np
import pytest
import xarray as xr

from transport_study import PACKAGE_ROOT
from transport_study.config import config
from transport_study.modules.normalization import STAT_NORMALIZATIONS
from transport_study.orchestration.organize_data import get_train_test_datasets
from transport_study.power_balance_transfer.power_balance_study import (
    HYPERPARAM_TARGET_SHOTS,
    PowerBalanceStudy,
)

SAMPLE_DIR = PACKAGE_ROOT / "datasets" / "sample"


def _working_dir_base() -> Path:
    working_dir_base = os.environ.get("PTPS_TEST_WORKING_DIR_BASE", None)
    if working_dir_base is None:
        working_dir_base = PACKAGE_ROOT / "tests" / "test_outputs"
    return Path(working_dir_base) / "power_balance_transfer"


def _make_config(study_name: str, **overrides) -> PowerBalanceStudy.Config:
    """Small Config against the bundled sample datasets, quick enough for smoke training."""
    defaults = dict(
        study_name=study_name,
        working_dir_base=_working_dir_base(),
        dataset_paths={
            "cmod-low1": SAMPLE_DIR / "cmod-low1.nc",
            "cmod-low2": SAMPLE_DIR / "cmod-low2.nc",
            "cmod-high": SAMPLE_DIR / "cmod-high.nc",
        },
        target_device="cmod-high",
        debug=True,
        max_ds_size=20,
        max_epochs=2,
        epochs_per_val=1,
        patience=2,
        model_types=("sciml-taue-nn", "mlp"),
        training_datasets=("cmod-low1", "cmod-low1_cmod-low2"),
        data_normalization_methods=("coral",),
        domain_adaptation_methods=(None, "weighted", "addition", "transfer"),
        freeze_submodules_options=(True,),
        num_target_shots_options=(0, 3),
        target_test_set_size=4,
    )
    defaults.update(overrides)
    return PowerBalanceStudy.Config(**defaults)


def _clean_case(study: PowerBalanceStudy, case: PowerBalanceStudy.Case):
    """Clear a case's checkpoints and results from previous runs before retraining.

    Without this, orbax keeps the old "best" checkpoint (max_to_keep=1,
    best_mode=min), so restore_best_checkpoint=True would load stale weights
    from a broken training run. The _latest resume dir must go too, otherwise
    launch_train resumes at max_epochs and trains for zero epochs, never
    writing a fresh best checkpoint.
    """
    shutil.rmtree(study.trained_model_dir(case), ignore_errors=True)
    shutil.rmtree(Path(f"{study.trained_model_dir(case)}_latest"), ignore_errors=True)
    if study.result_path(case).exists():
        os.remove(study.result_path(case))


def _hyperparam_case(study: PowerBalanceStudy, model_type: str) -> PowerBalanceStudy.Case:
    """The single tuning case of a model type, as the study itself derives it."""
    case = next(c for c in study.cases if c.model_type == model_type)
    return case.get_hyperparam_prereq()


def test_power_balance_transfer_cases():
    """Make sure the case graph is set up correctly (pure logic, no training)."""
    study = PowerBalanceStudy(
        _make_config(
            "xfer_test",
            model_types=("sciml-taue-nn", "mlp", "transformer"),
            data_normalization_methods=("raw", "coral"),
            freeze_submodules_options=(True, False),
            num_target_shots_options=(0, -1),
        )
    )

    # Ensure that each case with a prereq, has that prereq in the list of cases
    for case in study.cases:
        if case.prereqs is not None:
            for prereq in case.prereqs:
                assert prereq in study.cases, f"{case}\nhas prereq\n{prereq}\nwhich is not in the list of cases"

    # Ensure that there are no duplicate cases
    assert len(study.cases) == len(set(study.cases)), "There are duplicate cases in the study"

    # Ensure there is only one hyperparameter tuning case per model type
    hp_tuning_cases_by_model = {}
    for case in study.cases:
        if case.is_hyperparam_case():
            assert case.model_type not in hp_tuning_cases_by_model, (
                f"Multiple hyperparameter tuning cases found for model type {case.model_type}"
            )
            hp_tuning_cases_by_model[case.model_type] = case

    # Models without submodules only get one freeze variant
    for case in study.cases:
        if case.model_type in ("mlp", "transformer"):
            assert case.freeze_submodules == config.hyperparam_freeze_submodules

    # No impossible cases survive
    for case in study.cases:
        assert not case.is_impossible(), f"Impossible case in study: {case}"

    def _submodule_prereqs(case):
        return [
            PowerBalanceStudy.Case(
                model_type=submodule_type,
                training_data=case.training_data,
                data_normalization=case.data_normalization,
                domain_adaptation=case.domain_adaptation,
                freeze_submodules=config.hyperparam_freeze_submodules,
                num_target_shots=case.num_target_shots,
            )
            for submodule_type in ("p_oh", "p_rad")
        ]

    # For every case, check the prereqs match the hand-derived expectation
    for case in study.cases:
        if case.is_hyperparam_case():
            if case.model_type in ("p_oh", "p_rad", "mlp", "transformer"):
                assert case.prereqs is None, f"Hyperparameter case {case} should have no prereqs, has {case.prereqs}"
                continue
            expected_prereqs = _submodule_prereqs(case)
        elif case.domain_adaptation == "transfer":
            twin = PowerBalanceStudy.Case(
                model_type=case.model_type,
                training_data=case.training_data,
                data_normalization=case.data_normalization,
                domain_adaptation="transfer_pretrain",
                freeze_submodules=case.freeze_submodules,
                # Stat normalizations fit their per-device statistics on the
                # combined historic + target data of THIS case, so their twin
                # keeps its num_target_shots. Stateless normalizations have
                # nothing to fit, so all their transfer cases share one twin
                num_target_shots=case.num_target_shots if case.data_normalization in STAT_NORMALIZATIONS else HYPERPARAM_TARGET_SHOTS,
            )
            expected_prereqs = [_hyperparam_case(study, case.model_type), twin]
            if case.model_type in ("sciml-taue-nn", "sciml-taue-scalinglaw"):
                expected_prereqs += _submodule_prereqs(case)
        elif case.domain_adaptation in ("weighted", "addition", None, "transfer_pretrain"):
            expected_prereqs = [_hyperparam_case(study, case.model_type)]
            if case.model_type in ("sciml-taue-nn", "sciml-taue-scalinglaw"):
                expected_prereqs += _submodule_prereqs(case)
        else:
            continue

        assert set(case.prereqs) == set(expected_prereqs), (
            f"{case}\nexpected prereqs:\n"
            + "\n".join(str(p) for p in expected_prereqs)
            + "\nactual:\n"
            + "\n".join(str(p) for p in case.prereqs)
        )


@pytest.mark.slow
def test_weighted_device_weight():
    """Train a single p_oh case with weighted domain adaptation to exercise device weighting."""
    study = PowerBalanceStudy(
        _make_config(
            "test_weighted_device_weight",
            model_types=("sciml-taue-nn",),
            training_datasets=("cmod-low1_cmod-low2",),
            domain_adaptation_methods=("weighted",),
            num_target_shots_options=(3,),
        )
    )

    case_p_oh = PowerBalanceStudy.Case(
        model_type="p_oh",
        training_data="cmod-low1_cmod-low2",
        data_normalization="coral",
        domain_adaptation="weighted",
        freeze_submodules=True,
        num_target_shots=3,
    )

    weights = study.make_weighted_device_weights(case_p_oh)
    assert set(weights) == {"cmod-low1", "cmod-low2", "cmod-high"}
    assert all(w > 0 for w in weights.values())
    # Target gets the largest per-sample weight (few shots, half the budget)
    assert weights["cmod-high"] > weights["cmod-low1"]

    study.launch_train(case_p_oh)
    assert study.result_path(case_p_oh).exists()


def test_addition_no_device_weights():
    """An 'addition' case adds target shots as normal samples without loss weighting.

    make_train_config for a domain_adaptation='addition' case must NOT put a
    'device_weights' entry in loss_config (and neither in
    val_eval_suite_config['loss_config']), while an otherwise identical
    'weighted' case does. get_train_test_datasets must return identical
    train/test datasets for 'weighted' and 'addition' (same shots, same
    values), since the two methods differ only in the loss weighting.
    """
    study = PowerBalanceStudy(
        _make_config(
            "test_addition_no_device_weights",
            model_types=("mlp",),
            training_datasets=("cmod-low1_cmod-low2",),
            domain_adaptation_methods=("weighted", "addition"),
            num_target_shots_options=(3,),
        )
    )

    def _case(domain_adaptation):
        return PowerBalanceStudy.Case(
            model_type="mlp",
            training_data="cmod-low1_cmod-low2",
            data_normalization="coral",
            domain_adaptation=domain_adaptation,
            freeze_submodules=True,
            num_target_shots=3,
        )

    case_weighted = _case("weighted")
    case_addition = _case("addition")

    train_config_weighted = study.make_train_config(case_weighted)
    train_config_addition = study.make_train_config(case_addition)

    assert "device_weights" in train_config_weighted.loss_config
    assert "device_weights" in train_config_weighted.val_eval_suite_config["loss_config"]
    assert "device_weights" not in train_config_addition.loss_config
    assert "device_weights" not in train_config_addition.val_eval_suite_config["loss_config"]

    # The weighted case weights every training device, target included
    weights = train_config_weighted.loss_config["device_weights"]
    assert set(weights) == {"cmod-low1", "cmod-low2", "cmod-high"}

    # Identical datasets: the methods differ only in the loss weighting
    def _datasets(case):
        return get_train_test_datasets(
            case.training_data,
            domain_adaptation=case.domain_adaptation,
            num_target_shots=case.num_target_shots,
            target_test_set_size=config.target_test_set_size,
            study_type=PowerBalanceStudy.STUDY_TYPE,
        )

    train_ds_weighted, test_ds_weighted = _datasets(case_weighted)
    train_ds_addition, test_ds_addition = _datasets(case_addition)
    xr.testing.assert_identical(train_ds_weighted, train_ds_addition)
    xr.testing.assert_identical(test_ds_weighted, test_ds_addition)


@pytest.mark.slow
def test_transformer_training():
    """Smoke-train the transformer case and check the normalizer stats stay frozen."""
    study = PowerBalanceStudy(
        _make_config(
            "test_transformer_training",
            model_types=("transformer",),
            training_datasets=("cmod-low1",),
            domain_adaptation_methods=(None,),
            num_target_shots_options=(0,),
        )
    )

    case = PowerBalanceStudy.Case(
        model_type="transformer",
        training_data="cmod-low1",
        data_normalization="coral",
        domain_adaptation=None,
        freeze_submodules=True,
        num_target_shots=0,
    )

    _clean_case(study, case)

    study.launch_train(case)
    assert study.result_path(case).exists()

    trainer_init, _ = study.restore_trainer(case, restore_best_checkpoint=False)
    trainer_final, _ = study.restore_trainer(case, restore_best_checkpoint=True)
    module_init = trainer_init.train_state.model.module
    module_final = trainer_final.train_state.model.module

    # Normalizer statistics are frozen buffers, training must not touch them
    chex.assert_trees_all_equal(module_init.normalizer, module_final.normalizer)
    # The attention weights and head did train
    with pytest.raises(AssertionError):
        chex.assert_trees_all_equal(module_init.head, module_final.head)


@pytest.mark.slow
def test_submodule_freezing():
    study = PowerBalanceStudy(
        _make_config(
            "test_submodule_freezing",
            model_types=("sciml-taue-nn",),
            training_datasets=("cmod-low1_cmod-low2",),
            domain_adaptation_methods=(None,),
            freeze_submodules_options=(True, False),
            num_target_shots_options=(0,),
        )
    )

    def _case(model_type, freeze):
        return PowerBalanceStudy.Case(
            model_type=model_type,
            training_data="cmod-low1_cmod-low2",
            data_normalization="coral",
            domain_adaptation=None,
            freeze_submodules=freeze,
            num_target_shots=HYPERPARAM_TARGET_SHOTS,
        )

    case_p_oh = _case("p_oh", True)
    case_p_rad = _case("p_rad", True)
    case_frozen = _case("sciml-taue-nn", True)
    case_unfrozen = _case("sciml-taue-nn", False)

    for case in [case_p_oh, case_p_rad, case_frozen, case_unfrozen]:
        if not study.result_path(case).exists():
            study.launch_train(case)

    # Load the trained models for each case,
    # Ensure the weights for p_oh and p_rad are the same for the frozen case,
    # and different for the unfrozen case
    trainer_p_oh, _ = study.restore_trainer(case_p_oh)
    trainer_p_rad, _ = study.restore_trainer(case_p_rad)
    trainer_frozen, _ = study.restore_trainer(case_frozen)
    trainer_unfrozen, _ = study.restore_trainer(case_unfrozen)

    model_p_oh = trainer_p_oh.train_state.model.nn
    model_p_rad = trainer_p_rad.train_state.model.nn
    model_frozen = trainer_frozen.train_state.model.module
    model_unfrozen = trainer_unfrozen.train_state.model.module

    chex.assert_trees_all_equal(model_p_oh, model_frozen.p_oh_predictor.nn)
    chex.assert_trees_all_equal(model_p_rad, model_frozen.p_rad_predictor.nn)
    with pytest.raises(AssertionError):
        chex.assert_trees_all_equal(model_p_oh, model_unfrozen.p_oh_predictor.nn)
    with pytest.raises(AssertionError):
        chex.assert_trees_all_equal(model_p_rad, model_unfrozen.p_rad_predictor.nn)

    # The frozen submodules' normalizer stats survive unchanged too
    chex.assert_trees_all_equal(
        trainer_p_oh.train_state.model.normalizer,
        model_frozen.p_oh_predictor.normalizer,
    )


@pytest.mark.slow
def test_transfer_weights():
    """Transfer with a stat-based normalization runs as two cases: a
    transfer_pretrain twin trains on the historic data with the normalizer
    fitted on historic + target shots, then the transfer case restores that
    checkpoint (normalizer included) and fine-tunes the last layer on the
    target shots."""
    study = PowerBalanceStudy(
        _make_config(
            "test_transfer_weights",
            model_types=("mlp",),
            training_datasets=("cmod-low1_cmod-low2",),
            domain_adaptation_methods=("transfer",),
            num_target_shots_options=(3,),
        )
    )

    case_transfer = PowerBalanceStudy.Case(
        model_type="mlp",
        training_data="cmod-low1_cmod-low2",
        data_normalization="coral",
        domain_adaptation="transfer",
        freeze_submodules=True,
        num_target_shots=3,
    )
    # The pretrain twin keeps this case's num_target_shots, the old shared
    # source-only pretrain (da=None) is not a prereq anymore
    case_pretrain = case_transfer.replace(domain_adaptation="transfer_pretrain")
    assert case_pretrain in case_transfer.prereqs
    assert case_transfer.replace(domain_adaptation=None, num_target_shots=HYPERPARAM_TARGET_SHOTS) not in case_transfer.prereqs
    assert case_pretrain in study.cases

    _clean_case(study, case_pretrain)
    _clean_case(study, case_transfer)
    study.launch_train(case_pretrain)
    assert study.result_path(case_pretrain).exists()
    study.launch_train(case_transfer)
    assert study.result_path(case_transfer).exists()

    pretrain_final = study.restore_trainer(case_pretrain)[0].train_state.model.module
    # restore_best_checkpoint=False on the transfer case IS the pretrain best:
    # model_init restores the transfer_checkpoint before fine-tuning starts
    transfer_init = study.restore_trainer(case_transfer, restore_best_checkpoint=False)[0].train_state.model.module
    transfer_final = study.restore_trainer(case_transfer)[0].train_state.model.module

    # Important! The TARGET device's per-device stats are fitted by the pretrain twin
    # (identity before this change, the shared pretrain never saw the target device)
    target_idx = config.ds_source_to_idx[config.target_device]
    n = np.asarray(pretrain_final.normalizer.transforms[target_idx]).shape[0]
    assert not np.allclose(np.asarray(pretrain_final.normalizer.transforms[target_idx]), np.eye(n)), (
        "Target device CORAL transform is identity, the combined fit did not happen"
    )
    # The transfer case inherits the pretrain normalizer through the
    # checkpoint restore and never refits or trains it
    chex.assert_trees_all_equal(pretrain_final.normalizer, transfer_init.normalizer)
    chex.assert_trees_all_equal(pretrain_final.normalizer, transfer_final.normalizer)

    # The pretrain trained the full network from the random init
    pretrain_random_init = study.restore_trainer(case_pretrain, restore_best_checkpoint=False)[0].train_state.model.module
    for i in range(len(pretrain_final.nn.layers)):
        with pytest.raises(AssertionError):
            chex.assert_trees_all_equal(pretrain_random_init.nn.layers[i], pretrain_final.nn.layers[i])

    # The transfer case fine-tunes only the final layer on top of the pretrain best
    chex.assert_trees_all_equal(pretrain_final.nn, transfer_init.nn)
    for i in range(len(pretrain_final.nn.layers) - 1):
        chex.assert_trees_all_equal(pretrain_final.nn.layers[i], transfer_final.nn.layers[i])
    with pytest.raises(AssertionError):
        chex.assert_trees_all_equal(pretrain_final.nn.layers[-1], transfer_final.nn.layers[-1])


@pytest.mark.slow
@pytest.mark.parametrize("freeze_submodules", [True, False], ids=["frozen", "unfrozen"])
def test_transfer_weights_submodules(freeze_submodules):
    study = PowerBalanceStudy(
        _make_config(
            f"test_transfer_weights_submodules_{freeze_submodules}",
            model_types=("sciml-taue-nn",),
            training_datasets=("cmod-low1_cmod-low2",),
            domain_adaptation_methods=("transfer",),
            freeze_submodules_options=(freeze_submodules,),
            num_target_shots_options=(3,),
        )
    )

    def _case(model_type, domain_adaptation, freeze):
        return PowerBalanceStudy.Case(
            model_type=model_type,
            training_data="cmod-low1_cmod-low2",
            data_normalization="coral",
            domain_adaptation=domain_adaptation,
            freeze_submodules=freeze,
            num_target_shots=HYPERPARAM_TARGET_SHOTS if domain_adaptation is None else 3,
        )

    case_p_oh_transfer = _case("p_oh", "transfer", True)
    case_p_rad_transfer = _case("p_rad", "transfer", True)
    case_sciml_transfer = _case("sciml-taue-nn", "transfer", freeze_submodules)
    case_p_oh_pretrain = case_p_oh_transfer.replace(domain_adaptation="transfer_pretrain")
    case_p_rad_pretrain = case_p_rad_transfer.replace(domain_adaptation="transfer_pretrain")
    case_sciml_pretrain = case_sciml_transfer.replace(domain_adaptation="transfer_pretrain")

    # Prereq order:
    # submodule pretrains,
    # submodule transfers,
    # sciml pretrain (restores the pretrained submodules),
    # sciml transfer (restores the sciml pretrain plus the fine-tuned submodules)
    ordered_cases = [
        case_p_oh_pretrain,
        case_p_rad_pretrain,
        case_p_oh_transfer,
        case_p_rad_transfer,
        case_sciml_pretrain,
        case_sciml_transfer,
    ]
    for case in ordered_cases:
        _clean_case(study, case)
    for case in ordered_cases:
        study.launch_train(case)
        assert study.result_path(case).exists()

    # Get initial and final modules for each case
    sciml_transfer_init = study.restore_trainer(case_sciml_transfer, restore_best_checkpoint=False)[0].train_state.model.module
    p_oh_transfer_final = study.restore_trainer(case_p_oh_transfer)[0].train_state.model
    p_rad_transfer_final = study.restore_trainer(case_p_rad_transfer)[0].train_state.model
    sciml_transfer_final = study.restore_trainer(case_sciml_transfer)[0].train_state.model.module
    sciml_pretrain_final = study.restore_trainer(case_sciml_pretrain)[0].train_state.model.module

    # 1. The transfer case fine-tunes only the taue network's last layer on
    # top of the pretrain best
    taue_pretrain = sciml_pretrain_final.taue_predictor.nn
    taue_final = sciml_transfer_final.taue_predictor.nn
    for i in range(len(taue_pretrain.layers) - 1):
        chex.assert_trees_all_equal(taue_pretrain.layers[i], taue_final.layers[i])
    with pytest.raises(AssertionError):
        chex.assert_trees_all_equal(taue_pretrain.layers[-1], taue_final.layers[-1])

    # 2. Ensure the submodules were initialized from their own trained checkpoints
    chex.assert_trees_all_equal(p_oh_transfer_final.nn, sciml_transfer_init.p_oh_predictor.nn)
    chex.assert_trees_all_equal(p_rad_transfer_final.nn, sciml_transfer_init.p_rad_predictor.nn)

    if freeze_submodules:
        # 3a. Frozen submodules do not change at all during the sciml transfer run
        chex.assert_trees_all_equal(p_oh_transfer_final.nn, sciml_transfer_final.p_oh_predictor.nn)
        chex.assert_trees_all_equal(p_rad_transfer_final.nn, sciml_transfer_final.p_rad_predictor.nn)
    else:
        # 3b. Unfrozen submodules fine-tune only their last layers during the
        # sciml transfer run (the transfer restriction)
        for init_sub, final_sub in [
            (sciml_transfer_init.p_oh_predictor.nn, sciml_transfer_final.p_oh_predictor.nn),
            (sciml_transfer_init.p_rad_predictor.nn, sciml_transfer_final.p_rad_predictor.nn),
        ]:
            for i in range(len(init_sub.layers) - 1):
                chex.assert_trees_all_equal(init_sub.layers[i], final_sub.layers[i])
            with pytest.raises(AssertionError):
                chex.assert_trees_all_equal(init_sub.layers[-1], final_sub.layers[-1])

    # Normalizer stats never train, anywhere, and the transfer case inherits
    # the pretrain twin's combined-fit stats through the checkpoint restore
    chex.assert_trees_all_equal(sciml_pretrain_final.normalizer, sciml_transfer_final.normalizer)
    chex.assert_trees_all_equal(sciml_transfer_init.normalizer, sciml_transfer_final.normalizer)
    chex.assert_trees_all_equal(
        sciml_transfer_init.p_oh_predictor.normalizer,
        sciml_transfer_final.p_oh_predictor.normalizer,
    )


@pytest.mark.slow
def test_collect_results():
    study = PowerBalanceStudy(
        _make_config(
            "test_collect_results",
            model_types=("sciml-taue-nn",),
            training_datasets=("cmod-low1_cmod-low2",),
            domain_adaptation_methods=("addition",),
            num_target_shots_options=(3,),
        )
    )

    # Clean models and results but keep the working dir itself: the study's
    # open log handle leaves NFS ghost files behind that make a full
    # rmtree + mkdir of the working dir racy
    shutil.rmtree(study.model_dir, ignore_errors=True)
    shutil.rmtree(study.result_dir, ignore_errors=True)
    study.working_dir.mkdir(parents=True, exist_ok=True)

    unfinished_cases = [case for case in study.cases if not study.result_path(case).exists() and study.check_data_requirements(case)]
    while len(unfinished_cases) > 0:
        for case in unfinished_cases:
            study.run_case(case, skip_tuning=True, enable_parallelism=False)
        unfinished_cases = [case for case in unfinished_cases if not study.result_path(case).exists()]

    ds_merged = study.collect_results()

    for case in study.cases:
        assert case.model_type in ds_merged.coords["model_type"].values, f"{case.model_type} not found in collected results"
        assert str(case.training_data) in ds_merged.coords["training_data"].values, f"{case.training_data} not found in collected results"
        assert case.data_normalization in ds_merged.coords["data_normalization"].values, (
            f"{case.data_normalization} not found in collected results"
        )
        # None is normalized to "none" so the coord stays string-typed
        da = case.domain_adaptation if case.domain_adaptation is not None else "none"
        assert da in ds_merged.coords["domain_adaptation"].values, f"{da} not found in collected results"
        assert case.freeze_submodules in ds_merged.coords["freeze_submodules"].values, (
            f"{case.freeze_submodules} not found in collected results"
        )
        assert case.num_target_shots in ds_merged.coords["num_target_shots"].values, (
            f"{case.num_target_shots} not found in collected results"
        )

    for var in ds_merged.data_vars:
        vals = ds_merged[var].values
        assert (vals[~np.isnan(vals)] >= 0).all(), f"Found negative absolute error in {var}"
