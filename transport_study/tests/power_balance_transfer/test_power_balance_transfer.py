import os
import shutil
from pathlib import Path

import chex
import numpy as np
import pytest

from transport_study import PACKAGE_ROOT
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
        model_types=("sciml", "unstructured_nn"),
        training_datasets=("cmod-low1", "cmod-low1_cmod-low2"),
        data_normalization_methods=("coral",),
        domain_adaptation_methods=(None, "mixing", "transfer"),
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


def _hyperparam_case(model_type: str) -> PowerBalanceStudy.Case:
    from transport_study.config import config

    return PowerBalanceStudy.Case(
        model_type=model_type,
        training_data=PowerBalanceStudy._hyperparam_training_data(),
        data_normalization=config.hyperparam_data_normalization,
        domain_adaptation=config.hyperparam_domain_adaptation,
        freeze_submodules=config.hyperparam_freeze_submodules,
        num_target_shots=config.hyperparam_num_target_shots,
    )


def test_power_balance_transfer_cases():
    """Make sure the case graph is set up correctly (pure logic, no training)."""
    from transport_study.config import config

    study = PowerBalanceStudy(
        _make_config(
            "xfer_test",
            model_types=("sciml", "unstructured_nn", "transformer"),
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
        if case.model_type in ("unstructured_nn", "transformer"):
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
            if case.model_type in ("p_oh", "p_rad", "unstructured_nn", "transformer"):
                assert case.prereqs is None, f"Hyperparameter case {case} should have no prereqs, has {case.prereqs}"
                continue
            expected_prereqs = _submodule_prereqs(case)
        elif case.domain_adaptation == "transfer":
            expected_prereqs = [
                # The original training case with no domain adaptation to re-init the weights
                PowerBalanceStudy.Case(
                    model_type=case.model_type,
                    training_data=case.training_data,
                    data_normalization=case.data_normalization,
                    domain_adaptation=None,
                    freeze_submodules=case.freeze_submodules,
                    num_target_shots=HYPERPARAM_TARGET_SHOTS,
                ),
                _hyperparam_case(case.model_type),
            ]
            if case.model_type in ("sciml", "scaling_law"):
                expected_prereqs += _submodule_prereqs(case)
        elif case.domain_adaptation in ("mixing", None):
            expected_prereqs = [_hyperparam_case(case.model_type)]
            if case.model_type in ("sciml", "scaling_law"):
                expected_prereqs += _submodule_prereqs(case)
        else:
            continue

        assert set(case.prereqs) == set(expected_prereqs), (
            f"{case}\nexpected prereqs:\n"
            + "\n".join(str(p) for p in expected_prereqs)
            + "\nactual:\n"
            + "\n".join(str(p) for p in case.prereqs)
        )


def test_compatible_configs(tmp_path):
    cfg1 = _make_config("compat_test")
    cfg2 = _make_config("compat_test")
    assert cfg1.is_compatible(cfg2)

    assert not cfg1.is_compatible(_make_config("compat_test_other_name"))
    assert not cfg1.is_compatible(_make_config("compat_test", target_test_set_size=5))
    assert not cfg1.is_compatible(_make_config("compat_test", hyperparam_data_normalization="raw"))
    assert not cfg1.is_compatible(_make_config("compat_test", hyperparam_num_target_shots=1))
    # Case-grid axes do NOT affect compatibility (adding cases to a study is fine)
    assert cfg1.is_compatible(_make_config("compat_test", model_types=("transformer",)))

    # Save/load round trip through the config-lock TOML
    lock_path = tmp_path / "config_lock.toml"
    cfg1.save(lock_path)
    reloaded = PowerBalanceStudy.Config.from_toml(lock_path)
    assert cfg1.is_compatible(reloaded)


def test_mix_device_weight():
    """Train a single p_oh case with mixing domain adaptation to exercise device weighting."""
    study = PowerBalanceStudy(
        _make_config(
            "test_mix_device_weight",
            model_types=("sciml",),
            training_datasets=("cmod-low1_cmod-low2",),
            domain_adaptation_methods=("mixing",),
            num_target_shots_options=(3,),
        )
    )

    case_p_oh = PowerBalanceStudy.Case(
        model_type="p_oh",
        training_data="cmod-low1_cmod-low2",
        data_normalization="coral",
        domain_adaptation="mixing",
        freeze_submodules=True,
        num_target_shots=3,
    )

    weights = study._make_mixing_device_weights(case_p_oh)
    assert set(weights) == {"cmod-low1", "cmod-low2", "cmod-high"}
    assert all(w > 0 for w in weights.values())
    # Target gets the largest per-sample weight (few shots, half the budget)
    assert weights["cmod-high"] > weights["cmod-low1"]

    study.launch_train(case_p_oh)
    assert study.result_path(case_p_oh).exists()


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


def test_submodule_freezing():
    study = PowerBalanceStudy(
        _make_config(
            "test_submodule_freezing",
            model_types=("sciml",),
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
    case_frozen = _case("sciml", True)
    case_unfrozen = _case("sciml", False)

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


def test_transfer_weights():
    study = PowerBalanceStudy(
        _make_config(
            "test_transfer_weights",
            model_types=("unstructured_nn",),
            training_datasets=("cmod-low1_cmod-low2",),
            domain_adaptation_methods=("transfer",),
            num_target_shots_options=(3,),
        )
    )

    case_base = PowerBalanceStudy.Case(
        model_type="unstructured_nn",
        training_data="cmod-low1_cmod-low2",
        data_normalization="coral",
        domain_adaptation=None,
        freeze_submodules=True,
        num_target_shots=HYPERPARAM_TARGET_SHOTS,
    )
    case_transfer = PowerBalanceStudy.Case(
        model_type="unstructured_nn",
        training_data="cmod-low1_cmod-low2",
        data_normalization="coral",
        domain_adaptation="transfer",
        freeze_submodules=True,
        num_target_shots=3,
    )

    if not study.result_path(case_base).exists():
        study.launch_train(case_base)

    _clean_case(study, case_transfer)
    study.launch_train(case_transfer)

    base_trainer_init, _ = study.restore_trainer(case_base, restore_best_checkpoint=False)
    transfer_trainer_init, _ = study.restore_trainer(case_transfer, restore_best_checkpoint=False)
    base_model_init = base_trainer_init.train_state.model.module.nn
    transfer_model_init = transfer_trainer_init.train_state.model.module.nn
    # Can't simply call the restore_best_checkpoint on the trainer since it's a pass by reference
    base_trainer_final, _ = study.restore_trainer(case_base, restore_best_checkpoint=True)
    transfer_trainer_final, _ = study.restore_trainer(case_transfer, restore_best_checkpoint=True)
    base_model_final = base_trainer_final.train_state.model.module.nn
    transfer_model_final = transfer_trainer_final.train_state.model.module.nn

    # Base model final weights should be the same as the transfer model initial weights.
    chex.assert_trees_all_equal(base_model_final, transfer_model_init)

    # The source-fitted normalizer stats ride along through the transfer restore
    # and stay frozen through fine-tuning
    base_norm_final = base_trainer_final.train_state.model.module.normalizer
    transfer_norm_init = transfer_trainer_init.train_state.model.module.normalizer
    transfer_norm_final = transfer_trainer_final.train_state.model.module.normalizer
    chex.assert_trees_all_equal(base_norm_final, transfer_norm_init)
    chex.assert_trees_all_equal(transfer_norm_init, transfer_norm_final)

    # Transfer learning freezes all but the final layer

    # Base model initial weights should be different from base model final weights in every layer (whole model trained)
    for i in range(len(base_model_init.layers)):
        with pytest.raises(AssertionError):
            chex.assert_trees_all_equal(base_model_init.layers[i], base_model_final.layers[i])

    # Transfer model initial weights in layers 0-(n-1) should be the same as transfer model final weights in layers 0-(n-1)
    for i in range(len(transfer_model_init.layers) - 1):
        chex.assert_trees_all_equal(transfer_model_init.layers[i], transfer_model_final.layers[i])

    # Transfer model initial weights in layer n should be different from transfer model final weights in layer n
    with pytest.raises(AssertionError):
        chex.assert_trees_all_equal(transfer_model_init.layers[-1], transfer_model_final.layers[-1])


@pytest.mark.parametrize("freeze_submodules", [True, False], ids=["frozen", "unfrozen"])
def test_transfer_weights_submodules(freeze_submodules):
    study = PowerBalanceStudy(
        _make_config(
            f"test_transfer_weights_submodules_{freeze_submodules}",
            model_types=("sciml",),
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

    case_p_oh_base = _case("p_oh", None, True)
    case_p_oh_transfer = _case("p_oh", "transfer", True)
    case_p_rad_base = _case("p_rad", None, True)
    case_p_rad_transfer = _case("p_rad", "transfer", True)
    case_sciml_base = _case("sciml", None, freeze_submodules)
    case_sciml_transfer = _case("sciml", "transfer", freeze_submodules)

    # Cases that don't need to be re-run if something's broken
    for case in [case_p_oh_base, case_p_rad_base, case_sciml_base]:
        if not study.result_path(case).exists():
            study.launch_train(case)

    for case in [case_p_oh_transfer, case_p_rad_transfer, case_sciml_transfer]:
        _clean_case(study, case)
        study.launch_train(case)

    # Get initial and final modules for each case
    sciml_base_trainer_init, _ = study.restore_trainer(case_sciml_base, restore_best_checkpoint=False)
    sciml_transfer_trainer_init, _ = study.restore_trainer(case_sciml_transfer, restore_best_checkpoint=False)
    sciml_base_init = sciml_base_trainer_init.train_state.model.module
    sciml_transfer_init = sciml_transfer_trainer_init.train_state.model.module
    p_oh_base_final = study.restore_trainer(case_p_oh_base)[0].train_state.model
    p_oh_transfer_final = study.restore_trainer(case_p_oh_transfer)[0].train_state.model
    p_rad_base_final = study.restore_trainer(case_p_rad_base)[0].train_state.model
    p_rad_transfer_final = study.restore_trainer(case_p_rad_transfer)[0].train_state.model
    sciml_base_final = study.restore_trainer(case_sciml_base)[0].train_state.model.module
    sciml_transfer_final = study.restore_trainer(case_sciml_transfer)[0].train_state.model.module

    # 1. Ensure the modules themselves were correctly transferred
    for base_final, transfer_final in [
        (p_oh_base_final.nn, p_oh_transfer_final.nn),
        (p_rad_base_final.nn, p_rad_transfer_final.nn),
        (sciml_base_final.taue_predictor.nn, sciml_transfer_final.taue_predictor.nn),
    ]:
        for i in range(len(base_final.layers) - 1):
            chex.assert_trees_all_equal(base_final.layers[i], transfer_final.layers[i])
        with pytest.raises(AssertionError):
            chex.assert_trees_all_equal(base_final.layers[-1], transfer_final.layers[-1])

    # 2. Ensure the submodules were initialized properly
    chex.assert_trees_all_equal(p_oh_base_final.nn, sciml_base_init.p_oh_predictor.nn)
    chex.assert_trees_all_equal(p_rad_base_final.nn, sciml_base_init.p_rad_predictor.nn)
    chex.assert_trees_all_equal(p_oh_transfer_final.nn, sciml_transfer_init.p_oh_predictor.nn)
    chex.assert_trees_all_equal(p_rad_transfer_final.nn, sciml_transfer_init.p_rad_predictor.nn)

    if freeze_submodules:
        # 3a. Frozen submodules do not change at all during the sciml transfer
        chex.assert_trees_all_equal(p_oh_transfer_final.nn, sciml_transfer_final.p_oh_predictor.nn)
        chex.assert_trees_all_equal(p_rad_transfer_final.nn, sciml_transfer_final.p_rad_predictor.nn)
    else:
        # 3b. Unfrozen submodules only change their last layers (still transfer learning)
        for transfer_final, sciml_sub_final in [
            (p_oh_transfer_final.nn, sciml_transfer_final.p_oh_predictor.nn),
            (p_rad_transfer_final.nn, sciml_transfer_final.p_rad_predictor.nn),
        ]:
            for i in range(len(transfer_final.layers) - 1):
                chex.assert_trees_all_equal(transfer_final.layers[i], sciml_sub_final.layers[i])
            with pytest.raises(AssertionError):
                chex.assert_trees_all_equal(transfer_final.layers[-1], sciml_sub_final.layers[-1])

    # Normalizer stats never train, anywhere
    chex.assert_trees_all_equal(sciml_transfer_init.normalizer, sciml_transfer_final.normalizer)
    chex.assert_trees_all_equal(
        sciml_transfer_init.p_oh_predictor.normalizer,
        sciml_transfer_final.p_oh_predictor.normalizer,
    )


def test_collect_results():
    study = PowerBalanceStudy(
        _make_config(
            "test_collect_results",
            model_types=("sciml",),
            training_datasets=("cmod-low1_cmod-low2",),
            domain_adaptation_methods=("mixing",),
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
