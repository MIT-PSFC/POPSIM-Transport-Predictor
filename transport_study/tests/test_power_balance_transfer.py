import numpy as np
import pytest
import os
from transport_study import PACKAGE_ROOT
import chex

from transport_study.config import config

from transport_study.power_balance_transfer.run_study import PowerBalanceStudy


def test_power_balance_transfer_cases():
    """Make sure cases are set up correctly"""

    study = PowerBalanceStudy(
        name="xfer_test",
        working_dir_base=os.path.join(PACKAGE_ROOT, "tests", "test_outputs"),
        dataset_paths={
            "cmod": config.cmod_dataset_path,
            "tcv": config.tcv_dataset_path,
            "d3d_hp": config.d3d_hp_dataset_path,
        },
        model_types=["sciml", "unstructured_nn"],
        training_datasets=["cmod", "cmod_tcv"],
        data_normalization_methods=["raw", "coral"],
        domain_adaptation_methods=[None, "mixing", "transfer"],
        freeze_submodules_options=[True, False],
        num_hp_shots_options=[0, -1],
        hp_test_set_size=4,
    )

    # Ensure that each case with a prereq, has that prereq in the list of cases
    for case in study.cases:
        if case.prereqs is not None:
            for prereq in case.prereqs:
                assert prereq in study.cases, (
                    f"{case}\nhas prereq\n{prereq}\nwhich is not in the list of cases"
                )

    # Ensure that there are no duplicate cases
    assert len(study.cases) == len(set(study.cases)), (
        "There are duplicate cases in the study"
    )

    # Ensure there is only one hyperparameter tuning case per model type
    hp_tuning_cases = [case for case in study.cases if case.is_hyperparam_case()]
    hp_tuning_cases_by_model = {}
    for case in hp_tuning_cases:
        if case.model_type in hp_tuning_cases_by_model:
            raise ValueError(
                f"Multiple hyperparameter tuning cases found for model type {case.model_type}: {case} and {hp_tuning_cases_by_model[case.model_type]}"
            )
        hp_tuning_cases_by_model[case.model_type] = case

    # For a couple special cases, ensure the prereqs are set up correctly
    for case in study.cases:
        # Check that hyperparameter tuning case prereqs are set up correctly
        if case.is_hyperparam_case():
            if case.model_type in ["p_rad", "p_oh", "unstructured_nn"]:
                assert case.prereqs is None, (
                    f"Hyperparameter tuning case {case} for model type {case.model_type} should not have any prereqs, but has {case.prereqs}"
                )
                continue
            elif case.model_type in ["sciml", "scaling_law"]:
                expected_prereqs = [
                    PowerBalanceStudy.Case(
                        model_type="p_oh",
                        training_data=PowerBalanceStudy.HYPERPARAM_TRAINING_DATA,
                        data_normalization=PowerBalanceStudy.HYPERPARAM_DATA_NORMALIZATION,
                        domain_adaptation=PowerBalanceStudy.HYPERPARAM_DOMAIN_ADAPTATION,
                        freeze_submodules=PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES,
                        num_hp_shots=PowerBalanceStudy.HYPERPARAM_NUM_HP_SHOTS,
                    ),
                    PowerBalanceStudy.Case(
                        model_type="p_rad",
                        training_data=PowerBalanceStudy.HYPERPARAM_TRAINING_DATA,
                        data_normalization=PowerBalanceStudy.HYPERPARAM_DATA_NORMALIZATION,
                        domain_adaptation=PowerBalanceStudy.HYPERPARAM_DOMAIN_ADAPTATION,
                        freeze_submodules=PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES,
                        num_hp_shots=PowerBalanceStudy.HYPERPARAM_NUM_HP_SHOTS,
                    ),
                ]
        # Check that transfer learning depends on previous training case with same model type and training dataset, but no domain adaptation
        elif case.domain_adaptation == "transfer":
            expected_prereqs = [
                # The original training case with no domain adaptation to re-init the weights
                PowerBalanceStudy.Case(
                    model_type=case.model_type,
                    training_data=case.training_data,
                    data_normalization=case.data_normalization,
                    domain_adaptation=None,
                    freeze_submodules=case.freeze_submodules,
                    num_hp_shots=-1,
                ),
                # The hyperparameter tuning case
                PowerBalanceStudy.Case(
                    model_type=case.model_type,
                    training_data=PowerBalanceStudy.HYPERPARAM_TRAINING_DATA,
                    data_normalization=PowerBalanceStudy.HYPERPARAM_DATA_NORMALIZATION,
                    domain_adaptation=PowerBalanceStudy.HYPERPARAM_DOMAIN_ADAPTATION,
                    freeze_submodules=PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES,
                    num_hp_shots=PowerBalanceStudy.HYPERPARAM_NUM_HP_SHOTS,
                ),
            ]
            if case.model_type in ["sciml", "scaling_law"]:
                expected_prereqs += [
                    PowerBalanceStudy.Case(
                        model_type="p_oh",
                        training_data=case.training_data,
                        data_normalization=case.data_normalization,
                        domain_adaptation=case.domain_adaptation,
                        freeze_submodules=PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES,
                        num_hp_shots=case.num_hp_shots,
                    ),
                    PowerBalanceStudy.Case(
                        model_type="p_rad",
                        training_data=case.training_data,
                        data_normalization=case.data_normalization,
                        domain_adaptation=case.domain_adaptation,
                        freeze_submodules=PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES,
                        num_hp_shots=case.num_hp_shots,
                    ),
                ]
        elif case.domain_adaptation in ["mixing", None]:
            # For mixing and no domain adaptation, simply requires hyperparameter tuning
            expected_prereqs = [
                PowerBalanceStudy.Case(
                    model_type=case.model_type,
                    training_data=PowerBalanceStudy.HYPERPARAM_TRAINING_DATA,
                    data_normalization=PowerBalanceStudy.HYPERPARAM_DATA_NORMALIZATION,
                    domain_adaptation=PowerBalanceStudy.HYPERPARAM_DOMAIN_ADAPTATION,
                    freeze_submodules=PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES,
                    num_hp_shots=PowerBalanceStudy.HYPERPARAM_NUM_HP_SHOTS,
                ),
            ]
            if case.model_type in ["sciml", "scaling_law"]:
                expected_prereqs += [
                    PowerBalanceStudy.Case(
                        model_type="p_oh",
                        training_data=case.training_data,
                        data_normalization=case.data_normalization,
                        domain_adaptation=case.domain_adaptation,
                        freeze_submodules=PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES,
                        num_hp_shots=case.num_hp_shots,
                    ),
                    PowerBalanceStudy.Case(
                        model_type="p_rad",
                        training_data=case.training_data,
                        data_normalization=case.data_normalization,
                        domain_adaptation=case.domain_adaptation,
                        freeze_submodules=PowerBalanceStudy.HYPERPARAM_FREEZE_SUBMODULES,
                        num_hp_shots=case.num_hp_shots,
                    ),
                ]
        else:
            continue

        if set(case.prereqs) != set(expected_prereqs):
            print(f"{case}\n should have prereqs")
            for prereq in expected_prereqs:
                print(f"{prereq}")
            print(f"but has prereqs")
            for prereq in case.prereqs:
                print(f"{prereq}")
            raise AssertionError


def test_mix_device_weight():
    study = PowerBalanceStudy(
        name="xfer_test",
        working_dir_base=os.path.join(PACKAGE_ROOT, "tests", "test_outputs"),
        dataset_paths={
            "cmod": config.cmod_dataset_path,
            "tcv": config.tcv_dataset_path,
            "d3d_hp": config.d3d_hp_dataset_path,
        },
        model_types=["sciml"],
        training_datasets=["cmod_tcv"],
        data_normalization_methods=["coral"],
        domain_adaptation_methods=["mixing"],
        freeze_submodules_options=[True],
        num_hp_shots_options=[3],
        hp_test_set_size=4,
    )

    case_p_oh = PowerBalanceStudy.Case(
        model_type="p_oh",
        training_data="cmod_tcv",
        data_normalization="coral",
        domain_adaptation="mixing",
        freeze_submodules=True,
        num_hp_shots=3,
    )

    study.launch_train(case_p_oh)


def test_submodule_freezing():
    study = PowerBalanceStudy(
        name="xfer_test",
        working_dir_base=os.path.join(PACKAGE_ROOT, "tests", "test_outputs"),
        dataset_paths={
            "cmod": config.cmod_dataset_path,
            "tcv": config.tcv_dataset_path,
            "d3d_hp": config.d3d_hp_dataset_path,
        },
        model_types=["sciml"],
        training_datasets=["cmod_tcv"],
        data_normalization_methods=["coral"],
        domain_adaptation_methods=["mixing"],
        freeze_submodules_options=[True, False],
        num_hp_shots_options=[3],
        hp_test_set_size=4,
    )

    case_p_oh = PowerBalanceStudy.Case(
        model_type="p_oh",
        training_data="cmod_tcv",
        data_normalization="coral",
        domain_adaptation=None,
        freeze_submodules=True,
        num_hp_shots=-1,
    )

    case_p_rad = PowerBalanceStudy.Case(
        model_type="p_rad",
        training_data="cmod_tcv",
        data_normalization="coral",
        domain_adaptation=None,
        freeze_submodules=True,
        num_hp_shots=-1,
    )

    case_frozen = PowerBalanceStudy.Case(
        model_type="sciml",
        training_data="cmod_tcv",
        data_normalization="coral",
        domain_adaptation=None,
        freeze_submodules=True,
        num_hp_shots=-1,
    )

    case_unfrozen = PowerBalanceStudy.Case(
        model_type="sciml",
        training_data="cmod_tcv",
        data_normalization="coral",
        domain_adaptation=None,
        freeze_submodules=False,
        num_hp_shots=-1,
    )

    for case in [case_p_oh, case_p_rad, case_frozen, case_unfrozen]:
        if not os.path.exists(study.result_path(case)):
            study.launch_train(case)

    # Load the trained models for each case,
    # Ensure the weights for p_oh and p_rad are the same for the frozen case,
    # and different for the unfrozen case
    trainer_p_oh = study.restore_trainer(case_p_oh)
    trainer_p_rad = study.restore_trainer(case_p_rad)
    trainer_frozen = study.restore_trainer(case_frozen)
    trainer_unfrozen = study.restore_trainer(case_unfrozen)


def test_transfer_weights():
    study = PowerBalanceStudy(
        name="xfer_test",
        working_dir_base=os.path.join(PACKAGE_ROOT, "tests", "test_outputs"),
        dataset_paths={
            "cmod": config.cmod_dataset_path,
            "tcv": config.tcv_dataset_path,
            "d3d_hp": config.d3d_hp_dataset_path,
        },
        model_types=["sciml", "unstructured_nn"],
        training_datasets=["cmod_tcv"],
        data_normalization_methods=["coral"],
        domain_adaptation_methods=["mixing"],
        freeze_submodules_options=[True, False],
        num_hp_shots_options=[3],
        hp_test_set_size=4,
    )

    case_p_oh_orig = PowerBalanceStudy.Case(
        model_type="p_oh",
        training_data="cmod_tcv",
        data_normalization="coral",
        domain_adaptation=None,
        freeze_submodules=True,
        num_hp_shots=-1,
    )
    case_p_oh_transfer = PowerBalanceStudy.Case(
        model_type="p_oh",
        training_data="cmod_tcv",
        data_normalization="coral",
        domain_adaptation="transfer",
        freeze_submodules=True,
        num_hp_shots=-1,
    )
    case_p_rad_orig = PowerBalanceStudy.Case(
        model_type="p_rad",
        training_data="cmod_tcv",
        data_normalization="coral",
        domain_adaptation=None,
        freeze_submodules=True,
        num_hp_shots=-1,
    )
    case_p_rad_transfer = PowerBalanceStudy.Case(
        model_type="p_rad",
        training_data="cmod_tcv",
        data_normalization="coral",
        domain_adaptation="transfer",
        freeze_submodules=True,
        num_hp_shots=-1,
    )

    case_sciml_orig = PowerBalanceStudy.Case(
        model_type="sciml",
        training_data="cmod_tcv",
        data_normalization="coral",
        domain_adaptation=None,
        freeze_submodules=True,
        num_hp_shots=-1,
    )
    case_sciml_transfer_frozen = PowerBalanceStudy.Case(
        model_type="sciml",
        training_data="cmod_tcv",
        data_normalization="coral",
        domain_adaptation="transfer",
        freeze_submodules=True,
        num_hp_shots=-1,
    )
    case_sciml_transfer_unfrozen = PowerBalanceStudy.Case(
        model_type="sciml",
        training_data="cmod_tcv",
        data_normalization="coral",
        domain_adaptation="transfer",
        freeze_submodules=False,
        num_hp_shots=-1,
    )

    case_unstructured_nn_orig = PowerBalanceStudy.Case(
        model_type="unstructured_nn",
        training_data="cmod_tcv",
        data_normalization="coral",
        domain_adaptation=None,
        freeze_submodules=True,
        num_hp_shots=-1,
    )
    case_unstructured_nn_transfer = PowerBalanceStudy.Case(
        model_type="unstructured_nn",
        training_data="cmod_tcv",
        data_normalization="coral",
        domain_adaptation="transfer",
        freeze_submodules=True,
        num_hp_shots=-1,
    )

    # Restore models for each case
    p_oh_orig = study.restore_trainer(case_p_oh_orig).model
    p_oh_transfer = study.restore_trainer(case_p_oh_transfer).model
    p_rad_orig = study.restore_trainer(case_p_rad_orig).model
    p_rad_transfer = study.restore_trainer(case_p_rad_transfer).model
    sciml_orig = study.restore_trainer(case_sciml_orig).model
    sciml_transfer_frozen = study.restore_trainer(case_sciml_transfer_frozen).model
    sciml_transfer_unfrozen = study.restore_trainer(case_sciml_transfer_unfrozen).model
    unstructured_nn_orig = study.restore_trainer(case_unstructured_nn_orig).model
    unstructured_nn_transfer = study.restore_trainer(
        case_unstructured_nn_transfer
    ).model


def test_collect_results():
    study = PowerBalanceStudy(
        name="test_collect_results",
        working_dir_base=os.path.join(PACKAGE_ROOT, "tests", "test_outputs"),
        dataset_paths={
            "cmod": config.cmod_dataset_path,
            "tcv": config.tcv_dataset_path,
            "d3d_hp": config.d3d_hp_dataset_path,
        },
        model_types=["sciml"],
        training_datasets=["cmod_tcv"],
        data_normalization_methods=["coral"],
        domain_adaptation_methods=["mixing"],
        freeze_submodules_options=[True],
        num_hp_shots_options=[3],
        hp_test_set_size=4,
    )

    unfinished_cases = [
        case
        for case in study.cases
        if not os.path.exists(study.result_path(case))
        and study.check_data_requirements(case)
    ]
    while len(unfinished_cases) > 0:
        for case in unfinished_cases:
            study.run_case(case, skip_tuning=True, enable_parallelism=False)
        unfinished_cases = [
            case
            for case in unfinished_cases
            if not os.path.exists(study.result_path(case))
        ]

    ds_merged = study.collect_results()

    for case in study.cases:
        assert case.model_type in ds_merged.coords["model_type"].values, (
            f"{case.model_type} not found in collected results"
        )
        assert case.training_data in ds_merged.coords["training_data"].values, (
            f"{case.training_data} not found in collected results"
        )
        assert (
            case.data_normalization in ds_merged.coords["data_normalization"].values
        ), f"{case.data_normalization} not found in collected results"
        assert case.domain_adaptation in ds_merged.coords["domain_adaptation"].values, (
            f"{case.domain_adaptation} not found in collected results"
        )
        assert case.freeze_submodules in ds_merged.coords["freeze_submodules"].values, (
            f"{case.freeze_submodules} not found in collected results"
        )
        assert case.num_hp_shots in ds_merged.coords["num_hp_shots"].values, (
            f"{case.num_hp_shots} not found in collected results"
        )

    for var in ds_merged.data_vars:
        # Assert the value is positive
        assert (ds_merged[var] >= 0).all()


if __name__ == "__main__":
    # test_power_balance_transfer_cases()
    test_collect_results()
    # TODO(ZanderKeith), make sure the following things are happening:
    # 1) Cases properly restore their hyperparameters
    # 2) submodules within cases get their proper hyperparameters and restore properly
