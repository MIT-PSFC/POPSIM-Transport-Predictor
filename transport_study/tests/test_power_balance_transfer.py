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
        num_hp_shots_options=[0, None],
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
                    num_hp_shots=None,
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
        num_hp_shots=None,
    )

    case_p_rad = PowerBalanceStudy.Case(
        model_type="p_rad",
        training_data="cmod_tcv",
        data_normalization="coral",
        domain_adaptation=None,
        freeze_submodules=True,
        num_hp_shots=None,
    )

    case_frozen = PowerBalanceStudy.Case(
        model_type="sciml",
        training_data="cmod_tcv",
        data_normalization="coral",
        domain_adaptation=None,
        freeze_submodules=True,
        num_hp_shots=None,
    )


if __name__ == "__main__":
    # test_power_balance_transfer_cases()
    test_mix_device_weight()
    # TODO(ZanderKeith), make sure the following things are happening:
    # 1) Cases properly restore their hyperparameters
    # 2) submodules within cases get their proper hyperparameters and restore properly
