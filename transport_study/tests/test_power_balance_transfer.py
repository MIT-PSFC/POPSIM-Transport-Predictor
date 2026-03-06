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
        name="test_power_balance_transfer",
        working_dir_base=os.path.join(PACKAGE_ROOT, "tests", "test_outputs"),
        dataset_paths={
            "cmod": config.cmod_dataset_path,
            "tcv": config.tcv_dataset_path,
            "d3d_hp": config.d3d_hp_dataset_path,
        },
        model_types=["scaling_law", "sciml", "unstructured_nn"],
        training_datasets=["cmod", "cmod_tcv"],
        data_normalization_methods=["raw", "coral"],
        domain_adaptation_methods=[None, "mixing", "transfer"],
        freeze_submodules_options=[True, False],
        num_hp_shots_options=[0, None],
        debug=True,
    )

    # Ensure that each case with a prereq, has that prereq in the list of cases
    for case in study.cases:
        if case.prereq is not None:
            assert case.prereq in study.cases, (
                f"Case {case} has prereq {case.prereq} which is not in the list of cases"
            )

    # Ensure that there are no duplicate cases
    assert len(study.cases) == len(set(study.cases)), (
        "There are duplicate cases in the study"
    )

    # Ensure there is only one hyperparameter tuning case per model type
    # Hyperparameter tuning is defined as cases with no domain adaptation (train and test on same device) and no prereq (this is the one that others depend on)
    hp_tuning_cases = [
        case
        for case in study.cases
        if case.domain_adaptation is None and case.prereq is None
    ]
    hp_tuning_cases_by_model = {}
    for case in hp_tuning_cases:
        if case.model_type in hp_tuning_cases_by_model:
            raise ValueError(
                f"Multiple hyperparameter tuning cases found for model type {case.model_type}: {case} and {hp_tuning_cases_by_model[case.model_type]}"
            )
        hp_tuning_cases_by_model[case.model_type] = case


if __name__ == "__main__":
    test_power_balance_transfer_cases()
