"""The statistics the training losses read come from a case's own training shots (Study.training_loss_statistics).

The target device's held-out test shots must never enter them.
"""

import numpy as np

from transport_study import EPISODE_DIM
from transport_study.orchestration.organize_data import get_ds
from transport_study.orchestration.target_shots import TargetSplit, target_shot_split
from transport_study.power_balance_transfer.power_balance_study import PowerBalanceStudy


def test_target_scale_comes_from_its_training_shot_alone(synthetic_device_stores, tmp_path):
    # The two highest-hazard mast shots are the test set, the third is the one training shot of a targ_1 case
    study = PowerBalanceStudy(
        PowerBalanceStudy.Config(
            study_name="test-training-loss-statistics",
            working_dir_base=tmp_path,
            dataset_paths=synthetic_device_stores,
            target_device="mast",
            target_test_set_size=2,
            training_datasets=("cmod",),
            domain_adaptation_methods=("addition",),
            model_types=("mlp",),
            data_normalization_methods=("physics",),
            num_target_shots_options=(1,),
        )
    )
    case = next(case for case in study.cases if case.domain_adaptation == "addition")
    dataloader_config = study.model_train_spec(case, study.base_dataloader_config(case)).dataloader_config

    scales = study.training_loss_statistics(dataloader_config)["residual_scales"]

    ds_target = get_ds("mast", "power_balance_transfer")
    target_split = TargetSplit.from_dataloader_config(dataloader_config)
    train_shots, test_shots = target_shot_split(ds_target, target_split)
    energy_by_shot = ds_target["energy_mhd_MJ"].max("time_idx")
    train_energy = float(energy_by_shot.sel({EPISODE_DIM: train_shots}).item())
    test_energies = energy_by_shot.sel({EPISODE_DIM: test_shots}).values
    # Every synthetic shot holds one stored energy, and the test shots hold larger ones
    assert np.isclose(scales["mast"]["energy_mhd_MJ"], train_energy)
    assert (test_energies > train_energy).all()
    assert set(scales) == {"cmod", "mast"}
