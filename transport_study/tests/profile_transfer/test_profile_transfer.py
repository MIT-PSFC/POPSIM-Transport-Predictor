import fire
import pytest

from transport_study import PACKAGE_ROOT
from transport_study.config import StudyConfig, load_config
from transport_study.profile_transfer.profile_study import ProfileStudy, run_study


def test_study_cmod_to_cmod():
    print("beans")
    cfg = ProfileStudy.Config(
        study_name="test_study_cmod_to_cmod",
        working_dir_base=PACKAGE_ROOT / "tests" / "profile_transfer" / "working_dir_base",
        dataset_paths={
            "cmod-low": PACKAGE_ROOT / "datasets" / "sample" / "cmod_low_1.nc",
            "cmod-high": PACKAGE_ROOT / "datasets" / "sample" / "cmod_high.nc",
        },
        target_device="cmod-high",
        debug=True,
        dry_run=False,
        max_ds_size=20,
        hyperparam_sweeps=2,
        max_epochs=2,
        epochs_per_val=1,
        patience=2,
        model_types=["unstructured_nn"],
        training_datasets=[
            "exnihilo",
            "cmod-low",
        ],
        domain_adaptation_methods=[None, "mixing", "transfer"],
        num_hp_shots_options=[0, 1, -1],
        hp_test_set_size=60,
    )

    study = ProfileStudy(cfg)


if __name__ == "__main__":
    fire.Fire(
        {
            "test_study_cmod_to_cmod": test_study_cmod_to_cmod,
        }
    )
