import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import fire
import pytest
from loguru import logger

from transport_study import PACKAGE_ROOT
from transport_study.config import StudyConfig, load_config
from transport_study.profile_transfer.profile_study import ProfileStudy, run_study


def test_compatible_configs():
    # Base config to compare against
    cfg1 = ProfileStudy.Config(
        # Super
        study_name="test_study",
        dataset_paths={
            "cmod-low": "path/to/cmod_low.nc",
            "cmod-high": "path/to/cmod_high.nc",
        },
        target_device="cmod-high",
        # ProfileStudy
        working_dir_base=PACKAGE_ROOT / "tests" / "profile_transfer" / "test_compatible_configs",
        training_datasets=("exnihilo", "cmod-low"),
        target_test_set_size=60,
    )

    # Config that should match
    cfg2 = ProfileStudy.Config(
        # Super
        study_name="test_study",
        dataset_paths={
            "cmod-low": "path/to/cmod_low.nc",
            "cmod-high": "path/to/cmod_high.nc",
        },
        target_device="cmod-high",
        # ProfileStudy
        working_dir_base=PACKAGE_ROOT / "tests" / "profile_transfer" / "test_compatible_configs",
        training_datasets=("exnihilo", "cmod-low"),
        target_test_set_size=60,
    )
    assert cfg1.is_compatible(cfg2)

    # Config that differs in name
    cfg3 = ProfileStudy.Config(
        # Super
        study_name="test_study_different_name",
        dataset_paths={
            "cmod-low": "path/to/cmod_low.nc",
            "cmod-high": "path/to/cmod_high.nc",
        },
        target_device="cmod-high",
        # ProfileStudy
        working_dir_base=PACKAGE_ROOT / "tests" / "profile_transfer" / "test_compatible_configs",
        training_datasets=("exnihilo", "cmod-low"),
        target_test_set_size=60,
    )
    assert not cfg1.is_compatible(cfg3)

    # Config that differs in dataset paths
    cfg4 = ProfileStudy.Config(
        # Super
        study_name="test_study",
        dataset_paths={
            "cmod-low": "new/path/to/cmod_low.nc",
            "cmod-high": "path/to/cmod_high.nc",
        },
        target_device="cmod-high",
        # ProfileStudy
        working_dir_base=PACKAGE_ROOT / "tests" / "profile_transfer" / "test_compatible_configs",
        training_datasets=("exnihilo", "cmod-low"),
        target_test_set_size=60,
    )
    assert not cfg1.is_compatible(cfg4)

    # Config that differs in target device
    cfg5 = ProfileStudy.Config(
        # Super
        study_name="test_study",
        dataset_paths={
            "cmod-low": "path/to/cmod_low.nc",
            "cmod-high": "path/to/cmod_high.nc",
        },
        target_device="cmod-low",
        # ProfileStudy
        working_dir_base=PACKAGE_ROOT / "tests" / "profile_transfer" / "test_compatible_configs",
        training_datasets=("exnihilo", "cmod-low"),
        target_test_set_size=60,
    )
    assert not cfg1.is_compatible(cfg5)

    # Config that differs in target test set size
    cfg6 = ProfileStudy.Config(
        # Super
        study_name="test_study",
        dataset_paths={
            "cmod-low": "path/to/cmod_low.nc",
            "cmod-high": "path/to/cmod_high.nc",
        },
        target_device="cmod-high",
        # ProfileStudy
        working_dir_base=PACKAGE_ROOT / "tests" / "profile_transfer" / "test_compatible_configs",
        training_datasets=("exnihilo", "cmod-low"),
        target_test_set_size=20,
    )
    assert not cfg1.is_compatible(cfg6)

    # Config that differs in one of the hyperparameter tuning settings
    cfg7 = ProfileStudy.Config(
        # Super
        study_name="test_study",
        dataset_paths={
            "cmod-low": "path/to/cmod_low.nc",
            "cmod-high": "path/to/cmod_high.nc",
        },
        target_device="cmod-high",
        # ProfileStudy
        working_dir_base=PACKAGE_ROOT / "tests" / "profile_transfer" / "test_compatible_configs",
        training_datasets=("exnihilo", "cmod-low"),
        target_test_set_size=60,
        hyperparam_freeze_shapes=False,
    )
    assert not cfg1.is_compatible(cfg7)

    # Config that differs in training datasets, but that should be fine
    cfg8 = ProfileStudy.Config(
        # Super
        study_name="test_study",
        dataset_paths={
            "cmod-low": "path/to/cmod_low.nc",
            "cmod-high": "path/to/cmod_high.nc",
        },
        target_device="cmod-high",
        # ProfileStudy
        working_dir_base=PACKAGE_ROOT / "tests" / "profile_transfer" / "test_compatible_configs",
        training_datasets=("exnihilo"),
        target_test_set_size=60,
    )
    assert cfg1.is_compatible(cfg8)


def test_config_save_load():
    with TemporaryDirectory() as tmpdir:
        cfg = ProfileStudy.Config(
            study_name="test_study_save_load",
            dataset_paths={
                "cmod-low": "path/to/cmod_low.nc",
                "cmod-high": "path/to/cmod_high.nc",
            },
            target_device="cmod-high",
            working_dir_base=Path(tmpdir) / "working_dir_base",
            training_datasets=("exnihilo", "cmod-low"),
            target_test_set_size=60,
        )
        save_path = Path(tmpdir) / "config.toml"
        cfg.save(save_path)

        loaded_cfg = ProfileStudy.Config.from_toml(save_path)
        assert cfg == loaded_cfg


def test_config_save_load_with_dataset_weights():
    with TemporaryDirectory() as tmpdir:
        cfg = ProfileStudy.Config(
            study_name="test_study_save_load_weights",
            dataset_paths={
                "cmod-low": "path/to/cmod_low.nc",
                "cmod-high": "path/to/cmod_high.nc",
            },
            target_device="cmod-high",
            working_dir_base=Path(tmpdir) / "working_dir_base",
            training_datasets=("exnihilo", "cmod-low"),
            target_test_set_size=60,
            dataset_sizes={"cmod-low": 1000, "cmod-high": 500},
            dataset_fractions={"cmod-low": 0.25, "cmod-high": 0.75},
        )
        save_path = Path(tmpdir) / "config.toml"
        cfg.save(save_path)

        loaded_cfg = ProfileStudy.Config.from_toml(save_path)
        assert loaded_cfg.dataset_sizes == cfg.dataset_sizes
        assert loaded_cfg.dataset_fractions == cfg.dataset_fractions
        assert cfg == loaded_cfg


def test_study_cmod_to_cmod_no_tuning():
    working_dir_base = PACKAGE_ROOT / "tests" / "test_outputs" / "profile_transfer" / "cmod_to_cmod_no_tuning"

    cfg = ProfileStudy.Config(
        study_name="test_study_cmod_to_cmod",
        working_dir_base=working_dir_base,
        dataset_paths={
            "cmod-low": PACKAGE_ROOT / "datasets" / "sample" / "cmod-low1.nc",
            "cmod-high": PACKAGE_ROOT / "datasets" / "sample" / "cmod-high.nc",
        },
        target_device="cmod-high",
        debug=True,
        dry_run=False,
        max_ds_size=20,
        hyperparam_sweeps=2,
        max_epochs=2,
        epochs_per_val=1,
        patience=2,
        model_types=["torax"],
        training_datasets=[
            "exnihilo",
            "cmod-low",
        ],
        dataset_sizes={"cmod-low": 100, "cmod-high": 100},
        domain_adaptation_methods=[None, "mixing", "transfer"],
        num_hp_shots_options=[0, 1, -1],
        target_test_set_size=60,
    )

    study = ProfileStudy(cfg)
    study.setup_directories(
        enable_parallelism=False,
        skip_tuning=True,
        skip_visualization=True,
        clean_sweeps=True,
        clean_models=True,
        clean_results=True,
        clean_figures=True,
    )

    # Ensure each case can be run
    while len(study.get_unfinished_cases()) > 0:
        for case in study.get_unfinished_cases():
            if not study.result_path(case).exists():
                study.run_case(case, skip_tuning=True, enable_parallelism=False)

    ds_final = study.collect_results()
    ds_final.to_netcdf(study.collected_results_path())


def test_study_cmod_to_cmod_with_tuning():
    working_dir_base = PACKAGE_ROOT / "tests" / "test_outputs" / "profile_transfer" / "cmod_to_cmod_with_tuning"

    cfg = ProfileStudy.Config(
        study_name="test_study_cmod_to_cmod",
        working_dir_base=working_dir_base,
        dataset_paths={
            "cmod-low": PACKAGE_ROOT / "datasets" / "sample" / "cmod-low1.nc",
            "cmod-high": PACKAGE_ROOT / "datasets" / "sample" / "cmod-high.nc",
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
        num_hp_shots_options=[0, -1],
        target_test_set_size=60,
    )

    study = ProfileStudy(cfg)
    study.setup_directories(
        enable_parallelism=False,
        skip_tuning=False,
        skip_visualization=True,
        clean_sweeps=True,
        clean_models=True,
        clean_results=True,
        clean_figures=True,
    )

    # Ensure each case can be run
    while len(study.get_unfinished_cases()) > 0:
        for case in study.get_unfinished_cases():
            if not study.result_path(case).exists():
                study.run_case(case, skip_tuning=False, enable_parallelism=False)


def test_study_cmod_to_mast():
    working_dir_base = PACKAGE_ROOT / "tests" / "test_outputs" / "profile_transfer" / "cmod_to_mast"

    cfg = ProfileStudy.Config(
        study_name="test_study_cmod_to_mast",
        working_dir_base=working_dir_base,
        dataset_paths={
            "cmod-low": PACKAGE_ROOT / "datasets" / "sample" / "cmod-low1.nc",
            "cmod-high": PACKAGE_ROOT / "datasets" / "sample" / "cmod-high.nc",
            "mast": PACKAGE_ROOT / ".." / "scratch" / "datasets" / "mast" / "dataset_200" / "ds.zarr",
        },
        target_device="mast",
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
            "cmod-low_cmod-high",
        ],
        dataset_sizes={"cmod-low": 100, "cmod-high": 100, "mast": 41},
        domain_adaptation_methods=[None, "mixing", "transfer"],
        num_hp_shots_options=[0, 1, -1],
        target_test_set_size=20,
    )

    study = ProfileStudy(cfg)
    study.setup_directories(
        enable_parallelism=False,
        skip_tuning=True,
        skip_visualization=True,
        clean_sweeps=True,
        clean_models=True,
        clean_results=True,
        clean_figures=True,
    )

    # Ensure each case can be run
    while len(study.get_unfinished_cases()) > 0:
        for case in study.get_unfinished_cases():
            if not study.result_path(case).exists():
                study.run_case(case, skip_tuning=True, enable_parallelism=False)

    ds_final = study.collect_results()
    ds_final.to_netcdf(study.collected_results_path())
