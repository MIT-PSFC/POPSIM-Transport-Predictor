"""End-to-end runs of the profile transfer study on the sample datasets.

Every case in a small grid is trained for a couple of epochs and collected, so
the orchestration path (case grid, prereq chain, training, result collection)
is exercised for each model family. Slow: these actually train.
"""

import pytest

from transport_study import PACKAGE_ROOT
from transport_study.profile_transfer.profile_study import ProfileStudy
from transport_study.tests.sample_data import SAMPLE_DIR, requires_sample_data

WORKING_DIR_BASE = PACKAGE_ROOT / "tests" / "test_outputs" / "profile_transfer"

pytestmark = [pytest.mark.slow, requires_sample_data]


def _make_config(study_name: str, working_dir_name: str, **overrides) -> ProfileStudy.Config:
    """Small case grid over the sample datasets, quick enough for smoke training."""
    defaults = dict(
        study_name=study_name,
        working_dir_base=WORKING_DIR_BASE / working_dir_name,
        dataset_paths={
            "cmod-low": SAMPLE_DIR / "cmod-low1.nc",
            "cmod-high": SAMPLE_DIR / "cmod-high.nc",
        },
        target_device="cmod-high",
        debug=True,
        max_ds_size=20,
        hyperparam_sweeps=2,
        max_epochs=2,
        epochs_per_val=1,
        patience=2,
        training_datasets=["exnihilo", "cmod-low"],
        domain_adaptation_methods=[None, "weighted", "transfer"],
        num_target_shots_options=[0, 1, -1],
        # max_ds_size truncates the target device to 20 shots, so the test set
        # holdout must leave train candidates for the num_target_shots=1 case
        target_test_set_size=10,
    )
    defaults.update(overrides)
    return ProfileStudy.Config(**defaults)


def _run_every_case(study: ProfileStudy, skip_tuning: bool):
    while study.get_unfinished_cases():
        for case in study.get_unfinished_cases():
            if not study.result_path(case).exists():
                study.run_case(case, skip_tuning=skip_tuning, enable_parallelism=False)


@pytest.mark.parametrize(
    "model_type",
    [
        "shape-init-pca",
        "shape-init-kmeans",
        "mlp",
        "torax-constant",
        "torax-gyrobohm",
        "torax-qlknn",
    ],
)
def test_study_cmod_to_cmod_no_tuning(model_type):
    cfg = _make_config(
        f"test_study_cmod_to_cmod_{model_type}",
        "cmod_to_cmod_no_tuning",
        model_types=[model_type],
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

    _run_every_case(study, skip_tuning=True)

    study.collect_results().to_netcdf(study.collected_results_path())


def test_study_cmod_to_cmod_with_tuning():
    cfg = _make_config(
        "test_study_cmod_to_cmod_tuned",
        "cmod_to_cmod_with_tuning",
        model_types=["mlp"],
        num_target_shots_options=[0, -1],
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

    _run_every_case(study, skip_tuning=False)


def test_study_cmod_to_mast():
    """Cross-device transfer: two C-Mod sources, MAST as the target device."""
    cfg = _make_config(
        "test_study_cmod_to_mast",
        "cmod_to_mast",
        dataset_paths={
            "cmod-low": SAMPLE_DIR / "cmod-low1.nc",
            "cmod-high": SAMPLE_DIR / "cmod-high.nc",
            "mast": SAMPLE_DIR / "mast-high.nc",
        },
        target_device="mast",
        model_types=["mlp"],
        training_datasets=["exnihilo", "cmod-low_cmod-high"],
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

    _run_every_case(study, skip_tuning=True)

    study.collect_results().to_netcdf(study.collected_results_path())


def test_transfer_trainable_getter_is_last_layer_only():
    """ProfilePredictorTRB.get_trainable_getter with domain_adaptation='transfer'
    returns only the last-layer leaves of every network in the module
    (module.nn for shape-init / mlp / reservoir, nn_transport + nn_sources +
    nn_edge for torax-*), matching PowerBalanceEnv.get_trainable. Shapes,
    reservoir weights, and the normalizer statistics restored from the pretrain
    checkpoint stay frozen regardless of freeze_shapes. Every other
    domain_adaptation keeps the per-family getter.
    """
