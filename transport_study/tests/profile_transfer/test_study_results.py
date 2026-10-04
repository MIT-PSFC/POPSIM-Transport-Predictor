import os
from pathlib import Path

import pytest
import xarray as xr

from transport_study import PACKAGE_ROOT
from transport_study.orchestration.case_metrics import collected_metrics_path
from transport_study.orchestration.stages import STAGE_AGG_NAMES
from transport_study.profile_transfer.profile_study import ProfileStudy, run_study
from transport_study.tests.sample_data import SAMPLE_DIR, requires_sample_data


@pytest.mark.slow()
@requires_sample_data
class TestCmodToCmod:
    """Test to ensure all the important results are generated for a cmod to cmod profile transfer study"""

    def test_plots(self):
        working_dir_base = os.environ.get("PTPS_TEST_WORKING_DIR_BASE", None)
        if working_dir_base is None:
            working_dir_base = PACKAGE_ROOT / "tests" / "test_outputs" / "profile_transfer" / "test_study_results"

        study_config = ProfileStudy.Config(
            study_name="test_results_cmod_to_cmod_plots",
            working_dir_base=Path(working_dir_base),
            dataset_paths={
                "cmod-low1": SAMPLE_DIR / "cmod-low1.nc",
                "cmod-low2": SAMPLE_DIR / "cmod-low2.nc",
                "cmod-high": SAMPLE_DIR / "cmod-high.nc",
            },
            target_device="cmod-high",
            debug=True,
            # Must comfortably exceed target_test_set_size so a train candidate
            # pool remains after the test split (sample datasets have 100 shots)
            max_ds_size=100,
            hyperparam_sweeps=4,
            max_epochs=10,
            epochs_per_val=1,
            patience=2,
            model_types=[
                "shape-init-pca",
                "shape-init-kmeans",
                "mlp",
                "reservoir",
                "torax-constant",
                "torax-gyrobohm",
                "torax-qlknn",
            ],
            training_datasets=[
                "exnihilo",
                "cmod-low1",
                "cmod-low1_cmod-low2",
            ],
            domain_adaptation_methods=[None, "weighted", "transfer"],
            num_target_shots_options=[0, 1, 10, -1],
            target_test_set_size=60,
        )

        # Run study, reusing the same working directory to test that results are not overwritten and figures are regenerated
        run_study(
            config=study_config,
            enable_parallelism=True,
            skip_tuning=True,
            skip_visualization=False,
            clean_sweeps=False,
            clean_models=False,
            clean_results=False,
            clean_figures=False,
        )

        study = ProfileStudy(study_config)
        figure_dir = study.figure_dir

        # Ensure data visualization figures exist
        assert any((figure_dir / "data_visualization" / "domain_overlap").glob("*.png")), "Data visualization figures were not generated"

        # Stage-resolved metrics collected and cached for every finished case
        metrics_path = collected_metrics_path(study)
        assert metrics_path.exists(), "collected_metrics.nc was not generated"
        metrics_ds = xr.load_dataset(metrics_path)
        assert list(metrics_ds["stage"].values) == list(STAGE_AGG_NAMES)
        finished_cases = [case for case in study.cases if study.result_path(case).exists()]
        assert len(finished_cases) > 0
        # Cases whose predictions are all-NaN (diverged) are skipped by collect_metrics
        assert 0 < metrics_ds.sizes["case_idx"] <= len(finished_cases)

        # Comparison figures for every family
        for family in (
            "model_comparison",
            "training_dataset_comparison",
            "domain_adaptation_comparison",
        ):
            family_dir = figure_dir / "comparison" / family
            assert any(family_dir.glob("*.png")), f"No {family} figures were generated"

        # Per-case reports: best/worst timeslice PDF and profile evolution GIFs
        # for every case that produced valid metrics (case_idx indexes study.cases)
        for case_idx in metrics_ds["case_idx"].values:
            case = study.cases[int(case_idx)]
            case_dir = figure_dir / "case_reports" / str(case)
            assert (case_dir / "best_worst_timeslices.pdf").exists(), f"Missing best/worst PDF for case {case}"
            assert any(case_dir.glob("shot_*_evolution.gif")), f"Missing evolution GIFs for case {case}"

        # TORAX relaxation figure for the best torax case
        assert any((figure_dir / "torax").glob("relaxation_*.png")), "TORAX relaxation figure was not generated"
