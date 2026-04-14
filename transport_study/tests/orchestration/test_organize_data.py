import os

import pytest

from transport_study import EPISODE_DIM, PACKAGE_ROOT, TIME_COORD, TIME_DIM
from transport_study.orchestration import organize_data
from transport_study.orchestration.organize_data import (
    DatasetConfig,
    TrainingData,
    add_performance,
    get_ds,
    get_train_val_datasets,
)

# Sample dataset paths (real .nc files checked into git lfs)
SAMPLE_NAMES = ["cmod_low_1", "cmod_low_2", "cmod_high", "cmod_recent"]
SAMPLE_PATHS = {
    name: os.path.join(PACKAGE_ROOT, "datasets", "sample", f"{name}.nc")
    for name in SAMPLE_NAMES
}


@pytest.fixture()
def sample_dataset_config(monkeypatch):
    """Replace the module-level dataset_config singleton with one pointing
    at the checked-in sample .nc files so tests never need live env vars."""
    sorted_names = sorted(SAMPLE_PATHS)
    cfg = DatasetConfig(
        dataset_paths={k: str(v) for k, v in SAMPLE_PATHS.items()},
        ds_source_to_idx={name: i for i, name in enumerate(sorted_names)},
        target_device=None,
    )
    monkeypatch.setattr(organize_data, "dataset_config", cfg)
    return cfg


class TestGetDs:
    @pytest.mark.parametrize("source_ds", SAMPLE_NAMES)
    @pytest.mark.parametrize(
        "study_type", ["profile_transfer", "power_balance_transfer"]
    )
    def test_get_ds_returns_dataset_and_dims(
        self, sample_dataset_config, source_ds, study_type
    ):
        ds, episode_dim = get_ds(source_ds, study_type, debug=True)
        assert ds is not None
        assert episode_dim == EPISODE_DIM
        assert TIME_COORD in ds.coords
        assert TIME_DIM in ds.dims

    @pytest.mark.parametrize(
        "study_type", ["profile_transfer", "power_balance_transfer"]
    )
    def test_get_ds_unknown_source_raises(self, sample_dataset_config, study_type):
        with pytest.raises(ValueError, match="Unknown source dataset"):
            get_ds("nonexistent_device", study_type, debug=True)

    def test_get_ds_profile_transfer_has_shape_vars(self, sample_dataset_config):
        ds, _ = get_ds("cmod_low_1", "profile_transfer", debug=True)
        assert "Te_shape" in ds
        assert "ne_shape" in ds
        assert "Te_keV_line_avg" in ds

    def test_get_ds_power_balance_has_aux_power(self, sample_dataset_config):
        ds, _ = get_ds("cmod_low_1", "power_balance_transfer", debug=True)
        assert "P_aux_MW" in ds

    def test_get_ds_debug_limits_shots(self, sample_dataset_config):
        ds, _ = get_ds("cmod_low_1", "power_balance_transfer", debug=True)
        assert ds.sizes["shot"] <= 10


class TestAddPerformance:
    @pytest.mark.parametrize(
        "study_type", ["profile_transfer", "power_balance_transfer"]
    )
    def test_add_performance_adds_variables(self, sample_dataset_config, study_type):
        ds, episode_coord = get_ds("cmod_low_1", study_type, debug=True)
        result = add_performance(ds, episode_coord)
        assert "performance" in result
        assert "Ip_MA_p95" in result
        assert "Wtot_MJ_p95" in result
        assert result["performance"].dims == (episode_coord,)

    def test_add_performance_nonnegative(self, sample_dataset_config):
        ds, episode_coord = get_ds("cmod_recent", "power_balance_transfer", debug=True)
        result = add_performance(ds, episode_coord)
        valid = result["performance"].values
        assert all(v >= 0 for v in valid if not __import__("math").isnan(v))


class TestTrainingData:
    def test_training_data_known_sources_accepted(self, sample_dataset_config):
        td = TrainingData(sources=frozenset({"cmod_low_1", "cmod_high"}))
        assert "cmod_low_1" in td.sources

    def test_training_data_unknown_source_raises(self, sample_dataset_config):
        with pytest.raises(ValueError, match="Unknown dataset sources"):
            TrainingData(sources=frozenset({"d3d"}))

    def test_training_data_str_sorted(self, sample_dataset_config):
        td = TrainingData(sources=frozenset({"cmod_recent", "cmod_high"}))
        assert str(td) == "cmod_high_cmod_recent"

    def test_training_data_exnihilo_str(self, sample_dataset_config):
        td = TrainingData(sources=frozenset({"cmod_low_1"}), exnihilo=True)
        assert str(td) == "exnihilo"


class TestGetTrainValDatasets:
    @pytest.mark.parametrize("normalization", ["raw", "z_score"])
    def test_get_train_val_datasets_returns_split(
        self, sample_dataset_config, normalization
    ):
        td = TrainingData(sources=frozenset({"cmod_low_1", "cmod_low_2"}))
        train_ds, val_ds = get_train_val_datasets(
            td, normalization, study_type="power_balance_transfer", debug=True
        )
        assert train_ds.sizes["shot"] > 0
        assert val_ds.sizes["shot"] > 0

    def test_get_train_val_datasets_has_ds_source_idx(self, sample_dataset_config):
        td = TrainingData(sources=frozenset({"cmod_low_1", "cmod_high"}))
        train_ds, val_ds = get_train_val_datasets(
            td, "raw", study_type="power_balance_transfer", debug=True
        )
        assert "ds_source_idx" in train_ds
        assert "ds_source_idx" in val_ds

    def test_get_train_val_datasets_empty_sources_raises(self, sample_dataset_config):
        td = TrainingData.__new__(TrainingData)
        object.__setattr__(td, "sources", frozenset())
        object.__setattr__(td, "exnihilo", False)
        with pytest.raises(ValueError, match="sources is empty"):
            get_train_val_datasets(
                td, "raw", study_type="power_balance_transfer", debug=True
            )
