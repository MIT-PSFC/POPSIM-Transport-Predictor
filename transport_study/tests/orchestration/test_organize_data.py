import numpy as np
import pytest
import xarray as xr

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.config import StudyConfig, load_config
from transport_study.orchestration.organize_data import (
    TrainingData,
    add_hazard,
    get_ds,
    get_train_test_datasets,
    get_train_val_datasets,
    parse_training_data,
    reindex_to_uniform_timebase,
)
from transport_study.tests.sample_data import SAMPLE_DIR, requires_sample_data

SAMPLE_PATHS = {
    "cmod-low1": SAMPLE_DIR / "cmod-low1.nc",
    "cmod-low2": SAMPLE_DIR / "cmod-low2.nc",
    "cmod-high": SAMPLE_DIR / "cmod-high.nc",
    "mast-high": SAMPLE_DIR / "mast-high.nc",
}
MAX_DS_SIZE = 20


@pytest.fixture
def sample_dataset_config() -> StudyConfig:
    """Load the global config with the sample datasets, cmod-high as target."""
    return load_config(
        StudyConfig(
            study_name="test-organize-data",
            dataset_paths=dict(SAMPLE_PATHS),
            target_device="cmod-high",
            max_ds_size=MAX_DS_SIZE,
        )
    )


class TestGetDs:
    @requires_sample_data
    @pytest.mark.parametrize("source_ds", ["cmod-low1", "mast-high"])
    @pytest.mark.parametrize("study_type", ["profile_transfer", "power_balance_transfer"])
    def test_get_ds_returns_dataset_and_dims(self, sample_dataset_config, source_ds, study_type):
        ds, episode_dim = get_ds(source_ds, study_type)
        assert ds is not None
        assert episode_dim == EPISODE_DIM
        assert TIME_COORD in ds.coords
        assert TIME_DIM in ds.dims

    @pytest.mark.parametrize("study_type", ["profile_transfer", "power_balance_transfer"])
    def test_get_ds_unknown_source_raises(self, sample_dataset_config, study_type):
        with pytest.raises(ValueError, match="Unknown source dataset"):
            get_ds("nonexistent_device", study_type)

    @requires_sample_data
    def test_get_ds_unknown_study_type_raises(self, sample_dataset_config):
        with pytest.raises(ValueError, match="Unknown study type"):
            get_ds("cmod-low1", "not_a_study")

    @requires_sample_data
    def test_get_ds_profile_transfer_has_shape_vars(self, sample_dataset_config):
        ds, _ = get_ds("cmod-low1", "profile_transfer")
        assert "Te_shape" in ds
        assert "ne_shape" in ds

    @requires_sample_data
    def test_get_ds_power_balance_has_aux_power(self, sample_dataset_config):
        ds, _ = get_ds("cmod-low1", "power_balance_transfer")
        assert "P_aux_MW" in ds

    @requires_sample_data
    def test_get_ds_max_ds_size_limits_shots(self, sample_dataset_config):
        ds, _ = get_ds("cmod-low1", "power_balance_transfer")
        assert ds.sizes[EPISODE_DIM] <= MAX_DS_SIZE


@requires_sample_data
class TestAddHazard:
    @pytest.mark.parametrize("study_type", ["profile_transfer", "power_balance_transfer"])
    def test_add_hazard_adds_variables(self, sample_dataset_config, study_type):
        ds, episode_coord = get_ds("cmod-low1", study_type)
        result = add_hazard(ds, episode_coord)
        assert "hazard" in result
        assert "Ip_MA_p95" in result
        assert "Wtot_MJ_p95" in result
        assert result["hazard"].dims == (episode_coord,)

    def test_add_hazard_nonnegative(self, sample_dataset_config):
        ds, episode_coord = get_ds("cmod-high", "power_balance_transfer")
        result = add_hazard(ds, episode_coord)
        valid = result["hazard"].values
        assert all(v >= 0 for v in valid if not np.isnan(v))


class TestTrainingData:
    def test_training_data_str_sorted(self):
        td = TrainingData(sources_unsorted=["mast-high", "cmod-high"])
        assert str(td) == "cmod-high_mast-high"

    def test_training_data_exnihilo_str(self):
        td = TrainingData(sources_unsorted=["cmod-low1"], exnihilo=True)
        assert str(td) == "exnihilo"

    def test_training_data_source_order_deterministic(self):
        td_1 = TrainingData(sources_unsorted=["cmod-low1", "cmod-high"])
        td_2 = TrainingData(sources_unsorted=["cmod-high", "cmod-low1"])
        assert td_1.sources == td_2.sources
        assert hash(td_1) == hash(td_2)

    def test_training_data_source_idxs_use_config(self, sample_dataset_config):
        td = TrainingData(sources_unsorted=["cmod-low1", "cmod-high"])
        assert td.source_idxs == [sample_dataset_config.ds_source_to_idx[s] for s in td.sources]

    def test_parse_training_data_splits_on_underscore(self, sample_dataset_config):
        td = parse_training_data("cmod-low1_cmod-low2", dict(SAMPLE_PATHS), "cmod-high")
        assert td.sources == ["cmod-low1", "cmod-low2"]
        assert not td.exnihilo

    def test_parse_training_data_exnihilo_excludes_target(self, sample_dataset_config):
        td = parse_training_data("exnihilo", dict(SAMPLE_PATHS), "cmod-high")
        assert td.exnihilo
        assert "cmod-high" not in td.sources
        assert set(td.sources) == set(SAMPLE_PATHS) - {"cmod-high"}


class TestGetTrainValDatasets:
    @requires_sample_data
    def test_get_train_val_datasets_returns_split(self, sample_dataset_config):
        td = TrainingData(sources_unsorted=["cmod-low1", "cmod-low2"])
        train_ds, val_ds = get_train_val_datasets(td, study_type="power_balance_transfer")
        assert train_ds.sizes[EPISODE_DIM] > 0
        assert val_ds.sizes[EPISODE_DIM] > 0
        # Train and val sets are disjoint per source
        for source_idx in np.unique(train_ds["ds_source_idx"].values):
            train_subset = train_ds.where(train_ds["ds_source_idx"] == source_idx, drop=True)
            val_subset = val_ds.where(val_ds["ds_source_idx"] == source_idx, drop=True)
            train_shots = set(train_subset[EPISODE_DIM].values)
            val_shots = set(val_subset[EPISODE_DIM].values)
            assert train_shots.isdisjoint(val_shots)
            # Validation shots have the higher hazard metric within each source
            assert val_subset["hazard"].values.min() >= train_subset["hazard"].values.max()

    def test_get_train_val_datasets_empty_sources_raises(self, sample_dataset_config):
        td = TrainingData(sources_unsorted=[])
        with pytest.raises(ValueError, match="sources is empty"):
            get_train_val_datasets(td, study_type="power_balance_transfer")


@requires_sample_data
class TestGetTrainTestDatasets:
    def test_get_train_test_datasets_returns_split(self, sample_dataset_config):
        td = TrainingData(sources_unsorted=["cmod-low1", "cmod-low2"])
        train_ds, test_ds = get_train_test_datasets(
            td,
            domain_adaptation="addition",
            num_target_shots=2,
            target_test_set_size=5,
            study_type="power_balance_transfer",
        )
        assert train_ds.sizes[EPISODE_DIM] > 0
        assert test_ds.sizes[EPISODE_DIM] == 5

        # Only target device shots are in the test set
        target_idx = sample_dataset_config.ds_source_to_idx[sample_dataset_config.target_device]
        assert all(s == target_idx for s in test_ds["ds_source_idx"].values)

        # Target shots in training never overlap the held-out test shots
        train_target = train_ds.where(train_ds["ds_source_idx"] == target_idx, drop=True)
        train_shots = set(train_target[EPISODE_DIM].values)
        test_shots = set(test_ds[EPISODE_DIM].values)
        assert train_shots.isdisjoint(test_shots)

        # The test set holds the highest-hazard target shots
        assert test_ds["hazard"].values.min() >= train_target["hazard"].values.max()

    def test_get_train_test_datasets_zero_test_size_keeps_test_empty(self, sample_dataset_config):
        td = TrainingData(sources_unsorted=["cmod-low1"])
        _, test_ds = get_train_test_datasets(
            td,
            domain_adaptation="addition",
            num_target_shots=2,
            target_test_set_size=0,
            study_type="power_balance_transfer",
        )
        assert test_ds.sizes[EPISODE_DIM] == 0


class TestReindexToUniformTimebase:
    @staticmethod
    def _make_ds(times, values):
        times = np.asarray(times, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        return xr.Dataset(
            data_vars={
                TIME_COORD: ((EPISODE_DIM, TIME_DIM), times),
                "Wtot_MJ": ((EPISODE_DIM, TIME_DIM), values),
                "R0": ((EPISODE_DIM,), np.arange(times.shape[0], dtype=np.float32)),
            },
            coords={EPISODE_DIM: np.arange(times.shape[0])},
        )

    def test_gap_becomes_nan_slice(self):
        nan = np.nan
        # Shot 0 skips t=3,4 ms, shot 1 starts later and is shorter
        times = [
            [0.000, 0.001, 0.002, 0.005, 0.006],
            [0.010, 0.011, 0.012, nan, nan],
        ]
        values = [
            [10.0, 11.0, 12.0, 15.0, 16.0],
            [20.0, 21.0, 22.0, nan, nan],
        ]
        ds = reindex_to_uniform_timebase(self._make_ds(times, values))

        # Columns are absolute canonical-grid slots, so the dim spans t=0..12 ms
        assert ds.sizes[TIME_DIM] == 13
        w = ds["Wtot_MJ"].values
        # Values land on their absolute slots, the mid-shot gap is NaN
        assert np.allclose(w[0, :7], [10.0, 11.0, 12.0, nan, nan, 15.0, 16.0], equal_nan=True)
        assert np.isnan(w[0, 7:]).all()
        assert np.isnan(w[1, :10]).all()
        assert np.allclose(w[1, 10:], [20.0, 21.0, 22.0], equal_nan=True)

        t = ds[TIME_COORD].values
        # Time is filled on the grid inside each shot window, including the gap
        assert np.allclose(t[0, :7], [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006], atol=1e-6)
        assert np.isnan(t[0, 7:]).all()
        # Leading and trailing padding time stays NaN outside the shot window
        assert np.isnan(t[1, :10]).all()
        assert np.allclose(t[1, 10:], [0.010, 0.011, 0.012], atol=1e-6)

    def test_time_values_come_from_canonical_timebase(self):
        from transport_study.datasets import make_uniform_1khz_timebase

        times = [[0.000, 0.001, 0.003]]
        values = [[1.0, 2.0, 3.0]]
        ds = reindex_to_uniform_timebase(self._make_ds(times, values))
        expected = make_uniform_1khz_timebase(0.003)
        assert np.array_equal(ds[TIME_COORD].values[0], expected)

    def test_uniform_input_is_unchanged(self):
        times = [[0.000, 0.001, 0.002, 0.003]]
        values = [[1.0, 2.0, 3.0, 4.0]]
        ds = reindex_to_uniform_timebase(self._make_ds(times, values))
        assert ds.sizes[TIME_DIM] == 4
        assert np.allclose(ds["Wtot_MJ"].values, values)
        assert np.allclose(ds[TIME_COORD].values, times, atol=1e-6)

    def test_non_time_vars_untouched(self):
        times = [[0.000, 0.001, 0.003]]
        values = [[1.0, 2.0, 3.0]]
        ds = reindex_to_uniform_timebase(self._make_ds(times, values))
        assert np.allclose(ds["R0"].values, [0.0])

    def test_oversampled_data_raises(self):
        # Two samples 0.1 ms apart map to the same 1 ms grid slot
        times = [[0.0000, 0.0001, 0.0010]]
        values = [[1.0, 2.0, 3.0]]
        with pytest.raises(ValueError, match="same uniform-grid slot"):
            reindex_to_uniform_timebase(self._make_ds(times, values))
