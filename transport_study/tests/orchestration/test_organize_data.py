import numpy as np
import pytest
import xarray as xr

from transport_study import EPISODE_DIM, RADIAL_DIM, TIME_COORD, TIME_DIM
from transport_study.config import RHO_GRID, StudyConfig, load_config
from transport_study.orchestration.organize_data import (
    HAZARD_VARS,
    REQUIRED_SIGNALS,
    TrainingData,
    add_hazard,
    check_uniform_timebase,
    get_ds,
    get_train_test_datasets,
    get_train_val_datasets,
    keep_fresh_timeslices,
    parse_training_data,
)
from transport_study.tests.datasets.synthetic_store import (
    DT,
    SHOT_LENGTHS,
    T_START,
    write_synthetic_store,
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
        assert "t_e_shape" in ds
        assert "n_e_shape" in ds

    @requires_sample_data
    def test_get_ds_power_balance_has_aux_power(self, sample_dataset_config):
        ds, _ = get_ds("cmod-low1", "power_balance_transfer")
        assert "power_additional_MW" in ds

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
        assert "ip_MA_p95" in result
        assert "energy_mhd_MJ_p95" in result
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

    def test_parse_training_data_refuses_the_target_device(self):
        """The target's held-out test shots would land in the training set under weighted / addition."""
        with pytest.raises(ValueError, match="target device"):
            parse_training_data("cmod-low1_cmod-high", dict(SAMPLE_PATHS), "cmod-high")


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


def _time_series_ds(times, values) -> xr.Dataset:
    times = np.asarray(times, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    return xr.Dataset(
        data_vars={
            TIME_COORD: ((EPISODE_DIM, TIME_DIM), times),
            "energy_mhd_MJ": ((EPISODE_DIM, TIME_DIM), values),
            "ip_MA": ((EPISODE_DIM, TIME_DIM), 2.0 * values),
            "fresh_profile": ((EPISODE_DIM, TIME_DIM), np.ones_like(values)),
        },
        coords={EPISODE_DIM: 100 + np.arange(times.shape[0])},
    )


class TestCheckUniformTimebase:
    def test_contiguous_shots_with_trailing_padding_pass(self):
        nan = np.nan
        times = [[0.100, 0.101, 0.102, 0.103], [0.200, 0.201, nan, nan]]
        check_uniform_timebase(_time_series_ds(times, np.ones((2, 4))))

    def test_interior_gap_names_the_shot(self):
        nan = np.nan
        times = [[0.100, 0.101, 0.102, 0.103], [0.200, nan, 0.202, 0.203]]
        with pytest.raises(ValueError, match="101"):
            check_uniform_timebase(_time_series_ds(times, np.ones((2, 4))))

    def test_off_grid_step_names_the_shot(self):
        times = [[0.100, 0.101, 0.102, 0.103], [0.200, 0.201, 0.203, 0.204]]
        with pytest.raises(ValueError, match="101"):
            check_uniform_timebase(_time_series_ds(times, np.ones((2, 4))))


class TestAddHazardSynthetic:
    def test_p95_and_the_values_at_its_timeslice(self):
        """Hazard is the p95 of the normalized radius, the p95 columns hold ip and W at the closest timeslice."""
        nan = np.nan
        times = [[0.1, 0.101, 0.102, 0.103, 0.104], [0.1, 0.101, 0.102, nan, nan]]
        energy = [[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 1.5, 2.0, nan, nan]]
        ds = add_hazard(_time_series_ds(times, energy), EPISODE_DIM)

        # ip_MA = 2 W everywhere, so the normalized radius is W / W_max sqrt(1 + 1) with W_max = 5
        hazard_expected = np.array([np.percentile([1, 2, 3, 4, 5], 95), np.percentile([1, 1.5, 2], 95)]) / 5.0 * np.sqrt(2.0)
        np.testing.assert_allclose(ds["hazard"].values, hazard_expected, rtol=1e-6)
        assert ds["hazard"].dims == (EPISODE_DIM,)
        # The p95 sits closest to the last valid timeslice of each shot
        np.testing.assert_allclose(ds["energy_mhd_MJ_p95"].values, [5.0, 2.0])
        np.testing.assert_allclose(ds["ip_MA_p95"].values, [10.0, 4.0])

    def test_shot_without_values_has_nan_hazard(self):
        nan = np.nan
        times = [[0.1, 0.101], [0.1, 0.101]]
        energy = [[1.0, 2.0], [nan, nan]]
        ds = add_hazard(_time_series_ds(times, energy), EPISODE_DIM)

        assert np.isnan(ds["hazard"].values[1])
        assert np.isnan(ds["ip_MA_p95"].values[1])
        assert np.isfinite(ds["hazard"].values[0])


def test_keep_fresh_timeslices_masks_only_the_time_dependent_variables():
    times = [[0.1, 0.101, 0.102], [0.1, 0.101, 0.102]]
    energy = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    ds = _time_series_ds(times, energy)
    ds["fresh_profile"][0, 1] = 0.0
    ds["fresh_profile"][1, :] = 0.0
    ds["hazard"] = ((EPISODE_DIM,), np.array([0.5, 0.7], dtype=np.float32))

    ds_fresh = keep_fresh_timeslices(ds)

    assert ds_fresh[EPISODE_DIM].values.tolist() == [100], "a shot without a fresh slice is dropped"
    assert ds_fresh["hazard"].dims == (EPISODE_DIM,)
    np.testing.assert_allclose(ds_fresh["hazard"].values, [0.5])
    # The time_idx column no shot is fresh at is dropped with the shot
    np.testing.assert_allclose(ds_fresh["energy_mhd_MJ"].values, [[1.0, 3.0]])


@pytest.fixture
def synthetic_store_config(tmp_path) -> StudyConfig:
    """A tiny store in the TVD schema as the one device of the global config."""
    store_path = tmp_path / "synthetic.zarr"
    write_synthetic_store(store_path, internal=False)
    return load_config(
        StudyConfig(
            study_name="test-organize-data-synthetic",
            dataset_paths={"synthetic": store_path},
            target_device="synthetic",
        )
    )


class TestGetDsSyntheticStore:
    @pytest.mark.parametrize("study_type", list(REQUIRED_SIGNALS))
    def test_study_signals_hazard_and_time_coordinate(self, synthetic_store_config, study_type):
        ds, episode_dim = get_ds("synthetic", study_type)

        assert episode_dim == EPISODE_DIM
        assert TIME_COORD in ds.coords
        for name in REQUIRED_SIGNALS[study_type]:
            assert name in ds.variables, name
        assert "r0" not in ds and "power_ic_MW" not in ds, "only the study signals stay"
        for name in HAZARD_VARS:
            assert ds[name].dims == (EPISODE_DIM,), name
        assert ds["hazard"].notnull().all()
        # Most recent shots first
        assert ds[EPISODE_DIM].values.tolist() == sorted(SHOT_LENGTHS, reverse=True)

    @pytest.mark.parametrize("study_type", list(REQUIRED_SIGNALS))
    def test_hazard_is_the_same_for_every_study(self, synthetic_store_config, study_type):
        """The hazard comes from the full 0D series before any study prep, so every study holds out the same shots."""
        ds, _ = get_ds("synthetic", study_type)

        # The synthetic store grows ip and W with the shot index, so the hazard orders the shots as listed
        hazard_by_shot = ds["hazard"].sel({EPISODE_DIM: list(SHOT_LENGTHS)}).values
        assert np.all(np.diff(hazard_by_shot) > 0)

    def test_magnitudes_and_working_units(self, synthetic_store_config):
        ds, _ = get_ds("synthetic", "power_balance_transfer")

        assert (ds["ip_MA"].fillna(1.0) > 0).all(), "the signed shot is a magnitude now"
        np.testing.assert_allclose(ds["energy_mhd_MJ"].isel({EPISODE_DIM: 0}).dropna(TIME_DIM).values[0], 0.06 * 1.4, rtol=1e-5)
        time_shot = ds[TIME_COORD].isel({EPISODE_DIM: 0}).dropna(TIME_DIM).values
        np.testing.assert_allclose(time_shot[:3], T_START + DT * np.arange(3), atol=1e-6)

    @pytest.mark.parametrize("study_type", ["profile_transfer", "transport_transfer"])
    def test_profiles_on_rho_grid_with_shapes(self, synthetic_store_config, study_type):
        ds, _ = get_ds("synthetic", study_type)

        np.testing.assert_allclose(ds[RADIAL_DIM].values, RHO_GRID)
        assert {"t_e_shape", "n_e_shape"} <= set(ds.data_vars)
        assert all(not ds[name].attrs for name in ds.variables), "attrs are static jit metadata"
