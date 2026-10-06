"""The target shot split: the held-out test set and the three orders training shots are added in."""

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from transport_study import EPISODE_DIM, PACKAGE_ROOT, TIME_DIM
from transport_study.config import env_dataset_paths, load_config
from transport_study.orchestration.target_shots import (
    SPAN_SIGNALS,
    TARGET_SHOT_ORDERS,
    TargetSplit,
    held_out_shot_mask,
    target_shot_split,
    training_shot_picks,
)
from transport_study.power_balance_transfer.data_visualization import DataVisualization
from transport_study.power_balance_transfer.power_balance_study import PowerBalanceStudy

SHOTS = np.array([101, 102, 103, 104, 105, 106])
HAZARD = np.array([0.5, 0.9, 0.1, 0.7, 0.3, 0.8])

# The device stores in PTPS_DATASET_PATHS, each the target of one figure test
DEVICE_PATHS = env_dataset_paths()
# Target shot counts drawn in the figures, kept to those at most half the device's shots
FIGURE_SHOT_COUNTS = (0, 1, 3, 10, 32, 100)


def named_split(target_test_shots: tuple[int, ...]) -> TargetSplit:
    return TargetSplit(num_target_shots=0, target_shot_order="ascending", target_test_set_size=None, target_test_shots=target_test_shots)


def test_named_test_shots_pull_in_every_higher_hazard_shot():
    """104 (0.7) is the lowest named shot, so 106 (0.8) joins the test set too."""
    mask_test = held_out_shot_mask(SHOTS, HAZARD, named_split((104, 102)))
    assert set(SHOTS[mask_test].tolist()) == {102, 104, 106}


def test_named_test_shot_missing_from_loaded_data_raises():
    with pytest.raises(ValueError, match=r"\[999\]"):
        held_out_shot_mask(SHOTS, HAZARD, named_split((104, 999)))


@pytest.fixture
def target_ds() -> xr.Dataset:
    """A power balance view of 12 target shots, 20 timeslices each, with random signals and hazard."""
    rng = np.random.default_rng(0)
    n_shots, n_times = 12, 20
    data_vars = {name: ((EPISODE_DIM, TIME_DIM), rng.uniform(0.5, 2.0, (n_shots, n_times))) for name in SPAN_SIGNALS}
    data_vars["hazard"] = (EPISODE_DIM, rng.permutation(n_shots) / n_shots)
    return xr.Dataset(data_vars, coords={EPISODE_DIM: np.arange(200, 200 + n_shots)})


@pytest.mark.parametrize("target_shot_order", TARGET_SHOT_ORDERS)
def test_no_order_ever_picks_a_test_shot(target_ds, target_shot_order):
    hazard_by_shot = dict(zip(target_ds[EPISODE_DIM].values.tolist(), target_ds["hazard"].values.tolist(), strict=True))
    for num_target_shots in range(9):
        target_split = TargetSplit(num_target_shots, target_shot_order, target_test_set_size=4, target_test_shots=())
        train_shots, test_shots = target_shot_split(target_ds, target_split)
        assert len(train_shots) == num_target_shots
        assert not set(train_shots.tolist()) & set(test_shots.tolist())
        # The test set is the 4 highest-hazard shots
        test_hazard_min = min(hazard_by_shot[shot] for shot in test_shots.tolist())
        assert sum(hazard >= test_hazard_min for hazard in hazard_by_shot.values()) == 4


def test_ascending_and_descending_take_the_two_ends_of_the_pool():
    pool_hazard = np.array([0.4, 0.1, 0.5, 0.3, 0.2])
    footprints = [np.zeros((1, 1))] * 5
    ascending_picks = training_shot_picks(pool_hazard, footprints, 2, "ascending")
    descending_picks = training_shot_picks(pool_hazard, footprints, 2, "descending")
    assert ascending_picks.tolist() == [1, 4]
    assert descending_picks.tolist() == [2, 0]


def test_spanning_picks_the_typical_shot_then_the_extremes():
    """Five shots whose footprints sit along a line at -2, -1, 0, 1 and 2.

    One shot is the middle one, two are the ends, three add the middle back, nested from two up.
    """
    centers = np.array([1.0, -2.0, 2.0, 0.0, -1.0])
    pool_hazard = np.array([0.4, 0.1, 0.5, 0.3, 0.2])
    jitter = np.linspace(-0.05, 0.05, 8)
    footprints = [np.stack([center + jitter, jitter], axis=-1) for center in centers]

    picks_by_count = {n: training_shot_picks(pool_hazard, footprints, n, "spanning") for n in (1, 2, 3)}
    assert centers[picks_by_count[1]].tolist() == [0.0]
    assert sorted(centers[picks_by_count[2]].tolist()) == [-2.0, 2.0]
    assert sorted(centers[picks_by_count[3]].tolist()) == [-2.0, 0.0, 2.0]
    assert set(picks_by_count[2].tolist()) <= set(picks_by_count[3].tolist())


def _figure_dir() -> Path:
    working_dir_base = os.environ.get("PTPS_TEST_WORKING_DIR_BASE", None)
    if working_dir_base is None:
        working_dir_base = PACKAGE_ROOT / "tests" / "test_outputs"
    return Path(working_dir_base) / "target_shot_orders"


@pytest.mark.slow
@pytest.mark.parametrize("target_device", sorted(DEVICE_PATHS))
def test_target_shot_order_figures(target_device):
    """Render the target_shot_orders figures with each device store as the target, for a look by eye.

    The test set is the highest-hazard quarter of the device's shots.
    Figures land in PTPS_TEST_WORKING_DIR_BASE/target_shot_orders/data_visualization/target_shot_orders.
    """
    if not DEVICE_PATHS[target_device].exists():
        pytest.skip(f"No {target_device} store at {DEVICE_PATHS[target_device]}")
    with xr.open_dataset(DEVICE_PATHS[target_device]) as ds_store:
        n_shots = ds_store.sizes[EPISODE_DIM]
    load_config(
        PowerBalanceStudy.Config(
            study_name="test-target-shot-order-figures",
            dataset_paths=dict(DEVICE_PATHS),
            target_device=target_device,
            working_dir_base=_figure_dir(),
            training_datasets=("exnihilo",),
            target_test_set_size=n_shots // 4,
            num_target_shots_options=tuple(n for n in FIGURE_SHOT_COUNTS if n <= n_shots // 2),
            target_shot_orders=TARGET_SHOT_ORDERS,
        )
    )

    figure_dir = _figure_dir()
    fig_paths = [
        figure_dir / "data_visualization" / "target_shot_orders" / f"{target_device}_order_{order}.png" for order in TARGET_SHOT_ORDERS
    ]
    # The visualization skips figures that exist, so stale ones are cleared first
    for fig_path in fig_paths:
        fig_path.unlink(missing_ok=True)
    DataVisualization.target_shot_orders(figure_dir)
    assert all(fig_path.exists() for fig_path in fig_paths)
