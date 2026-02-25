import pytest

from popsim_transport_predictor.config import config
from popsim_transport_predictor.trajectory_optimization.setup import (
    make_optimization_dataset,
    IP_RAMP_SHOTS,
)


def test_optimization_dataset():
    """Make sure the dataset is reproducible with the same seed, and different seeds are truly different
    make sure that exactly the original IP ramp shots are in there when debug is true"""

    ds_path = config.d3d_dataset_path

    ds_debug_1 = make_optimization_dataset(ds_path, debug=True)
    ds_debug_2 = make_optimization_dataset(ds_path, debug=True)

    assert len(ds_debug_1["shot"].values) == len(IP_RAMP_SHOTS.keys()), (
        "Debug dataset should only contain the IP ramp shots"
    )

    assert ds_debug_1.equals(ds_debug_2), (
        "Datasets with the same seed should be identical"
    )

    ds_normal_1 = make_optimization_dataset(
        ds_path, permutations_per_shot=4, debug=False
    )
    ds_normal_2 = make_optimization_dataset(
        ds_path, permutations_per_shot=4, debug=False
    )
    ds_normal_3 = make_optimization_dataset(
        ds_path, permutations_per_shot=4, prng_seed=43, debug=False
    )

    assert ds_normal_1.equals(ds_normal_2), (
        "Datasets with the same seed should be identical"
    )
    assert not ds_normal_1.equals(ds_normal_3), (
        "Datasets with different seeds should not be identical"
    )
