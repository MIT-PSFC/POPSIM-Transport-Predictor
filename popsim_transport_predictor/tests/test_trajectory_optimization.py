import numpy as np
import pytest
import os

from popsim_transport_predictor.config import config
from popsim_transport_predictor.trajectory_optimization.setup import (
    make_optimization_dataset,
    IP_RAMP_SHOTS,
)
from popsim_transport_predictor.modules.profile_trajectory.data import get_ds
from popsim_transport_predictor.modules.profile_trajectory.module import (
    ProfileTrajectoryOptimizer,
)


def test_optimization_dataset():
    """Make sure the dataset is reproducible with the same seed, and different seeds are truly different
    make sure that exactly the original IP ramp shots are in there when debug is true"""

    ds_path = config.d3d_dataset_path
    ds, _ = get_ds(ds_path)

    ds_debug_1 = make_optimization_dataset(ds, debug=True)
    ds_debug_2 = make_optimization_dataset(ds, debug=True)

    assert len(ds_debug_1["shot"].values) == len(IP_RAMP_SHOTS.keys()), (
        "Debug dataset should only contain the IP ramp shots"
    )

    assert ds_debug_1.equals(ds_debug_2), (
        "Datasets with the same seed should be identical"
    )

    ds_normal_1 = make_optimization_dataset(ds, permutations_per_shot=4, debug=False)
    ds_normal_2 = make_optimization_dataset(ds, permutations_per_shot=4, debug=False)
    ds_normal_3 = make_optimization_dataset(
        ds, permutations_per_shot=4, prng_seed=43, debug=False
    )

    assert ds_normal_1.equals(ds_normal_2), (
        "Datasets with the same seed should be identical"
    )
    assert not ds_normal_1.equals(ds_normal_3), (
        "Datasets with different seeds should not be identical"
    )


def test_resolve_shapes():
    input_ranges = {
        "R0": (1.0, 2.0),
        "a_minor": (0.1, 0.2),
        "kappa": (1.0, 1.2),
        "delta_top": (-0.5, 0.5),
        "delta_bottom": (-0.6, 0.6),
    }

    shape_times = np.array([2.0, 3.0, 4.0])

    config = ProfileTrajectoryOptimizer.Config(
        shape_times=shape_times,
        input_ranges=input_ranges,
    )

    # Increase shape parameters linearly between shape times so we can see some difference
    trajectory = {
        "R0": np.array([1.0, 1.25, 1.7]),
        "a_minor": np.array([0.1, 0.15, 0.2]),
        "kappa": np.array([0.8, 1.1, 12]),
        "delta_top": np.array([-0.5, 0.0, 0.5]),
        "delta_bottom": np.array([-0.6, 0.0, 0.6]),
    }

    optimizer = ProfileTrajectoryOptimizer(
        psigrid=np.linspace(0, 1, 10),
        config=config,
        profile_predictor=None,
        trajectory=trajectory,
    )

    # Assert all shapes are within the ranges at all times
    shapes_1 = optimizer.resolve_shapes(1.0)
    shapes_2 = optimizer.resolve_shapes(2.0)
    shapes_3 = optimizer.resolve_shapes(3.0)
    shapes_4 = optimizer.resolve_shapes(4.0)
    shapes_5 = optimizer.resolve_shapes(5.0)

    for shapes, time in zip(
        [shapes_1, shapes_2, shapes_3, shapes_4, shapes_5],
        [1.0, 2.0, 3.0, 4.0, 5.0],
    ):
        for shape_name, (min_val, max_val) in input_ranges.items():
            clipped_min = min_val - 0.001
            clipped_max = max_val + 0.001
            assert clipped_min <= shapes[shape_name] <= clipped_max, (
                f"Shape {shape_name} at time {time} is out of range: {shapes[shape_name]} not in [{min_val}, {max_val}]"
            )
