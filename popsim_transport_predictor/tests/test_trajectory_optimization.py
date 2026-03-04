import numpy as np
import pytest
import os
from popsim_transport_predictor import PACKAGE_ROOT
import chex

from popsim_transport_predictor.config import config
from popsim_transport_predictor.trajectory_optimization.setup import (
    make_optimization_dataset,
    IP_RAMP_SHOTS,
)
from popsim_transport_predictor.modules.profile_trajectory.data import get_ds
from popsim_transport_predictor.modules.profile_trajectory.module import (
    ProfileTrajectoryOptimizer,
)
from popsim.ml import TrainConfig, Trainer
from popsim.ml.launch import launch_train
from popsim_transport_predictor.modules.profile_trajectory.profile_predictor.trb import (
    ProfilePredictorTRB,
)
from popsim_transport_predictor.modules.profile_trajectory.trb import (
    ProfileTrajectoryOptimizerTRB,
)
from popsim_transport_predictor.trajectory_optimization.optimize import (
    setup_optimization_config,
    train_profile_predictor,
)
from popsim_transport_predictor.trajectory_optimization.plotting import (
    profile_comparison,
    trajectory_performance_comparison,
)
from popsim_transport_predictor.trajectory_optimization.setup import (
    make_optimization_dataset,
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


def test_optimization_training():
    """Ensure the right things are getting changed in the trajectory optimization
    1. Confirm we aren't accidentally modifying our world model during optimization
    2. Confirm the optimization actually modifies the trajectory parameters
    """
    save_dir = os.path.join(PACKAGE_ROOT, "tests", "test_trajectory_optimization")
    case_dir = os.path.join(save_dir, "predictor_unchanged")
    ds_path = config.d3d_dataset_path
    model_type = "direct_points"

    checkpoint_dir = os.path.join(case_dir, "checkpoints")
    shape_times = np.linspace(2.0, 5.0, 4).tolist()
    shape_times = [round(t, 1) for t in shape_times]  # Round to nearest 10th
    optimization_config = setup_optimization_config(
        ds_path,
        model_type,
        debug=True,
        checkpoint_dir=checkpoint_dir,
        shape_times=shape_times,
    )
    optimization_config.model_init_config["submodules"]["profile_predictor"][
        "checkpoint_dir"
    ] = os.path.join(save_dir, model_type, "checkpoints")
    profile_predictor_checkpoint_dir = os.path.join(save_dir, model_type, "checkpoints")

    predictor_trainer, _, _ = train_profile_predictor(
        ds_path,
        model_type,
        profile_predictor_checkpoint_dir,
        debug=True,
        clean=True,
    )
    optimization_config.model_init_config["submodules"]["profile_predictor"][
        "checkpoint_dir"
    ] = profile_predictor_checkpoint_dir

    optimization_trainer, _, aug_dl, _, _test_results = launch_train(
        optimization_config.model_dump(), use_wandb=False
    )

    _, train_dl, _, _test_dl = ProfileTrajectoryOptimizerTRB.get_dataloaders(
        optimization_config.dataloader_config
    )
    optimization_trainer_restored = Trainer(
        model=ProfileTrajectoryOptimizerTRB.model_init(
            train_dl, optimization_config.model_init_config
        ),
        loss_fn=ProfileTrajectoryOptimizerTRB.get_loss_fn(
            optimization_config.loss_config
        ),
        optimizer=ProfileTrajectoryOptimizerTRB.get_optimizer(
            optimization_config.optimizer_config
        ),
        checkpoint_dir=checkpoint_dir,
    )

    optimization_trainer_restored.restore_best_checkpoint()

    # The parameters of the profile predictor submodule should be identical between all three cases:
    # 1. The profile predictor trained on its own
    # 2. The profile predictor as part of the optimization trainer, after training
    # 3. The profile predictor as part of the optimization trainer, after restoring from checkpoint
    predictor_params = predictor_trainer.train_state.model.nn
    optimization_params = (
        optimization_trainer.train_state.model.module.profile_predictor.nn
    )
    restored_params = (
        optimization_trainer_restored.train_state.model.module.profile_predictor.nn
    )
    (
        chex.assert_trees_all_equal(predictor_params, optimization_params),
        "Profile predictor parameters should be unchanged during optimization training",
    )
    (
        chex.assert_trees_all_equal(predictor_params, restored_params),
        "Profile predictor parameters should be unchanged after restoring optimization checkpoint",
    )

    initial_model = ProfileTrajectoryOptimizerTRB.model_init(
        train_dl, optimization_config.model_init_config
    )

    initial_trajectory = {
        var: getattr(initial_model.module, var)
        for var in optimization_config.model_init_config["input_ranges"].keys()
    }
    optimized_trajectory = {
        var: getattr(optimization_trainer.train_state.model.module, var)
        for var in optimization_config.model_init_config["input_ranges"].keys()
    }
    restored_trajectory = {
        var: getattr(optimization_trainer_restored.train_state.model.module, var)
        for var in optimization_config.model_init_config["input_ranges"].keys()
    }

    # Assert optimized and restored trajectories are the same, and different from the initial trajectory
    for var in optimization_config.model_init_config["input_ranges"].keys():
        (
            chex.assert_trees_all_equal(
                optimized_trajectory[var], restored_trajectory[var]
            ),
            f"Optimized and restored trajectories should be the same for {var}",
        )
        with pytest.raises(AssertionError):
            (
                chex.assert_trees_all_equal(
                    initial_trajectory[var], optimized_trajectory[var]
                ),
                f"Optimized trajectory should be different from initial trajectory for {var}",
            )
