from popsim.ml import TrainConfig, Trainer

from popsim_transport_predictor.config import config
from popsim_transport_predictor.modules.profile_trajectory.data import get_ds
from popsim_transport_predictor.modules.profile_trajectory.profile_predictor.trb import (
    ProfilePredictorTRB,
)
from popsim_transport_predictor.modules.profile_trajectory.trb import (
    ProfileTrajectoryOptimizerTRB,
)
from popsim_transport_predictor.trajectory_optimization.plotting import (
    profile_comparison,
)
from popsim_transport_predictor.trajectory_optimization.setup import (
    make_optimization_dataset,
)


def eval_profile_predictor(transport_predictor_config: TrainConfig):
    """
    Evaluate how well the profile predictor works within the trajectory optimization shots
    """
    _, train_dl, _, test_dl = ProfileTrajectoryOptimizerTRB.get_dataloaders(
        transport_predictor_config.dataloader_config
    )
    trainer = Trainer(
        model=ProfilePredictorTRB.model_init(
            train_dl, transport_predictor_config.model_init_config
        ),
        loss_fn=ProfilePredictorTRB.get_loss_fn(transport_predictor_config.loss_config),
        optimizer=ProfilePredictorTRB.get_optimizer(
            transport_predictor_config.optimizer_config
        ),
        checkpoint_dir=transport_predictor_config.checkpoint_dir,
    )
    trainer.restore_best_checkpoint()

    # Run evaluation on the test set and extract the input and output datasets
    eval_data = trainer.run_evals(test_dl)

    return eval_data


def plot_profiles(
    ds_path: str,
    fig_dir: str,
):
    ds, _episode_coord = get_ds(
        ds_path,
        fresh_profiles=True,  # Only use timesteps where profile data is fresh
        debug=False,
    )
    ds_opt = make_optimization_dataset(
        ds, debug=True
    )  # Get original shots for plotting

    profile_comparison(
        profile_dir=fig_dir,
        ds_targ=ds_opt,  # Original dataset is the target since it has the true profiles
        ds_pred=None,  # No predictions for now, just plotting the original profiles
    )


# A bit hacky, just saving all these things here for now
if __name__ == "__main__":
    ds_path = config.d3d_dataset_path
    fig_dir = "scratch/trajectory_optimization_profiles"
    plot_profiles(ds_path, fig_dir)
