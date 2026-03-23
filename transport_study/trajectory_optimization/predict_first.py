import fire

from transport_study.config import config
from transport_study.profile_transfer.restore_predictor import (
    restore_profile_predictor_from_checkpoint,
)


def run_preshot_prediction(
    profile_predictor_checkpoint_dir: str,
    optimized_trajectory_checkpoint_dir: str | None = None,
    ds_path: str = config.d3d_hp_dataset_path,
):
    """Run selected profile predictor to get distribution of profiles over time

    0. If an optimized trajectory is provided, go get that and overwrite programmed trajectory with its values
    1. Make augmented dataset where input parameters are randomly perturbed within their typical error ranges
    2. Create time-independent dataloader for profile predictor
    3. Restore profile predictor from checkpoint
    4. Run profile predictor on dataloader
    5. Save predicted profiles as a dataset for later use
    6 + nice plots and gifs with error bars over time, etc.
    """

    profile_predictor = restore_profile_predictor_from_checkpoint(
        profile_predictor_checkpoint_dir
    )

    print(profile_predictor)


if __name__ == "__main__":
    fire.Fire(
        {
            "run_preshot_prediction": run_preshot_prediction,
        }
    )
