from loguru import logger

from transport_study.modules.profile_trajectory.data import get_ds
from transport_study.trajectory_optimization import (
    get_controllable_input_ranges,
    get_trajectory_input_ranges,
)

############################################################################################################
# Scope the dataset to determine the distribution of targets / allowable ranges for optimization variables #
############################################################################################################


def characterize_dataset(ds_path: str, debug: bool = False) -> None:
    """Load the dataset and print out some basic statistics to help scope the optimization problem.

    Args:
        ds_path (str): Path to the dataset.
        debug (bool, optional): Whether to enable debug mode, reducing dataset size to a select few shots.
    """

    ds, _ = get_ds(ds_path, debug=debug)

    input_ranges = get_trajectory_input_ranges(
        ds, ["gapin", "R0", "rxpt1", "zxpt1", "rxpt2", "zxpt2"]
    )

    for input_var, stats in input_ranges.items():
        logger.info(
            f"Input variable {input_var} has the following statistics during the trajectory portion of the shots:"
        )
        for stat_name, stat_value in stats.items():
            logger.info(f"    {stat_name}: {stat_value:.5f}")

    controllable_input_ranges = get_controllable_input_ranges(
        ds, ["iptipp_MA", "B0", "beta", "dstdenp"]
    )

    for input_var, error in controllable_input_ranges.items():
        logger.info(
            f"Controllable input variable {input_var} has a characteristic error of {error:.5f} during the trajectory portion of the shots"
        )
