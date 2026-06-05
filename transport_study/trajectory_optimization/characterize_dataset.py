from pathlib import Path

import fire
import numpy as np
from loguru import logger

from transport_study.config import config
from transport_study.modules.profile_trajectory.data import get_ds
from transport_study.trajectory_optimization.setup_data import (
    FEEDBACK_CONTROL_SHOTS,
    IP_RAMP_SHOTS,
)

############################################################################################################
# Scope the dataset to determine the distribution of targets / allowable ranges for optimization variables #
############################################################################################################


def get_trajectory_input_ranges(
    ds_path: Path | str,
    inputs: list[str],
    shots_times: dict[int, dict[str, float]] = IP_RAMP_SHOTS,
) -> dict[str, dict[str, float]]:
    """Find the typical ranges of trajectory parameters during the portion of the shot we are interested in
    The goal is to understand the distribution to set the bounds of what the parameter is allowed to vary by during trajectory optimization

    Args:
        ds_path: str
            The path to the dataset
        inputs: list[str]
            The list of input parameters to find the ranges for

    Returns:
        dict[str, dict[str, float]]: A dictionary mapping each input parameter to a dictionary with keys "min", "max", "mean", "std", "median", "q1", and "q3" for the respective statistics of that parameter across the relevant portion of the shots
    """

    ds_trajectory, _ = get_ds(
        ds_path,
        selected_shots=shots_times,
        fresh_profiles=False,
        debug=False,
    )

    input_ranges = {}
    for input_var in inputs:
        # Get data for this input var where it's not nan
        input_data = ds_trajectory[input_var].values
        input_data = input_data[~np.isnan(input_data)]
        input_ranges[input_var] = {
            "min": float(input_data.min()),
            "max": float(input_data.max()),
            "mean": float(input_data.mean()),
            "std": float(input_data.std()),
            "median": float(np.median(input_data)),
            "q1": float(np.percentile(input_data, 25)),
            "q3": float(np.percentile(input_data, 75)),
        }

    return input_ranges


def get_controllable_input_errors(
    ds_path: Path | str,
    inputs: list[tuple[str, str]],
    shots_times: dict[int, dict[str, float]] = FEEDBACK_CONTROL_SHOTS,
) -> dict[str, float]:
    """Find the characteristic distributions of controllable input parameters during the portion of the shot we are interested in
    The goal is to understand the typical error between the programmed and measured values,
    so we can optimize the trajectory across a range of possible values that the parameters might take when we actually go to run the thing
    this makes the resulting trajectory robust to control errors

    This is also used for the pre-shot profile predictions, using Monte-Carlo sampling of the input parameters according to these error distributions

    Args:
        ds_path: str
            The path to the dataset
        inputs: list[tuple[str, str]]
            The list of input parameter pairs to find the errors for.
            Format is [(input_prog1, input_meas1), (input_prog2, input_meas2), ...]

    Returns:
        dict[str, float]: A dictionary mapping each input parameter to a characteristic error value (e.g. standard deviation of the error)
    """

    ds_trajectory, _ = get_ds(
        ds_path,
        selected_shots=shots_times,
        fresh_profiles=False,
        debug=False,
    )

    input_errors = {}
    for input_prog, input_meas in inputs:
        if input_prog not in ds_trajectory or input_meas not in ds_trajectory:
            raise ValueError(f"Input variables {input_prog} and/or {input_meas} not found in dataset")

        absolute_error = np.abs(ds_trajectory[input_prog].values - ds_trajectory[input_meas].values)
        absolute_error = absolute_error[~np.isnan(absolute_error)]
        error_stat = float(absolute_error.std())
        input_errors[input_prog] = error_stat

    return input_errors


def characterize_dataset(
    ds_path: Path | str | None = None,
) -> None:
    """Load the dataset and print out some basic statistics to help scope the optimization problem.

    Args:
        ds_path (str): Path to the dataset.
    """
    if ds_path is None:
        ds_path = config.dataset_paths.get(config.target_device)

    try:
        input_ranges = get_trajectory_input_ranges(
            ds_path,
            [
                "ne20_edge",
                "R0",
                "gapin",
                "rxbot",
                "zxbot",
                "rxtop",
                "zxtop",
                "a_minor",
                "kappa",
                "delta_top",
                "delta_bot",
            ],
            shots_times=IP_RAMP_SHOTS,
        )

        for input_var, stats in input_ranges.items():
            logger.info(f"Input variable {input_var} has the following statistics during the trajectory portion of the shots:")
            for stat_name, stat_value in stats.items():
                logger.info(f"    {stat_name}: {stat_value:.5f}")
    except Exception as e:
        logger.error(f"Error characterizing trajectory input ranges: {e}")

    controllable_input_ranges = get_controllable_input_errors(
        ds_path,
        [
            ("Ip_MA_prog", "Ip_MA"),
            ("B0_prog", "B0"),
            ("betan_prog", "betan"),
            ("ne20_edge_prog", "ne20_edge"),
            ("R0_prog", "R0"),
            ("gapin_prog", "gapin"),
            ("rxbot_prog", "rxbot"),
            ("zxbot_prog", "zxbot"),
            ("rxtop_prog", "rxtop"),
            ("zxtop_prog", "zxtop"),
        ],
        shots_times=FEEDBACK_CONTROL_SHOTS,
    )

    logger.opt(colors=True).info("<bold><cyan>Controllable input parameter errors:</cyan></bold>")
    for input_var, error in controllable_input_ranges.items():
        logger.info(f"{input_var}:\t{error:.5f}")


if __name__ == "__main__":
    fire.Fire(
        {
            "characterize_dataset": characterize_dataset,
        }
    )
