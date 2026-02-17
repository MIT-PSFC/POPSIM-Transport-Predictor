import xarray as xr

IP_RAMP_SHOTS = {
    199121: {"start": 3.0, "end": 5.0},
    199122: {"start": 2.5, "end": 5.0},
    199125: {"start": 2.0, "end": 5.5},
    199126: {"start": 2.0, "end": 4.3},
    # These 2019XX series are the ones I'll be mostly targeting, still using the above to see what's possible though
    201907: {"start": 1.5, "end": 5.2},
    201908: {"start": 1.5, "end": 5.3},
    201910: {"start": 1.5, "end": 5.4},
    201911: {"start": 1.5, "end": 5.4},
    201912: {"start": 1.5, "end": 5.4},
    201913: {"start": 1.5, "end": 5.4},  # I thought Arunav said there wasn't any ECRH?
    201914: {"start": 1.5, "end": 5.4},
    201927: {"start": 1.5, "end": 4.7},
    201934: {"start": 1.5, "end": 5.4},
}


def get_trajectory_input_ranges(
    ds: xr.Dataset,
    inputs: list[str],
    shots_times: dict[int, dict[str, float]] = IP_RAMP_SHOTS,
) -> dict[str, tuple[float, float]]:
    """Find the typical ranges of trajectory parameters during the portion of the shot we are interested in

    For this study, this finds the ranges of R0, a_minor, kappa, delta_top, and delta_bottom

    Parameters:
    -----------
    ds: xr.Dataset
        The full DIII-D HBP dataset
    inputs: list[str]
        The list of input parameters to find the ranges for

    Returns:
    --------
    dict[str, tuple[float, float]]: A dictionary mapping each input parameter to a tuple of (min, max) values that characterize the range of that parameter during the trajectory
    """


def get_controllable_input_ranges(
    ds: xr.Dataset,
    inputs: list[str],
    shots_times: dict[int, dict[str, float]] = IP_RAMP_SHOTS,
) -> dict[str, float]:
    """Find the characteristic distributions of controllable input parameters during the portion of the shot we are interested in

    For this study, this finds the typical error of Ip, B0, ne20_edge, and beta
    TODO(ZanderKeith): Yeah yeah I know to do this rigorously I'd want to look at the difference to the actual control waveforms, I'll do that if I have time

    Parameters:
    -----------
    ds: xr.Dataset
        The full DIII-D HBP dataset
    inputs: list[str]
        The list of input parameters to find the ranges for

    Returns:
    --------
    dict[str, float]: A dictionary mapping each input parameter to a characteristic error value (e.g. standard deviation of the error)
    """


def make_optimization_dataset():
    """Make a dataset with episode_dim being 'shot_alt', with shot x N episodes
    The idea is that we have historic data for ~10 shots that we're trying to model the scenario off of,
    but of course there's going to be differences when we actually go to run the thing
    So, optimize the trajectory across a distribution of shots

    Right now doing that in a simple fashion, where we find the typical distribution of input parameters,
    sample some offset for each one independently, and add that to a past trajectory to make a bunch of different trajectories to optimize across
    """
