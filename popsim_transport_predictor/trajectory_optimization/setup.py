import numpy as np
import xarray as xr

from popsim_transport_predictor.trajectory_optimization import IP_RAMP_SHOTS

PROG_INPUT_ERRORS = {
    "iptipp_MA": 0.01,
    "B0": 0.01,
    "dstdenp": 0.4,
    "beta": 0.22,
}


def get_trajectory_input_ranges(
    ds: xr.Dataset,
    inputs: list[str],
    shots_times: dict[int, dict[str, float]] = IP_RAMP_SHOTS,
) -> dict[str, dict[str, float]]:
    """Find the typical ranges of trajectory parameters during the portion of the shot we are interested in

    For this study, this finds the ranges of R0, a_minor, kappa, delta_top, and delta_bottom

    Args:
        ds: xr.Dataset
            The full DIII-D HBP dataset
        inputs: list[str]
            The list of input parameters to find the ranges for

    Returns:
        dict[str, dict[str, float]]: A dictionary mapping each input parameter to a dictionary with keys "min", "max", "mean", "std", "median", "q1", and "q3" for the respective statistics of that parameter across the relevant portion of the shots
    """

    shot_datasets = []
    for shot, times in shots_times.items():
        if shot in ds["shot"]:
            ds_shot = ds.sel(shot=shot)
            ds_ramp = ds_shot.where(
                (ds_shot["time"] >= times["start"]) & (ds_shot["time"] <= times["end"]),
                drop=True,
            )
            shot_datasets.append(ds_ramp)

    ds_trajectory = xr.concat(
        shot_datasets, dim="time_idx", coords="minimal", compat="override"
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


def get_controllable_input_ranges(
    ds: xr.Dataset,
    inputs: list[str],
    shots_times: dict[int, dict[str, float]] = IP_RAMP_SHOTS,
) -> dict[str, float]:
    """Find the characteristic distributions of controllable input parameters during the portion of the shot we are interested in

    For this study, this finds the typical error of iptipp_MA, B0, dstdenp, and beta
    TODO(ZanderKeith): Yeah yeah I know to do this rigorously I'd want to look at the difference to the actual control waveforms, I'll do that if I have time

    Args:
        ds: xr.Dataset
            The full DIII-D HBP dataset
        inputs: list[str]
            The list of input parameters to find the ranges for

    Returns:
        dict[str, float]: A dictionary mapping each input parameter to a characteristic error value (e.g. standard deviation of the error)
    """

    all_chunk_stds = []

    for shot, times in shots_times.items():
        if shot in ds["shot"]:
            ds_shot = ds.sel(shot=shot)[inputs]
            # Slice to the specific window of interest
            ds_ramp = ds_shot.where(
                (ds_shot["time"] >= times["start"]) & (ds_shot["time"] <= times["end"]),
                drop=True,
            )

            # Coarsen this shot individually
            # This avoids "bleeding" data from Shot A into Shot B
            shot_chunks = ds_ramp.coarsen(time_idx=100, boundary="trim").std()
            shot_chunks = shot_chunks.rename({"time_idx": "chunk_idx"})
            all_chunk_stds.append(shot_chunks)

    # Combine all the standard deviation "snippets" from all shots
    # We can just use a simple list merge or xr.concat if we want to keep it as an xarray object
    ds_all_stds = xr.concat(all_chunk_stds, dim="chunk_idx", coords="different")

    input_ranges = {}
    for input_var in inputs:
        # Average the standard deviations across ALL chunks from ALL shots
        input_ranges[input_var] = float(ds_all_stds[input_var].mean())

    return input_ranges


def make_optimization_dataset(
    ds: xr.Dataset,
    shots_times: dict[int, dict[str, float]] = IP_RAMP_SHOTS,
    prog_input_errors: dict[str, float] = PROG_INPUT_ERRORS,
    permutations_per_shot: int = 20,
    prng_seed: int = 42,
    debug: bool | None = False,
) -> xr.Dataset:
    """Make a dataset with episode_dim being 'shot_alt', with shot x N episodes
    The idea is that we have historic data for ~10 shots that we're trying to model the scenario off of,
    but of course there's going to be differences when we actually go to run the thing
    So, optimize the trajectory across a distribution of shots

    Right now doing that in a simple fashion, where we find the typical distribution of input parameters,
    sample some offset for each one independently, and add that to a past trajectory to make a bunch of different trajectories to optimize across

    Args:
        ds: xr.Dataset
            The dataset to use for making the optimization dataset
        shots_times: dict[int, dict[str, float]]
            The time windows for each shot to use for the trajectory portion of the dataset
        prog_input_errors: dict[str, float]
            The characteristic errors of the controllable input parameters during the trajectory time window, as determined from the dataset
        permutations_per_shot: int
            The number of different trajectories to make for each shot (including the unmodified one)
        prng_seed: int
            The seed to use for the pseudo-random number generator when sampling offsets for the input parameters
        debug: bool, optional
            If True, returns a dataset with unmodified source shots.

    Returns:
        xr.Dataset: Dataset with episode_dim being 'shot_alt', with the modified trajectories to optimize across
    """
    rng = np.random.default_rng(seed=prng_seed)

    ds_shot_list = []
    for shot, times in shots_times.items():
        if shot not in ds["shot"]:
            raise ValueError(f"Shot {shot} not found in dataset")
        ds_shot = ds.sel(shot=shot)
        ds_ramp = ds_shot.where(
            (ds_shot["time"] >= times["start"]) & (ds_shot["time"] <= times["end"]),
            drop=True,
        )
        for i in range(permutations_per_shot):
            ds_ramp_permuted = ds_ramp.copy()
            # Leave the first one unmodified, so we have the actual trajectory in there as well to optimize across
            if i != 0:
                for prog_input, error in prog_input_errors.items():
                    offset = rng.normal(loc=0.0, scale=error)
                    ds_ramp_permuted[prog_input] = ds_ramp_permuted[prog_input] + offset
            ds_ramp_permuted = ds_ramp_permuted.expand_dims(
                {"shot_alt": [f"{shot}_{i}"]}, axis=0
            )
            ds_shot_list.append(ds_ramp_permuted)

            if debug:
                # If we're in debug mode, only make one permutation per shot (the unmodified one)
                break

    # Pad all shots to the same length and combine into one big dataset
    max_time_len = max(len(ds_shot["time_idx"]) for ds_shot in ds_shot_list)
    ds_pad_list = []
    for ds_shot in ds_shot_list:
        shot_size = len(ds_shot["time_idx"])
        if shot_size < max_time_len:
            # Pad this shot with nans to reach the max length
            padding = max_time_len - shot_size
            ds_shot_padded = ds_shot.pad(time_idx=(0, padding), constant_values=np.nan)
            ds_pad_list.append(ds_shot_padded)
        else:
            ds_pad_list.append(ds_shot)

    ds_optimization = xr.concat(ds_pad_list, dim="shot_alt", coords="all")
    return ds_optimization
