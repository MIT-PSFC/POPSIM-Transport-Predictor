import numpy as np
import xarray as xr

# Shots that feature our Ip ramp that we can use to characterize ranges of inputs
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
    201913: {"start": 1.5, "end": 5.4},
    201914: {"start": 1.5, "end": 5.4},
    201927: {"start": 1.5, "end": 4.7},  # OUR BASE SHOT
    201934: {"start": 1.5, "end": 5.4},
}

# Shots that have our exact feedback control scheme (matching 201927)
# that we can use to characterize typical error between our target signal and the actual signal
FEEDBACK_CONTROL_SHOTS = {201927: {"start": 0.7, "end": 4.7}}

# Obtained by running characterize_dataset.py
PROG_INPUT_ERRORS = {
    "Ip_MA_prog": 0.00449,
    "B0_prog": 0.02,  # Actual value was 0.00831 but we're cheating a little so bump it up
    "betan_prog": 0.10511,
    "ne20_edge_prog": 0.03995,
    "R0_prog": 0.00258,
    "gapin_prog": 0.002,  # Actual value was 0.00146 but we're cheating a little so bump it up
    "rxbot_prog": 0.00254,
    "zxbot_prog": 0.00300,
    "rxtop_prog": 0.00416,
    "zxtop_prog": 0.00407,
}


def make_augmented_dataset(
    ds: xr.Dataset,
    prog_input_errors: dict[str, float] = PROG_INPUT_ERRORS,
    permutations: int = 100,
    prng_seed: int = 42,
    debug: bool | None = False,
) -> xr.Dataset:
    """Make a dataset with episode_dim being 'shot_alt', with shot x N episodes
    The idea is that we have a trajectory with programmed target waveforms,
    but of course there's going to be differences when we actually go to run the thing
    So, optimize the trajectory across a distribution of shots

    Right now doing that in a simple fashion, where we find the typical distribution of input parameters,
    sample some offset for each one independently, and add that to a past trajectory to make a bunch of different trajectories to optimize across

    Args:
        ds: xr.Dataset
            The dataset to use for making the optimization dataset. Expects there to be a single shot present.
        shots_times: dict[int, dict[str, float]]
            The time windows for each shot to use for the trajectory portion of the dataset
        prog_input_errors: dict[str, float]
            The characteristic errors of the controllable input parameters during the trajectory time window, as determined from the dataset
        permutations: int
            The number of different trajectories to make (including the unmodified one)
        prng_seed: int
            The seed to use for the pseudo-random number generator when sampling offsets for the input parameters
        debug: bool, optional
            If True, returns a dataset with only 2 permutations

    Returns:
        xr.Dataset: Dataset with episode_dim being 'shot_alt', with the modified trajectories to optimize across
    """
    rng = np.random.default_rng(seed=prng_seed)
    shot = ds["shot"].item()
    if debug:
        permutations = 2

    # Trim the dataset to where all the input signals are valid (not null and some non-zero)
    ds = ds.where(ds.notnull().all(dim="time_idx"), drop=True).isel(shot=0)
    if "betan_prog" in ds:
        ds = ds.where(ds["betan_prog"] > 0.01, drop=True)

    ds_shot_list = []
    for i in range(permutations):
        ds_permuted = ds.copy()
        # Leave the first one unmodified, so we have the actual trajectory in there as well to optimize across
        if i != 0:
            for prog_input, error in prog_input_errors.items():
                offset = rng.normal(loc=0.0, scale=error)
                ds_permuted[prog_input] = ds_permuted[prog_input] + offset
        ds_permuted = ds_permuted.expand_dims({"shot_alt": [f"{shot}_{i}"]}, axis=0)
        ds_shot_list.append(ds_permuted)

    ds_augmented = xr.concat(ds_shot_list, dim="shot_alt", coords="all")
    return ds_augmented
