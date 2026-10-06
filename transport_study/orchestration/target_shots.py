"""Which target-device shots a case is tested on and which it trains on.

The test set is never trained on.
It is either the target_test_set_size highest-hazard target shots,
or every shot at or above the hazard rank of the lowest-ranked named target_test_shots shot.
Every other target shot is the training pool, and a case adds num_target_shots of it in one of three orders:
- ascending, the base extrapolation: the lowest-hazard shots first
- descending: the highest-hazard pool shots first, the ones closest to the test regime
- spanning: the shots whose footprints span the power balance input and output space

A shot's footprint is the distribution of its timeslices over the power balance inputs and the stored energy,
so a shot is described by everything it passes through (ramps, flattop, heated phases), not by one summary point.
Two footprints differ by their maximum mean discrepancy (MMD) under a Gaussian kernel.
One shot is the medoid, the most typical footprint.
From two shots up the picks start from the most different pair,
then add the shot farthest from every pick so far (farthest-first, the greedy cover of the pool).

Every study decides the split on the power balance view of the target device,
so a transport case and its power balance and profile prereqs train and test on the same shots.
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr

from transport_study import EPISODE_DIM, TIME_DIM
from transport_study.config import config
from transport_study.modules.normalization import NORM_INPUT_VARS

TARGET_SHOT_ORDERS = ("ascending", "descending", "spanning")
# The base extrapolation, suppressed from case names
BASE_TARGET_SHOT_ORDER = "ascending"

# The power balance inputs and its output
SPAN_SIGNALS = (*NORM_INPUT_VARS, "energy_mhd_MJ")
# Evenly spaced complete timeslices per shot, so every shot weighs the same whatever its length
SPAN_TIMESLICES_PER_SHOT = 32
# Pool points the kernel width is measured on, a fixed stride through the pool keeps it deterministic
KERNEL_WIDTH_SAMPLE_SIZE = 2000


@dataclass(frozen=True)
class TargetSplit:
    """Everything that decides which target shots a case is tested and trained on."""

    num_target_shots: int
    target_shot_order: str
    # Exactly one of the two defines the test set
    target_test_set_size: int | None
    target_test_shots: tuple[int, ...]

    @classmethod
    def from_dataloader_config(cls, dataloader_config: dict) -> "TargetSplit":
        return cls(
            num_target_shots=dataloader_config["num_target_shots"],
            target_shot_order=dataloader_config["target_shot_order"],
            target_test_set_size=dataloader_config["target_test_set_size"],
            target_test_shots=tuple(dataloader_config["target_test_shots"]),
        )

    def dataloader_config(self) -> dict:
        """The dataloader-config entries from_dataloader_config reads back, as plain values YAML can load."""
        return {
            "num_target_shots": self.num_target_shots,
            "target_shot_order": self.target_shot_order,
            "target_test_set_size": self.target_test_set_size,
            "target_test_shots": list(self.target_test_shots),
        }


def configured_target_split(num_target_shots: int, target_shot_order: str = BASE_TARGET_SHOT_ORDER) -> TargetSplit:
    """A TargetSplit under the loaded config's test set."""
    return TargetSplit(
        num_target_shots=num_target_shots,
        target_shot_order=target_shot_order,
        target_test_set_size=config.target_test_set_size,
        target_test_shots=tuple(config.target_test_shots),
    )


def held_out_shot_mask(shots: np.ndarray, hazard: np.ndarray, target_split: TargetSplit) -> np.ndarray:
    """Which of the target shots are the test set.

    Ties in hazard keep the dataset order (most recent shot first), through a stable sort.
    """
    hazard_order = np.argsort(hazard, kind="stable")
    hazard_rank = np.empty(hazard.size, dtype=int)
    hazard_rank[hazard_order] = np.arange(hazard.size)
    if target_split.target_test_shots:
        missing = sorted(set(target_split.target_test_shots) - set(shots.tolist()))
        if missing:
            raise ValueError(
                f"Named test shots {missing} are not in the loaded target data "
                "(not in the store, dropped for lack of a hazard, or cut by max_ds_size)"
            )
        mask_named = np.isin(shots, target_split.target_test_shots)
        cutoff_rank = hazard_rank[mask_named].min()
        return hazard_rank >= cutoff_rank
    if target_split.target_test_set_size is None:
        raise ValueError("The target split names no test set, set target_test_set_size or target_test_shots")
    if target_split.target_test_set_size > shots.size:
        raise ValueError(f"target_test_set_size={target_split.target_test_set_size} exceeds the {shots.size} loaded target shots")
    return hazard_rank >= shots.size - target_split.target_test_set_size


def shot_footprints(ds: xr.Dataset) -> list[np.ndarray]:
    """Each shot's footprint, up to SPAN_TIMESLICES_PER_SHOT evenly spaced timeslices where every SPAN_SIGNALS value is finite.

    Each footprint is (timeslices, SPAN_SIGNALS), unscaled, and empty for a shot without a complete timeslice.
    """
    signal_values = np.stack([ds[name].transpose(EPISODE_DIM, TIME_DIM).values for name in SPAN_SIGNALS], axis=-1)
    mask_complete = np.isfinite(signal_values).all(axis=-1)
    footprints = []
    for shot_values, shot_mask_complete in zip(signal_values, mask_complete, strict=True):
        complete_values = shot_values[shot_mask_complete]
        n_kept = min(SPAN_TIMESLICES_PER_SHOT, len(complete_values))
        kept_idx = np.linspace(0, len(complete_values) - 1, n_kept).round().astype(int)
        footprints.append(complete_values[kept_idx])
    return footprints


def footprint_mmd(footprints: list[np.ndarray]) -> np.ndarray:
    """The (shots, shots) matrix of MMD between unscaled footprints.

    Each signal is z-scored over every footprint point, and a signal with no spread is dropped.
    The Gaussian kernel width is the median distance between pool points (the median heuristic).
    MMD^2(A, B) = mean k(a, a') + mean k(b, b') - 2 mean k(a, b), exact, built one shot's block at a time.
    """
    n_points_per_shot = np.array([len(footprint) for footprint in footprints])
    if (n_points_per_shot == 0).any():
        empty_idx = np.flatnonzero(n_points_per_shot == 0).tolist()
        raise ValueError(f"Pool shots at indices {empty_idx} have no timeslice with every one of {SPAN_SIGNALS} finite")

    points_raw = np.concatenate(footprints)
    signal_std = points_raw.std(axis=0)
    mask_spread = signal_std > 0
    signal_mean = points_raw.mean(axis=0)
    points = (points_raw[:, mask_spread] - signal_mean[mask_spread]) / signal_std[mask_spread]

    points_sq_norm = (points**2).sum(axis=1)
    stride = max(1, len(points) // KERNEL_WIDTH_SAMPLE_SIZE)
    width_sample = points[::stride]
    width_sample_sq_dist = _pairwise_sq_dist(width_sample, points_sq_norm[::stride], width_sample, points_sq_norm[::stride])
    upper_idx = np.triu_indices(len(width_sample), k=1)
    width_sample_dist = np.sqrt(width_sample_sq_dist[upper_idx])
    kernel_width = float(np.median(width_sample_dist))

    shot_ends = np.cumsum(n_points_per_shot)
    shot_starts = shot_ends - n_points_per_shot
    n_shots = len(footprints)
    kernel_means = np.empty((n_shots, n_shots))
    for shot_idx in range(n_shots):
        block_slice = slice(shot_starts[shot_idx], shot_ends[shot_idx])
        sq_dist = _pairwise_sq_dist(points[block_slice], points_sq_norm[block_slice], points, points_sq_norm)
        kernel = np.exp(-sq_dist / (2 * kernel_width**2))
        kernel_sum_per_point = kernel.sum(axis=0)
        kernel_sum_per_shot = np.add.reduceat(kernel_sum_per_point, shot_starts)
        kernel_means[shot_idx] = kernel_sum_per_shot / (n_points_per_shot[shot_idx] * n_points_per_shot)

    self_kernel_means = np.diag(kernel_means)
    mmd_sq = self_kernel_means[:, None] + self_kernel_means[None, :] - 2 * kernel_means
    # Round-off can leave a tiny negative MMD^2 between near-identical footprints
    mmd_sq_clipped = np.clip(mmd_sq, 0, None)
    return np.sqrt(mmd_sq_clipped)


def _pairwise_sq_dist(a: np.ndarray, a_sq_norm: np.ndarray, b: np.ndarray, b_sq_norm: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) squared Euclidean distances from the row norms, clipped at the round-off floor of 0."""
    sq_dist = a_sq_norm[:, None] + b_sq_norm[None, :] - 2 * a @ b.T
    return np.clip(sq_dist, 0, None)


def spanning_picks(mmd: np.ndarray, num_target_shots: int) -> np.ndarray:
    """Pool indices of the num_target_shots most spanning shots under a pool MMD matrix, in pick order.

    One shot is the medoid.
    From two up: the most different pair, then farthest-first, so the picks are nested from two shots up.
    Ties go to the lower index.
    """
    if num_target_shots == 0:
        return np.array([], dtype=int)
    if num_target_shots == 1:
        summed_mmd = mmd.sum(axis=1)
        return np.array([np.argmin(summed_mmd)])
    most_different_flat_idx = np.argmax(mmd)
    first_pick, second_pick = np.unravel_index(most_different_flat_idx, mmd.shape)
    picks = [int(first_pick), int(second_pick)]
    nearest_pick_mmd = np.minimum(mmd[first_pick], mmd[second_pick])
    nearest_pick_mmd[picks] = -np.inf
    while len(picks) < num_target_shots:
        next_pick = int(np.argmax(nearest_pick_mmd))
        picks.append(next_pick)
        nearest_pick_mmd = np.minimum(nearest_pick_mmd, mmd[next_pick])
        nearest_pick_mmd[picks] = -np.inf
    return np.array(picks)


def training_shot_picks(
    pool_hazard: np.ndarray,
    pool_footprints: list[np.ndarray],
    num_target_shots: int,
    target_shot_order: str,
) -> np.ndarray:
    """Pool indices of the num_target_shots training shots, in pick order."""
    if num_target_shots > pool_hazard.size:
        raise ValueError(
            f"num_target_shots={num_target_shots} requested but only {pool_hazard.size} target shots remain outside the test set. "
            "Is the dataset smaller than expected (max_ds_size truncation)?"
        )
    ascending_idx = np.argsort(pool_hazard, kind="stable")
    if target_shot_order == "ascending":
        return ascending_idx[:num_target_shots]
    if target_shot_order == "descending":
        descending_idx = ascending_idx[::-1]
        return descending_idx[:num_target_shots]
    if target_shot_order == "spanning":
        # In ascending hazard order, so ties go to the lower hazard
        ascending_footprints = [pool_footprints[idx] for idx in ascending_idx]
        mmd = footprint_mmd(ascending_footprints)
        ascending_picks = spanning_picks(mmd, num_target_shots)
        return ascending_idx[ascending_picks]
    raise ValueError(f"Unknown target shot order: {target_shot_order}. Must be one of {TARGET_SHOT_ORDERS}")


def target_shot_split(ds_power_balance: xr.Dataset, target_split: TargetSplit) -> tuple[np.ndarray, np.ndarray]:
    """(training shots in pick order, test shots) as shot numbers, from the power balance view of the target device."""
    shots = ds_power_balance[EPISODE_DIM].values
    hazard = ds_power_balance["hazard"].values
    mask_test = held_out_shot_mask(shots, hazard, target_split)
    pool_idx = np.flatnonzero(~mask_test)
    ds_pool = ds_power_balance.isel({EPISODE_DIM: pool_idx})
    pool_footprints = shot_footprints(ds_pool)
    pool_hazard = hazard[pool_idx]
    picks = training_shot_picks(pool_hazard, pool_footprints, target_split.num_target_shots, target_split.target_shot_order)
    train_shots = shots[pool_idx[picks]]
    # Test shots lowest hazard first
    hazard_order = np.argsort(hazard, kind="stable")
    test_order = hazard_order[mask_test[hazard_order]]
    test_shots = shots[test_order]
    return train_shots, test_shots
