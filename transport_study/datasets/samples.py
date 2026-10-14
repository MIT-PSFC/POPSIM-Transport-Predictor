"""Sample device datasets for the tests.

Each sample is SAMPLE_N_SHOTS shots of a device store in its on-disk schema (IMAS names, SI units),
so the tests read it through organize_data.get_ds exactly like a store.
A store is split by the hazard metric into three disjoint samples,
<device>-low1.nc and <device>-low2.nc (the two lowest-hazard blocks) and <device>-high.nc (the highest),
and the tests treat each file as a device of its own, two sources and a target out of one device.
The files live outside the repo, in the directory PTPS_TEST_SAMPLE_DIR names (tests/sample_data.py).
"""

from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.orchestration.organize_data import add_hazard
from transport_study.signals import convert_to_working_units

SAMPLE_N_SHOTS = 80
# The hazard blocks a store is split into, each SAMPLE_N_SHOTS shots
SAMPLE_BLOCKS = ("low1", "low2", "high")
# netCDF compression of every numeric sample variable
SAMPLE_ENCODING = {"zlib": True, "shuffle": True, "complevel": 4}


def _sample_path(sample_dir: Path | str, device: str, block: str) -> Path:
    """The sample file of one hazard block of a device."""
    return Path(sample_dir) / f"{device}-{block}.nc"


def _shots_by_hazard(ds_store: xr.Dataset) -> np.ndarray:
    """Shots of a device store in ascending order of the hazard metric the study ranks shots by, those without one left out."""
    ds_scalars = ds_store[["energy_mhd", "ip"]].load()
    ds_working = convert_to_working_units(ds_scalars)
    ds_hazard = add_hazard(ds_working)
    # A shot without a valid energy or current timeslice has no hazard and belongs to no sample
    hazard = ds_hazard["hazard"].dropna(EPISODE_DIM)
    hazard_sorted = hazard.sortby(hazard)
    return hazard_sorted[EPISODE_DIM].values


def _write_sample(ds_store: xr.Dataset, shots: np.ndarray, path: Path) -> None:
    """Write the given shots of a device store as one netCDF sample.

    The time_idx columns none of the shots uses (time NaN for every shot) are dropped.
    """
    shots_sorted = np.sort(shots)
    ds_sample = ds_store.sel({EPISODE_DIM: shots_sorted})
    mask_time_used = ds_sample[TIME_COORD].notnull().any(EPISODE_DIM).values
    ds_sample = ds_sample.isel({TIME_DIM: mask_time_used}).load()

    # The zarr chunk and compressor encodings do not carry over to netCDF
    for variable in ds_sample.variables.values():
        variable.encoding = {}
    numeric_vars = [name for name, da in ds_sample.data_vars.items() if np.issubdtype(da.dtype, np.number)]
    encoding = dict.fromkeys(numeric_vars, SAMPLE_ENCODING)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds_sample.to_netcdf(path, encoding=encoding)
    logger.info(f"Wrote {len(shots_sorted)} shots over {ds_sample.sizes[TIME_DIM]} time_idx to {path}")


def make_samples(device: str, store_path: Path | str, sample_dir: Path | str) -> list[Path]:
    """Split a device store by hazard into the SAMPLE_BLOCKS samples of SAMPLE_N_SHOTS shots each.

    low1 and low2 are the first and second SAMPLE_N_SHOTS shots in ascending hazard, high the last.

    Raises:
        ValueError: If the store has fewer shots than the disjoint samples need.
    """
    ds_store = xr.open_zarr(store_path)
    n_shots = ds_store.sizes[EPISODE_DIM]
    n_needed = len(SAMPLE_BLOCKS) * SAMPLE_N_SHOTS
    if n_shots < n_needed:
        raise ValueError(f"{store_path} has {n_shots} shots, {len(SAMPLE_BLOCKS)} disjoint samples need {n_needed}")

    shots_sorted = _shots_by_hazard(ds_store)
    block_shots = {
        "low1": shots_sorted[:SAMPLE_N_SHOTS],
        "low2": shots_sorted[SAMPLE_N_SHOTS : 2 * SAMPLE_N_SHOTS],
        "high": shots_sorted[-SAMPLE_N_SHOTS:],
    }
    paths = []
    for block in SAMPLE_BLOCKS:
        path = _sample_path(sample_dir, device, block)
        _write_sample(ds_store, block_shots[block], path)
        paths.append(path)
    return paths
