"""Builds one device's POPSIM store from its transport-validation-datasets store.

transport-validation-datasets (TVD) builds every device: C-Mod and MAST up to a published store,
TCV and DIII-D up to an internal one, since their data has no release permission.
Either holds the shared schema, IMAS names in SI units on (shot, time_idx[, rho_tor_norm]) with time on (shot, time_idx),
already filtered and fitted, and the internal one also carries the raw readings and the equilibrium.
A shot here only needs the signals the studies read (signals.STUDY_STORE_SIGNALS) selected,
its trailing NaN padding trimmed, and ip and b0 taken as magnitudes.
Shot quality is the TVD store's responsibility, nothing is culled here.

A store is not uniform in time: filtering drops interior timeslices,
so consumers needing a uniform grid reindex it (see organize_data.reindex_to_uniform_timebase).
"""

from functools import cached_property
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from transport_study import EPISODE_DIM, RADIAL_DIM, TIME_COORD, TIME_DIM
from transport_study.datasets.plotting import (
    ds_profile_plot,
    ds_profile_time_plot,
    ds_summary_report,
)
from transport_study.signals import STUDY_STORE_SIGNALS

# The study store's name under final_ds_dir, <name>.zarr
STUDY_STORE_NAME = "ds"
# Signals TVD keeps signed, the studies use magnitudes
MAGNITUDE_SIGNALS = ("ip", "b0")


class StoreWorkflow:
    """One device's POPSIM store, built from a TVD published or internal store."""

    def __init__(
        self,
        ds_name: str,
        data_assembly_dir: Path | str,
        source_store_path: Path | str,
        max_num_shots: int | None = None,
    ):
        """
        Parameters
        ----------
        ds_name : str
            Name of the dataset, the directory under data_assembly_dir the store is written to
        data_assembly_dir : Path | str
            Directory the store is written under
        source_store_path : Path | str
            The TVD Zarr store to read, <ds>_published.zarr or <ds>_internal.zarr
        max_num_shots : int | None
            Maximum number of shots to process (for testing). If None, process all shots.
        """
        self.ds_name = ds_name
        self.data_assembly_dir = Path(data_assembly_dir)
        self.source_store_path = Path(source_store_path)
        self.max_num_shots = max_num_shots
        dataset_dir_name = "dataset_full" if max_num_shots is None else f"dataset_{max_num_shots}"
        self.final_ds_dir = self.data_assembly_dir / ds_name / dataset_dir_name

    @property
    def study_store_path(self) -> Path:
        """Path of the Zarr store this workflow writes."""
        return self.final_ds_dir / f"{STUDY_STORE_NAME}.zarr"

    @cached_property
    def source_ds(self) -> xr.Dataset:
        """The source store's study signals and times, loaded into memory.

        Loaded once, since the profiles are chunked over many shots and reading
        them shot by shot would decompress every chunk once per shot.
        """
        ds_store = xr.open_zarr(self.source_store_path)
        ds_selected = ds_store[[*STUDY_STORE_SIGNALS, TIME_COORD]].isel({EPISODE_DIM: slice(0, self.max_num_shots)})
        logger.info(f"Loading {ds_selected.sizes[EPISODE_DIM]} shots from {self.source_store_path}")
        ds_loaded = ds_selected.load()
        # Drop the source store's chunking and compressor encodings, the new store sets its own
        for variable in ds_loaded.variables.values():
            variable.encoding = {}
        return ds_loaded

    def shot_identifiers(self) -> list[int]:
        """Every shot in the source store, truncated to max_num_shots."""
        return [int(shot) for shot in self.source_ds[EPISODE_DIM].values]

    def max_dim_sizes(self) -> dict[str, int]:
        """The source store's padded sizes, upper bounds on every shot."""
        return {TIME_DIM: self.source_ds.sizes[TIME_DIM], RADIAL_DIM: self.source_ds.sizes[RADIAL_DIM]}

    def load_shot(self, shot: int) -> xr.Dataset:
        """One shot of the source store in the study store's schema, with time as a coordinate."""
        shot_ds = self.source_ds.sel({EPISODE_DIM: [shot]})
        # The source store pads the end of each shot with NaN times
        mask_time_valid = shot_ds[TIME_COORD].notnull().squeeze(EPISODE_DIM).values
        shot_ds = shot_ds.isel({TIME_DIM: mask_time_valid}).set_coords(TIME_COORD)
        for name in MAGNITUDE_SIGNALS:
            magnitude = np.abs(shot_ds[name].values)
            shot_ds[name] = shot_ds[name].copy(data=magnitude)
        return shot_ds

    def log_ds_details(self, ds: xr.Dataset):
        logger.info(f"Final dataset dimensions: {ds.dims}")
        logger.info(f"Final dataset variables: {list(ds.data_vars)}")
        # For each variable, log the maximum value and the shot in which it occurs, to check for any outliers that might indicate issues with the processing
        for var in ds.data_vars:
            # A per-shot constant (r0) has no time to take statistics over
            if TIME_DIM not in ds[var].dims:
                continue
            # Compute statistics (needed for dask arrays)
            max_per_shot = ds[var].max(dim=TIME_DIM, skipna=True)
            if len(max_per_shot.sizes) > 1:  # Handle multidimensional case
                collapse_dims = [dim for dim in max_per_shot.dims if dim != EPISODE_DIM]
                max_per_shot = max_per_shot.max(dim=collapse_dims, skipna=True)
            try:
                max_shot_idx = np.nanargmax(max_per_shot.values)
            except ValueError:
                logger.warning(f"Variable {var} has no valid values, skipping stats logging")
                continue
            max_shot = ds[EPISODE_DIM].values[max_shot_idx]
            max_val = max_per_shot.values[max_shot_idx]

            min_per_shot = ds[var].min(dim=TIME_DIM, skipna=True)
            if len(min_per_shot.sizes) > 1:  # Handle multidimensional case
                collapse_dims = [dim for dim in min_per_shot.dims if dim != EPISODE_DIM]
                min_per_shot = min_per_shot.min(dim=collapse_dims, skipna=True)
            min_shot_idx = np.nanargmin(min_per_shot.values)
            min_shot = ds[EPISODE_DIM].values[min_shot_idx]
            min_val = min_per_shot.values[min_shot_idx]

            # float64, since SI densities squared overflow float32
            da_var_f64 = ds[var].astype(np.float64)
            mean = da_var_f64.mean(skipna=True).compute()
            std = da_var_f64.std(skipna=True).compute()

            logger.info(f"Stats for {var}")
            logger.info(f"  Max is {max_val:.6g} at shot {max_shot}")
            logger.info(f"  Min is {min_val:.6g} at shot {min_shot}")
            logger.info(f"  Mean is {mean:.6g}")
            logger.info(f"  Std is {std:.6g}")

    def run_processed_data_workflow(self):
        """Build the study store, then log and plot it."""
        from popsim.data.dataset_utils import build_tensorized_dataset

        if self.study_store_path.exists():
            logger.info(f"Dataset already exists at {self.study_store_path}, skipping processing.")
            return

        identifiers = self.shot_identifiers()
        logger.info(f"Processing {len(identifiers)} shots to build dataset")
        # Passing upper bounds lets every shot be padded to a fixed shape up front,
        # so the store never needs to be extended when a later shot is larger
        dim_sizes = self.max_dim_sizes()
        logger.info(f"Maximum dimension sizes across shots: {dim_sizes}")
        build_tensorized_dataset(
            process_fn=self.load_shot,
            identifiers=identifiers,
            zarr_path=str(self.study_store_path),
            time_dim=TIME_DIM,
            episode_dim=EPISODE_DIM,
            extend_existing=False,
            mb_per_chunk=100,
            dim_sizes=dim_sizes,
        )
        logger.info(f"Saved the study store to {self.study_store_path}")

        ds = xr.open_zarr(self.study_store_path)
        self.log_ds_details(ds)

        # Diagnostic plots to sanity-check the resulting dataset (signal ranges,
        # profile coverage, remaining issues)
        try:
            ds_profile_time_plot(
                self.study_store_path,
                self.final_ds_dir / "time_traces",
                title=f"{self.ds_name.upper()} Dataset Time Traces",
            )
        except Exception as e:
            logger.error(f"Error generating time trace plots: {e}")
        try:
            ds_profile_plot(
                self.study_store_path,
                self.final_ds_dir / "profile_traces",
                title=f"{self.ds_name.upper()} Dataset Profile Traces",
            )
        except Exception as e:
            logger.error(f"Error generating profile plots: {e}")
        try:
            ds_summary_report(
                self.study_store_path,
                self.final_ds_dir / "summary_report.pdf",
                title=f"{self.ds_name.upper()} Dataset",
            )
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
