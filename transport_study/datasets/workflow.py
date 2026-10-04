"""Builds one device's study store from its transport-validation-datasets store.

transport-validation-datasets (TVD) builds every device: C-Mod and MAST up to a published store,
TCV and DIII-D up to an internal one, since their data has no release permission.
Either holds the shared schema, IMAS names in SI units on (shot, time_idx[, rho_tor_norm]) with time on (shot, time_idx),
already filtered and fitted, one contiguous 1 kHz segment per shot padded with NaN to the longest shot,
and the internal one also carries the raw readings and the equilibrium.
The build here is a lazy xarray pass: it selects the signals the studies read (signals.STUDY_STORE_SIGNALS),
cuts the time axis back to the longest shot and rechunks into ds.zarr.
Shot quality is the TVD store's responsibility, nothing is culled here,
and the studies convert units and take the ip and b0 magnitudes on load (signals.convert_to_working_units).
"""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr
from loguru import logger
from transport_validation_datasets.dataset_utils import (
    episode_chunk_specs,
    write_rechunked_store,
)

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.datasets.plotting import (
    ds_profile_plot,
    ds_profile_time_plot,
    ds_summary_report,
    variable_stats,
)
from transport_study.signals import STUDY_STORE_SIGNALS

if TYPE_CHECKING:
    from collections.abc import Callable

# The study store's name under final_ds_dir, <name>.zarr
STUDY_STORE_NAME = "ds"
# The TVD fit mode whose stores hold one contiguous segment per shot, the windowed modes do not
CONTIGUOUS_FIT_MODE = "sample"
# Target chunk size of every variable in the written store
MB_PER_CHUNK = 100


class StoreWorkflow:
    """One device's study store, built from a TVD published or internal store."""

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

    @property
    def partial_store_path(self) -> Path:
        """Where the store is written before it is complete, so a crashed run leaves nothing at study_store_path."""
        return self.final_ds_dir / f"{STUDY_STORE_NAME}.partial.zarr"

    def select_source(self) -> xr.Dataset:
        """The source store's study signals and times, lazily, cut to max_num_shots and to the longest shot."""
        ds_store = xr.open_zarr(self.source_store_path)
        fit_mode = ds_store.attrs.get("fit_mode")
        if fit_mode != CONTIGUOUS_FIT_MODE:
            raise ValueError(
                f"{self.source_store_path} has fit_mode {fit_mode!r}, only {CONTIGUOUS_FIT_MODE!r} stores are one contiguous segment per shot"
            )
        ds_selected = ds_store[[*STUDY_STORE_SIGNALS, TIME_COORD]]
        # time is written as a data variable, the store writer keeps only the index coordinates in memory.
        # The source encoding would still list it as a coordinate on read-back, so it goes
        if TIME_COORD in ds_selected.coords:
            ds_selected = ds_selected.reset_coords(TIME_COORD)
        for variable in ds_selected.variables.values():
            variable.encoding.pop("coordinates", None)
        ds_selected = ds_selected.isel({EPISODE_DIM: slice(0, self.max_num_shots)})
        # Every shot is padded with NaN times to the longest shot of the source store
        n_time_longest = int(ds_selected[TIME_COORD].notnull().sum(TIME_DIM).max())
        return ds_selected.isel({TIME_DIM: slice(0, n_time_longest)})

    def write_store(self, ds_selected: xr.Dataset) -> None:
        """Write the selection to the partial store, check its shot count, then move it into place."""
        if self.partial_store_path.exists():
            logger.warning(f"Removing the partial store of an earlier run at {self.partial_store_path}")
            shutil.rmtree(self.partial_store_path)
        chunk_specs = episode_chunk_specs(ds_selected, EPISODE_DIM, mb_per_chunk=MB_PER_CHUNK)
        write_rechunked_store(ds_selected, self.partial_store_path, EPISODE_DIM, chunk_specs)
        n_shots_source = ds_selected.sizes[EPISODE_DIM]
        n_shots_written = xr.open_zarr(self.partial_store_path).sizes[EPISODE_DIM]
        if n_shots_written != n_shots_source:
            raise RuntimeError(f"Wrote {n_shots_written} shots to {self.partial_store_path}, the source selection has {n_shots_source}")
        self.partial_store_path.rename(self.study_store_path)

    def log_ds_details(self, ds: xr.Dataset):
        logger.info(f"Final dataset dimensions: {ds.dims}")
        logger.info(f"Final dataset variables: {list(ds.data_vars)}")
        # The extreme shot of every variable points at outliers the processing let through
        for name, stats in variable_stats(ds).items():
            logger.info(f"Stats for {name}")
            logger.info(f"  Max is {stats['max']:.6g} at shot {stats['shot_max']}")
            logger.info(f"  Min is {stats['min']:.6g} at shot {stats['shot_min']}")
            logger.info(f"  Mean is {stats['mean']:.6g}")
            logger.info(f"  Std is {stats['std']:.6g}")

    def plot_store(self) -> None:
        """Diagnostic plots of the written store, each logged with its traceback when it fails."""
        plots: list[tuple[Callable[..., None], Path, str]] = [
            (ds_profile_time_plot, self.final_ds_dir / "time_traces", f"{self.ds_name.upper()} Dataset Time Traces"),
            (ds_profile_plot, self.final_ds_dir / "profile_traces", f"{self.ds_name.upper()} Dataset Profile Traces"),
            (ds_summary_report, self.final_ds_dir / "summary_report.pdf", f"{self.ds_name.upper()} Dataset"),
        ]
        for plot_fn, out_path, title in plots:
            try:
                plot_fn(self.study_store_path, out_path, title=title)
            except Exception:
                logger.opt(exception=True).error(f"{plot_fn.__name__} failed for {self.study_store_path}")

    def run_processed_data_workflow(self):
        """Build the study store, then log and plot it."""
        if self.study_store_path.exists():
            logger.info(f"Dataset already exists at {self.study_store_path}, skipping processing.")
            return

        ds_selected = self.select_source()
        logger.info(f"Writing {ds_selected.sizes[EPISODE_DIM]} shots of {self.source_store_path} to {self.study_store_path}")
        self.final_ds_dir.mkdir(parents=True, exist_ok=True)
        self.write_store(ds_selected)
        logger.info(f"Saved the study store to {self.study_store_path}")

        ds = xr.open_zarr(self.study_store_path)
        self.log_ds_details(ds)
        self.plot_store()
