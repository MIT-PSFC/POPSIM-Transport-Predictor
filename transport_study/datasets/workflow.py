import gc
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.datasets import UNIFORM_TIMEBASE_DT_S
from transport_study.datasets.plotting import (
    ds_profile_plot,
    ds_profile_time_plot,
    ds_summary_report,
)

GC_INTERVAL = 40  # Every 40 shots force garbage collection

# The HDF5 library netCDF4 bundles is not built thread safe: concurrent calls
# into it from the prepare_shot thread pool corrupt its internal state and
# segfault the process (seen in H5SL__insert_common). Every netCDF read/write
# reached from a staging thread goes through this lock.
NETCDF_LOCK = threading.Lock()


def write_netcdf(ds: xr.Dataset, path: Path) -> None:
    """Write a dataset to netCDF, serialized against other threads."""
    with NETCDF_LOCK:
        ds.to_netcdf(path)


def load_netcdf(path: Path) -> xr.Dataset:
    """Read a netCDF file fully into memory, serialized against other threads."""
    with NETCDF_LOCK:
        return xr.load_dataset(path)


# GP-fit profile outputs: the values plus their error-bar and gradient companions.
# A timeslice culled during processing must have all of them NaNed together, or
# downstream sees an error bar for a profile that is not there.
PROFILE_FIT_VARS = (
    "Te_keV_rho",
    "Te_keV_rho_error",
    "Te_keV_rho_grad",
    "Te_keV_rho_grad_error",
    "ne20_rho",
    "ne20_rho_error",
    "ne20_rho_grad",
    "ne20_rho_grad_error",
)

# Variables every raw per-shot file carries, in a fixed order
# (see the device workflows' standardize_signal_names)
RAW_DATASET_VARS = (
    # POWER BALANCE
    "Wtot_MJ",
    "P_oh_MW",
    "P_rad_MW",
    "P_ICRF_MW",
    "P_LH_MW",
    "P_NBI_MW",
    "P_ECRH_MW",
    # PROFILE PREDICTOR TRAINING
    *PROFILE_FIT_VARS,
    "Ip_MA",
    "B0",
    "betan",
    "ne20_line_avg",
    "R0",
    "kappa",
    "a_minor",
    "delta_top",
    "delta_bot",
    # OTHER
    "beta_p",
    "ne20_edge",
)


class DataWorkflow(ABC):
    """Class that handles organization of data processing steps

    For this study, the general workflow is:
    1. Create raw data files from source.
    - One file per shot
    - On a common timebase (1 kHz)
    - Standardized signal names
    2. Process and filter data as needed to remove bad shots / fix signals where possible
    - Logging of issues encountered, with plots where relevant to see what went wrong
    3. Combine all shots together into a single xarray Dataset and save to disk

    Only the raw per-shot files are strictly uniform at 1 kHz. Filtering in
    step 2 drops interior timeslices, so the processed dataset can have
    mid-shot dt gaps; consumers needing a uniform grid must reindex
    (see organize_data.reindex_to_uniform_timebase).
    """

    # GP fitting options used by fit_batch / the cluster worker; overridden per device
    fit_min_points = 1
    fit_scale_per_slice = False

    # Seconds of data cut from the end of the Ip record (see filter_ds).
    # Set per device since the same margin can land in completely different plasma states
    # C-Mod cuts Ip sooner after disruption, MAST keeps recording all the way through
    end_margin_s = 0.05

    # A transient_filter_config signal has to cross its threshold on
    # transient_persistence_slices timeslices inside a window of
    # transient_persistence_window_slices before it counts as an event. Counting
    # crossings in a window rather than requiring them consecutive matters:
    # C-Mod 1160503009's UFO holds P_rad above 2.5 MW for 10 ms but dips 0.03 MW
    # under it midway. See _transient_cutoff_time.
    transient_persistence_slices = 5
    transient_persistence_window_slices = 10

    # A GP-fitted density profile is inconsistent with the interferometer when
    # mean_{rho<=1}(ne20_rho) / ne20_line_avg falls outside this range.
    ne_fit_ratio_bounds = (0.7, 1.4)

    # Smallest physical profile value ([keV] for Te_keV_rho, [1e20 m^-3] for
    # ne20_rho) and the open rho interval it is enforced over; see mask_bad_profile_fits.
    profile_floor_value = 0.001
    profile_floor_rho_range = (0.0, 1.0)

    # Furthest a profile is carried forward to bridge a hole made by
    # mask_bad_profile_fits. One C-Mod Thomson block is ~20 slices and one MAST
    # block ~4, so this bridges a single culled block and no more.
    profile_ffill_limit_slices = 25

    def __init__(
        self,
        ds_name: str,
        shotlist_file: Path | str | None,
        data_assembly_dir: Path | str,
        max_num_shots: int | None = None,
        min_shot_duration: float = 0.5,
        cluster_config=None,
        fit_workers: int = 1,
        prepare_workers: int = 1,
    ):
        """
        Parameters
        ----------
        ds_name : str
            Name of the dataset ('d3d', 'tcv', 'cmod')
        shotlist_file : Path | str | None
            Path to file containing list of shots to process. If None, will call
            _get_shotlist_from_source() to retrieve shotlist from device-specific source.
        data_assembly_dir : Path | str
            Directory where data files are stored and final dataset will be saved
        max_num_shots : int | None
            Maximum number of shots to process (for testing). If None, process all shots.
        min_shot_duration : float
            Minimum duration (in seconds) for a shot to be included in the dataset.
        cluster_config : ClusterFitConfig | None
            If provided, GP profile fitting is dispatched to a SLURM cluster
            (see datasets/gp_fitting/dispatcher.py). If None, fitting runs in-process.
        fit_workers : int
            Number of local processes for in-process GP fitting (serial mode only).
        prepare_workers : int
            Threads used to stage source data (see stage_shots). Only raise it
            for sources that tolerate concurrent reads: MAST reads public S3 and
            does, disruption_py's MDSplus connections do not.
        """

        self.ds_name = ds_name
        self.data_assembly_dir = Path(data_assembly_dir)
        self.raw_data_dir = self.data_assembly_dir / ds_name / "raw_data"
        self.fit_staging_dir = self.data_assembly_dir / ds_name / "fit_staging"
        if max_num_shots is None:
            self.final_ds_dir = self.data_assembly_dir / ds_name / "dataset_full"
        else:
            self.final_ds_dir = self.data_assembly_dir / ds_name / f"dataset_{max_num_shots}"

        self.max_num_shots = max_num_shots
        self.min_shot_duration = min_shot_duration
        self.cluster_config = cluster_config
        self.fit_workers = fit_workers
        self.prepare_workers = prepare_workers

        # Signals that flag a transient event (UFO, minor disruption). Subclasses
        # set this to {signal: max_value}; see filter_ds / _transient_cutoff_time.
        self.transient_filter_config = None

        # Shots excluded from the processed dataset
        # Raw data is still fetched, so re-including a shot only needs a dataset rebuild
        self.shot_blacklist: set[int] = set()

        if shotlist_file is None:
            logger.info("No shotlist file provided, retrieving shotlist from device-specific source")
            self.shotlist = self._get_shotlist_from_source()
            logger.info(f"Retrieved {len(self.shotlist)} shots from source")
        else:
            with open(shotlist_file) as f:
                lines = f.readlines()
                self.shotlist = [int(line.strip()) for line in lines if line.strip().isdigit()]
            logger.info(f"Loaded {len(self.shotlist)} shots from {shotlist_file}")

    @abstractmethod
    def _get_shotlist_from_source(self) -> list[int]:
        """Retrieve shotlist from device-specific source.

        This method is called when no shotlist file is provided. Subclasses should
        implement their own logic (e.g., SQL database query, reading from existing dataset).

        Returns
        -------
        list[int]
            List of shot numbers to process
        """

    @abstractmethod
    def make_raw_data_files(self):
        """Create the raw data files by pulling from source

        The resulting files should be one per shot, on a common timebase,
        and have standardized signal names
        """

    def prepare_shot(self, shot: int):
        """Download and validate source data for one shot, returning a ShotFitInput.

        Implemented by workflows that support distributed GP fitting (C-Mod, MAST).
        Must be idempotent: cache downloaded data in fit_staging_dir so a restart
        does not hit the source again. Returns None if the shot is invalid.
        """
        raise NotImplementedError(f"{self.ds_name} workflow does not support distributed GP fitting")

    def assemble_shot(self, shot: int, fit_output) -> bool:
        """Combine staged source data with GP fit results into the raw data file.

        Returns True if the raw file was written. Implemented by workflows that
        support distributed GP fitting (C-Mod, MAST).
        """
        raise NotImplementedError(f"{self.ds_name} workflow does not support distributed GP fitting")

    def clean_cluster_state(self):
        """Cancel this workflow's queued/running cluster fitting jobs and clear staged batches.

        No-op unless cluster_config is set. Called by the CLI's --clean so a
        from-scratch run doesn't adopt stale jobs or reuse leftover batch
        outputs left on the cluster from a previous invocation.
        """
        if self.cluster_config is None:
            return

        from transport_study.datasets.gp_fitting.dispatcher import ClusterFitDispatcher

        dispatcher = ClusterFitDispatcher(self.cluster_config, self.ds_name, self.fit_staging_dir)
        dispatcher.clean()

    def stage_shots(self, target: int) -> tuple[int, dict]:
        """Stage source data for shots that have no raw file yet, in shotlist order.

        Returns (raw files already present, {shot: ShotFitInput}), the two
        together capped at target. Shots whose source data is missing or invalid
        (prepare_shot returns None) are skipped and do not count toward it, so
        the loop keeps going until target shots are actually in hand.

        With prepare_workers > 1 the staging runs in a thread pool. It is
        dominated by source-read latency (S3 round trips for MAST), so threads
        help despite the GIL. Only as many shots as are still needed are ever
        submitted, so the selected set matches the serial order exactly.
        """
        n_existing = 0
        pending: dict[int, object] = {}
        workers = max(1, self.prepare_workers)
        shots = iter(self.shotlist)
        n_seen = 0

        while n_existing + len(pending) < target:
            batch: list[int] = []
            exhausted = False
            while len(batch) < min(workers, target - n_existing - len(pending)):
                shot = next(shots, None)
                if shot is None:
                    exhausted = True
                    break
                n_seen += 1
                if n_seen % GC_INTERVAL == 0:
                    gc.collect()  # Source datasets can pin a lot of memory
                if (self.raw_data_dir / f"{shot}.nc").exists():
                    n_existing += 1
                    continue
                batch.append(shot)

            pending.update({shot: fit_input for shot, fit_input in self._stage_batch(batch, workers) if fit_input is not None})
            if exhausted:
                break

        return n_existing, pending

    def _stage_batch(self, shots: list[int], workers: int) -> list[tuple[int, object]]:
        """prepare_shot over a batch of shots, in shot order, one thread each.

        A shot that raises is logged and dropped rather than killing a staging
        run that may already be hours in.
        """
        if not shots:
            return []

        def stage(shot: int):
            try:
                return self.prepare_shot(shot)
            except Exception as e:
                logger.warning(f"Failed to stage source data for shot {shot}: {e}")
                return None

        if workers == 1 or len(shots) == 1:
            return [(shot, stage(shot)) for shot in shots]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(zip(shots, pool.map(stage, shots), strict=True))

    def make_raw_data_files_distributed(self):
        """Create raw data files with GP fitting dispatched to a SLURM cluster.

        Three phases:
        1. Prepare: download and validate source data locally (the cluster has
           no access to the data source), staging per-shot files and fit inputs.
        2. Fit: ship fit inputs to the cluster in batches and wait for results
           (see ClusterFitDispatcher for batching, dedup, and concurrency).
        3. Assemble: combine staged data and fitted profiles into one raw
           netCDF per shot, identical to the serial workflow's output.
        """
        from transport_study.datasets.gp_fitting.dispatcher import ClusterFitDispatcher

        if self.cluster_config is None:
            raise ValueError("make_raw_data_files_distributed requires a cluster_config")

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.fit_staging_dir.mkdir(parents=True, exist_ok=True)

        target = self.max_num_shots if self.max_num_shots is not None else len(self.shotlist)
        n_existing, pending = self.stage_shots(target)

        logger.info(f"{n_existing} raw files already exist, {len(pending)} shots need GP fitting")
        if not pending:
            logger.info("Nothing to fit, finished making raw data files.")
            return

        dispatcher = ClusterFitDispatcher(self.cluster_config, self.ds_name, self.fit_staging_dir)
        results = dispatcher.run(
            pending,
            x_star=self.gp_fit_rho,
            min_points=self.fit_min_points,
            scale_per_slice=self.fit_scale_per_slice,
        )

        n_assembled = 0
        for shot, fit_output in results.items():
            if fit_output is None:
                logger.warning(f"No fit results for shot {shot} (batch failed); staging kept for retry")
                continue
            if self.assemble_shot(shot, fit_output):
                n_assembled += 1

        logger.info(f"Assembled {n_assembled}/{len(pending)} shots. Finished making raw data files.")

    @abstractmethod
    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename signals in the dataset to match the POPSIM convention"""

    @abstractmethod
    def device_specific_processing(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply any device-specific processing steps before the general workflow"""

    def device_specific_culling(self, ds: xr.Dataset) -> bool:
        """Device-specific culling logic, True if this shot should be excluded from the dataset.

        Default: cull when either profile is entirely missing after processing
        and filtering (e.g. every individual profile was culled, or filtering
        cut the shot down to a window with no valid profiles).
        """
        shot_id = ds["shot"].values[0] if "shot" in ds else "unknown"
        for signal in ["Te_keV_rho", "ne20_rho"]:
            if ds[signal].isnull().all():
                logger.warning(f"Culling shot {shot_id}: {signal} is all NaN after processing and filtering")
                return True
        return False

    # Auxiliary input power signals a device may carry, summed with ohmic power
    # for the energy sanity check
    AUX_POWER_SIGNALS = ("P_NBI_MW", "P_ICRF_MW", "P_LH_MW", "P_ECRH_MW")

    def energy_sanity_cull(self, ds: xr.Dataset) -> bool:
        """Energy sanity check, True if this shot should be excluded.

        The stored energy rise from the start of the (filtered) window to its
        peak cannot exceed the total input energy (ohmic + auxiliary) delivered
        over that same interval. A ratio above 1 means an input power record is
        broken or missing, e.g. MAST shots reaching 0.2 MJ with zero recorded
        NBI power. The rise (not the absolute peak) is used because filtering
        can drop the ramp-up, and energy stored before the window start needs
        no input inside the window.
        Give a little bit of leeway (10%) to account for measurement noise and integration error.
        Note this ignores radiated power, so it's a conservative check.
        Missing power samples count as zero, which only lowers the input estimate,
        so a healthy shot (integrated input energy far above stored energy) is never culled.
        """
        shot_id = ds["shot"].values[0] if "shot" in ds else "unknown"
        if "Wtot_MJ" not in ds or "P_oh_MW" not in ds:
            return False
        time = np.asarray(ds[TIME_COORD].values).reshape(-1)
        wtot = np.asarray(ds["Wtot_MJ"].values).reshape(-1)
        valid = np.isfinite(time) & np.isfinite(wtot)
        if valid.sum() < 2:
            return False
        p_total = np.nan_to_num(np.asarray(ds["P_oh_MW"].values, dtype=float).reshape(-1), nan=0.0)
        for sig in self.AUX_POWER_SIGNALS:
            if sig in ds:
                p_total = p_total + np.nan_to_num(np.asarray(ds[sig].values, dtype=float).reshape(-1), nan=0.0)
        order = np.argsort(time[valid])
        time_v = time[valid][order]
        wtot_v = wtot[valid][order]
        power_v = p_total[valid][order]
        peak = int(np.argmax(wtot_v))
        energy_in_MJ = float(np.trapezoid(power_v[: peak + 1], time_v[: peak + 1]))
        wtot_rise_MJ = wtot_v[peak] - wtot_v[0]
        if wtot_rise_MJ > (energy_in_MJ * 1.05):
            logger.info(
                f"Culling shot {shot_id}: stored energy rise {wtot_rise_MJ:.3f} MJ exceeds "
                f"integrated input energy {energy_in_MJ:.3f} MJ, input power record is broken or missing"
            )
            return True
        return False

    def has_all_nan_signal(self, ds: xr.Dataset, signals: list[str]) -> bool:
        """True (with a warning naming the signal) if any given signal is entirely NaN."""
        shot_id = ds["shot"].item() if "shot" in ds else "unknown"
        for signal in signals:
            if ds[signal].isnull().all():
                logger.warning(f"Signal {signal} is all NaN for shot {shot_id}, skipping shot.")
                return True
        return False

    def standardize_dim_names(self, ds: xr.Dataset) -> xr.Dataset:
        """Make episode dim, time dim, and time coordinate names consistent with POPSIM conventions."""
        if TIME_DIM not in ds.dims:
            ds = ds.rename_dims({"time": TIME_DIM})
        if EPISODE_DIM not in ds.dims:
            ds = ds.rename_dims({"shot": EPISODE_DIM})
        if TIME_COORD not in ds.coords:
            ds = ds.rename_vars({"time": TIME_COORD})
        return ds

    def log_ds_details(self, ds: xr.Dataset):
        logger.info(f"Final dataset dimensions: {ds.dims}")
        logger.info(f"Final dataset variables: {list(ds.data_vars)}")
        # For each variable, log the maximum value and the shot in which it occurs, to check for any outliers that might indicate issues with the processing
        for var in ds.data_vars:
            # Compute statistics (needed for dask arrays)
            max_per_shot = ds[var].max(dim="time_idx", skipna=True)
            if len(max_per_shot.sizes) > 1:  # Handle multidimensional case
                collapse_dims = [dim for dim in max_per_shot.dims if dim != "shot"]
                max_per_shot = max_per_shot.max(dim=collapse_dims, skipna=True)
            try:
                max_shot_idx = np.nanargmax(max_per_shot.values)
            except ValueError:
                logger.warning(f"Variable {var} has no valid values, skipping stats logging")
                continue
            max_shot = ds["shot"].values[max_shot_idx]
            max_val = max_per_shot.values[max_shot_idx]

            min_per_shot = ds[var].min(dim="time_idx", skipna=True)
            if len(min_per_shot.sizes) > 1:  # Handle multidimensional case
                collapse_dims = [dim for dim in min_per_shot.dims if dim != "shot"]
                min_per_shot = min_per_shot.min(dim=collapse_dims, skipna=True)
            min_shot_idx = np.nanargmin(min_per_shot.values)
            min_shot = ds["shot"].values[min_shot_idx]
            min_val = min_per_shot.values[min_shot_idx]

            mean = ds[var].mean(skipna=True).compute()
            std = ds[var].std(skipna=True).compute()

            logger.info(f"Stats for {var}")
            logger.info(f"  Max is {max_val:.6g} at shot {max_shot}")
            logger.info(f"  Min is {min_val:.6g} at shot {min_shot}")
            logger.info(f"  Mean is {mean:.6g}")
            logger.info(f"  Std is {std:.6g}")

    def run_processed_data_workflow(self):
        """Run the data processing workflow"""
        from popsim.data.dataset_utils import build_tensorized_dataset

        if not int(np.version.version.split(".")[0]) >= 2:
            raise RuntimeError("Numpy version must be greater than 2 to run data processing workflow on all devices.")

        zarr_path = self.final_ds_dir / "ds.zarr"
        if zarr_path.exists():
            logger.info(f"Dataset already exists at {zarr_path}, skipping processing.")
            return

        identifiers = [int(p.stem) for p in self.raw_data_dir.glob("*.nc")]
        if self.shot_blacklist:
            n_before = len(identifiers)
            identifiers = [s for s in identifiers if s not in self.shot_blacklist]
            logger.info(f"Excluded {n_before - len(identifiers)} blacklisted shots")
        if self.max_num_shots:
            identifiers = identifiers[: self.max_num_shots]

        logger.info(f"Processing {len(identifiers)} shots to build dataset")

        # Pre-scan the raw files for the maximum size of each dimension. Processing only ever
        # drops timeslices, so the raw sizes are a valid upper bound for the processed shots.
        # Passing these to build_tensorized_dataset lets every shot be padded to a fixed shape
        # up front, so the zarr store never needs to be extended when a later shot is larger.
        dim_sizes: dict[str, int] = {}
        for identifier in identifiers:
            with xr.open_dataset(self.raw_data_dir / f"{identifier}.nc") as raw_ds:
                for dim, size in raw_ds.sizes.items():
                    dim_sizes[dim] = max(dim_sizes.get(dim, 0), size)
        dim_sizes.pop(EPISODE_DIM, None)
        logger.info(f"Maximum dimension sizes across raw files: {dim_sizes}")

        # Run the data processing workflow and save to a POPSIM tensorized dataset
        ds = build_tensorized_dataset(
            process_fn=self.process_fn,
            identifiers=identifiers,
            zarr_path=str(zarr_path),
            time_dim=TIME_DIM,
            episode_dim=EPISODE_DIM,
            extend_existing=False,
            mb_per_chunk=100,
            dim_sizes=dim_sizes,
        )

        logger.info(f"Saved processed dataset to {zarr_path}")

        # Log some stats about the resulting dataset
        self.log_ds_details(ds)

        # Diagnostic plots to sanity-check the resulting dataset (signal ranges,
        # profile coverage, remaining issues)
        try:
            ds_profile_time_plot(
                zarr_path,
                self.final_ds_dir / "time_traces",
                title=f"{self.ds_name.upper()} Dataset Time Traces",
            )
        except Exception as e:
            logger.error(f"Error generating time trace plots: {e}")
        try:
            ds_profile_plot(
                zarr_path,
                self.final_ds_dir / "profile_traces",
                title=f"{self.ds_name.upper()} Dataset Profile Traces",
            )
        except Exception as e:
            logger.error(f"Error generating profile plots: {e}")
        try:
            ds_summary_report(
                zarr_path,
                self.final_ds_dir / "summary_report.pdf",
                title=f"{self.ds_name.upper()} Dataset",
            )
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")

    def _transient_cutoff_time(self, shot_ds: xr.Dataset) -> float | None:
        """Earliest time [s] any transient_filter_config signal exceeds its threshold.

        These signals flag transient events (UFO, minor disruption) that break
        the pre-shot prediction we're after, so once one crosses its threshold
        the shot is no longer usable. A crossing only counts once the signal has
        crossed on transient_persistence_slices timeslices inside a window of
        transient_persistence_window_slices: C-Mod's P_oh_MW is a numerically
        differentiated quantity carrying 0.33 MW of 1 ms noise, so isolated
        samples past 5 MW are noise rather than events, and treating them as
        events truncated 43% of C-Mod shots at mid-flattop (a median of 3 slices
        over threshold, none of which survive an 11 ms median filter). Returns
        the time of the first crossing inside the first qualifying window, or
        None if nothing qualifies (or no config). filter_ds drops data from 10ms
        before this time to end of shot.
        """
        if not self.transient_filter_config:
            return None

        n_persist = max(1, int(self.transient_persistence_slices))
        window = max(n_persist, int(self.transient_persistence_window_slices))
        cutoff_idx = None
        for var, threshold in self.transient_filter_config.items():
            if var not in shot_ds:
                logger.debug(f"Transient filter variable {var} not in dataset for shot {shot_ds.shot.values[0]}")
                continue
            exceed = shot_ds[var] > threshold
            reduce_dims = [dim for dim in exceed.dims if dim != TIME_DIM]
            if reduce_dims:
                exceed = exceed.any(dim=reduce_dims)
            exceed_vals = np.asarray(exceed.values).reshape(-1).astype(int)
            if exceed_vals.size < window:
                continue
            # Crossings per window: entry i counts the window starting at slice i
            crossings = np.convolve(exceed_vals, np.ones(window, dtype=int), mode="valid")
            qualifying = np.where(crossings >= n_persist)[0]
            if not qualifying.size:
                continue
            start = int(qualifying[0])
            first_crossing = start + int(np.argmax(exceed_vals[start : start + window] > 0))
            cutoff_idx = first_crossing if cutoff_idx is None else min(cutoff_idx, first_crossing)

        return None if cutoff_idx is None else float(shot_ds.time[cutoff_idx])

    def filter_ds(self, shot_ds: xr.Dataset) -> xr.Dataset:
        """Apply filtering steps based on device config"""

        # Cut all data end_margin_s before Ip_MA is NAN to avoid including obviously disruptive data
        valid_time = shot_ds["Ip_MA"].notnull().any(dim=EPISODE_DIM)
        last_valid_idx = int(np.where(valid_time.values)[0][-1])
        valid_mask = shot_ds.time <= shot_ds.time[last_valid_idx] - self.end_margin_s

        # Drop everything from 10ms before the first transient event to end of shot
        cutoff_time = self._transient_cutoff_time(shot_ds)
        if cutoff_time is not None:
            logger.info(f"Shot {shot_ds.shot.values[0]}: transient event at t={cutoff_time:.3f}s, cutting from 10ms before")
            valid_mask = valid_mask & (shot_ds.time < cutoff_time - 0.01)

        # Apply full-timeslice filters
        for var, valid_range in self.filter_config.items():
            if var in shot_ds:
                var_mask = shot_ds[var].notnull() & (shot_ds[var] >= valid_range["min"]) & (shot_ds[var] <= valid_range["max"])
                valid_mask = valid_mask & var_mask
            else:
                logger.debug(f"Variable {var} specified in filter_config not found in dataset for shot {shot_ds.shot.values[0]}")

        if valid_mask.sum() == 0:
            logger.warning(f"Excluding shot {shot_ds.shot.values[0]} because all data points are invalid after filtering")
            return None

        # Apply individual filters that set out-of-range values to NaN, but don't drop the entire timeslice.
        if self.individual_filter_config is not None:
            for var, valid_range in self.individual_filter_config.items():
                if var in shot_ds:
                    shot_ds[var] = shot_ds[var].where(
                        (shot_ds[var].notnull()) & (shot_ds[var] >= valid_range["min"]) & (shot_ds[var] <= valid_range["max"]),
                        other=np.nan,
                    )
                else:
                    logger.debug(
                        f"Variable {var} specified in individual_filter_config not found in dataset for shot {shot_ds.shot.values[0]}"
                    )

        shot_ds = shot_ds.where(valid_mask, drop=True)
        return shot_ds

    def mask_bad_profile_fits(self, shot_ds: xr.Dataset) -> xr.Dataset:
        """NaN GP-fit profiles that disagree with the interferometer or collapse to zero.

        Two checks, applied to every PROFILE_FIT_VARS variable together:

        1. `ne_fit_ratio = mean_{rho<=1}(ne20_rho) / ne20_line_avg`. The fit and
           the interferometer normally agree to about 1% (device medians 1.01
           and 1.02), so a ratio outside ne_fit_ratio_bounds means the Thomson
           density channels disagree with each other and the fit split the
           difference (C-Mod 1160527001 sits at 0.69, with core points spanning
           0.19 to 1.05e20 at the same rho). The GP error bars do not widen in
           this case, so no uncertainty threshold catches this class.
        2. Either profile falling below profile_floor_value anywhere inside
           profile_floor_rho_range (open interval), i.e. a fit that collapsed to
           zero inside the plasma.

        The ratio is kept as `ne_fit_ratio` for auditing and re-thresholding
        without another rebuild. It describes the fit as it arrived, so it is
        deliberately not filled along with the profiles below.

        Holes punched by these checks are forward filled - the raw files are
        already ffilled from Thomson times onto the 1 kHz grid at assembly, so
        this only restores what the check removed, and it keeps a culled Thomson
        block from splitting the shot into two shorter segments. The fill is
        capped at profile_ffill_limit_slices so a shot whose fits are bad
        throughout ends up with holes rather than one stale profile smeared over
        the whole discharge. Slices that had no profile to begin with stay empty,
        and process_fn's fresh_profiles then marks the filled slices stale.

        Call from device_specific_processing before any signal derived from the
        profiles (ne20_edge, Te_keV_core) is re-read.
        """
        shot_id = shot_ds["shot"].item() if "shot" in shot_ds else "unknown"
        rho_vals = np.asarray(shot_ds["rho"].values, dtype=float)
        in_plasma = rho_vals[rho_vals <= 1.0]
        if in_plasma.size < 2:
            logger.warning(f"Shot {shot_id}: rho grid has no interior span, skipping profile fit checks")
            return shot_ds

        ne_fit_mean = shot_ds["ne20_rho"].sel(rho=slice(None, 1.0)).integrate("rho") / float(in_plasma[-1] - in_plasma[0])
        shot_ds["ne_fit_ratio"] = ne_fit_mean / shot_ds["ne20_line_avg"]
        ratio_min, ratio_max = self.ne_fit_ratio_bounds
        inconsistent = (shot_ds["ne_fit_ratio"] < ratio_min) | (shot_ds["ne_fit_ratio"] > ratio_max)

        rho_lo, rho_hi = self.profile_floor_rho_range
        interior = (shot_ds["rho"] > rho_lo) & (shot_ds["rho"] < rho_hi)
        floor = self.profile_floor_value
        collapsed = (shot_ds["ne20_rho"].where(interior) < floor).any(dim="rho") | (shot_ds["Te_keV_rho"].where(interior) < floor).any(
            dim="rho"
        )

        bad_profile = inconsistent | collapsed
        n_bad = int(bad_profile.sum())
        if n_bad:
            logger.info(
                f"Shot {shot_id}: masking {n_bad} timeslices with bad profile fits "
                f"({int(inconsistent.sum())} inconsistent with the interferometer, {int(collapsed.sum())} collapsed to zero)"
            )

        had_profile = shot_ds["ne20_rho"].notnull().any(dim="rho")
        for var in PROFILE_FIT_VARS:
            filled = shot_ds[var].where(~bad_profile).ffill(TIME_DIM, limit=self.profile_ffill_limit_slices)
            shot_ds[var] = filled.where(had_profile)

        return shot_ds

    def _debug_plots(self, shot_ds: xr.Dataset):
        debug_fig_dir = self.final_ds_dir / "debug_plots"
        debug_fig_dir.mkdir(parents=True, exist_ok=True)
        ds_profile_time_plot(shot_ds, debug_fig_dir, title=f"Debug: {shot_ds.shot.values[0]}")
        ds_profile_plot(shot_ds, debug_fig_dir, title=f"Debug: {shot_ds.shot.values[0]}")

    def process_fn(self, shot_id: int) -> xr.Dataset:
        raw_ds_path = self.raw_data_dir / f"{shot_id}.nc"
        shot_ds = xr.open_dataset(raw_ds_path)

        # Processing that is specific to the device, implemented in the subclass
        shot_ds = self.device_specific_processing(shot_ds)
        if shot_ds is None:
            logger.warning(f"Skipping shot {shot_id} due to device-specific processing failure")
            return None

        # Processing that is common across devices
        # Ensure powers are non-negative
        power_signals = [sig for sig in shot_ds.data_vars if "P_" in sig and sig.endswith("_MW")]
        for sig in power_signals:
            shot_ds[sig] = shot_ds[sig].clip(min=0)

        # Label where the profiles are fresh (not made by ffill)
        if "fresh_profiles" not in shot_ds:
            ne20 = shot_ds["ne20_rho"]
            ne20_filled = ne20.fillna(0)
            diff_result = ne20_filled != ne20_filled.shift(time_idx=1, fill_value=0)
            first_valid_is_fresh = ne20.notnull().cumsum("time_idx") == 1
            # A slice where the GP fit failed (all-NaN, per fit_worker's NaN-on-
            # failure convention) reads as 0 after fillna and so looks "changed"
            # from the previous slice - require notnull too, so a failed fit
            # isn't mislabeled fresh.
            fresh_profiles_1D = (diff_result | first_valid_is_fresh) & ne20.notnull()
            fresh_profiles = fresh_profiles_1D.any("rho")
            shot_ds["fresh_profiles"] = fresh_profiles.astype(np.float32)

        debug_ds = shot_ds.copy()  # Copy for plotting later if need be
        # Subclass filter config: range checks, disruption cutoff, transient events
        shot_ds = self.filter_ds(shot_ds)
        if shot_ds is None:
            self._debug_plots(debug_ds)
            return None

        # Device-specific culling, after all processing so it sees the corrected data
        if self.device_specific_culling(shot_ds):
            logger.warning(f"Excluding shot {shot_id} based on device-specific culling criteria")
            self._debug_plots(debug_ds)
            return None

        # Culling that is common across devices
        if self.common_culling(shot_id, shot_ds):
            self._debug_plots(debug_ds)
            return None

        return shot_ds

    def common_culling(self, shot_id: int, shot_ds: xr.Dataset) -> bool:
        """Device-independent culling, True if this shot should be excluded."""
        # If input power record cannot account for the stored energy, exclude it
        if self.energy_sanity_cull(shot_ds):
            return True

        # If shot is too short after processing, exclude it
        cleaned_ds = shot_ds.dropna("time_idx", how="all")
        valid_time_duration = 0 if cleaned_ds.time.size == 0 else float(cleaned_ds.time.max() - cleaned_ds.time.min())
        if valid_time_duration < self.min_shot_duration:
            logger.warning(f"Excluding shot {shot_id} because duration after processing is only {valid_time_duration:.2f} seconds")
            return True

        # Also require the accumulated valid time (slice count on the 1 kHz grid) to reach the minimum.
        valid_data_duration = cleaned_ds.sizes["time_idx"] * UNIFORM_TIMEBASE_DT_S
        if valid_data_duration < self.min_shot_duration:
            logger.warning(
                f"Excluding shot {shot_id} because only {valid_data_duration:.3f} seconds of valid data "
                f"remain after processing (span {valid_time_duration:.2f} seconds)"
            )
            return True

        return False
