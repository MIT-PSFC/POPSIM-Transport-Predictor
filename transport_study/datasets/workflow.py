from abc import ABC, abstractmethod
from functools import cached_property, partial
from pathlib import Path
from typing import ClassVar

import numpy as np
import xarray as xr
from loguru import logger

from transport_study import EPISODE_DIM, RADIAL_DIM, TIME_COORD, TIME_DIM
from transport_study.datasets import UNIFORM_TIMEBASE_DT_S
from transport_study.datasets.plotting import (
    ds_profile_plot,
    ds_profile_time_plot,
    ds_summary_report,
)
from transport_study.signals import (
    PREDICTION_STORE_NAME,
    STORE_HEATING_POWERS,
    STORE_POWERS,
    STORE_SIGNAL_UNITS,
    STORE_SIGNALS,
)


class DataWorkflow(ABC):
    """Builds one device's tensorized POPSIM stores.

    Each entry of STORE_VARIABLES is one Zarr store under final_ds_dir,
    built from the same per-shot processing (process_fn) with its own variable selection.
    The prediction store (PREDICTION_STORE_NAME) holds the shared on-disk schema, signals.STORE_SIGNALS:
    IMAS names in SI units, on (shot, time_idx[, rho_tor_norm]) with time on (shot, time_idx).

    A store is not uniform in time: filtering drops interior timeslices,
    so consumers needing a uniform grid reindex it (see organize_data.reindex_to_uniform_timebase).
    """

    STORE_VARIABLES: ClassVar[dict[str, tuple[str, ...]]] = {PREDICTION_STORE_NAME: STORE_SIGNALS}
    # Attributes set on every store shot: per variable or coordinate (description, units, ref = IMAS path),
    # and dataset level. Units of the shared schema always come from signals.STORE_SIGNAL_UNITS.
    SIGNAL_ATTRS: ClassVar[dict[str, dict[str, str]]] = {}
    STORE_ATTRS: ClassVar[dict[str, str]] = {}

    def __init__(self, ds_name: str, data_assembly_dir: Path | str, max_num_shots: int | None = None):
        """
        Parameters
        ----------
        ds_name : str
            Name of the dataset, the directory under data_assembly_dir everything is written to
        data_assembly_dir : Path | str
            Directory the stores (and any raw per-shot files) are written under
        max_num_shots : int | None
            Maximum number of shots to process (for testing). If None, process all shots.
        """
        self.ds_name = ds_name
        self.data_assembly_dir = Path(data_assembly_dir)
        self.max_num_shots = max_num_shots
        dataset_dir_name = "dataset_full" if max_num_shots is None else f"dataset_{max_num_shots}"
        self.final_ds_dir = self.data_assembly_dir / ds_name / dataset_dir_name

    @abstractmethod
    def shot_identifiers(self) -> list[int]:
        """Shots to process, in processing order."""

    @abstractmethod
    def max_dim_sizes(self, identifiers: list[int]) -> dict[str, int]:
        """Upper bound on every non-episode dimension across the given shots."""

    @abstractmethod
    def load_shot(self, shot: int) -> xr.Dataset | None:
        """One shot in the on-disk schema, with a shot dim of size 1 and time as a coordinate.

        May carry extra processing-only variables, process_fn selects the stored ones.
        Returns None when the shot cannot be used.
        """

    def cull_shot(self, shot_ds: xr.Dataset) -> bool:
        """True if this shot should be left out of the stores. Default: keep everything."""
        return False

    def process_fn(self, shot: int, variables: tuple[str, ...]) -> xr.Dataset | None:
        """One shot of one store: loaded, culled, cut down to the store's variables, and labelled with attributes."""
        shot_ds = self.load_shot(shot)
        if shot_ds is None:
            logger.warning(f"Skipping shot {shot}, it could not be loaded")
            return None
        if self.cull_shot(shot_ds):
            return None
        store_ds = shot_ds[list(variables)]
        store_ds.attrs.update(self.STORE_ATTRS)
        for name, variable in store_ds.variables.items():
            name_str = str(name)
            variable.attrs.update(self.SIGNAL_ATTRS.get(name_str, {}))
            if name_str in STORE_SIGNAL_UNITS:
                variable.attrs["units"] = STORE_SIGNAL_UNITS[name_str]
        return store_ds

    def log_ds_details(self, ds: xr.Dataset):
        logger.info(f"Final dataset dimensions: {ds.dims}")
        logger.info(f"Final dataset variables: {list(ds.data_vars)}")
        # For each variable, log the maximum value and the shot in which it occurs, to check for any outliers that might indicate issues with the processing
        for var in ds.data_vars:
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

    def store_path(self, store_name: str) -> Path:
        """Path of one of this workflow's Zarr stores."""
        return self.final_ds_dir / f"{store_name}.zarr"

    def run_processed_data_workflow(self):
        """Build every STORE_VARIABLES store, then log and plot the prediction store."""
        from popsim.data.dataset_utils import build_tensorized_dataset

        prediction_store_path = self.store_path(PREDICTION_STORE_NAME)
        if prediction_store_path.exists():
            logger.info(f"Dataset already exists at {prediction_store_path}, skipping processing.")
            return

        identifiers = self.shot_identifiers()
        logger.info(f"Processing {len(identifiers)} shots to build dataset")
        # Passing upper bounds lets every shot be padded to a fixed shape up front,
        # so a store never needs to be extended when a later shot is larger
        dim_sizes = self.max_dim_sizes(identifiers)
        logger.info(f"Maximum dimension sizes across shots: {dim_sizes}")

        for store_name, variables in self.STORE_VARIABLES.items():
            store_path = self.store_path(store_name)
            process_store_shot = partial(self.process_fn, variables=variables)
            build_tensorized_dataset(
                process_fn=process_store_shot,
                identifiers=identifiers,
                zarr_path=str(store_path),
                time_dim=TIME_DIM,
                episode_dim=EPISODE_DIM,
                extend_existing=False,
                mb_per_chunk=100,
                dim_sizes=dim_sizes,
            )
            logger.info(f"Saved the {store_name} store to {store_path}")
        self._assert_stores_aligned()

        ds = xr.open_zarr(prediction_store_path)
        self.log_ds_details(ds)

        # Diagnostic plots to sanity-check the resulting dataset (signal ranges,
        # profile coverage, remaining issues)
        try:
            ds_profile_time_plot(
                prediction_store_path,
                self.final_ds_dir / "time_traces",
                title=f"{self.ds_name.upper()} Dataset Time Traces",
            )
        except Exception as e:
            logger.error(f"Error generating time trace plots: {e}")
        try:
            ds_profile_plot(
                prediction_store_path,
                self.final_ds_dir / "profile_traces",
                title=f"{self.ds_name.upper()} Dataset Profile Traces",
            )
        except Exception as e:
            logger.error(f"Error generating profile plots: {e}")
        try:
            ds_summary_report(
                prediction_store_path,
                self.final_ds_dir / "summary_report.pdf",
                title=f"{self.ds_name.upper()} Dataset",
            )
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")

    def _assert_stores_aligned(self):
        """Every extra store must hold exactly the prediction store's shots and times.

        build_tensorized_dataset logs and skips a shot whose processing raises,
        so a failure in one store's pass alone would silently misalign the stores.
        """
        ds_prediction = xr.open_zarr(self.store_path(PREDICTION_STORE_NAME))
        shots_prediction = ds_prediction[EPISODE_DIM].values
        time_prediction = ds_prediction[TIME_COORD].values
        for store_name in self.STORE_VARIABLES:
            if store_name == PREDICTION_STORE_NAME:
                continue
            ds_store = xr.open_zarr(self.store_path(store_name))
            if not np.array_equal(ds_store[EPISODE_DIM].values, shots_prediction):
                raise ValueError(f"The {store_name} store holds different shots than the {PREDICTION_STORE_NAME} store")
            if not np.array_equal(ds_store[TIME_COORD].values, time_prediction, equal_nan=True):
                raise ValueError(f"The {store_name} store has different times than the {PREDICTION_STORE_NAME} store")


class RawFileWorkflow(DataWorkflow):
    """Devices built from raw per-shot files pulled from source (TCV, DIII-D).

    1. make_raw_data_files: one netCDF per shot under raw_data/,
       on a uniform 1 kHz timebase with signals in the on-disk schema.
    2. run_processed_data_workflow: each raw file goes through device_specific_processing,
       power clipping, fresh_profile labelling, filter_ds, and the culls, into the stores.
    """

    # Seconds of data cut from the end of the ip record (see filter_ds).
    # Set per device since the same margin can land in completely different plasma states
    end_margin_s = 0.05

    # Set by each device
    # filter_config: {signal: {"min", "max"}}, a slice with any signal out of range is dropped
    # individual_filter_config: {signal: {"min", "max"}}, an out-of-range value is NaNed
    filter_config: dict[str, dict[str, float]]
    individual_filter_config: dict[str, dict[str, float]] | None

    def __init__(
        self,
        ds_name: str,
        shotlist_file: Path | str | None,
        data_assembly_dir: Path | str,
        max_num_shots: int | None = None,
        min_shot_duration: float = 0.5,
    ):
        """
        Parameters
        ----------
        ds_name : str
            Name of the dataset ('d3d', 'tcv')
        shotlist_file : Path | str | None
            Path to file containing list of shots to process. If None, will call
            _get_shotlist_from_source() to retrieve shotlist from device-specific source.
        data_assembly_dir : Path | str
            Directory where data files are stored and final dataset will be saved
        max_num_shots : int | None
            Maximum number of shots to process (for testing). If None, process all shots.
        min_shot_duration : float
            Minimum duration (in seconds) for a shot to be included in the dataset.
        """
        super().__init__(ds_name, data_assembly_dir, max_num_shots)
        self.raw_data_dir = self.data_assembly_dir / ds_name / "raw_data"
        self.min_shot_duration = min_shot_duration

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

        Called when no shotlist file is provided.
        """

    @abstractmethod
    def make_raw_data_files(self):
        """Create the raw data files by pulling from source

        The resulting files should be one per shot, on a common timebase,
        and have the on-disk signal names
        """

    @abstractmethod
    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset | None:
        """Rename and convert source signals into the on-disk schema"""

    @abstractmethod
    def device_specific_processing(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply any device-specific processing steps before the general workflow"""

    def shot_identifiers(self) -> list[int]:
        """Every shot with a raw file, truncated to max_num_shots."""
        identifiers = sorted(int(path.stem) for path in self.raw_data_dir.glob("*.nc"))
        return identifiers[: self.max_num_shots]

    def max_dim_sizes(self, identifiers: list[int]) -> dict[str, int]:
        """Largest size of every dimension across the raw files.

        Processing only ever drops timeslices, so the raw sizes bound the processed shots.
        """
        dim_sizes: dict[str, int] = {}
        for identifier in identifiers:
            with xr.open_dataset(self.raw_data_dir / f"{identifier}.nc") as raw_ds:
                for dim, size in raw_ds.sizes.items():
                    dim_sizes[str(dim)] = max(dim_sizes.get(str(dim), 0), size)
        dim_sizes.pop(EPISODE_DIM, None)
        return dim_sizes

    def load_shot(self, shot: int) -> xr.Dataset | None:
        """A raw file processed, power-clipped, fresh-labelled, and filtered."""
        with xr.open_dataset(self.raw_data_dir / f"{shot}.nc") as raw_file:
            shot_ds = raw_file.load()

        shot_ds = self.device_specific_processing(shot_ds)
        if shot_ds is None:
            return None

        # No heating or radiated power is physically negative
        for power in STORE_POWERS:
            shot_ds[power] = shot_ds[power].clip(min=0)

        # Label where the profiles are fresh (not made by ffill), BEFORE filter_ds compacts the record
        shot_ds["fresh_profile"] = self._fresh_profile_flags(shot_ds["n_e"])

        shot_ds_filtered = self.filter_ds(shot_ds.copy())
        if shot_ds_filtered is None:
            self._debug_plots(shot_ds)
        return shot_ds_filtered

    @staticmethod
    def _fresh_profile_flags(n_e: xr.DataArray) -> xr.DataArray:
        """1 where a forward-filled profile record carries a new profile, 0 where it holds an earlier one.

        A slice whose profile is all NaN (a failed fit) is never fresh,
        even though after fillna it reads as changed from the slice before.
        """
        n_e_filled = n_e.fillna(0)
        mask_changed = n_e_filled != n_e_filled.shift({TIME_DIM: 1}, fill_value=0)
        mask_first_valid = n_e.notnull().cumsum(TIME_DIM) == 1
        mask_fresh = (mask_changed | mask_first_valid) & n_e.notnull()
        return mask_fresh.any(RADIAL_DIM).astype(np.float32)

    def cull_shot(self, shot_ds: xr.Dataset) -> bool:
        """Device-specific, energy-sanity, and duration culls."""
        shot = int(shot_ds[EPISODE_DIM].values[0])
        if self.device_specific_culling(shot_ds):
            logger.warning(f"Excluding shot {shot} based on device-specific culling criteria")
            culled = True
        else:
            culled = self.energy_sanity_cull(shot_ds) or self.is_too_short(shot_ds)
        if culled:
            self._debug_plots(shot_ds)
        return culled

    def device_specific_culling(self, ds: xr.Dataset) -> bool:
        """Device-specific culling logic, True if this shot should be excluded from the dataset.

        Default: cull when either profile is entirely missing after processing
        and filtering (e.g. filtering cut the shot down to a window with no valid profiles).
        """
        shot_id = ds[EPISODE_DIM].values[0]
        for signal in ["t_e", "n_e"]:
            if ds[signal].isnull().all():
                logger.warning(f"Culling shot {shot_id}: {signal} is all NaN after processing and filtering")
                return True
        return False

    def energy_sanity_cull(self, ds: xr.Dataset) -> bool:
        """Energy sanity check, True if this shot should be excluded.

        The stored energy rise from the start of the (filtered) window to its
        peak cannot exceed the total input energy (ohmic + heating) delivered
        over that same interval. A ratio above 1 means an input power record is
        broken or missing, e.g. MAST shots reaching 0.2 MJ with zero recorded
        NBI power. The rise (not the absolute peak) is used because filtering
        can drop the ramp-up, and energy stored before the window start needs
        no input inside the window.
        Give 5% of leeway to account for measurement noise and integration error.
        Note this ignores radiated power, so it's a conservative check.
        Missing power samples count as zero, which only lowers the input estimate,
        so a healthy shot (integrated input energy far above stored energy) is never culled.
        """
        shot_id = ds[EPISODE_DIM].values[0]
        time = np.asarray(ds[TIME_COORD].values).reshape(-1)
        energy_mhd = np.asarray(ds["energy_mhd"].values).reshape(-1)
        mask_valid = np.isfinite(time) & np.isfinite(energy_mhd)
        if mask_valid.sum() < 2:
            return False
        power_input = np.nan_to_num(np.asarray(ds["power_ohm"].values, dtype=float).reshape(-1), nan=0.0)
        for power in STORE_HEATING_POWERS:
            power_heating = np.asarray(ds[power].values, dtype=float).reshape(-1)
            power_input = power_input + np.nan_to_num(power_heating, nan=0.0)
        order = np.argsort(time[mask_valid])
        time_valid = time[mask_valid][order]
        energy_valid = energy_mhd[mask_valid][order]
        power_valid = power_input[mask_valid][order]
        idx_peak = int(np.argmax(energy_valid))
        energy_input_J = float(np.trapezoid(power_valid[: idx_peak + 1], time_valid[: idx_peak + 1]))
        energy_rise_J = energy_valid[idx_peak] - energy_valid[0]
        if energy_rise_J > (energy_input_J * 1.05):
            logger.info(
                f"Culling shot {shot_id}: stored energy rise {energy_rise_J / 1e6:.3f} MJ exceeds "
                f"integrated input energy {energy_input_J / 1e6:.3f} MJ, input power record is broken or missing"
            )
            return True
        return False

    def is_too_short(self, shot_ds: xr.Dataset) -> bool:
        """True if the shot's valid span or its accumulated valid time is below min_shot_duration.

        filter_ds drops invalid slices outright, so holes show up as time jumps,
        which the span check cannot see. The accumulated time (slice count on the 1 kHz grid) does.
        """
        shot = int(shot_ds[EPISODE_DIM].values[0])
        cleaned_ds = shot_ds.dropna(TIME_DIM, how="all")
        valid_time_duration = 0 if cleaned_ds[TIME_COORD].size == 0 else float(cleaned_ds[TIME_COORD].max() - cleaned_ds[TIME_COORD].min())
        if valid_time_duration < self.min_shot_duration:
            logger.warning(f"Excluding shot {shot} because duration after processing is only {valid_time_duration:.2f} seconds")
            return True

        valid_data_duration = cleaned_ds.sizes[TIME_DIM] * UNIFORM_TIMEBASE_DT_S
        if valid_data_duration < self.min_shot_duration:
            logger.warning(
                f"Excluding shot {shot} because only {valid_data_duration:.3f} seconds of valid data "
                f"remain after processing (span {valid_time_duration:.2f} seconds)"
            )
            return True
        return False

    def has_all_nan_signal(self, ds: xr.Dataset, signals: list[str]) -> bool:
        """True (with a warning naming the signal) if any given signal is entirely NaN."""
        shot_id = ds[EPISODE_DIM].item() if EPISODE_DIM in ds else "unknown"
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

    def filter_ds(self, shot_ds: xr.Dataset) -> xr.Dataset | None:
        """Apply filtering steps based on device config"""

        # Cut all data end_margin_s before ip is NaN to avoid including obviously disruptive data
        valid_time = shot_ds["ip"].notnull().any(dim=EPISODE_DIM)
        last_valid_idx = int(np.where(valid_time.values)[0][-1])
        valid_mask = shot_ds[TIME_COORD] <= shot_ds[TIME_COORD][last_valid_idx] - self.end_margin_s

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

    def _debug_plots(self, shot_ds: xr.Dataset):
        debug_fig_dir = self.final_ds_dir / "debug_plots"
        debug_fig_dir.mkdir(parents=True, exist_ok=True)
        ds_profile_time_plot(shot_ds, debug_fig_dir, title=f"Debug: {shot_ds.shot.values[0]}")
        ds_profile_plot(shot_ds, debug_fig_dir, title=f"Debug: {shot_ds.shot.values[0]}")


class PublishedStoreWorkflow(DataWorkflow):
    """Devices built from a transport-validation-datasets published store (C-Mod, MAST).

    The published store is already filtered, fitted, and in IMAS names and SI units,
    so a shot only needs its stored signals selected,
    its trailing NaN padding trimmed, and ip / b0 taken as magnitudes.
    Shot quality is the published store's responsibility, nothing is culled here.
    """

    def __init__(
        self,
        ds_name: str,
        data_assembly_dir: Path | str,
        published_store_path: Path | str,
        max_num_shots: int | None = None,
    ):
        """
        Parameters
        ----------
        ds_name : str
            Name of the dataset, the directory under data_assembly_dir the stores are written to
        data_assembly_dir : Path | str
            Directory the stores are written under
        published_store_path : Path | str
            The published Zarr store to read
        max_num_shots : int | None
            Maximum number of shots to process (for testing). If None, process all shots.
        """
        super().__init__(ds_name, data_assembly_dir, max_num_shots)
        self.published_store_path = Path(published_store_path)

    @cached_property
    def published_ds(self) -> xr.Dataset:
        """The published store's stored signals and times, loaded into memory.

        Loaded once, since the profiles are chunked over many shots and reading
        them shot by shot would decompress every chunk once per shot.
        A heating system the store does not carry is absent on the device, load_shot zero-fills it.
        """
        ds_store = xr.open_zarr(self.published_store_path)
        published_signals = [name for name in STORE_SIGNALS if name not in STORE_HEATING_POWERS or name in ds_store]
        ds_selected = ds_store[[*published_signals, TIME_COORD]].isel({EPISODE_DIM: slice(0, self.max_num_shots)})
        logger.info(f"Loading {ds_selected.sizes[EPISODE_DIM]} shots from {self.published_store_path}")
        ds_loaded = ds_selected.load()
        # Drop the published store's chunking and compressor encodings, the new stores set their own
        for variable in ds_loaded.variables.values():
            variable.encoding = {}
        return ds_loaded

    def shot_identifiers(self) -> list[int]:
        """Every shot in the published store, truncated to max_num_shots."""
        return [int(shot) for shot in self.published_ds[EPISODE_DIM].values]

    def max_dim_sizes(self, identifiers: list[int]) -> dict[str, int]:
        """The published store's padded sizes."""
        return {TIME_DIM: self.published_ds.sizes[TIME_DIM], RADIAL_DIM: self.published_ds.sizes[RADIAL_DIM]}

    def load_shot(self, shot: int) -> xr.Dataset | None:
        """One shot of the published store in the on-disk schema."""
        shot_ds = self.published_ds.sel({EPISODE_DIM: [shot]})
        # The published store pads the end of each shot with NaN times
        mask_time_valid = shot_ds[TIME_COORD].notnull().squeeze(EPISODE_DIM).values
        shot_ds = shot_ds.isel({TIME_DIM: mask_time_valid}).set_coords(TIME_COORD)

        # The C-Mod store keeps the source sign of ip and b0, the study uses magnitudes
        for name in ["ip", "b0"]:
            magnitude = np.abs(shot_ds[name].values)
            shot_ds[name] = shot_ds[name].copy(data=magnitude)

        for power in STORE_HEATING_POWERS:
            if power not in shot_ds:
                power_absent = np.zeros_like(shot_ds["power_ohm"].values)
                shot_ds[power] = (shot_ds["power_ohm"].dims, power_absent)
        return shot_ds
