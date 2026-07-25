"""Makes the 'raw' CMOD dataset on mfews, to be processed later by POPSIM"""

import os
from pathlib import Path

import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import RetrievalSettings
from disruption_py.settings.output_setting import DatasetOutputSetting
from disruption_py.workflow import get_shots_data
from dynaconf import Dynaconf
from loguru import logger
from threadpoolctl import threadpool_limits

from transport_study import PACKAGE_ROOT
from transport_study.datasets import make_uniform_1khz_timebase
from transport_study.datasets.cmod import (
    CMOD_DATASET_SIGNALS,
)
from transport_study.datasets.dispy_utils import passive_log_settings, summary
from transport_study.datasets.gp_fitting.fit_worker import (
    ShotFitInput,
    ShotFitOutput,
    fit_batch,
)
from transport_study.datasets.plotting import ts_fit_pdf
from transport_study.datasets.workflow import (
    PROFILE_FIT_VARS,
    RAW_DATASET_VARS,
    DataWorkflow,
    load_netcdf,
    write_netcdf,
)

DEFAULT_SHOTLIST_FILE = PACKAGE_ROOT / "datasets" / "cmod" / "cmod_shotlist"

config = Dynaconf(settings_files=[Path(PACKAGE_ROOT) / "datasets/cmod/config.toml"])

DEBUG = os.environ.get("PTPS_DEBUG", "False").lower() in ("1", "true")


class CModDataWorkflow(DataWorkflow):
    """C-Mod specific data workflow for creating and processing datasets.

    This workflow retrieves data from C-Mod's MDSPlus server, fits Thomson scattering
    profiles using Gaussian processes, standardizes signal names, and creates a uniform
    1 kHz timebase dataset suitable for POPSIM transport prediction studies.

    Note: This workflow can execute on the present cluster with C-Mod data access.
    """

    # Normalize each slice to O(1) before the GP fit. C-Mod Te (keV) and ne
    # (1e20 m^-3) span different magnitudes, so a single set of absolute
    # hyperparameter bounds only makes sense on normalized data; without this the
    # optimizer sometimes collapsed the amplitude to ~0 and returned a flat,
    # near-zero profile.
    fit_scale_per_slice = True

    def __init__(
        self,
        ds_name: str,
        shotlist_file: Path | str | None,
        data_assembly_dir: Path | str,
        max_num_shots: int | None = None,
        gp_fit_rho: np.ndarray | None = None,
        cluster_config=None,
        fit_workers: int = 1,
    ):
        """Initialize the C-Mod data workflow.

        Parameters
        ----------
        ds_name : str
            Name of the dataset/study, used for directory naming
        shotlist_file : str | None
            Path to file containing list of shots to process. If None, retrieves
            shotlist from SQL database using parameters in config.toml.
        data_assembly_dir : str
            Directory where data files are stored and final dataset will be saved
        max_num_shots : int | None
            Maximum number of shots to process (for testing). If None, process all shots.
        gp_fit_rho : np.ndarray | None
            Radial locations (normalized minor radius rho) for GP profile fitting.
            If None, uses default from config.
        cluster_config : ClusterFitConfig | None
            If provided, GP fitting is dispatched to a SLURM cluster. The C-Mod
            data source is only reachable locally, so data retrieval and dataset
            assembly always run here; only the fitting is shipped out.
        fit_workers : int
            Number of local processes for in-process GP fitting (serial mode only).
        """

        # Use the C-Mod dataset config from datasets/cmod/config.toml
        self.config = config

        # Set up GP fitting rho grid
        if gp_fit_rho is not None:
            self.gp_fit_rho = gp_fit_rho
        else:
            # Use config values
            prof_config = self.config["profile_fitting"]
            self.gp_fit_rho = np.linspace(
                prof_config["rho_min"],
                prof_config["rho_max"],
                prof_config["num_rho_points"],
            )

        # Call parent init (which will call _get_shotlist_from_source if needed)
        super().__init__(
            ds_name,
            shotlist_file,
            data_assembly_dir,
            max_num_shots=max_num_shots,
            cluster_config=cluster_config,
            fit_workers=fit_workers,
        )

        self.filter_config = {
            "Wtot_MJ": {"min": 0.002, "max": 2},
            "ne20_line_avg": {"min": 0.01, "max": 4},
            "betan": {"min": 0, "max": 1.5},
        }
        # Shots from run day with UFO
        self.shot_blacklist = {
            1160503001,
            1160503002,
            1160503003,
            1160503004,
            1160621001,
        }
        self.individual_filter_config = None
        # {signal: max_value}: once any of these exceeds its threshold the shot is
        # cut from 10ms before to the end (transient event, see filter_ds). Set
        # thresholds per signal, e.g. {"P_rad_MW": 5.0}.
        self.transient_filter_config = {
            "P_oh_MW": 5.0,  # Shot 1160503009 at t=0.7 has a UFO
            "P_rad_MW": 2.5,
        }

    def _get_shotlist_from_source(self) -> list[int]:
        """Retrieve shotlist from C-Mod SQL database.

        Uses the summary() function to query the C-Mod database for shots
        matching the criteria in config.toml, then filters to blessed Thomson days.

        Returns
        -------
        list[int]
            List of shot numbers to process
        """
        query_config = self.config["shotlist_query"]

        # Query the SQL database
        data = summary(
            summary_table=query_config["summary_table"],
            ipmax=query_config["ipmax"],
            pulse_length=query_config["pulse_length"],
            min_shot=query_config["min_shot"],
            max_shot=query_config["max_shot"],
            shots=False,
        )
        shotlist = data[:, 0].astype(int).tolist()

        # Build list of blessed Thomson days
        thomson_config = self.config["thomson_filtering"]
        blessed_days = list(thomson_config["blessed_days"])
        for day_range in thomson_config["blessed_day_ranges"]:
            blessed_days.extend(range(day_range[0], day_range[1]))

        # Filter to blessed Thomson days
        shotlist = [shot for shot in shotlist if int(shot / 1000) in blessed_days]

        return shotlist

    def _get_thomson_dataset(self, shot: int) -> xr.Dataset | None:
        """Retrieve Thomson scattering data for a shot.

        Parameters
        ----------
        shot : int
            Shot number to retrieve

        Returns
        -------
        xr.Dataset | None
            Raw Thomson scattering data, or None if retrieval fails
        """
        retrieval_settings = RetrievalSettings(
            run_methods=["get_thomson_channels"],
            only_requested_columns=False,
        )
        result = get_shots_data(
            tokamak=Tokamak.CMOD,
            shotlist_setting=[shot],
            retrieval_settings=retrieval_settings,
            output_setting=DatasetOutputSetting(path=False),
            log_settings=passive_log_settings(),
            num_processes=1,
        )
        if len(result) == 0:
            return None
        result = result.set_index(idx=["shot", "time"]).unstack("idx")
        return result

    def _extract_fit_input(self, ds_thomson: xr.Dataset) -> ShotFitInput:
        """Build GP fit input arrays from raw Thomson scattering data.

        This assumes the input data is raw Thomson scattering data from
        _get_thomson_dataset() where Te is in keV and ne is in m^-3. The fit
        input arrays are Te in keV and ne in 1e20 m^-3. Channel locations are
        the normalized minor radius (rho) computed by disruption_py from the
        channel midplane radius relative to the magnetic axis and LCFS.
        """
        ds_shot = ds_thomson.squeeze(dim="shot", drop=True)
        data_x = ds_shot["ts_channel_rho"].values.T  # shape (time, channel)

        arrays = {}
        for variable in ["te", "ne"]:
            data_y = ds_shot[f"ts_channel_{variable}"].values.T  # shape (time, channel)
            err_y = ds_shot[f"ts_channel_{variable}_error"].values.T  # shape (time, channel)

            if variable == "ne":
                data_y = data_y * 1e-20  # Convert to [1e20 m^-3]
                err_y = err_y * 1e-20
                # Drop density channels too uncertain to constrain the fit
                # (error > 1e20 m^-3). These are typically bad edge/SOL channels
                # Seen on shot 1160609014: a ne~4, err~2 channel past the
                # separatrix (rho~1.05) drove a spike to ne~19 at rho=1.0
                data_y = np.where(err_y > 1.0, np.nan, data_y)
                # Drop density points past the separatrix (rho>1.0) reading
                # > 0.9e20: SOL density is low out there, so such a point is a
                # bad channel, and a lone high one beyond the last pedestal
                # channel makes the GP overshoot upward toward it (shot
                # 1160503029: a rho~1.07, ne~1.45 point with a small error bar -
                # so not caught above - drove a spike to ne~13 at rho=1.0).
                # Restricted to rho>1.0 so genuine H-mode density pedestals at
                # rho 0.9-1.0 are kept.
                data_y = np.where((data_x > 1.0) & (data_y > 0.9), np.nan, data_y)

            # If data or error bar is incredibly small, set to NaN since it's probably bad data
            # At this point, ne is in 1e20 m^-3 and Te is in keV
            data_y = np.where(data_y < 0.001, np.nan, data_y)
            err_y = np.where(err_y < 0.001, np.nan, err_y)

            if variable == "te":
                # Near the magnetic axis, Te this low is not physically real -
                # almost certainly a broken channel, not a genuine reading
                # (unlike near the edge, where Te legitimately falls this low).
                core_problem = (data_x >= 0.0) & (data_x < 0.4) & (data_y < 0.4)
                data_y = np.where(core_problem, np.nan, data_y)

            # Historic data, we're mostly going off vibes anyway
            err_y = np.where(err_y < 0.1, 0.1, err_y)

            arrays[f"{variable}_y"] = data_y
            arrays[f"{variable}_err"] = err_y

        return ShotFitInput(x=data_x, **arrays)

    def _checked_fit_input(self, shot: int, ds_thomson: xr.Dataset) -> ShotFitInput | None:
        """Extract fit inputs, skipping shots the GP fit could only return all NaN for
        (e.g. when the rho mapping failed and ts_channel_rho is all NaN)."""
        fit_input = self._extract_fit_input(ds_thomson)
        if not fit_input.has_fittable_points():
            logger.warning(f"Shot {shot}: no finite (rho, te, ne) channel data to fit, skipping")
            return None
        return fit_input

    def profiles_dataset_from_fit(self, shot: int, times: np.ndarray, fit_output: ShotFitOutput) -> xr.Dataset:
        """Build the GP-fitted profile dataset (with shot dimension) from fit results."""
        ds_profiles = xr.Dataset(
            data_vars={
                "Te_keV_rho": (("time", "rho"), fit_output.te_fit),
                "Te_keV_rho_error": (("time", "rho"), fit_output.te_std),
                "ne20_rho": (("time", "rho"), fit_output.ne_fit),
                "ne20_rho_error": (("time", "rho"), fit_output.ne_std),
                "Te_keV_rho_grad": (("time", "rho"), fit_output.te_grad),
                "Te_keV_rho_grad_error": (("time", "rho"), fit_output.te_grad_std),
                "ne20_rho_grad": (("time", "rho"), fit_output.ne_grad),
                "ne20_rho_grad_error": (("time", "rho"), fit_output.ne_grad_std),
            },
            coords={
                "time": times,
                "rho": self.gp_fit_rho,
            },
        )
        return ds_profiles.expand_dims({"shot": [shot]})

    def staging_paths(self, shot: int) -> tuple[Path, Path]:
        return (
            self.fit_staging_dir / f"{shot}_thomson.nc",
            self.fit_staging_dir / f"{shot}_efit.nc",
        )

    def prepare_shot(self, shot: int) -> ShotFitInput | None:
        """Retrieve and stage source data for one shot, returning GP fit inputs.

        Downloads EFIT/0D and Thomson data from the C-Mod MDSplus server and
        caches them as netCDF in fit_staging_dir, so restarts (and the later
        assembly step) don't hit the server again. Returns None if the shot
        has no valid data.
        """
        thomson_path, efit_path = self.staging_paths(shot)
        self.fit_staging_dir.mkdir(parents=True, exist_ok=True)

        if thomson_path.exists() and efit_path.exists():
            logger.info(f"Using staged source data for shot {shot}")
            ds_thomson = load_netcdf(thomson_path)
            return self._checked_fit_input(shot, ds_thomson)

        # Get EFIT and 0D data
        try:
            ds_efit = self._get_efit_dataset(shot)
        except Exception as e:
            logger.warning(f"Failed to retrieve EFIT data for shot {shot}: {e}")
            return None

        # Get Thomson data
        ds_thomson = self._get_thomson_dataset(shot)
        if ds_thomson is None:
            logger.warning(f"Skipping shot {shot} since no Thomson data was retrieved")
            return None

        write_netcdf(ds_efit, efit_path)
        write_netcdf(ds_thomson, thomson_path)
        return self._checked_fit_input(shot, ds_thomson)

    def assemble_shot(self, shot: int, fit_output: ShotFitOutput) -> bool:
        """Combine staged source data and GP fit results into the raw data file."""
        thomson_path, efit_path = self.staging_paths(shot)
        ds_path = self.raw_data_dir / f"{shot}.nc"

        # Kept at the TS measurement times for the fit diagnostic plot below
        ds_thomson = xr.load_dataset(thomson_path)
        ds_efit = xr.load_dataset(efit_path)
        times = ds_thomson.squeeze(dim="shot", drop=True)["time"].values
        ds_profiles = self.profiles_dataset_from_fit(shot, times, fit_output)

        # Put each dataset on a 1 kHz timebase, using previous value fill
        max_time = max(
            ds_thomson["time"].max().item(),
            ds_profiles["time"].max().item(),
            ds_efit["time"].max().item(),
        )
        timebase = make_uniform_1khz_timebase(max_time)

        ds_assembly = xr.merge(
            [
                ds_thomson.reindex(time=timebase, method="ffill"),
                ds_profiles.reindex(time=timebase, method="ffill"),
                ds_efit.interp(time=timebase, method="nearest"),  # EFIT is already at high time resolution
            ],
            compat="override",
        )

        ds_standardized = self.standardize_signal_names(ds_assembly)
        if ds_standardized is None:
            logger.warning(f"Standardization failed for shot {shot}, skipping")
            return False

        ds_standardized.to_netcdf(ds_path)
        logger.info(f"Saved raw dataset for shot {shot} to {ds_path}")

        # Only make fit diagnostic plots for shots that are kept
        try:
            self.debug_plot_profiles(shot, ds_thomson, fit_output)
        except Exception as e:
            logger.error(f"Failed to make TS fit diagnostic plot for shot {shot}: {e}")

        # Staged source data is no longer needed once the raw file exists
        thomson_path.unlink(missing_ok=True)
        efit_path.unlink(missing_ok=True)
        return True

    def debug_plot_profiles(
        self,
        shot: int,
        ds_thomson: xr.Dataset,
        fit_output: ShotFitOutput,
        debug_plot_dir: Path | str | None = None,
    ) -> None:
        """Save the TS-fit diagnostic PDF for one shot (see plotting.ts_fit_pdf).

        ds_thomson is the raw Thomson channel dataset at the TS measurement
        times, whose rows align with fit_output's. The plotted points are the
        exact channel data the fit consumed (NaN-masked, error floored, ne in
        1e20 units), not the raw staged signals.
        """
        if debug_plot_dir is None:
            debug_plot_dir = self.data_assembly_dir / self.ds_name / "ts_fit_plots"
        pdf_path = Path(debug_plot_dir) / f"{shot}_ts_gp_fit.pdf"

        fit_input = self._extract_fit_input(ds_thomson)
        ds_thomson = ds_thomson.squeeze("shot", drop=True)
        is_core = ds_thomson["ts_array"].values == "core"

        n_pages = ts_fit_pdf(
            pdf_path,
            shot,
            ds_thomson["time"].values,
            fit_input.x,
            {"te": (fit_input.te_y, fit_input.te_err), "ne": (fit_input.ne_y, fit_input.ne_err)},
            fit_output,
            self.gp_fit_rho,
            channel_groups=[(is_core, "tab:blue", "TS core"), (~is_core, "tab:orange", "TS edge")],
        )
        if n_pages:
            logger.info(f"Saved TS fit diagnostic plot ({n_pages} slices) to {pdf_path}")

    def _get_efit_dataset(self, shot: int) -> xr.Dataset:
        """Retrieve EFIT and 0D signals for a shot.

        Parameters
        ----------
        shot : int
            Shot number to retrieve

        Returns
        -------
        xr.Dataset
            Dataset with EFIT and 0D signals
        """
        retrieval_settings = RetrievalSettings(
            run_columns=CMOD_DATASET_SIGNALS,
            time_setting="efit",
            only_requested_columns=True,
        )
        result = get_shots_data(
            tokamak=Tokamak.CMOD,
            shotlist_setting=shot,
            retrieval_settings=retrieval_settings,
            output_setting=DatasetOutputSetting(path=False),
            log_settings=passive_log_settings(),
            num_processes=1,
        )
        result = result.set_index(idx=["shot", "time"]).unstack("idx")
        return result

    def make_raw_data_files(self):
        """Create raw data files from source for C-Mod dataset.

        This method retrieves Thomson scattering data, performs GP fitting for profiles,
        retrieves EFIT and 0D signals, combines them on a uniform 1 kHz timebase,
        standardizes signal names, and saves one netCDF file per shot.

        GP fitting runs in-process. If a cluster_config was provided, fitting is
        dispatched to the cluster via make_raw_data_files_distributed() instead.
        """
        if DEBUG:
            logger.warning("PTPS_DEBUG is set: in-process GP fitting truncates to 21 slices per shot - do not use for production datasets")
        if self.cluster_config is not None:
            self.make_raw_data_files_distributed()
            return

        processed_shots = 0
        for shot in self.shotlist:
            if self.max_num_shots is not None and processed_shots >= self.max_num_shots:
                logger.info(f"Reached maximum number of shots to process: {self.max_num_shots}")
                break

            ds_path = self.raw_data_dir / f"{shot}.nc"
            if ds_path.exists():
                logger.info(f"Raw dataset for shot {shot} already exists at {ds_path}")
                processed_shots += 1
                continue

            # Retrieve and stage source data, then fit profiles in-process
            fit_input = self.prepare_shot(shot)
            if fit_input is None:
                continue
            # numpy is already imported by this point (this module imports it
            # directly above fit_worker), so fit_worker's own
            # OPENBLAS_NUM_THREADS=1 setdefault came too late to take effect and
            # OpenBLAS defaults to one thread per core. Cap it here instead: GP
            # fit matrices are tiny, so multi-threaded BLAS is pure overhead.
            with threadpool_limits(1):
                outputs = fit_batch(
                    {shot: fit_input},
                    x_star=self.gp_fit_rho,
                    min_points=self.fit_min_points,
                    scale_per_slice=self.fit_scale_per_slice,
                    num_workers=self.fit_workers,
                    max_slices_per_shot=21 if DEBUG else None,
                )
            if self.assemble_shot(shot, outputs[shot]):
                processed_shots += 1

        logger.info("Finished making raw data files.")

    def standardize_signal_names(self, ds: xr.Dataset) -> xr.Dataset | None:
        """Rename signals in the dataset to match the POPSIM convention.

        This includes unit conversions and creating derived quantities.
        Also validates that critical signals are present.

        Parameters
        ----------
        ds : xr.Dataset
            Raw dataset with device-specific signal names

        Returns
        -------
        xr.Dataset | None
            Standardized dataset, or None if critical signals are missing
        """
        # POWER BALANCE
        ds["Wtot_MJ"] = ds["wmhd"] / 1e6  # Convert J to MJ
        ds["B0"] = np.abs(ds["btor"])
        ds["Ip_MA"] = np.abs(ds["ip"]) / 1e6  # Convert A to MA
        ds["R0"] = ds["rout"]
        ds["ne20_line_avg"] = ds["n_e"] / 1e20  # Convert m^-3 to 10^20 m^-3
        ds["P_oh_MW"] = ds["p_oh"] / 1e6
        ds["P_rad_MW"] = ds["p_rad"] / 1e6
        ds["P_ICRF_MW"] = ds["p_icrf"] / 1e6
        ds["P_LH_MW"] = ds["p_lh"] / 1e6
        # C-Mod doesn't have NBI or ECRH, set to zero where Ip_MA is valid
        ds["P_NBI_MW"] = xr.zeros_like(ds["Ip_MA"])
        ds["P_ECRH_MW"] = xr.zeros_like(ds["Ip_MA"])

        # PROFILE PREDICTOR TRAINING
        # Te_keV_rho
        # ne20_rho
        # Ip_MA
        # B0
        ds["betan"] = ds["beta_n"]
        ds["ne20_edge"] = ds["ne20_rho"].sel(
            rho=0.9, method="nearest"
        )  # C-Mod doesn't have edge interferometry, get the density from rho=0.9
        # R0
        # kappa
        # a_minor
        ds["delta_top"] = ds["tritop"]
        ds["delta_bot"] = ds["tribot"]

        ds = ds[list(RAW_DATASET_VARS)]

        # If any *important* signal is all NaN, return None to skip this shot
        if self.has_all_nan_signal(ds, ["Te_keV_rho", "ne20_rho", "Ip_MA"]):
            return None

        return self.standardize_dim_names(ds)

    def device_specific_processing(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply C-Mod specific processing steps.

        Parameters
        ----------
        ds : xr.Dataset
            Standardized dataset

        Returns
        -------
        xr.Dataset
            Processed dataset ready for general workflow
        """

        # Cull obviously bad fits, such as when the point at rho = 0 is super low (1160503009 0.83)
        # Or when any profile value at rho < 1.0 is negative
        negative_profile_mask = (ds["ne20_rho"].where(ds["rho"] < 1.0) < 0).any(dim="rho") | (
            ds["Te_keV_rho"].where(ds["rho"] < 1.0) < 0
        ).any(dim="rho")
        low_value_mask = (ds["Te_keV_rho"].sel(rho=0) < 1.0) | (ds["ne20_rho"].sel(rho=0) < 0.3)
        valid_profile_mask = ~(negative_profile_mask | low_value_mask)
        if valid_profile_mask.sum() == 0:
            logger.warning(f"All profiles are invalid for shot {ds['shot'].item()}")
        else:
            logger.debug(
                f"Culled {(~valid_profile_mask).sum().item() / (valid_profile_mask.sum().item()) * 100:.2f}% invalid profiles for shot {ds['shot'].item()}"
            )
        for var in PROFILE_FIT_VARS:
            ds[var] = ds[var].where(valid_profile_mask)

        return ds
