"""Makes the 'raw' CMOD dataset on mfews, to be processed later by POPSIM"""

import os
from pathlib import Path

import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import (
    LogSettings,
    RetrievalSettings,
)
from disruption_py.settings.output_setting import DatasetOutputSetting
from disruption_py.workflow import get_shots_data
from dynaconf import Dynaconf
from loguru import logger
from threadpoolctl import threadpool_limits

from transport_study import EPISODE_DIM, PACKAGE_ROOT, TIME_COORD, TIME_DIM
from transport_study.datasets import make_uniform_1khz_timebase
from transport_study.datasets.cmod import (
    CMOD_DATASET_SIGNALS,
)
from transport_study.datasets.dispy_utils import summary
from transport_study.datasets.gp_fitting.fit_worker import (
    ShotFitInput,
    ShotFitOutput,
    fit_batch,
)
from transport_study.datasets.workflow import DataWorkflow

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
        skip_profiles: bool = False,
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
        skip_profiles : bool
            If True, skip profile fitting and use zero arrays instead. Useful for testing.
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
            "Wtot_MJ": {"min": 0.001, "max": 2},
            "ne20_line_avg": {"min": 0.01, "max": 4},
            "betan": {"min": 0, "max": 5},
        }
        self.individual_filter_config = None
        # {signal: max_value}: once any of these exceeds its threshold the shot is
        # cut from 10ms before to the end (transient event, see filter_ds). Set
        # thresholds per signal, e.g. {"P_rad_MW": 5.0}.
        self.transient_filter_config = {
            "P_oh_MW": 5.0,  # Shot 1160503009 at t=0.7 has a UFO
            "P_rad_MW": 2.5,
        }
        self.skip_profiles = skip_profiles

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
            log_settings=LogSettings(file_path=None),
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

    def _profiles_dataset_from_fit(self, shot: int, times: np.ndarray, fit_output: ShotFitOutput) -> xr.Dataset:
        """Build the GP-fitted profile dataset (with shot dimension) from fit results."""
        ds_profiles = xr.Dataset(
            data_vars={
                "Te_keV_rho": (("time", "rho"), fit_output.te_fit),
                "Te_keV_rho_error": (("time", "rho"), fit_output.te_std),
                "ne20_rho": (("time", "rho"), fit_output.ne_fit),
                "ne20_rho_error": (("time", "rho"), fit_output.ne_std),
            },
            coords={
                "time": times,
                "rho": self.gp_fit_rho,
            },
        )
        return ds_profiles.expand_dims({"shot": [shot]})

    def _staging_paths(self, shot: int) -> tuple[Path, Path]:
        return (
            self.fit_staging_dir / f"{shot}_thomson.nc",
            self.fit_staging_dir / f"{shot}_efit.nc",
        )

    def _prepare_shot(self, shot: int) -> ShotFitInput | None:
        """Retrieve and stage source data for one shot, returning GP fit inputs.

        Downloads EFIT/0D and Thomson data from the C-Mod MDSplus server and
        caches them as netCDF in fit_staging_dir, so restarts (and the later
        assembly step) don't hit the server again. Returns None if the shot
        has no valid data.
        """
        thomson_path, efit_path = self._staging_paths(shot)
        self.fit_staging_dir.mkdir(parents=True, exist_ok=True)

        if thomson_path.exists() and efit_path.exists():
            logger.info(f"Using staged source data for shot {shot}")
            ds_thomson = xr.load_dataset(thomson_path)
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

        ds_efit.to_netcdf(efit_path)
        ds_thomson.to_netcdf(thomson_path)
        return self._checked_fit_input(shot, ds_thomson)

    def _assemble_shot(self, shot: int, fit_output: ShotFitOutput) -> bool:
        """Combine staged source data and GP fit results into the raw data file."""
        thomson_path, efit_path = self._staging_paths(shot)
        ds_path = self.raw_data_dir / f"{shot}.nc"

        ds_thomson = xr.load_dataset(thomson_path)
        ds_efit = xr.load_dataset(efit_path)
        times = ds_thomson.squeeze(dim="shot", drop=True)["time"].values
        ds_profiles = self._profiles_dataset_from_fit(shot, times, fit_output)

        # Keep the datasets at the TS measurement times for fit diagnostics
        ds_thomson_at_ts_times = ds_thomson
        ds_profiles_at_ts_times = ds_profiles

        # Put each dataset on a 1 kHz timebase, using previous value fill
        max_time = max(
            ds_thomson["time"].max().item(),
            ds_profiles["time"].max().item(),
            ds_efit["time"].max().item(),
        )
        timebase = make_uniform_1khz_timebase(max_time)

        ds_thomson = ds_thomson.reindex(time=timebase, method="ffill")
        ds_profiles = ds_profiles.reindex(time=timebase, method="ffill")
        ds_efit = ds_efit.interp(time=timebase, method="nearest")  # EFIT is already at high time resolution

        ds_assembly = xr.merge([ds_thomson, ds_profiles, ds_efit], compat="override")

        ds_standardized = self.standardize_signal_names(ds_assembly)
        if ds_standardized is None:
            logger.warning(f"Standardization failed for shot {shot}, skipping")
            return False

        ds_standardized.to_netcdf(ds_path)
        logger.info(f"Saved raw dataset for shot {shot} to {ds_path}")

        # Only make fit diagnostic plots for shots that are kept
        try:
            self._debug_plot_profiles(shot, ds_thomson_at_ts_times, ds_profiles_at_ts_times, fit_output=fit_output)
        except Exception as e:
            logger.error(f"Failed to make TS fit diagnostic plot for shot {shot}: {e}")

        # Staged source data is no longer needed once the raw file exists
        thomson_path.unlink(missing_ok=True)
        efit_path.unlink(missing_ok=True)
        return True

    def _debug_plot_profiles(
        self,
        shot: int,
        ds_thomson: xr.Dataset,
        ds_profiles: xr.Dataset,
        debug_plot_dir: Path | str | None = None,
        Te_keV_lim: float | None = 5.0,
        ne20_lim: float | None = 1.8,
        fit_output: ShotFitOutput | None = None,
    ) -> None:
        """Save a PDF comparing the GP fits to the raw TS measurements.

        Core and edge TS channels are differentiated and shown with error bars,
        with the GP fit mean and +-1 sigma band overlaid. One page per sampled
        TS measurement time.

        Parameters
        ----------
        shot : int
            Shot number, used for the file name and plot titles
        ds_thomson : xr.Dataset
            Raw Thomson channel dataset at the TS measurement times
        ds_profiles : xr.Dataset
            GP-fitted profile dataset at the TS measurement times
        debug_plot_dir : Path | str | None
            Directory to save the PDF. If None, uses '<ds_name>/ts_fit_plots'
            next to the raw data directory.
        fit_output : ShotFitOutput | None
            When given, its te_hyps/ne_hyps (rows aligned to ds_thomson's time
            index) are annotated on each panel.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        if debug_plot_dir is None:
            debug_plot_dir = self.data_assembly_dir / self.ds_name / "ts_fit_plots"
        debug_plot_dir = Path(debug_plot_dir)
        debug_plot_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = debug_plot_dir / f"{shot}_ts_gp_fit.pdf"

        # Plot the exact channel data the GP fit consumed (NaN-masked, error
        # floored at 0.1, ne in 1e20 units), not the raw staged signals, so the
        # plotted points and error bars match what the fit actually saw.
        fit_input = self._extract_fit_input(ds_thomson)

        ds_thomson = ds_thomson.squeeze("shot", drop=True)
        ds_profiles = ds_profiles.squeeze("shot", drop=True)

        times = ds_thomson["time"].values
        is_core = ds_thomson["ts_array"].values == "core"

        n_t = len(times)
        step = max(1, n_t // 20)

        with PdfPages(pdf_path) as pdf:
            for i_time in range(0, n_t, step):
                time = times[i_time]
                ds_prof_t = ds_profiles.sel(time=time, method="nearest")
                rho_ch = fit_input.x[i_time]

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                for ax, variable, gp_var, label, ylim, hyps_arr in [
                    (axes[0], "te", "Te_keV_rho", "Te [keV]", Te_keV_lim, None if fit_output is None else fit_output.te_hyps),
                    (axes[1], "ne", "ne20_rho", "ne [1e20 m^-3]", ne20_lim, None if fit_output is None else fit_output.ne_hyps),
                ]:
                    data_y = getattr(fit_input, f"{variable}_y")[i_time]
                    err_y = getattr(fit_input, f"{variable}_err")[i_time]

                    for mask, color, name in [
                        (is_core, "tab:blue", "TS core"),
                        (~is_core, "tab:orange", "TS edge"),
                    ]:
                        valid = mask & np.isfinite(rho_ch) & np.isfinite(data_y) & np.isfinite(err_y)
                        if valid.any():
                            ax.errorbar(
                                rho_ch[valid],
                                data_y[valid],
                                yerr=err_y[valid],
                                fmt="o",
                                ms=4,
                                color=color,
                                label=name,
                                zorder=3,
                            )

                    gp_y = ds_prof_t[gp_var].values
                    gp_err = ds_prof_t[f"{gp_var}_error"].values
                    gp_valid = np.isfinite(gp_y)
                    if gp_valid.any():
                        ax.plot(self.gp_fit_rho[gp_valid], gp_y[gp_valid], color="black", label="GP fit")
                        ax.fill_between(
                            self.gp_fit_rho[gp_valid],
                            (gp_y - gp_err)[gp_valid],
                            (gp_y + gp_err)[gp_valid],
                            color="black",
                            alpha=0.2,
                            label="GP +-1 sigma",
                        )

                    ax.set_xlabel("rho")
                    ax.set_ylabel(label)
                    ax.set_ylim(bottom=0, top=ylim)
                    ax.set_title(f"shot {shot}  t={time:.3f} s")
                    ax.grid(alpha=0.3)
                    if ax.get_legend_handles_labels()[0]:
                        ax.legend(fontsize=8)
                    if hyps_arr is not None and i_time < len(hyps_arr) and np.isfinite(hyps_arr[i_time]).all():
                        var, l1, l2, lw, x0 = hyps_arr[i_time]
                        ax.text(
                            0.98,
                            0.98,
                            f"var={var:.2f}  l1={l1:.2f}  l2={l2:.2f}\nlw={lw:.2f}  x0={x0:.2f}",
                            transform=ax.transAxes,
                            ha="right",
                            va="top",
                            fontsize=7,
                            family="monospace",
                        )

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        logger.info(f"Saved TS fit diagnostic plot to {pdf_path}")

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
            log_settings=LogSettings(file_path=None),
            num_processes=1,
        )
        result = result.set_index(idx=["shot", "time"]).unstack("idx")
        return result

    def make_raw_data_files(self):
        """Create raw data files from source for C-Mod dataset.

        This method retrieves Thomson scattering data, performs GP fitting for profiles,
        retrieves EFIT and 0D signals, combines them on a uniform 1 kHz timebase,
        standardizes signal names, and saves one netCDF file per shot.

        GP fitting runs in-process. If a cluster_config was provided (and
        profiles are not skipped), fitting is dispatched to the cluster via
        make_raw_data_files_distributed() instead.
        """
        if self.cluster_config is not None and not self.skip_profiles:
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

            if not self.skip_profiles:
                # Retrieve and stage source data, then fit profiles in-process
                fit_input = self._prepare_shot(shot)
                if fit_input is None:
                    continue
                # numpy is already imported by this point (this module imports
                # it directly above fit_worker), so fit_worker's own
                # OPENBLAS_NUM_THREADS=1 setdefault came too late to take effect
                # and OpenBLAS defaults to one thread per core. Cap it here
                # instead: GP fit matrices are tiny (tens of points), so
                # multi-threaded BLAS is pure overhead, not speedup.
                with threadpool_limits(1):
                    outputs = fit_batch(
                        {shot: fit_input},
                        x_star=self.gp_fit_rho,
                        min_points=self.fit_min_points,
                        scale_per_slice=self.fit_scale_per_slice,
                        num_workers=self.fit_workers,
                        max_slices_per_shot=21 if DEBUG else None,
                    )
                if self._assemble_shot(shot, outputs[shot]):
                    processed_shots += 1
                continue

            # skip_profiles=True: build the raw file with zero profiles
            # Get EFIT and 0D data
            try:
                ds_efit = self._get_efit_dataset(shot)
            except Exception as e:
                logger.warning(f"Failed to retrieve EFIT data for shot {shot}: {e}")
                continue

            # Skip profile fitting, use zeros instead
            logger.info(f"Skipping profile fitting for shot {shot} (skip_profiles=True)")
            max_time = ds_efit["time"].max().item()
            timebase = make_uniform_1khz_timebase(max_time)

            # Create dummy profile dataset with zeros
            ds_profiles = xr.Dataset(
                data_vars={
                    "Te_keV_rho": (
                        ("time", "rho"),
                        np.zeros((len(timebase), len(self.gp_fit_rho))),
                    ),
                    "Te_keV_rho_error": (
                        ("time", "rho"),
                        np.zeros((len(timebase), len(self.gp_fit_rho))),
                    ),
                    "ne20_rho": (
                        ("time", "rho"),
                        np.zeros((len(timebase), len(self.gp_fit_rho))),
                    ),
                    "ne20_rho_error": (
                        ("time", "rho"),
                        np.zeros((len(timebase), len(self.gp_fit_rho))),
                    ),
                },
                coords={
                    "time": timebase,
                    "rho": self.gp_fit_rho,
                    "shot": shot,
                },
            )
            ds_profiles = ds_profiles.expand_dims("shot")

            ds_efit = ds_efit.interp(time=timebase, method="nearest")
            ds_assembly = xr.merge([ds_profiles, ds_efit], compat="override")

            ds_standardized = self.standardize_signal_names(ds_assembly)
            if ds_standardized is None:
                logger.warning(f"Standardization failed for shot {shot}, skipping")
                continue

            ds_standardized.to_netcdf(ds_path)
            logger.info(f"Saved raw dataset for shot {shot} to {ds_path}")
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

        # C-Mod doesn't have tau_conf from standard diagnostics
        ds["tau_conf"] = xr.zeros_like(ds["Ip_MA"])

        # Only keep variables of interest
        kept_vars = {
            # POWER BALANCE
            "Wtot_MJ",
            "P_oh_MW",
            "P_rad_MW",
            "P_ICRF_MW",
            "P_LH_MW",
            "P_NBI_MW",
            "P_ECRH_MW",
            # PROFILE PREDICTOR TRAINING
            "Te_keV_rho",
            "ne20_rho",
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
            "beta_p",  # EFIT
            "ne20_edge",
        }

        ds = ds[list(kept_vars)]

        # If any *important* signal is all NaN, return None to skip this shot
        for signal in ["Te_keV_rho", "ne20_rho", "Ip_MA"]:
            if ds[signal].isnull().all():
                logger.warning(f"Signal {signal} is all NaN for shot {ds['shot'].item()}, skipping shot.")
                return None

        # Make episode dimension, time dimension, and time coordinate names consistent
        if TIME_DIM not in ds.dims:
            ds = ds.rename_dims({"time": TIME_DIM})
        if EPISODE_DIM not in ds.dims:
            ds = ds.rename_dims({"shot": EPISODE_DIM})
        if TIME_COORD not in ds.coords:
            ds = ds.rename_vars({"time": TIME_COORD})

        return ds

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
        ds["ne20_rho"] = ds["ne20_rho"].where(valid_profile_mask)
        ds["Te_keV_rho"] = ds["Te_keV_rho"].where(valid_profile_mask)

        return ds

    def device_specific_culling(self, ds: xr.Dataset) -> bool:
        """Apply C-Mod specific culling criteria to the dataset

        Returns True if the dataset should be culled, False otherwise
        """
        shot_id = ds.shot.values[0] if "shot" in ds else "unknown"

        # Profiles can become all NaN after the raw-file stage, e.g. when
        # device_specific_processing culls every individual profile or when
        # filtering cuts the shot down to a window with no valid profiles
        for signal in ["Te_keV_rho", "ne20_rho"]:
            if ds[signal].isnull().all():
                logger.warning(f"Culling shot {shot_id}: {signal} is all NaN after processing and filtering")
                return True

        return False
