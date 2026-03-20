import glob
import io
import os
import tarfile
import tempfile
import warnings

import matplotlib.pyplot as plt
import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import (
    TimeSetting,
    TimeSettingParams,
)
from disruption_py.settings.time_setting import _postprocess
from freeqdsk import aeqdsk
from loguru import logger

from transport_study import PACKAGE_ROOT

warnings.filterwarnings(
    "ignore",
    message=r"Encountered variables at the end of an A-EQDSK file that are not recognised by FreeQDSK*",
    category=UserWarning,
    module=r"freeqdsk\.aeqdsk",
)


DEFAULT_SHOTLIST_FILE = os.path.join(
    PACKAGE_ROOT, "datasets", "d3d", "HBP_shotlist_2024"
)


class Uniform1kHzTimeSetting(TimeSetting):
    """
    Time setting for creating a uniform timebase at 1 kHz, based on the maximum EFIT time.
    """

    def _get_times(self, params: TimeSettingParams) -> np.ndarray:
        """
        Parameters
        ----------
        params : TimeSettingParams
            Parameters needed to retrieve the timebase.

        Returns
        -------
        np.ndarray
            Array of times in the timebase.
        """
        (efit_time,) = params.mds_conn.get_dims(
            r"\efit_aeqdsk:ali", tree_name="_efit_tree"
        )
        # If timebase is much slower than 1 kHz, log a warning
        typical_delta = np.median(np.diff(efit_time))
        if typical_delta > 2:
            logger.critical(
                f"EFIT timebase is much slower than 1 kHz (typical delta: {typical_delta:.3f} ms). This may cause issues with interpolation and data quality."
            )

        max_time = np.max(efit_time)
        if params.tokamak == Tokamak.CMOD:
            times = np.round(np.arange(0, max_time + 1e-3, 1e-3), 3)
            efit_time_unit = "s"
        if params.tokamak == Tokamak.D3D:
            times = np.round(np.arange(0, max_time + 1, 1), 0)
            efit_time_unit = "ms"
        return _postprocess(times=times, units=efit_time_unit)


class Uniform1MHzTimeSetting(TimeSetting):
    """
    Time setting for creating a uniform timebase at 1 MHz, based on the maximum EFIT time.
    """

    def _get_times(self, params: TimeSettingParams) -> np.ndarray:
        """
        Parameters
        ----------
        params : TimeSettingParams
            Parameters needed to retrieve the timebase.

        Returns
        -------
        np.ndarray
            Array of times in the timebase.
        """
        (efit_time,) = params.mds_conn.get_dims(
            r"\efit_aeqdsk:ali", tree_name="_efit_tree"
        )

        max_time = np.max(efit_time)
        if params.tokamak == Tokamak.CMOD:
            times = np.round(np.arange(0, max_time + 1e-6, 1e-6), 6)
            efit_time_unit = "s"
        if params.tokamak == Tokamak.D3D:
            times = np.round(np.arange(0, max_time + 1e-3, 1e-3), 3)
            efit_time_unit = "ms"
        return _postprocess(times=times, units=efit_time_unit)


def _clean_aeqdsk_file(f):
    """Clean AEQDSK file by removing invalid text lines that don't fit Fortran format.

    Some AEQDSK files have stray text labels (like "MAG", "EQU") at the end that are not
    valid floating-point values for Fortran format descriptors. This function removes
    lines that contain only alphabetic characters (pure garbage text).

    Parameters
    ----------
    f : file-like object
        File handle to read from

    Returns
    -------
    io.StringIO
        A file-like object with cleaned content
    """

    try:
        content = f.read()
        lines = content.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                # Keep blank lines as they're part of the format
                cleaned_lines.append(line)
            elif any(c.isdigit() for c in stripped):
                # Keep lines that contain at least one digit (numeric data)
                # This preserves all data lines while removing pure text like "EQU" or "MAG"
                cleaned_lines.append(line)
            else:
                # Skip pure text lines (like "EQU", "MAG", etc.)
                logger.debug(f"Skipping invalid AEQDSK line: {stripped}")

        # Reconstruct the file content and return as StringIO
        cleaned_content = "\n".join(cleaned_lines)
        string_io = io.StringIO(cleaned_content)

    except Exception as e:
        logger.error(f"Error while cleaning AEQDSK file: {e}")
        # If there's an error, return the original content to avoid data loss
        f.seek(0)
        return f

    return string_io


def disruption_efit(efit_tgz_path: str, shot: int) -> xr.Dataset:
    """Directly parse the saved 1 kHz EFIT results into an xarray dataset for a shot

    We re-computed EFIT01 at 1 kHz, but the DIII-D data curators did not allow us to have a dedicated tree in MDSPlus,
    so we had to put the results under scratch paths like EFIT02-EFIT06 or something
    Of course some shots got overwritten by other researchers using the same scratch paths, so we can't rely on it being consistent.
    So here I'm just going to where we have the results saved and parse the data directly into Xarray
    """

    if not os.path.exists(efit_tgz_path):
        logger.warning(f"EFIT tgz file for shot {shot} not found at {efit_tgz_path}")
        return None

    archive_dir = os.path.dirname(efit_tgz_path)
    efit_nc_path = os.path.join(archive_dir, f"{shot}.nc")
    if os.path.exists(efit_nc_path):
        logger.info(
            f"Found pre-extracted EFIT netCDF file for shot {shot} at {efit_nc_path}, loading from it instead of re-extracting from tgz"
        )
        ds = xr.open_dataset(efit_nc_path)
        # Make sure all the data vars have ("shot", "time") as their dimensions
        for var in ds.data_vars:
            if set(ds[var].dims) != {"shot", "time"}:
                ds[var] = (
                    ds[var].expand_dims({"shot": [shot]}).transpose("shot", "time")
                )
        return ds

    start_dir = os.getcwd()

    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(efit_tgz_path, "r:gz") as tar:
            tar.extractall(path=tmpdir)
        os.chdir(os.path.join(tmpdir, "efit"))
        a_files = glob.glob("a*")  # Mainly 0D scalars
        g_files = glob.glob("g*")  # Grids and boundaries

        # Ensure they're sorted from low to high
        for files in [g_files, a_files]:
            files.sort()
        if len(a_files) != len(g_files):
            raise ValueError("Inconstent number of EFIT files!")

        datasets = []
        for _i, a_file in enumerate(a_files):  # Limit to first 100 files
            with open(a_file) as f:
                cleaned_f = _clean_aeqdsk_file(f)
                a_data = aeqdsk.read(cleaned_f)

            # Create dataset for this time slice
            # The only things we're using in this study are as follows:
            # TODO(ZanderKeith): Complete this after you verify they line up
            # major radius
            # minor radius
            # X point R and Z
            data_vars = {
                "gapin": a_data["oleft"] / 100,  # Inner gap
                "rxpt1": a_data["rseps1"] / 100,  # Lower X-point R
                "zxpt1": a_data["zseps1"] / 100,  # Lower X-point Z
                "rxpt2": a_data["rseps2"] / 100,  # Upper X-point R
                "zxpt2": a_data["zseps2"] / 100,  # Upper X-point Z
                "rsurf": a_data["rout"] / 100,  # Geometric major radius
                "aminor": a_data["aout"] / 100,  # Geometric minor radius
                "kappa": a_data["eout"],  # Elongation
                "tritop": a_data["doutu"],  # Triangularity at the top
                "tribot": a_data["doutl"],  # Triangularity at the bottom
                "beta_p": a_data["betap"],  # Poloidal beta
                "betat": a_data["betat"],  # Toroidal beta
            }

            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": a_data["time"] / 1e3,  # Convert ms to s
                },
            )

            # If the x-point coordinates are broken (e.g. -9.9), set them to NaN
            # TODO(ZanderKeith): This should be done in a common location for things obtained from EFIT, too
            valid_xpoint_mask = (ds["rxpt1"] > 0) & (ds["rxpt2"] > 0)
            for var in ["rxpt1", "zxpt1", "rxpt2", "zxpt2"]:
                ds[var] = ds[var].where(valid_xpoint_mask, other=np.nan)

            datasets.append(ds)

        final_dataset = xr.concat(datasets, dim="time")

    os.chdir(start_dir)
    # Add shot dimension
    final_dataset = final_dataset.expand_dims({"shot": [shot]}).transpose(
        "shot", "time"
    )
    final_dataset.to_netcdf(efit_nc_path)
    return final_dataset


def compare_efits(ds_fast: xr.Dataset, ds_slow: xr.Dataset, shot: int, fig_path: str):
    num_data_vars = len(ds_fast.data_vars)

    ds_fast_shot = ds_fast.sel(shot=shot)
    ds_slow_shot = ds_slow.sel(shot=shot)

    fig, axs = plt.subplots(num_data_vars, 1, figsize=(10, 5 * num_data_vars))
    for i, var in enumerate(ds_fast.data_vars):
        axs[i].plot(
            ds_fast_shot["time"],
            ds_fast_shot[var].data,
            label="Fast EFIT",
            color="blue",
        )
        axs[i].plot(
            ds_slow_shot["time"],
            ds_slow_shot[var].data,
            label="Slow EFIT",
            color="orange",
            linestyle="dashed",
        )
        axs[i].set_title(f"{var} comparison for shot {shot}")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel(var)
        axs[i].legend()
        axs[i].grid()

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def compare_densities(ds: xr.Dataset, shot: int, fig_path: str):
    ds = ds.isel(shot=0)
    dssneped = ds["dssneped"].data / 10
    ne_line_avg = ds["ne_line_avg"].data / 2e20
    dstdenp = ds["dstdenp"].data / 10

    ne_prof_90 = ds["ne_psi"].sel(psi_n=0.9, method="nearest") / 1e20
    ne_prof_95 = ds["ne_psi"].sel(psi_n=0.95, method="nearest") / 1e20
    ne_prof_100 = ds["ne_psi"].sel(psi_n=1.0, method="nearest") / 1e20

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.plot(
        ds["time"],
        dssneped,
        label="dssneped",
        color="blue",
    )
    axs.plot(
        ds["time"],
        ne_line_avg,
        label="ne_line_avg",
        color="orange",
        linestyle="dashed",
    )
    axs.plot(
        ds["time"],
        dstdenp,
        label="dstdenp",
        color="green",
        linestyle="dotted",
    )
    for ne_prof, psi_n in zip(
        [ne_prof_90, ne_prof_95, ne_prof_100],
        [0.9, 0.95, 1.0],
        strict=True,
    ):
        axs.plot(
            ds["time"],
            ne_prof.data,
            label=f"ne_psi(psi_n={psi_n})",
            linestyle="dashdot",
        )

    axs.set_title(f"Edge density comparison for shot {shot}")
    axs.set_xlabel("Time [s]")
    axs.set_ylabel("Density [10^20 m^-3]")
    axs.legend()
    axs.grid()

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
