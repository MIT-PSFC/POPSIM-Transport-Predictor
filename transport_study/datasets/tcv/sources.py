"""Readers for the two TCV sources.

DEFUSE exports are MATLAB v7.3 files (HDF5), read with h5py.
The LIUQE reconstructions of the MEQ databases are MATLAB v5 files, read with scipy.
Only what the workflow uses is read: a few MB of each DEFUSE file and the liuqe_data struct of each MEQ database.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.io
from loguru import logger

from transport_study.datasets.tcv import config
from transport_study.datasets.tcv.profiles import DefuseProfile, LiuqeEquilibria

DEFUSE_FILE_RE = re.compile(r"TCVno(\d+)\.h5")
MEQDB_FILE_RE = re.compile(r"TCV(\d+)_meqdb\.mat")

# DEFUSE signals stored one row per actuator, summed into one trace (ECRH: one row per gyrotron in newer shots)
SUMMED_ROW_SIGNALS = ("ECRH",)


@dataclass(frozen=True)
class DefuseSignal:
    """One DEFUSE 0D signal on its own timebase."""

    time: np.ndarray  # (n,) sorted, unique times [s]
    values: np.ndarray  # (n,)


def defuse_path(shot: int) -> Path:
    return Path(config["data_sources"]["defuse_dir"]) / f"TCVno{shot}.h5"


def meqdb_path(shot: int) -> Path:
    return Path(config["data_sources"]["meqdb_dir"]) / f"TCV{shot}_meqdb.mat"


def _shots_in(directory: Path, file_re: re.Pattern) -> set[int]:
    shots = set()
    for path in directory.iterdir():
        match = file_re.fullmatch(path.name)
        if match:
            shots.add(int(match.group(1)))
    return shots


def find_tcv_shots() -> list[int]:
    """Every shot with both a DEFUSE export and a LIUQE MEQ database, sorted."""
    defuse_shots = _shots_in(Path(config["data_sources"]["defuse_dir"]), DEFUSE_FILE_RE)
    meqdb_shots = _shots_in(Path(config["data_sources"]["meqdb_dir"]), MEQDB_FILE_RE)
    return sorted(defuse_shots & meqdb_shots)


def _is_placeholder(dataset: h5py.Dataset) -> bool:
    """True for a DEFUSE stand-in for a missing signal rather than data.

    MATLAB v7.3 stores an empty array as its shape, flagged by a MATLAB_empty attribute.
    DEFUSE also marks a missing fit with a uint64 [0 0], while real data is always single or double.
    """
    is_empty = bool(dataset.attrs.get("MATLAB_empty", 0))
    is_floating = dataset.attrs.get("MATLAB_class") in (b"single", b"double")
    return is_empty or not is_floating


def _unique_finite_times(time: np.ndarray) -> np.ndarray:
    """Indices that sort the finite times and drop repeats, since the hold onto the timebase needs increasing times."""
    idx_finite = np.flatnonzero(np.isfinite(time))
    _, idx_unique = np.unique(time[idx_finite], return_index=True)
    return idx_finite[idx_unique]


def _read_signal(group: h5py.Group, name: str) -> DefuseSignal | None:
    """One 0D signal, None when absent, empty, or laid out in a way it should not be."""
    if "signal" not in group or "time" not in group:
        return None
    signal = group["signal"]
    if not isinstance(signal, h5py.Dataset) or _is_placeholder(signal) or _is_placeholder(group["time"]):
        return None
    time = np.asarray(group["time"][()], dtype=np.float64).ravel()
    values_rows = np.atleast_2d(np.asarray(signal[()], dtype=np.float64))
    if values_rows.shape[0] == time.size and values_rows.shape[1] != time.size:
        values_rows = values_rows.T
    if values_rows.shape[1] != time.size:
        logger.warning(f"DEFUSE {name} has shape {values_rows.shape} against {time.size} times, treating it as absent")
        return None
    if values_rows.shape[0] > 1 and name not in SUMMED_ROW_SIGNALS:
        logger.warning(f"DEFUSE {name} has {values_rows.shape[0]} rows, treating it as absent")
        return None
    # An actuator row that is NaN contributes nothing, a time with every row NaN stays NaN
    mask_any_finite = np.isfinite(values_rows).any(axis=0)
    values_summed = np.nansum(values_rows, axis=0)
    values = np.where(mask_any_finite, values_summed, np.nan)
    idx_keep = _unique_finite_times(time)
    return DefuseSignal(time=time[idx_keep], values=values[idx_keep])


def _read_profile(group: h5py.Group, name: str) -> DefuseProfile | None:
    """One profile fit, None when absent or empty."""
    fit = group.get("signal")
    if not isinstance(fit, h5py.Group) or not all(key in fit for key in ("t", "x", "z")):
        return None
    if any(_is_placeholder(fit[key]) for key in ("t", "x", "z")):
        return None
    time = np.asarray(fit["t"][()], dtype=np.float64).ravel()
    rho_pol = np.asarray(fit["x"][()], dtype=np.float64).ravel()
    values = np.asarray(fit["z"][()], dtype=np.float64)
    if values.shape == (rho_pol.size, time.size) and rho_pol.size != time.size:
        values = values.T
    if values.shape != (time.size, rho_pol.size):
        logger.warning(f"DEFUSE {name} fit has shape {values.shape} against {time.size} times and {rho_pol.size} points")
        return None
    if not (np.diff(rho_pol) > 0).all():
        logger.warning(f"DEFUSE {name} fit points are not increasing in rho_pol")
        return None
    idx_keep = _unique_finite_times(time)
    return DefuseProfile(time=time[idx_keep], rho_pol=rho_pol, values=values[idx_keep])


def read_defuse(
    path: Path, signal_names: tuple[str, ...], profile_names: tuple[str, ...]
) -> tuple[dict[str, DefuseSignal], dict[str, DefuseProfile]]:
    """The requested 0D signals and profile fits of one DEFUSE export, each left out when absent or empty.

    Units are DEFUSE's own: SI, except NBI, NBI2 and ECRH in MW.
    """
    signals = {}
    profiles = {}
    with h5py.File(path, "r") as defuse_file:
        signal_root = defuse_file["SIG"]
        for name in signal_names:
            if name in signal_root:
                signal = _read_signal(signal_root[name], name)
                if signal is not None:
                    signals[name] = signal
        for name in profile_names:
            if name in signal_root:
                profile = _read_profile(signal_root[name], name)
                if profile is not None:
                    profiles[name] = profile
    return signals, profiles


def read_liuqe(path: Path) -> LiuqeEquilibria:
    """The LIUQE reconstructions of one MEQ database.

    Loads only the liuqe_data struct, about 250 MB in memory while it is read.
    """
    mat = scipy.io.loadmat(path, variable_names=["liuqe_data"], simplify_cells=True)
    liuqe = mat["liuqe_data"]
    reconstructions = liuqe["LY"]
    time = np.atleast_1d(np.asarray(reconstructions["t"], dtype=np.float64))
    rho_pol = np.asarray(liuqe["L"]["pQ"], dtype=np.float64).ravel()
    surfaces_by_time = (rho_pol.size, time.size)
    inverse_q = np.asarray(reconstructions["iqQ"], dtype=np.float64).reshape(surfaces_by_time).T
    return LiuqeEquilibria(time=time, rho_pol=rho_pol, inverse_q=inverse_q)
