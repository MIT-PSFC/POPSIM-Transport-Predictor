"""Convert a trajectory optimization output directory to an Excel spreadsheet.

Usage:
    python trajopt_to_excel.py <output_dir> [--out <path.xlsx>]

    <output_dir> is the path to a single trajopt output, e.g.:
        /path/to/working_dir/case.../outputs/trajopt.N.od_True.case.../

The resulting spreadsheet has two sheets:
    Waypoints    — the optimizer control points, matching the instructions.txt table
    Full Waveform — all time steps, including derived shape quantities
"""

import sys
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import xarray as xr

from transport_study.datasets.d3d.d3d_dataset import INNER_WALL


def _derive_shape(
    R0: np.ndarray,
    gapin: np.ndarray,
    rxbot: np.ndarray,
    zxbot: np.ndarray,
    rxtop: np.ndarray,
    zxtop: np.ndarray,
) -> dict[str, np.ndarray]:
    a_minor = R0 - gapin - INNER_WALL
    kappa = np.abs(zxtop - zxbot) / (a_minor * 2)
    delta_bot = (R0 - rxbot) / a_minor
    delta_top = (R0 - rxtop) / a_minor
    return {
        "a_minor": a_minor,
        "kappa": kappa,
        "delta_bot": delta_bot,
        "delta_top": delta_top,
    }


def _nearest_valid_idx(time_arr: np.ndarray, gapin_arr: np.ndarray, t: float) -> int:
    """Return the index of the time point nearest to t that has a valid (non-NaN) gapin."""
    valid = ~np.isnan(gapin_arr)
    if not valid.any():
        return int(np.argmin(np.abs(time_arr - t)))
    dists = np.where(valid, np.abs(time_arr - t), np.inf)
    return int(np.argmin(dists))


def _build_waypoints_df(ds: xr.Dataset) -> pd.DataFrame:
    """Rows = signals, columns = traj_time waypoints. Matches instructions.txt layout."""
    time_arr = ds["time"].values

    traj_times = ds.attrs.get("traj_times", None)
    if traj_times is None:
        # Fall back: 5 evenly-spaced points
        idxs = np.linspace(0, len(time_arr) - 1, 5, dtype=int)
        traj_times = time_arr[idxs]

    traj_times = np.atleast_1d(np.array(traj_times, dtype=float))
    gapin_arr = ds["gapin_opt"].values
    idxs = [_nearest_valid_idx(time_arr, gapin_arr, t) for t in traj_times]
    time_cols = [f"t={time_arr[i]:.3f}s" for i in idxs]

    shape = _derive_shape(
        ds["idtrp"].values,
        ds["gapin_opt"].values,
        ds["idtrxbot"].values,
        ds["idtzxbot"].values,
        ds["idtrxtop"].values,
        ds["idtzxtop"].values,
    )

    def row(label, status, arr):
        d = {"Signal": label, "Status": status}
        for col, idx in zip(time_cols, idxs, strict=False):
            d[col] = float(arr[idx])
        return d

    rows = []

    # Unchanged signals (reference waveforms, not optimized)
    unchanged = [
        ("iptipp / Ip [A]", "iptipp"),
        ("bttbt / B0 [T]", "bttbt"),
        ("bmtpwrtar / betan [-]", "bmtpwrtar"),
    ]
    for label, var in unchanged:
        rows.append(row(label, "UNCHANGED", ds[var].values))

    rows.append({"Signal": "---", "Status": ""})  # separator

    # Optimized PCS signals
    optimized = [
        ("dstdenp [1e19 m\u207b\u00b3]", ds["dstdenp"].values),
        ("idtrp / R0 [m]", ds["idtrp"].values),
        ("gapin [m]", ds["gapin_opt"].values),
        ("idtrxbot [m]", ds["idtrxbot"].values),
        ("idtzxbot [m]", ds["idtzxbot"].values),
        ("idtrxtop [m]", ds["idtrxtop"].values),
        ("idtzxtop [m]", ds["idtzxtop"].values),
    ]
    for label, arr in optimized:
        rows.append(row(label, "OPTIMIZED", arr))

    rows.append({"Signal": "--- desired shape ---", "Status": ""})  # separator

    # Derived shape quantities
    derived = [
        ("a_minor [m]", shape["a_minor"]),
        ("kappa [-]", shape["kappa"]),
        ("delta_top [-]", shape["delta_top"]),
        ("delta_bot [-]", shape["delta_bot"]),
    ]
    for label, arr in derived:
        rows.append(row(label, "DERIVED", arr))

    return pd.DataFrame(rows)


def _build_full_df(ds: xr.Dataset) -> pd.DataFrame:
    """All time steps, all variables including derived shape."""
    shape = _derive_shape(
        ds["idtrp"].values,
        ds["gapin_opt"].values,
        ds["idtrxbot"].values,
        ds["idtzxbot"].values,
        ds["idtrxtop"].values,
        ds["idtzxtop"].values,
    )

    cols = {
        "time [s]": ds["time"].values,
        "iptipp / Ip [A]": ds["iptipp"].values,
        "bttbt / B0 [T]": ds["bttbt"].values,
        "bmtpwrtar / betan [-]": ds["bmtpwrtar"].values,
        "dstdenp [1e19 m\u207b\u00b3]": ds["dstdenp"].values,
        "idtrp / R0 [m]": ds["idtrp"].values,
        "gapin [m]": ds["gapin_opt"].values,
        "idtrxbot [m]": ds["idtrxbot"].values,
        "idtzxbot [m]": ds["idtzxbot"].values,
        "idtrxtop [m]": ds["idtrxtop"].values,
        "idtzxtop [m]": ds["idtzxtop"].values,
        "a_minor [m]": shape["a_minor"],
        "kappa [-]": shape["kappa"],
        "delta_top [-]": shape["delta_top"],
        "delta_bot [-]": shape["delta_bot"],
    }

    return pd.DataFrame(cols)


def _autofit_columns(ws):
    for col in ws.columns:
        max_len = max((len(str(cell.value or "")) for cell in col), default=0)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 32)


def trajopt_to_excel(output_dir: Path | str, out: Path | str | None = None):
    """Convert a trajectory optimization output directory to an Excel file.

    Args:
        output_dir: Path to a single trajopt output directory containing
                    optimized_trajectory.nc.
        out: Output .xlsx path. Defaults to <output_dir>/optimized_trajectory.xlsx.
    """
    traj_path = Path(output_dir) / "optimized_trajectory.nc"
    if not traj_path.exists():
        print(f"ERROR: {traj_path} not found", file=sys.stderr)
        sys.exit(1)

    ds = xr.open_dataset(traj_path)

    if out is None:
        out = Path(output_dir) / "optimized_trajectory.xlsx"

    df_wp = _build_waypoints_df(ds)
    df_full = _build_full_df(ds)

    case_name = ds.attrs.get("case", Path(output_dir).name)
    ref_shot_desc = ds.attrs.get("description", "")

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        # --- Sheet 1: Waypoints ---
        df_wp.to_excel(writer, sheet_name="Waypoints", index=False, startrow=3)
        ws = writer.sheets["Waypoints"]
        ws["A1"] = "Trajectory Optimization Output"
        ws["A2"] = case_name
        ws["A3"] = ref_shot_desc
        _autofit_columns(ws)

        # --- Sheet 2: Full waveform ---
        df_full.to_excel(writer, sheet_name="Full Waveform", index=False)
        _autofit_columns(writer.sheets["Full Waveform"])

    print(f"Saved: {out}")


if __name__ == "__main__":
    fire.Fire(trajopt_to_excel)
