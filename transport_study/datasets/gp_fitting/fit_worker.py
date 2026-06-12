"""Standalone GP profile fitting worker.

This file is shipped by itself to remote SLURM clusters (see dispatcher.py),
where it runs as `python fit_worker.py input.npz output.npz --num-workers N`.
It must therefore remain self-contained: stdlib + numpy + gptools only, no
transport_study imports. The serial (no-cluster) path imports fit_batch() from
here so local and remote fitting share one implementation.

Batch file format (npz):
    Input:
        format_version : int
        shots          : (n_shots,) int64
        x_star         : (n_psi,) target psi_n grid
        min_points     : int, minimum valid channels per slice to attempt a fit
        scale_per_slice: bool, normalize each slice by its max before fitting
        {shot}:psi     : (n_t, n_ch) psi_n of each channel at each slice
        {shot}:te_y    : (n_t, n_ch) Te [keV], NaN where invalid
        {shot}:te_err  : (n_t, n_ch) Te error [keV]
        {shot}:ne_y    : (n_t, n_ch) ne [1e20 m^-3], NaN where invalid
        {shot}:ne_err  : (n_t, n_ch) ne error [1e20 m^-3]
    Output:
        format_version, shots, x_star as above
        {shot}:te_fit, {shot}:te_std, {shot}:ne_fit, {shot}:ne_std
            each (n_t, n_psi), NaN where the slice was skipped or failed
"""

import argparse
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path

import gptools
import numpy as np

FORMAT_VERSION = 1
VARIABLES = ("te", "ne")


# ----------------------------------------------------------------------
# GP fitting
# ----------------------------------------------------------------------
def _build_gp() -> gptools.GaussianProcess:
    """Construct a GP instance with the standard prior/kernel settings."""
    hp = gptools.UniformJointPrior([[0.0, 20.0]]) * gptools.GammaJointPriorAlt([1.0, 0.5, 0.0, 1.0], [0.3, 0.25, 0.1, 0.1])
    k_gibbs = gptools.GibbsKernel1dTanh(hyperprior=hp)
    return gptools.GaussianProcess(k_gibbs)


def _add_data_and_bcs(gp, data_X, data_y, err_y) -> bool:
    """Add measurements and edge boundary conditions to the GP.

    Returns False if no valid input data remains after NaN filtering.
    """
    valid_mask = ~np.isnan(data_y) & ~np.isnan(data_X) & ~np.isnan(err_y)
    if np.sum(valid_mask) == 0:
        return False
    data_X = data_X[valid_mask]
    data_y = data_y[valid_mask]
    err_y = err_y[valid_mask]

    gp.add_data(data_X, data_y, err_y)
    gp.remove_outliers(sigma=2)

    # Boundary conditions, informed by Chilenski 2016
    val_bc = np.array([[1.1, 0, 0.01], [1.2, 0, 0.01], [1.3, 0, 0.01], [1.4, 0, 0.01]])
    grad_bc = np.array([[0, 0, 0], [1.1, 0, 0.1], [1.2, 0, 0.1], [1.3, 0, 0.1], [1.4, 0, 0.1]])
    gp.add_data(val_bc[:, 0], val_bc[:, 1], err_y=val_bc[:, 2], n=0)
    gp.add_data(grad_bc[:, 0], grad_bc[:, 1], err_y=grad_bc[:, 2], n=1)
    return True


def fit_gp_hyperparameters(
    data_X: np.ndarray,
    data_y: np.ndarray,
    err_y: np.ndarray,
    num_proc: int = 4,
) -> np.ndarray | None:
    """Fit GP hyperparameters once and return them for later reuse.

    Returns
    -------
    np.ndarray | None
        Optimized free hyperparameters, or None if no valid input data remains
        after NaN filtering.
    """
    gp = _build_gp()
    if not _add_data_and_bcs(gp, data_X, data_y, err_y):
        return None
    gp.optimize_hyperparameters(verbose=False, random_starts=8, max_tries=4, num_proc=num_proc)
    return np.asarray(gp.free_params, dtype=float)


def gp_profile(
    data_X: np.ndarray,
    data_y: np.ndarray,
    err_y: np.ndarray,
    X_star: np.ndarray,
    calc_gradient: bool = False,
    hyperparams: np.ndarray | None = None,
    optimize_hyperparams: bool = True,
    num_proc: int = 4,
):
    gp = _build_gp()
    if not _add_data_and_bcs(gp, data_X, data_y, err_y):
        return None, None, None, None

    if hyperparams is not None:
        gp.update_hyperparameters(np.asarray(hyperparams, dtype=float))
    elif optimize_hyperparams:
        gp.optimize_hyperparameters(verbose=False, random_starts=8, max_tries=4, num_proc=num_proc)

    y_star, std_y_star = gp.predict(X_star)

    if not calc_gradient:
        return y_star, std_y_star, None, None
    else:
        grad_y_star, std_grad_y_star = gp.predict(X_star, n=1)
        return y_star, std_y_star, grad_y_star, std_grad_y_star


# ----------------------------------------------------------------------
# Batch containers and (de)serialization
# ----------------------------------------------------------------------
@dataclass
class ShotFitInput:
    """Raw Thomson channel data for one shot, ready for GP fitting.

    All arrays are (n_t, n_ch). psi is shared between te and ne since both
    come from the same channels. Invalid points are NaN.
    """

    psi: np.ndarray
    te_y: np.ndarray
    te_err: np.ndarray
    ne_y: np.ndarray
    ne_err: np.ndarray


@dataclass
class ShotFitOutput:
    """GP-fitted profiles for one shot, each (n_t, n_psi)."""

    te_fit: np.ndarray
    te_std: np.ndarray
    ne_fit: np.ndarray
    ne_std: np.ndarray


def pack_fit_batch(
    path: Path | str,
    shot_inputs: dict[int, ShotFitInput],
    x_star: np.ndarray,
    min_points: int,
    scale_per_slice: bool,
) -> None:
    """Write a batch of shot fit inputs to a single npz file."""
    arrays = {
        "format_version": np.int64(FORMAT_VERSION),
        "shots": np.array(sorted(shot_inputs), dtype=np.int64),
        "x_star": np.asarray(x_star, dtype=np.float64),
        "min_points": np.int64(min_points),
        "scale_per_slice": np.bool_(scale_per_slice),
    }
    for shot, si in shot_inputs.items():
        arrays[f"{shot}:psi"] = np.asarray(si.psi, dtype=np.float32)
        arrays[f"{shot}:te_y"] = np.asarray(si.te_y, dtype=np.float32)
        arrays[f"{shot}:te_err"] = np.asarray(si.te_err, dtype=np.float32)
        arrays[f"{shot}:ne_y"] = np.asarray(si.ne_y, dtype=np.float32)
        arrays[f"{shot}:ne_err"] = np.asarray(si.ne_err, dtype=np.float32)
    _atomic_savez(path, arrays)


def unpack_fit_batch(
    path: Path | str,
) -> tuple[dict[int, ShotFitInput], np.ndarray, int, bool]:
    """Read a batch input npz. Returns (shot_inputs, x_star, min_points, scale_per_slice)."""
    with np.load(path) as data:
        version = int(data["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(f"Batch file {path} has format version {version}, expected {FORMAT_VERSION}")
        shots = data["shots"].tolist()
        x_star = data["x_star"]
        min_points = int(data["min_points"])
        scale_per_slice = bool(data["scale_per_slice"])
        shot_inputs = {
            shot: ShotFitInput(
                psi=data[f"{shot}:psi"],
                te_y=data[f"{shot}:te_y"],
                te_err=data[f"{shot}:te_err"],
                ne_y=data[f"{shot}:ne_y"],
                ne_err=data[f"{shot}:ne_err"],
            )
            for shot in shots
        }
    return shot_inputs, x_star, min_points, scale_per_slice


def read_batch_shots(path: Path | str) -> list[int]:
    """Read only the shot list from a batch input npz (cheap)."""
    with np.load(path) as data:
        return data["shots"].tolist()


def pack_fit_results(
    path: Path | str,
    outputs: dict[int, ShotFitOutput],
    x_star: np.ndarray,
) -> None:
    """Write fitted profiles to a single npz file (atomically)."""
    arrays = {
        "format_version": np.int64(FORMAT_VERSION),
        "shots": np.array(sorted(outputs), dtype=np.int64),
        "x_star": np.asarray(x_star, dtype=np.float64),
    }
    for shot, so in outputs.items():
        arrays[f"{shot}:te_fit"] = np.asarray(so.te_fit, dtype=np.float32)
        arrays[f"{shot}:te_std"] = np.asarray(so.te_std, dtype=np.float32)
        arrays[f"{shot}:ne_fit"] = np.asarray(so.ne_fit, dtype=np.float32)
        arrays[f"{shot}:ne_std"] = np.asarray(so.ne_std, dtype=np.float32)
    _atomic_savez(path, arrays)


def unpack_fit_results(path: Path | str) -> dict[int, ShotFitOutput]:
    """Read a batch output npz into per-shot fit outputs."""
    with np.load(path) as data:
        version = int(data["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(f"Result file {path} has format version {version}, expected {FORMAT_VERSION}")
        return {
            shot: ShotFitOutput(
                te_fit=data[f"{shot}:te_fit"],
                te_std=data[f"{shot}:te_std"],
                ne_fit=data[f"{shot}:ne_fit"],
                ne_std=data[f"{shot}:ne_std"],
            )
            for shot in data["shots"].tolist()
        }


def _atomic_savez(path: Path | str, arrays: dict) -> None:
    """Write npz to a temp file then rename, so readers never see partial files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp_path, path)


# ----------------------------------------------------------------------
# Batch fitting
# ----------------------------------------------------------------------
def _fit_slice(task: tuple) -> tuple[tuple[int, str, int], np.ndarray | None, np.ndarray | None]:
    """Fit a single (shot, variable, time slice). Returns (key, y, std) or (key, None, None)."""
    key, psi, y, err, x_star, min_points, scale_per_slice, gp_num_proc, optimize, hyperparams = task

    valid = np.isfinite(psi) & np.isfinite(y) & np.isfinite(err)
    if int(valid.sum()) < min_points:
        return key, None, None

    scale = 1.0
    if scale_per_slice:
        # Normalize to O(1) before GP fit to prevent amplitude collapse
        # when channels don't cover the full psi_n range.
        with np.errstate(all="ignore"):
            scale = float(np.nanmax(y))
        if not np.isfinite(scale) or scale < 1e-6:
            return key, None, None

    y_star, std_y_star, _, _ = gp_profile(
        data_X=np.asarray(psi, dtype=float),
        data_y=np.asarray(y, dtype=float) / scale,
        err_y=np.asarray(err, dtype=float) / scale,
        X_star=x_star,
        calc_gradient=False,
        hyperparams=hyperparams,
        optimize_hyperparams=optimize,
        num_proc=gp_num_proc,
    )
    if y_star is None:
        return key, None, None

    # Last resort: the GP mean can ring below zero between the outermost
    # measurement and the edge boundary conditions, so clamp to non-negative
    y_out = np.clip(np.asarray(y_star, dtype=float).ravel() * scale, 0.0, None)
    std_out = np.asarray(std_y_star, dtype=float).ravel() * scale
    return key, y_out, std_out


def fit_batch(
    shot_inputs: dict[int, ShotFitInput],
    x_star: np.ndarray,
    min_points: int = 1,
    scale_per_slice: bool = False,
    num_workers: int = 1,
    optimize_hyperparams: bool = True,
    hyperparams: np.ndarray | None = None,
    max_slices_per_shot: int | None = None,
) -> dict[int, ShotFitOutput]:
    """GP fit every (shot, variable, time slice) in the batch.

    Hyperparameters are optimized for each individual profile, since plasma
    conditions (and thus profile shapes) change over the course of a shot.

    With num_workers == 1, slices are fit serially and the GP hyperparameter
    optimizer uses its own internal parallelism (num_proc=4, the historical
    serial behavior). With num_workers > 1, slices are fit in parallel
    processes and the internal optimizer parallelism is disabled to avoid
    oversubscription.

    Parameters
    ----------
    max_slices_per_shot : int | None
        If set, only fit the first N time slices of each shot (debug aid).
    """
    x_star = np.asarray(x_star, dtype=float)
    gp_num_proc = 4 if num_workers == 1 else 1

    tasks = []
    for shot, si in shot_inputs.items():
        for var in VARIABLES:
            y_all = getattr(si, f"{var}_y")
            err_all = getattr(si, f"{var}_err")
            n_t = y_all.shape[0]
            if max_slices_per_shot is not None:
                n_t = min(n_t, max_slices_per_shot)
            tasks.extend(
                (
                    (shot, var, i_time),
                    si.psi[i_time, :],
                    y_all[i_time, :],
                    err_all[i_time, :],
                    x_star,
                    min_points,
                    scale_per_slice,
                    gp_num_proc,
                    optimize_hyperparams,
                    hyperparams,
                )
                for i_time in range(n_t)
            )

    n_psi = len(x_star)
    outputs = {
        shot: ShotFitOutput(
            te_fit=np.full((si.te_y.shape[0], n_psi), np.nan),
            te_std=np.full((si.te_y.shape[0], n_psi), np.nan),
            ne_fit=np.full((si.ne_y.shape[0], n_psi), np.nan),
            ne_std=np.full((si.ne_y.shape[0], n_psi), np.nan),
        )
        for shot, si in shot_inputs.items()
    }

    def _store(result):
        (shot, var, i_time), y_out, std_out = result
        if y_out is None:
            return
        so = outputs[shot]
        getattr(so, f"{var}_fit")[i_time, :] = y_out
        getattr(so, f"{var}_std")[i_time, :] = std_out

    n_total = len(tasks)
    n_done = 0
    if num_workers <= 1:
        for task in tasks:
            _store(_fit_slice(task))
            n_done += 1
            if n_done % 10 == 0:
                print(f"[fit_worker] {n_done}/{n_total} slice fits done", flush=True)
    else:
        with multiprocessing.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(_fit_slice, tasks, chunksize=1):
                _store(result)
                n_done += 1
                if n_done % 50 == 0:
                    print(f"[fit_worker] {n_done}/{n_total} slice fits done", flush=True)

    print(f"[fit_worker] finished {n_total} slice fits for {len(shot_inputs)} shots", flush=True)
    return outputs


# ----------------------------------------------------------------------
# CLI entry point (used on the cluster)
# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GP profile fitting worker")
    parser.add_argument("input", help="Path to batch input npz")
    parser.add_argument("output", help="Path to write batch output npz")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Slice-level worker processes. Defaults to SLURM_CPUS_PER_TASK or cpu count.",
    )
    args = parser.parse_args(argv)

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    shot_inputs, x_star, min_points, scale_per_slice = unpack_fit_batch(args.input)
    print(
        f"[fit_worker] fitting {len(shot_inputs)} shots with {num_workers} workers "
        f"(min_points={min_points}, scale_per_slice={scale_per_slice})",
        flush=True,
    )
    outputs = fit_batch(
        shot_inputs,
        x_star,
        min_points=min_points,
        scale_per_slice=scale_per_slice,
        num_workers=num_workers,
    )
    pack_fit_results(args.output, outputs, x_star)
    print(f"[fit_worker] wrote results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
