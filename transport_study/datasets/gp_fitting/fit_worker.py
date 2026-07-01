"""Standalone GP profile fitting worker.

This file is shipped by itself to remote SLURM clusters (see dispatcher.py),
where it runs as `python fit_worker.py input.npz output.npz --num-workers N`.
It must therefore remain self-contained: stdlib + numpy + mkgp only, no
transport_study imports. The serial (no-cluster) path imports fit_batch() from
here so local and remote fitting share one implementation.

Batch file format (npz):
    Input:
        format_version : int
        shots          : (n_shots,) int64
        x_star         : (n_x,) target radial grid (normalized minor radius rho)
        min_points     : int, minimum valid channels per slice to attempt a fit
        scale_per_slice: bool, normalize each slice by its max before fitting
        {shot}:x       : (n_t, n_ch) radial location of each channel at each slice
        {shot}:te_y    : (n_t, n_ch) Te [keV], NaN where invalid
        {shot}:te_err  : (n_t, n_ch) Te error [keV]
        {shot}:ne_y    : (n_t, n_ch) ne [1e20 m^-3], NaN where invalid
        {shot}:ne_err  : (n_t, n_ch) ne error [1e20 m^-3]
    Output:
        format_version, shots, x_star as above
        {shot}:te_fit, {shot}:te_std, {shot}:ne_fit, {shot}:ne_std
            each (n_t, n_x), NaN where the slice was skipped or failed
"""

import argparse
import contextlib
import hashlib
import io
import multiprocessing
import os
import time
from dataclasses import dataclass
from pathlib import Path

# Limit BLAS threads before numpy loads so slice-level multiprocessing
# (fit_batch num_workers) does not oversubscribe cores.
# mkgp is single-threaded numpy/scipy
# one thread per worker is the right default.
# setdefault keeps any explicit override.
# Effective only when this module is the program entry point
# (the cluster `python fit_worker.py` path), does nothing otherwise
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from mkgp.core.baseclasses import _WarpingFunction
from mkgp.core.kernels import Gibbs_Kernel
from mkgp.core.routines import GaussianProcess

FORMAT_VERSION = 3


# ----------------------------------------------------------------------
# GP fitting (mkgp Gibbs kernel with tanh-warped length scale)
# ----------------------------------------------------------------------
# Hyperparameters, order [var, l1, l2, lw, x0]:
# amplitude
# core (small-rho) length scale,
# edge (large-rho) length scale,
# tanh transition width
# and transition center.
_HYP_START = np.array([2.0, 1.0, 0.5, 0.1, 1.0])
# Bounds define the optimizer's random-restart ranges (drawn uniform in log10)
_HYP_BOUNDS = np.array([[1.0e-2, 0.4, 0.1, 0.05, 0.95], [2.0e1, 2.0, 0.5, 0.2, 1.05]])
# Edge boundary conditions, informed by Chilenski 2016. Columns: (rho, value, error).
# Value BCs pull the profile to ~0 past the separatrix
# gradient BCs flatten it at the axis (rho=0) and past the edge
# The axis gradient uses a small positive error (mkgp needs a positive diagonal entry to stay invertible)
_VALUE_BC = np.array([[1.1, 0.0, 0.01], [1.2, 0.0, 0.01], [1.3, 0.0, 0.01], [1.4, 0.0, 0.01]])
_GRAD_BC = np.array([[0.0, 0.0, 0.01], [1.1, 0.0, 0.1], [1.2, 0.0, 0.1], [1.3, 0.0, 0.1], [1.4, 0.0, 0.1]])
# Half-width (in rho) of the x0 window used to pin the pedestal location when
# tying Te to the ne fit. Narrow enough to hold x0, wide enough to stay a valid
# (lower < upper) bound after clamping to the global x0 range.
_X0_PIN_HALFWIDTH = 1.0e-3
# Extra optimizer attempts (beyond the first) when a fit pins a hyperparameter
# at its bound - a different random restart usually escapes the same basin.
_MAX_HYP_RETRIES = 2


class Tanh_WarpingFunction(_WarpingFunction):
    """tanh length-scale warp for the mkgp Gibbs kernel.

    l(z) = 0.5 * ((l1 + l2) - (l1 - l2) * tanh((z - x0) / lw))

    mkgp ships only Constant/Linear/IG warps, so this reproduces the old gptools
    GibbsKernel1dTanh length scale. hyps = [l1, l2, lw, x0]. Analytic z- and
    hyperparameter-derivatives are provided (verified against finite differences)
    so mkgp's analytic LML-gradient optimizer path stays valid.
    """

    def __calc_warp(self, zz, der=0, hder=None):
        l1, l2, lw, x0 = self.hyperparameters
        u = (zz - x0) / lw
        tt = np.tanh(u)
        ss = 1.0 - tt * tt
        warp = np.zeros(np.shape(zz), dtype=self._dtype)
        if der == 0:
            if hder is None:
                warp = 0.5 * ((l1 + l2) - (l1 - l2) * tt)
            elif hder == 0:
                warp = 0.5 * (1.0 - tt)
            elif hder == 1:
                warp = 0.5 * (1.0 + tt)
            elif hder == 2:
                warp = 0.5 * (l1 - l2) * ss * u / lw
            elif hder == 3:
                warp = 0.5 * (l1 - l2) * ss / lw
        elif der == 1:
            if hder is None:
                warp = -0.5 * (l1 - l2) * ss / lw
            elif hder == 0:
                warp = -0.5 * ss / lw
            elif hder == 1:
                warp = 0.5 * ss / lw
            elif hder == 2:
                warp = -0.5 * (l1 - l2) * ss / (lw * lw) * (2.0 * tt * u - 1.0)
            elif hder == 3:
                warp = -0.5 * (l1 - l2) * 2.0 * tt * ss / (lw * lw)
        return warp

    def __init__(self, l1=1.0, l2=0.5, lw=0.1, x0=1.0, dtype=None):
        hyps = np.array([float(l1), float(l2), float(lw), float(x0)])
        super().__init__("Wtanh", self.__calc_warp, True, hyps, dtype=dtype)

    def __copy__(self):
        hyps = self.hyperparameters
        bnds = self.bounds
        kcopy = Tanh_WarpingFunction(hyps[0], hyps[1], hyps[2], hyps[3], dtype=self._dtype)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy


def _build_kernel(hyperparams: np.ndarray | None = None) -> Gibbs_Kernel:
    """Gibbs kernel with the tanh warp, at the start or given hyperparameters.

    Bound enforcement is turned on for both the kernel and its warp. mkgp's
    gradient-ascent optimizer never clamps to kbounds, so without this the
    hyperparameters can wander out of the physical region into the degenerate
    "all noise" fit (amplitude -> 0, edge length scale -> inf, profile pulled to
    ~0). Enforcement also lets _run_gp pin the pedestal location by narrowing the
    x0 bounds. set_kernel/__copy__ both preserve the enforce flag.
    """
    hyps = _HYP_START if hyperparams is None else np.asarray(hyperparams, dtype=float)
    kernel = Gibbs_Kernel(hyps[0], wfunc=Tanh_WarpingFunction(*hyps[1:]))
    kernel.enforce_bounds(True)
    kernel._wfunc.enforce_bounds(True)
    return kernel


def _is_pedestal_resolved(x0: float) -> bool:
    """True if a fitted pedestal location sits inside (not pushed to) the x0 bounds.

    An x0 pinned at a bound means the optimizer found no clear pedestal in range,
    so it should not be trusted to drive the other profile's location.
    """
    lo, hi = _HYP_BOUNDS[0, 4], _HYP_BOUNDS[1, 4]
    margin = 0.02 * (hi - lo)
    return lo + margin < x0 < hi - margin


def _pinned_hyperparams(hyps: np.ndarray) -> bool:
    """True if the optimizer pushed a hyperparameter to (not just near) its bound,
    in a way a differently-seeded restart could plausibly escape.

    Bound enforcement (_build_kernel) exists so a bad restart can't wander into
    the degenerate collapse mkgp is otherwise prone to (see the mkgp-bounds-not-
    enforced writeup); one of these hyperparameters still sitting at that bound
    after optimization means the search ran out of room in that basin rather
    than converging inside the physical range. _run_gp retries from a different
    restart when this happens instead of accepting the degenerate fit.

    Two edges are excluded because a different restart provably re-lands on
    the same edge, making a retry pure waste rather than a chance to escape:
    - x0 (pedestal location): its bounds - the base [0.9, 1.1] physical
      pedestal window, or the narrower pin_x0 window tying Te to ne - are
      already tight by design, not slack search room.
    - l2's ceiling (2.0): once x0 is held near the edge, the region beyond it
      often has no independent short-scale structure left to fit, so mkgp is
      happy pushing l2 as long/smooth as the box allows - not a collapse.
    Confirmed by profiling a real shot: x0 pinned in 10/10 sampled slices and
    l2's ceiling in half of them, every one re-landing on the same edge across
    all _MAX_HYP_RETRIES attempts, tripling fit time for zero change in
    outcome. var, l1, lw, and l2's floor still trigger retries: those bounds
    guard the genuine degenerate collapse (amplitude -> 0, edge scale ->
    infinity) that _build_kernel's enforcement exists to prevent, where a bad
    restart really can land somewhere better.

    The margin itself is measured in log10 space, matching how restarts are
    drawn (uniform in log10 - see _HYP_BOUNDS). var and lw span 2-3 decades,
    so a margin taken as a fraction of the raw range is huge in log terms: a
    var of 0.2755 against bounds [0.01, 20] falls inside a linear 2% margin
    (~0.4) while actually sitting at 44% of the way up the log-uniform range,
    nowhere near either wall - a converged interior optimum mislabeled as
    pinned, burning a retry that only ever re-finds the same interior point.
    """
    lo, hi = _HYP_BOUNDS[0], _HYP_BOUNDS[1]
    log_lo, log_hi, log_hyps = np.log10(lo), np.log10(hi), np.log10(hyps)
    margin = 0.02 * (log_hi - log_lo)
    pinned_lo = log_hyps <= log_lo + margin
    pinned_hi = log_hyps >= log_hi - margin
    pinned_lo[4] = pinned_hi[4] = False  # x0
    pinned_hi[2] = False  # l2 ceiling
    return bool((pinned_lo | pinned_hi).any())


def _deterministic_seed(*arrays: np.ndarray, salt: int = 0) -> int:
    """Stable RNG seed derived from the fit's own input data, not call order.

    mkgp draws its optimizer restarts from the global numpy RNG (see
    GaussianProcess.GPRFit), so without reseeding here a fit's result depends
    on whatever else already consumed random draws earlier in the process:
    slice processing order in serial mode, or multiprocessing.Pool scheduling
    and fork-inherited RNG state in parallel mode. Hashing the fit's own inputs
    makes every fit reproducible regardless of how the batch happens to be
    scheduled. `salt` distinguishes retry attempts on the same input.
    """
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    h.update(int(salt).to_bytes(8, "little", signed=True))
    return int(h.hexdigest()[:8], 16)


def _clean_inputs(data_X, data_y, err_y):
    """Drop NaN points. Returns (X, y, err) or None if nothing valid remains."""
    valid = ~np.isnan(data_y) & ~np.isnan(data_X) & ~np.isnan(err_y)
    if not valid.any():
        return None
    return data_X[valid], data_y[valid], err_y[valid]


def _remove_local_outliers(data_X, data_y, err_y, sigma_neighbor=2.0, sigma_local=3.0):
    """Drop a point that disagrees with both its immediate neighbors in x, when
    those neighbors agree with each other.

    Independent of any GP fit or hyperparameters, unlike _remove_outliers: a GP
    reference fit (however it is built - generic or self-tuned) can be flexible
    enough to bend down and absorb a single bad point along with its
    genuinely-consistent neighbors, which is exactly what let a near-zero
    misfired channel escape _remove_outliers on a C-Mod Te slice (a short core
    length scale dove down to chase it instead of the reference flagging it).
    Comparing a point only to its immediate left/right neighbors in rho catches
    an isolated single-channel spike regardless of how flexible the eventual
    fit is allowed to be. Runs before the GP-based _remove_outliers.

    A point (not the first or last, by rho) is dropped when its neighbors agree
    with each other (within sigma_neighbor combined sigma) but it disagrees
    with their average (by more than sigma_local combined sigma). A genuine
    trend - where the neighbors themselves disagree - never trips this, since
    the neighbor-agreement precondition fails first.
    """
    n = data_X.size
    if n < 3:
        return data_X, data_y, err_y
    order = np.argsort(data_X)
    x, y, e = data_X[order], data_y[order], err_y[order]

    y_left, y_right = y[:-2], y[2:]
    e_left, e_right = e[:-2], e[2:]
    y_mid, e_mid = y[1:-1], e[1:-1]

    neighbors_agree = np.abs(y_left - y_right) <= sigma_neighbor * np.sqrt(e_left**2 + e_right**2)
    neighbor_mean = 0.5 * (y_left + y_right)
    neighbor_mean_err = 0.5 * np.sqrt(e_left**2 + e_right**2)
    point_disagrees = np.abs(y_mid - neighbor_mean) > sigma_local * np.sqrt(e_mid**2 + neighbor_mean_err**2)

    drop = np.zeros(n, dtype=bool)
    drop[1:-1] = neighbors_agree & point_disagrees
    if not drop.any():
        return data_X, data_y, err_y
    keep = ~drop
    return x[keep], y[keep], e[keep]


def _run_gp(data_X, data_y, err_y, x_eval, hyperparams=None, optimize=True, pin_x0=None):
    """Set up the GP with edge BCs and fit. Returns the GaussianProcess or None.

    With optimize=True and hyperparams=None the hyperparameters are tuned (8 random
    restarts, mkgp's native LML maximization). Otherwise the GP predicts at the
    given (or default start) hyperparameters with no optimization.

    pin_x0 narrows the x0 (pedestal location) bounds to a tight window around the
    given value, so bound enforcement holds the pedestal there (used to tie the
    Te pedestal location to the ne fit).

    The restarts are seeded from the fit's own input data (_deterministic_seed),
    so the result only depends on (data_X, data_y, err_y), never on multiprocessing
    scheduling or slice processing order. When optimizing, a fit that pins a
    hyperparameter at its bound (_pinned_hyperparams) is retried from a fresh,
    differently-seeded restart set up to _MAX_HYP_RETRIES times; the attempt with
    the best log marginal likelihood is kept even if every attempt stays pinned
    (a genuinely unresolvable slice should still return its least-bad fit).
    """
    kbounds = _HYP_BOUNDS
    if pin_x0 is not None:
        kbounds = _HYP_BOUNDS.astype(float).copy()
        kbounds[0, 4] = max(_HYP_BOUNDS[0, 4], pin_x0 - _X0_PIN_HALFWIDTH)
        kbounds[1, 4] = min(_HYP_BOUNDS[1, 4], pin_x0 + _X0_PIN_HALFWIDTH)
    xdata = np.concatenate([data_X, _VALUE_BC[:, 0]])
    ydata = np.concatenate([data_y, _VALUE_BC[:, 1]])
    yerr = np.concatenate([err_y, _VALUE_BC[:, 2]])

    do_optimize = optimize and hyperparams is None
    n_attempts = (1 + _MAX_HYP_RETRIES) if do_optimize else 1

    best_gp, best_lml = None, -np.inf
    for attempt in range(n_attempts):
        gp = GaussianProcess()
        gp.set_kernel(kernel=_build_kernel(hyperparams), kbounds=kbounds, regpar=1.0)
        gp.set_raw_data(
            xdata=xdata,
            ydata=ydata,
            yerr=yerr,
            dxdata=_GRAD_BC[:, 0],
            dydata=_GRAD_BC[:, 1],
            dyerr=_GRAD_BC[:, 2],
        )
        gp.set_search_parameters(epsilon=1.0e-2)
        if do_optimize:
            nrestarts = 8
            np.random.seed(_deterministic_seed(data_X, data_y, err_y, salt=attempt))
        else:
            # predict-only at fixed hyperparameters. The public maxiter clamps
            # to >=50, so poke _imax=0 to skip the gradient-ascent loop entirely.
            gp._imax = 0
            nrestarts = 0
        try:
            # mkgp prints optimizer status to stdout; keep worker logs clean.
            with contextlib.redirect_stdout(io.StringIO()):
                gp.GPRFit(np.asarray(x_eval, dtype=float), hsgp_flag=False, nrestarts=nrestarts)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue

        if not do_optimize:
            return gp

        hyps = np.asarray(gp.get_gp_kernel_details()[1], dtype=float)
        lml = gp.get_gp_lml()
        if lml is not None and lml > best_lml:
            best_gp, best_lml = gp, lml
        if not _pinned_hyperparams(hyps):
            return gp  # converged inside the physical range, no need to retry

    return best_gp


def _remove_outliers(data_X, data_y, err_y, sigma=3.0, max_drop_frac=0.3, ref_hyperparams=None):
    """Drop points lying > sigma combined-sigma from a reference GP fit.

    Mirrors the old gptools remove_outliers ordering: detect outliers with a
    reference fit, then the caller optimizes on the clean data.

    ref_hyperparams, when given, fixes the reference fit at those (already
    optimized) hyperparameters instead of the generic, un-tuned _HYP_START.
    gp_profile's two-pass call supplies its own rough-optimized hyperparameters
    here: a single generic reference can be too smooth for a genuinely steep or
    locally noisy slice, and once it is wrong in one spot it can flag several
    real points as outliers together. Seen on a C-Mod Te slice: a near-zero
    misfired channel sandwiched between two ~4 keV core points dragged a
    generic reference down enough that both real points looked like outliers
    too, and all three got dropped, leaving the core with no supporting data
    at all. A reference built from the slice's own shape does not have that
    failure mode. sigma=3.0 (rather than the stricter 2.0) further tolerates
    reference/data mismatch on genuinely steep slices.

    Never drops more than max_drop_frac of the points: a reference fit that flags
    a large fraction as outliers is itself unreliable (a genuinely bad slice, or
    a real pedestal the reference fit cannot follow), so in that case all points
    are kept rather than gutting the profile - which was producing completely
    wrong fits when most points got dropped.
    """
    n = data_X.size
    if n < 3:
        return data_X, data_y, err_y
    gp = _run_gp(data_X, data_y, err_y, data_X, hyperparams=ref_hyperparams, optimize=False)
    if gp is None:
        return data_X, data_y, err_y
    mean = gp.get_gp_mean()
    std = gp.get_gp_std(noise_flag=False)
    keep = np.abs(data_y - mean) <= sigma * np.sqrt(std**2 + err_y**2)
    n_keep = int(keep.sum())
    if n_keep == n or n_keep < 3 or (n - n_keep) > max(1, round(max_drop_frac * n)):
        return data_X, data_y, err_y
    return data_X[keep], data_y[keep], err_y[keep]


def _rough_hyperparameters(data_X, data_y, err_y) -> np.ndarray | None:
    """One optimize pass on (possibly outlier-contaminated) data.

    Used only to get a locally-representative reference for outlier removal -
    see _remove_outliers's ref_hyperparams. Not returned to callers as a real
    fit result.
    """
    gp = _run_gp(data_X, data_y, err_y, data_X, optimize=True)
    if gp is None:
        return None
    return np.asarray(gp.get_gp_kernel_details()[1], dtype=float)


def fit_gp_hyperparameters(
    data_X: np.ndarray,
    data_y: np.ndarray,
    err_y: np.ndarray,
) -> np.ndarray | None:
    """Fit GP hyperparameters once and return them for later reuse.

    Returns
    -------
    np.ndarray | None
        Optimized hyperparameters [var, l1, l2, lw, x0], or None if no valid
        input data remains after NaN filtering.
    """
    cleaned = _clean_inputs(data_X, data_y, err_y)
    if cleaned is None:
        return None
    data_X, data_y, err_y = cleaned
    data_X, data_y, err_y = _remove_local_outliers(data_X, data_y, err_y)
    rough_hyps = _rough_hyperparameters(data_X, data_y, err_y)
    data_X, data_y, err_y = _remove_outliers(data_X, data_y, err_y, ref_hyperparams=rough_hyps)
    gp = _run_gp(data_X, data_y, err_y, np.asarray(data_X, dtype=float), optimize=True)
    if gp is None:
        return None
    return np.asarray(gp.get_gp_kernel_details()[1], dtype=float)


def gp_profile(
    data_X: np.ndarray,
    data_y: np.ndarray,
    err_y: np.ndarray,
    X_star: np.ndarray,
    calc_gradient: bool = False,
    hyperparams: np.ndarray | None = None,
    optimize_hyperparams: bool = True,
    pin_x0: float | None = None,
):
    """Fit one profile and predict on X_star.

    Returns (mean, std, grad_mean, grad_std, hyperparams), with the gradient
    entries None unless calc_gradient, and all entries None if no valid data
    remains. The fitted hyperparameters are returned so callers can read the
    pedestal location (x0). pin_x0 holds the pedestal at a given location.

    When optimizing with no fixed hyperparams, outlier removal runs in two
    passes: a cheap neighbor-agreement check first drops isolated single-point
    spikes independent of any fit (_remove_local_outliers), then a rough
    optimize on the survivors supplies hyperparameters that already capture
    the slice's own shape, and remaining outliers are judged against that fit
    rather than a generic un-tuned reference (see _remove_outliers) before the
    real optimize on the cleaned data.
    """
    cleaned = _clean_inputs(data_X, data_y, err_y)
    if cleaned is None:
        return None, None, None, None, None
    data_X, data_y, err_y = cleaned

    if hyperparams is None and optimize_hyperparams:
        data_X, data_y, err_y = _remove_local_outliers(data_X, data_y, err_y)
        rough_hyps = _rough_hyperparameters(data_X, data_y, err_y)
        data_X, data_y, err_y = _remove_outliers(data_X, data_y, err_y, ref_hyperparams=rough_hyps)

    gp = _run_gp(
        data_X,
        data_y,
        err_y,
        X_star,
        hyperparams=hyperparams,
        optimize=optimize_hyperparams,
        pin_x0=pin_x0,
    )
    if gp is None:
        return None, None, None, None, None

    y_star = gp.get_gp_mean()
    std_y_star = gp.get_gp_std(noise_flag=False)
    hyps_out = np.asarray(gp.get_gp_kernel_details()[1], dtype=float)
    if not calc_gradient:
        return y_star, std_y_star, None, None, hyps_out
    return y_star, std_y_star, gp.get_gp_drv_mean(), gp.get_gp_drv_std(noise_flag=False), hyps_out


# ----------------------------------------------------------------------
# Batch containers and (de)serialization
# ----------------------------------------------------------------------
@dataclass
class ShotFitInput:
    """Raw Thomson channel data for one shot, ready for GP fitting.

    All arrays are (n_t, n_ch). x is the radial coordinate of each channel
    (normalized minor radius rho), shared between te and ne since both come
    from the same channels. Invalid points are NaN.
    """

    x: np.ndarray
    te_y: np.ndarray
    te_err: np.ndarray
    ne_y: np.ndarray
    ne_err: np.ndarray

    def has_fittable_points(self) -> bool:
        """True if te and ne each have at least one finite (x, y, err) point.

        A shot failing this can only come back all NaN from the fit (and would
        then be culled at assembly), so callers should skip it before staging
        or batching. Both variables are required because downstream assembly
        culls the shot if either profile is all NaN.
        """
        x_ok = np.isfinite(self.x)
        te_ok = x_ok & np.isfinite(self.te_y) & np.isfinite(self.te_err)
        ne_ok = x_ok & np.isfinite(self.ne_y) & np.isfinite(self.ne_err)
        return bool(te_ok.any() and ne_ok.any())


@dataclass
class ShotFitOutput:
    """GP-fitted profiles for one shot.

    te_fit/te_std/ne_fit/ne_std are (n_t, n_x). te_hyps/ne_hyps are (n_t, 5),
    columns [var, l1, l2, lw, x0], NaN where a slice was skipped, failed, or
    fit at fixed (non-optimized) hyperparameters.
    """

    te_fit: np.ndarray
    te_std: np.ndarray
    ne_fit: np.ndarray
    ne_std: np.ndarray
    te_hyps: np.ndarray
    ne_hyps: np.ndarray


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
        arrays[f"{shot}:x"] = np.asarray(si.x, dtype=np.float32)
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
                x=data[f"{shot}:x"],
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
        arrays[f"{shot}:te_hyps"] = np.asarray(so.te_hyps, dtype=np.float32)
        arrays[f"{shot}:ne_hyps"] = np.asarray(so.ne_hyps, dtype=np.float32)
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
                te_hyps=data[f"{shot}:te_hyps"],
                ne_hyps=data[f"{shot}:ne_hyps"],
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
def _fit_variable(x, y, err, x_star, min_points, scale_per_slice, optimize, hyperparams, pin_x0):
    """Fit one variable of one time slice.

    Returns (y_out, std_out, hyps) or (None, None, None). hyps is the fitted
    [var, l1, l2, lw, x0] array (x0 is used to tie Te to ne); None when the
    slice is skipped or fails.
    """
    # Force float32 here, the same precision the cluster path is stuck at after
    # its pack_fit_batch npz roundtrip. Without this, in-process fits (which
    # otherwise keep whatever dtype the source data arrived in, often float64)
    # hash different bytes into _deterministic_seed than the cluster does for
    # the same physical shot, landing on a different optimizer restart.
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    err = np.asarray(err, dtype=np.float32)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(err)
    if int(valid.sum()) < min_points:
        return None, None, None

    scale = 1.0
    if scale_per_slice:
        # Normalize to O(1) before GP fit to prevent amplitude collapse
        # when channels don't cover the full radial range.
        with np.errstate(all="ignore"):
            scale = float(np.nanmax(y))
        if not np.isfinite(scale) or scale < 1e-6:
            return None, None, None

    y_star, std_y_star, _, _, hyps = gp_profile(
        data_X=np.asarray(x, dtype=float),
        data_y=np.asarray(y, dtype=float) / scale,
        err_y=np.asarray(err, dtype=float) / scale,
        X_star=x_star,
        calc_gradient=False,
        hyperparams=hyperparams,
        optimize_hyperparams=optimize,
        pin_x0=pin_x0,
    )
    if y_star is None:
        return None, None, None

    # Last resort: the GP mean can ring below zero between the outermost
    # measurement and the edge boundary conditions, so clamp to non-negative
    y_out = np.clip(np.asarray(y_star, dtype=float).ravel() * scale, 0.0, None)
    std_out = np.asarray(std_y_star, dtype=float).ravel() * scale
    return y_out, std_out, hyps


def _fit_slice(task: tuple):
    """Fit Te and ne for one (shot, time slice).

    ne is fit first; when optimizing, its pedestal location pins Te's so both
    profiles place the high-gradient region at the same rho. (Density is the
    cleaner pedestal indicator in C-Mod H-mode.) If the ne pedestal is not
    clearly resolved, Te is fit freely. The fixed-hyperparameter path leaves
    both profiles independent.

    Returns (shot, i_time, te_y, te_std, ne_y, ne_std, te_hyps, ne_hyps); any
    value is None when that variable's slice was skipped or failed.
    """
    (shot, i_time), x, te_y, te_err, ne_y, ne_err, x_star, min_points, scale_per_slice, optimize, hyperparams = task
    tie_x0 = optimize and hyperparams is None

    ne_y_out, ne_std_out, ne_hyps = _fit_variable(x, ne_y, ne_err, x_star, min_points, scale_per_slice, optimize, hyperparams, None)
    ne_x0 = None if ne_hyps is None else float(ne_hyps[4])
    pin_x0 = ne_x0 if (tie_x0 and ne_x0 is not None and _is_pedestal_resolved(ne_x0)) else None
    te_y_out, te_std_out, te_hyps = _fit_variable(x, te_y, te_err, x_star, min_points, scale_per_slice, optimize, hyperparams, pin_x0)

    return shot, i_time, te_y_out, te_std_out, ne_y_out, ne_std_out, te_hyps, ne_hyps


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
    """GP fit every (shot, time slice) in the batch.

    Hyperparameters are optimized for each individual profile, since plasma
    conditions (and thus profile shapes) change over the course of a shot. Te and
    ne of a slice are fit together so they can share one pedestal location (see
    _fit_slice).

    Slices are fit serially (num_workers == 1) or across worker processes
    (num_workers > 1). mkgp fits are single-threaded, so parallelism comes only
    from the slice-level pool. BLAS threads are pinned to 1 (see module top) to
    avoid oversubscription.

    Parameters
    ----------
    max_slices_per_shot : int | None
        If set, only fit the first N time slices of each shot (debug aid).
    """
    x_star = np.asarray(x_star, dtype=float)

    tasks = []
    for shot, si in shot_inputs.items():
        n_t = si.te_y.shape[0]
        if max_slices_per_shot is not None:
            n_t = min(n_t, max_slices_per_shot)
        tasks.extend(
            (
                (shot, i_time),
                si.x[i_time, :],
                si.te_y[i_time, :],
                si.te_err[i_time, :],
                si.ne_y[i_time, :],
                si.ne_err[i_time, :],
                x_star,
                min_points,
                scale_per_slice,
                optimize_hyperparams,
                hyperparams,
            )
            for i_time in range(n_t)
        )

    n_x = len(x_star)
    outputs = {
        shot: ShotFitOutput(
            te_fit=np.full((si.te_y.shape[0], n_x), np.nan),
            te_std=np.full((si.te_y.shape[0], n_x), np.nan),
            ne_fit=np.full((si.ne_y.shape[0], n_x), np.nan),
            ne_std=np.full((si.ne_y.shape[0], n_x), np.nan),
            te_hyps=np.full((si.te_y.shape[0], 5), np.nan),
            ne_hyps=np.full((si.ne_y.shape[0], 5), np.nan),
        )
        for shot, si in shot_inputs.items()
    }

    def _store(result):
        shot, i_time, te_y_out, te_std_out, ne_y_out, ne_std_out, te_hyps, ne_hyps = result
        so = outputs[shot]
        if te_y_out is not None:
            so.te_fit[i_time, :] = te_y_out
            so.te_std[i_time, :] = te_std_out
            if te_hyps is not None:
                so.te_hyps[i_time, :] = te_hyps
        if ne_y_out is not None:
            so.ne_fit[i_time, :] = ne_y_out
            so.ne_std[i_time, :] = ne_std_out
            if ne_hyps is not None:
                so.ne_hyps[i_time, :] = ne_hyps

    n_total = len(tasks)
    n_done = 0
    t_start = time.monotonic()
    if num_workers <= 1:
        for task in tasks:
            _store(_fit_slice(task))
            n_done += 1
            if n_done % 10 == 0:
                print(f"[fit_worker] {n_done}/{n_total} slices (Te+ne) done", flush=True)
    else:
        with multiprocessing.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(_fit_slice, tasks, chunksize=1):
                _store(result)
                n_done += 1
                if n_done % 50 == 0:
                    print(f"[fit_worker] {n_done}/{n_total} slices (Te+ne) done", flush=True)

    elapsed = time.monotonic() - t_start
    print(
        f"[fit_worker] finished {n_total} slices (Te+ne) for {len(shot_inputs)} shots "
        f"in {elapsed:.0f}s ({elapsed / max(n_total, 1):.2f}s per slice)",
        flush=True,
    )
    return outputs


# ----------------------------------------------------------------------
# CLI entry point (used on the cluster)
# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    # Typically I prefer using fire but to minimize deps we're using argparse here
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
