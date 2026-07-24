"""Tests for the GP fitting batch format, worker, and dispatcher logic.

The real cluster interaction (srunx submission, rsync) needs a cluster, so the
dispatcher loop is exercised against a fake backend. Everything else here runs
exactly the code the cluster path uses: the npz round-trip, deterministic batch
planning for restart safety, and the fitting math the worker executes.
"""

import multiprocessing
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from transport_study import PACKAGE_ROOT
from transport_study.datasets.gp_fitting.dispatcher import (
    BatchState,
    ClusterFitConfig,
    ClusterFitDispatcher,
    PartitionSpec,
    batch_id,
    parse_partition_specs,
    plan_batches,
)
from transport_study.datasets.gp_fitting.fit_worker import (
    ShotFitInput,
    ShotFitOutput,
    data_envelope,
    fit_batch,
    fit_variable,
    gp_profile,
    main,
    nonphysical_peak,
    pack_fit_batch,
    pack_fit_results,
    read_batch_shots,
    unpack_fit_batch,
    unpack_fit_results,
)

N_T = 3
N_CH = 15
X_STAR = np.linspace(0, 1.05, 30)

# Plausible free hyperparameters for GibbsKernel1dTanh (sigma_f, l1, l2, lw, x0),
# used to make fit tests fast by skipping hyperparameter optimization
FIXED_HYPERPARAMS = np.array([2.0, 0.8, 0.3, 0.1, 0.95])


def _synthetic_input(seed: int = 0, amplitude: float = 2.0) -> ShotFitInput:
    """Pedestal-like tanh profiles measured at N_CH channels for N_T slices."""
    rng = np.random.default_rng(seed)
    rho = np.tile(np.linspace(0.0, 1.05, N_CH), (N_T, 1))
    profile = amplitude * (1 - np.tanh((rho - 0.95) / 0.08)) / 2 + 0.05
    arrays = {}
    for var in ("te", "ne"):
        y = profile + rng.normal(0, 0.02, size=profile.shape)
        err = np.maximum(0.1 * np.abs(y), 0.05)
        arrays[f"{var}_y"] = y
        arrays[f"{var}_err"] = err
    return ShotFitInput(x=rho, **arrays)


def _tracking_data(rho_ch, fit, rho_fit, rel_err=0.1):
    """Scatter that sits exactly on the fitted profile with rel_err error bars."""
    y = np.interp(rho_ch, rho_fit, fit)
    return rho_ch, y, np.maximum(rel_err * np.abs(y), 0.02)


def test_data_envelope_includes_nearest_neighbors():
    # A gap between channels at 0.72 and 0.90: envelope at rho 0.84 must still include the nearest neighbors (the zero-margin rule culled these).
    # see the high inner neighbor so pedestal interpolation never reads as
    # overshoot.
    x = np.array([0.5, 0.72, 0.90, 1.0])
    y = np.array([3.5, 3.0, 0.2, 0.05])
    e = np.array([0.1, 0.1, 0.05, 0.02])
    assert data_envelope(x, y, e, 0.84) >= 3.0

    x_nan = np.full(3, np.nan)
    y_nan = np.full(3, np.nan)
    e_nan = np.full(3, np.nan)

    assert np.isnan(data_envelope(x_nan, y_nan, e_nan, 0.9))


def test_nonphysical_peak():
    rho = np.linspace(0.0, 1.1, 56)
    rho_ch = np.linspace(0.02, 1.05, 20)

    # Monotonic core-peaked profile tracking its data: healthy.
    mono = np.clip(9.0 * (1 - rho / 1.15), 0, None)
    x, y, e = _tracking_data(rho_ch, mono, rho)
    assert nonphysical_peak(mono, rho, x, y, e) is None

    # Flat profile whose global max lands in the edge region by noise
    # the margin keeps it
    flat = np.full_like(rho, 2.7)
    flat[(rho > 0.9) & (rho < 0.96)] = 2.75
    x, y, e = _tracking_data(rho_ch, flat, rho)
    assert nonphysical_peak(flat, rho, x, y, e) is None

    # Data-supported hollow profile (ramp-up ne): off-axis bump 1.5x the core
    # but the scatter shows the same shape, so it is real physics.
    hollow = 1.0 + 0.6 * np.exp(-(((rho - 0.6) / 0.2) ** 2))
    hollow[rho > 1.0] = 0.1
    x, y, e = _tracking_data(rho_ch, hollow, rho)
    assert nonphysical_peak(hollow, rho, x, y, e) is None

    # Edge bump above the whole interior: flagged at the bump even when the
    # scatter supports it (miscalibrated edge channels).
    edge_spike = np.clip(8.0 * (1 - rho / 1.0), 0, None)
    edge_spike[(rho > 0.92) & (rho < 0.99)] = 14.0
    x, y, e = _tracking_data(rho_ch, edge_spike, rho)
    peak = nonphysical_peak(edge_spike, rho, x, y, e)
    assert peak is not None and 0.9 <= peak <= 1.0

    # Sub-core ringing: a 6 keV spike under an 8 keV core passes the edge rule
    # but exceeds the local data envelope (scatter decayed to ~0.2 there).
    ring = np.clip(8.0 * (1 - rho / 0.9), 0.05, None)
    ring[(rho > 0.92) & (rho < 0.99)] = 6.0
    x, y, e = _tracking_data(rho_ch, np.clip(8.0 * (1 - rho / 0.9), 0.05, None), rho)
    peak = nonphysical_peak(ring, rho, x, y, e)
    assert peak is not None and 0.9 <= peak <= 1.0

    # Steep pedestal interpolated across a data gap stays healthy: the fit at
    # the gap sits below its inner neighbor, which the envelope includes.
    ped = 3.0 * (1 - np.tanh((rho - 0.85) / 0.06)) / 2 + 0.05
    gap_ch = np.array([0.1, 0.3, 0.5, 0.72, 0.95, 1.02])
    x, y, e = _tracking_data(gap_ch, ped, rho)
    assert nonphysical_peak(ped, rho, x, y, e) is None

    # All-NaN slice is not flagged (nothing to cull).
    nan_data = np.full(5, np.nan)
    assert nonphysical_peak(np.full_like(rho, np.nan), rho, nan_data, nan_data, nan_data) is None


def test_fit_variable_repairs_then_culls(monkeypatch):
    """Repair path: first fit nonphysical -> refit without the offending
    channels; healthy refit keeps the slice as "repaired", a still-bad refit
    culls it. gp_profile is faked so the shapes are exact."""
    from transport_study.datasets.gp_fitting import fit_worker

    rho = np.linspace(0.0, 1.1, 56)
    n_ch = 12
    x = np.linspace(0.02, 1.05, n_ch)
    good = np.clip(8.0 * (1 - x / 1.0), 0.05, None)
    y = good.copy()
    y[9] = 5.0  # stray edge channel near rho 0.9
    err = np.maximum(0.1 * good, 0.05)

    spiked = np.clip(8.0 * (1 - rho / 1.0), 0.0, None)
    spiked[(rho > 0.9) & (rho < 0.98)] = 14.0
    clean = np.clip(8.0 * (1 - rho / 1.0), 0.0, None)
    band = np.full_like(rho, 0.3)
    hyps = np.array([2.0, 0.8, 0.3, 0.1, 0.95])

    calls = {"n": 0}  # This is so stupid but needed to make the fit worker get its stuff

    def fake_gp_profile(data_X, data_y, err_y, X_star, **kwargs):
        calls["n"] += 1
        out = spiked if calls["n"] == 1 else clean
        return out, band, np.gradient(out, rho), band, hyps

    monkeypatch.setattr(fit_worker, "gp_profile", fake_gp_profile)
    y_out, _, _, _, _, status = fit_variable(x, y, err, rho, 3, False, True, None, None)
    assert status == "repaired"
    assert calls["n"] == 2
    assert np.nanmax(y_out) <= 8.5

    # Always-spiked fit: repair does not help, slice is culled.
    calls["n"] = 0
    monkeypatch.setattr(
        fit_worker,
        "gp_profile",
        lambda *a, **k: (spiked, band, np.gradient(spiked, rho), band, hyps),
    )
    y_out, _, _, _, _, status = fit_variable(x, y, err, rho, 3, False, True, None, None)
    assert status == "culled"
    assert y_out is None


def test_has_fittable_points():
    si = _synthetic_input()
    assert si.has_fittable_points()

    # All-NaN radial coordinate (e.g. failed rho mapping) makes the shot unfittable
    si_bad_x = _synthetic_input()
    si_bad_x.x[:] = np.nan
    assert not si_bad_x.has_fittable_points()

    # One all-NaN variable is enough to skip, since assembly culls the shot anyway
    si_bad_te = _synthetic_input()
    si_bad_te.te_y[:] = np.nan
    assert not si_bad_te.has_fittable_points()

    # A single finite point per variable keeps the shot
    si_sparse = _synthetic_input()
    si_sparse.x[:] = np.nan
    si_sparse.x[0, 0] = 0.5
    assert si_sparse.has_fittable_points()


# ----------------------------------------------------------------------
# Batch file round-trips
# ----------------------------------------------------------------------
def test_fit_batch_npz_roundtrip(tmp_path):
    inputs = {1234: _synthetic_input(0), 5678: _synthetic_input(1)}
    path = tmp_path / "batch_test.npz"
    pack_fit_batch(path, inputs, X_STAR, min_points=4, scale_per_slice=True)

    assert read_batch_shots(path) == [1234, 5678]

    loaded, x_star, min_points, scale_per_slice = unpack_fit_batch(path)
    assert min_points == 4
    assert scale_per_slice is True
    np.testing.assert_allclose(x_star, X_STAR)
    assert set(loaded) == {1234, 5678}
    for shot, si in inputs.items():
        for attr in ("x", "te_y", "te_err", "ne_y", "ne_err"):
            np.testing.assert_allclose(getattr(loaded[shot], attr), getattr(si, attr), rtol=1e-6, atol=1e-6)


def test_fit_results_npz_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    outputs = {
        42: ShotFitOutput(
            te_fit=rng.random((N_T, len(X_STAR))),
            te_std=rng.random((N_T, len(X_STAR))),
            ne_fit=rng.random((N_T, len(X_STAR))),
            ne_std=rng.random((N_T, len(X_STAR))),
            te_grad=rng.random((N_T, len(X_STAR))),
            te_grad_std=rng.random((N_T, len(X_STAR))),
            ne_grad=rng.random((N_T, len(X_STAR))),
            ne_grad_std=rng.random((N_T, len(X_STAR))),
            te_hyps=rng.random((N_T, 5)),
            ne_hyps=rng.random((N_T, 5)),
        )
    }
    path = tmp_path / "batch_test_out.npz"
    pack_fit_results(path, outputs, X_STAR)

    loaded = unpack_fit_results(path)
    assert set(loaded) == {42}
    for attr in ("te_fit", "te_std", "ne_fit", "ne_std", "te_grad", "te_grad_std", "ne_grad", "ne_grad_std", "te_hyps", "ne_hyps"):
        np.testing.assert_allclose(getattr(loaded[42], attr), getattr(outputs[42], attr), rtol=1e-6, atol=1e-6)


def test_atomic_write_leaves_no_tmp_file(tmp_path):
    path = tmp_path / "batch.npz"
    pack_fit_batch(path, {1: _synthetic_input()}, X_STAR, min_points=1, scale_per_slice=False)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))


# ----------------------------------------------------------------------
# Batch planning (restart safety)
# ----------------------------------------------------------------------
def test_batch_id_deterministic():
    assert batch_id("cmod", [3, 1, 2]) == batch_id("cmod", [1, 2, 3])
    assert batch_id("cmod", [1, 2, 3]) != batch_id("cmod", [1, 2, 4])
    assert batch_id("cmod", [1, 2, 3]) != batch_id("mast", [1, 2, 3])


def test_plan_batches_chunks_new_shots(tmp_path):
    plan = plan_batches("cmod", list(range(10)), tmp_path, shots_per_batch=4)
    assert sorted(s for shots in plan.values() for s in shots) == list(range(10))
    sizes = sorted(len(shots) for shots in plan.values())
    assert sizes == [2, 4, 4]
    # Batch ids match their shot lists
    for bid, shots in plan.items():
        assert bid == batch_id("cmod", shots)


def test_plan_batches_reuses_existing_batch_files(tmp_path):
    # A previous run packed shots 0-3 into one batch
    old_shots = [0, 1, 2, 3]
    old_bid = batch_id("cmod", old_shots)
    pack_fit_batch(
        tmp_path / f"batch_{old_bid}.npz",
        {s: _synthetic_input(s) for s in old_shots},
        X_STAR,
        min_points=1,
        scale_per_slice=False,
    )

    # This run still needs shots 2 and 3 (0 and 1 are done), plus new shots 10-12
    plan = plan_batches("cmod", [2, 3, 10, 11, 12], tmp_path, shots_per_batch=4)

    # Shots 2 and 3 stay claimed by the existing batch so its id (and job name)
    # lines up with any job already in the cluster queue
    assert plan[old_bid] == [2, 3]
    new_bids = set(plan) - {old_bid}
    assert len(new_bids) == 1
    assert plan[new_bids.pop()] == [10, 11, 12]


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------
def test_fit_batch_fixed_hyperparams_shapes_and_clipping():
    inputs = {7: _synthetic_input()}
    outputs = fit_batch(
        inputs,
        X_STAR,
        min_points=1,
        scale_per_slice=False,
        num_workers=1,
        optimize_hyperparams=False,
        hyperparams=FIXED_HYPERPARAMS,
    )
    out = outputs[7]
    assert out.te_fit.shape == (N_T, len(X_STAR))
    assert out.ne_fit.shape == (N_T, len(X_STAR))
    for arr in (out.te_fit, out.ne_fit):
        assert np.isfinite(arr).all()
        assert (arr >= 0).all()  # clamped to non-negative
    for arr in (out.te_std, out.ne_std):
        assert np.isfinite(arr).all()
    # Core value should be near the synthetic amplitude
    assert abs(out.te_fit[0, 0] - 2.0) < 0.5


def test_fit_batch_min_points_skips_sparse_slices():
    si = _synthetic_input()
    si.te_y[1, :] = np.nan  # slice 1 has no valid te points
    si.te_err[1, :] = np.nan
    outputs = fit_batch(
        {7: si},
        X_STAR,
        min_points=4,
        scale_per_slice=False,
        num_workers=1,
        optimize_hyperparams=False,
        hyperparams=FIXED_HYPERPARAMS,
    )
    out = outputs[7]
    assert np.isnan(out.te_fit[1, :]).all()
    assert np.isfinite(out.te_fit[0, :]).all()
    assert np.isfinite(out.ne_fit[1, :]).all()  # ne slice untouched


def test_fit_batch_scale_per_slice_recovers_amplitude():
    # Large amplitude (like ne in raw units) needs per-slice normalization
    inputs = {7: _synthetic_input(amplitude=4.0)}
    outputs = fit_batch(
        inputs,
        X_STAR,
        min_points=1,
        scale_per_slice=True,
        num_workers=1,
        optimize_hyperparams=False,
        hyperparams=FIXED_HYPERPARAMS,
    )
    out = outputs[7]
    assert abs(out.te_fit[0, 0] - 4.0) < 1.0


def test_fit_batch_parallel_matches_serial():
    inputs = {7: _synthetic_input()}
    kwargs = dict(
        x_star=X_STAR,
        min_points=1,
        scale_per_slice=False,
        optimize_hyperparams=False,
        hyperparams=FIXED_HYPERPARAMS,
    )
    serial = fit_batch(inputs, num_workers=1, **kwargs)[7]
    parallel = fit_batch(inputs, num_workers=2, **kwargs)[7]
    np.testing.assert_allclose(serial.te_fit, parallel.te_fit, rtol=1e-10)
    np.testing.assert_allclose(serial.ne_fit, parallel.ne_fit, rtol=1e-10)


def test_fit_batch_max_slices_per_shot():
    inputs = {7: _synthetic_input()}
    outputs = fit_batch(
        inputs,
        X_STAR,
        num_workers=1,
        optimize_hyperparams=False,
        hyperparams=FIXED_HYPERPARAMS,
        max_slices_per_shot=1,
    )
    out = outputs[7]
    assert np.isfinite(out.te_fit[0, :]).all()
    assert np.isnan(out.te_fit[1:, :]).all()


# ----------------------------------------------------------------------
# Monotonic-edge constraint (virtual zero-slope observations, MONO_CHECK_RHO)
# ----------------------------------------------------------------------
MONO_X_STAR = np.linspace(0.0, 1.1, 56)
MONO_CH = np.linspace(0.02, 1.05, 22)


def _edge_bump_slice():
    """Pedestal with a data-supported bump on its shoulder, tight error bars so
    the short edge length scale of FIXED_HYPERPARAMS tracks it."""
    base = 1.0 * (1 - np.tanh((MONO_CH - 0.75) / 0.06)) / 2 + 0.05
    y = base + 0.30 * np.exp(-(((MONO_CH - 0.95) / 0.04) ** 2))
    return MONO_CH, y, np.full_like(y, 0.01)


@contextmanager
def _mono_check_rho(values):
    """Temporarily swap the constraint's check grid (empty disables it)."""
    from transport_study.datasets.gp_fitting import fit_worker

    saved = fit_worker.MONO_CHECK_RHO
    fit_worker.MONO_CHECK_RHO = np.asarray(values, dtype=float)
    try:
        yield
    finally:
        fit_worker.MONO_CHECK_RHO = saved


def _fit(x, y, err, **kwargs):
    return gp_profile(x, y, err, MONO_X_STAR, calc_gradient=True, **kwargs)


def test_mono_constraint_suppresses_edge_bump():
    """A slice whose unconstrained fit rises past rho 0.6 comes back flattened.

    The constraint is soft (the virtual observations carry MONO_GRAD_ERR as
    their error bar), so the bump becomes a plateau rather than exactly zero
    slope: what is pinned down here is that the constrained fit cuts the edge
    gradient by most of its value and leaves no rise the unconstrained fit did
    not already have.
    """
    x, y, err = _edge_bump_slice()
    with threadpool_limits(1):
        with _mono_check_rho([]):
            free_mean, _, free_grad, _, _ = _fit(x, y, err, hyperparams=FIXED_HYPERPARAMS, optimize_hyperparams=False)
        mean, _, grad, _, _ = _fit(x, y, err, hyperparams=FIXED_HYPERPARAMS, optimize_hyperparams=False)

    edge = MONO_X_STAR >= 0.6
    assert np.nanmax(free_grad[edge]) > 0.5, "test data does not produce a rising edge without the constraint"
    assert np.nanmax(grad[edge]) < 0.4 * np.nanmax(free_grad[edge])
    # The bump itself is gone from the mean, not just from the gradient
    shoulder = (MONO_X_STAR >= 0.85) & (MONO_X_STAR <= 1.0)
    assert np.max(np.diff(mean[shoulder])) < np.max(np.diff(free_mean[shoulder]))
    # The returned gradient belongs to the same posterior as the returned mean
    fd = np.gradient(mean, MONO_X_STAR)
    interior = (MONO_X_STAR > 0.05) & (MONO_X_STAR < 1.05)
    assert np.nanmax(np.abs(fd[interior] - grad[interior])) < 0.1 * np.nanmax(np.abs(grad[interior]))


def test_mono_constraint_leaves_monotone_slice_untouched():
    """A cleanly monotone pedestal never trips the check, so the fit must be
    bit-identical to one with the check grid emptied out (no virtual
    observations, no extra refit, unchanged deterministic seeding)."""
    y = 1.0 * (1 - np.tanh((MONO_CH - 0.9) / 0.08)) / 2 + 0.05
    err = np.full_like(y, 0.02)
    with threadpool_limits(1):
        mean, std, grad, grad_std, hyps = _fit(MONO_CH, y, err, hyperparams=FIXED_HYPERPARAMS, optimize_hyperparams=False)
        with _mono_check_rho([]):
            free = _fit(MONO_CH, y, err, hyperparams=FIXED_HYPERPARAMS, optimize_hyperparams=False)

    for got, expected in zip((mean, std, grad, grad_std, hyps), free, strict=True):
        np.testing.assert_array_equal(got, expected)


def test_mono_constraint_preserves_hollow_core():
    """A hollow profile keeps its positive core gradient: the check grid starts
    at rho 0.6, so the core is never constrained even when the same slice picks
    up virtual observations further out."""
    y = np.where(MONO_CH > 1.0, 0.05, 0.6 + 0.5 * np.exp(-(((MONO_CH - 0.4) / 0.2) ** 2)))
    err = np.full_like(y, 0.03)
    with threadpool_limits(1):
        mean, _, grad, _, _ = _fit(MONO_CH, y, err, hyperparams=FIXED_HYPERPARAMS, optimize_hyperparams=False)
        with _mono_check_rho([]):
            free_mean, _, free_grad, _, _ = _fit(MONO_CH, y, err, hyperparams=FIXED_HYPERPARAMS, optimize_hyperparams=False)

    core = MONO_X_STAR < 0.5
    assert np.nanmax(free_grad[core]) > 0.1, "test data has no hollow core to preserve"
    np.testing.assert_allclose(grad[core], free_grad[core], atol=0.02)
    np.testing.assert_allclose(mean[core], free_mean[core], atol=0.01)


def test_mono_constraint_refit_failure_keeps_unconstrained_fit(monkeypatch):
    """A constrained refit that fails inside _run_gp leaves the unconstrained
    fit standing instead of losing the slice."""
    from transport_study.datasets.gp_fitting import fit_worker

    x, y, err = _edge_bump_slice()
    with threadpool_limits(1):
        expected = _fit(x, y, err, hyperparams=FIXED_HYPERPARAMS, optimize_hyperparams=False)

        real_run_gp = fit_worker._run_gp

        def failing_refit(*args, extra_grad_bc=None, **kwargs):
            if extra_grad_bc is not None:
                return None
            return real_run_gp(*args, **kwargs)

        monkeypatch.setattr(fit_worker, "_run_gp", failing_refit)
        got = _fit(x, y, err, hyperparams=FIXED_HYPERPARAMS, optimize_hyperparams=False)

    assert got[0] is not None
    with _mono_check_rho([]):
        with threadpool_limits(1):
            monkeypatch.undo()
            free = _fit(x, y, err, hyperparams=FIXED_HYPERPARAMS, optimize_hyperparams=False)
    # Failed refit -> the unconstrained fit, not the one the constraint produced
    np.testing.assert_array_equal(got[0], free[0])
    assert not np.array_equal(got[0], expected[0])


def test_mono_constraint_second_pass_adds_only_new_points(monkeypatch):
    """Virtual observations accumulate without duplicates and the loop is
    bounded: a violation that persists at an already-constrained point adds
    nothing and stops the loop, while one at a new point earns another pass."""
    from transport_study.datasets.gp_fitting import fit_worker

    check_rho = np.array([0.7, 0.8, 0.9])
    n_out = MONO_X_STAR.size
    calls = []

    class FakeGP:
        """Reports a violation at 0.7 on the first fit, then at 0.7 and 0.8."""

        def __init__(self, n_calls):
            self.n_calls = n_calls

        def get_gp_drv_mean(self):
            drv = np.zeros(n_out + check_rho.size)
            drv[n_out + 0] = 1.0  # 0.7 always violates
            if self.n_calls >= 1:
                drv[n_out + 1] = 1.0  # 0.8 starts violating after the first refit
            return drv

        def get_gp_mean(self):
            return np.zeros(n_out + check_rho.size)

        def get_gp_std(self, noise_flag=True):
            return np.zeros(n_out + check_rho.size)

        def get_gp_drv_std(self, noise_flag=False):
            return np.zeros(n_out + check_rho.size)

        def get_gp_kernel_details(self):
            return None, FIXED_HYPERPARAMS

    def fake_run_gp(*args, extra_grad_bc=None, **kwargs):
        calls.append(None if extra_grad_bc is None else np.array(extra_grad_bc))
        return FakeGP(len(calls) - 1)

    monkeypatch.setattr(fit_worker, "_run_gp", fake_run_gp)
    with _mono_check_rho(check_rho):
        _fit(np.array([0.1, 0.5, 0.9]), np.array([1.0, 0.8, 0.1]), np.array([0.1, 0.1, 0.1]), hyperparams=FIXED_HYPERPARAMS)

    # First call is the unconstrained fit, then one refit per newly violated point
    assert len(calls) == 3
    assert calls[0] is None
    np.testing.assert_allclose(calls[1][:, 0], [0.7])
    np.testing.assert_allclose(calls[2][:, 0], [0.7, 0.8])  # 0.7 not duplicated
    assert (calls[2][:, 1] == 0.0).all()
    assert (calls[2][:, 2] == fit_worker.MONO_GRAD_ERR).all()


def test_mono_constraint_hyps_come_from_unconstrained_optimize():
    """The returned hyperparameters (whose x0 pins Te to the ne fit) come from
    the original optimized fit, since the refit runs at fixed hyperparameters."""
    from transport_study.datasets.gp_fitting import fit_worker

    seen = []
    real_run_gp = fit_worker._run_gp

    def recording_run_gp(*args, **kwargs):
        gp = real_run_gp(*args, **kwargs)
        if gp is not None:
            seen.append((kwargs.get("extra_grad_bc"), np.asarray(gp.get_gp_kernel_details()[1], dtype=float)))
        return gp

    x, y, err = _edge_bump_slice()
    with threadpool_limits(1):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(fit_worker, "_run_gp", recording_run_gp)
            _, _, _, _, hyps = _fit(x, y, err, hyperparams=FIXED_HYPERPARAMS, optimize_hyperparams=False)

    unconstrained = [h for bc, h in seen if bc is None]
    assert unconstrained, "no unconstrained fit was recorded"
    np.testing.assert_array_equal(hyps, unconstrained[0])


# ----------------------------------------------------------------------
# Dispatcher loop (against a fake cluster backend)
# ----------------------------------------------------------------------
class _FakeBackend:
    """Stands in for the srunx SSH/local backends.

    Jobs are submitted as rendered sbatch scripts; the fake parses the batch
    npz names, partition, and constraint back out of the script text. Unless a
    failure/pending knob applies, jobs "complete" immediately: a zero-filled
    result npz is written to the fake remote workdir, where the dispatcher's
    pull will find it.

    Knobs:
    - fail_batch_ids: these batches always end FAILED.
    - fail_first_attempts: bid -> number of leading attempts that end FAILED
      before the batch completes.
    - pending_partitions: submissions to these partitions stay PENDING; if
      release_pending_after is set, they complete after that many job_states
      polls.
    """

    def __init__(
        self,
        remote_dir: Path,
        fail_batch_ids: set[str] | None = None,
        fail_first_attempts: dict[str, int] | None = None,
        pending_partitions: set[str] | None = None,
        release_pending_after: int | None = None,
    ):
        self.remote_dir = Path(remote_dir)
        self.remote_dir.mkdir(parents=True, exist_ok=True)
        self.fail_batch_ids = fail_batch_ids or set()
        self.fail_first_attempts = fail_first_attempts or {}
        self.pending_partitions = pending_partitions or set()
        self.release_pending_after = release_pending_after
        self.submitted_names: list[str] = []
        self.submissions: list[tuple[str, str, str | None]] = []  # (name, partition, constraint)
        self.cancelled_ids: list[int] = []
        self._states: dict[int, str] = {}
        self._queued: dict[str, int] = {}
        self._job_paths: dict[int, tuple[Path, Path]] = {}  # job_id -> (in, out)
        self._pending_polls: dict[int, int] = {}
        self._next_id = 100

    def push_file(self, local: Path, remote_dir: str) -> None:
        shutil.copy2(local, self.remote_dir / Path(local).name)

    def pull_file(self, remote_path: str, local_dir: Path) -> bool:
        src = self.remote_dir / Path(remote_path).name
        if not src.exists():
            return False
        shutil.copy2(src, Path(local_dir) / src.name)
        return True

    def _write_output(self, in_path: Path, out_path: Path) -> None:
        shot_inputs, x_star, _, _ = unpack_fit_batch(in_path)
        outputs = {
            shot: ShotFitOutput(
                te_fit=np.zeros((si.te_y.shape[0], len(x_star))),
                te_std=np.zeros((si.te_y.shape[0], len(x_star))),
                ne_fit=np.zeros((si.ne_y.shape[0], len(x_star))),
                ne_std=np.zeros((si.ne_y.shape[0], len(x_star))),
                te_grad=np.zeros((si.te_y.shape[0], len(x_star))),
                te_grad_std=np.zeros((si.te_y.shape[0], len(x_star))),
                ne_grad=np.zeros((si.ne_y.shape[0], len(x_star))),
                ne_grad_std=np.zeros((si.ne_y.shape[0], len(x_star))),
                te_hyps=np.zeros((si.te_y.shape[0], 5)),
                ne_hyps=np.zeros((si.ne_y.shape[0], 5)),
            )
            for shot, si in shot_inputs.items()
        }
        pack_fit_results(out_path, outputs, x_star)

    def submit_script(self, script: str, job_name: str) -> int:
        assert script.startswith("#!/bin/bash")
        assert f"#SBATCH --job-name={job_name}" in script
        self.submitted_names.append(job_name)
        job_id = self._next_id
        self._next_id += 1

        bid = re.search(r"batch_([0-9a-f]+)\.npz", script).group(1)
        partition = re.search(r"--partition=(\S+)", script).group(1)
        constraint_m = re.search(r"--constraint=(\S+)", script)
        constraint = constraint_m.group(1) if constraint_m else None
        attempt = int(job_name.rsplit("-a", 1)[-1])
        self.submissions.append((job_name, partition, constraint))

        in_path = self.remote_dir / f"batch_{bid}.npz"
        out_path = self.remote_dir / f"batch_{bid}_out.npz"
        self._job_paths[job_id] = (in_path, out_path)

        if partition in self.pending_partitions:
            self._states[job_id] = "PENDING"
            self._pending_polls[job_id] = 0
        elif bid in self.fail_batch_ids or attempt <= self.fail_first_attempts.get(bid, 0):
            self._states[job_id] = "FAILED"
        else:
            self._write_output(in_path, out_path)
            self._states[job_id] = "COMPLETED"
        return job_id

    def ensure_dir(self, path: str) -> None:
        pass

    def queued_job_names(self) -> dict[str, int]:
        return dict(self._queued)

    def job_states(self, job_ids: list[int]) -> dict[int, str]:
        for jid in job_ids:
            if self._states.get(jid) == "PENDING" and self.release_pending_after is not None:
                self._pending_polls[jid] = self._pending_polls.get(jid, 0) + 1
                if self._pending_polls[jid] >= self.release_pending_after:
                    in_path, out_path = self._job_paths[jid]
                    self._write_output(in_path, out_path)
                    self._states[jid] = "COMPLETED"
        return {jid: self._states[jid] for jid in job_ids if jid in self._states}

    def cancel(self, job_id: int) -> None:
        self.cancelled_ids.append(job_id)
        self._states[job_id] = "CANCELLED"

    def remove_glob(self, remote_dir: str, pattern: str) -> None:
        for p in self.remote_dir.glob(pattern):
            p.unlink(missing_ok=True)


def _make_dispatcher(
    tmp_path,
    monkeypatch,
    fail_batch_ids=None,
    fail_first_attempts=None,
    pending_partitions=None,
    release_pending_after=None,
    **config_overrides,
):
    fake = _FakeBackend(
        tmp_path / "remote",
        fail_batch_ids,
        fail_first_attempts,
        pending_partitions,
        release_pending_after,
    )
    monkeypatch.setattr(ClusterFitDispatcher, "_create_backend", staticmethod(lambda config: fake))
    config_kwargs = dict(
        profile="fake",
        partitions="cpu@7:50:00",
        remote_workdir=str(tmp_path / "remote"),
        venv_path="/fake/.venv",
        shots_per_batch=2,
        max_concurrent_jobs=1,
        poll_interval_s=0.01,
    )
    config_kwargs.update(config_overrides)
    config = ClusterFitConfig(**config_kwargs)
    dispatcher = ClusterFitDispatcher(config, "cmod", tmp_path / "staging")
    return dispatcher, fake


def test_dispatcher_run_completes_batches(tmp_path, monkeypatch):
    dispatcher, fake = _make_dispatcher(tmp_path, monkeypatch)
    inputs = {s: _synthetic_input(s) for s in (1, 2, 3)}

    results = dispatcher.run(inputs, X_STAR, min_points=1, scale_per_slice=False)

    assert set(results) == {1, 2, 3}
    assert all(isinstance(out, ShotFitOutput) for out in results.values())
    # 3 shots with shots_per_batch=2 -> two jobs, despite max_concurrent_jobs=1
    assert len(fake.submitted_names) == 2
    assert all(name.startswith("gpfit-cmod-") and name.endswith("-a1") for name in fake.submitted_names)
    # Outputs were pulled back into local staging
    assert len(list((tmp_path / "staging" / "batches").glob("batch_*_out.npz"))) == 2


def test_dispatcher_failed_batch_returns_none(tmp_path, monkeypatch):
    failed_bid = batch_id("cmod", [3])
    dispatcher, fake = _make_dispatcher(tmp_path, monkeypatch, fail_batch_ids={failed_bid}, max_retries=0)
    inputs = {s: _synthetic_input(s) for s in (1, 2, 3)}

    results = dispatcher.run(inputs, X_STAR, min_points=1, scale_per_slice=False)

    assert isinstance(results[1], ShotFitOutput)
    assert isinstance(results[2], ShotFitOutput)
    assert results[3] is None
    # max_retries=0: the failed batch was submitted exactly once
    assert len([n for n in fake.submitted_names if failed_bid in n]) == 1


def test_dispatcher_reuses_local_outputs_without_jobs(tmp_path, monkeypatch):
    # First run completes everything
    dispatcher, fake = _make_dispatcher(tmp_path, monkeypatch)
    inputs = {s: _synthetic_input(s) for s in (1, 2, 3)}
    dispatcher.run(inputs, X_STAR, min_points=1, scale_per_slice=False)
    assert len(fake.submitted_names) == 2

    # Second run finds the local outputs and submits nothing
    dispatcher2, fake2 = _make_dispatcher(tmp_path, monkeypatch)
    results = dispatcher2.run(inputs, X_STAR, min_points=1, scale_per_slice=False)
    assert len(fake2.submitted_names) == 0
    assert all(isinstance(out, ShotFitOutput) for out in results.values())


def test_dispatcher_clean_cancels_jobs_and_removes_batches(tmp_path, monkeypatch):
    dispatcher, fake = _make_dispatcher(tmp_path, monkeypatch)
    inputs = {s: _synthetic_input(s) for s in (1, 2, 3)}
    dispatcher.run(inputs, X_STAR, min_points=1, scale_per_slice=False)
    batches_dir = tmp_path / "staging" / "batches"
    assert len(list(batches_dir.glob("batch_*.npz"))) > 0
    assert len(list(fake.remote_dir.glob("batch_*.npz"))) > 0

    # A stale job for this device should be cancelled; another device's job left alone
    fake._queued = {"gpfit-cmod-stale00000-a1": 555, "gpfit-mast-other000000-a1": 777}

    dispatcher.clean()

    assert fake.cancelled_ids == [555]
    assert list(batches_dir.glob("batch_*.npz")) == []
    assert list(fake.remote_dir.glob("batch_*.npz")) == []


def test_dispatcher_clean_removes_orphaned_remote_outputs(tmp_path, monkeypatch):
    """Remote batch files with no local counterpart must not survive clean.

    Regression: clean used to derive the remote deletion list from the local
    batch files, so a stale remote output (e.g. from a run whose staging was
    already wiped) survived and was adopted as pre-existing results by the
    next run, silently reusing old fits.
    """
    dispatcher, fake = _make_dispatcher(tmp_path, monkeypatch)
    (fake.remote_dir / "batch_deadbeef00_out.npz").write_bytes(b"stale")
    assert list((tmp_path / "staging" / "batches").glob("batch_*.npz")) == []

    dispatcher.clean()

    assert list(fake.remote_dir.glob("batch_*.npz")) == []


# ----------------------------------------------------------------------
# Retry, partition fallback, and reconciliation
# ----------------------------------------------------------------------
def test_parse_partition_specs():
    specs = parse_partition_specs("p1@8:00:00,p2@1-00:00:00@rocky8")
    assert specs == [
        PartitionSpec(name="p1", time_limit="8:00:00"),
        PartitionSpec(name="p2", time_limit="1-00:00:00", constraint="rocky8"),
    ]
    # Fire may pre-split a comma-separated argument into a tuple
    assert parse_partition_specs(("p1@8:00:00", "p2@1:00:00")) == parse_partition_specs("p1@8:00:00,p2@1:00:00")
    with pytest.raises(ValueError):
        parse_partition_specs("p1")
    with pytest.raises(ValueError):
        parse_partition_specs("p1@1:00:00@rocky8@extra")


def test_cluster_fit_config_parses_partitions():
    config = ClusterFitConfig(profile="fake", partitions="p1@1:00:00,p2@2:00:00@rocky8", remote_workdir="/r", venv_path="/v")
    assert config.partitions == [
        PartitionSpec(name="p1", time_limit="1:00:00"),
        PartitionSpec(name="p2", time_limit="2:00:00", constraint="rocky8"),
    ]
    # An already-parsed list passes through untouched
    specs = [PartitionSpec(name="p1", time_limit="1:00:00")]
    config = ClusterFitConfig(profile="fake", partitions=specs, remote_workdir="/r", venv_path="/v")
    assert config.partitions == specs
    with pytest.raises(ValueError):
        ClusterFitConfig(profile="fake", partitions="p1", remote_workdir="/r", venv_path="/v")


def test_dispatcher_retries_failed_batch_on_next_partition(tmp_path, monkeypatch):
    bid = batch_id("cmod", [1, 2])
    dispatcher, fake = _make_dispatcher(
        tmp_path,
        monkeypatch,
        fail_first_attempts={bid: 1},
        partitions="p1@1:00:00,p2@2:00:00@rocky8",
    )
    inputs = {s: _synthetic_input(s) for s in (1, 2)}

    results = dispatcher.run(inputs, X_STAR, min_points=1, scale_per_slice=False)

    assert all(isinstance(out, ShotFitOutput) for out in results.values())
    assert [name.rsplit("-", 1)[-1] for name, _, _ in fake.submissions] == ["a1", "a2"]
    # First attempt on p1 (no constraint), retry lands on p2 with its constraint
    assert fake.submissions[0][1:] == ("p1", None)
    assert fake.submissions[1][1:] == ("p2", "rocky8")


def test_dispatcher_retry_exhaustion_wraps_partitions(tmp_path, monkeypatch):
    bid = batch_id("cmod", [1, 2])
    dispatcher, fake = _make_dispatcher(
        tmp_path,
        monkeypatch,
        fail_batch_ids={bid},
        partitions="p1@1:00:00,p2@2:00:00",
        max_retries=3,
    )
    inputs = {s: _synthetic_input(s) for s in (1, 2)}

    results = dispatcher.run(inputs, X_STAR, min_points=1, scale_per_slice=False)

    assert results[1] is None and results[2] is None
    # 1 initial + 3 retries, wrapping around the partition list
    assert [partition for _, partition, _ in fake.submissions] == ["p1", "p2", "p1", "p2"]


def test_dispatcher_pending_timeout_falls_back(tmp_path, monkeypatch):
    dispatcher, fake = _make_dispatcher(
        tmp_path,
        monkeypatch,
        pending_partitions={"stuck"},
        partitions="stuck@1:00:00,ok@2:00:00",
        pending_timeout_s=0.0,
    )
    inputs = {s: _synthetic_input(s) for s in (1, 2)}

    results = dispatcher.run(inputs, X_STAR, min_points=1, scale_per_slice=False)

    assert all(isinstance(out, ShotFitOutput) for out in results.values())
    assert len(fake.cancelled_ids) == 1
    assert [partition for _, partition, _ in fake.submissions] == ["stuck", "ok"]


def test_dispatcher_single_partition_never_cancels_pending(tmp_path, monkeypatch):
    dispatcher, fake = _make_dispatcher(
        tmp_path,
        monkeypatch,
        pending_partitions={"stuck"},
        release_pending_after=3,
        partitions="stuck@1:00:00",
        pending_timeout_s=0.0,
    )
    inputs = {s: _synthetic_input(s) for s in (1, 2)}

    results = dispatcher.run(inputs, X_STAR, min_points=1, scale_per_slice=False)

    assert all(isinstance(out, ShotFitOutput) for out in results.values())
    assert fake.cancelled_ids == []
    assert len(fake.submitted_names) == 1


def test_dispatcher_adopts_suffixed_job_on_restart(tmp_path, monkeypatch):
    bid = batch_id("cmod", [1, 2])
    dispatcher, fake = _make_dispatcher(
        tmp_path,
        monkeypatch,
        release_pending_after=1,
    )
    # A previous run left attempt 2 in the queue (plus a stale attempt 1);
    # once polled, the pending job completes and writes its output.
    fake._queued = {f"gpfit-cmod-{bid}-a1": 444, f"gpfit-cmod-{bid}-a2": 555}
    fake._states = {444: "PENDING", 555: "PENDING"}
    staging_input = tmp_path / "staging" / "batches" / f"batch_{bid}.npz"
    fake._job_paths = {
        444: (staging_input, fake.remote_dir / f"batch_{bid}_out.npz"),
        555: (staging_input, fake.remote_dir / f"batch_{bid}_out.npz"),
    }
    inputs = {s: _synthetic_input(s) for s in (1, 2)}

    results = dispatcher.run(inputs, X_STAR, min_points=1, scale_per_slice=False)

    assert all(isinstance(out, ShotFitOutput) for out in results.values())
    # Adopted the highest attempt instead of resubmitting, cancelled the stale one
    assert fake.submitted_names == []
    assert fake.cancelled_ids == [444]


def test_run_summary_lists_failed_shots(tmp_path):
    batches = [
        BatchState(
            bid="aaaa",
            shots=[1, 2],
            input_path=tmp_path / "batch_aaaa.npz",
            output_path=tmp_path / "batch_aaaa_out.npz",
            job_base_name="gpfit-cmod-aaaa",
            attempt=3,
            failed=True,
            fail_reason="job 42 ended in state TIMEOUT",
        ),
        BatchState(
            bid="bbbb",
            shots=[3],
            input_path=tmp_path / "batch_bbbb.npz",
            output_path=tmp_path / "batch_bbbb_out.npz",
            job_base_name="gpfit-cmod-bbbb",
            attempt=1,
            done=True,
        ),
    ]
    results = {1: None, 2: None, 3: ShotFitOutput(*[np.zeros((1, 1))] * 10)}

    summary = ClusterFitDispatcher._run_summary(batches, results)

    assert "3 shots requested, 1 fitted, 2 FAILED" in summary
    assert "TIMEOUT" in summary
    assert "1, 2" in summary
    assert "bbbb" not in summary


@pytest.mark.slow
def test_worker_main_end_to_end(tmp_path):
    """Full cluster code path: pack input npz, run worker CLI, unpack results.

    Uses a single time slice with real hyperparameter optimization, as on the
    cluster (several seconds).
    """
    si = _synthetic_input()
    si_single = ShotFitInput(
        x=si.x[:1],
        te_y=si.te_y[:1],
        te_err=si.te_err[:1],
        ne_y=si.ne_y[:1],
        ne_err=si.ne_err[:1],
    )
    in_path = tmp_path / "batch_in.npz"
    out_path = tmp_path / "batch_out.npz"
    pack_fit_batch(in_path, {99: si_single}, X_STAR, min_points=4, scale_per_slice=True)

    main([str(in_path), str(out_path), "--num-workers", "1"])

    outputs = unpack_fit_results(out_path)
    out = outputs[99]
    assert out.te_fit.shape == (1, len(X_STAR))
    assert np.isfinite(out.te_fit).all()
    assert (out.te_fit >= 0).all()
    assert abs(out.te_fit[0, 0] - 2.0) < 0.5


# ----------------------------------------------------------------------
# Real-data spot checks (pull source data, fit a few slices, save plots)
# ----------------------------------------------------------------------
# One PDF per shot lands here for manual eyeballing of fit quality.
GP_FIT_PLOT_DIR = PACKAGE_ROOT / "tests" / "test_outputs" / "gp_fitting"

# mkgp's optimizer draws its random restarts from the global numpy RNG. Seed it
# so these spot-check fits and the plots they save are reproducible run to run.
# this pins one realization and hides the run-to-run restart variability
# that is itself a failure mode of the unseeded production fit.
GP_FIT_SEED = 0


@contextmanager
def _spawn_for_disruption_py():
    """disruption_py's get_shots_data always builds a multiprocessing.Pool, even
    for num_processes=1. Pool() forks by default, and forking while pytest holds
    internal logging/thread locks deadlocks the child forever at 0% CPU (verified:
    reproduces with no fitting code at all, and persists with output capture
    disabled, so it isn't a captured-stdout pipe issue - it's a fork-inherited
    lock). "spawn" starts each worker from a fresh interpreter instead of forking,
    which sidesteps the inherited-lock deadlock. Only the C-Mod prepare_shot call
    goes through disruption_py's SQL/MDSplus retrieval, so this is scoped tightly
    around that rather than changed for the whole test session (MAST retrieval and
    fit_worker's own slice-level multiprocessing.Pool are unaffected either way).
    """
    orig = multiprocessing.get_start_method(allow_none=True)
    multiprocessing.set_start_method("spawn", force=True)
    try:
        yield
    finally:
        multiprocessing.set_start_method(orig, force=True)


def _nearest_indices(times: np.ndarray, targets) -> list[int]:
    return [int(np.argmin(np.abs(times - t))) for t in targets]


def _slice_input(fit_input: ShotFitInput, idxs) -> ShotFitInput:
    """A ShotFitInput holding only the given time-slice rows (in idxs order)."""
    return ShotFitInput(
        x=fit_input.x[idxs],
        te_y=fit_input.te_y[idxs],
        te_err=fit_input.te_err[idxs],
        ne_y=fit_input.ne_y[idxs],
        ne_err=fit_input.ne_err[idxs],
    )


#  TS measurement times [s] to spot-check, one (shot, time) pair per case. These
# slices are the ones known to produce questionable fits, so they are the ones
# worth inspecting individually.
CMOD_SPOT_CHECK = [
    (1160503001, 0.310),
    (1160503001, 0.410),
    (1160503001, 1.010),
    (1160503001, 1.110),
    (1160503001, 1.610),
    (1160503003, 0.710),
    (1160503003, 1.610),
    (1160503004, 1.210),
    (1160503008, 1.410),
    (1160503009, 0.810),
]


@pytest.mark.slow
class TestGPFitCMOD:
    """Spot check the GP fitting on select C-Mod shots and timesteps.

    Pulls Thomson + EFIT through the same prepare_shot codepath the cmod CLI
    uses (so TS channels get mapped onto rho), fits one TS measurement time per
    test case with the production GP path, and writes a diagnostic PDF to
    tests/test_outputs/gp_fitting/{shot}_t{time}/ for eyeballing. Parametrized
    one timestep at a time (rather than bundling a shot's timesteps into one
    test) so a single slice can be run, debugged, or inspected in isolation.
    The plots show the exact (floored, unit-converted) channel data the fit
    consumed. Requires local C-Mod MDSplus access. Skips otherwise.
    """

    SPOT_CHECK = CMOD_SPOT_CHECK

    @pytest.fixture(scope="class")
    def workflow(self, tmp_path_factory):
        try:
            from transport_study.datasets.cmod.cmod_dataset import CModDataWorkflow
        except ImportError as e:
            pytest.skip(f"C-Mod workflow deps unavailable: {e}")
        tmp = tmp_path_factory.mktemp("cmod_gpfit")
        shots = sorted({shot for shot, _ in self.SPOT_CHECK})
        shotlist = tmp / "shotlist"
        shotlist.write_text("\n".join(str(s) for s in shots) + "\n")
        return CModDataWorkflow(
            ds_name="cmod_gpfit_test",
            shotlist_file=shotlist,
            data_assembly_dir=tmp,
            max_num_shots=len(shots),
        )

    @pytest.mark.parametrize(("shot", "t"), SPOT_CHECK, ids=[f"{s}-t{t:.3f}" for s, t in SPOT_CHECK])
    def test_spot_check_timestep(self, workflow, shot, t):
        import xarray as xr

        try:
            # prepare_shot stages source data to netCDF and returns early from
            # that cache on repeat calls, so re-calling it for each of a shot's
            # timestep cases only hits MDSplus once per shot, not once per case.
            with _spawn_for_disruption_py():
                fit_input = workflow.prepare_shot(shot)
        except Exception as e:
            pytest.skip(f"C-Mod data unreachable for shot {shot}: {e}")
        if fit_input is None:
            pytest.skip(f"C-Mod shot {shot} returned no fittable data (data access?)")

        thomson_path, _ = workflow.staging_paths(shot)
        ds_thomson = xr.load_dataset(thomson_path)
        times = ds_thomson.squeeze("shot", drop=True)["time"].values
        idx = _nearest_indices(times, [t])[0]

        np.random.seed(GP_FIT_SEED)
        # By the time this test module is collected, numpy/OpenBLAS is already
        # loaded (pytest plugins, other test modules), so fit_worker's own
        # OPENBLAS_NUM_THREADS=1 setdefault came too late and OpenBLAS would
        # otherwise spin up one thread per core. These per-slice fit matrices
        # are tiny (tens of points), so that's pure thread overhead - it turned
        # a ~30s/slice fit into something that didn't finish in 15+ minutes.
        with threadpool_limits(1):
            out = fit_batch(
                {shot: _slice_input(fit_input, [idx])},
                x_star=workflow.gp_fit_rho,
                min_points=workflow.fit_min_points,
                scale_per_slice=workflow.fit_scale_per_slice,
                num_workers=1,
            )[shot]

        assert np.isfinite(out.te_fit[0]).any(), f"shot {shot} t={t}: Te fit all NaN"
        assert np.isfinite(out.ne_fit[0]).any(), f"shot {shot} t={t}: ne fit all NaN"

        # One directory per (shot, time) case: debug_plot_profiles always names
        # its file "{shot}_ts_gp_fit.pdf", so separate cases for the same shot
        # would otherwise overwrite each other's output.
        plot_dir = GP_FIT_PLOT_DIR / f"{shot}_t{t:.3f}"
        workflow.debug_plot_profiles(shot, ds_thomson.isel(time=[idx]), out, debug_plot_dir=plot_dir)
        assert (plot_dir / f"{shot}_ts_gp_fit.pdf").exists()


@pytest.mark.slow
class TestGPFitMAST:
    """Spot check the GP fitting on a MAST shot, one timestep at a time.

    Mirrors TestGPFitCMOD against the open-access MAST S3 store (shot 30284),
    using the mast CLI's prepare_shot codepath. Each test case fits one of the
    best-covered TS slices (ranked by valid-channel count) and writes a
    diagnostic PDF to tests/test_outputs/gp_fitting/{shot}_t{time}/. Requires
    network access to the MAST store. Skips otherwise.
    """

    SHOT = 30284
    N_SPOT_CHECK = 3  # number of best-covered TS slices to fit and plot

    @pytest.fixture(scope="class")
    def workflow(self, tmp_path_factory):
        try:
            from transport_study.datasets.mast.mast_dataset import (
                MASTDataWorkflow,
                check_required_signals,
                config,
            )
        except ImportError as e:
            pytest.skip(f"MAST workflow deps unavailable: {e}")
        try:
            reachable = check_required_signals(self.SHOT, config["data_sources"])
        except Exception:
            reachable = False
        if not reachable:
            pytest.skip(f"MAST store unreachable or shot {self.SHOT} missing")
        tmp = tmp_path_factory.mktemp("mast_gpfit")
        shotlist = tmp / "shotlist"
        shotlist.write_text(f"{self.SHOT}\n")
        return MASTDataWorkflow(
            ds_name="mast_gpfit_test",
            shotlist_file=shotlist,
            data_assembly_dir=tmp,
            max_num_shots=1,
        )

    @pytest.mark.parametrize("rank", range(N_SPOT_CHECK))
    def test_spot_check_timestep(self, workflow, rank):
        """rank=0 is the best-covered TS slice, rank=1 the next best, etc."""
        import xarray as xr

        # prepare_shot stages to netCDF and returns early from that cache on
        # repeat calls, so re-calling it per rank only hits S3 once per class.
        fit_input = workflow.prepare_shot(self.SHOT)
        if fit_input is None:
            pytest.skip(f"MAST shot {self.SHOT} returned no fittable data")

        ds_staging = xr.load_dataset(workflow.staging_path(self.SHOT))
        ts_time = ds_staging["ts_time"].values
        te_eV = ds_staging["ts_te_eV"].values
        ne_m3 = ds_staging["ts_ne_m3"].values
        rho_ts = ds_staging["ts_rho"].values

        # Rank slices by valid-channel count (mid-shot, hot plasma tends to win)
        valid_per_slice = np.sum(np.isfinite(fit_input.x) & np.isfinite(fit_input.te_y), axis=1)
        idxs = sorted(int(i) for i in np.argsort(valid_per_slice)[::-1][: self.N_SPOT_CHECK])
        i = idxs[rank]

        np.random.seed(GP_FIT_SEED)
        with threadpool_limits(1):
            out = fit_batch(
                {self.SHOT: _slice_input(fit_input, [i])},
                x_star=workflow.gp_fit_rho,
                min_points=workflow.fit_min_points,
                scale_per_slice=workflow.fit_scale_per_slice,
                num_workers=1,
            )[self.SHOT]

        assert np.isfinite(out.te_fit[0]).any(), f"slice {i}: Te fit all NaN"
        assert np.isfinite(out.ne_fit[0]).any(), f"slice {i}: ne fit all NaN"

        # One directory per rank: debug_plot_profiles always names its file
        # "{shot}_ts_gp_fit.pdf", so separate cases would otherwise clobber
        # each other's output.
        plot_dir = GP_FIT_PLOT_DIR / f"{self.SHOT}_t{ts_time[i]:.3f}"
        workflow.debug_plot_profiles(self.SHOT, ts_time[[i]], te_eV[[i]], ne_m3[[i]], rho_ts[[i]], out, plot_dir)
        assert (plot_dir / f"{self.SHOT}_ts_gp_fit.pdf").exists()
