"""Tests for the GP fitting batch format, worker, and dispatcher logic.

The real cluster interaction (srunx submission, rsync) needs a cluster, so the
dispatcher loop is exercised against a fake backend; everything else here runs
exactly the code the cluster path uses: the npz round-trip, deterministic batch
planning for restart safety, and the fitting math the worker executes.
"""

import multiprocessing
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from transport_study import PACKAGE_ROOT
from transport_study.datasets.gp_fitting.dispatcher import (
    ClusterFitConfig,
    ClusterFitDispatcher,
    batch_id,
    plan_batches,
)
from transport_study.datasets.gp_fitting.fit_worker import (
    ShotFitInput,
    ShotFitOutput,
    fit_batch,
    main,
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
        )
    }
    path = tmp_path / "batch_test_out.npz"
    pack_fit_results(path, outputs, X_STAR)

    loaded = unpack_fit_results(path)
    assert set(loaded) == {42}
    for attr in ("te_fit", "te_std", "ne_fit", "ne_std"):
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
# Dispatcher loop (against a fake cluster backend)
# ----------------------------------------------------------------------
class _FakeBackend:
    """Stands in for the srunx SSH/local backends.

    Submitted jobs "complete" immediately: unless the batch is in
    fail_batch_ids, a zero-filled result npz is written to the fake remote
    workdir, where the dispatcher's pull will find it.
    """

    def __init__(self, remote_dir: Path, fail_batch_ids: set[str] | None = None):
        self.remote_dir = Path(remote_dir)
        self.remote_dir.mkdir(parents=True, exist_ok=True)
        self.fail_batch_ids = fail_batch_ids or set()
        self.submitted_names: list[str] = []
        self._states: dict[int, str] = {}
        self._next_id = 100

    def push_file(self, local: Path, remote_dir: str) -> None:
        shutil.copy2(local, self.remote_dir / Path(local).name)

    def pull_file(self, remote_path: str, local_dir: Path) -> bool:
        src = self.remote_dir / Path(remote_path).name
        if not src.exists():
            return False
        shutil.copy2(src, Path(local_dir) / src.name)
        return True

    def submit(self, job) -> int:
        self.submitted_names.append(job.name)
        job_id = self._next_id
        self._next_id += 1

        in_path = self.remote_dir / Path(job.command[2]).name
        out_path = self.remote_dir / Path(job.command[3]).name
        bid = job.name.rsplit("-", 1)[-1]
        if bid in self.fail_batch_ids:
            self._states[job_id] = "FAILED"
            return job_id

        shot_inputs, x_star, _, _ = unpack_fit_batch(in_path)
        outputs = {
            shot: ShotFitOutput(
                te_fit=np.zeros((si.te_y.shape[0], len(x_star))),
                te_std=np.zeros((si.te_y.shape[0], len(x_star))),
                ne_fit=np.zeros((si.ne_y.shape[0], len(x_star))),
                ne_std=np.zeros((si.ne_y.shape[0], len(x_star))),
            )
            for shot, si in shot_inputs.items()
        }
        pack_fit_results(out_path, outputs, x_star)
        self._states[job_id] = "COMPLETED"
        return job_id

    def queued_job_names(self) -> dict[str, int]:
        return {}

    def job_states(self, job_ids: list[int]) -> dict[int, str]:
        return {jid: self._states[jid] for jid in job_ids if jid in self._states}


def _make_dispatcher(tmp_path, monkeypatch, fail_batch_ids=None):
    fake = _FakeBackend(tmp_path / "remote", fail_batch_ids)
    monkeypatch.setattr(ClusterFitDispatcher, "_create_backend", staticmethod(lambda config: fake))
    config = ClusterFitConfig(
        profile="fake",
        partition="cpu",
        remote_workdir=str(tmp_path / "remote"),
        venv_path="/fake/.venv",
        shots_per_batch=2,
        max_concurrent_jobs=1,
        poll_interval_s=0.01,
    )
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
    assert all(name.startswith("gpfit-cmod-") for name in fake.submitted_names)
    # Outputs were pulled back into local staging
    assert len(list((tmp_path / "staging" / "batches").glob("batch_*_out.npz"))) == 2


def test_dispatcher_failed_batch_returns_none(tmp_path, monkeypatch):
    failed_bid = batch_id("cmod", [3])
    dispatcher, _ = _make_dispatcher(tmp_path, monkeypatch, fail_batch_ids={failed_bid})
    inputs = {s: _synthetic_input(s) for s in (1, 2, 3)}

    results = dispatcher.run(inputs, X_STAR, min_points=1, scale_per_slice=False)

    assert isinstance(results[1], ShotFitOutput)
    assert isinstance(results[2], ShotFitOutput)
    assert results[3] is None


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
# blue: this pins one realization and hides the run-to-run restart variability
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
    which sidesteps the inherited-lock deadlock; only the C-Mod _prepare_shot call
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


@pytest.mark.slow
class TestGPFitCMOD:
    """Spot check the GP fitting on select C-Mod shots and timesteps.

    Pulls Thomson + EFIT through the same _prepare_shot codepath the cmod CLI
    uses (so TS channels get mapped onto rho), fits a handful of TS measurement
    times with the production GP path, and writes one diagnostic PDF per shot to
    tests/test_outputs/gp_fitting for eyeballing. The plots show the exact
    (floored, unit-converted) channel data the fit consumed. Requires local
    C-Mod MDSplus access; skips otherwise.
    """

    # TS measurement times [s] to spot-check per shot. These slices are the ones
    # known to produce questionable fits, so they are the ones worth inspecting.
    SPOT_CHECK = {
        1160503003: [0.710, 1.610],
        1160503001: [1.010, 1.110, 1.610],
    }

    @pytest.fixture(scope="class")
    def workflow(self, tmp_path_factory):
        try:
            from transport_study.datasets.cmod.cmod_dataset import CModDataWorkflow
        except ImportError as e:
            pytest.skip(f"C-Mod workflow deps unavailable: {e}")
        tmp = tmp_path_factory.mktemp("cmod_gpfit")
        shotlist = tmp / "shotlist"
        shotlist.write_text("\n".join(str(s) for s in self.SPOT_CHECK) + "\n")
        return CModDataWorkflow(
            ds_name="cmod_gpfit_test",
            shotlist_file=shotlist,
            data_assembly_dir=tmp,
            max_num_shots=len(self.SPOT_CHECK),
        )

    @pytest.mark.parametrize("shot", list(SPOT_CHECK))
    def test_spot_check_shot(self, workflow, shot):
        import xarray as xr

        try:
            with _spawn_for_disruption_py():
                fit_input = workflow._prepare_shot(shot)
        except Exception as e:
            pytest.skip(f"C-Mod data unreachable for shot {shot}: {e}")
        if fit_input is None:
            pytest.skip(f"C-Mod shot {shot} returned no fittable data (data access?)")

        thomson_path, _ = workflow._staging_paths(shot)
        ds_thomson = xr.load_dataset(thomson_path)
        times = ds_thomson.squeeze("shot", drop=True)["time"].values
        idxs = _nearest_indices(times, self.SPOT_CHECK[shot])

        np.random.seed(GP_FIT_SEED)
        # By the time this test module is collected, numpy/OpenBLAS is already
        # loaded (pytest plugins, other test modules), so fit_worker's own
        # OPENBLAS_NUM_THREADS=1 setdefault came too late and OpenBLAS would
        # otherwise spin up one thread per core. These per-slice fit matrices
        # are tiny (tens of points), so that's pure thread overhead - it turned
        # a ~30s/slice fit into something that didn't finish in 15+ minutes.
        with threadpool_limits(1):
            out = fit_batch(
                {shot: _slice_input(fit_input, idxs)},
                x_star=workflow.gp_fit_rho,
                min_points=workflow.fit_min_points,
                scale_per_slice=workflow.fit_scale_per_slice,
                num_workers=1,
            )[shot]

        # Every requested slice must produce a usable (finite, non-empty) fit
        for k, t in enumerate(self.SPOT_CHECK[shot]):
            assert np.isfinite(out.te_fit[k]).any(), f"shot {shot} t={t}: Te fit all NaN"
            assert np.isfinite(out.ne_fit[k]).any(), f"shot {shot} t={t}: ne fit all NaN"

        ds_profiles = workflow._profiles_dataset_from_fit(shot, times[idxs], out)
        workflow._debug_plot_profiles(
            shot,
            ds_thomson.isel(time=idxs),
            ds_profiles,
            debug_plot_dir=GP_FIT_PLOT_DIR,
        )
        assert (GP_FIT_PLOT_DIR / f"{shot}_ts_gp_fit.pdf").exists()


@pytest.mark.slow
class TestGPFitMAST:
    """Spot check the GP fitting on a MAST shot and a few timesteps.

    Mirrors TestGPFitCMOD against the open-access MAST S3 store (shot 30284),
    using the mast CLI's _prepare_shot codepath. The best-covered TS slices are
    fit and a diagnostic PDF is written to tests/test_outputs/gp_fitting.
    Requires network access to the MAST store; skips otherwise.
    """

    SHOT = 30284
    N_SPOT_CHECK = 3  # number of best-covered TS slices to fit and plot

    @pytest.fixture(scope="class")
    def workflow(self, tmp_path_factory):
        try:
            from transport_study.datasets.mast.mast_dataset import (
                MASTDataWorkflow,
                _check_required_signals,
                config,
            )
        except ImportError as e:
            pytest.skip(f"MAST workflow deps unavailable: {e}")
        try:
            reachable = _check_required_signals(self.SHOT, config["data_sources"])
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

    def test_spot_check(self, workflow):
        import xarray as xr

        fit_input = workflow._prepare_shot(self.SHOT)
        if fit_input is None:
            pytest.skip(f"MAST shot {self.SHOT} returned no fittable data")

        ds_staging = xr.load_dataset(workflow._staging_path(self.SHOT))
        ts_time = ds_staging["ts_time"].values
        te_eV = ds_staging["ts_te_eV"].values
        ne_m3 = ds_staging["ts_ne_m3"].values
        rho_ts = ds_staging["ts_rho"].values

        # Plot the slices with the most valid channels (mid-shot, hot plasma)
        valid_per_slice = np.sum(np.isfinite(fit_input.x) & np.isfinite(fit_input.te_y), axis=1)
        idxs = sorted(int(i) for i in np.argsort(valid_per_slice)[::-1][: self.N_SPOT_CHECK])

        np.random.seed(GP_FIT_SEED)
        with threadpool_limits(1):
            out = fit_batch(
                {self.SHOT: _slice_input(fit_input, idxs)},
                x_star=workflow.gp_fit_rho,
                min_points=workflow.fit_min_points,
                scale_per_slice=workflow.fit_scale_per_slice,
                num_workers=1,
            )[self.SHOT]

        for k, i in enumerate(idxs):
            assert np.isfinite(out.te_fit[k]).any(), f"slice {i}: Te fit all NaN"
            assert np.isfinite(out.ne_fit[k]).any(), f"slice {i}: ne fit all NaN"

        workflow._debug_plot_profiles(
            self.SHOT,
            ts_time[idxs],
            te_eV[idxs] / 1e3,
            ne_m3[idxs] / 1e20,
            rho_ts[idxs],
            out.te_fit,
            out.ne_fit,
            GP_FIT_PLOT_DIR,
        )
        assert (GP_FIT_PLOT_DIR / f"{self.SHOT}_ts_gp_fit.pdf").exists()
