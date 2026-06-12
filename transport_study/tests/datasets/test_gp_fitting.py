"""Tests for the GP fitting batch format, worker, and dispatcher logic.

The real cluster interaction (srunx submission, rsync) needs a cluster, so the
dispatcher loop is exercised against a fake backend; everything else here runs
exactly the code the cluster path uses: the npz round-trip, deterministic batch
planning for restart safety, and the fitting math the worker executes.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

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
