"""Tests for orchestration/topk_results.py, the top-K checkpoint results.

Aggregation is driven with small hand-built xr.Datasets shaped like the
study_results outputs (data vars error_abs_ts / error_rel_shot / energy_mhd_MJ_pred /
energy_mhd_MJ_targ over dims shot, time_idx), no training needed. The checkpoint
retention test exercises the real orbax manager on tmp_path.
"""

from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import xarray as xr
from popsim.ml import TrainConfig
from popsim.ml.checkpointing import (
    TrainState,
    create_default_checkpoint_manager,
    restore_model,
    save_train_state,
)

import transport_study.orchestration.study as study_module
from transport_study.orchestration.topk_results import (
    BEST_EPOCH_ATTR,
    CKPT_DIM,
    aggregate_topk_results,
    compute_topk_study_results,
    topk_checkpoint_steps,
    topk_statistics,
)
from transport_study.tests.stubs import StubCase


def _result_ds(error_offset: float, pred_value: float) -> xr.Dataset:
    """A study_results-shaped dataset whose error variables sit at a known offset."""
    err_ts = np.arange(6, dtype=float).reshape(2, 3) + error_offset
    return xr.Dataset(
        data_vars={
            "error_abs_ts": (("shot", "time_idx"), err_ts),
            "error_rel_shot": ("shot", np.array([1.0, 2.0]) + error_offset),
            "energy_mhd_MJ_pred": (("shot", "time_idx"), np.full((2, 3), pred_value)),
            "energy_mhd_MJ_targ": (("shot", "time_idx"), np.ones((2, 3))),
        },
        coords={"shot": ["a", "b"], "time_idx": np.arange(3)},
    )


def _three_checkpoint_results() -> dict[int, xr.Dataset]:
    """Three fake checkpoints with error offsets 0, 1, 2 and epoch-tagged predictions."""
    return {10: _result_ds(0.0, 10.0), 40: _result_ds(1.0, 40.0), 25: _result_ds(2.0, 25.0)}


def test_aggregate_stacks_error_vars_over_checkpoints(tmp_path: Path):
    """Every error variable keeps each checkpoint's values on CKPT_DIM, epochs in order, as float32,
    the predictions and targets stay the best checkpoint's (a mean trajectory would be smoother than any model),
    and the ckpt coord and best epoch survive a netcdf round trip."""
    per_step = _three_checkpoint_results()
    out = aggregate_topk_results(per_step, best_step=25)
    assert out[CKPT_DIM].values.tolist() == [10, 25, 40]
    assert out["error_abs_ts"].dtype == np.float32
    np.testing.assert_allclose(out["error_abs_ts"].sel({CKPT_DIM: 40}).values, np.arange(6).reshape(2, 3) + 1.0)
    np.testing.assert_allclose(out["error_rel_shot"].sel({CKPT_DIM: 25}).values, [3.0, 4.0])
    assert CKPT_DIM not in out["energy_mhd_MJ_pred"].dims
    assert (out["energy_mhd_MJ_pred"].values == 25.0).all()

    nc_path = tmp_path / "result.nc"
    out.to_netcdf(nc_path)
    back = xr.load_dataset(nc_path)
    assert back[CKPT_DIM].values.tolist() == [10, 25, 40]
    assert int(back.attrs[BEST_EPOCH_ATTR]) == 25


def test_topk_statistics_are_best_mean_and_std_of_the_case_scores():
    """One case score per checkpoint gives the top-K mean, the best epoch's score and the std over checkpoints (ddof 0).
    A checkpoint whose score is NaN (nothing finite to score) is skipped by the mean and std."""
    per_ckpt = xr.Dataset({"score": (CKPT_DIM, [1.0, 2.0, 4.0, np.nan])}, coords={CKPT_DIM: [10, 25, 40, 55]})
    out = topk_statistics(per_ckpt, best_epoch=25)
    assert np.isclose(out["score"], 7.0 / 3.0)
    assert np.isclose(out["score_best"], 2.0)
    assert np.isclose(out["score_ckpt_std"], np.std([1.0, 2.0, 4.0]))


def test_compute_topk_single_checkpoint_keeps_the_ckpt_dim():
    """With max_to_keep 1 nothing is restored or re-evaluated,
    and the best result still comes back on a length-1 ckpt dim so every consumer reads one layout."""
    best_ds = _result_ds(0.0, 7.0)
    step_losses = {7: 0.1, 9: 0.5}
    manager = SimpleNamespace(all_steps=lambda: list(step_losses), metrics=lambda step: {"loss": step_losses[step]}, best_step=lambda: 7)
    trainer = SimpleNamespace(checkpoint_manager=manager)
    train_config = SimpleNamespace(checkpoint_max_to_keep=1)
    out = compute_topk_study_results(trainer, None, train_config, {"test/study_results": best_ds})
    assert out[CKPT_DIM].values.tolist() == [7]
    assert out.attrs[BEST_EPOCH_ATTR] == 7
    np.testing.assert_allclose(out["error_abs_ts"].sel({CKPT_DIM: 7}).values, best_ds["error_abs_ts"].values)


def test_topk_steps_exclude_latest_outside_top_k(tmp_path: Path):
    """The checkpoint dir keeps the best max_to_keep steps plus the latest one.
    Save 5 epochs through a max_to_keep=2 manager with the final epoch outside the best 2,
    and topk_checkpoint_steps must return only the 2 lowest-loss epochs.
    An unvalidated latest step (saved with no loss) must never rank either."""
    manager = create_default_checkpoint_manager(tmp_path / "ckpt", max_to_keep=2)
    for epoch, loss in {1: 0.9, 2: 0.2, 3: 0.4, 4: 0.3, 5: 0.8}.items():
        state = TrainState(step=epoch, epoch=epoch, model=_TinyModel(w=jnp.zeros(2)), opt_state={"dummy": jnp.zeros(1)})
        save_train_state(state, manager, loss=loss)
    assert sorted(manager.all_steps()) == [2, 4, 5]
    assert topk_checkpoint_steps(manager, 2) == [2, 4]

    state = TrainState(step=6, epoch=6, model=_TinyModel(w=jnp.zeros(2)), opt_state={"dummy": jnp.zeros(1)})
    save_train_state(state, manager, loss=None)
    assert 6 in manager.all_steps()
    assert topk_checkpoint_steps(manager, 2) == [2, 4]


def test_launch_train_wires_num_result_checkpoints(make_stub_study, monkeypatch):
    """Study.launch_train must copy config.num_result_checkpoints into the
    train config's checkpoint_max_to_keep, while launch_sweep forces
    checkpoint_max_to_keep to 1 regardless of the study config (sweep
    trials are compared on val loss only). Drive with StubStudy from
    tests/stubs.py and inspect the TrainConfig each path builds."""
    case = StubCase("case.stub")
    study = make_stub_study([case], num_result_checkpoints=7)
    base_config = TrainConfig(
        project="stub_project",
        train_run_builder="stub.module.StubTRB",
        max_epochs=3,
        epochs_per_val=1,
        dataloader_config={},
        model_init_config={},
        loss_config={},
        optimizer_config={},
    )
    monkeypatch.setattr(study, "make_train_config", lambda _case: base_config)
    captured: dict[str, TrainConfig] = {}

    def fake_launch_train(train_config):
        captured["train"] = train_config
        return None, None, None, None, None

    monkeypatch.setattr(study_module, "launch_train", fake_launch_train)
    study.launch_train(case, enable_parallelism=False)
    assert captured["train"].checkpoint_max_to_keep == 7

    def fake_launch_agent(train_config, sweep_id, kwargs_agent=None):
        captured["sweep"] = train_config

    monkeypatch.setattr(study_module, "get_sweep_id", lambda _project: "stub-sweep-id")
    monkeypatch.setattr(study_module, "launch_agent", fake_launch_agent)
    study.launch_sweep(case, enable_parallelism=False)
    assert captured["sweep"].checkpoint_max_to_keep == 1


class _TinyModel(eqx.Module):
    w: jnp.ndarray


def test_reopening_topk_dir_with_default_manager_preserves_checkpoints(tmp_path: Path):
    """Restore paths open checkpoint dirs with the default max_to_keep=1
    manager. Save 3 steps through a max_to_keep=3 manager, reopen the
    directory with create_default_checkpoint_manager defaults, restore, and
    verify all 3 steps still exist on disk afterwards (orbax deletes only
    on save) and best_step() picks the step with the lowest saved loss."""
    ckpt_dir = tmp_path / "ckpt"
    manager = create_default_checkpoint_manager(ckpt_dir, max_to_keep=3)
    for epoch, loss in {1: 0.9, 2: 0.2, 3: 0.5}.items():
        state = TrainState(
            step=epoch * 10,
            epoch=epoch,
            model=_TinyModel(w=jnp.full((2,), float(epoch))),
            opt_state={"dummy": jnp.zeros(1)},
        )
        save_train_state(state, manager, loss=loss)
    assert sorted(manager.all_steps()) == [1, 2, 3]

    reopened = create_default_checkpoint_manager(ckpt_dir)
    assert sorted(reopened.all_steps()) == [1, 2, 3]
    assert reopened.best_step() == 2
    template = _TinyModel(w=jnp.zeros((2,)))
    best = restore_model(reopened, template)
    assert float(best.w[0]) == 2.0
    at_step = restore_model(reopened, template, step=3)
    assert float(at_step.w[0]) == 3.0
    on_disk = sorted(int(p.name) for p in ckpt_dir.iterdir() if p.name.isdigit())
    assert on_disk == [1, 2, 3]
