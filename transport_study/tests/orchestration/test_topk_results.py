"""Tests for orchestration/topk_results.py, the top-K checkpoint result averaging.

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
    aggregate_topk_results,
    compute_topk_study_results,
)
from transport_study.tests.stubs import StubCase, StubConfig

CKPT_STD_012 = float(np.std([0.0, 1.0, 2.0]))


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


def test_aggregate_error_vars_are_checkpoint_means():
    """Build per-step datasets for 3 fake checkpoints whose error variables
    differ by known offsets. aggregate_topk_results must replace every data
    variable whose name contains 'error' with the elementwise mean over the
    3 checkpoints, under the ORIGINAL variable name (downstream consumers
    like _summarize_case_errors read these names unchanged), and the values
    must match a hand-computed mean."""
    out = aggregate_topk_results(_three_checkpoint_results(), best_step=25)
    expected_ts = np.arange(6, dtype=float).reshape(2, 3) + 1.0
    assert np.allclose(out["error_abs_ts"].values, expected_ts)
    assert np.allclose(out["error_rel_shot"].values, [2.0, 3.0])


def test_aggregate_adds_ckpt_std_companions():
    """Same setup as the mean test: every error variable gains a
    <name>_ckpt_std companion holding the per-point std over checkpoints
    (ddof 0), and non-error variables gain no companion."""
    out = aggregate_topk_results(_three_checkpoint_results(), best_step=25)
    assert np.allclose(out["error_abs_ts_ckpt_std"].values, CKPT_STD_012)
    assert np.allclose(out["error_rel_shot_ckpt_std"].values, CKPT_STD_012)
    assert "energy_mhd_MJ_pred_ckpt_std" not in out.data_vars
    assert "energy_mhd_MJ_targ_ckpt_std" not in out.data_vars


def test_aggregate_pred_and_targ_from_best_checkpoint():
    """Give each fake checkpoint a distinct energy_mhd_MJ_pred. The aggregated
    dataset's energy_mhd_MJ_pred and energy_mhd_MJ_targ must be bit-identical to the
    best checkpoint's (predictions are never averaged across checkpoints,
    a mean trajectory would be smoother than any actual model), where
    'best' is the step passed as best_step, not the lowest or highest."""
    per_step = _three_checkpoint_results()
    out = aggregate_topk_results(per_step, best_step=25)
    assert (out["energy_mhd_MJ_pred"].values == 25.0).all()
    assert (out["energy_mhd_MJ_targ"].values == per_step[25]["energy_mhd_MJ_targ"].values).all()


def test_aggregate_records_checkpoint_epochs_in_attrs(tmp_path: Path):
    """The aggregated dataset attrs must carry result_checkpoint_epochs
    (sorted list of retained epochs) and result_checkpoint_best_epoch, and
    both must survive a to_netcdf round trip."""
    out = aggregate_topk_results(_three_checkpoint_results(), best_step=25)
    assert out.attrs["result_checkpoint_epochs"] == [10, 25, 40]
    assert out.attrs["result_checkpoint_best_epoch"] == 25
    nc_path = tmp_path / "result.nc"
    out.to_netcdf(nc_path)
    back = xr.load_dataset(nc_path)
    assert list(back.attrs["result_checkpoint_epochs"]) == [10, 25, 40]
    assert int(back.attrs["result_checkpoint_best_epoch"]) == 25


def test_aggregate_nan_handling():
    """A timeslice that is NaN in one checkpoint but finite in the others
    averages over the finite ones (skipna), while a timeslice that is NaN
    in every checkpoint (the padded tail) stays NaN in both the mean and
    the _ckpt_std companion."""
    per_step = _three_checkpoint_results()
    for step in per_step:
        per_step[step]["error_abs_ts"].values[0, 0] = np.nan
    per_step[40]["error_abs_ts"].values[1, 2] = np.nan
    out = aggregate_topk_results(per_step, best_step=25)
    assert np.isnan(out["error_abs_ts"].values[0, 0])
    assert np.isnan(out["error_abs_ts_ckpt_std"].values[0, 0])
    # Element (1, 2) is 5.0 + offset, finite only in the offset 0 and 2 checkpoints
    assert np.isclose(out["error_abs_ts"].values[1, 2], 6.0)


def test_compute_topk_single_checkpoint_passthrough():
    """With a checkpoint manager retaining a single step (or max_to_keep 1),
    compute_topk_study_results must return result_dict['test/study_results']
    unchanged without restoring anything, recovering the old best-only
    behavior byte for byte."""
    best_ds = _result_ds(0.0, 7.0)
    trainer = SimpleNamespace(checkpoint_manager=SimpleNamespace(all_steps=lambda: [7]))
    out = compute_topk_study_results(trainer, None, None, {"test/study_results": best_ds})
    assert out is best_ds


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


def test_config_lock_includes_num_result_checkpoints(tmp_path: Path):
    """is_compatible must reject two otherwise identical study configs that
    differ in num_result_checkpoints: their result files mix top-K means
    with single-checkpoint draws and must not share a study directory."""
    cfg = StubConfig(
        study_name="lock_study",
        working_dir_base=tmp_path,
        dataset_paths={},
        target_device="cmod",
        num_result_checkpoints=10,
    )
    assert cfg.is_compatible(cfg.model_copy())
    other = cfg.model_copy(update={"num_result_checkpoints": 5})
    assert not cfg.is_compatible(other)
    assert not other.is_compatible(cfg)


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
