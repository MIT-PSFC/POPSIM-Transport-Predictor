"""Tests for orchestration/seed_results.py and the seed runs of Study.launch_train and Study.merge_seed_results.

Aggregation is driven with small hand-built xr.Datasets shaped like the
study_results outputs (data vars error_abs_ts / error_rel_shot / energy_mhd_MJ_pred /
energy_mhd_MJ_targ over dims shot, time_idx), no training needed.
The merge tests save real orbax checkpoints on tmp_path.
"""

import os
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
from transport_study.orchestration.seed_results import (
    BEST_SEED_ATTR,
    SEED_BEST_EPOCH_VAR,
    SEED_DIM,
    SEED_VAL_LOSS_VAR,
    aggregate_seed_results,
    seed_result,
    seed_statistics,
)
from transport_study.orchestration.study import write_netcdf_atomic
from transport_study.tests.stubs import StubCase, with_seed_attrs


class _TinyModel(eqx.Module):
    w: jnp.ndarray


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


def test_aggregate_stacks_error_vars_over_seeds(tmp_path: Path):
    """Every error variable keeps each seed's values on SEED_DIM in seed order, as float32.

    The best seed is the one with the lowest validation loss, a NaN loss never wins.
    The predictions and targets stay the best seed's (a mean trajectory would be smoother than any model),
    and the seed coord, each seed's best epoch and loss and the best seed survive a netcdf round trip.
    """
    per_seed = {
        0: with_seed_attrs(_result_ds(0.0, 10.0), val_loss=0.5, best_epoch=12),
        1: with_seed_attrs(_result_ds(1.0, 40.0), val_loss=np.nan, best_epoch=30),
        2: with_seed_attrs(_result_ds(2.0, 25.0), val_loss=0.2, best_epoch=7),
    }
    out = aggregate_seed_results(per_seed)
    assert out[SEED_DIM].values.tolist() == [0, 1, 2]
    assert out.attrs[BEST_SEED_ATTR] == 2
    assert out["error_abs_ts"].dtype == np.float32
    np.testing.assert_allclose(out["error_abs_ts"].sel({SEED_DIM: 1}).values, np.arange(6).reshape(2, 3) + 1.0)
    np.testing.assert_allclose(out["error_rel_shot"].sel({SEED_DIM: 2}).values, [3.0, 4.0])
    assert SEED_DIM not in out["energy_mhd_MJ_pred"].dims
    assert (out["energy_mhd_MJ_pred"].values == 25.0).all()
    assert out[SEED_BEST_EPOCH_VAR].values.tolist() == [12, 30, 7]

    nc_path = tmp_path / "result.nc"
    out.to_netcdf(nc_path)
    back = xr.load_dataset(nc_path)
    assert back[SEED_DIM].values.tolist() == [0, 1, 2]
    assert int(back.attrs[BEST_SEED_ATTR]) == 2
    np.testing.assert_allclose(back[SEED_VAL_LOSS_VAR].values, [0.5, np.nan, 0.2])


def test_seed_statistics_are_best_mean_and_std_of_the_case_scores():
    """One case score per seed gives the seed mean, the best seed's score and the sample std over seeds (ddof 1).
    A seed whose score is NaN (nothing finite to score) is skipped by the mean and std."""
    per_seed = xr.Dataset({"score": (SEED_DIM, [1.0, 2.0, 4.0, np.nan])}, coords={SEED_DIM: [0, 1, 2, 3]})
    out = seed_statistics(per_seed, best_seed=1)
    assert np.isclose(out["score"], 7.0 / 3.0)
    assert np.isclose(out["score_best"], 2.0)
    assert np.isclose(out["score_seed_std"], np.std([1.0, 2.0, 4.0], ddof=1))


def test_seed_result_reads_the_best_checkpoint():
    """The trainer's own best-checkpoint test result comes back with that checkpoint's epoch and validation loss."""
    best_ds = _result_ds(0.0, 7.0)
    step_losses = {7: 0.1, 9: 0.5}
    manager = SimpleNamespace(metrics=lambda step: {"loss": step_losses[step]}, best_step=lambda: 7)
    trainer = SimpleNamespace(checkpoint_manager=manager)

    out = seed_result(trainer, {"test/study_results": best_ds})

    aggregated = aggregate_seed_results({4: out})
    assert aggregated.attrs[BEST_SEED_ATTR] == 4
    assert aggregated[SEED_BEST_EPOCH_VAR].values.tolist() == [7]
    np.testing.assert_allclose(aggregated[SEED_VAL_LOSS_VAR].values, [0.1])
    np.testing.assert_allclose(aggregated["error_abs_ts"].sel({SEED_DIM: 4}).values, best_ds["error_abs_ts"].values)


def test_launch_train_seeds_its_init_shuffle_and_checkpoint_dir(make_stub_study, monkeypatch):
    """Each seed trains its own initialization and shuffle into its own checkpoint dir, keeping one checkpoint,
    and sweep trials keep one checkpoint too."""
    case = StubCase("case.stub")
    study = make_stub_study([case], num_seeds=4)
    base_config = TrainConfig(
        project="stub_project",
        train_run_builder="stub.module.StubTRB",
        max_epochs=3,
        epochs_per_val=1,
        dataloader_config={"shuffle_seed": 0},
        model_init_config={"prng_seed": 42},
        loss_config={},
        optimizer_config={},
    )
    monkeypatch.setattr(study, "make_train_config", lambda _case: base_config)
    captured: dict[str, TrainConfig] = {}

    def fake_launch_train(train_config):
        captured["train"] = train_config
        return None, None, None, None, None

    monkeypatch.setattr(study_module, "launch_train", fake_launch_train)
    study.launch_train(case, 3, enable_parallelism=False)
    seeded_config = captured["train"]
    assert seeded_config.model_init_config["prng_seed"] == 3
    assert seeded_config.dataloader_config["shuffle_seed"] == 3
    assert seeded_config.checkpoint_dir == str(study.seed_model_dir(case, 3))
    assert seeded_config.checkpoint_max_to_keep == 1

    def fake_launch_agent(train_config, sweep_id, kwargs_agent=None):
        captured["sweep"] = train_config

    monkeypatch.setattr(study_module, "get_sweep_id", lambda _project: "stub-sweep-id")
    monkeypatch.setattr(study_module, "launch_agent", fake_launch_agent)
    study.launch_sweep(case, enable_parallelism=False)
    assert captured["sweep"].checkpoint_max_to_keep == 1


def _train_seed(study, case, seed: int, epoch_losses: dict[int, float]) -> int:
    """Save one seed's checkpoints as training would, then write its seed result. Returns its best epoch."""
    manager = create_default_checkpoint_manager(study.seed_model_dir(case, seed), max_to_keep=1)
    for epoch, loss in epoch_losses.items():
        # Each checkpoint's weights tag the seed and the epoch
        weights = jnp.full((2,), 10.0 * seed + epoch)
        state = TrainState(step=epoch, epoch=epoch, model=_TinyModel(w=weights), opt_state={"dummy": jnp.zeros(1)})
        save_train_state(state, manager, loss=loss)
    best_epoch = min(epoch_losses, key=epoch_losses.__getitem__)
    seed_ds = with_seed_attrs(_result_ds(float(seed), float(seed)), val_loss=epoch_losses[best_epoch], best_epoch=best_epoch)
    write_netcdf_atomic(seed_ds, study.seed_result_path(case, seed))
    return best_epoch


def test_merge_keeps_only_the_best_seeds_best_checkpoint(make_stub_study):
    """The best seed's best checkpoint moves up into the case's model dir, where every restore reads,
    the seed dirs and seed results go, and the case result is written."""
    case = StubCase("case.merge")
    study = make_stub_study([case], num_seeds=2)
    _train_seed(study, case, 0, {1: 0.9, 2: 0.4, 3: 0.6})
    best_epoch = _train_seed(study, case, 1, {1: 0.8, 2: 0.3, 3: 0.5})

    study.merge_seed_results(case)

    result_ds = xr.load_dataset(study.result_path(case))
    assert result_ds.attrs[BEST_SEED_ATTR] == 1
    model_dir = study.trained_model_dir(case)
    assert [path.name for path in model_dir.iterdir()] == [str(best_epoch)]
    assert not study.seed_result_path(case, 0).exists()
    restored = restore_model(create_default_checkpoint_manager(model_dir), _TinyModel(w=jnp.zeros(2)))
    assert float(restored.w[0]) == 10.0 * 1 + best_epoch


def test_interrupted_merge_reruns_cleanly(make_stub_study):
    """A merge cut short after moving the best checkpoint up finishes on the next pass."""
    case = StubCase("case.rerun")
    study = make_stub_study([case], num_seeds=2)
    best_epoch = _train_seed(study, case, 0, {1: 0.2, 2: 0.7})
    _train_seed(study, case, 1, {1: 0.9, 2: 0.8})
    os.rename(study.seed_model_dir(case, 0) / str(best_epoch), study.trained_model_dir(case) / str(best_epoch))

    study.merge_seed_results(case)

    assert study.result_path(case).exists()
    assert [path.name for path in study.trained_model_dir(case).iterdir()] == [str(best_epoch)]
