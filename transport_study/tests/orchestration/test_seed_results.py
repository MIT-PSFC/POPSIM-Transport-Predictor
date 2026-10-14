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
from transport_study.modules.trb_utils import restore_from_checkpoint
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


def _stub_train_config(**updates) -> TrainConfig:
    """A TrainConfig carrying only what launch_train reads, with updates on top."""
    config_fields = {
        "project": "stub_project",
        "train_run_builder": "stub.module.StubTRB",
        "max_epochs": 3,
        "epochs_per_val": 1,
        "dataloader_config": {"shuffle_seed": 0},
        "model_init_config": {"prng_seed": 42},
        "loss_config": {},
        "optimizer_config": {},
    }
    return TrainConfig(**{**config_fields, **updates})


def _capture_launch_train(monkeypatch) -> dict[str, TrainConfig]:
    """Swap the serial launch_train for one recording its train config under "train"."""
    captured: dict[str, TrainConfig] = {}

    def fake_launch_train(train_config):
        captured["train"] = train_config
        return None, None, None, None, None

    monkeypatch.setattr(study_module, "launch_train", fake_launch_train)
    return captured


def test_launch_train_seeds_its_init_shuffle_and_checkpoint_dir(make_stub_study, monkeypatch):
    """Each seed trains its own initialization and shuffle into its own checkpoint dir, keeping one checkpoint,
    and sweep trials keep one checkpoint too."""
    case = StubCase("case.stub")
    study = make_stub_study([case], num_seeds=4)
    base_config = _stub_train_config()
    monkeypatch.setattr(study, "make_train_config", lambda _case: base_config)
    captured = _capture_launch_train(monkeypatch)
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


def test_restored_case_merge_keeps_each_seeds_best_checkpoint(make_stub_study):
    """The merge of a case other cases restore seed by seed (keeps_seed_checkpoints).

    Before the merge:
    - each seed dir holds its best checkpoint (epoch 2) and its resume checkpoint (epoch 3)
    - seed 1 has the lower validation loss, so it is the best seed

    The merge then:
    1. copies seed 1's epoch 2 checkpoint up into the case dir, where unseeded restores read
    2. deletes epoch 3 from each seed dir, so each keeps only its own best checkpoint

    After it the case dir holds 2/, seed_0/2/ and seed_1/2/,
    and a restore of the case dir reads 2/ and skips the seed dirs.
    """
    case = StubCase("case.restored", keeps_seed_checkpoints=True)
    study = make_stub_study([case], num_seeds=2)
    seed_best_epochs = {
        0: _train_seed(study, case, 0, {1: 0.9, 2: 0.4, 3: 0.6}),
        1: _train_seed(study, case, 1, {1: 0.8, 2: 0.3, 3: 0.5}),
    }

    study.merge_seed_results(case)

    # 1. The best seed's best checkpoint sits in the case dir beside the seed dirs
    model_dir = study.trained_model_dir(case)
    assert sorted(path.name for path in model_dir.iterdir()) == [str(seed_best_epochs[1]), "seed_0", "seed_1"]
    # 2. Each seed dir keeps only its own best checkpoint
    for seed, best_epoch in seed_best_epochs.items():
        assert [path.name for path in study.seed_model_dir(case, seed).iterdir()] == [str(best_epoch)]
    # The restore reads the copied-up checkpoint, seed 1's weights
    restored = restore_model(create_default_checkpoint_manager(model_dir), _TinyModel(w=jnp.zeros(2)))
    assert float(restored.w[0]) == 10.0 * 1 + seed_best_epochs[1]


def test_seed_restores_same_seed_of_twin_and_submodules(make_stub_study, monkeypatch):
    """Seed s of a case restores seed s of its transfer_pretrain twin and of each submodule prereq, not their best seed.

    Before the launch:
    - the twin and the submodule each train 2 seeds, with weights 10 x seed + epoch
    - seed 0's best checkpoint is epoch 2 (weights 2), seed 1's is epoch 1 (weights 11)
    - seed 1 has the lower validation loss, so each merge copies seed 1's checkpoint up into the case dir
    - so each case dir holds 1/ (weights 11), seed_0/2/ (weights 2) and seed_1/1/ (weights 11)
    - the consumer's train config points transfer_checkpoint and the submodule checkpoint_dir at the two case dirs,
      as make_train_config builds them

    The launch of the consumer's seed 0 then:
    1. repoints transfer_checkpoint at the twin's seed_0/
    2. repoints the submodule checkpoint_dir at the submodule's seed_0/

    Restoring from both repointed dirs gives weights 2, seed 0's, never 11, the merged best.
    """
    twin = StubCase("case.twin", keeps_seed_checkpoints=True)
    submodule = StubCase("case.submodule", keeps_seed_checkpoints=True)
    case = StubCase("case.finetune")
    study = make_stub_study([twin, submodule, case], num_seeds=2)
    for prereq in (twin, submodule):
        seed_0_best_epoch = _train_seed(study, prereq, 0, {1: 0.9, 2: 0.4})
        _train_seed(study, prereq, 1, {1: 0.3, 2: 0.5})
        study.merge_seed_results(prereq)
    submodule_config = _stub_train_config(checkpoint_dir=str(study.trained_model_dir(submodule)))
    model_init_config = {
        "prng_seed": 42,
        "transfer_checkpoint": str(study.trained_model_dir(twin)),
        "submodules": {"p_oh_predictor": submodule_config},
    }
    base_config = _stub_train_config(model_init_config=model_init_config)
    monkeypatch.setattr(study, "make_train_config", lambda _case: base_config)
    captured = _capture_launch_train(monkeypatch)

    study.launch_train(case, 0, enable_parallelism=False)

    seeded_init_config = captured["train"].model_init_config
    template = _TinyModel(w=jnp.zeros(2))
    # 1. The twin restore reads seed 0's checkpoint
    restored_twin = restore_from_checkpoint(template, seeded_init_config["transfer_checkpoint"])
    # 2. The submodule restore reads seed 0's checkpoint
    seeded_submodule_dir = seeded_init_config["submodules"]["p_oh_predictor"].checkpoint_dir
    restored_submodule = restore_from_checkpoint(template, seeded_submodule_dir)
    assert float(restored_twin.w[0]) == seed_0_best_epoch
    assert float(restored_submodule.w[0]) == seed_0_best_epoch
