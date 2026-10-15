"""A diverged transport training rollout costs only its own sample (TransportPredictorTRB training loss, popsim train_epoch)."""

import numpy as np
import pytest
from popsim.ml.dataloading import XarrayPreppedDataset
from popsim.ml.launch import build_train_run
from popsim.ml.trainer import train_epoch

from transport_study.tests.tuned_configs import write_hyperparam_tuned_configs
from transport_study.transport_transfer.transport_transfer_study import TransportStudy


@pytest.mark.slow
def test_diverged_segment_masked_not_skipped(synthetic_device_stores, tmp_path, monkeypatch):
    """One segment whose rollout goes non-finite is masked out of its step, and the rest of the batch still trains.

    An infinite heating power partway through the segment stands in for a rollout that diverged there.
    Its training loss has to stay non-finite, so popsim can find it and mask it.
    A finite stand-in loss leaves the gradient NaN through the rollout's own backward pass,
    popsim then skips the whole step, and an epoch of skipped steps aborts the run.
    """
    study = TransportStudy(
        TransportStudy.Config(
            study_name="test-training-divergence",
            working_dir_base=tmp_path,
            dataset_paths=synthetic_device_stores,
            target_device="mast",
            # A weighted case validates on the target test set, and popsim rejects a single-shot whole-episode set
            target_test_set_size=2,
            training_datasets=("cmod",),
            model_types=("transformer",),
            domain_adaptation_methods=("none", "weighted"),
            num_target_shots_options=(1,),
        )
    )
    # The synthetic shots are shorter than the study's training segments
    base_dataloader_config = study.base_dataloader_config
    monkeypatch.setattr(
        study,
        "base_dataloader_config",
        lambda case: {**base_dataloader_config(case), "segment_length_train": 20, "segment_overlap_train": 10},
    )
    write_hyperparam_tuned_configs(study)
    weighted_case = next(case for case in study.cases if case.domain_adaptation == "weighted")
    trainer, train_dl, _, _ = build_train_run(study.make_train_config(weighted_case))
    batch = next(iter(train_dl))
    assert len(batch) - batch.n_padded > 1
    sample_dim = batch.training_metadata.sample_dim
    time_dim = batch.training_metadata.time_dep_metadata.time_dim
    ds_diverged = batch.ds.copy(deep=True)
    ds_diverged["power_additional_MW"][{sample_dim: 0, time_dim: 10}] = np.inf
    batch_diverged = XarrayPreppedDataset(ds_diverged, batch.training_metadata, n_padded=batch.n_padded)

    # The study's runs average their weights, so train_epoch needs a horizon, any one will do here
    train_state, train_metrics = train_epoch(
        trainer.train_state,
        trainer.partition_fn,
        trainer.loss_fn,
        trainer.optimizer,
        [batch_diverged],
        average_horizon_steps=len(train_dl),
    )

    assert train_state.step == 1
    assert (train_metrics["train/nan_steps_masked"], train_metrics["train/nan_steps_skipped"]) == (1, 0)
    assert np.isfinite(float(train_metrics["train/loss"]))
