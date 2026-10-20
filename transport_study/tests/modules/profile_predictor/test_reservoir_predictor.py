"""Tests for the reservoir profile predictor.

The reservoir weights are drawn once at init and must stay frozen:
only the readout network trains.
These pin that contract plus the reservoir's own spectral-radius scaling and determinism,
and the per-sample state cache model_init attaches to the loaders.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from popsim.ml.eval import batched_model_eval_and_loss
from popsim.ml.launch import launch_train

from transport_study import RADIAL_DIM
from transport_study.config import RHO_GRID, StudyConfig, load_config, reset_config
from transport_study.modules.normalization import CoralFeatureNormalizer
from transport_study.modules.profile_predictor.module import (
    N_NN_INPUTS,
    RESERVOIR_FINGERPRINT_VAR,
    RESERVOIR_STATE_VAR,
    Inputs,
    ProfilePredictorReservoir,
)
from transport_study.modules.profile_predictor.trb import ProfilePredictorTRB
from transport_study.profile_transfer.profile_study import ProfileStudy
from transport_study.tests.sample_data import SAMPLE_DIR, requires_sample_data

RESERVOIR_SIZE = 32


@pytest.fixture(autouse=True)
def loaded_config():
    return load_config(
        StudyConfig(
            study_name="test_reservoir_predictor",
            dataset_paths={
                "cmod-low1": SAMPLE_DIR / "cmod-low1.nc",
                "cmod-high": SAMPLE_DIR / "cmod-high.nc",
            },
            target_device="cmod-high",
        )
    )


@pytest.fixture
def make_module():
    def _make(**overrides) -> ProfilePredictorReservoir:
        return ProfilePredictorReservoir(
            reservoir_size=RESERVOIR_SIZE,
            rhogrid=RHO_GRID,
            key=jax.random.PRNGKey(0),
            normalizer=CoralFeatureNormalizer.identity(1, N_NN_INPUTS),
            **overrides,
        )

    return _make


def _inputs() -> Inputs:
    return Inputs(
        ip_MA=1.0,
        b0=5.56,
        b_geo=5.4,
        beta_tor_norm=1.2,
        n_e_line_average_1e20=1.5,
        geometric_axis_r=0.68,
        minor_radius=0.22,
        elongation=1.6,
        triangularity_upper=0.4,
        triangularity_lower=0.5,
        rho=jnp.asarray(RHO_GRID),
        ds_source_idx=jnp.asarray(0.0),
    )


def test_reservoir_output_shapes(make_module):
    out = make_module()(_inputs())
    for channel in (out.ne, out.te):
        assert channel.shape == RHO_GRID.shape
        assert np.all(np.isfinite(channel.values))


def test_reservoir_spectral_radius(make_module):
    module = make_module(spectral_radius=0.8)
    eig_max = float(np.max(np.abs(np.linalg.eigvals(np.asarray(module.w_res)))))
    assert eig_max == pytest.approx(0.8, rel=1e-5)


def test_reservoir_state_deterministic_and_input_sensitive(make_module):
    module = make_module()
    x1 = jnp.arange(10, dtype=jnp.float32) * 0.1
    x2 = x1 + 0.5
    assert jnp.allclose(module.reservoir_state(x1), module.reservoir_state(x1))
    assert not jnp.allclose(module.reservoir_state(x1), module.reservoir_state(x2))


def test_reservoir_trainable_getter_only_readout(make_module):
    """The reservoir tensors must never appear in the trainable selection."""
    module = make_module()
    trainable = ProfilePredictorTRB.get_trainable_getter({"model_type": "reservoir", "domain_adaptation": None})(module)

    trainable_ids = {id(leaf) for leaf in trainable}
    nn_ids = {id(leaf) for leaf in jax.tree.leaves(module.nn)}
    reservoir_ids = {id(leaf) for leaf in jax.tree.leaves((module.w_in, module.w_res, module.res_bias))}
    assert trainable_ids == nn_ids
    assert not (trainable_ids & reservoir_ids)


def _sample_reservoir_study(study_name: str, working_dir_base) -> ProfileStudy:
    """A one-case reservoir study on the sample dataset, loading its own config in place of the module fixture's."""
    reset_config()
    cfg = ProfileStudy.Config(
        study_name=study_name,
        working_dir_base=working_dir_base,
        dataset_paths={
            "cmod-low1": SAMPLE_DIR / "cmod-low1.nc",
            "cmod-high": SAMPLE_DIR / "cmod-high.nc",
        },
        target_device="cmod-high",
        max_ds_size=10,
        max_epochs=2,
        epochs_per_val=1,
        model_types=("reservoir",),
        data_normalization_methods=("physics-coral",),
        hyperparam_data_normalization="physics-coral",
        training_datasets=("cmod-low1",),
        domain_adaptation_methods=(None,),
        num_target_shots_options=(0,),
        target_test_set_size=4,
    )
    return ProfileStudy(cfg)


@pytest.fixture
def cached_reservoir_run(tmp_path):
    """The train config, loaders and module model_init returns for the sample reservoir case, the loaders holding the cache."""
    study = _sample_reservoir_study("test_reservoir_state_cache", tmp_path)
    (case,) = study.cases
    train_config = study.make_train_config(case)
    _, train_dl, val_dl, _ = ProfilePredictorTRB.get_dataloaders(train_config.dataloader_config)
    module = ProfilePredictorTRB.model_init(train_dl, train_config.model_init_config)
    return train_config, train_dl, val_dl, module


@requires_sample_data
def test_cached_states_match_uncached_losses(cached_reservoir_run):
    """The cached states give the training and validation losses the reservoir itself gives."""
    train_config, train_dl, val_dl, module = cached_reservoir_run
    train_loss_fn = ProfilePredictorTRB.get_loss_fn(train_config.loss_config)
    val_loss_fn = ProfilePredictorTRB.get_val_loss_fn(train_config.val_eval_suite_config["loss_config"])
    for loader, loss_fn in ((train_dl, train_loss_fn), (val_dl, val_loss_fn)):
        batch = next(iter(loader))
        inputs, targets = batch.get_inputs_and_targets()
        assert RESERVOIR_STATE_VAR in inputs
        inputs_uncached = inputs.drop_vars([RESERVOIR_STATE_VAR, RESERVOIR_FINGERPRINT_VAR])
        losses_cached = batched_model_eval_and_loss(module, loss_fn, inputs, targets)
        losses_uncached = batched_model_eval_and_loss(module, loss_fn, inputs_uncached, targets)
        np.testing.assert_allclose(losses_cached, losses_uncached, rtol=1e-10)


@requires_sample_data
def test_cached_states_refuse_other_reservoir(cached_reservoir_run):
    """Another seed's reservoir evaluating the cache raises, as after Study.restore_trainer restores the best seed."""
    train_config, train_dl, _, module = cached_reservoir_run
    other_seed_module = ProfilePredictorReservoir(
        reservoir_size=train_config.model_init_config["reservoir_size"],
        rhogrid=np.asarray(train_dl.ds[RADIAL_DIM]),
        key=jax.random.PRNGKey(train_config.model_init_config["prng_seed"] + 1),
        normalizer=module.normalizer,
    )
    loss_fn = ProfilePredictorTRB.get_loss_fn(train_config.loss_config)
    batch = next(iter(train_dl))
    inputs, targets = batch.get_inputs_and_targets()
    with pytest.raises(RuntimeError, match="made by another reservoir"):
        batched_model_eval_and_loss(other_seed_module, loss_fn, inputs, targets)


@pytest.mark.slow
@requires_sample_data
def test_reservoir_training_smoke(tmp_path):
    """End-to-end launch_train of the production reservoir config on the sample dataset."""
    study = _sample_reservoir_study("test_reservoir_training_smoke", tmp_path)
    (case,) = study.cases
    launch_train(study.make_train_config(case))
