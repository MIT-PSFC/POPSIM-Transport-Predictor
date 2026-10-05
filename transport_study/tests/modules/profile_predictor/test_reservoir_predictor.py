"""Tests for the reservoir profile predictor.

The reservoir weights are drawn once at init and must stay frozen: only the
readout network trains. These pin that contract plus the reservoir's own
spectral-radius scaling and determinism.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from popsim.ml import TrainConfig
from popsim.ml.launch import launch_train

from transport_study.config import RHO_GRID, StudyConfig, load_config
from transport_study.modules.normalization import CoralFeatureNormalizer
from transport_study.modules.profile_predictor.module import (
    N_NN_INPUTS,
    Inputs,
    ProfilePredictorReservoir,
)
from transport_study.modules.profile_predictor.trb import ProfilePredictorTRB
from transport_study.orchestration.organize_data import PROFILE_TARGET_VARS
from transport_study.profile_transfer.profile_study import PROFILE_INPUT_VARS
from transport_study.tests.sample_data import SAMPLE_DIR, requires_sample_data

RESERVOIR_SIZE = 32


@pytest.fixture(autouse=True)
def loaded_config():
    return load_config(
        StudyConfig(
            study_name="test_reservoir_predictor",
            dataset_paths={
                "cmod-low": SAMPLE_DIR / "cmod-low1.nc",
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


@pytest.mark.slow
@requires_sample_data
def test_reservoir_training_smoke():
    """End-to-end launch_train on the sample dataset."""
    train_config = TrainConfig(
        project="test_reservoir_training_smoke",
        train_run_builder="transport_study.modules.profile_predictor.trb.ProfilePredictorTRB",
        max_epochs=2,
        epochs_per_val=1,
        checkpoint_dir=None,
        dataloader_config={
            "input_vars": PROFILE_INPUT_VARS,
            "target_vars": [*PROFILE_TARGET_VARS, "ds_source_idx"],
            "training_data": {"sources": ["cmod-low"], "exnihilo": False},
            "batch_size": None,
        },
        model_init_config={
            "model_type": "reservoir",
            # model_init indexes this strictly, the study normally supplies it
            "data_normalization": "physics-coral",
            "reservoir_size": RESERVOIR_SIZE,
            "spectral_radius": 0.9,
            "input_scaling": 0.5,
            "leak_rate": 1.0,
            "n_steps": 10,
            "prng_seed": 42,
        },
        loss_config={
            "huber_delta": 0.5,
            "gradient_weight": 0.1,
            "huber_delta_grad": 5.0,
        },
        optimizer_config={
            "lr0": 3e-3,
            "transition_steps": 500,
            "decay_rate": 0.5,
            "lrf": 5e-4,
            "weight_decay": 2e-4,
        },
    )

    launch_train(train_config)
