import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from popsim.ml import TrainConfig
from popsim.ml.launch import launch_train
from torax import ToraxConfig
from torax import experimental as torax_experimental

from transport_study import RADIAL_DIM
from transport_study.config import RHO_GRID, StudyConfig, load_config
from transport_study.modules.profile_predictor.module import Inputs
from transport_study.modules.profile_predictor.torax_module import (
    TRANSPORT_COEFFICIENT_NAMES,
    bound_transport_coefficients,
    transport_provider_mapping,
    validate_transport_model_name,
)
from transport_study.modules.profile_predictor.train_configs import (
    PROFILE_PREDICTOR_TORAX_CONFIGS,
    TORAX_CONFIG_BASE,
    TORAX_TRANSPORT_BLOCKS,
)
from transport_study.modules.profile_predictor.trb import resolve_relaxation_overrides
from transport_study.orchestration.organize_data import PROFILE_TARGET_VARS
from transport_study.tests.sample_data import SAMPLE_DIR, requires_sample_data

TRANSPORT_MODELS = ("constant", "gyrobohm", "qlknn")

# TORAX runtime-param field -> the bounded NN coefficient that must land in it.
# gyrobohm applies one multiplier to both species
RUNTIME_FIELD_TO_COEFFICIENT = {
    "constant": {"chi_i": "chi_i", "chi_e": "chi_e", "D_e": "D_e", "V_e": "V_e"},
    "gyrobohm": {
        "chi_e_bohm_multiplier": "chi_bohm_multiplier",
        "chi_i_bohm_multiplier": "chi_bohm_multiplier",
        "chi_e_gyrobohm_multiplier": "chi_gyrobohm_multiplier",
        "chi_i_gyrobohm_multiplier": "chi_gyrobohm_multiplier",
        "D_face_c1": "D_face_c1",
        "D_face_c2": "D_face_c2",
        "V_face_coeff": "V_face_coeff",
    },
    "qlknn": {
        "ITG_flux_ratio_correction": "ITG_flux_ratio_correction",
        "ETG_correction_factor": "ETG_correction_factor",
        "collisionality_multiplier": "collisionality_multiplier",
    },
}


def _torax_config(transport: dict) -> ToraxConfig:
    config_dict = copy.deepcopy(TORAX_CONFIG_BASE)
    config_dict["transport"] = copy.deepcopy(transport)
    return ToraxConfig.from_dict(config_dict)


def _batch_inputs() -> Inputs:
    """Two C-Mod-like timeslices differing in Ip, beta_tor_norm, and density."""
    n_batch = 2
    return Inputs(
        ip_MA=jnp.array([1.0, 0.8]),
        b0=jnp.full(n_batch, 5.56),
        b_geo=jnp.full(n_batch, 5.4),
        beta_tor_norm=jnp.array([1.2, 0.9]),
        n_e_line_average_1e20=jnp.array([1.5, 1.2]),
        geometric_axis_r=jnp.full(n_batch, 0.68),
        minor_radius=jnp.full(n_batch, 0.22),
        elongation=jnp.full(n_batch, 1.6),
        triangularity_upper=jnp.full(n_batch, 0.4),
        triangularity_lower=jnp.full(n_batch, 0.5),
        ds_source_idx=jnp.zeros(n_batch),
        rho=jnp.tile(jnp.asarray(RHO_GRID), (n_batch, 1)),
    )


@pytest.mark.parametrize("transport_model", TRANSPORT_MODELS)
def test_transport_provider_mapping_reaches_runtime_params(transport_model):
    """Every NN transport coefficient lands in the TORAX runtime params of the single core model.

    Regression test for the TORAX transport schema:
    the config must validate, validate_transport_model_name must accept it,
    and each provider path in transport_provider_mapping must resolve and replace the placeholder.
    """
    torax_config = _torax_config(TORAX_TRANSPORT_BLOCKS[transport_model])
    validate_transport_model_name(torax_config, transport_model)
    provider = torax_experimental.make_step_fn(torax_config).runtime_params_provider

    n_coefficients = len(TRANSPORT_COEFFICIENT_NAMES[transport_model])
    raw_outputs = 0.5 + 0.37 * jnp.arange(n_coefficients)
    coeffs = bound_transport_coefficients(transport_model, raw_outputs)
    mapping = transport_provider_mapping(transport_model, coeffs)
    updated_provider = provider.update_provider_from_mapping(mapping)

    params_placeholder = provider(t=0.0).transport.core_transport_model_params[transport_model]
    params_updated = updated_provider(t=0.0).transport.core_transport_model_params[transport_model]
    field_to_coefficient = RUNTIME_FIELD_TO_COEFFICIENT[transport_model]
    assert {key.rsplit(".", 1)[1] for key in mapping} == set(field_to_coefficient)
    for field, coefficient in field_to_coefficient.items():
        expected = float(np.squeeze(np.asarray(coeffs[coefficient])))
        updated = np.asarray(getattr(params_updated, field))
        placeholder = np.asarray(getattr(params_placeholder, field))
        # Constant-model coefficients are flat radial profiles, every face carries the value
        np.testing.assert_allclose(updated, np.full(updated.shape, expected), err_msg=field)
        assert not np.allclose(placeholder, expected), field


@pytest.mark.parametrize(
    ("transport", "transport_model"),
    [
        (TORAX_TRANSPORT_BLOCKS["qlknn"], "gyrobohm"),
        # No core models: TORAX silently injects a single prescribed one
        ({}, "gyrobohm"),
        ({"core_transport_models": {"other": {"model_name": "bohm-gyrobohm"}}}, "gyrobohm"),
        (
            {
                "core_transport_models": {
                    "gyrobohm": {"model_name": "bohm-gyrobohm"},
                    "extra": {"model_name": "prescribed", "rho_min": 0.9},
                }
            },
            "gyrobohm",
        ),
        (
            {
                "core_transport_models": {"gyrobohm": {"model_name": "bohm-gyrobohm"}},
                "pedestal_transport_models": {"ped": {"model_name": "prescribed"}},
            },
            "gyrobohm",
        ),
    ],
    ids=["wrong_model", "default_prescribed", "wrong_key", "extra_core_model", "pedestal_model"],
)
def test_validate_transport_model_name_rejects_mismatch(transport, transport_model):
    """Any transport config the NN overrides cannot address exactly is rejected."""
    torax_config = _torax_config(transport)
    with pytest.raises(ValueError, match="requires exactly the core transport models"):
        validate_transport_model_name(torax_config, transport_model)


@pytest.mark.slow
@requires_sample_data
@pytest.mark.parametrize("transport_model", ["constant", "gyrobohm", "qlknn"])
def test_torax_predictor(transport_model):
    config = StudyConfig(
        study_name=f"test_torax_predictor_{transport_model}",
        dataset_paths={
            "cmod-low": SAMPLE_DIR / "cmod-low1.nc",
            "cmod-high": SAMPLE_DIR / "cmod-high.nc",
        },
        target_device="cmod-high",
        # large batch with the the full ~100 shot sample dataset OOMs the GPU
        # 10 shots keeps this a cheap smoke test
        max_ds_size=10,
    )
    load_config(config)

    train_config = TrainConfig(**PROFILE_PREDICTOR_TORAX_CONFIGS[transport_model])
    training_data = {
        "sources": ["cmod-low"],
        "exnihilo": False,
    }
    train_config = train_config.model_copy(
        update={
            "project": config.study_name,
            "max_epochs": 4,
            "epochs_per_val": 2,
            "dataloader_config": {
                **train_config.dataloader_config,
                "training_data": training_data,
                "target_vars": [*PROFILE_TARGET_VARS, "ds_source_idx"],
                "batch_size": 512,
            },
            "model_init_config": {
                **train_config.model_init_config,
                # model_init indexes this strictly, the study normally supplies it
                "data_normalization": "physics-coral",
            },
        }
    )

    _trainer, _train_dl, _val_dl, _test_dl, _ = launch_train(train_config)


# qlknn training blew up on MAST samples with the circular geometry,
# so it and gyrobohm, the other stiff model, are the smoke coverage for the miller builder
@pytest.mark.slow
@requires_sample_data
@pytest.mark.parametrize("transport_model", ["gyrobohm", "qlknn"])
def test_torax_predictor_mast_miller(transport_model):
    config = StudyConfig(
        study_name=f"test_torax_predictor_mast_miller_{transport_model}",
        dataset_paths={
            "mast-low": SAMPLE_DIR / "mast-low1.nc",
            "mast-high": SAMPLE_DIR / "mast-high.nc",
        },
        target_device="mast-high",
        max_ds_size=10,
    )
    load_config(config)

    train_config = TrainConfig(**PROFILE_PREDICTOR_TORAX_CONFIGS[transport_model])
    training_data = {
        "sources": ["mast-low"],
        "exnihilo": False,
    }
    train_config = train_config.model_copy(
        update={
            "project": config.study_name,
            "max_epochs": 4,
            "epochs_per_val": 2,
            "dataloader_config": {
                **train_config.dataloader_config,
                "training_data": training_data,
                "target_vars": [*PROFILE_TARGET_VARS, "ds_source_idx"],
                "batch_size": 512,
            },
            "model_init_config": {
                **train_config.model_init_config,
                "geometry_builder": "miller",
                # model_init indexes this strictly, the study normally supplies it
                "data_normalization": "physics-coral",
            },
        }
    )

    _trainer, _train_dl, _val_dl, _test_dl, _ = launch_train(train_config)


@pytest.mark.slow
@requires_sample_data
def test_torax_heat_source_response(make_torax_module, sample_timeslices):
    # Pins the generic_heat wiring end to end: prescribing more auxiliary
    # power through the NN-controlled source must heat the relaxed profile
    module = make_torax_module("gyrobohm")
    timeslice = sample_timeslices("cmod-high.nc")[0]

    steps_cold, coeffs_cold = module.evolve(timeslice, prescribed={"P_aux_total": 0.0})
    steps_hot, coeffs_hot = module.evolve(timeslice, prescribed={"P_aux_total": 10.0})

    assert coeffs_cold["P_aux_total"] == pytest.approx(0.0)
    assert coeffs_hot["P_aux_total"] == pytest.approx(10.0)
    for steps in (steps_cold, steps_hot):
        for step in steps:
            assert np.all(np.isfinite(step["n_e_1e20"]))
            assert np.all(np.isfinite(step["t_e_keV"]))

    te_cold = steps_cold[-1]["t_e_keV"].mean()
    te_hot = steps_hot[-1]["t_e_keV"].mean()
    assert te_hot > te_cold * 1.05


@pytest.mark.slow
@requires_sample_data
def test_torax_output_hits_edge_bc_and_smooth_init(make_torax_module, sample_timeslices):
    module = make_torax_module("gyrobohm")
    timeslice = sample_timeslices("cmod-high.nc")[0]

    # The 51-point output must pass through the exact Dirichlet edge BC at
    # rho = 1 instead of flat-holding the outermost cell value
    outputs = module(timeslice)
    steps, coeffs = module.evolve(timeslice)
    assert outputs.te.values[-1] == pytest.approx(coeffs["T_e_right_bc"], rel=1e-3)
    assert outputs.ne.values[-1] == pytest.approx(coeffs["n_e_right_bc"], rel=1e-3)

    # The initial condition sampled on the cell grid must be a smooth parabola
    for key in ("t_e_keV", "n_e_1e20"):
        init = steps[0][key]
        d2 = np.diff(init, n=2)
        scale = np.abs(init).max()
        assert np.all(np.abs(d2 - d2.mean()) < 1e-3 * scale), key


def test_resolve_relaxation_overrides():
    # Nothing set: no overrides, torax_config numerics stay authoritative
    assert resolve_relaxation_overrides({}) == {}
    assert resolve_relaxation_overrides({"t_final": None, "fixed_dt": None, "n_solver_steps": None}) == {}

    # Explicit t_final / fixed_dt pass through unchanged
    assert resolve_relaxation_overrides({"t_final": 0.2}) == {"t_final": 0.2}
    assert resolve_relaxation_overrides({"t_final": 0.2, "fixed_dt": 0.02}) == {"t_final": 0.2, "fixed_dt": 0.02}

    # n_solver_steps derives fixed_dt = t_final / n_solver_steps, so the
    # swept horizon does not multiply per-sample solver cost
    assert resolve_relaxation_overrides({"t_final": 0.4, "n_solver_steps": 10}) == {
        "t_final": 0.4,
        "fixed_dt": pytest.approx(0.04),
    }
    assert resolve_relaxation_overrides({"t_final": 0.1, "n_solver_steps": 5, "fixed_dt": None}) == {
        "t_final": 0.1,
        "fixed_dt": pytest.approx(0.02),
    }

    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_relaxation_overrides({"t_final": 0.2, "fixed_dt": 0.02, "n_solver_steps": 5})
    with pytest.raises(ValueError, match="requires t_final"):
        resolve_relaxation_overrides({"n_solver_steps": 5})


def test_torax_max_steps_from_n_solver_steps(make_torax_module):
    # The module derives max_steps = ceil(t_final / fixed_dt) + 1, so an
    # n_solver_steps override must bound the scan length to n_solver_steps + 1
    module = make_torax_module("gyrobohm", numerics_overrides=resolve_relaxation_overrides({"t_final": 0.4, "n_solver_steps": 10}))
    assert module.max_steps == 11


@pytest.mark.slow
def test_torax_relaxation_loop_matches_step_replay(make_torax_module):
    """The jitted bounded relaxation loop in __call__ runs to t_final and matches evolve's step-by-step replay.

    Both apply the same steps and clamps, so the final cell profiles must agree
    once interpolated onto the output grid the way __call__ does.
    """
    module = make_torax_module("constant")
    inputs = jax.tree_util.tree_map(lambda x: x[0], _batch_inputs())

    outputs = module(inputs)
    steps, coeffs = module.evolve(inputs)

    numerics = module.step_fn.runtime_params_provider.numerics
    assert steps[-1]["t"] == pytest.approx(float(numerics.t_final))
    rho_full = np.concatenate([[0.0], steps[-1][RADIAL_DIM], [1.0]])
    ne_full = np.concatenate([steps[-1]["n_e_1e20"][:1], steps[-1]["n_e_1e20"], [coeffs["n_e_right_bc"]]])
    te_full = np.concatenate([steps[-1]["t_e_keV"][:1], steps[-1]["t_e_keV"], [coeffs["T_e_right_bc"]]])
    np.testing.assert_allclose(outputs.ne.values, np.interp(RHO_GRID, rho_full, ne_full), rtol=1e-6)
    np.testing.assert_allclose(outputs.te.values, np.interp(RHO_GRID, rho_full, te_full), rtol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("transport_model", ["constant", "gyrobohm", "qlknn"])
def test_torax_batched_gradients_finite(make_torax_module, transport_model):
    """Gradient of a vmapped batch through the relaxation loop is finite for every trainable network.

    Production training differentiates the vmapped forward, which exercises the
    bounded while loop's batching rule and custom VJP rather than a single sample.
    """
    module = make_torax_module(transport_model, geometry_builder="miller")
    inputs = _batch_inputs()

    def loss(mod):
        def one_sample(sample_inputs):
            out = mod(sample_inputs)
            return jnp.sum(out.te.data) + jnp.sum(out.ne.data)

        per_sample = jax.vmap(one_sample)(inputs)
        return jnp.sum(per_sample)

    value, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss))(module)
    assert np.isfinite(float(value))
    for name in ("nn_transport", "nn_sources", "nn_edge"):
        leaves = jax.tree_util.tree_leaves(eqx.filter(getattr(grads, name), eqx.is_inexact_array))
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves), name
        assert any(bool(jnp.any(leaf != 0.0)) for leaf in leaves), name
