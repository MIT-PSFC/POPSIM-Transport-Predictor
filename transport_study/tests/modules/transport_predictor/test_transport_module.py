"""Module-level tests for the transport predictor architectures.

Covers TransportPredictorEnv state seeding, trainable selections, the
transformer's history buffer semantics, and the TORAX modules' measured
P_aux feed-through and NN-predicted absorption_fraction. Study-level
wiring lives in tests/transport_transfer/.
"""

import copy
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from transport_study import RADIAL_DIM
from transport_study.config import RHO_GRID
from transport_study.modules.normalization import (
    CoralFeatureNormalizer,
    RawNormalizer,
    ZScoreFeatureNormalizer,
    feature_fit_arrays,
)
from transport_study.modules.power_balance.module import (
    PowerBalance,
    PowerBalanceSciML,
)
from transport_study.modules.power_balance.p_oh.module import OhmicPower
from transport_study.modules.power_balance.p_rad.module import RadiatedPower
from transport_study.modules.profile_predictor.module import (
    N_NN_INPUTS,
    ProfilePredictorUnstructuredNN,
)
from transport_study.modules.profile_predictor.torax_module import (
    TAU_REF_S,
    ProfilePredictorTorax,
)
from transport_study.modules.profile_predictor.train_configs import (
    TORAX_CONFIG_BASE,
    TORAX_TRANSPORT_BLOCKS,
)
from transport_study.modules.transport_predictor.module import (
    MIN_W_MJ,
    N_TRANSPORT_NN_INPUTS,
    NE_SEED_FLOOR_20,
    SOURCE_SHAPE_COEFFICIENT_NAMES,
    TE_SEED_FLOOR_KEV,
    TRANSPORT_NN_INPUT_NAMES,
    Inputs,
    TransportPredictorEnv,
    TransportPredictorSciML,
    TransportPredictorTorax,
    TransportPredictorToraxBase,
    TransportPredictorToraxSimState,
    TransportPredictorTransformer,
    energy_mhd_from_profiles,
    make_transport_nn_input_normalizer,
    transport_nn_input_matrix,
)
from transport_study.modules.transport_predictor.train_configs import (
    make_transport_torax_config,
)
from transport_study.signals import convert_to_working_units
from transport_study.tests.sample_data import SAMPLE_DIR, requires_sample_data

RHO = np.asarray(RHO_GRID)
N_RHO = len(RHO)

HISTORY_LEN = 4
D_MODEL = 8

# Measured t0 profiles with exact-zero edge points, as GP-fit rampdown
# timeslices produce, so the TORAX seed floors are exercised
NE0 = 1.5 * (1.0 - 0.5 * RHO**2)
NE0[-1] = 0.0
TE0 = 2.0 * (1.0 - RHO**2) + 0.05
TE0[-3:] = 0.0

INPUT_SCALARS = {
    "ip_MA": 1.0,
    "b0": 5.56,
    "b_geo": 5.4,
    "n_e_line_average_1e20": 1.5,
    "geometric_axis_r": 0.68,
    "minor_radius": 0.22,
    "elongation": 1.6,
    "triangularity_upper": 0.35,
    "triangularity_lower": 0.45,
    "power_additional_MW": 2.0,
    "ds_source_idx": 0.0,
}


def make_observations(**overrides) -> xr.Dataset:
    """t0 slice of the state_init_vars the dataloader hands create_state."""
    data = {
        "n_e_1e20": (RADIAL_DIM, NE0.copy()),
        "t_e_keV": (RADIAL_DIM, TE0.copy()),
        "energy_mhd_MJ": 0.15,
        **INPUT_SCALARS,
        **overrides,
    }
    return xr.Dataset(data)


def make_inputs(**overrides) -> Inputs:
    return Inputs(**{**INPUT_SCALARS, **overrides})


def scalar(value) -> float:
    """Squeeze a shape (1,) coefficient array to a python float."""
    return float(np.squeeze(np.asarray(value)))


def leaf_in_selection(selection, leaf) -> bool:
    """Whether an exact array object appears among the selection's leaves."""
    return any(sel_leaf is leaf for sel_leaf in jax.tree_util.tree_leaves(selection))


def normalizer_leaves(module) -> list:
    return jax.tree_util.tree_leaves(eqx.filter(module.normalizer, eqx.is_array))


@pytest.fixture(scope="module")
def transformer_module() -> TransportPredictorTransformer:
    return TransportPredictorTransformer.init(
        d_model=D_MODEL,
        num_heads=2,
        history_len=HISTORY_LEN,
        nn_width=8,
        nn_depth=1,
        rhogrid=RHO,
        normalizer=make_transport_nn_input_normalizer("physics", None, 2),
        prng_seed=0,
    )


@pytest.fixture(scope="module")
def sciml_module() -> TransportPredictorSciML:
    p_oh = OhmicPower.init(in_size=7, out_size=1, nn_width=4, nn_depth=1, prng_seed=0, normalizer=RawNormalizer())
    p_rad = RadiatedPower.init(in_size=7, out_size=1, nn_width=4, nn_depth=1, prng_seed=1, normalizer=RawNormalizer())
    power_balance = PowerBalanceSciML.init(
        in_size=7,
        out_size=1,
        nn_width=4,
        nn_depth=1,
        p_oh_predictor=p_oh,
        p_rad_predictor=p_rad,
        normalizer=RawNormalizer(),
    )
    profile_predictor = ProfilePredictorUnstructuredNN(
        nn_width=4,
        nn_depth=1,
        rhogrid=tuple(RHO.tolist()),
        key=jax.random.PRNGKey(0),
        normalizer=CoralFeatureNormalizer.identity(2, N_NN_INPUTS),
    )
    return TransportPredictorSciML.init(power_balance=power_balance, profile_predictor=profile_predictor)


@pytest.fixture(scope="module")
def torax_rebuild_module() -> TransportPredictorTorax:
    return TransportPredictorTorax.init(
        rhogrid=RHO,
        torax_config=make_transport_torax_config("constant"),
        nn_width=8,
        nn_depth=2,
        prng_seed=0,
        normalizer=make_transport_nn_input_normalizer("physics", None, 2),
        sim_dt=0.001,
        transport_model="constant",
        geometry_builder="circular",
    )


@pytest.fixture(scope="module")
def torax_carry_module() -> TransportPredictorToraxSimState:
    return TransportPredictorToraxSimState.init(
        rhogrid=RHO,
        torax_config=make_transport_torax_config("constant"),
        nn_width=8,
        nn_depth=2,
        prng_seed=0,
        normalizer=make_transport_nn_input_normalizer("physics", None, 2),
        sim_dt=0.001,
        transport_model="constant",
        geometry_builder="circular",
    )


def test_env_create_state_per_model_type(transformer_module, sciml_module, torax_rebuild_module, torax_carry_module):
    """TransportPredictorEnv.create_state seeds the right state per module:
    transformer gets a (history_len, 2 n_rho) buffer tiled from the measured
    t0 profiles, sciml gets PowerBalance.State with the measured energy_mhd_MJ,
    torax rebuild gets ne/te from the measured profiles, and torax carry gets
    a full ToraxSimState built from the measured profiles with the edge points
    pinned to the NN boundary conditions."""
    obs = make_observations()

    # Transformer: whole buffer tiled from the raw measured t0 profiles
    state_tf = TransportPredictorEnv(module=transformer_module).create_state(obs, obs)
    assert isinstance(state_tf, TransportPredictorTransformer.State)
    assert state_tf.profiles.shape == (HISTORY_LEN, 2 * N_RHO)
    row = np.concatenate([NE0, TE0])
    np.testing.assert_array_equal(np.asarray(state_tf.profiles), np.tile(row, (HISTORY_LEN, 1)))

    # Sciml: PowerBalance.State seeded with the measured stored energy
    state_sciml = TransportPredictorEnv(module=sciml_module).create_state(obs, obs)
    assert isinstance(state_sciml, PowerBalance.State)
    assert float(state_sciml.energy_mhd_MJ) == pytest.approx(0.15)

    # Torax rebuild: measured profiles with the seed floors applied
    state_rb = TransportPredictorEnv(module=torax_rebuild_module).create_state(obs, obs)
    assert isinstance(state_rb, TransportPredictorTorax.State)
    np.testing.assert_allclose(np.asarray(state_rb.ne), np.maximum(NE0, NE_SEED_FLOOR_20))
    np.testing.assert_allclose(np.asarray(state_rb.te), np.maximum(TE0, TE_SEED_FLOOR_KEV))
    # The exact-zero measured edge points really hit the floors
    assert np.all(TE0[-3:] == 0.0)
    np.testing.assert_allclose(np.asarray(state_rb.te[-3:]), TE_SEED_FLOOR_KEV)
    np.testing.assert_allclose(np.asarray(state_rb.ne[-1]), NE_SEED_FLOOR_20)

    # Torax carry: full ToraxSimState at t_initial, edge points pinned to the
    # NN Dirichlet boundary conditions computed from the floored seeds
    state_carry = TransportPredictorEnv(module=torax_carry_module).create_state(obs, obs)
    assert isinstance(state_carry, TransportPredictorToraxSimState.State)
    sim_state, post_processed = state_carry.unwrap()
    assert sim_state is not None
    assert post_processed is not None
    assert float(sim_state.t) == pytest.approx(0.0)

    inputs0 = TransportPredictorEnv.create_inputs(obs)
    ne_seed = jnp.maximum(jnp.asarray(NE0), NE_SEED_FLOOR_20)
    te_seed = jnp.maximum(jnp.asarray(TE0), TE_SEED_FLOOR_KEV)
    wtot0 = energy_mhd_from_profiles(ne_seed, te_seed, jnp.asarray(RHO), inputs0.volume_approx)
    coeffs = torax_carry_module.nn_coefficients(inputs0, wtot0)
    core_profiles = sim_state.core_profiles
    np.testing.assert_allclose(
        np.asarray(core_profiles.T_e.right_face_constraint),
        np.squeeze(np.asarray(coeffs["T_e_right_bc"])),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(core_profiles.n_e.right_face_constraint),
        1e20 * np.squeeze(np.asarray(coeffs["n_e_right_bc"])),
        rtol=1e-6,
    )
    assert np.all(np.isfinite(np.asarray(core_profiles.T_e.value)))
    assert np.all(np.asarray(core_profiles.T_e.value) > 0.0)
    assert np.all(np.asarray(core_profiles.n_e.value) > 0.0)


def test_env_get_trainable_selections(transformer_module, sciml_module, torax_rebuild_module):
    """get_trainable never includes normalizer statistics; for transfer it
    returns only last-layer leaves (transformer head, the three torax MLPs,
    sciml taue/profile last layers); for sciml with freeze_submodules both
    submodules drop out of the selection entirely; for the transformer at
    da=none the selection is exactly feature_embed, profile_embed, pos_embed,
    attention and head."""
    # Transformer, da=none: exactly the five architecture components
    sel_tf = TransportPredictorEnv(module=transformer_module, domain_adaptation="none").get_trainable()
    assert sorted(sel_tf.keys()) == ["attention", "feature_embed", "head", "pos_embed", "profile_embed"]
    assert leaf_in_selection(sel_tf, transformer_module.pos_embed)
    for leaf in normalizer_leaves(transformer_module):
        assert not leaf_in_selection(sel_tf, leaf)

    # Transformer, transfer: only the head's last layer
    sel_tf_transfer = TransportPredictorEnv(module=transformer_module, domain_adaptation="transfer").get_trainable()
    head_last = transformer_module.head.layers[-1]
    assert len(sel_tf_transfer) == 2
    assert sel_tf_transfer[0] is head_last.weight
    assert sel_tf_transfer[1] is head_last.bias
    for leaf in normalizer_leaves(transformer_module):
        assert not leaf_in_selection(sel_tf_transfer, leaf)

    # Torax, transfer: last layers of the three MLPs, nothing else
    sel_tx_transfer = TransportPredictorEnv(module=torax_rebuild_module, domain_adaptation="transfer").get_trainable()
    expected = []
    for nn in (torax_rebuild_module.nn_transport, torax_rebuild_module.nn_sources, torax_rebuild_module.nn_edge):
        expected += [nn.layers[-1].weight, nn.layers[-1].bias]
    assert len(sel_tx_transfer) == len(expected)
    for got, want in zip(sel_tx_transfer, expected, strict=True):
        assert got is want
    for leaf in normalizer_leaves(torax_rebuild_module):
        assert not leaf_in_selection(sel_tx_transfer, leaf)

    # Torax, da=none: exactly the three MLPs, normalizer excluded
    sel_tx = TransportPredictorEnv(module=torax_rebuild_module, domain_adaptation="none").get_trainable()
    assert sorted(sel_tx.keys()) == ["nn_edge", "nn_sources", "nn_transport"]
    for leaf in normalizer_leaves(torax_rebuild_module):
        assert not leaf_in_selection(sel_tx, leaf)

    # Sciml, da=none: both submodule subtrees present, dropped when frozen
    sel_sciml = TransportPredictorEnv(module=sciml_module, domain_adaptation="none").get_trainable()
    assert sorted(sel_sciml.keys()) == ["power_balance", "profile_predictor"]
    sel_sciml_frozen = TransportPredictorEnv(
        module=sciml_module,
        domain_adaptation="none",
        freeze_submodules=["power_balance", "profile_predictor"],
    ).get_trainable()
    assert sel_sciml_frozen == {}

    # Sciml, transfer: taue / p_oh / p_rad / profile last layers only
    sel_sciml_transfer = TransportPredictorEnv(module=sciml_module, domain_adaptation="transfer").get_trainable()
    pb = sciml_module.power_balance
    expected_sciml = [
        pb.taue_predictor.nn.layers[-1].weight,
        pb.taue_predictor.nn.layers[-1].bias,
        pb.p_oh_predictor.nn.layers[-1].weight,
        pb.p_oh_predictor.nn.layers[-1].bias,
        pb.p_rad_predictor.nn.layers[-1].weight,
        pb.p_rad_predictor.nn.layers[-1].bias,
        sciml_module.profile_predictor.nn.layers[-1].weight,
        sciml_module.profile_predictor.nn.layers[-1].bias,
    ]
    assert len(sel_sciml_transfer) == len(expected_sciml)
    for want in expected_sciml:
        assert leaf_in_selection(sel_sciml_transfer, want)
    for leaf in normalizer_leaves(sciml_module.profile_predictor):
        assert not leaf_in_selection(sel_sciml_transfer, leaf)
        assert not leaf_in_selection(sel_sciml, leaf)

    # Sciml, transfer with both submodules frozen: nothing left to train
    sel_sciml_transfer_frozen = TransportPredictorEnv(
        module=sciml_module,
        domain_adaptation="transfer",
        freeze_submodules=["power_balance", "profile_predictor"],
    ).get_trainable()
    assert sel_sciml_transfer_frozen == []


def test_transformer_position_embedding_orders_history(transformer_module):
    """The transformer output must change when the profile history buffer is
    reversed (position embedding breaks permutation invariance), and
    pos_embed has shape (history_len, d_model) with a small nonzero init so
    the t0-seeded constant buffer still yields slot-distinguishable tokens."""
    pos_embed = np.asarray(transformer_module.pos_embed)
    assert pos_embed.shape == (HISTORY_LEN, D_MODEL)
    assert np.any(pos_embed != 0.0)
    # Small init: breaks slot symmetry without drowning the profile tokens
    assert np.max(np.abs(pos_embed)) < 0.2

    inputs = make_inputs()
    obs = make_observations()
    seeded = TransportPredictorEnv(module=transformer_module).create_state(obs, obs)
    # Nonconstant history, then swap two rows that are NOT the newest one so
    # the current profile (and the features derived from it) stay identical
    profiles = seeded.profiles.at[0].mul(1.2).at[1].mul(0.8)
    permuted = profiles.at[0].set(profiles[1]).at[1].set(profiles[0])
    state = TransportPredictorTransformer.State(profiles=profiles)
    state_perm = TransportPredictorTransformer.State(profiles=permuted)

    next_state, output = transformer_module(state, inputs)
    next_state_perm, output_perm = transformer_module(state_perm, inputs)
    # The reported output (profile at time t) is order-independent by design
    np.testing.assert_array_equal(np.asarray(output.ne), np.asarray(output_perm.ne))
    # But the predicted next profile must see the history order
    assert not np.allclose(np.asarray(next_state.profiles[-1]), np.asarray(next_state_perm.profiles[-1]))

    # Zeroing pos_embed restores permutation invariance, so the position
    # embedding is exactly what breaks it
    no_pos = eqx.tree_at(lambda m: m.pos_embed, transformer_module, jnp.zeros_like(transformer_module.pos_embed))
    next_no_pos, _ = no_pos(state, inputs)
    next_no_pos_perm, _ = no_pos(state_perm, inputs)
    np.testing.assert_allclose(
        np.asarray(next_no_pos.profiles[-1]),
        np.asarray(next_no_pos_perm.profiles[-1]),
        rtol=1e-6,
        atol=1e-8,
    )


def test_transformer_history_holds_profiles_only(transformer_module):
    """The rolling State buffer contains only predicted ne/te profile rows,
    (history_len, 2 n_rho): past input features are never stored, only the
    current timestep's features form the attention query."""
    field_names = [f.name for f in dataclasses.fields(TransportPredictorTransformer.State)]
    assert field_names == ["profiles"]

    obs = make_observations()
    state = TransportPredictorEnv(module=transformer_module).create_state(obs, obs)
    assert state.profiles.shape == (HISTORY_LEN, 2 * N_RHO)

    # The buffer update is a pure shift plus the newest prediction: the rows
    # carried over are byte-identical to the old rows regardless of the
    # current input features, so no feature values ever enter the buffer
    next_a, _ = transformer_module(state, make_inputs(power_additional_MW=0.0))
    next_b, _ = transformer_module(state, make_inputs(power_additional_MW=6.0))
    np.testing.assert_array_equal(np.asarray(next_a.profiles[:-1]), np.asarray(state.profiles[1:]))
    np.testing.assert_array_equal(np.asarray(next_b.profiles[:-1]), np.asarray(state.profiles[1:]))
    # Only the newest row responds to the changed features (via the query)
    assert not np.allclose(np.asarray(next_a.profiles[-1]), np.asarray(next_b.profiles[-1]))

    # The next call reads its Output from the newest buffered profile
    _, output = transformer_module(next_a, make_inputs())
    np.testing.assert_array_equal(np.asarray(output.ne), np.asarray(next_a.profiles[-1, :N_RHO]))
    np.testing.assert_array_equal(np.asarray(output.te), np.asarray(next_a.profiles[-1, N_RHO:]))


def test_torax_p_aux_feed_through(torax_rebuild_module):
    """The measured power_additional_MW input is wired directly to the TORAX
    generic_heat.P_total runtime update (MW to W) for the rebuild variant,
    the carry variant, and the env's initial TORAX state construction (all
    three route through TransportPredictorToraxBase.build_provider_and_geo);
    the sources network predicts only the deposition shape
    (gaussian_location, gaussian_width, electron_heat_fraction), the
    gas-puff fueling, and the absorption_fraction, so no NN output can
    override the measured injected heating magnitude."""
    module = torax_rebuild_module

    # The sources network has no heating-magnitude output slot
    assert SOURCE_SHAPE_COEFFICIENT_NAMES == (
        "S_total",
        "gaussian_location",
        "gaussian_width",
        "electron_heat_fraction",
        "absorption_fraction",
    )
    assert not any("P_aux" in name or "P_total" in name for name in SOURCE_SHAPE_COEFFICIENT_NAMES)
    assert module.nn_sources.out_size == len(SOURCE_SHAPE_COEFFICIENT_NAMES)

    # Rebuild, carry, and the env's initial-state construction all share the
    # base implementation, so the provider-level checks below cover them all
    assert TransportPredictorTorax.build_provider_and_geo is TransportPredictorToraxBase.build_provider_and_geo
    assert TransportPredictorToraxSimState.build_provider_and_geo is TransportPredictorToraxBase.build_provider_and_geo

    for p_aux in (0.0, 2.0, 6.0):
        inputs = make_inputs(power_additional_MW=p_aux)
        coeffs = module.nn_coefficients(inputs, 0.15)
        provider, _geo = module.build_provider_and_geo(inputs, coeffs)
        heat = provider(t=0.0).sources["generic_heat"]
        # Measured heating magnitude, MW to W, straight through
        np.testing.assert_allclose(np.asarray(heat.P_total), p_aux * 1e6)
        # The NN-predicted deposition shape reaches the runtime params
        np.testing.assert_allclose(np.asarray(heat.gaussian_location), np.squeeze(np.asarray(coeffs["gaussian_location"])))
        np.testing.assert_allclose(np.asarray(heat.gaussian_width), np.squeeze(np.asarray(coeffs["gaussian_width"])))
        np.testing.assert_allclose(
            np.asarray(heat.electron_heat_fraction),
            np.squeeze(np.asarray(coeffs["electron_heat_fraction"])),
        )
        # Bounds on the NN-predicted source shape coefficients
        assert 0.0 <= scalar(coeffs["gaussian_location"]) <= 0.8
        assert 0.02 <= scalar(coeffs["gaussian_width"]) <= 0.4
        assert 0.2 <= scalar(coeffs["electron_heat_fraction"]) <= 0.95
        assert scalar(coeffs["S_total"]) >= 0.0

    # Transport coefficient bounds saturate for any raw network output
    for raw in (50.0, -50.0):
        bounded = module.transport_coefficients(jnp.full((4,), raw))
        assert 0.1 <= scalar(bounded["chi_i"]) <= 5.0
        assert 0.1 <= scalar(bounded["chi_e"]) <= 10.0
        assert 0.1 <= scalar(bounded["D_e"]) <= 2.0
        assert -5.0 <= scalar(bounded["V_e"]) <= 5.0

    # Perturbing the sources network changes its predictions but can never
    # touch the measured heating magnitude
    inputs = make_inputs()
    arrays, static = eqx.partition(module.nn_sources, eqx.is_inexact_array)
    perturbed_nn = eqx.combine(jax.tree_util.tree_map(lambda x: x + 0.7, arrays), static)
    perturbed = eqx.tree_at(lambda m: m.nn_sources, module, perturbed_nn)
    coeffs = module.nn_coefficients(inputs, 0.15)
    coeffs_perturbed = perturbed.nn_coefficients(inputs, 0.15)
    assert not np.allclose(
        np.asarray(coeffs["absorption_fraction"]),
        np.asarray(coeffs_perturbed["absorption_fraction"]),
    )
    heat = module.build_provider_and_geo(inputs, coeffs)[0](t=0.0).sources["generic_heat"]
    heat_perturbed = perturbed.build_provider_and_geo(inputs, coeffs_perturbed)[0](t=0.0).sources["generic_heat"]
    np.testing.assert_allclose(np.asarray(heat.P_total), inputs.power_additional_MW * 1e6)
    np.testing.assert_allclose(np.asarray(heat_perturbed.P_total), inputs.power_additional_MW * 1e6)


@pytest.mark.slow
def test_torax_p_aux_change_reaches_one_step_output(torax_rebuild_module):
    """Changing power_additional_MW changes the one-step evolved profiles with the
    module weights held fixed, so the measured heating magnitude feeds
    through the whole TORAX solve, not just the runtime params."""
    obs = make_observations()
    state = TransportPredictorEnv(module=torax_rebuild_module).create_state(obs, obs)
    next_low, _ = torax_rebuild_module(state, make_inputs(power_additional_MW=0.0))
    next_high, _ = torax_rebuild_module(state, make_inputs(power_additional_MW=8.0))
    assert np.all(np.isfinite(np.asarray(next_low.te)))
    assert np.all(np.isfinite(np.asarray(next_high.te)))
    assert not np.allclose(np.asarray(next_low.te), np.asarray(next_high.te))


@pytest.mark.slow
@pytest.mark.parametrize("transport_model", ["constant", "gyrobohm"])
def test_torax_rebuild_step_gradients_finite(transport_model):
    """One rebuild step (initial state built from the carried profiles, checkpointed solver step)
    gives finite, nonzero gradients for every trainable network."""
    module = TransportPredictorTorax.init(
        rhogrid=RHO,
        torax_config=make_transport_torax_config(transport_model),
        nn_width=8,
        nn_depth=2,
        prng_seed=0,
        normalizer=make_transport_nn_input_normalizer("physics", None, 2),
        sim_dt=0.001,
        transport_model=transport_model,
        geometry_builder="miller",
    )
    obs = make_observations()
    state = TransportPredictorEnv(module=module).create_state(obs, obs)
    inputs = make_inputs()

    def loss(mod):
        next_state, _output = mod(state, inputs)
        return jnp.sum(next_state.te) + jnp.sum(next_state.ne)

    value, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss))(module)
    assert np.isfinite(float(value))
    for name in ("nn_transport", "nn_sources", "nn_edge"):
        leaves = jax.tree_util.tree_leaves(eqx.filter(getattr(grads, name), eqx.is_inexact_array))
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves), name
        assert any(bool(jnp.any(leaf != 0.0)) for leaf in leaves), name


def test_torax_absorption_fraction_nn(torax_rebuild_module):
    """The transport sources network's last output sets absorption_fraction
    via the saturating Beer-Lambert form 1 - exp(-n_e_line_average_1e20 * softplus(nn
    output)): always in (0, 1), linear in line density when optically thin,
    smoothly saturating toward 1 with no gradient-dead cap. The value reaches
    the generic_heat.absorption_fraction runtime update in
    build_provider_and_geo, and the profile predictor's TORAX modules keep
    the fixed config value 0.9."""
    module = torax_rebuild_module
    wtot = 0.15

    # Manual reconstruction of the Beer-Lambert form from the raw network output
    for ne in (0.05, 0.5, 1.5, 5.0):
        inputs = make_inputs(n_e_line_average_1e20=ne)
        coeffs = module.nn_coefficients(inputs, wtot)
        nn_inputs = module.normalizer(inputs.transport_nn_inputs(wtot), inputs.ds_source_idx)
        opacity = jax.nn.softplus(module.nn_sources(nn_inputs)[4:5])
        expected = 1.0 - jnp.exp(-ne * opacity)
        np.testing.assert_allclose(np.asarray(coeffs["absorption_fraction"]), np.asarray(expected), rtol=1e-6)
        assert 0.0 < scalar(coeffs["absorption_fraction"]) < 1.0

    # Fix the opacity and probe the functional form in line density
    inputs = make_inputs()
    nn_inputs = module.normalizer(inputs.transport_nn_inputs(wtot), inputs.ds_source_idx)
    opacity = float(jax.nn.softplus(module.nn_sources(nn_inputs)[4]))
    assert opacity > 0.0

    def beer_lambert(ne_line):
        return 1.0 - jnp.exp(-ne_line * opacity)

    # Optically thin: linear in line density
    thin = 1e-4
    assert float(beer_lambert(thin)) == pytest.approx(thin * opacity, rel=1e-3)
    # Saturates toward 1 while the gradient stays strictly positive (no dead cap)
    assert float(beer_lambert(1e3)) > 0.999
    assert float(jax.grad(beer_lambert)(50.0)) > 0.0

    # The predicted value reaches the generic_heat runtime params
    coeffs = module.nn_coefficients(inputs, wtot)
    heat = module.build_provider_and_geo(inputs, coeffs)[0](t=0.0).sources["generic_heat"]
    np.testing.assert_allclose(np.asarray(heat.absorption_fraction), np.squeeze(np.asarray(coeffs["absorption_fraction"])))

    # The profile predictor's TORAX modules keep the fixed config absorption:
    # 0.9 in the shared config skeleton, never overridden by their provider
    assert TORAX_CONFIG_BASE["sources"]["generic_heat"]["absorption_fraction"] == 0.9
    assert make_transport_torax_config("constant")["sources"]["generic_heat"]["absorption_fraction"] == 0.9
    profile_torax_config = copy.deepcopy(TORAX_CONFIG_BASE)
    profile_torax_config["transport"] = copy.deepcopy(TORAX_TRANSPORT_BLOCKS["constant"])
    profile_module = ProfilePredictorTorax(
        nn_width=8,
        nn_depth=2,
        rhogrid=tuple(RHO.tolist()),
        torax_config=profile_torax_config,
        key=jax.random.PRNGKey(0),
        normalizer=CoralFeatureNormalizer.identity(1, N_NN_INPUTS),
        transport_model="constant",
        geometry_builder="circular",
    )
    profile_inputs = inputs.to_profile_predictor_inputs(rho=jnp.asarray(RHO), beta_tor_norm=1.5)
    profile_coeffs = profile_module.nn_coefficients(profile_inputs)
    profile_heat = profile_module.build_provider_and_geo(profile_inputs, profile_coeffs)[0](t=0.0).sources["generic_heat"]
    np.testing.assert_allclose(np.asarray(profile_heat.absorption_fraction), 0.9)


@pytest.mark.slow
def test_torax_absorbed_power_matches_absorption_fraction(torax_carry_module):
    """The absorbed power inside TORAX equals P_total * absorption_fraction:
    after one solver step the post-processed generic_heat total is the
    measured P_aux scaled by the NN-predicted absorption fraction."""
    obs = make_observations()
    inputs = TransportPredictorEnv.create_inputs(obs)
    state = TransportPredictorEnv(module=torax_carry_module).create_state(obs, obs)

    # Recompute the coefficients exactly as __call__ does, from the stored
    # energy implied by the carried TORAX core profiles
    core_profiles = state.unwrap()[0].core_profiles
    rho_cells = jnp.asarray(torax_carry_module.rho_norm_grid)
    wtot = energy_mhd_from_profiles(
        core_profiles.n_e.value / 1e20,
        core_profiles.T_e.value,
        rho_cells,
        inputs.volume_approx,
    )
    coeffs = torax_carry_module.nn_coefficients(inputs, wtot)

    next_state, _output = torax_carry_module(state, inputs)
    absorbed = float(np.squeeze(np.asarray(next_state.unwrap()[1].P_aux_generic_total)))
    expected = float(inputs.power_additional_MW) * 1e6 * float(np.squeeze(np.asarray(coeffs["absorption_fraction"])))
    assert absorbed == pytest.approx(expected, rel=1e-3)


@requires_sample_data
def test_normalizer_fit_features():
    """make_transport_nn_input_normalizer builds an (N, 11) feature matrix
    whose columns match Inputs.transport_nn_inputs evaluated with the
    measured energy_mhd_MJ, drops rows with NaN device index, and returns identity
    stats for devices with too few samples."""
    assert N_TRANSPORT_NN_INPUTS == 11
    assert len(TRANSPORT_NN_INPUT_NAMES) == N_TRANSPORT_NN_INPUTS
    assert TRANSPORT_NN_INPUT_NAMES[-1] == "paux_norm"

    ds_sample = xr.open_dataset(SAMPLE_DIR / "cmod-low1.nc").isel(shot=slice(0, 40), time_idx=slice(0, 600, 3))
    ds = convert_to_working_units(ds_sample)
    source_idx_per_shot = np.zeros(ds.sizes["shot"])
    source_idx_per_shot[30:35] = 1.0  # a device with too few shots for CORAL
    source_idx_per_shot[35:] = np.nan  # unattributed shots (NaN-padded concat)
    ds["ds_source_idx"] = ("shot", source_idx_per_shot)
    n_devices = 3  # device 2 is registered but absent from the fit data

    matrix = transport_nn_input_matrix(ds)
    n_times = ds.sizes["time_idx"]
    assert matrix.shape == (ds.sizes["shot"] * n_times, N_TRANSPORT_NN_INPUTS)

    # Columns match Inputs.transport_nn_inputs evaluated with the MEASURED
    # stored energy, checked on one complete flattened row
    reference = ds["ip_MA"]

    def flat(var: str) -> np.ndarray:
        return np.asarray(ds[var].broadcast_like(reference).values, dtype=float).ravel()

    source_vars = (
        "ip_MA",
        "b0",
        "b_geo",
        "n_e_line_average_1e20",
        "geometric_axis_r",
        "minor_radius",
        "elongation",
        "triangularity_upper",
        "triangularity_lower",
        "power_additional_MW",
    )
    columns = {var: flat(var) for var in (*source_vars, "energy_mhd_MJ")}
    stacked = np.column_stack(list(columns.values()))
    row = int(np.argmax(~np.isnan(stacked).any(axis=1)))
    assert not np.isnan(stacked[row]).any()
    scalar_inputs = Inputs(**{var: columns[var][row] for var in source_vars}, ds_source_idx=0.0)
    np.testing.assert_allclose(
        np.asarray(scalar_inputs.transport_nn_inputs(columns["energy_mhd_MJ"][row])),
        matrix[row],
        rtol=1e-6,
    )
    # The aux power slot is the dimensionless TAU_REF_S * P_aux / Wtot scale,
    # log1p compressed so a near-zero stored energy cannot send it out of
    # distribution (P_aux = 0 still maps to exactly 0)
    expected_paux_norm = np.log1p(TAU_REF_S * columns["power_additional_MW"] / np.maximum(columns["energy_mhd_MJ"], MIN_W_MJ))
    np.testing.assert_allclose(matrix[:, -1], expected_paux_norm, rtol=1e-6, equal_nan=True)

    # Rows with a NaN device index are dropped before fitting
    features, source_idx, _shot_idx = feature_fit_arrays(ds, matrix)
    n_attributed_shots = int(np.sum(~np.isnan(source_idx_per_shot)))
    assert features.shape == (n_attributed_shots * n_times, N_TRANSPORT_NN_INPUTS)
    assert not np.isnan(source_idx).any()
    assert set(np.unique(source_idx)) == {0, 1}

    # physics-coral: the well-populated device gets fitted stats, the
    # low-shot device and the absent device keep identity rows
    normalizer = make_transport_nn_input_normalizer("physics-coral", ds, n_devices)
    assert isinstance(normalizer, CoralFeatureNormalizer)
    eye = np.eye(N_TRANSPORT_NN_INPUTS)
    assert not np.allclose(np.asarray(normalizer.transforms[0]), eye)
    complete = ~np.isnan(features).any(axis=1)
    device0 = complete & (source_idx == 0)
    np.testing.assert_allclose(np.asarray(normalizer.means[0]), features[device0].mean(axis=0), rtol=1e-6)
    for low_count_device in (1, 2):
        np.testing.assert_array_equal(np.asarray(normalizer.transforms[low_count_device]), eye)
        np.testing.assert_array_equal(np.asarray(normalizer.means[low_count_device]), 0.0)

    # physics-zscore: fitted for present devices, identity for the absent one
    zscore = make_transport_nn_input_normalizer("physics-zscore", ds, n_devices)
    assert isinstance(zscore, ZScoreFeatureNormalizer)
    assert not np.allclose(np.asarray(zscore.means[0]), 0.0)
    np.testing.assert_array_equal(np.asarray(zscore.means[2]), 0.0)
    np.testing.assert_array_equal(np.asarray(zscore.stds[2]), 1.0)

    # physics keeps identity buffers, and fit_ds None yields identity stats
    # with the checkpoint's pytree structure
    for identity_normalizer in (
        make_transport_nn_input_normalizer("physics", ds, n_devices),
        make_transport_nn_input_normalizer("physics-coral", None, n_devices),
    ):
        assert isinstance(identity_normalizer, CoralFeatureNormalizer)
        np.testing.assert_array_equal(
            np.asarray(identity_normalizer.transforms),
            np.tile(eye, (n_devices, 1, 1)),
        )
    assert isinstance(make_transport_nn_input_normalizer("physics-zscore", None, n_devices), ZScoreFeatureNormalizer)
