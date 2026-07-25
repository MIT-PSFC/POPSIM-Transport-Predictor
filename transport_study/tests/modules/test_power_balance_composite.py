"""Tests for the power balance composite training loss (submodule anchor terms)
and the grouped optimizer (reduced submodule learning rate).
"""

import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from popsim.ml.partition import make_partition_by_members

from transport_study import PACKAGE_ROOT
from transport_study.config import StudyConfig, load_config
from transport_study.modules.normalization import make_normalizer
from transport_study.modules.power_balance.module import (
    PowerBalance,
    PowerBalanceScalingLaw,
    PowerBalanceSciML,
    PowerBalanceTransformer,
    PowerBalanceUnstructuredNN,
    TauePredictorOutputs,
)
from transport_study.modules.power_balance.p_oh.module import OhmicPower
from transport_study.modules.power_balance.p_rad.module import RadiatedPower
from transport_study.modules.power_balance.trb import PowerBalanceTRB
from transport_study.modules.trb_utils import make_grouped_exponential_adamw
from transport_study.power_balance_transfer.power_balance_study import (
    HYPERPARAM_TARGET_SHOTS,
    PowerBalanceStudy,
)

SAMPLE_DIR = PACKAGE_ROOT / "datasets" / "sample"

##################
# Loss functions #
##################

# device_weights passed explicitly so the loss builder does not consult the
# global config (only the device-weight test loads one)
BASE_LOSS_CONFIG = {
    "huber_delta": 0.5,
    "anchor_weight_p_oh": 0.1,
    "anchor_weight_p_rad": 0.1,
    "device_weights": {},
}


def _make_output(n: int, p_oh: float | None = None, p_rad: float | None = None) -> PowerBalance.Output:
    kwargs = {}
    if p_oh is not None:
        kwargs["P_oh_MW_pred"] = jnp.full(n, p_oh)
    if p_rad is not None:
        kwargs["P_rad_MW_pred"] = jnp.full(n, p_rad)
    return PowerBalance.Output(
        Wtot_MJ_pred=jnp.ones(n),
        P_cond_MW=jnp.ones(n),
        taue_predictor_output=TauePredictorOutputs(taue_pred=jnp.ones(n), debug_info={}),
        **kwargs,
    )


def _make_targets(n: int, with_powers: bool = True, ds_source_idx=None) -> dict:
    if ds_source_idx is None:
        ds_source_idx = np.zeros(n, dtype=np.float32)
    targ = {
        "Wtot_MJ": xr.DataArray(np.ones(n, dtype=np.float32)),
        "ds_source_idx": xr.DataArray(np.asarray(ds_source_idx, dtype=np.float32)),
    }
    if with_powers:
        targ["P_oh_MW"] = xr.DataArray(np.full(n, 2.0, dtype=np.float32))
        targ["P_rad_MW"] = xr.DataArray(np.full(n, 1.0, dtype=np.float32))
    return targ


def test_anchor_terms_enter_train_loss_for_sciml():
    inst = PowerBalanceTRB.get_loss_fn(BASE_LOSS_CONFIG).instantaneous_loss
    targ = _make_targets(4)

    loss_match = inst(_make_output(4, p_oh=2.0, p_rad=1.0), targ)
    loss_drift = inst(_make_output(4, p_oh=5.0, p_rad=1.0), targ)

    # Wtot term is zero (pred == targ), so the difference is exactly the
    # P_oh anchor: anchor_weight * mean(|5 - 2|)
    assert float(loss_match) == 0.0
    assert np.isclose(float(loss_drift), 0.1 * 3.0, atol=1e-6)


def test_anchor_terms_absent_from_val_loss():
    inst = PowerBalanceTRB.get_val_loss_fn(BASE_LOSS_CONFIG).instantaneous_loss
    targ = _make_targets(4)

    loss_match = inst(_make_output(4, p_oh=2.0, p_rad=1.0), targ)
    loss_drift = inst(_make_output(4, p_oh=5.0, p_rad=4.0), targ)

    assert np.isclose(float(loss_match), float(loss_drift))


def test_anchor_terms_noop_without_power_targets():
    # mlp / transformer target_vars carry no powers, and their Output keeps
    # the NaN defaults, which must never reach the loss
    inst = PowerBalanceTRB.get_loss_fn(BASE_LOSS_CONFIG).instantaneous_loss
    targ = _make_targets(4, with_powers=False)

    loss = inst(_make_output(4), targ)

    assert np.isfinite(float(loss))
    assert float(loss) == 0.0


def test_anchor_weight_zero_disables_term():
    loss_config = {**BASE_LOSS_CONFIG, "anchor_weight_p_oh": 0.0, "anchor_weight_p_rad": 0.0}
    inst = PowerBalanceTRB.get_loss_fn(loss_config).instantaneous_loss
    targ = _make_targets(4)

    loss_drift = inst(_make_output(4, p_oh=5.0, p_rad=4.0), targ)

    assert float(loss_drift) == 0.0


def test_anchor_terms_respect_device_weights():
    load_config(
        StudyConfig(
            study_name="composite_loss_test",
            dataset_paths={"cmod": Path("unused_a.nc"), "mast": Path("unused_b.nc")},
            target_device="mast",
        )
    )
    # ds_source_to_idx sorts the dataset names: cmod -> 0, mast -> 1
    loss_config = {**BASE_LOSS_CONFIG, "device_weights": {"cmod": 1.0, "mast": 5.0}}
    inst = PowerBalanceTRB.get_loss_fn(loss_config).instantaneous_loss
    targ = _make_targets(4, ds_source_idx=[0.0, 0.0, 1.0, 1.0])

    # P_oh anchor error is 1.0 everywhere, so the term is
    # anchor_weight * mean(sample_weights) = 0.1 * mean([1, 1, 5, 5])
    loss = inst(_make_output(4, p_oh=3.0, p_rad=1.0), targ)

    assert np.isclose(float(loss), 0.1 * 3.0, atol=1e-6)


##################
# Module outputs #
##################


def _scalar_inputs() -> PowerBalance.Inputs:
    return PowerBalance.Inputs(
        Ip_MA=jnp.asarray(1.0),
        B0=jnp.asarray(5.0),
        R0=jnp.asarray(1.7),
        a_minor=jnp.asarray(0.5),
        kappa=jnp.asarray(1.7),
        ne20_line_avg=jnp.asarray(1.5),
        P_aux_MW=jnp.asarray(2.0),
        ds_source_idx=jnp.asarray(0),
    )


def _make_submodules() -> tuple[OhmicPower, RadiatedPower]:
    p_oh = OhmicPower.init(
        in_size=7, out_size=1, nn_width=4, nn_depth=1, min_val=0, max_val=16, prng_seed=1, normalizer=make_normalizer("raw", None, 1)
    )
    p_rad = RadiatedPower.init(
        in_size=7, out_size=1, nn_width=4, nn_depth=1, min_val=0, max_val=16, prng_seed=2, normalizer=make_normalizer("raw", None, 1)
    )
    return p_oh, p_rad


def test_structured_outputs_carry_submodule_predictions():
    p_oh, p_rad = _make_submodules()
    inputs = _scalar_inputs()
    state = PowerBalance.State(Wtot_MJ=jnp.asarray(0.1))

    expected_p_oh = p_oh(inputs.to_normalizer_inputs()).P_oh_MW_pred
    expected_p_rad = p_rad(inputs.to_normalizer_inputs()).P_rad_MW_pred

    sciml = PowerBalanceSciML.init(
        in_size=7,
        out_size=1,
        nn_width=4,
        nn_depth=1,
        p_oh_predictor=p_oh,
        p_rad_predictor=p_rad,
        normalizer=make_normalizer("raw", None, 1),
    )
    scalinglaw = PowerBalanceScalingLaw.init(p_oh_predictor=p_oh, p_rad_predictor=p_rad)
    for module in (sciml, scalinglaw):
        _, output = module(state, inputs)
        assert np.isclose(float(output.P_oh_MW_pred), float(expected_p_oh))
        assert np.isclose(float(output.P_rad_MW_pred), float(expected_p_rad))

    mlp = PowerBalanceUnstructuredNN.init(in_size=7, out_size=1, nn_width=4, nn_depth=1, normalizer=make_normalizer("raw", None, 1))
    _, output = mlp(state, inputs)
    assert np.isnan(float(output.P_oh_MW_pred))
    assert np.isnan(float(output.P_rad_MW_pred))

    transformer = PowerBalanceTransformer.init(
        d_model=8, num_heads=2, history_len=4, nn_width=4, nn_depth=1, normalizer=make_normalizer("raw", None, 1)
    )
    transformer_state = PowerBalanceTransformer.State(Wtot_MJ=jnp.asarray(0.1), history=jnp.zeros((4, 8)))
    _, output = transformer(transformer_state, inputs)
    assert np.isnan(float(output.P_oh_MW_pred))
    assert np.isnan(float(output.P_rad_MW_pred))


#####################
# Grouped optimizer #
#####################


class FakeSubmodule(eqx.Module):
    nn: eqx.nn.MLP


class FakeModule(eqx.Module):
    taue_predictor: FakeSubmodule
    p_oh_predictor: FakeSubmodule
    p_rad_predictor: FakeSubmodule


OPTIMIZER_CONFIG = {
    "lr0": 1e-3,
    "lrf": 1e-5,
    "transition_steps": 100,
    "decay_rate": 0.9,
    "weight_decay": 0.0,
    "submodule_lr_factors": {"p_oh_predictor": 0.1, "p_rad_predictor": 0.1},
}


def _make_fake_module() -> FakeModule:
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    return FakeModule(
        taue_predictor=FakeSubmodule(eqx.nn.MLP(2, 1, 8, 2, key=k1)),
        p_oh_predictor=FakeSubmodule(eqx.nn.MLP(2, 1, 8, 2, key=k2)),
        p_rad_predictor=FakeSubmodule(eqx.nn.MLP(2, 1, 8, 2, key=k3)),
    )


def _first_step_updates(optimizer, trainable):
    opt_state = optimizer.init(trainable)
    grads = jax.tree.map(jnp.ones_like, trainable)
    updates, _ = optimizer.update(grads, opt_state, trainable)
    return updates


def test_grouped_adamw_scales_submodule_updates():
    # Adam normalizes the first-step update to the learning rate, so the
    # update ratio between groups is exactly the configured factor
    trainable, _ = make_partition_by_members(lambda m: m)(_make_fake_module())
    updates = _first_step_updates(make_grouped_exponential_adamw(OPTIMIZER_CONFIG), trainable)

    u_taue = jnp.abs(updates.taue_predictor.nn.layers[0].weight).mean()
    for submodule in ("p_oh_predictor", "p_rad_predictor"):
        u_sub = jnp.abs(getattr(updates, submodule).nn.layers[0].weight).mean()
        assert np.isclose(float(u_sub / u_taue), 0.1, atol=1e-3)


def test_grouped_adamw_falls_back_to_plain_adamw():
    trainable, _ = make_partition_by_members(lambda m: m)(_make_fake_module())

    # The grouped state is a partition keyed by label, the plain adamw state is not
    grouped_state = make_grouped_exponential_adamw(OPTIMIZER_CONFIG).init(trainable)
    assert hasattr(grouped_state, "inner_states")

    for optimizer_config in (
        {k: v for k, v in OPTIMIZER_CONFIG.items() if k != "submodule_lr_factors"},
        {**OPTIMIZER_CONFIG, "submodule_lr_factors": {}},
        {**OPTIMIZER_CONFIG, "submodule_lr_factors": {"p_oh_predictor": 1.0, "p_rad_predictor": 1.0}},
    ):
        plain_state = make_grouped_exponential_adamw(optimizer_config).init(trainable)
        assert not hasattr(plain_state, "inner_states")


def test_grouped_adamw_labels_transfer_partition():
    # Transfer mode trains only last-layer leaves, the rest of the pytree is
    # None. Path labeling must still find the submodule leaves.
    model = _make_fake_module()
    partition_fn = make_partition_by_members(
        lambda m: [
            m.taue_predictor.nn.layers[-1].weight,
            m.p_oh_predictor.nn.layers[-1].weight,
        ]
    )
    trainable, _ = partition_fn(model)
    updates = _first_step_updates(make_grouped_exponential_adamw(OPTIMIZER_CONFIG), trainable)

    u_taue = jnp.abs(updates.taue_predictor.nn.layers[-1].weight).mean()
    u_p_oh = jnp.abs(updates.p_oh_predictor.nn.layers[-1].weight).mean()
    assert np.isclose(float(u_p_oh / u_taue), 0.1, atol=1e-3)


def test_base_optimizer_config_carries_lr_factors():
    working_dir_base = os.environ.get("PTPS_TEST_WORKING_DIR_BASE", PACKAGE_ROOT / "tests" / "test_outputs")
    study = PowerBalanceStudy(
        PowerBalanceStudy.Config(
            study_name="composite_optimizer_test",
            working_dir_base=Path(working_dir_base) / "power_balance_transfer",
            dataset_paths={
                "cmod-low1": SAMPLE_DIR / "cmod-low1.nc",
                "cmod-high": SAMPLE_DIR / "cmod-high.nc",
            },
            target_device="cmod-high",
            model_types=("sciml-taue-nn",),
            training_datasets=("cmod-low1",),
            data_normalization_methods=("raw",),
            domain_adaptation_methods=(None,),
            freeze_submodules_options=(True,),
            num_target_shots_options=(HYPERPARAM_TARGET_SHOTS,),
            target_test_set_size=4,
        )
    )
    case = next(c for c in study.cases if c.model_type == "sciml-taue-nn")
    train_config = study.make_train_config(case)

    expected = {"p_oh_predictor": 0.1, "p_rad_predictor": 0.1}
    assert train_config.optimizer_config["submodule_lr_factors"] == expected
