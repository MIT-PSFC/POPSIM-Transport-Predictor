"""Loss split of the p_oh / p_rad predictors, the same one every other study TRB follows."""

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
import xarray as xr

from transport_study.config import StudyConfig, load_config
from transport_study.modules.power_balance.p_oh.trb import OhmicPowerTRB


@pytest.fixture
def loaded_config():
    return load_config(
        StudyConfig(
            study_name="test-scalar-power-loss",
            dataset_paths={"cmod": Path("path/to/cmod.zarr"), "mast": Path("path/to/mast.zarr")},
            target_device="mast",
        )
    )


def _loss(loss_fn, residual: float, device: str, config) -> float:
    pred = SimpleNamespace(power_ohm_MW_pred=jnp.array([1.0 + residual]))
    targ = {
        "power_ohm_MW": xr.DataArray([1.0]),
        "ds_source_idx": xr.DataArray([float(config.ds_source_to_idx[device])]),
    }
    return float(loss_fn(pred, targ))


@pytest.mark.parametrize("residual", [0.05, 3.0])
def test_device_weight_scales_the_loss_linearly(loaded_config, residual):
    """The weight multiplies the per-sample loss in the quadratic and in the linear huber regime alike."""
    weighted = OhmicPowerTRB.get_loss_fn({"huber_delta": 0.5, "device_weights": {"cmod": 1.0, "mast": 2.0}})
    unweighted = OhmicPowerTRB.get_loss_fn({"huber_delta": 0.5, "device_weights": {"cmod": 1.0, "mast": 1.0}})

    assert _loss(weighted, residual, "mast", loaded_config) == pytest.approx(2.0 * _loss(unweighted, residual, "mast", loaded_config))


def test_val_loss_is_delta_free(loaded_config):
    """The sweep metric reads the validation loss, which must not depend on the swept huber_delta."""
    val_small = OhmicPowerTRB.get_val_loss_fn({"huber_delta": 0.01})
    val_large = OhmicPowerTRB.get_val_loss_fn({"huber_delta": 1.0})
    train_small = OhmicPowerTRB.get_loss_fn({"huber_delta": 0.01})
    train_large = OhmicPowerTRB.get_loss_fn({"huber_delta": 1.0})

    assert _loss(val_small, 0.3, "cmod", loaded_config) == pytest.approx(0.3)
    assert _loss(val_large, 0.3, "cmod", loaded_config) == pytest.approx(0.3)
    assert _loss(train_small, 0.3, "cmod", loaded_config) != pytest.approx(_loss(train_large, 0.3, "cmod", loaded_config))
