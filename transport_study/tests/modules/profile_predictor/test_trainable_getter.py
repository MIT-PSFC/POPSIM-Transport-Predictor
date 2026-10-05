"""Which profile predictor leaves a transfer case fine-tunes (ProfilePredictorTRB.get_trainable_getter)."""

import jax
import pytest

from transport_study.config import RHO_GRID
from transport_study.modules.normalization import CoralFeatureNormalizer
from transport_study.modules.profile_predictor.module import (
    N_NN_INPUTS,
    ProfilePredictorReservoir,
    ProfilePredictorUnstructuredNN,
)
from transport_study.modules.profile_predictor.trb import ProfilePredictorTRB

MODULE_BUILDERS = {
    "mlp": lambda: ProfilePredictorUnstructuredNN(
        nn_width=4,
        nn_depth=2,
        rhogrid=tuple(RHO_GRID.tolist()),
        key=jax.random.PRNGKey(0),
        normalizer=CoralFeatureNormalizer.identity(2, N_NN_INPUTS),
    ),
    "reservoir": lambda: ProfilePredictorReservoir(
        reservoir_size=16,
        rhogrid=tuple(RHO_GRID.tolist()),
        key=jax.random.PRNGKey(0),
        normalizer=CoralFeatureNormalizer.identity(2, N_NN_INPUTS),
    ),
}


@pytest.mark.parametrize("model_type", list(MODULE_BUILDERS))
def test_transfer_trains_only_the_last_layer(model_type):
    """The reservoir weights and the normalizer statistics restored from the pretrain stay frozen, as do the earlier layers."""
    module = MODULE_BUILDERS[model_type]()
    get_trainable = ProfilePredictorTRB.get_trainable_getter({"model_type": model_type, "domain_adaptation": "transfer"})

    trainable = get_trainable(module)

    last_layer_ids = {id(leaf) for leaf in jax.tree.leaves(module.nn.layers[-1])}
    assert {id(leaf) for leaf in trainable} == last_layer_ids
