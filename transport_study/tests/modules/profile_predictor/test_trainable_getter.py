"""Which profile predictor leaves a case trains (ProfilePredictorTRB.get_trainable_getter)."""

import jax
import pytest

from transport_study.config import RHO_GRID
from transport_study.modules.normalization import CoralFeatureNormalizer
from transport_study.modules.profile_predictor.module import (
    N_NN_INPUTS,
    ProfilePredictorReservoir,
    ProfilePredictorShapeInit,
    ProfilePredictorUnstructuredNN,
    ShapeType,
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


@pytest.mark.parametrize("freeze_shapes", [True, False])
def test_shape_init_trains_the_network_and_unfrozen_shape_points(freeze_shapes):
    """The network always trains and the shape points only with freeze_shapes off,
    never the rho grid a shape is interpolated on or the normalizer statistics."""
    module = ProfilePredictorShapeInit.init(
        n_shapes=2,
        rhogrid=RHO_GRID,
        nn_width=4,
        nn_depth=2,
        in_size=N_NN_INPUTS,
        shape_type=ShapeType.CONVEX_COMBINATION,
        prng_seed=0,
        normalizer=CoralFeatureNormalizer.identity(2, N_NN_INPUTS),
    )
    get_trainable = ProfilePredictorTRB.get_trainable_getter(
        {"model_type": "shape-init-kmeans", "domain_adaptation": None, "freeze_shapes": freeze_shapes}
    )

    trainable_ids = {id(leaf) for leaf in get_trainable(module)}

    shapes = (*module.te_shapes, *module.ne_shapes)
    expected_leaves = jax.tree.leaves(module.nn)
    if not freeze_shapes:
        expected_leaves += [shape.coeffs for shape in shapes]
    assert trainable_ids == {id(leaf) for leaf in expected_leaves}
    grid_ids = {id(shape.basis.grid) for shape in shapes}
    assert not trainable_ids & grid_ids
