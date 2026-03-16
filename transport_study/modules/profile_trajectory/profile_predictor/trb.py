from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from popsim.ml import DataLoader, TrainRunBuilder

from transport_study.modules.profile_trajectory.profile_predictor.module import (
    ProfilePredictorDirectPoints,
    ProfilePredictorShapeInit,
    ShapeType,
    kmeans_initial_guess,
    pca_initial_guess,
)


class ProfilePredictorTRB(TrainRunBuilder):
    """Training run builder for the profile predictor module,
    based on `popsim.modules.profile_predictor.training_run_builder.ProfilePredictorTrainRunBuilder`
    """

    @staticmethod
    def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
        raise NotImplementedError(
            "The profile predictor will only ever be trained as a submodule of the trajectory optimization, so this method is not implemented."
        )

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        """
        Instantiate and return your model given a training DataLoader
        and a model config dict.
        """
        if model_init_config["model_type"] == "shape_init":
            shape_type = model_init_config["shape_type"]
            te_shape_var = model_init_config["te_shape_var"]
            ne_shape_var = model_init_config["ne_shape_var"]
            n_shapes = model_init_config["n_shapes"]

            module = ProfilePredictorShapeInit.init(
                n_shapes=model_init_config["n_shapes"],
                psigrid=train_dl.ds["psi"].data,
                nn_width=model_init_config["nn_width"],
                nn_depth=model_init_config["nn_depth"],
                shape_type=shape_type,
                softmax_temp=model_init_config["softmax_temp"],
                prng_seed=model_init_config["prng_seed"],
            )

            # PCA/K-means initial guess for the shapes.
            ds = train_dl.ds
            sample_dim = train_dl.dataset.training_metadata.sample_dim

            if shape_type == ShapeType.PCA_LIKE:
                te_shapes, ne_shapes = pca_initial_guess(
                    n_shapes, ds[te_shape_var], ds[ne_shape_var], sample_dim
                )
            elif shape_type == ShapeType.CONVEX_COMBINATION:
                te_shapes, ne_shapes = kmeans_initial_guess(
                    n_shapes, ds[te_shape_var], ds[ne_shape_var], sample_dim
                )
            else:
                raise ValueError(f"Invalid shape type: {shape_type}")

            # Overwrite the initial shapes in the module with the initial guess.
            module = eqx.tree_at(
                lambda m: (m.te_shapes, m.ne_shapes),
                module,
                (te_shapes, ne_shapes),
            )
        elif model_init_config["model_type"] == "direct_points":
            module = ProfilePredictorDirectPoints(
                n_points=model_init_config["n_points"],
                nn_width=model_init_config["nn_width"],
                nn_depth=model_init_config["nn_depth"],
                psigrid=train_dl.ds["psi"].data,
                key=jax.random.PRNGKey(model_init_config["prng_seed"]),
            )
        else:
            raise ValueError(
                f"Invalid model type {model_init_config['model_type']}, must be either ProfilePredictorShapeInit or ProfilePredictorDirectPoints"
            )

        return module

    @staticmethod
    def get_loss_fn(config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        def loss_fn(pred, targ):
            ne_psi_loss = jnp.trapezoid(
                optax.huber_loss(
                    pred.ne.data, targ["ne20_psi"].data, delta=config["huber_delta"]
                ),
                x=pred.ne.psi.data,
            )
            te_psi_loss = jnp.trapezoid(
                optax.huber_loss(
                    pred.te.data, targ["Te_keV_psi"].data, delta=config["huber_delta"]
                ),
                x=pred.te.psi.data,
            )
            return ne_psi_loss + te_psi_loss

        return loss_fn

    @staticmethod
    def get_optimizer(config: dict) -> optax.GradientTransformation:
        schedule = optax.exponential_decay(
            init_value=config["lr0"],
            transition_steps=config["transition_steps"],
            decay_rate=config["decay_rate"],
            end_value=config["lrf"],
        )
        opt = optax.adamw(learning_rate=schedule, weight_decay=config["weight_decay"])
        return opt

    @staticmethod
    def get_trainable_getter(model_init_config: dict) -> Callable[[Any], Any] | None:
        """Optionally return a function that takes in the trainable parameters of your model and returns the trainable parameters."""

        def get_trainable_shape_init(module: ProfilePredictorShapeInit):
            # Get all leaves that are not a part of te_shapes and ne_shapes.
            # All of these leaves are trainable.
            ids_of_shape_leaves = [
                id(x) for x in jax.tree.leaves((module.te_shapes, module.ne_shapes))
            ]
            return [
                x for x in jax.tree.leaves(module) if id(x) not in ids_of_shape_leaves
            ]

        def get_trainable_direct_points(module: ProfilePredictorDirectPoints):
            ids_of_nn_leaves = [id(x) for x in jax.tree.leaves(module.nn)]
            return [x for x in jax.tree.leaves(module) if id(x) in ids_of_nn_leaves]

        if model_init_config["model_type"] == "shape_init":
            return get_trainable_shape_init
        elif model_init_config["model_type"] == "direct_points":
            return get_trainable_direct_points
        else:
            raise ValueError(
                f"Invalid model type {model_init_config['model_type']}, must be either 'shape_init' or 'direct_points'"
            )
