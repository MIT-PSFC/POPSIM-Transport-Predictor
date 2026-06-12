from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from loguru import logger
from popsim.ml import DataLoader, TrainRunBuilder
from popsim.ml.checkpointing import create_default_checkpoint_manager, restore_model
from popsim.ml.dataloading import make_dataloaders
from popsim.ml.eval import EvalData, EvaluationSuite, batched_model_eval_and_loss

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.config import config
from transport_study.modules.profile_predictor.module import (
    ProfilePredictorShapeInit,
    ProfilePredictorUnstructuredNN,
    ShapeType,
    kmeans_initial_guess,
    pca_initial_guess,
)
from transport_study.modules.profile_predictor.torax_module import ProfilePredictorTorax
from transport_study.orchestration.organize_data import (
    TrainingData,
    get_train_test_datasets,
    get_train_val_datasets,
)


class ProfilePredictorTRB(TrainRunBuilder):
    """Training run builder for the profile predictor module,
    based on `popsim.modules.profile_predictor.training_run_builder.ProfilePredictorTrainRunBuilder`
    """

    @staticmethod
    def get_dataloaders(
        dataloader_config: dict,
    ) -> tuple[xr.Dataset, tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Get the dataset and dataloaders for training.

        For both standard learning and transfer learning, we essentially have two datasets.
        For standard learning, it's the standard train/val for hyperparameter tuning. No test is needed, so we can just return None.
        For transfer learning, we have the training dataset composed of all historic data and a small amount of new data,
        and the test dataset composed of a set amount of new data that is held out of training.
        We are not doing hyperparameter tuning for transfer learning.

        Also, slightly different from the POPSIM version, we're just returning the validation dataset.
        """

        training_data = dataloader_config["training_data"]

        if isinstance(training_data, dict):  # when the config is passed from WandB, it's a dict
            training_data = TrainingData(**training_data)

        if dataloader_config.get("domain_adaptation") is None:
            logger.info("Using standard learning dataloader")
            if not training_data.exnihilo:
                ds_train, ds_val = get_train_val_datasets(
                    training_data=training_data,
                    study_type="profile_transfer",
                )
            else:
                ds_train, ds_val = get_train_test_datasets(
                    training_data=training_data,
                    domain_adaptation=None,
                    num_target_shots=dataloader_config["num_target_shots"],
                    target_test_set_size=dataloader_config.get("target_test_set_size", None),
                    study_type="profile_transfer",
                )
                # Double check there's no source (non-target) data anywhere in here
                non_target = set(config.dataset_paths.keys()) - {config.target_device}
                if any((ds_train["ds_source"] == src).any() for src in non_target):
                    raise ValueError(
                        "Historic data found in training set for exnihilo training_data option. Please check the dataset construction logic."
                    )
        else:
            logger.info(f"Using transfer learning dataloader with domain adaptation {dataloader_config['domain_adaptation']}")
            ds_train, ds_val = get_train_test_datasets(
                training_data=training_data,
                domain_adaptation=dataloader_config["domain_adaptation"],
                num_target_shots=dataloader_config["num_target_shots"],
                target_test_set_size=dataloader_config.get("target_test_set_size", None),
                study_type="profile_transfer",
            )

        # Drop time_idx as a shared coordinate — it has duplicate values across shots and
        # causes groupby("shot") to fail when reassembling. The dataloader uses "time" instead.
        ds_train = ds_train.drop_vars(TIME_DIM, errors="ignore")
        ds_val = ds_val.drop_vars(TIME_DIM, errors="ignore")

        input_vars = dataloader_config["input_vars"]
        target_vars = dataloader_config["target_vars"]
        extra_vars = dataloader_config.get("extra_vars", None)

        train_dl, val_dl = make_dataloaders(
            datasets=(ds_train, ds_val),
            time_coord=TIME_COORD,
            episode_coord=EPISODE_DIM,
            input_vars=input_vars,
            target_vars=target_vars,
            extra_vars=extra_vars,
            batch_size=dataloader_config.get("batch_size", None),
            shuffle=[True, False],
            convert_xr_to_jnp=False,  # Needed to keep the coords for calculating loss
        )
        # Running test evaluation on the validation set, since we don't need a dedicated test set
        # In the no domain adaptation case, we are hyperparameter tuning on all historic data, pick the best one and test on it
        # In the domain adaptation case, we are training on all historic data + some new data, and testing on the rest of the new data
        # No hyperparameter tuning is happening, so we treat the validation set as the test set and just return it for evaluation after training
        return ds_val, train_dl, val_dl, val_dl

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        """
        Instantiate and return your model given a training DataLoader
        and a model config dict.
        """
        if model_init_config["model_type"] in ["shape_init_pca", "shape_init_kmeans"]:
            te_shape_var = model_init_config["te_shape_var"]
            ne_shape_var = model_init_config["ne_shape_var"]
            n_shapes = model_init_config["n_shapes"]

            if model_init_config["model_type"] == "shape_init_pca":
                shape_type = ShapeType.PCA_LIKE
            elif model_init_config["model_type"] == "shape_init_kmeans":
                shape_type = ShapeType.CONVEX_COMBINATION

            module = ProfilePredictorShapeInit.init(
                n_shapes=model_init_config["n_shapes"],
                rhogrid=np.asarray(train_dl.ds["rho"]),
                nn_width=model_init_config["nn_width"],
                nn_depth=model_init_config["nn_depth"],
                in_size=model_init_config["in_size"],
                shape_type=shape_type,
                softmax_temp=model_init_config["softmax_temp"],
                prng_seed=model_init_config["prng_seed"],
            )

            # PCA/K-means initial guess for the shapes.
            ds = train_dl.ds
            sample_dim = train_dl.dataset.training_metadata.sample_dim

            if shape_type == ShapeType.PCA_LIKE:
                te_shapes, ne_shapes = pca_initial_guess(n_shapes, ds[te_shape_var], ds[ne_shape_var], sample_dim)
            elif shape_type == ShapeType.CONVEX_COMBINATION:
                te_shapes, ne_shapes = kmeans_initial_guess(n_shapes, ds[te_shape_var], ds[ne_shape_var], sample_dim)
            else:
                raise ValueError(f"Invalid shape type: {shape_type}")

            # Overwrite the initial shapes in the module with the initial guess.
            module = eqx.tree_at(
                lambda m: (m.te_shapes, m.ne_shapes),
                module,
                (te_shapes, ne_shapes),
            )
        elif model_init_config["model_type"] == "unstructured_nn":
            module = ProfilePredictorUnstructuredNN(
                n_points=model_init_config["n_points"],
                nn_width=model_init_config["nn_width"],
                nn_depth=model_init_config["nn_depth"],
                rhogrid=np.asarray(train_dl.ds["rho"]),
                key=jax.random.PRNGKey(model_init_config["prng_seed"]),
            )
        elif model_init_config["model_type"].startswith("torax-"):
            # model_type is "torax-<transport_model>", e.g. "torax-cgm"
            module = ProfilePredictorTorax(
                nn_width=model_init_config["nn_width"],
                nn_depth=model_init_config["nn_depth"],
                rhogrid=np.asarray(train_dl.ds["rho"]),
                torax_config=model_init_config["torax_config"],
                key=jax.random.PRNGKey(model_init_config["prng_seed"]),
                transport_model=model_init_config["model_type"].removeprefix("torax-"),
            )
        else:
            raise ValueError(f"Invalid model type {model_init_config['model_type']}")

        if model_init_config.get("transfer_checkpoint", False):
            transfer_manager = create_default_checkpoint_manager(model_init_config["transfer_checkpoint"])
            module = restore_model(transfer_manager, module)
            logger.debug(f"Restoring module from transfer learning pretrained checkpoint\n{model_init_config['transfer_checkpoint']}")

        return module

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        if "device_weights" not in loss_config:
            device_weights = dict.fromkeys(config.dataset_paths, 1.0)
        else:
            device_weights = loss_config["device_weights"]

        def loss_fn(pred, targ):
            ne_huber = optax.huber_loss(
                pred.ne.data,
                targ["ne20_rho"].data,
                delta=loss_config["huber_delta"],
            )
            te_huber = optax.huber_loss(
                pred.te.data,
                targ["Te_keV_rho"].data,
                delta=loss_config["huber_delta"],
            )

            # Build per-sample device weight
            ds_source_idx = targ["ds_source_idx"].data
            sample_weights = jnp.ones(ds_source_idx.shape, dtype=ne_huber.dtype)
            for device, weight in device_weights.items():
                sample_weights = jnp.where(
                    ds_source_idx == config.ds_source_to_idx[device],
                    weight,
                    sample_weights,
                )

            # Broadcast sample weights across profile/time axes
            while sample_weights.ndim < ne_huber.ndim:
                sample_weights = sample_weights[..., None]

            ne_weighted = sample_weights * ne_huber
            te_weighted = sample_weights * te_huber

            ne_rho_loss = jnp.trapezoid(ne_weighted, x=pred.ne.rho.data)
            te_rho_loss = jnp.trapezoid(te_weighted, x=pred.te.rho.data)

            return ne_rho_loss + te_rho_loss

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
            ids_of_shape_leaves = [id(x) for x in jax.tree.leaves((module.te_shapes, module.ne_shapes))]
            if model_init_config["freeze_shapes"]:
                return [x for x in jax.tree.leaves(module) if id(x) not in ids_of_shape_leaves]
            else:
                return jax.tree.leaves(module)

        def get_trainable_nn(module: ProfilePredictorUnstructuredNN):
            ids_of_nn_leaves = [id(x) for x in jax.tree.leaves(module.nn)]
            return [x for x in jax.tree.leaves(module) if id(x) in ids_of_nn_leaves]

        def get_trainable_torax(module: ProfilePredictorTorax):
            ids_of_nn_leaves = [id(x) for x in jax.tree.leaves((module.nn_transport, module.nn_sources, module.nn_edge))]
            return [x for x in jax.tree.leaves(module) if id(x) in ids_of_nn_leaves]

        if model_init_config["model_type"] in ["shape_init_pca", "shape_init_kmeans"]:
            return get_trainable_shape_init
        elif model_init_config["model_type"] == "unstructured_nn":
            return get_trainable_nn
        elif model_init_config["model_type"].startswith("torax-"):
            return get_trainable_torax
        else:
            raise ValueError(f"Invalid model type {model_init_config['model_type']}")

    @staticmethod
    def get_val_eval_suite(suite_config) -> EvaluationSuite:
        """Evaluation suite for validation during training

        We are running with very large datasets.
        This means that the regular eval function will be uploading too much data to wandb
        This eval suite basically does the same thing but cuts the vec to be at most 100 long
        Can still see the distribution, but without all the data
        """
        # If not specified, return None
        if suite_config is None:
            return None

        # Must be an exact copy of the loss_config used in training
        loss_config = suite_config["loss_config"]
        loss_fn = ProfilePredictorTRB.get_loss_fn(loss_config)

        def eval_fn(inp: EvalData) -> float:
            loss_vecs = []
            for batch in inp.dataloader:
                inputs, targets = batch.get_inputs_and_targets()
                loss_vec = batched_model_eval_and_loss(
                    inp.model,
                    loss_fn,
                    inputs,
                    targets,
                )
                loss_vecs.append(loss_vec)
            loss_vec = jnp.concatenate(loss_vecs)
            loss_vec_mean = loss_vec.mean()
            # Sort loss vec and sample at most 100 points evenly for logging
            if loss_vec.shape[0] > 100:
                sorted_indices = jnp.argsort(loss_vec)
                selected_indices = sorted_indices[jnp.linspace(0, loss_vec.shape[0] - 1, num=100, dtype=int)]
                loss_vec = loss_vec[selected_indices]
            out = {
                "mean": loss_vec_mean,
                "vec": loss_vec,
            }

            return out

        eval_suite = {"loss": eval_fn}

        return eval_suite

    @staticmethod
    def get_test_eval_suite(config) -> EvaluationSuite:
        """Evaluation suite for testing after training."""

        def study_results(eval_data: EvalData) -> xr.Dataset:
            """Calculate final study results for profile prediction.
                - Target vs predicted ne20_rho and Te_keV_rho profiles
                - Per-timeslice profile-integrated absolute/relative errors
                - Per-shot time-integrated errors
            Keeps shot and ds_source coordinates for downstream analysis.
            """

            def _unstack_and_rename_time(da: xr.DataArray) -> xr.DataArray:
                da = da.unstack("sample")
                # Keep episode/time axes even when they have length 1.
                # Single-shot eval splits are valid and should retain the shot dimension.
                protected_dims = {EPISODE_DIM, TIME_DIM, TIME_DIM + "_input"}
                squeeze_dims = [d for d, n in da.sizes.items() if n == 1 and d not in protected_dims]
                if squeeze_dims:
                    da = da.squeeze(dim=squeeze_dims, drop=True)
                if TIME_DIM + "_input" in da.dims:
                    da = da.rename({TIME_DIM + "_input": TIME_DIM})
                return da

            def _get_first_present(ds: xr.Dataset, candidates: list[str]) -> xr.DataArray:
                for name in candidates:
                    if name in ds.data_vars:
                        return ds[name]
                raise KeyError(f"None of the candidate output vars were found: {candidates}")

            # Targets from input dataset
            ne_targ = _unstack_and_rename_time(eval_data.input_ds["ne20_rho"])
            te_targ = _unstack_and_rename_time(eval_data.input_ds["Te_keV_rho"])
            time_2d = _unstack_and_rename_time(eval_data.input_ds[TIME_COORD])

            # Predictions from output dataset.

            ne_pred_raw = _get_first_present(
                eval_data.output_ds,
                ["ne", "output.ne", "output.profile_predictor_output.ne"],
            )
            te_pred_raw = _get_first_present(
                eval_data.output_ds,
                ["te", "output.te", "output.profile_predictor_output.te"],
            )
            ne_pred = _unstack_and_rename_time(ne_pred_raw)
            te_pred = _unstack_and_rename_time(te_pred_raw)

            # Per-point profile errors
            ne_error_abs_profile = xr.apply_ufunc(np.abs, ne_pred - ne_targ)
            te_error_abs_profile = xr.apply_ufunc(np.abs, te_pred - te_targ)

            ne_error_rel_profile = ne_error_abs_profile / (xr.apply_ufunc(np.abs, ne_targ) + 0.1)
            te_error_rel_profile = te_error_abs_profile / (xr.apply_ufunc(np.abs, te_targ) + 0.1)

            # Integrate profile error over rho for each timeslice
            ne_error_abs_ts = ne_error_abs_profile.integrate("rho")
            te_error_abs_ts = te_error_abs_profile.integrate("rho")
            ne_error_rel_ts = ne_error_rel_profile.integrate("rho")
            te_error_rel_ts = te_error_rel_profile.integrate("rho")

            # Combined timeslice errors (equal weighting between ne and Te channels)
            error_abs_ts = 0.5 * (ne_error_abs_ts + te_error_abs_ts)
            error_rel_ts = 0.5 * (ne_error_rel_ts + te_error_rel_ts)

            # Integrate over time for each shot, handling NaN-padded entries safely
            def _trapezoid_dropna(y, x):
                mask = ~np.isnan(x) & ~np.isnan(y)
                if mask.sum() < 2:
                    return np.nan
                y_valid, x_valid = y[mask], x[mask]
                sort_idx = np.argsort(x_valid)
                return np.trapezoid(y_valid[sort_idx], x_valid[sort_idx])

            ne_error_abs_shot = xr.apply_ufunc(
                _trapezoid_dropna,
                ne_error_abs_ts,
                time_2d,
                input_core_dims=[[TIME_DIM], [TIME_DIM]],
                vectorize=True,
            )
            te_error_abs_shot = xr.apply_ufunc(
                _trapezoid_dropna,
                te_error_abs_ts,
                time_2d,
                input_core_dims=[[TIME_DIM], [TIME_DIM]],
                vectorize=True,
            )
            ne_error_rel_shot = xr.apply_ufunc(
                _trapezoid_dropna,
                ne_error_rel_ts,
                time_2d,
                input_core_dims=[[TIME_DIM], [TIME_DIM]],
                vectorize=True,
            )
            te_error_rel_shot = xr.apply_ufunc(
                _trapezoid_dropna,
                te_error_rel_ts,
                time_2d,
                input_core_dims=[[TIME_DIM], [TIME_DIM]],
                vectorize=True,
            )

            error_abs_shot = 0.5 * (ne_error_abs_shot + te_error_abs_shot)
            error_rel_shot = 0.5 * (ne_error_rel_shot + te_error_rel_shot)

            # Keep ds_source coord aligned to shot
            ds_source = eval_data.input_ds["ds_source"]
            if "sample" in ds_source.dims:
                ds_source_unstacked = ds_source.unstack("sample")
                if EPISODE_DIM in ds_source_unstacked.dims and ds_source_unstacked.ndim > 1:
                    other_dims = [d for d in ds_source_unstacked.dims if d != EPISODE_DIM]
                    ds_source_stacked = ds_source_unstacked.stack(_other=other_dims).transpose(EPISODE_DIM, "_other")
                    ds_source_vals = ds_source_stacked.values
                    ds_source_valid = ds_source_stacked.notnull().values
                    ds_source_array = np.array(
                        [
                            row[np.flatnonzero(valid_mask)[0]] if np.any(valid_mask) else np.nan
                            for row, valid_mask in zip(ds_source_vals, ds_source_valid, strict=True)
                        ],
                        dtype=object,
                    )
                else:
                    ds_source_array = ds_source_unstacked.values
            else:
                ds_source_array = np.array([ds_source.values.item() for _ in range(ne_targ.sizes[EPISODE_DIM])])

            ds = xr.Dataset(
                data_vars={
                    "ne20_rho_targ": ne_targ,
                    "ne20_rho_pred": ne_pred,
                    "Te_keV_rho_targ": te_targ,
                    "Te_keV_rho_pred": te_pred,
                    "ne_error_abs_ts": ne_error_abs_ts,
                    "te_error_abs_ts": te_error_abs_ts,
                    "ne_error_rel_ts": ne_error_rel_ts,
                    "te_error_rel_ts": te_error_rel_ts,
                    "error_abs_ts": error_abs_ts,
                    "error_rel_ts": error_rel_ts,
                    "ne_error_abs_shot": ne_error_abs_shot,
                    "te_error_abs_shot": te_error_abs_shot,
                    "ne_error_rel_shot": ne_error_rel_shot,
                    "te_error_rel_shot": te_error_rel_shot,
                    "error_abs_shot": error_abs_shot,
                    "error_rel_shot": error_rel_shot,
                }
            )

            ds = ds.assign_coords(ds_source=(EPISODE_DIM, ds_source_array))
            ds = ds.drop_vars("quantile", errors="ignore")
            ds = ds.drop_vars("input_batch", errors="ignore")
            return ds

        if config:
            eval_suite = {
                "study_results": study_results,
            }
            return eval_suite
        else:
            # Hyperparameter tuning, do not run test evals
            return None
