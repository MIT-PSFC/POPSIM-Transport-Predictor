import copy
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
from popsim.ml.eval import EvalData, EvaluationSuite

from transport_study import EPISODE_DIM, RADIAL_DIM, TIME_COORD, TIME_DIM
from transport_study.config import config
from transport_study.modules.profile_predictor.module import (
    MODEL_TYPES_WITH_SHAPES,
    ProfilePredictorReservoir,
    ProfilePredictorShapeInit,
    ProfilePredictorUnstructuredNN,
    ShapeType,
    kmeans_initial_guess,
    make_nn_input_normalizer,
    pca_initial_guess,
)
from transport_study.modules.profile_predictor.torax_module import ProfilePredictorTorax
from transport_study.modules.trb_utils import (
    CHI_ERROR_VARS,
    PROFILE_SCALE_FLOOR,
    chi_gradient,
    chi_value,
    integrate_error_over_time,
    make_exponential_adamw,
    make_loss_eval_suite,
    peak_scale,
    per_sample_device_values,
    per_sample_sigma_floor,
    resolve_case_datasets,
    target_device_idx,
    to_mid,
)


def resolve_relaxation_overrides(model_init_config: dict) -> dict:
    """Numerics overrides for the sweepable torax relaxation window.

    n_solver_steps fixes the solver step count and derives fixed_dt = t_final / n_solver_steps,
    which decouples the swept relaxation horizon from per-sample solver cost
    (compute scales with step count, not physical time).
    Returns the dict of numerics keys to override, empty when nothing is set.
    """
    t_final = model_init_config.get("t_final")
    fixed_dt = model_init_config.get("fixed_dt")
    n_solver_steps = model_init_config.get("n_solver_steps")
    overrides: dict[str, float] = {}
    if t_final is not None:
        overrides["t_final"] = float(t_final)
    if n_solver_steps is not None:
        if fixed_dt is not None:
            raise ValueError("n_solver_steps and fixed_dt are mutually exclusive, sweep one or the other")
        if t_final is None:
            raise ValueError("n_solver_steps requires t_final to derive fixed_dt")
        overrides["fixed_dt"] = float(t_final) / int(n_solver_steps)
    elif fixed_dt is not None:
        overrides["fixed_dt"] = float(fixed_dt)
    return overrides


class ProfilePredictorTRB(TrainRunBuilder):
    """Training run builder for the profile predictor module,
    based on `popsim.modules.profile_predictor.training_run_builder.ProfilePredictorTrainRunBuilder`
    """

    @staticmethod
    def get_dataloaders(
        dataloader_config: dict,
    ) -> tuple[xr.Dataset, DataLoader, DataLoader, DataLoader]:
        """Dataset and dataloaders for profile predictor training.

        Dataset selection and preparation is the shared
        trb_utils.resolve_case_datasets, this only builds the
        time-independent dataloaders on top.
        """
        ds_train, ds_val, input_vars, normalizer_fit_ds = resolve_case_datasets(dataloader_config, "profile_transfer")

        train_dl, val_dl = make_dataloaders(
            datasets=(ds_train, ds_val),
            time_coord=TIME_COORD,
            episode_coord=EPISODE_DIM,
            input_vars=input_vars,
            target_vars=dataloader_config["target_vars"],
            extra_vars=dataloader_config.get("extra_vars", None),
            batch_size=dataloader_config.get("batch_size", None),
            shuffle=[True, False],
            convert_xr_to_jnp=False,  # Needed to keep the coords for calculating loss
            # Keep every batch the same shape to avoid an extra XLA compilation
            # for the final partial batch, which is expensive for TORAX models.
            # Train drops the ragged tail (reshuffled every epoch, so no data is
            # permanently lost), val pads it and consumers trim the duplicates.
            drop_last=[True, False],
            pad_last=[False, True],
        )
        # Transfer pretrain fits the normalizer on more data than it trains on
        # (historic + target shots). model_init reads this attribute off the
        # train dataloader, every other case fits on train_dl.ds itself
        # TODO(ZanderKeith): Must be *extremely clear* about why you're doing this
        if normalizer_fit_ds is not None:
            train_dl.normalizer_fit_ds = normalizer_fit_ds
        # The validation set doubles as the test set (see resolve_case_datasets)
        return ds_val, train_dl, val_dl, val_dl

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        """
        Instantiate and return your model given a training DataLoader
        and a model config dict.
        """
        # Stat stage (CORAL or z-score) on the dimensionless nn_inputs.
        # Fitted from the training data only. When a transfer checkpoint will overwrite
        # the module anyway, skip the fit (a fit on a handful of target shots is
        # ill-conditioned and the restored stats, fitted on historic + target
        # shots by the transfer_pretrain prereq case, are the correct ones).
        # transfer_pretrain dataloaders carry that combined fit dataset as an
        # attribute (see get_dataloaders).
        # skip_data_init is the same escape hatch for a caller building this
        # module as a submodule skeleton on a dataset that cannot support the
        # data-driven init at all - the transport study's sciml model type builds
        # one on its own dataloader, which carries neither a measured beta_tor_norm nor
        # the profile shape variables (see transport_predictor/trb.py)
        n_devices = len(config.ds_source_to_idx)
        skip_data_init = model_init_config.get("skip_data_init", False)
        if model_init_config.get("transfer_checkpoint") or skip_data_init:
            fit_ds = None
        else:
            fit_ds = getattr(train_dl, "normalizer_fit_ds", train_dl.ds)
        normalizer = make_nn_input_normalizer(model_init_config["data_normalization"], fit_ds, n_devices, target_device_idx())

        if model_init_config["model_type"] in MODEL_TYPES_WITH_SHAPES:
            te_shape_var = model_init_config["te_shape_var"]
            ne_shape_var = model_init_config["ne_shape_var"]
            n_shapes = model_init_config["n_shapes"]

            if model_init_config["model_type"] == "shape-init-pca":
                shape_type = ShapeType.PCA_LIKE
            elif model_init_config["model_type"] == "shape-init-kmeans":
                shape_type = ShapeType.CONVEX_COMBINATION

            module = ProfilePredictorShapeInit.init(
                n_shapes=model_init_config["n_shapes"],
                rhogrid=np.asarray(train_dl.ds[RADIAL_DIM]),
                nn_width=model_init_config["nn_width"],
                nn_depth=model_init_config["nn_depth"],
                in_size=model_init_config["in_size"],
                shape_type=shape_type,
                softmax_temp=model_init_config["softmax_temp"],
                prng_seed=model_init_config["prng_seed"],
                normalizer=normalizer,
            )

            # PCA/K-means initial guess for the shapes.
            # Skipped when restoring from a transfer checkpoint, which overwrites the
            # shapes anyway and whose fine-tune dataset may have fewer samples than shapes.
            # Also skipped for a restore-bound submodule skeleton (skip_data_init),
            # whose dataloader need not carry the shape variables at all.
            if not model_init_config.get("transfer_checkpoint", False) and not skip_data_init:
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
        elif model_init_config["model_type"] == "mlp":
            module = ProfilePredictorUnstructuredNN(
                nn_width=model_init_config["nn_width"],
                nn_depth=model_init_config["nn_depth"],
                rhogrid=np.asarray(train_dl.ds[RADIAL_DIM]),
                key=jax.random.PRNGKey(model_init_config["prng_seed"]),
                normalizer=normalizer,
            )
        elif model_init_config["model_type"] == "reservoir":
            module = ProfilePredictorReservoir(
                reservoir_size=model_init_config["reservoir_size"],
                spectral_radius=model_init_config.get("spectral_radius", 0.9),
                input_scaling=model_init_config.get("input_scaling", 0.5),
                leak_rate=model_init_config.get("leak_rate", 1.0),
                n_steps=model_init_config.get("n_steps", 20),
                rhogrid=np.asarray(train_dl.ds[RADIAL_DIM]),
                key=jax.random.PRNGKey(model_init_config["prng_seed"]),
                normalizer=normalizer,
            )
        elif model_init_config["model_type"].startswith("torax-"):
            # model_type is "torax-<transport_model>", e.g. "torax-gyrobohm"
            torax_config = model_init_config["torax_config"]
            numerics_overrides = resolve_relaxation_overrides(model_init_config)
            if numerics_overrides:
                if not isinstance(torax_config, dict):
                    raise ValueError("t_final / fixed_dt / n_solver_steps overrides require torax_config as a dict")
                torax_config = copy.deepcopy(torax_config)
                torax_config["numerics"].update(numerics_overrides)
            module = ProfilePredictorTorax(
                nn_width=model_init_config["nn_width"],
                nn_depth=model_init_config["nn_depth"],
                rhogrid=np.asarray(train_dl.ds[RADIAL_DIM]),
                torax_config=torax_config,
                key=jax.random.PRNGKey(model_init_config["prng_seed"]),
                normalizer=normalizer,
                transport_model=model_init_config["model_type"].removeprefix("torax-"),
                geometry_builder=model_init_config["geometry_builder"],
                delta_exponent=model_init_config["delta_exponent"],
            )
        else:
            raise ValueError(f"Invalid model type {model_init_config['model_type']}")

        if model_init_config.get("transfer_checkpoint", False):
            transfer_manager = create_default_checkpoint_manager(model_init_config["transfer_checkpoint"])
            module = restore_model(transfer_manager, module)
            logger.debug(f"Restoring module from transfer learning pretrained checkpoint\n{model_init_config['transfer_checkpoint']}")

        return module

    # Softening fraction for the relative-error denominator in study_results.
    # The floor added to abs(targ) is this fraction of the profile's own peak (per timeslice, floored by PROFILE_SCALE_FLOOR),
    # so it reads as the same relative amount for ne and Te on every device
    REL_ERROR_FLOOR_FRAC = 0.1

    # The training gradient loss only applies below this rho.
    # Beyond it the GP fits extrapolate into the pedestal and scrape-off layer, where the measured gradients are unreliable
    GRAD_LOSS_RHO_MAX = 0.9

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        """Training loss: huber on the peak-normalized residual of the values and of the gradients.

        Each target profile is scaled so its peak is 1 and the prediction by the same per-sample scale,
        so the channels and devices are commensurate and huber_delta / huber_delta_grad read as fractional errors.
        The gradient targets are the GP-fit gradients, masked to rho below GRAD_LOSS_RHO_MAX.
        The swept deltas make this loss unfit for the sweep metric, validation uses get_val_loss_fn.
        """
        device_weights = loss_config.get("device_weights", {})
        gradient_weight = loss_config["gradient_weight"]
        huber_delta = loss_config["huber_delta"]
        # Normalized gradients are larger than normalized values (a pedestal can reach d/drho of order 10),
        # so the gradient huber transition has its own delta
        huber_delta_grad = loss_config["huber_delta_grad"]

        def loss_fn(pred, targ):
            rho = pred.ne[RADIAL_DIM].data
            rho_mid = to_mid(rho)
            mask_grad_rho = rho_mid < ProfilePredictorTRB.GRAD_LOSS_RHO_MAX
            sample_weights = per_sample_device_values(targ["ds_source_idx"].data, device_weights, 1.0)
            loss = 0.0
            for channel, pred_channel in (("n_e_1e20", pred.ne.data), ("t_e_keV", pred.te.data)):
                targ_channel = targ[channel].data
                scale = peak_scale(targ_channel)
                value_err = optax.huber_loss(pred_channel / scale - targ_channel / scale, delta=huber_delta)
                grad_pred = jnp.diff(pred_channel / scale, axis=-1) / jnp.diff(rho)
                grad_targ = to_mid(targ[f"{channel}_gradient"].data) / scale
                grad_err = optax.huber_loss(grad_pred - grad_targ, delta=huber_delta_grad)
                loss = loss + jnp.trapezoid(value_err, x=rho, axis=-1)
                loss = loss + gradient_weight * jnp.trapezoid(mask_grad_rho * grad_err, x=rho_mid, axis=-1)
            return sample_weights * loss

        return loss_fn

    @staticmethod
    def get_val_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        """Validation loss: chi, the residual in units of the GP-fit error bar (trb_utils.chi_value / chi_gradient).

        Summed over ne and Te, value chi plus gradient_weight times gradient chi, device-weighted.
        The error bars are floored per device at chi_sigma_floors from the loss config.
        Chi reads no huber delta, so the sweep metric val/loss.mean cannot be gamed by shrinking one.
        """
        device_weights = loss_config.get("device_weights", {})
        gradient_weight = loss_config["gradient_weight"]
        sigma_floors = loss_config["chi_sigma_floors"]

        def loss_fn(pred, targ):
            rho = pred.ne[RADIAL_DIM].data
            ds_source_idx = targ["ds_source_idx"].data
            sample_weights = per_sample_device_values(ds_source_idx, device_weights, 1.0)
            loss = 0.0
            for channel, pred_channel in (("n_e_1e20", pred.ne.data), ("t_e_keV", pred.te.data)):
                targ_channel = targ[channel].data
                value_error_var, grad_error_var = CHI_ERROR_VARS[channel]
                value_floor = per_sample_sigma_floor(ds_source_idx, sigma_floors, value_error_var)
                grad_floor = per_sample_sigma_floor(ds_source_idx, sigma_floors, grad_error_var)
                loss = loss + chi_value(pred_channel, targ_channel, targ[value_error_var].data, value_floor, rho)
                loss = loss + gradient_weight * chi_gradient(
                    pred_channel,
                    targ_channel,
                    targ[f"{channel}_gradient"].data,
                    targ[grad_error_var].data,
                    grad_floor,
                    rho,
                )
            return sample_weights * loss

        return loss_fn

    @staticmethod
    def get_optimizer(optimizer_config: dict) -> optax.GradientTransformation:
        # Standard AdamW plus a global-norm cap
        # the differentiated TORAX solve can spike gradients and NaN a run without it
        return optax.chain(
            optax.clip_by_global_norm(optimizer_config.get("grad_clip_max_norm", 1.0)),
            make_exponential_adamw(optimizer_config),
        )

    @staticmethod
    def get_trainable_getter(model_init_config: dict) -> Callable[[Any], Any] | None:
        """Optionally return a function that takes in the trainable parameters of your model and returns the trainable parameters."""

        def get_trainable_shape_init(module: ProfilePredictorShapeInit):
            # The normalizer statistics are never trainable,
            # the shapes only trainable when freeze_shapes is off
            # Everything else trains.
            if model_init_config["freeze_shapes"]:
                frozen = (module.te_shapes, module.ne_shapes, module.normalizer)
            else:
                frozen = (module.normalizer,)
            ids_of_frozen_leaves = [id(x) for x in jax.tree.leaves(frozen)]
            return [x for x in jax.tree.leaves(module) if id(x) not in ids_of_frozen_leaves]

        def get_trainable_nn(module: ProfilePredictorUnstructuredNN):
            ids_of_nn_leaves = [id(x) for x in jax.tree.leaves(module.nn)]
            return [x for x in jax.tree.leaves(module) if id(x) in ids_of_nn_leaves]

        def get_trainable_torax(module: ProfilePredictorTorax):
            ids_of_nn_leaves = [id(x) for x in jax.tree.leaves((module.nn_transport, module.nn_sources, module.nn_edge))]
            return [x for x in jax.tree.leaves(module) if id(x) in ids_of_nn_leaves]

        def _networks(module) -> tuple:
            """The module's networks, whatever the family calls them."""
            if isinstance(module, ProfilePredictorTorax):
                return (module.nn_transport, module.nn_sources, module.nn_edge)
            return (module.nn,)

        def get_trainable_transfer(module):
            # Transfer fine-tunes the last layer of every network and nothing
            # else, matching the power balance study (PowerBalanceEnv.get_trainable).
            # Shapes and reservoir weights stay frozen regardless of
            # freeze_shapes, and the normalizer statistics restored from the
            # pretrain checkpoint are never touched.
            last_layers = tuple(nn.layers[-1] for nn in _networks(module))
            ids_of_last_layer_leaves = [id(x) for x in jax.tree.leaves(last_layers)]
            return [x for x in jax.tree.leaves(module) if id(x) in ids_of_last_layer_leaves]

        if model_init_config["model_type"] in MODEL_TYPES_WITH_SHAPES:
            getter = get_trainable_shape_init
        elif model_init_config["model_type"] in ["mlp", "reservoir"]:
            # For the reservoir, only the readout (module.nn) is trainable, the
            # fixed random reservoir weights stay frozen
            getter = get_trainable_nn
        elif model_init_config["model_type"].startswith("torax-"):
            getter = get_trainable_torax
        else:
            raise ValueError(f"Invalid model type {model_init_config['model_type']}")

        if model_init_config.get("domain_adaptation") == "transfer":
            return get_trainable_transfer
        return getter

    @staticmethod
    def get_val_eval_suite(suite_config) -> EvaluationSuite:
        """Evaluation suite for validation during training."""
        if suite_config is None:
            return None

        # Shares device_weights / gradient_weight with the training loss_config,
        # but scores chi, which no swept delta can shrink
        return make_loss_eval_suite(ProfilePredictorTRB.get_val_loss_fn(suite_config["loss_config"]))

    @staticmethod
    def get_test_eval_suite(suite_config) -> EvaluationSuite:
        """Evaluation suite for testing after training."""
        return ProfilePredictorTRB.make_test_eval_suite(suite_config, fresh_only=False)

    @staticmethod
    def make_test_eval_suite(suite_config, fresh_only: bool) -> EvaluationSuite:
        """The test suite, with fresh_only scoring only the timeslices whose target is a fresh profile.

        The transport study keeps forward-filled timeslices for contiguous rollouts,
        its targets there are stale measurements, so it scores with fresh_only.
        """

        def study_results(eval_data: EvalData) -> xr.Dataset:
            """Calculate final study results for profile prediction.
                - Target vs predicted n_e_1e20 and t_e_keV profiles
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
                # Time-dependent dataloaders suffix every input-side dim
                # (the transport study reuses this suite)
                # rename them back (time_idx_input -> time_idx, rho_tor_norm_input -> rho_tor_norm)
                # so targets and predictions share one grid
                # No-op for time-independent evals.
                renames = {d: d.removesuffix("_input") for d in da.dims if isinstance(d, str) and d.endswith("_input")}
                if renames:
                    da = da.rename(renames)
                return da

            def _get_first_present(ds: xr.Dataset, candidates: list[str]) -> xr.DataArray:
                for name in candidates:
                    if name in ds.data_vars:
                        return ds[name]
                raise KeyError(f"None of the candidate output vars were found: {candidates}")

            def _ensure_rho_dim(pred: xr.DataArray, targ: xr.DataArray) -> xr.DataArray:
                # Time-dependent stepper outputs are bare arrays whose profile
                # axis gets a generic auto-generated dim name
                # (the transport study reuses this suite)
                # Identify it as the one dim the target does not have and rename it,
                # so the error math never silently outer-broadcasts pred rho against targ rho
                if RADIAL_DIM in pred.dims:
                    return pred
                extra = [d for d in pred.dims if d not in targ.dims]
                if len(extra) != 1 or pred.sizes[extra[0]] != targ.sizes[RADIAL_DIM]:
                    raise ValueError(f"Cannot identify the profile axis of the prediction, dims {pred.dims} vs target {targ.dims}")
                return pred.rename({extra[0]: RADIAL_DIM}).assign_coords({RADIAL_DIM: targ[RADIAL_DIM].values})

            # Targets from input dataset
            ne_targ = _unstack_and_rename_time(eval_data.input_ds["n_e_1e20"])
            te_targ = _unstack_and_rename_time(eval_data.input_ds["t_e_keV"])
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
            ne_pred = _ensure_rho_dim(_unstack_and_rename_time(ne_pred_raw), ne_targ)
            te_pred = _ensure_rho_dim(_unstack_and_rename_time(te_pred_raw), te_targ)

            # Per-point profile errors
            ne_error_abs_profile = xr.apply_ufunc(np.abs, ne_pred - ne_targ)
            te_error_abs_profile = xr.apply_ufunc(np.abs, te_pred - te_targ)
            if fresh_only:
                # Stale timeslices score NaN, every error below and the shot integrals then skip them
                mask_fresh = _unstack_and_rename_time(eval_data.input_ds["fresh_profile"]) == 1
                ne_error_abs_profile = ne_error_abs_profile.where(mask_fresh)
                te_error_abs_profile = te_error_abs_profile.where(mask_fresh)

            # Softening floor scales with each profile's own peak (per timeslice)
            # instead of a fixed absolute value, so it means the same relative
            # amount for ne and Te on every device (see REL_ERROR_FLOOR_FRAC)
            floor_frac = ProfilePredictorTRB.REL_ERROR_FLOOR_FRAC
            scale_floor = PROFILE_SCALE_FLOOR
            ne_peak = np.maximum(xr.apply_ufunc(np.abs, ne_targ).max(dim=RADIAL_DIM), scale_floor)
            te_peak = np.maximum(xr.apply_ufunc(np.abs, te_targ).max(dim=RADIAL_DIM), scale_floor)

            ne_error_rel_profile = ne_error_abs_profile / (xr.apply_ufunc(np.abs, ne_targ) + floor_frac * ne_peak)
            te_error_rel_profile = te_error_abs_profile / (xr.apply_ufunc(np.abs, te_targ) + floor_frac * te_peak)

            # Integrate profile error over rho for each timeslice
            ne_error_abs_ts = ne_error_abs_profile.integrate(RADIAL_DIM)
            te_error_abs_ts = te_error_abs_profile.integrate(RADIAL_DIM)
            ne_error_rel_ts = ne_error_rel_profile.integrate(RADIAL_DIM)
            te_error_rel_ts = te_error_rel_profile.integrate(RADIAL_DIM)

            # Combined timeslice errors (equal weighting between ne and Te channels)
            error_abs_ts = 0.5 * (ne_error_abs_ts + te_error_abs_ts)
            error_rel_ts = 0.5 * (ne_error_rel_ts + te_error_rel_ts)

            # Integrate over time for each shot, handling NaN-padded entries safely
            ne_error_abs_shot = integrate_error_over_time(ne_error_abs_ts, time_2d)
            te_error_abs_shot = integrate_error_over_time(te_error_abs_ts, time_2d)
            ne_error_rel_shot = integrate_error_over_time(ne_error_rel_ts, time_2d)
            te_error_rel_shot = integrate_error_over_time(te_error_rel_ts, time_2d)

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
                    "n_e_1e20_targ": ne_targ,
                    "n_e_1e20_pred": ne_pred,
                    "t_e_keV_targ": te_targ,
                    "t_e_keV_pred": te_pred,
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

        if suite_config:
            eval_suite = {
                "study_results": study_results,
            }
            return eval_suite
        else:
            # Hyperparameter tuning, do not run test evals
            return None
