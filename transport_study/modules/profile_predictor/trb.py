import copy
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from popsim.ml import DataLoader, TrainRunBuilder
from popsim.ml.dataloading import make_dataloaders
from popsim.ml.eval import EvalData, EvaluationSuite

from transport_study import EPISODE_DIM, RADIAL_DIM, TIME_COORD
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
    attach_normalizer_fit_ds,
    chi_gradient,
    chi_profile_loss,
    chi_value,
    ds_source_per_shot,
    fixed_steps_train_dataloader,
    huber_profile_loss,
    integrate_error_over_time,
    make_exponential_adamw,
    make_loss_eval_suite,
    normalizer_fit_dataset,
    per_sample_device_values,
    per_sample_sigma_floor,
    resolve_case_datasets,
    restore_from_checkpoint,
    target_device_idx,
    unstack_samples,
)
from transport_study.orchestration.organize_data import merge_parts

# Profile relaxation solver step [s].
# The Pereverzev linear step makes the same progress per step at any dt,
# so the step count alone sets how far the relaxation gets:
# the output is n_solver_steps damped steps from the parabolic initial profiles, never a steady state.
# 25 ms is the step the production relaxation was benchmarked at (8 steps, 0.2 s).
RELAXATION_DT_S = 0.025


def relaxation_numerics(n_solver_steps: int) -> dict:
    """TORAX numerics window of an n_solver_steps relaxation at the fixed RELAXATION_DT_S step."""
    return {"t_final": n_solver_steps * RELAXATION_DT_S, "fixed_dt": RELAXATION_DT_S}


def _per_shot_sigma_floor(
    sigma_floors: dict[str, dict[str, float]], ds_source: np.ndarray, error_var: str, shots: np.ndarray
) -> xr.DataArray:
    """The chi_sigma_floors entry of error_var for each shot's device, on EPISODE_DIM."""
    floors = np.array([sigma_floors[device][error_var] for device in ds_source])
    return xr.DataArray(floors, dims=EPISODE_DIM, coords={EPISODE_DIM: shots})


def _profile_chi(chi_fn: Callable, profiles: tuple[xr.DataArray, ...], floor: xr.DataArray, rho: np.ndarray) -> xr.DataArray:
    """chi_fn (trb_utils.chi_value or chi_gradient) of every timeslice of the profiles, the floor given per shot.

    Evaluated on the CPU, the whole test set at once would otherwise land on the training GPU.
    """

    def chi_of_arrays(*arrays: np.ndarray) -> np.ndarray:
        *profile_arrays, floor_array = arrays
        with jax.default_device(jax.devices("cpu")[0]):
            chi = chi_fn(*profile_arrays, floor_array[..., None], rho)
        return np.asarray(chi)

    core_dims = [[RADIAL_DIM]] * len(profiles)
    return xr.apply_ufunc(chi_of_arrays, *profiles, floor, input_core_dims=[*core_dims, []])


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
        The training parts are merged first, cheap since the profile view keeps only the fresh timeslices of each shot.
        """
        train_parts, ds_val, input_vars, normalizer_fit_ds = resolve_case_datasets(dataloader_config, "profile_transfer")

        train_dl, val_dl = make_dataloaders(
            datasets=(merge_parts(train_parts), ds_val),
            time_coord=TIME_COORD,
            episode_coord=EPISODE_DIM,
            input_vars=input_vars,
            target_vars=dataloader_config["target_vars"],
            extra_vars=dataloader_config.get("extra_vars"),
            batch_size=dataloader_config["batch_size"],
            shuffle=[True, False],
            convert_xr_to_jnp=False,  # Needed to keep the coords for calculating loss
            # Keep every batch the same shape to avoid an extra XLA compilation
            # for the final partial batch, which is expensive for TORAX models.
            # Train drops the ragged tail (reshuffled every epoch, so no data is
            # permanently lost), val pads it and consumers trim the duplicates.
            drop_last=[True, False],
            pad_last=[False, True],
        )
        train_dl = fixed_steps_train_dataloader(train_dl, dataloader_config["steps_per_epoch"])
        attach_normalizer_fit_ds(train_dl, normalizer_fit_ds)
        # The validation set doubles as the test set (see resolve_case_datasets)
        return ds_val, train_dl, val_dl, val_dl

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        """The profile predictor of a case, restored from the transfer checkpoint when one is set.

        The data-driven init (normalizer statistics, PCA / k-means shape guess) is skipped when a restore will overwrite it:
        for a transfer case, and for a skeleton built with skip_data_init on a dataset that cannot support it
        (the transport sciml submodule, see transport_predictor/trb.py).
        """
        skip_data_init = model_init_config.get("skip_data_init", False)
        # Stat stage (CORAL or z-score) on the dimensionless nn_inputs
        normalizer = make_nn_input_normalizer(
            model_init_config["data_normalization"],
            None if skip_data_init else normalizer_fit_dataset(train_dl, model_init_config),
            len(config.ds_source_to_idx),
            target_device_idx(),
        )

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
                prng_seed=model_init_config["prng_seed"],
                normalizer=normalizer,
            )

            # PCA / k-means initial guess for the shapes, unless a restore overwrites them
            # (a transfer fine-tune dataset can also hold fewer samples than shapes)
            if not model_init_config.get("transfer_checkpoint") and not skip_data_init:
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
                spectral_radius=model_init_config["spectral_radius"],
                input_scaling=model_init_config["input_scaling"],
                leak_rate=model_init_config["leak_rate"],
                n_steps=model_init_config["n_steps"],
                rhogrid=np.asarray(train_dl.ds[RADIAL_DIM]),
                key=jax.random.PRNGKey(model_init_config["prng_seed"]),
                normalizer=normalizer,
            )
        elif model_init_config["model_type"].startswith("torax-"):
            # model_type is "torax-<transport_model>", e.g. "torax-gyrobohm"
            torax_config = copy.deepcopy(model_init_config["torax_config"])
            torax_config["numerics"].update(relaxation_numerics(model_init_config["n_solver_steps"]))
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

        if model_init_config.get("transfer_checkpoint"):
            module = restore_from_checkpoint(module, model_init_config["transfer_checkpoint"])
        return module

    # Softening fraction for the relative-error denominator in study_results.
    # The floor added to abs(targ) is this fraction of the profile's own peak (per timeslice, floored by PROFILE_SCALE_FLOOR),
    # so it reads as the same relative amount for ne and Te on every device
    REL_ERROR_FLOOR_FRAC = 0.1

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        """Training loss: huber on the peak-normalized residual of the values and of the gradients (trb_utils.huber_profile_loss).

        Each target profile is scaled so its peak is 1 and the prediction by the same per-sample scale,
        so the channels and devices are commensurate and huber_delta / huber_delta_grad read as fractional errors.
        The gradient targets are the GP-fit gradients, masked to rho below GRAD_RHO_MAX.
        The swept deltas make this loss unfit for the sweep metric, validation uses get_val_loss_fn.
        """
        device_weights = loss_config.get("device_weights", {})

        def loss_fn(pred, targ):
            rho = pred.ne[RADIAL_DIM].data
            sample_weights = per_sample_device_values(targ["ds_source_idx"].data, device_weights, 1.0)
            loss = 0.0
            for channel, pred_channel in (("n_e_1e20", pred.ne.data), ("t_e_keV", pred.te.data)):
                loss = loss + huber_profile_loss(
                    pred_channel,
                    targ[channel].data,
                    targ[f"{channel}_gradient"].data,
                    rho,
                    loss_config["huber_delta"],
                    loss_config["huber_delta_grad"],
                    loss_config["gradient_weight"],
                )
            return sample_weights * loss

        return loss_fn

    @staticmethod
    def get_val_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        """Validation loss: chi, the residual in units of the GP-fit error bar (trb_utils.chi_profile_loss).

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
                loss = loss + chi_profile_loss(
                    pred_channel,
                    targ_channel,
                    targ[value_error_var].data,
                    targ[f"{channel}_gradient"].data,
                    targ[grad_error_var].data,
                    value_floor,
                    grad_floor,
                    rho,
                    gradient_weight,
                )
            return sample_weights * loss

        return loss_fn

    @staticmethod
    def get_optimizer(optimizer_config: dict) -> optax.GradientTransformation:
        # Standard AdamW plus a global-norm cap
        # the differentiated TORAX solve can spike gradients and NaN a run without it
        return optax.chain(
            optax.clip_by_global_norm(optimizer_config["grad_clip_max_norm"]),
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

        if model_init_config["domain_adaptation"] == "transfer":
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
    def get_test_eval_suite(suite_config) -> EvaluationSuite | None:
        """Evaluation suite for testing after training."""
        return ProfilePredictorTRB.make_test_eval_suite(suite_config, fresh_only=False)

    @staticmethod
    def make_test_eval_suite(suite_config, fresh_only: bool) -> EvaluationSuite | None:
        """The test suite, with fresh_only scoring only the timeslices whose target is a fresh profile.

        The transport study keeps forward-filled timeslices for contiguous rollouts,
        its targets there are stale measurements, so it scores with fresh_only.
        The chi reads gradient_weight and chi_sigma_floors from the suite's loss_config.
        None for sweep trials, which run no test evals.
        """
        if not suite_config:
            return None
        gradient_weight = suite_config["loss_config"]["gradient_weight"]
        sigma_floors = suite_config["loss_config"]["chi_sigma_floors"]

        def study_results(eval_data: EvalData) -> xr.Dataset:
            """Calculate final study results for profile prediction.
                - Target vs predicted n_e_1e20 and t_e_keV profiles
                - Per-timeslice profile-integrated absolute/relative errors
                - Per-timeslice chi summed over the channels (error_chi_value_ts, error_chi_grad_ts),
                  and error_chi_ts = value + gradient_weight * grad, the validation loss less the device weight
                - error_diverged_ts: 1 where the prediction went non-finite, 0 where finite, NaN without a target
                - Per-shot time-integrated errors
            Keeps shot and ds_source coordinates for downstream analysis.
            """

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

            ne_targ = unstack_samples(eval_data.input_ds["n_e_1e20"])
            te_targ = unstack_samples(eval_data.input_ds["t_e_keV"])
            time_2d = unstack_samples(eval_data.input_ds[TIME_COORD])
            ds_source = ds_source_per_shot(eval_data.input_ds["ds_source"], ne_targ.sizes[EPISODE_DIM])

            # The time-dependent modules (the transport study) nest their outputs under output.
            output_prefix = "" if "ne" in eval_data.output_ds else "output."
            ne_pred = _ensure_rho_dim(unstack_samples(eval_data.output_ds[f"{output_prefix}ne"]), ne_targ)
            te_pred = _ensure_rho_dim(unstack_samples(eval_data.output_ds[f"{output_prefix}te"]), te_targ)

            # A prediction with any non-finite point in either channel diverged.
            # Its errors are NaN, never inf, and error_diverged_ts counts it instead,
            # outside the freshness mask like the validation divergence penalty
            ne_pred_finite = xr.apply_ufunc(np.isfinite, ne_pred).all(RADIAL_DIM)
            te_pred_finite = xr.apply_ufunc(np.isfinite, te_pred).all(RADIAL_DIM)
            mask_pred_finite = ne_pred_finite & te_pred_finite
            mask_targ_present = ne_targ.notnull().all(RADIAL_DIM) & te_targ.notnull().all(RADIAL_DIM)
            error_diverged_ts = (~mask_pred_finite).astype(float).where(mask_targ_present)
            mask_scored = mask_pred_finite
            if fresh_only:
                # Stale timeslices score NaN, every error below and the shot integrals then skip them
                mask_fresh = unstack_samples(eval_data.input_ds["fresh_profile"]) == 1
                mask_scored = mask_scored & mask_fresh

            # Per-point profile errors
            ne_error_abs_profile = xr.apply_ufunc(np.abs, ne_pred - ne_targ).where(mask_scored)
            te_error_abs_profile = xr.apply_ufunc(np.abs, te_pred - te_targ).where(mask_scored)

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

            # Chi of the validation loss (chi_profile_loss), summed over the channels
            rho = ne_targ[RADIAL_DIM].values
            shots = ne_targ[EPISODE_DIM].values
            chi_value_ts = xr.zeros_like(error_abs_ts)
            chi_grad_ts = xr.zeros_like(error_abs_ts)
            for channel, pred_channel, targ_channel in (("n_e_1e20", ne_pred, ne_targ), ("t_e_keV", te_pred, te_targ)):
                value_error_var, grad_error_var = CHI_ERROR_VARS[channel]
                sigma = unstack_samples(eval_data.input_ds[value_error_var])
                grad_targ = unstack_samples(eval_data.input_ds[f"{channel}_gradient"])
                grad_sigma = unstack_samples(eval_data.input_ds[grad_error_var])
                value_floor = _per_shot_sigma_floor(sigma_floors, ds_source, value_error_var, shots)
                grad_floor = _per_shot_sigma_floor(sigma_floors, ds_source, grad_error_var, shots)
                chi_value_ts = chi_value_ts + _profile_chi(chi_value, (pred_channel, targ_channel, sigma), value_floor, rho)
                chi_grad_ts = chi_grad_ts + _profile_chi(chi_gradient, (pred_channel, targ_channel, grad_targ, grad_sigma), grad_floor, rho)
            error_chi_value_ts = chi_value_ts.where(mask_scored)
            error_chi_grad_ts = chi_grad_ts.where(mask_scored)
            error_chi_ts = error_chi_value_ts + gradient_weight * error_chi_grad_ts

            # Integrate over time for each shot, handling NaN-padded entries safely
            ne_error_abs_shot = integrate_error_over_time(ne_error_abs_ts, time_2d)
            te_error_abs_shot = integrate_error_over_time(te_error_abs_ts, time_2d)
            ne_error_rel_shot = integrate_error_over_time(ne_error_rel_ts, time_2d)
            te_error_rel_shot = integrate_error_over_time(te_error_rel_ts, time_2d)

            error_abs_shot = 0.5 * (ne_error_abs_shot + te_error_abs_shot)
            error_rel_shot = 0.5 * (ne_error_rel_shot + te_error_rel_shot)

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
                    "error_chi_value_ts": error_chi_value_ts,
                    "error_chi_grad_ts": error_chi_grad_ts,
                    "error_chi_ts": error_chi_ts,
                    "error_diverged_ts": error_diverged_ts,
                    "ne_error_abs_shot": ne_error_abs_shot,
                    "te_error_abs_shot": te_error_abs_shot,
                    "ne_error_rel_shot": ne_error_rel_shot,
                    "te_error_rel_shot": te_error_rel_shot,
                    "error_abs_shot": error_abs_shot,
                    "error_rel_shot": error_rel_shot,
                }
            )
            return ds.assign_coords(ds_source=(EPISODE_DIM, ds_source)).drop_vars(["quantile", "input_batch"], errors="ignore")

        return {"study_results": study_results}
