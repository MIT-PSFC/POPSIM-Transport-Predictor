from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from loguru import logger
from popsim.ml import DataLoader, IntegralLoss, TrainConfig, TrainRunBuilder
from popsim.ml.checkpointing import create_default_checkpoint_manager, restore_model
from popsim.ml.eval import EvaluationSuite

from transport_study.config import config
from transport_study.modules.power_balance.trb import PowerBalanceTRB
from transport_study.modules.profile_predictor.trb import ProfilePredictorTRB
from transport_study.modules.transport_predictor.module import (
    TransportPredictorEnv,
    TransportPredictorSciML,
    TransportPredictorTorax,
    TransportPredictorToraxSimState,
    TransportPredictorTransformer,
    make_transport_nn_input_normalizer,
)
from transport_study.modules.trb_utils import (
    get_time_dep_dataloaders,
    make_grouped_exponential_adamw,
    make_loss_eval_suite,
)

STUDY_TYPE = "transport_transfer"

# Anchor terms in the sciml training loss, keyed by measured target signal:
# (Output attribute holding the model's own prediction, loss_config key for the weight)
ANCHOR_SIGNALS = {
    "Wtot_MJ": ("Wtot_MJ_pred", "anchor_weight_wtot"),
    "P_oh_MW": ("P_oh_MW_pred", "anchor_weight_p_oh"),
    "P_rad_MW": ("P_rad_MW_pred", "anchor_weight_p_rad"),
}


def _submodule_config_dict(submodule_config: TrainConfig | dict) -> dict:
    if isinstance(submodule_config, TrainConfig):
        return submodule_config.model_dump()
    return submodule_config


class TransportPredictorTRB(TrainRunBuilder):
    """TrainRunBuilder for the transport predictor modules used in transfer learning."""

    @staticmethod
    def get_dataloaders(
        dataloader_config: dict,
    ) -> tuple[xr.Dataset, DataLoader, DataLoader, DataLoader]:
        """Dataset and dataloaders for training, see trb_utils.get_time_dep_dataloaders."""
        return get_time_dep_dataloaders(dataloader_config, STUDY_TYPE)

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        """Instantiate the transport predictor env for a case."""

        def _build_module(train_dl: DataLoader, model_init_config: dict) -> Any:
            model_type = model_init_config["model_type"]
            # Stat stage (CORAL or z-score) on the 11 transport features, fitted
            # from the training data only. When a transfer checkpoint will
            # overwrite the module anyway, skip the fit (a fit on a handful of
            # target shots is ill-conditioned and the restored stats, fitted on
            # historic + target shots by the transfer_pretrain prereq case, are
            # the correct ones). transfer_pretrain dataloaders carry that
            # combined fit dataset as an attribute (see get_time_dep_dataloaders)
            n_devices = len(config.ds_source_to_idx)
            if model_init_config.get("transfer_checkpoint"):
                fit_ds = None
            else:
                fit_ds = getattr(train_dl, "normalizer_fit_ds", train_dl.ds)
            normalizer = make_transport_nn_input_normalizer(model_init_config["data_normalization"], fit_ds, n_devices)

            if model_type == "sciml":
                # Submodule skeletons come from their own TRBs, then their
                # trained weights are restored from the prereq case checkpoints
                pb_config = _submodule_config_dict(model_init_config["submodules"]["power_balance"])
                prof_config = _submodule_config_dict(model_init_config["submodules"]["profile_predictor"])
                pb_env = PowerBalanceTRB.model_init(train_dl, pb_config["model_init_config"])
                # The profile TRB derives two things from its dataloader that this
                # study's cannot supply: the normalizer stats over the 10 nn_inputs
                # (one is a measured betan the transport modules deliberately do
                # without, deriving beta from the evolving stored-energy state -
                # see Inputs.betan_from_wtot and REQUIRED_SIGNALS_TRANSPORT_TRANSFER)
                # and the PCA / k-means shape guess (needs Te_shape / ne_shape,
                # which the transport dataloader does not carry). Both are moot
                # here: this is only a skeleton, and restore_submodules below
                # overwrites every leaf with the profile prereq case's trained
                # weights, fitted on the profile study's own dataset - the feature
                # space this submodule actually consumes at runtime
                prof_init_config = {**prof_config["model_init_config"], "skip_data_init": True}
                profile_module = ProfilePredictorTRB.model_init(train_dl, prof_init_config)
                if model_init_config["restore_submodules"]:
                    pb_manager = create_default_checkpoint_manager(pb_config["checkpoint_dir"])
                    pb_env = restore_model(pb_manager, pb_env)
                    prof_manager = create_default_checkpoint_manager(prof_config["checkpoint_dir"])
                    profile_module = restore_model(prof_manager, profile_module)
                module = TransportPredictorSciML.init(
                    power_balance=pb_env.module,
                    profile_predictor=profile_module,
                )
            elif model_type == "transformer":
                module = TransportPredictorTransformer.init(
                    d_model=model_init_config["d_model"],
                    num_heads=model_init_config["num_heads"],
                    history_len=model_init_config["history_len"],
                    nn_width=model_init_config["nn_width"],
                    nn_depth=model_init_config["nn_depth"],
                    rhogrid=np.asarray(train_dl.ds["rho"]),
                    normalizer=normalizer,
                    prng_seed=model_init_config.get("prng_seed", 42),
                )
            elif model_type.startswith("torax-"):
                torax_cls = {
                    "rebuild": TransportPredictorTorax,
                    "carry": TransportPredictorToraxSimState,
                }[model_init_config["torax_state"]]
                module = torax_cls.init(
                    rhogrid=np.asarray(train_dl.ds["rho"]),
                    torax_config=model_init_config["torax_config"],
                    nn_width=model_init_config["nn_width"],
                    nn_depth=model_init_config["nn_depth"],
                    prng_seed=model_init_config.get("prng_seed", 42),
                    normalizer=normalizer,
                    sim_dt=model_init_config["sim_dt"],
                    transport_model=model_type.removeprefix("torax-"),
                    geometry_builder=model_init_config["geometry_builder"],
                    delta_exponent=model_init_config["delta_exponent"],
                )
            else:
                raise ValueError(f"Invalid model type: {model_type}")

            return module

        module = _build_module(train_dl, model_init_config)

        if model_init_config.get("freeze_submodules", False):
            freeze_submodules = ["power_balance", "profile_predictor"]
        else:
            freeze_submodules = []

        env = TransportPredictorEnv(
            module=module,
            domain_adaptation=model_init_config["domain_adaptation"],
            freeze_submodules=freeze_submodules,
        )

        if model_init_config.get("transfer_checkpoint", False):
            transfer_manager = create_default_checkpoint_manager(model_init_config["transfer_checkpoint"])
            env = restore_model(transfer_manager, env)
            # Restoring the whole env overwrote the freshly restored submodule weights, restore them again from their own checkpoints
            if model_init_config["model_type"] == "sciml":
                pb_config = _submodule_config_dict(model_init_config["submodules"]["power_balance"])
                prof_config = _submodule_config_dict(model_init_config["submodules"]["profile_predictor"])

                # The power balance case checkpointed a PowerBalanceEnv, so the
                # restore skeleton is rebuilt exactly the way that case built it
                pb_env = PowerBalanceTRB.model_init(train_dl, pb_config["model_init_config"])
                pb_manager = create_default_checkpoint_manager(pb_config["checkpoint_dir"])
                pb_env = restore_model(pb_manager, pb_env)
                prof_manager = create_default_checkpoint_manager(prof_config["checkpoint_dir"])
                prof_restored = restore_model(prof_manager, env.module.profile_predictor)
                env = eqx.tree_at(
                    lambda e: (e.module.power_balance, e.module.profile_predictor),
                    env,
                    (pb_env.module, prof_restored),
                )

            logger.debug(f"Restoring module from transfer learning pretrained checkpoint\n{model_init_config['transfer_checkpoint']}")

        # This restoration of the main module is separate from the transfer learning restoration
        # This would get the post-trained model, AFTER transfer learning has already been done
        if model_init_config.get("restore_main_module", False):
            manager = create_default_checkpoint_manager(model_init_config["checkpoint_dir"])
            env = restore_model(manager, env)
            logger.debug(f"Restoring module from post-training checkpoint\n{model_init_config['checkpoint_dir']}")
        else:
            logger.debug("Not restoring main module from post-training checkpoint.")

        return env

    @staticmethod
    def _make_profile_loss_fn(loss_config: dict, use_huber: bool, include_anchors: bool = False) -> IntegralLoss:
        """Device-weighted loss on the predicted ne/te profiles, wrapped for time integration.

        Profiles are peak-normalized per timeslice (scale from the target
        only, floored) so the two channels and all devices are commensurate
        and the huber delta reads as a fractional error, same convention as
        the profile study loss.

        Both losses count only timeslices with a fresh profile measurement:
        the profile residual is masked by the fresh_profiles target var, so
        forward-filled (stale) profile slices steer neither training nor
        checkpoint selection. The anchor terms are exempt, their measured
        signals exist at every timeslice.

        use_huber selects the training loss (huber on the raw residual, with
        the swept huber_delta) or the validation loss (delta-free absolute
        error, softened inside the GP-fit error bars). Same split as the
        profile study: the sweep metric val/loss.mean cannot be gamed by
        shrinking delta, and checkpoint selection does not chase fit noise
        the measurement cannot distinguish (within_error_weight down-weights
        the part of the residual inside the <v>_error bars, a 0 error is the
        sentinel for a zero-width bar and an absent error var behaves the
        same). The training loss never reads the error bars, its robustness
        to fit noise comes from the huber delta alone.

        include_anchors adds the ANCHOR_SIGNALS terms pulling the sciml
        submodule predictions (the power balance's Wtot plus its own p_oh and
        p_rad submodules) toward the measured signals, weighted by the
        anchor_weight_* loss_config keys. Training loss only: validation stays
        pure profile error so the sweep metric is comparable across model
        types. The terms drop out at trace time for model types whose
        target_vars do not carry the measured signals (transformer, torax-*).

        Anchor errors are plain absolute error, not huber.
        huber_delta is swept on the peak-normalized profile residuals and is
        meaningless for the MJ / MW scale anchor signals,
        and the anchors are not worth a second delta hyperparameter
        """
        if "device_weights" not in loss_config:
            device_weights = dict.fromkeys(config.dataset_paths, 1.0)
        else:
            device_weights = loss_config["device_weights"]

        anchor_weights = {}
        if include_anchors:
            for signal, (_, weight_key) in ANCHOR_SIGNALS.items():
                anchor_weights[signal] = loss_config.get(weight_key, 0.0)

        if use_huber:
            huber_delta = loss_config["huber_delta"]

            def value_err(residual):
                return optax.huber_loss(residual, delta=huber_delta)

            def _residual(pred, targ, sigma):
                # Training residual: plain distance to the GP fit mean, the
                # error bars do not soften it
                return jnp.abs(pred - targ)
        else:

            def value_err(residual):
                return residual

            # Down-weighting of the residual inside the measurement error bar,
            # read strictly (see ProfilePredictorTRB)
            within_error_weight = loss_config["within_error_weight"]

            def _residual(pred, targ, sigma):
                # Piecewise-linear shrink of the residual, same as the profile
                # study validation loss: full weight on the part beyond the
                # error bar, within_error_weight on the part inside it. sigma
                # is clamped at 0 so a degenerate negative error bar cannot
                # inflate the residual
                abs_residual = jnp.abs(pred - targ)
                sigma = jnp.maximum(sigma, 0.0)
                outside = jnp.maximum(abs_residual - sigma, 0.0)
                inside = jnp.minimum(abs_residual, sigma)
                return outside + within_error_weight * inside

        def _sigma_from_targ(targ, var, scale):
            # Error-bar target var, normalized like the profiles. An absent
            # var behaves like the 0 sentinel (zero-width error bar)
            if var in targ:
                return targ[var].data / scale
            return 0.0

        def loss_fn(pred, targ):
            ne_targ = targ["ne20_rho"].data
            te_targ = targ["Te_keV_rho"].data

            floor = ProfilePredictorTRB.PROFILE_SCALE_FLOOR
            ne_scale = jnp.maximum(jnp.max(jnp.abs(ne_targ), axis=-1, keepdims=True), floor)
            te_scale = jnp.maximum(jnp.max(jnp.abs(te_targ), axis=-1, keepdims=True), floor)

            ne_sigma = _sigma_from_targ(targ, "ne20_rho_error", ne_scale)
            te_sigma = _sigma_from_targ(targ, "Te_keV_rho_error", te_scale)

            # A diverged rollout is a failure of the model, not a missing
            # measurement, so it is charged instead of being allowed through as
            # NaN. Sanitize BEFORE the arithmetic: a single jnp.where after the
            # fact still drags NaN through the backward pass, because reverse
            # mode differentiates the discarded branch too. Replacing the bad
            # values with the target makes the residual exactly 0 there, so the
            # only thing those timeslices contribute is the explicit penalty
            # term below.
            ne_finite = jnp.isfinite(pred.ne)
            te_finite = jnp.isfinite(pred.te)
            ne_pred = jnp.where(ne_finite, pred.ne, ne_targ)
            te_pred = jnp.where(te_finite, pred.te, te_targ)
            # Per-timeslice divergence flag (any bad point on either channel)
            diverged = ~(jnp.all(ne_finite, axis=-1) & jnp.all(te_finite, axis=-1))

            ne_err = value_err(_residual(ne_pred / ne_scale, ne_targ / ne_scale, ne_sigma))
            te_err = value_err(_residual(te_pred / te_scale, te_targ / te_scale, te_sigma))

            # Build per-sample weights from device labels
            ds_source_idx = targ["ds_source_idx"].data
            sample_weights = jnp.ones(ds_source_idx.shape, dtype=ne_err.dtype)
            for device, weight in device_weights.items():
                sample_weights = jnp.where(
                    ds_source_idx == config.ds_source_to_idx[device],
                    weight,
                    sample_weights,
                )

            # Freshness mask: only timeslices with a fresh profile measurement
            # contribute to the profile terms, forward-filled slices are zeroed
            profile_weights = sample_weights * targ["fresh_profiles"].data
            # Broadcast across the rho axis
            while profile_weights.ndim < ne_err.ndim:
                profile_weights = profile_weights[..., None]

            loss = 0.5 * (jnp.mean(profile_weights * ne_err) + jnp.mean(profile_weights * te_err))

            # Anchor terms are scalar signals measured at every timeslice, so
            # they take the unbroadcast per-sample weights without the
            # freshness mask (freshness only applies to the profiles)
            for signal, anchor_weight in anchor_weights.items():
                if anchor_weight <= 0.0 or signal not in targ:
                    continue
                pred_attr = ANCHOR_SIGNALS[signal][0]
                anchor_pred = getattr(pred, pred_attr)
                anchor_finite = jnp.isfinite(anchor_pred)
                anchor_targ = targ[signal].data
                # Same sanitize-then-charge treatment as the profiles
                anchor_errors = jnp.abs(jnp.where(anchor_finite, anchor_pred, anchor_targ) - anchor_targ)
                loss = loss + anchor_weight * jnp.mean(sample_weights * anchor_errors)
                diverged = diverged | ~anchor_finite

            # Divergence penalty, deliberately OUTSIDE the freshness mask:
            # a rollout that went non-finite is broken whether or not those
            # timeslices happened to carry a fresh profile measurement, and
            # masking it would let a fully diverged run score as though the
            # stale slices simply did not count.
            #
            # Applied to BOTH the training and validation loss (one builder),
            # which is the point: previously a diverged trial paid nothing.
            # popsim's own guard only counts SKIPPED steps, so steps recovered
            # by NaN-masking left train/nan_skip_fraction at 0, and the sweep
            # metric val/loss.mean never saw the divergence at all - so the
            # bayes sweep was free to walk into the divergent high-lr corner
            # TODO(ZanderKeith): This deserves a revisit
            divergence_penalty = loss_config["divergence_penalty"]
            loss = loss + divergence_penalty * jnp.mean(sample_weights * diverged.astype(loss.dtype))

            return loss

        return IntegralLoss(loss_fn)

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        return TransportPredictorTRB._make_profile_loss_fn(loss_config, use_huber=True, include_anchors=True)

    @staticmethod
    def get_val_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        return TransportPredictorTRB._make_profile_loss_fn(loss_config, use_huber=False)

    @staticmethod
    def get_val_eval_suite(suite_config) -> EvaluationSuite | None:
        """Validation suite computing the delta-free loss (sweep metric val/loss.mean)."""
        if suite_config is None:
            return None
        return make_loss_eval_suite(TransportPredictorTRB.get_val_loss_fn(suite_config["loss_config"]))

    @staticmethod
    def get_optimizer(optimizer_config: dict) -> optax.GradientTransformation:
        # zero_nans FIRST, because the global-norm cap cannot help against a
        # non-finite gradient: clip_by_global_norm(NaN) is still NaN, so one
        # bad backward pass poisons every parameter and the run is dead even
        # though the forward losses were all finite (the transformer rollout
        # does this - the loss sanitizes non-finite PREDICTIONS, but a finite
        # loss can still have a NaN derivative). Zeroing turns that step into
        # a no-op for the affected leaves instead of ending the run, and the
        # divergence_penalty term still charges the sample in the loss so the
        # sweep metric keeps seeing it.
        #
        # Then the global-norm cap as in the profile study (the differentiated
        # TORAX solve can spike gradients and NaN a run without it), on top of
        # the power balance study's grouped schedule: submodule_lr_factors runs
        # the sciml power_balance subtree at a reduced learning rate while
        # the profile predictor keeps the full one (pytree-path labeling, a
        # no-op for model types without a matching path)
        return optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(optimizer_config.get("grad_clip_max_norm", 1.0)),
            make_grouped_exponential_adamw(optimizer_config),
        )

    @staticmethod
    def get_test_eval_suite(suite_config) -> EvaluationSuite:
        """Evaluation suite for testing after training.

        The profile study's study_results already handles time-dependent
        outputs (it renames the stepper's time_idx_input dim back) and emits
        exactly the per-channel and combined error variables the analysis
        stack reads, so it is reused as-is.
        """
        return ProfilePredictorTRB.get_test_eval_suite(suite_config)
