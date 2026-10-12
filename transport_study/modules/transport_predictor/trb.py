from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from popsim.ml import DataLoader, IntegralLoss, TrainRunBuilder
from popsim.ml.eval import EvaluationSuite

from transport_study import RADIAL_DIM
from transport_study.config import config
from transport_study.modules.power_balance.module import (
    FREEZABLE_SUBMODULES,
    SCALING_LAW_FIELDS,
)
from transport_study.modules.power_balance.trb import PowerBalanceTRB
from transport_study.modules.profile_predictor.trb import ProfilePredictorTRB
from transport_study.modules.transport_predictor.module import (
    TransportPredictorEnv,
    TransportPredictorSciML,
    TransportPredictorTorax,
    TransportPredictorToraxCarry,
    TransportPredictorTransformer,
    make_transport_nn_input_normalizer,
)
from transport_study.modules.trb_utils import (
    CHI_ERROR_VARS,
    TimeAveragedLoss,
    chi_profile_loss,
    freshness_weight,
    get_time_dep_dataloaders,
    huber_profile_loss,
    make_grouped_exponential_adamw,
    make_loss_eval_suite,
    module_build_config,
    normalizer_fit_dataset,
    per_sample_device_values,
    per_sample_residual_scale,
    per_sample_sigma_floor,
    restore_from_checkpoint,
    submodule_config_dict,
    target_device_idx,
    transfer_checkpoint_to_restore,
)


def _sciml_submodules(train_dl: DataLoader, model_init_config: dict) -> tuple:
    """The power balance module and the profile predictor of a sciml case, restored from their prereq case checkpoints.

    The skeletons come from the submodules' own TRBs under skip_data_init,
    and the power balance one is a whole PowerBalanceEnv, as its case checkpointed.
    The profile skeleton could not take its data-driven init anyway:
    the transport dataloader carries no measured beta_tor_norm and no shape variables,
    and the restore overwrites every leaf with the weights fitted on the profile study's own data.
    A skip_data_init model_init_config gets the unrestored skeletons, its caller restores them (see trb_utils.module_build_config).
    """
    pb_config = submodule_config_dict(model_init_config["submodules"]["power_balance"])
    profile_config = submodule_config_dict(model_init_config["submodules"]["profile_predictor"])
    pb_env = PowerBalanceTRB.model_init(train_dl, {**pb_config["model_init_config"], "skip_data_init": True})
    profile_predictor = ProfilePredictorTRB.model_init(train_dl, {**profile_config["model_init_config"], "skip_data_init": True})
    if not model_init_config.get("skip_data_init", False):
        pb_env = restore_from_checkpoint(pb_env, pb_config["checkpoint_dir"])
        profile_predictor = restore_from_checkpoint(profile_predictor, profile_config["checkpoint_dir"])
    return pb_env.module, profile_predictor


class TransportPredictorTRB(TrainRunBuilder):
    """TrainRunBuilder for the transport predictor modules used in transfer learning."""

    # The organize_data prep its dataloaders load every device with
    STUDY_TYPE: ClassVar[str] = "transport_transfer"

    # Anchor terms in the training loss (sciml and multiobjective cases), keyed by measured target signal:
    # (Output attribute holding the model's own prediction, loss_config key for the weight)
    ANCHOR_SIGNALS: ClassVar[dict[str, tuple[str, str]]] = {
        "energy_mhd_MJ": ("energy_mhd_MJ_pred", "anchor_weight_energy_mhd"),
        "power_ohm_MW": ("power_ohm_MW_pred", "anchor_weight_power_ohm"),
        "power_radiated_MW": ("power_radiated_MW_pred", "anchor_weight_power_radiated"),
    }

    @staticmethod
    def get_dataloaders(
        dataloader_config: dict,
    ) -> tuple[xr.Dataset, DataLoader, DataLoader, DataLoader]:
        """Dataset and dataloaders for training, see trb_utils.get_time_dep_dataloaders."""
        return get_time_dep_dataloaders(dataloader_config, TransportPredictorTRB.STUDY_TYPE)

    @staticmethod
    def _build_module(train_dl: DataLoader, build_config: dict) -> Any:
        """The transport predictor of build_config["model_type"] (see trb_utils.module_build_config)."""
        model_type = build_config["model_type"]
        if model_type == "sciml":
            power_balance, profile_predictor = _sciml_submodules(train_dl, build_config)
            return TransportPredictorSciML.init(power_balance=power_balance, profile_predictor=profile_predictor)

        # Stat stage (CORAL or z-score) on the 11 transport features, sciml has none of its own
        normalizer = make_transport_nn_input_normalizer(
            build_config["data_normalization"],
            normalizer_fit_dataset(train_dl, build_config),
            len(config.ds_source_to_idx),
            target_device_idx(),
        )
        rhogrid = np.asarray(train_dl.ds[RADIAL_DIM])
        if model_type == "transformer":
            return TransportPredictorTransformer.init(
                d_model=build_config["d_model"],
                num_heads=build_config["num_heads"],
                history_len=build_config["history_len"],
                nn_width=build_config["nn_width"],
                nn_depth=build_config["nn_depth"],
                rhogrid=rhogrid,
                normalizer=normalizer,
                multiobjective=build_config["multiobjective"],
                prng_seed=build_config["prng_seed"],
            )
        if model_type.startswith("torax-"):
            torax_cls = {"rebuild": TransportPredictorTorax, "carry": TransportPredictorToraxCarry}[build_config["torax_state"]]
            return torax_cls.init(
                rhogrid=rhogrid,
                torax_config=build_config["torax_config"],
                nn_width=build_config["nn_width"],
                nn_depth=build_config["nn_depth"],
                prng_seed=build_config["prng_seed"],
                normalizer=normalizer,
                sim_dt=build_config["sim_dt"],
                transport_model=model_type.removeprefix("torax-"),
                geometry_builder=build_config["geometry_builder"],
                delta_exponent=build_config["delta_exponent"],
            )
        raise ValueError(f"Invalid model type: {model_type}")

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> TransportPredictorEnv:
        """The TransportPredictorEnv of a case, the sciml submodules restored from their prereq cases.

        A transfer case builds a skeleton (see trb_utils.module_build_config),
        restores the whole env from its transfer_pretrain checkpoint,
        then a sciml case puts back the submodules of its own power_balance / profile prereq cases,
        which ran their own pretrain and finetune on the target device.
        Under skip_data_init it is an unrestored skeleton.
        """
        # A frozen sciml case keeps its power balance's p_oh / p_rad predictors fixed, as the power balance study's freeze does
        freeze_submodules = list(FREEZABLE_SUBMODULES) if model_init_config.get("freeze_submodules", False) else []
        build_config = module_build_config(model_init_config)
        env = TransportPredictorEnv(
            module=TransportPredictorTRB._build_module(train_dl, build_config),
            domain_adaptation=model_init_config["domain_adaptation"],
            freeze_submodules=freeze_submodules,
        )
        transfer_checkpoint = transfer_checkpoint_to_restore(model_init_config)
        if transfer_checkpoint:
            env = restore_from_checkpoint(env, transfer_checkpoint)
            if model_init_config["model_type"] == "sciml":
                env = eqx.tree_at(
                    lambda e: (e.module.power_balance, e.module.profile_predictor),
                    env,
                    _sciml_submodules(train_dl, model_init_config),
                )
        return env

    @staticmethod
    def _make_profile_loss_fn(loss_config: dict, training: bool) -> IntegralLoss:
        """Device-weighted loss on the predicted ne/te profiles, wrapped for time integration.

        Training: the huber of chi, the residuals of the values and of the GP-fit gradients in error bars
        floored per device at train_chi_sigma_floors from the case's own training data,
        with the swept huber_delta / huber_delta_grad in error bars (trb_utils.huber_profile_loss, the profile study's training loss),
        plus the anchor terms.
        Validation: value plus gradient chi, the residuals in units of the GP-fit error bars
        floored per device at chi_sigma_floors (trb_utils.chi_profile_loss), so no swept delta can shrink the sweep metric.

        Both count only timeslices with a fresh profile measurement:
        the profile terms are multiplied by the fresh_profile target var,
        so forward-filled (stale) slices steer neither training nor checkpoint selection.
        Training integrates over the equal-length segments,
        validation averages over each episode's fresh profile time (TimeAveragedLoss), so a long shot weighs no more than a short one.

        The anchor terms pull the sciml submodule predictions (the power balance Wtot and its own p_oh / p_rad)
        and a multiobjective case's stored energy and ohmic power toward the measured signals,
        weighted by the anchor_weight_* loss_config keys.
        They are training only, the stored energy and ohmic power counted at the fresh equilibrium timeslices
        and P_rad at every one (trb_utils.freshness_weight).
        Each is a huber on the residual over the device's typical value (loss_config residual_scales),
        with the delta tuned on the study's hyperparam case of the model predicting it (loss_config anchor_huber_deltas),
        and drops out at trace time for cases whose target_vars lack the signal (the plain transformer and torax-*).
        """
        device_weights = loss_config.get("device_weights", {})
        gradient_weight = loss_config["gradient_weight"]
        if training:
            sigma_floors = loss_config["train_chi_sigma_floors"]
            divergence_penalty = loss_config["divergence_penalty"]
            residual_scales = loss_config["residual_scales"]
            anchor_signals = TransportPredictorTRB.ANCHOR_SIGNALS
        else:
            sigma_floors = loss_config["chi_sigma_floors"]
            divergence_penalty = loss_config["divergence_penalty_val"]
            residual_scales = {}
            anchor_signals = {}
        anchor_weights = {signal: loss_config[weight_key] for signal, (_, weight_key) in anchor_signals.items()}

        def loss_fn(pred, targ):
            ne_targ = targ["n_e_1e20"].data
            te_targ = targ["t_e_keV"].data
            ds_source_idx = targ["ds_source_idx"].data
            sample_weights = per_sample_device_values(ds_source_idx, device_weights, 1.0)

            # A diverged rollout is a failure of the model, not a missing measurement,
            # so it is charged by the divergence penalty below instead of passing through as NaN.
            # The non-finite values are replaced by the target BEFORE the arithmetic,
            # because reverse mode also differentiates the discarded branch of a later jnp.where
            ne_finite = jnp.isfinite(pred.ne)
            te_finite = jnp.isfinite(pred.te)
            ne_pred = jnp.where(ne_finite, pred.ne, ne_targ)
            te_pred = jnp.where(te_finite, pred.te, te_targ)
            diverged = ~(jnp.all(ne_finite, axis=-1) & jnp.all(te_finite, axis=-1))

            # Only timeslices with a fresh profile measurement count for the profile terms, values and gradients alike
            profile_weights = sample_weights * targ["fresh_profile"].data
            channel_losses = []
            for channel, pred_channel, targ_channel in (("n_e_1e20", ne_pred, ne_targ), ("t_e_keV", te_pred, te_targ)):
                value_error_var, grad_error_var = CHI_ERROR_VARS[channel]
                profile_terms = (
                    pred_channel,
                    targ_channel,
                    targ[value_error_var].data,
                    targ[f"{channel}_gradient"].data,
                    targ[grad_error_var].data,
                    per_sample_sigma_floor(ds_source_idx, sigma_floors, value_error_var),
                    per_sample_sigma_floor(ds_source_idx, sigma_floors, grad_error_var),
                    pred.rho,
                )
                if training:
                    channel_loss = huber_profile_loss(
                        *profile_terms, loss_config["huber_delta"], loss_config["huber_delta_grad"], gradient_weight
                    )
                else:
                    channel_loss = chi_profile_loss(*profile_terms, gradient_weight)
                channel_losses.append(channel_loss)
            # Training averages the two channels, chi sums them, the scales the divergence penalties are sized to
            channel_sum = channel_losses[0] + channel_losses[1]
            loss = jnp.mean(profile_weights * (0.5 * channel_sum if training else channel_sum))

            for signal, anchor_weight in anchor_weights.items():
                if anchor_weight <= 0.0 or signal not in targ:
                    continue
                anchor_pred = getattr(pred, anchor_signals[signal][0])
                anchor_finite = jnp.isfinite(anchor_pred)
                anchor_targ = targ[signal].data
                # Same sanitize-then-charge treatment as the profiles
                anchor_residual = jnp.where(anchor_finite, anchor_pred, anchor_targ) - anchor_targ
                anchor_scale = per_sample_residual_scale(ds_source_idx, residual_scales, signal)
                anchor_errors = optax.huber_loss(anchor_residual / anchor_scale, delta=loss_config["anchor_huber_deltas"][signal])
                anchor_sample_weights = sample_weights * freshness_weight(signal, targ)
                loss = loss + anchor_weight * jnp.mean(anchor_sample_weights * anchor_errors)
                diverged = diverged | ~anchor_finite

            # Training charges the divergence penalty outside the freshness mask:
            # a rollout that went non-finite is broken whether or not those slices carry a fresh measurement.
            # Validation charges it at the fresh slices, so the whole validation loss is one fresh-time average,
            # a fully diverged episode scoring the penalty itself, and a diverged trial cannot win the sweep
            diverged_weights = sample_weights if training else profile_weights
            return loss + divergence_penalty * jnp.mean(diverged_weights * diverged.astype(loss.dtype))

        # Validation scores whole episodes, where an integral would weigh each shot by its length,
        # so it divides by the fresh profile time and every shot counts once.
        # Training stays an integral, its segments are equal length so dividing by their duration would only rescale the loss.
        if training:
            return IntegralLoss(loss_fn)
        return TimeAveragedLoss(loss_fn, "fresh_profile")

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        return TransportPredictorTRB._make_profile_loss_fn(loss_config, training=True)

    @staticmethod
    def get_val_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        return TransportPredictorTRB._make_profile_loss_fn(loss_config, training=False)

    @staticmethod
    def get_val_eval_suite(suite_config) -> EvaluationSuite | None:
        """Validation suite computing the chi loss (sweep metric val/loss.mean)."""
        if suite_config is None:
            return None
        return make_loss_eval_suite(TransportPredictorTRB.get_val_loss_fn(suite_config["loss_config"]))

    @staticmethod
    def get_optimizer(optimizer_config: dict) -> optax.GradientTransformation:
        # The power balance study's grouped schedule, after popsim's clip_by_global_norm(0.5):
        # submodule_lr_factors runs the sciml power_balance subtree at a reduced learning rate,
        # and its scaling-law coefficients skip the weight decay,
        # both no-ops for model types without a matching pytree path.
        # A non-finite gradient reaches popsim's NaN skip count and TrainingDivergedError abort unmasked
        return make_grouped_exponential_adamw(optimizer_config, no_decay_names=SCALING_LAW_FIELDS)

    @staticmethod
    def get_test_eval_suite(suite_config) -> EvaluationSuite | None:
        """The profile study's test suite, scored on fresh profile timeslices only like the losses.

        It already handles the time-dependent outputs (renaming the stepper's _input dims back)
        and emits the per-channel and combined error variables the analysis stack reads.
        """
        return ProfilePredictorTRB.make_test_eval_suite(suite_config, fresh_only=True)
