from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from popsim.ml import DataLoader, IntegralLoss, TrainRunBuilder
from popsim.ml.eval import EvaluationSuite

from transport_study import RADIAL_DIM
from transport_study.config import config
from transport_study.modules.power_balance.module import SCALING_LAW_FIELDS
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
    chi_value,
    get_time_dep_dataloaders,
    make_grouped_exponential_adamw,
    make_loss_eval_suite,
    normalizer_fit_dataset,
    peak_scale,
    per_sample_device_values,
    per_sample_sigma_floor,
    restore_from_checkpoint,
    submodule_config_dict,
    target_device_idx,
)

STUDY_TYPE = "transport_transfer"

# Anchor terms in the sciml training loss, keyed by measured target signal:
# (Output attribute holding the model's own prediction, loss_config key for the weight)
ANCHOR_SIGNALS = {
    "energy_mhd_MJ": ("energy_mhd_MJ_pred", "anchor_weight_energy_mhd"),
    "power_ohm_MW": ("power_ohm_MW_pred", "anchor_weight_power_ohm"),
    "power_radiated_MW": ("power_radiated_MW_pred", "anchor_weight_power_radiated"),
}


def _restored_sciml_submodules(train_dl: DataLoader, model_init_config: dict) -> tuple:
    """The power balance module and the profile predictor of a sciml case, restored from their prereq case checkpoints.

    The skeletons come from the submodules' own TRBs, the power balance case checkpointed a whole PowerBalanceEnv.
    The profile skeleton skips its data-driven init (skip_data_init):
    the transport dataloader carries no measured beta_tor_norm and no shape variables,
    and the restore overwrites every leaf with the weights fitted on the profile study's own data.
    """
    pb_config = submodule_config_dict(model_init_config["submodules"]["power_balance"])
    profile_config = submodule_config_dict(model_init_config["submodules"]["profile_predictor"])
    pb_env = PowerBalanceTRB.model_init(train_dl, pb_config["model_init_config"])
    profile_skeleton = ProfilePredictorTRB.model_init(train_dl, {**profile_config["model_init_config"], "skip_data_init": True})
    return (
        restore_from_checkpoint(pb_env, pb_config["checkpoint_dir"]).module,
        restore_from_checkpoint(profile_skeleton, profile_config["checkpoint_dir"]),
    )


class TransportPredictorTRB(TrainRunBuilder):
    """TrainRunBuilder for the transport predictor modules used in transfer learning."""

    @staticmethod
    def get_dataloaders(
        dataloader_config: dict,
    ) -> tuple[xr.Dataset, DataLoader, DataLoader, DataLoader]:
        """Dataset and dataloaders for training, see trb_utils.get_time_dep_dataloaders."""
        return get_time_dep_dataloaders(dataloader_config, STUDY_TYPE)

    @staticmethod
    def _build_module(train_dl: DataLoader, model_init_config: dict) -> Any:
        """The transport predictor of model_init_config["model_type"]."""
        model_type = model_init_config["model_type"]
        if model_type == "sciml":
            power_balance, profile_predictor = _restored_sciml_submodules(train_dl, model_init_config)
            return TransportPredictorSciML.init(power_balance=power_balance, profile_predictor=profile_predictor)

        # Stat stage (CORAL or z-score) on the 11 transport features, sciml has none of its own
        normalizer = make_transport_nn_input_normalizer(
            model_init_config["data_normalization"],
            normalizer_fit_dataset(train_dl, model_init_config),
            len(config.ds_source_to_idx),
            target_device_idx(),
        )
        rhogrid = np.asarray(train_dl.ds[RADIAL_DIM])
        if model_type == "transformer":
            return TransportPredictorTransformer.init(
                d_model=model_init_config["d_model"],
                num_heads=model_init_config["num_heads"],
                history_len=model_init_config["history_len"],
                nn_width=model_init_config["nn_width"],
                nn_depth=model_init_config["nn_depth"],
                rhogrid=rhogrid,
                normalizer=normalizer,
                prng_seed=model_init_config["prng_seed"],
            )
        if model_type.startswith("torax-"):
            torax_cls = {"rebuild": TransportPredictorTorax, "carry": TransportPredictorToraxSimState}[model_init_config["torax_state"]]
            return torax_cls.init(
                rhogrid=rhogrid,
                torax_config=model_init_config["torax_config"],
                nn_width=model_init_config["nn_width"],
                nn_depth=model_init_config["nn_depth"],
                prng_seed=model_init_config["prng_seed"],
                normalizer=normalizer,
                sim_dt=model_init_config["sim_dt"],
                transport_model=model_type.removeprefix("torax-"),
                geometry_builder=model_init_config["geometry_builder"],
                delta_exponent=model_init_config["delta_exponent"],
            )
        raise ValueError(f"Invalid model type: {model_type}")

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> TransportPredictorEnv:
        """The TransportPredictorEnv of a case, the sciml submodules restored from their prereq cases.

        A transfer case restores the whole env from its transfer_pretrain checkpoint,
        then a sciml case puts back the submodules of its own power_balance / profile prereq cases,
        which ran their own pretrain and finetune on the target device.
        """
        freeze_submodules = ["power_balance", "profile_predictor"] if model_init_config.get("freeze_submodules", False) else []
        env = TransportPredictorEnv(
            module=TransportPredictorTRB._build_module(train_dl, model_init_config),
            domain_adaptation=model_init_config["domain_adaptation"],
            freeze_submodules=freeze_submodules,
        )
        if model_init_config.get("transfer_checkpoint"):
            env = restore_from_checkpoint(env, model_init_config["transfer_checkpoint"])
            if model_init_config["model_type"] == "sciml":
                env = eqx.tree_at(
                    lambda e: (e.module.power_balance, e.module.profile_predictor),
                    env,
                    _restored_sciml_submodules(train_dl, model_init_config),
                )
        return env

    @staticmethod
    def _make_profile_loss_fn(loss_config: dict, use_chi: bool) -> IntegralLoss:
        """Device-weighted loss on the predicted ne/te profiles, wrapped for time integration.

        Training (use_chi False): huber on the peak-normalized residual with the swept huber_delta,
        the same convention as the profile study training loss, plus the sciml anchor terms.
        Validation (use_chi True): value chi, the residual in units of the GP-fit error bar
        floored per device at chi_sigma_floors (trb_utils.chi_value), so no swept delta can shrink the sweep metric.

        Both count only timeslices with a fresh profile measurement:
        the profile terms are multiplied by the fresh_profile target var,
        so forward-filled (stale) slices steer neither training nor checkpoint selection.

        The anchor terms pull the sciml submodule predictions (the power balance Wtot and its own p_oh / p_rad)
        toward the measured signals, weighted by the anchor_weight_* loss_config keys.
        They are training only, measured at every timeslice so exempt from the freshness mask,
        plain absolute error (huber_delta is sized for the profile residuals, not MJ / MW signals),
        and drop out at trace time for model types whose target_vars lack the signals (transformer, torax-*).
        """
        device_weights = loss_config.get("device_weights", {})
        if use_chi:
            sigma_floors = loss_config["chi_sigma_floors"]
            divergence_penalty = loss_config["divergence_penalty_val"]
            anchor_weights = {}
        else:
            huber_delta = loss_config["huber_delta"]
            divergence_penalty = loss_config["divergence_penalty"]
            anchor_weights = {signal: loss_config[weight_key] for signal, (_, weight_key) in ANCHOR_SIGNALS.items()}

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

            # Only timeslices with a fresh profile measurement count for the profile terms
            profile_weights = sample_weights * targ["fresh_profile"].data
            if use_chi:
                chi_ne = chi_value(
                    ne_pred,
                    ne_targ,
                    targ["n_e_1e20_error"].data,
                    per_sample_sigma_floor(ds_source_idx, sigma_floors, "n_e_1e20_error"),
                    pred.rho,
                )
                chi_te = chi_value(
                    te_pred,
                    te_targ,
                    targ["t_e_keV_error"].data,
                    per_sample_sigma_floor(ds_source_idx, sigma_floors, "t_e_keV_error"),
                    pred.rho,
                )
                loss = jnp.mean(profile_weights * (chi_ne + chi_te))
            else:
                ne_scale = peak_scale(ne_targ)
                te_scale = peak_scale(te_targ)
                ne_err = optax.huber_loss(ne_pred / ne_scale - ne_targ / ne_scale, delta=huber_delta)
                te_err = optax.huber_loss(te_pred / te_scale - te_targ / te_scale, delta=huber_delta)
                loss = 0.5 * (jnp.mean(profile_weights[..., None] * ne_err) + jnp.mean(profile_weights[..., None] * te_err))

            for signal, anchor_weight in anchor_weights.items():
                if anchor_weight <= 0.0 or signal not in targ:
                    continue
                anchor_pred = getattr(pred, ANCHOR_SIGNALS[signal][0])
                anchor_finite = jnp.isfinite(anchor_pred)
                anchor_targ = targ[signal].data
                # Same sanitize-then-charge treatment as the profiles
                anchor_errors = jnp.abs(jnp.where(anchor_finite, anchor_pred, anchor_targ) - anchor_targ)
                loss = loss + anchor_weight * jnp.mean(sample_weights * anchor_errors)
                diverged = diverged | ~anchor_finite

            # The divergence penalty is deliberately outside the freshness mask:
            # a rollout that went non-finite is broken whether or not those slices carry a fresh measurement.
            # It applies to validation too, so a diverged trial cannot win the sweep
            return loss + divergence_penalty * jnp.mean(sample_weights * diverged.astype(loss.dtype))

        return IntegralLoss(loss_fn)

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        return TransportPredictorTRB._make_profile_loss_fn(loss_config, use_chi=False)

    @staticmethod
    def get_val_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        return TransportPredictorTRB._make_profile_loss_fn(loss_config, use_chi=True)

    @staticmethod
    def get_val_eval_suite(suite_config) -> EvaluationSuite | None:
        """Validation suite computing the chi loss (sweep metric val/loss.mean)."""
        if suite_config is None:
            return None
        return make_loss_eval_suite(TransportPredictorTRB.get_val_loss_fn(suite_config["loss_config"]))

    @staticmethod
    def get_optimizer(optimizer_config: dict) -> optax.GradientTransformation:
        # zero_nans first, the global-norm cap cannot fix a non-finite gradient (clip_by_global_norm(NaN) is NaN).
        # A finite loss can still have a NaN derivative (the transformer rollout does this),
        # so zeroing makes that step a no-op for the affected leaves instead of ending the run.
        # The divergence_penalty term still charges the sample, so the sweep metric sees it.
        # Then the global-norm cap as in the profile study, the differentiated TORAX solve can spike gradients.
        # Last the power balance study's grouped schedule:
        # submodule_lr_factors runs the sciml power_balance subtree at a reduced learning rate,
        # and its scaling-law coefficients skip the weight decay,
        # both no-ops for model types without a matching pytree path.
        return optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(optimizer_config["grad_clip_max_norm"]),
            make_grouped_exponential_adamw(optimizer_config, no_decay_names=SCALING_LAW_FIELDS),
        )

    @staticmethod
    def get_test_eval_suite(suite_config) -> EvaluationSuite:
        """The profile study's test suite, scored on fresh profile timeslices only like the losses.

        It already handles the time-dependent outputs (renaming the stepper's _input dims back)
        and emits the per-channel and combined error variables the analysis stack reads.
        """
        return ProfilePredictorTRB.make_test_eval_suite(suite_config, fresh_only=True)
