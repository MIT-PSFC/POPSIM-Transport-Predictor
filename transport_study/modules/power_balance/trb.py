from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
import optax
import xarray as xr
from popsim.ml import DataLoader, IntegralLoss, TrainRunBuilder
from popsim.ml.eval import EvaluationSuite

from transport_study.config import config
from transport_study.modules.normalization import make_normalizer
from transport_study.modules.power_balance.module import (
    FREEZABLE_SUBMODULES,
    MODEL_TYPES_WITH_ENERGY_INPUT,
    MODEL_TYPES_WITH_SUBMODULES,
    SCALING_LAW_FIELDS,
    PowerBalanceEnv,
    PowerBalanceMLP,
    PowerBalanceScalingLaw,
    PowerBalanceSciML,
    PowerBalanceTransformer,
)
from transport_study.modules.power_balance.p_oh.trb import OhmicPowerTRB
from transport_study.modules.power_balance.p_rad.trb import RadiatedPowerTRB
from transport_study.modules.trb_utils import (
    TimeAveragedLoss,
    freshness_weight,
    get_time_dep_dataloaders,
    make_grouped_exponential_adamw,
    make_loss_eval_suite,
    module_build_config,
    normalizer_fit_dataset,
    per_sample_device_values,
    restore_from_checkpoint,
    scalar_study_results,
    submodule_config_dict,
    target_device_idx,
    transfer_checkpoint_to_restore,
)

# Anchor terms in the training loss of the models with p_oh / p_rad submodules, keyed by measured target signal:
# (Output attribute holding the submodule's prediction, loss_config key of its weight).
# They keep the submodule predictions close to the measured signals.
# Not cheating, a real application has these signals for training alongside the target Wtot
ANCHOR_SIGNALS = {
    "power_ohm_MW": ("power_ohm_MW_pred", "anchor_weight_power_ohm"),
    "power_radiated_MW": ("power_radiated_MW_pred", "anchor_weight_power_radiated"),
}


def _prereq_submodules(train_dl: DataLoader, model_init_config: dict) -> tuple:
    """The p_oh and p_rad predictors, built by their own TRBs as skeletons and restored from their prereq case checkpoints.

    A skip_data_init model_init_config gets the unrestored skeletons, its caller restores them (see trb_utils.module_build_config).
    """
    submodules = []
    for name, submodule_trb in (("p_oh_predictor", OhmicPowerTRB), ("p_rad_predictor", RadiatedPowerTRB)):
        submodule_config = submodule_config_dict(model_init_config["submodules"][name])
        submodule = submodule_trb.model_init(train_dl, {**submodule_config["model_init_config"], "skip_data_init": True})
        if not model_init_config.get("skip_data_init", False):
            submodule = restore_from_checkpoint(submodule, submodule_config["checkpoint_dir"])
        submodules.append(submodule)
    return tuple(submodules)


class PowerBalanceTRB(TrainRunBuilder):
    """TrainRunBuilder for the power balance modules used in transfer learning"""

    # The organize_data prep its dataloaders load every device with
    STUDY_TYPE: ClassVar[str] = "power_balance_transfer"

    @staticmethod
    def get_dataloaders(
        dataloader_config: dict,
    ) -> tuple[xr.Dataset, DataLoader, DataLoader, DataLoader]:
        """Dataset and dataloaders for training, see trb_utils.get_time_dep_dataloaders."""
        return get_time_dep_dataloaders(dataloader_config, PowerBalanceTRB.STUDY_TYPE)

    @staticmethod
    def _build_module(train_dl: DataLoader, build_config: dict) -> Any:
        """The power balance module of build_config["model_type"], normalizer fitted on the training data (see module_build_config)."""
        model_type = build_config["model_type"]
        if model_type == "sciml-taue-scalinglaw":
            # The scaling law consumes physical units and holds no normalizer,
            # data_normalization reaches only its p_oh / p_rad prereq cases
            p_oh_predictor, p_rad_predictor = _prereq_submodules(train_dl, build_config)
            return PowerBalanceScalingLaw.init(p_oh_predictor=p_oh_predictor, p_rad_predictor=p_rad_predictor)
        normalizer = make_normalizer(
            build_config["data_normalization"],
            normalizer_fit_dataset(train_dl, build_config),
            len(config.ds_source_to_idx),
            target_device_idx(),
            with_energy=model_type in MODEL_TYPES_WITH_ENERGY_INPUT,
        )
        if model_type == "sciml-taue-nn":
            p_oh_predictor, p_rad_predictor = _prereq_submodules(train_dl, build_config)
            return PowerBalanceSciML.init(
                p_oh_predictor=p_oh_predictor,
                p_rad_predictor=p_rad_predictor,
                normalizer=normalizer,
                nn_width=build_config["nn_width"],
                nn_depth=build_config["nn_depth"],
                prng_seed=build_config["prng_seed"],
            )
        if model_type == "mlp":
            return PowerBalanceMLP.init(
                nn_width=build_config["nn_width"],
                nn_depth=build_config["nn_depth"],
                normalizer=normalizer,
                prng_seed=build_config["prng_seed"],
            )
        if model_type == "transformer":
            return PowerBalanceTransformer.init(
                d_model=build_config["d_model"],
                num_heads=build_config["num_heads"],
                history_len=build_config["history_len"],
                nn_width=build_config["nn_width"],
                nn_depth=build_config["nn_depth"],
                normalizer=normalizer,
                predicts_powers=build_config["multiobjective"],
                prng_seed=build_config["prng_seed"],
            )
        raise ValueError(f"Invalid model type: {model_type}")

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> PowerBalanceEnv:
        """The PowerBalanceEnv of a case, its submodules restored from their prereq cases.

        A transfer case builds a skeleton (see trb_utils.module_build_config),
        restores the whole env from its transfer_pretrain checkpoint,
        then puts back the submodules of its own p_oh / p_rad prereq cases,
        which ran their own pretrain and finetune on the target device.
        Under skip_data_init it is an unrestored skeleton.
        """
        freeze_submodules = list(FREEZABLE_SUBMODULES) if model_init_config.get("freeze_submodules", False) else []
        build_config = module_build_config(model_init_config)
        env = PowerBalanceEnv(
            module=PowerBalanceTRB._build_module(train_dl, build_config),
            domain_adaptation=model_init_config["domain_adaptation"],
            freeze_submodules=freeze_submodules,
        )
        transfer_checkpoint = transfer_checkpoint_to_restore(model_init_config)
        if transfer_checkpoint:
            env = restore_from_checkpoint(env, transfer_checkpoint)
            if model_init_config["model_type"] in MODEL_TYPES_WITH_SUBMODULES:
                env = eqx.tree_at(
                    lambda e: (e.module.p_oh_predictor, e.module.p_rad_predictor),
                    env,
                    _prereq_submodules(train_dl, model_init_config),
                )
        return env

    @staticmethod
    def _make_wtot_loss_fn(loss_config: dict, use_huber: bool) -> IntegralLoss:
        """Device-weighted loss on energy_mhd_MJ_pred, wrapped for time integration.

        use_huber selects the training loss (huber, with the swept huber_delta)
        or the delta-free validation loss (plain absolute error),
        so the sweep metric val/loss.mean cannot be gamed by shrinking delta.

        Wtot counts only the timeslices a reconstruction landed on (trb_utils.freshness_weight),
        the stores hold it forward between reconstructions.
        Training integrates over the equal-length segments,
        validation averages over each episode's fresh time (TimeAveragedLoss), so a long shot weighs no more than a short one.

        The training loss adds the ANCHOR_SIGNALS terms pulling the submodule
        predictions toward the measured signals, weighted by the anchor_weight_* loss_config keys,
        P_oh at the fresh timeslices only and P_rad at every one.
        Validation stays pure Wtot so the sweep metric is comparable across model types.
        The terms drop out at trace time for cases whose target_vars do not
        carry the measured signals (mlp, the plain transformer).
        Anchor errors are plain absolute error: huber_delta is swept on the MJ-scale Wtot residuals,
        meaningless for the MW-scale powers, and the anchors are not worth a second delta
        """
        device_weights = loss_config.get("device_weights", {})
        anchor_weights = {signal: loss_config[weight_key] for signal, (_, weight_key) in ANCHOR_SIGNALS.items()} if use_huber else {}

        def loss_fn(pred, targ):
            residual = pred.energy_mhd_MJ_pred - targ["energy_mhd_MJ"].data
            errors = optax.huber_loss(residual, delta=loss_config["huber_delta"]) if use_huber else jnp.abs(residual)
            sample_weights = per_sample_device_values(targ["ds_source_idx"].data, device_weights, 1.0)
            wtot_weights = sample_weights * freshness_weight("energy_mhd_MJ", targ)
            loss = jnp.mean(wtot_weights * errors)

            for signal, anchor_weight in anchor_weights.items():
                if anchor_weight <= 0.0 or signal not in targ:
                    continue
                anchor_errors = jnp.abs(getattr(pred, ANCHOR_SIGNALS[signal][0]) - targ[signal].data)
                anchor_sample_weights = sample_weights * freshness_weight(signal, targ)
                loss = loss + anchor_weight * jnp.mean(anchor_sample_weights * anchor_errors)
            return loss

        # Validation scores whole episodes, where an integral would weigh each shot by its length,
        # so it divides by the fresh time and every shot counts once.
        # Training stays an integral, its segments are equal length so dividing by their duration would only rescale the loss.
        # Dividing by their fresh time would weigh each fresh timeslice by 1 / (fresh time of its segment)
        if use_huber:
            return IntegralLoss(loss_fn)
        return TimeAveragedLoss(loss_fn, "fresh_equilibrium")

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        return PowerBalanceTRB._make_wtot_loss_fn(loss_config, use_huber=True)

    @staticmethod
    def get_val_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        return PowerBalanceTRB._make_wtot_loss_fn(loss_config, use_huber=False)

    @staticmethod
    def get_val_eval_suite(suite_config) -> EvaluationSuite | None:
        """Validation suite computing the delta-free loss (sweep metric val/loss.mean)."""
        if suite_config is None:
            return None
        return make_loss_eval_suite(PowerBalanceTRB.get_val_loss_fn(suite_config["loss_config"]))

    @staticmethod
    def get_optimizer(optimizer_config: dict) -> optax.GradientTransformation:
        return make_grouped_exponential_adamw(optimizer_config, no_decay_names=SCALING_LAW_FIELDS)

    @staticmethod
    def get_test_eval_suite(suite_config) -> EvaluationSuite | None:
        """The stored-energy study results (trb_utils.scalar_study_results), None for sweep trials."""
        if not suite_config:
            return None
        return {"study_results": lambda eval_data: scalar_study_results(eval_data, "energy_mhd_MJ", "output.energy_mhd_MJ_pred")}
