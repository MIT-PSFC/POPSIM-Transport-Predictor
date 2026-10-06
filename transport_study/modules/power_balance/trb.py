from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import optax
import xarray as xr
from popsim.ml import DataLoader, IntegralLoss, TrainRunBuilder
from popsim.ml.eval import EvaluationSuite

from transport_study.config import config
from transport_study.modules.normalization import make_normalizer
from transport_study.modules.power_balance.module import (
    MODEL_TYPES_WITH_ENERGY_INPUT,
    MODEL_TYPES_WITH_SUBMODULES,
    MULTIOBJECTIVE_MODEL_TYPES,
    SCALING_LAW_FIELDS,
    TRANSFORMER_MODEL_TYPES,
    PowerBalanceEnv,
    PowerBalanceMLP,
    PowerBalanceScalingLaw,
    PowerBalanceSciML,
    PowerBalanceTransformer,
)
from transport_study.modules.power_balance.p_oh.trb import OhmicPowerTRB
from transport_study.modules.power_balance.p_rad.trb import RadiatedPowerTRB
from transport_study.modules.trb_utils import (
    get_time_dep_dataloaders,
    make_grouped_exponential_adamw,
    make_loss_eval_suite,
    normalizer_fit_dataset,
    per_sample_device_values,
    restore_from_checkpoint,
    scalar_study_results,
    submodule_config_dict,
    target_device_idx,
)

STUDY_TYPE = "power_balance_transfer"

# Anchor terms in the training loss of the models with p_oh / p_rad submodules, keyed by measured target signal:
# (Output attribute holding the submodule's prediction, loss_config key of its weight).
# They keep the submodule predictions close to the measured signals.
# Not cheating, a real application has these signals for training alongside the target Wtot
ANCHOR_SIGNALS = {
    "power_ohm_MW": ("power_ohm_MW_pred", "anchor_weight_power_ohm"),
    "power_radiated_MW": ("power_radiated_MW_pred", "anchor_weight_power_radiated"),
}


def _restored_submodules(train_dl: DataLoader, model_init_config: dict) -> tuple:
    """The p_oh and p_rad predictors, built by their own TRBs and restored from their prereq case checkpoints."""
    restored = []
    for name, submodule_trb in (("p_oh_predictor", OhmicPowerTRB), ("p_rad_predictor", RadiatedPowerTRB)):
        submodule_config = submodule_config_dict(model_init_config["submodules"][name])
        submodule = submodule_trb.model_init(train_dl, submodule_config["model_init_config"])
        restored.append(restore_from_checkpoint(submodule, submodule_config["checkpoint_dir"]))
    return tuple(restored)


class PowerBalanceTRB(TrainRunBuilder):
    """TrainRunBuilder for the power balance modules used in transfer learning"""

    @staticmethod
    def get_dataloaders(
        dataloader_config: dict,
    ) -> tuple[xr.Dataset, DataLoader, DataLoader, DataLoader]:
        """Dataset and dataloaders for training, see trb_utils.get_time_dep_dataloaders."""
        return get_time_dep_dataloaders(dataloader_config, STUDY_TYPE)

    @staticmethod
    def _build_module(train_dl: DataLoader, model_init_config: dict) -> Any:
        """The power balance module of model_init_config["model_type"], normalizer fitted on the training data."""
        model_type = model_init_config["model_type"]
        if model_type == "sciml-taue-scalinglaw":
            # The scaling law consumes physical units and holds no normalizer,
            # data_normalization reaches only its p_oh / p_rad prereq cases
            p_oh_predictor, p_rad_predictor = _restored_submodules(train_dl, model_init_config)
            return PowerBalanceScalingLaw.init(p_oh_predictor=p_oh_predictor, p_rad_predictor=p_rad_predictor)
        normalizer = make_normalizer(
            model_init_config["data_normalization"],
            normalizer_fit_dataset(train_dl, model_init_config),
            len(config.ds_source_to_idx),
            target_device_idx(),
            with_energy=model_type in MODEL_TYPES_WITH_ENERGY_INPUT,
        )
        if model_type == "sciml-taue-nn":
            p_oh_predictor, p_rad_predictor = _restored_submodules(train_dl, model_init_config)
            return PowerBalanceSciML.init(
                p_oh_predictor=p_oh_predictor,
                p_rad_predictor=p_rad_predictor,
                normalizer=normalizer,
                nn_width=model_init_config["nn_width"],
                nn_depth=model_init_config["nn_depth"],
                prng_seed=model_init_config["prng_seed"],
            )
        if model_type == "mlp":
            return PowerBalanceMLP.init(
                nn_width=model_init_config["nn_width"],
                nn_depth=model_init_config["nn_depth"],
                normalizer=normalizer,
                prng_seed=model_init_config["prng_seed"],
            )
        if model_type in TRANSFORMER_MODEL_TYPES:
            return PowerBalanceTransformer.init(
                d_model=model_init_config["d_model"],
                num_heads=model_init_config["num_heads"],
                history_len=model_init_config["history_len"],
                nn_width=model_init_config["nn_width"],
                nn_depth=model_init_config["nn_depth"],
                normalizer=normalizer,
                predicts_powers=model_type in MULTIOBJECTIVE_MODEL_TYPES,
                prng_seed=model_init_config["prng_seed"],
            )
        raise ValueError(f"Invalid model type: {model_type}")

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> PowerBalanceEnv:
        """The PowerBalanceEnv of a case, its submodules restored from their prereq cases.

        A transfer case restores the whole env from its transfer_pretrain checkpoint,
        then puts back the submodules of its own p_oh / p_rad prereq cases,
        which ran their own pretrain and finetune on the target device.
        """
        freeze_submodules = ["p_oh_predictor", "p_rad_predictor"] if model_init_config.get("freeze_submodules", False) else []
        env = PowerBalanceEnv(
            module=PowerBalanceTRB._build_module(train_dl, model_init_config),
            domain_adaptation=model_init_config["domain_adaptation"],
            freeze_submodules=freeze_submodules,
        )
        if model_init_config.get("transfer_checkpoint"):
            env = restore_from_checkpoint(env, model_init_config["transfer_checkpoint"])
            if model_init_config["model_type"] in MODEL_TYPES_WITH_SUBMODULES:
                env = eqx.tree_at(
                    lambda e: (e.module.p_oh_predictor, e.module.p_rad_predictor),
                    env,
                    _restored_submodules(train_dl, model_init_config),
                )
        return env

    @staticmethod
    def _make_wtot_loss_fn(loss_config: dict, use_huber: bool) -> IntegralLoss:
        """Device-weighted loss on energy_mhd_MJ_pred, wrapped for time integration.

        use_huber selects the training loss (huber, with the swept huber_delta)
        or the delta-free validation loss (plain absolute error),
        so the sweep metric val/loss.mean cannot be gamed by shrinking delta.

        The training loss adds the ANCHOR_SIGNALS terms pulling the submodule
        predictions toward the measured signals, weighted by the anchor_weight_* loss_config keys.
        Validation stays pure Wtot so the sweep metric is comparable across model types.
        The terms drop out at trace time for model types whose target_vars do not
        carry the measured signals (mlp, transformer).
        Anchor errors are plain absolute error: huber_delta is swept on the MJ-scale Wtot residuals,
        meaningless for the MW-scale powers, and the anchors are not worth a second delta
        """
        device_weights = loss_config.get("device_weights", {})
        anchor_weights = {signal: loss_config[weight_key] for signal, (_, weight_key) in ANCHOR_SIGNALS.items()} if use_huber else {}

        def loss_fn(pred, targ):
            residual = pred.energy_mhd_MJ_pred - targ["energy_mhd_MJ"].data
            errors = optax.huber_loss(residual, delta=loss_config["huber_delta"]) if use_huber else jnp.abs(residual)
            sample_weights = per_sample_device_values(targ["ds_source_idx"].data, device_weights, 1.0)
            loss = jnp.mean(sample_weights * errors)

            for signal, anchor_weight in anchor_weights.items():
                if anchor_weight <= 0.0 or signal not in targ:
                    continue
                anchor_errors = jnp.abs(getattr(pred, ANCHOR_SIGNALS[signal][0]) - targ[signal].data)
                loss = loss + anchor_weight * jnp.mean(sample_weights * anchor_errors)
            return loss

        return IntegralLoss(loss_fn)

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
