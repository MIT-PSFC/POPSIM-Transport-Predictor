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
from popsim.ml.eval import EvalData, EvaluationSuite

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.config import config
from transport_study.modules.normalization import make_normalizer
from transport_study.modules.power_balance.module import (
    PowerBalanceEnv,
    PowerBalanceScalingLaw,
    PowerBalanceSciML,
    PowerBalanceTransformer,
    PowerBalanceUnstructuredNN,
)
from transport_study.modules.power_balance.p_oh.trb import OhmicPowerTRB
from transport_study.modules.power_balance.p_rad.trb import RadiatedPowerTRB
from transport_study.modules.trb_utils import (  # noqa: F401 re-exported, historical import location
    get_time_dep_dataloaders,
    integrate_error_over_time,
    make_exponential_adamw,
    make_grouped_exponential_adamw,
    make_loss_eval_suite,
    mask_to_largest_contiguous_segment,
)

STUDY_TYPE = "power_balance_transfer"

# Anchor terms in the training loss, keyed by measured target signal:
# (Output attribute holding the model's own prediction, loss_config key for the weight)
ANCHOR_SIGNALS = {
    "P_oh_MW": ("P_oh_MW_pred", "anchor_weight_p_oh"),
    "P_rad_MW": ("P_rad_MW_pred", "anchor_weight_p_rad"),
}


class PowerBalanceTRB(TrainRunBuilder):
    """TrainRunBuilder for the power balance modules used in transfer learning"""

    @staticmethod
    def get_dataloaders(
        dataloader_config: dict,
    ) -> tuple[xr.Dataset, DataLoader, DataLoader, DataLoader]:
        """Dataset and dataloaders for training, see trb_utils.get_time_dep_dataloaders."""
        return get_time_dep_dataloaders(dataloader_config, STUDY_TYPE)

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        """
        Instantiate and return your model given a training DataLoader
        and a model config dict.
        """

        def _build_module(train_dl: DataLoader, model_init_config: dict) -> Any:
            model_type = model_init_config["model_type"]
            # Fit normalization stats from the training data only. When a
            # transfer checkpoint will overwrite the module anyway, skip the
            # fit (a CORAL fit on a handful of target shots is ill-conditioned
            # and the restored stats, fitted on historic + target shots by the
            # transfer_pretrain prereq case, are the correct ones).
            # transfer_pretrain dataloaders carry that combined fit dataset as
            # an attribute (see get_dataloaders)
            n_devices = len(config.ds_source_to_idx)
            if model_init_config.get("transfer_checkpoint"):
                fit_ds = None
            else:
                fit_ds = getattr(train_dl, "normalizer_fit_ds", train_dl.ds)
            normalizer = make_normalizer(model_init_config["data_normalization"], fit_ds, n_devices)
            if model_type in ["sciml-taue-scalinglaw", "sciml-taue-nn"]:
                p_oh_config = model_init_config["submodules"]["p_oh_predictor"]
                if isinstance(p_oh_config, TrainConfig):
                    p_oh_config = p_oh_config.model_dump()
                p_oh_predictor = OhmicPowerTRB.model_init(train_dl, p_oh_config["model_init_config"])
                p_rad_config = model_init_config["submodules"]["p_rad_predictor"]
                if isinstance(p_rad_config, TrainConfig):
                    p_rad_config = p_rad_config.model_dump()
                p_rad_predictor = RadiatedPowerTRB.model_init(train_dl, p_rad_config["model_init_config"])
                if model_init_config["restore_submodules"]:
                    p_oh_manager = create_default_checkpoint_manager(p_oh_config["checkpoint_dir"])
                    p_oh_predictor = restore_model(p_oh_manager, p_oh_predictor)
                    p_rad_manager = create_default_checkpoint_manager(p_rad_config["checkpoint_dir"])
                    p_rad_predictor = restore_model(p_rad_manager, p_rad_predictor)

            if model_type == "sciml-taue-scalinglaw":
                module = PowerBalanceScalingLaw.init(
                    p_oh_predictor=p_oh_predictor,
                    p_rad_predictor=p_rad_predictor,
                    min_taue=model_init_config.get("min_taue", None),
                    max_taue=model_init_config.get("max_taue", None),
                )
            elif model_type == "sciml-taue-nn":
                module = PowerBalanceSciML.init(
                    p_oh_predictor=p_oh_predictor,
                    p_rad_predictor=p_rad_predictor,
                    normalizer=normalizer,
                    in_size=model_init_config["in_size"],
                    out_size=model_init_config["out_size"],
                    nn_width=model_init_config["nn_width"],
                    nn_depth=model_init_config["nn_depth"],
                    min_taue=model_init_config.get("min_taue", None),
                    max_taue=model_init_config.get("max_taue", None),
                    prng_seed=model_init_config.get("prng_seed", 42),
                )
            elif model_type == "mlp":
                module = PowerBalanceUnstructuredNN.init(
                    in_size=model_init_config["in_size"],
                    out_size=model_init_config["out_size"],
                    nn_width=model_init_config["nn_width"],
                    nn_depth=model_init_config["nn_depth"],
                    normalizer=normalizer,
                    prng_seed=model_init_config.get("prng_seed", 42),
                )
            elif model_type == "transformer":
                module = PowerBalanceTransformer.init(
                    d_model=model_init_config["d_model"],
                    num_heads=model_init_config["num_heads"],
                    history_len=model_init_config["history_len"],
                    nn_width=model_init_config["nn_width"],
                    nn_depth=model_init_config["nn_depth"],
                    normalizer=normalizer,
                    prng_seed=model_init_config.get("prng_seed", 42),
                )
            else:
                raise ValueError(f"Invalid model type: {model_type}")

            return module

        module = _build_module(train_dl, model_init_config)

        if model_init_config.get("freeze_submodules", False):
            freeze_submodules = ["p_oh_predictor", "p_rad_predictor"]
        else:
            freeze_submodules = []

        env = PowerBalanceEnv(
            module=module,
            domain_adaptation=model_init_config["domain_adaptation"],
            freeze_submodules=freeze_submodules,
        )

        if model_init_config.get("transfer_checkpoint", False):
            transfer_manager = create_default_checkpoint_manager(model_init_config["transfer_checkpoint"])
            env = restore_model(transfer_manager, env)
            # Restoring the whole env overwrote the freshly restored submodule weights, restore them again from their own checkpoints
            if model_init_config["model_type"] in ["sciml-taue-scalinglaw", "sciml-taue-nn"]:
                p_oh_config = model_init_config["submodules"]["p_oh_predictor"]
                if isinstance(p_oh_config, TrainConfig):
                    p_oh_config = p_oh_config.model_dump()
                p_rad_config = model_init_config["submodules"]["p_rad_predictor"]
                if isinstance(p_rad_config, TrainConfig):
                    p_rad_config = p_rad_config.model_dump()

                p_oh_manager = create_default_checkpoint_manager(p_oh_config["checkpoint_dir"])
                p_oh_restored = restore_model(p_oh_manager, env.module.p_oh_predictor)
                p_rad_manager = create_default_checkpoint_manager(p_rad_config["checkpoint_dir"])
                p_rad_restored = restore_model(p_rad_manager, env.module.p_rad_predictor)
                env = eqx.tree_at(
                    lambda e: (e.module.p_oh_predictor, e.module.p_rad_predictor),
                    env,
                    (p_oh_restored, p_rad_restored),
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
    def _make_wtot_loss_fn(loss_config: dict, use_huber: bool, include_anchors: bool = False) -> IntegralLoss:
        """Device-weighted loss on Wtot_MJ_pred, wrapped for time integration.

        use_huber selects the training loss (huber, with the swept
        huber_delta) or the delta-free validation loss (plain absolute error),
        so the sweep metric val/loss.mean cannot be gamed by shrinking delta.

        include_anchors adds the ANCHOR_SIGNALS terms pulling the submodule
        predictions toward the measured signals, weighted by the
        anchor_weight_* loss_config keys. Training loss only: validation stays
        pure Wtot so the sweep metric is comparable across model types. The
        terms drop out at trace time for model types whose target_vars do not
        carry the measured signals (mlp, transformer).

        Anchor errors are plain absolute error, not huber.
        huber_delta is swept on the MJ-scale Wtot residuals and is meaningless for the MW-scale powers,
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

        def loss_fn(pred, targ):
            if use_huber:
                errors = optax.huber_loss(
                    pred.Wtot_MJ_pred,
                    targ["Wtot_MJ"].data,
                    delta=loss_config["huber_delta"],
                )
            else:
                errors = jnp.abs(pred.Wtot_MJ_pred - targ["Wtot_MJ"].data)

            # Build per-sample weights from device labels
            ds_source_idx = targ["ds_source_idx"].data
            sample_weights = jnp.ones(ds_source_idx.shape, dtype=errors.dtype)
            for device, weight in device_weights.items():
                sample_weights = jnp.where(
                    ds_source_idx == config.ds_source_to_idx[device],
                    weight,
                    sample_weights,
                )

            # Broadcast sample weights to match the error shape if needed
            while sample_weights.ndim < errors.ndim:
                sample_weights = sample_weights[..., None]

            loss = jnp.mean(sample_weights * errors)

            for signal, anchor_weight in anchor_weights.items():
                if anchor_weight <= 0.0 or signal not in targ:
                    continue
                pred_attr = ANCHOR_SIGNALS[signal][0]
                anchor_errors = jnp.abs(getattr(pred, pred_attr) - targ[signal].data)
                loss = loss + anchor_weight * jnp.mean(sample_weights * anchor_errors)

            return loss

        return IntegralLoss(loss_fn)

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        return PowerBalanceTRB._make_wtot_loss_fn(loss_config, use_huber=True, include_anchors=True)

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
    def get_optimizer(config: dict) -> optax.GradientTransformation:
        return make_grouped_exponential_adamw(config)

    @staticmethod
    def get_test_eval_suite(config) -> EvaluationSuite:
        """Evaluation suite for testing after training."""

        def study_results(eval_data: EvalData) -> xr.Dataset:
            """Calculate final study results
                - Target vs predicted Wtot_MJ
                - Absolute and relative error on a per-timeslice basis
                - Integrated error over time for each shot
            This should maintain the coordinates of the original dataset, in particular `ds_source` and `shot`
            """

            # Unstack sample MultiIndex -> (shot, time_idx) and sqeeze out batch dimension so we can integrate per shot
            targ = eval_data.input_ds.Wtot_MJ.unstack("sample").squeeze()
            pred = eval_data.output_ds["output.Wtot_MJ_pred"].unstack("sample").squeeze()
            time_2d = eval_data.input_ds[TIME_COORD].unstack("sample").squeeze()

            # time-dependent modules modify the time dimension name, change it back to avoid confusion
            targ = targ.rename({TIME_DIM + "_input": TIME_DIM})
            pred = pred.rename({TIME_DIM + "_input": TIME_DIM})
            time_2d = time_2d.rename({TIME_DIM + "_input": TIME_DIM})

            # Get relative error on a per-timeslice basis
            error_abs_ts = xr.apply_ufunc(np.abs, pred - targ)
            error_rel_ts = error_abs_ts / (xr.apply_ufunc(np.abs, targ) + 0.1)

            # Integrate absolute error over time for each shot, ignoring NaN-padded entries
            error_abs_shot = integrate_error_over_time(error_abs_ts, time_2d)
            error_rel_shot = integrate_error_over_time(error_rel_ts, time_2d)

            ds_source = eval_data.input_ds["ds_source"]
            if "sample" in ds_source.dims:
                # Case when there are multiple source datasets present
                ds_source_array = eval_data.input_ds["ds_source"].unstack("sample").squeeze().values
            else:
                # Case when there is a single source dataset present
                ds_source_array = np.array([ds_source.values.item() for _ in range(targ.sizes["shot"])])

            ds = xr.Dataset(
                data_vars={
                    "Wtot_MJ_targ": targ,
                    "Wtot_MJ_pred": pred,
                    "error_abs_ts": error_abs_ts,
                    "error_rel_ts": error_rel_ts,
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
