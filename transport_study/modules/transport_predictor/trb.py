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
from transport_study.modules.normalization import (
    CoralFeatureNormalizer,
    FeatureNormalizer,
    ZScoreFeatureNormalizer,
)
from transport_study.modules.power_balance.trb import PowerBalanceTRB
from transport_study.modules.profile_predictor.trb import ProfilePredictorTRB
from transport_study.modules.transport_predictor.module import (
    N_TRANSPORT_NN_INPUTS,
    Inputs,
    TransportPredictorEnv,
    TransportPredictorSciML,
    TransportPredictorTorax,
    TransportPredictorToraxSimState,
    TransportPredictorTransformer,
)
from transport_study.modules.trb_utils import (
    get_time_dep_dataloaders,
    make_loss_eval_suite,
)

STUDY_TYPE = "transport_transfer"


def _fit_transport_input_normalizer(train_ds: xr.Dataset, n_devices: int, data_normalization: str) -> FeatureNormalizer:
    """Fit the per-device stat stage (CORAL or z-score) on the 11 transport_nn_inputs.

    Evaluates the module's own feature math over the flattened training data,
    with the beta-derived entries computed from the MEASURED stored energy
    (at runtime the modules use the state-implied Wtot instead). Rows with
    incomplete features or an unattributable device index are dropped by the fit.
    """
    reference = train_ds["Ip_MA"]

    def col(var: str) -> np.ndarray:
        return np.asarray(train_ds[var].broadcast_like(reference).values, dtype=float).ravel()

    source_idx = col("ds_source_idx")
    inputs = Inputs(
        Ip_MA=col("Ip_MA"),
        B0=col("B0"),
        ne20_line_avg=col("ne20_line_avg"),
        R0=col("R0"),
        a_minor=col("a_minor"),
        kappa=col("kappa"),
        delta_top=col("delta_top"),
        delta_bot=col("delta_bot"),
        P_aux_MW=col("P_aux_MW"),
        ds_source_idx=source_idx,
    )
    features = np.asarray(inputs.transport_nn_inputs(col("Wtot_MJ"))).T  # (N, N_TRANSPORT_NN_INPUTS)
    attributed = ~np.isnan(source_idx)
    normalizer_cls = ZScoreFeatureNormalizer if data_normalization == "physics-zscore" else CoralFeatureNormalizer
    return normalizer_cls.fit_from_features(features[attributed], source_idx[attributed].astype(int), n_devices)


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
            # Stat stage (CORAL or z-score) on the 11 transport features.
            # Fitted from the training data, or left at identity when the
            # physics features are used as-is or a transfer checkpoint will
            # overwrite the buffers anyway (a stat fit on a handful of target
            # shots is ill-conditioned, the restored stats are the correct
            # ones; the identity class must still match the checkpoint's
            # pytree). transfer_pretrain dataloaders carry a combined historic
            # + target fit dataset as an attribute (see get_time_dep_dataloaders)
            data_normalization = model_init_config.get("data_normalization", "physics-coral")
            if data_normalization not in ("physics", "physics-coral", "physics-zscore"):
                raise ValueError(f"Unknown transport data normalization method: {data_normalization}")
            n_devices = len(config.ds_source_to_idx)
            if data_normalization == "physics":
                normalizer: FeatureNormalizer = CoralFeatureNormalizer.identity(n_devices, N_TRANSPORT_NN_INPUTS)
            elif model_init_config.get("transfer_checkpoint"):
                identity_cls = ZScoreFeatureNormalizer if data_normalization == "physics-zscore" else CoralFeatureNormalizer
                normalizer = identity_cls.identity(n_devices, N_TRANSPORT_NN_INPUTS)
            else:
                normalizer = _fit_transport_input_normalizer(
                    getattr(train_dl, "normalizer_fit_ds", train_dl.ds), n_devices, data_normalization
                )

            if model_type == "sciml":
                # Submodule skeletons come from their own TRBs, then their
                # trained weights are restored from the prereq case checkpoints
                pb_config = _submodule_config_dict(model_init_config["submodules"]["power_balance"])
                prof_config = _submodule_config_dict(model_init_config["submodules"]["profile_predictor"])
                pb_env = PowerBalanceTRB.model_init(train_dl, pb_config["model_init_config"])
                profile_module = ProfilePredictorTRB.model_init(train_dl, prof_config["model_init_config"])
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
                }[model_init_config.get("torax_state", "rebuild")]
                module = torax_cls.init(
                    rhogrid=np.asarray(train_dl.ds["rho"]),
                    torax_config=model_init_config["torax_config"],
                    nn_width=model_init_config["nn_width"],
                    nn_depth=model_init_config["nn_depth"],
                    prng_seed=model_init_config.get("prng_seed", 42),
                    normalizer=normalizer,
                    sim_dt=model_init_config["sim_dt"],
                    transport_model=model_type.removeprefix("torax-"),
                    geometry_builder=model_init_config.get("geometry_builder", "circular"),
                    delta_exponent=model_init_config.get("delta_exponent", 2.0),
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
    def _make_profile_loss_fn(loss_config: dict, use_huber: bool) -> IntegralLoss:
        """Device-weighted loss on the predicted ne/te profiles, wrapped for time integration.

        Profiles are peak-normalized per timeslice (scale from the target
        only, floored) so the two channels and all devices are commensurate
        and the huber delta reads as a fractional error, same convention as
        the profile study loss. No gradient or error-bar terms: the transport
        Output carries profile values only.

        use_huber selects the training loss (huber, with the swept
        huber_delta) or the delta-free validation loss (plain absolute error),
        so the sweep metric val/loss.mean cannot be gamed by shrinking delta.
        """
        if "device_weights" not in loss_config:
            device_weights = dict.fromkeys(config.dataset_paths, 1.0)
        else:
            device_weights = loss_config["device_weights"]

        def loss_fn(pred, targ):
            ne_targ = targ["ne20_rho"].data
            te_targ = targ["Te_keV_rho"].data

            floor = ProfilePredictorTRB.PROFILE_SCALE_FLOOR
            ne_scale = jnp.maximum(jnp.max(jnp.abs(ne_targ), axis=-1, keepdims=True), floor)
            te_scale = jnp.maximum(jnp.max(jnp.abs(te_targ), axis=-1, keepdims=True), floor)

            ne_resid = (pred.ne - ne_targ) / ne_scale
            te_resid = (pred.te - te_targ) / te_scale
            if use_huber:
                ne_err = optax.huber_loss(ne_resid, delta=loss_config["huber_delta"])
                te_err = optax.huber_loss(te_resid, delta=loss_config["huber_delta"])
            else:
                ne_err = jnp.abs(ne_resid)
                te_err = jnp.abs(te_resid)

            # Build per-sample weights from device labels
            ds_source_idx = targ["ds_source_idx"].data
            sample_weights = jnp.ones(ds_source_idx.shape, dtype=ne_err.dtype)
            for device, weight in device_weights.items():
                sample_weights = jnp.where(
                    ds_source_idx == config.ds_source_to_idx[device],
                    weight,
                    sample_weights,
                )

            # Broadcast sample weights across the rho axis
            while sample_weights.ndim < ne_err.ndim:
                sample_weights = sample_weights[..., None]

            return 0.5 * (jnp.mean(sample_weights * ne_err) + jnp.mean(sample_weights * te_err))

        return IntegralLoss(loss_fn)

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        return TransportPredictorTRB._make_profile_loss_fn(loss_config, use_huber=True)

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
        # Same clipped AdamW as the profile study: the differentiated TORAX
        # solve can spike gradients and NaN a run without the global-norm cap
        return ProfilePredictorTRB.get_optimizer(optimizer_config)

    @staticmethod
    def get_test_eval_suite(suite_config) -> EvaluationSuite:
        """Evaluation suite for testing after training.

        The profile study's study_results already handles time-dependent
        outputs (it renames the stepper's time_idx_input dim back) and emits
        exactly the per-channel and combined error variables the analysis
        stack reads, so it is reused as-is.
        """
        return ProfilePredictorTRB.get_test_eval_suite(suite_config)
