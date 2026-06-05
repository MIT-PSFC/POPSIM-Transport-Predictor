from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from popsim.cfspopcon_jax.density_peaking import calc_effective_collisionality
from popsim.ml import DataLoader, IntegralLoss, TrainRunBuilder
from popsim.ml.checkpointing import create_default_checkpoint_manager, restore_model
from popsim.ml.dataloading import DEFAULT_SAMPLE_DIM, make_dataloaders
from popsim.ml.eval import EvalData, EvaluationSuite
from popsim.ml.train_config import load_dict

from transport_study.modules.profile_trajectory.data import get_ds
from transport_study.modules.profile_trajectory.module import (
    ProfileTrajectoryOptimizer,
    ProfileTrajectoryOptimizerEnv,
)
from transport_study.profile_transfer.restore_predictor import (
    restore_profile_predictor,
)
from transport_study.trajectory_optimization.setup_data import (
    make_augmented_dataset,
)


class ProfileTrajectoryOptimizerTRB(TrainRunBuilder):
    @staticmethod
    def get_dataloaders(
        dataloader_config: dict,
    ) -> tuple[xr.Dataset, DataLoader, DataLoader, DataLoader]:
        """
        Get the dataset and dataloaders for training.

        Original dataset creation is shared between trajectory optimization and the profile predictor
        Profile predictor gets time-independent dataloaders (just train and val),
        while trajectory optimization gets two identical time-dependent dataloaders with a lot of augmented traces
        """
        if dataloader_config.get("module") == "profile_trajectory":
            ds_ref_dir = Path(dataloader_config["scratch_dir"]) / "predict_first" / "raw_data"
            ds_ref, _ = get_ds(
                ds_ref_dir / f"{dataloader_config['ref_shot']}.nc",
                selected_shots={
                    # TODO(ZanderKeith) make this not hardcoded
                    dataloader_config["ref_shot"]: {
                        "start": 2.6,
                        "end": 5.1,
                    }
                },
                fresh_profiles=False,
                debug=dataloader_config["debug"],
            )

            ds_aug = make_augmented_dataset(
                ds=ds_ref,
                debug=dataloader_config["debug"],
                prng_seed=dataloader_config["prng_seed"],
            )
            train_dl, val_dl = make_dataloaders(
                datasets=[
                    ds_aug,
                    ds_aug,
                ],  # Using the same augmented dataset to both train and validate the trajectory
                time_coord="time",
                episode_coord="shot_alt",
                input_vars=dataloader_config["input_vars"],
                target_vars=dataloader_config["target_vars"],
                convert_xr_to_jnp=dataloader_config["convert_xr_to_jnp"],
                state_init_vars=dataloader_config["state_vars"],
                extra_vars=dataloader_config.get("extra_vars", None),
                batch_size=dataloader_config["batch_size"],
                shuffle=[True, False],
            )
            ds = ds_aug
        else:
            raise ValueError(f"Called with a module that isn't a profile trajectory optimizer: {dataloader_config.get('module')}")

        return ds, train_dl, val_dl, val_dl

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        """Initialize the model to be trained"""
        submodule_configs = model_init_config["submodules"]

        config = ProfileTrajectoryOptimizer.Config(
            traj_times=jnp.asarray(model_init_config["traj_times"]),
            input_ranges=model_init_config["input_ranges"],
            derived_shape_ranges=model_init_config.get("derived_shape_ranges", {}),
        )

        profile_predictor = restore_profile_predictor(load_dict(submodule_configs["profile_predictor"]))

        # Seed all trajectory variables from the programmed waveforms of sample 0.
        # Use nearest-index lookup since the time coordinate is 2D after make_augmented_dataset.
        meta = train_dl.dataset.training_metadata
        sample_dim = meta.sample_dim
        time_dim = meta.time_dep_metadata.time_dim
        ds_first = train_dl.ds.isel({sample_dim: 0})
        time_arr = ds_first["time"].values if "time" in ds_first else ds_first[time_dim].values

        def _at_traj_times(sig_name: str) -> jnp.ndarray:
            arr = ds_first[sig_name].values
            return jnp.array([arr[int(np.argmin(np.abs(time_arr - float(t))))] for t in config.traj_times])

        trajectory = {
            "R0": _at_traj_times("R0_prog"),
            "gapin": _at_traj_times("gapin_prog"),
            "rxpt1": _at_traj_times("rxbot_prog"),
            "zxpt1": _at_traj_times("zxbot_prog"),
            "rxpt2": _at_traj_times("rxtop_prog"),
            "zxpt2": _at_traj_times("zxtop_prog"),
            "ne20_edge": _at_traj_times("ne20_edge_prog"),
        }

        module = ProfileTrajectoryOptimizer.init(
            config=config,
            profile_predictor=profile_predictor,
            psigrid=profile_predictor.psigrid,
            trajectory=trajectory,
        )

        # Wrap the module in an environment since it's time-dependent
        env = ProfileTrajectoryOptimizerEnv(
            module=module,
            optimize_density=model_init_config.get("optimize_density", False),
        )

        if model_init_config.get("restore_main_module", False):
            manager = create_default_checkpoint_manager(model_init_config["checkpoint_dir"])
            env = restore_model(manager, env)

        return env

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        # Typical Zeff for DIII-D H-mode plasmas
        Z_EFF = 1.5

        def loss_fn(pred, targ):
            """Loss function based on D-level stability metrics:

            D0 is low risk, D3 is high risk
            D0 :    q_star > 3.5, nu_e* < 0.3
            D1 :    3.0 < q_star <= 3.5, 0.3 <= nu_e* < 0.6
            D2 :    2.5 < q_star <= 3.0, 0.6 <= nu_e* < 1.0
            D3 :    q_star <= 2.5 or nu_e* >= 1.0

            1. Minimize pressure peaking (most important, high-beta shots typically disrupt from MHD activity)
            2. Maintain high q_star (proxy for q_min), penalized in D-level brackets
            3. Have low effective collisionality (nu_e*), penalized in D-level brackets
            4. Soft Greenwald density limit, penalize approach to fGW = 1.5
            """
            ne20_psi = jnp.asarray(pred.profile_predictor_output.ne.data)
            Te_keV_psi = jnp.asarray(pred.profile_predictor_output.te.data)

            # 1. Pressure peaking (most important)
            P_psi = ne20_psi * Te_keV_psi
            avg_P = jnp.mean(P_psi)
            safe_avg_P = jnp.maximum(avg_P, 1e-6)
            peaking = jnp.max(P_psi) / safe_avg_P

            # 2. q_star in D-level brackets (lower q_star = higher risk)
            q_star = pred.q_star
            q_loss = (
                jax.nn.relu(3.5 - q_star)  # D0 -> D1 boundary
                + jax.nn.relu(3.0 - q_star)  # D1 -> D2 boundary
                + jax.nn.relu(2.5 - q_star)  # D2 -> D3 boundary
            )

            # 3. Effective collisionality in D-level brackets
            # calc_effective_collisionality expects ne in [1e19 m^-3]
            ne_avg_1e19 = jnp.mean(ne20_psi) * 10.0
            Te_avg_keV = jnp.mean(Te_keV_psi)
            nu_star = calc_effective_collisionality(ne_avg_1e19, Te_avg_keV, pred.R0, Z_EFF)
            nu_loss = (
                jax.nn.relu(nu_star - 0.3)  # D0 -> D1 boundary
                + jax.nn.relu(nu_star - 0.6)  # D1 -> D2 boundary
                + jax.nn.relu(nu_star - 1.0)  # D2 -> D3 boundary
            )

            # 4. Soft Greenwald limit: penalize fGW approaching 1.5
            gw_loss = jax.nn.softplus(10.0 * (pred.fGW - 1.3))

            return 3.0 * peaking + q_loss + nu_loss + 0.5 * gw_loss + pred.shape_penalty

        return IntegralLoss(loss_fn, nan_strategy="zero")

    @staticmethod
    def get_test_eval_suite(config: dict) -> EvaluationSuite | None:
        if not config:
            return None

        def predicted_profiles(eval_data: EvalData) -> xr.Dataset:
            """Mean and std of predicted ne/Te profiles across augmented shots."""
            ne = eval_data.output_ds["output.profile_predictor_output.ne"]
            te = eval_data.output_ds["output.profile_predictor_output.te"]

            ne_mean = ne.mean(dim=DEFAULT_SAMPLE_DIM, skipna=True)
            ne_std = ne.std(dim=DEFAULT_SAMPLE_DIM, skipna=True)
            te_mean = te.mean(dim=DEFAULT_SAMPLE_DIM, skipna=True)
            te_std = te.std(dim=DEFAULT_SAMPLE_DIM, skipna=True)

            return xr.Dataset(
                {
                    "ne_mean": ne_mean,
                    "ne_std": ne_std,
                    "te_mean": te_mean,
                    "te_std": te_std,
                }
            )

        return {"predicted_profiles": predicted_profiles}

    @staticmethod
    def get_optimizer(optimizer_config: dict) -> optax.GradientTransformation:
        schedule = optax.exponential_decay(
            init_value=optimizer_config["lr0"],
            transition_steps=optimizer_config["transition_steps"],
            decay_rate=optimizer_config["decay_rate"],
            end_value=optimizer_config["lrf"],
        )
        opt = optax.chain(
            optax.clip_by_global_norm(optimizer_config.get("max_grad_norm", 1.0)),
            optax.adamw(learning_rate=schedule, weight_decay=optimizer_config["weight_decay"]),
        )
        return opt
