"""Training run builder for the TORAX-based profile predictor.

:class:`ProfilePredictorTorax` is JIT-compilable and end-to-end differentiable
through the six transport/source parameters.  However, training directly through
the full TORAX PDE solver with gradient descent is expensive and unstable.
This TRB therefore uses a **two-stage fitting** strategy that is more robust:

Stage 1 - Offline parameter fitting
    For each training sample, find the 7 transport parameters
    (chi_e, chi_i, source_rho, source_width, D_e, ne_peaking, P_scale) that
    minimise the profile error when passed directly to TORAX.  This is done via
    ``scipy.optimize`` (Nelder-Mead) on the *decoded* parameter space (i.e. the
    raw NN output vector).  The fitted parameter vectors are stored as a
    ``fitted_params`` ``xr.DataArray`` in the dataset.  This stage uses
    ``torax.run_simulation`` directly (not the JIT-compiled step function) since
    it runs outside of JAX tracing.

Stage 2 - NN supervised learning
    Train the NN (RtdMLP) to predict the fitted raw parameter vectors from the
    9 equilibrium inputs.  This stage is fully differentiable and uses standard
    JAX gradient descent via the POPSIM ``TrainRunBuilder`` machinery.

The TRB exposes both stages through its public interface.  Stage 1 is invoked
once (or periodically), and its outputs are cached.  Stage 2 runs every epoch.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torax
import xarray as xr
from loguru import logger
from popsim.ml import DataLoader, TrainRunBuilder
from popsim.ml.dataloading import make_dataloaders
from popsim.ml.eval import EvalData, EvaluationSuite, batched_model_eval_and_loss

from transport_study import EPISODE_DIM, TIME_COORD, TIME_DIM
from transport_study.modules.profile_predictor.module import Inputs
from transport_study.modules.profile_predictor.torax_module import (
    _CHI_MAX,
    _CHI_MIN,
    _D_E_MAX,
    _D_E_MIN,
    _NE_PEAK_MAX,
    _NE_PEAK_MIN,
    _P_SCALE_MAX,
    _P_SCALE_MIN,
    _P_TOTAL_MAX,
    _P_TOTAL_MIN,
    _SRC_RHO_MAX,
    _SRC_RHO_MIN,
    _SRC_WIDTH_MAX,
    _SRC_WIDTH_MIN,
    ProfilePredictorTorax,
    _build_torax_config,
    _estimate_P_total,
    _rho_to_psigrid,
)
from transport_study.orchestration.organize_data import (
    get_train_test_datasets,
    get_train_val_datasets,
)

# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 helpers - fitting transport parameters per sample
# ──────────────────────────────────────────────────────────────────────────────

_NN_OUT_SIZE = 7  # must match torax_module._NN_OUT_SIZE


def _decode_raw(raw: np.ndarray) -> tuple[float, ...]:
    """Convert raw 7-element NN output vector to physical parameter tuple.

    Returns:
        (chi_e, chi_i, source_rho, source_width, D_e, ne_peaking, P_scale)
    """

    chi_e = float(np.clip(np.exp(raw[0]), _CHI_MIN, _CHI_MAX))
    chi_i = float(np.clip(chi_e * np.exp(raw[1]), _CHI_MIN, _CHI_MAX))
    source_rho = float(
        np.clip(1.0 / (1.0 + np.exp(-raw[2])), _SRC_RHO_MIN, _SRC_RHO_MAX)
    )
    source_wid = float(np.clip(np.exp(raw[3]), _SRC_WIDTH_MIN, _SRC_WIDTH_MAX))
    D_e = float(np.clip(np.exp(raw[4]), _D_E_MIN, _D_E_MAX))
    ne_peaking = float(np.clip(np.exp(raw[5]), _NE_PEAK_MIN, _NE_PEAK_MAX))
    P_scale = float(np.clip(np.exp(raw[6]), _P_SCALE_MIN, _P_SCALE_MAX))
    return chi_e, chi_i, source_rho, source_wid, D_e, ne_peaking, P_scale


def _torax_profile_error(
    raw: np.ndarray,
    inputs: Inputs,
    ne_target: np.ndarray,
    te_target: np.ndarray,
    psi_n_target: np.ndarray,
    psigrid: tuple,
    t_edge_keV: float,
    dt_steady: float,
    n_rho: int,
) -> float:
    """Objective function for Stage-1 per-sample parameter fitting.

    Runs TORAX with the decoded parameters and returns the Huber profile error
    against the target profiles.  Used by ``scipy.optimize`` (no gradients).

    Args:
        raw: 7-element raw NN parameter vector.
        inputs: Equilibrium parameters for this sample.
        ne_target: Target electron density [10²⁰ m⁻³] on ``psi_n_target``.
        te_target: Target electron temperature [keV] on ``psi_n_target``.
        psi_n_target: psi_n grid of the targets.
        psigrid: psi_n grid for TORAX output interpolation.
        t_edge_keV: Edge temperature BC [keV].
        dt_steady: Single TORAX timestep [s].
        n_rho: TORAX radial grid resolution.

    Returns:
        Scalar loss value (mean Huber loss over the psi grid).
    """
    chi_e, chi_i, source_rho, source_wid, D_e, ne_peaking, P_scale = _decode_raw(raw)
    # _estimate_P_total returns a JAX scalar; float() converts it for numpy/scipy.
    P_physics = float(_estimate_P_total(inputs))
    P_total = float(np.clip(P_physics * P_scale, _P_TOTAL_MIN, _P_TOTAL_MAX))

    try:
        cfg = _build_torax_config(
            inputs=inputs,
            chi_e=chi_e,
            chi_i=chi_i,
            source_rho=source_rho,
            source_width=source_wid,
            D_e=D_e,
            ne_peaking=ne_peaking,
            P_total=P_total,
            t_edge_keV=t_edge_keV,
            dt_steady=dt_steady,
            n_rho=n_rho,
        )
        torax_cfg = torax.ToraxConfig.from_dict(cfg)
        _, state_history = torax.run_simulation(
            torax_cfg, log_timestep_info=False, progress_bar=False
        )
        if state_history.sim_error != torax.SimError.NO_ERROR:
            return 1e6

        final = state_history.core_profiles[-1]
        rho_grid = np.asarray(state_history.rho_cell_norm)
        te_rho = np.asarray(final.T_e.value)
        ne_rho = np.asarray(final.n_e.value) / 1e20
        te_psi, ne_psi = _rho_to_psigrid(te_rho, ne_rho, rho_grid, psigrid)

        # Interpolate onto the target grid for loss computation.
        te_pred_t = np.interp(psi_n_target, np.asarray(psigrid), te_psi)
        ne_pred_t = np.interp(psi_n_target, np.asarray(psigrid), ne_psi)

        delta = 0.5
        te_err = np.where(
            np.abs(te_pred_t - te_target) <= delta,
            0.5 * (te_pred_t - te_target) ** 2,
            delta * (np.abs(te_pred_t - te_target) - 0.5 * delta),
        ).mean()
        ne_err = np.where(
            np.abs(ne_pred_t - ne_target) <= delta,
            0.5 * (ne_pred_t - ne_target) ** 2,
            delta * (np.abs(ne_pred_t - ne_target) - 0.5 * delta),
        ).mean()
        return float(te_err + ne_err)

    except Exception as exc:
        logger.debug(f"TORAX error in objective: {exc}")
        return 1e6


def fit_transport_params_for_sample(
    inputs: Inputs,
    ne_target: np.ndarray,
    te_target: np.ndarray,
    psi_n_target: np.ndarray,
    psigrid: tuple,
    t_edge_keV: float = 0.2,
    dt_steady: float = 5.0,
    n_rho: int = 25,
    raw_init: np.ndarray | None = None,
    max_iter: int = 50,
) -> np.ndarray:
    """Fit the 7 raw transport parameters for a single sample using Nelder-Mead.

    This is Stage 1 of the two-stage training strategy: for each sample, find
    the raw parameter vector ``raw`` such that TORAX produces profiles closest
    to the targets.  The result is used as the supervised training label for
    Stage 2.

    Args:
        inputs: Equilibrium parameters for this sample.
        ne_target: Target n_e profile [10²⁰ m⁻³] on ``psi_n_target``.
        te_target: Target T_e profile [keV] on ``psi_n_target``.
        psi_n_target: psi_n coordinate of the target profiles.
        psigrid: psi_n output grid for TORAX.
        t_edge_keV: Edge temperature boundary condition [keV].
        dt_steady: TORAX single timestep [s].
        n_rho: TORAX radial grid resolution.
        raw_init: Optional starting point (7-element array in raw NN space).
            Defaults to zeros (chi_e = chi_i = 1 m²/s, D_e = 1 m²/s, etc.).
        max_iter: Maximum Nelder-Mead iterations.

    Returns:
        Fitted 7-element raw parameter vector (in NN output space, not decoded).
    """
    from scipy.optimize import minimize

    if raw_init is None:
        raw_init = np.zeros(_NN_OUT_SIZE, dtype=np.float64)

    result = minimize(
        _torax_profile_error,
        x0=raw_init,
        args=(
            inputs,
            ne_target,
            te_target,
            psi_n_target,
            psigrid,
            t_edge_keV,
            dt_steady,
            n_rho,
        ),
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-4},
    )
    return result.x.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 TRB - NN supervised learning on fitted params
# ──────────────────────────────────────────────────────────────────────────────


class ProfilePredictorToraxTRB(TrainRunBuilder):
    """Training run builder for :class:`ProfilePredictorTorax`.

    Training proceeds in two stages:

    **Stage 1** (offline, called once before training):
        :meth:`fit_all_transport_params` loops over the training dataset and
        runs TORAX to fit the 7 raw transport parameters per sample.  The
        resulting ``fitted_params`` variable (shape ``(sample, 7)``) is added
        to the dataset.

    **Stage 2** (online, every epoch):
        The NN is trained in the standard supervised fashion to predict the
        ``fitted_params`` from the 9 equilibrium ``nn_inputs``.  This stage is
        fully differentiable.

    To use Stage 1 before calling :func:`run_study` or the POPSIM training
    loop, call::

        fitted_ds = ProfilePredictorToraxTRB.fit_all_transport_params(
            ds_train, model, psigrid, ...
        )
        # Then pass fitted_ds as the training dataset.
    """

    @staticmethod
    def get_dataloaders(
        dataloader_config: dict,
    ) -> tuple[xr.Dataset, tuple[DataLoader, DataLoader, DataLoader]]:
        """Same data-loading logic as :class:`ProfilePredictorTRB`."""
        training_data = dataloader_config["training_data"]
        if dataloader_config["domain_adaptation"] is None:
            ds_train, ds_val = get_train_val_datasets(
                training_data=training_data,
                data_normalization=dataloader_config["data_normalization"],
                study_type="profile_transfer",
                debug=dataloader_config.get("debug", False),
            )
        else:
            ds_train, ds_val = get_train_test_datasets(
                training_data=training_data,
                data_normalization=dataloader_config["data_normalization"],
                domain_adaptation=dataloader_config["domain_adaptation"],
                num_hp_shots=dataloader_config["num_hp_shots"],
                hp_test_set_size=dataloader_config.get("hp_test_set_size", None),
                study_type="profile_transfer",
                debug=dataloader_config.get("debug", False),
            )

        ds_train = ds_train.drop_vars(TIME_DIM, errors="ignore")
        ds_val = ds_val.drop_vars(TIME_DIM, errors="ignore")

        input_vars = dataloader_config["input_vars"]
        target_vars = dataloader_config["target_vars"]

        train_dl, val_dl = make_dataloaders(
            datasets=(ds_train, ds_val),
            time_coord=TIME_COORD,
            episode_coord=EPISODE_DIM,
            input_vars=input_vars,
            target_vars=target_vars,
            batch_size=dataloader_config.get("batch_size", None),
            shuffle=[True, False],
            convert_xr_to_jnp=False,
        )
        return ds_val, train_dl, val_dl, val_dl

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        """Initialise a :class:`ProfilePredictorTorax` from config."""
        psigrid = np.asarray(train_dl.ds["psi_n"])
        return ProfilePredictorTorax.init(
            psigrid=psigrid,
            nn_width=model_init_config["nn_width"],
            nn_depth=model_init_config["nn_depth"],
            dt_steady=model_init_config.get("dt_steady", 5.0),
            n_rho=model_init_config.get("n_rho", 25),
            t_edge_keV=model_init_config.get("t_edge_keV", 0.2),
            prng_seed=model_init_config.get("prng_seed", 42),
        )

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        """MSE loss on the 7 fitted transport parameters (Stage 2).

        The NN predicts the raw 7-element parameter vector; the target is the
        ``fitted_params`` variable added by :meth:`fit_all_transport_params`.
        """

        def loss_fn(nn_raw_pred: jnp.ndarray, targ: dict) -> jnp.ndarray:
            fitted = targ["fitted_params"].data  # shape (..., 7)
            return jnp.mean((nn_raw_pred - fitted) ** 2)

        return loss_fn

    @staticmethod
    def get_optimizer(config: dict) -> optax.GradientTransformation:
        schedule = optax.exponential_decay(
            init_value=config["lr0"],
            transition_steps=config["transition_steps"],
            decay_rate=config["decay_rate"],
            end_value=config["lrf"],
        )
        return optax.adamw(learning_rate=schedule, weight_decay=config["weight_decay"])

    @staticmethod
    def get_trainable_getter(model_init_config: dict) -> Callable[[Any], Any] | None:
        """Only the NN weights are trainable (psigrid and hyper-params are static)."""

        def get_nn_leaves(module: ProfilePredictorTorax):
            return jax.tree.leaves(module.nn)

        return get_nn_leaves

    @staticmethod
    def get_val_eval_suite(suite_config: dict) -> EvaluationSuite:
        """Validation evaluates the mean squared error on fitted parameters."""
        loss_config = suite_config["loss_config"]
        loss_fn = ProfilePredictorToraxTRB.get_loss_fn(loss_config)

        def eval_fn(inp: EvalData) -> dict:
            loss_vecs = []
            for batch in inp.dataloader:
                inputs, targets = batch.get_inputs_and_targets()
                loss_vec = batched_model_eval_and_loss(
                    inp.model, loss_fn, inputs, targets
                )
                loss_vecs.append(loss_vec)
            loss_vec = jnp.concatenate(loss_vecs)
            return {"mean": loss_vec.mean(), "vec": loss_vec}

        return {"loss": eval_fn}

    @staticmethod
    def get_test_eval_suite(config: dict) -> EvaluationSuite:
        """Test evaluation runs the full TORAX forward pass and measures profile error."""
        loss_config = config["loss_config"]
        huber_delta = loss_config.get("huber_delta", 0.5)

        def eval_fn(inp: EvalData) -> dict:
            model: ProfilePredictorTorax = inp.model
            results = []

            for batch in inp.dataloader:
                raw_inputs, targets = batch.get_inputs_and_targets()
                for i in range(len(raw_inputs)):
                    sample_inputs = jax.tree.map(lambda x, i=i: x[i], raw_inputs)
                    sample_target = jax.tree.map(lambda x, i=i: x[i], targets)
                    try:
                        pred = model(sample_inputs)
                        ne_targ = np.asarray(sample_target["ne20_psi"].data)
                        te_targ = np.asarray(sample_target["Te_keV_psi"].data)
                        ne_pred = np.asarray(pred.ne.data)
                        te_pred = np.asarray(pred.te.data)
                        psi_n = np.asarray(pred.ne.coords["psi_n"])

                        te_hub = np.where(
                            np.abs(te_pred - te_targ) <= huber_delta,
                            0.5 * (te_pred - te_targ) ** 2,
                            huber_delta
                            * (np.abs(te_pred - te_targ) - 0.5 * huber_delta),
                        )
                        ne_hub = np.where(
                            np.abs(ne_pred - ne_targ) <= huber_delta,
                            0.5 * (ne_pred - ne_targ) ** 2,
                            huber_delta
                            * (np.abs(ne_pred - ne_targ) - 0.5 * huber_delta),
                        )
                        sample_loss = float(
                            np.trapz(te_hub, psi_n) + np.trapz(ne_hub, psi_n)
                        )
                        results.append(sample_loss)
                    except Exception as exc:
                        logger.warning(f"TORAX evaluation failed for sample {i}: {exc}")
                        results.append(float("nan"))

            loss_arr = np.array(results, dtype=np.float32)
            valid = loss_arr[~np.isnan(loss_arr)]
            return {
                "mean": float(np.mean(valid)) if len(valid) > 0 else float("nan"),
                "vec": jnp.array(loss_arr),
            }

        return {"loss": eval_fn}

    # ------------------------------------------------------------------
    # Stage 1: offline transport-parameter fitting
    # ------------------------------------------------------------------

    @staticmethod
    def fit_all_transport_params(
        ds: xr.Dataset,
        model: ProfilePredictorTorax,
        max_iter: int = 50,
        show_progress: bool = True,
    ) -> xr.Dataset:
        """Fit per-sample transport parameters via TORAX and add to dataset.

        For each sample in ``ds``, runs :func:`fit_transport_params_for_sample`
        to find the 7 raw parameter values that minimise the TORAX profile error
        against the ground-truth profiles.  Results are stored in a new
        ``fitted_params`` variable (shape ``(sample, 7)``).

        This is a **slow, embarrassingly parallel** operation that should be run
        once and cached.  Use multiprocessing externally if needed.

        Args:
            ds: Dataset with variables ``Ip_MA``, ``B0``, ``betan``,
                ``ne20_edge``, ``R0``, ``a_minor``, ``kappa``, ``delta_top``,
                ``delta_bot``, ``ne20_psi``, ``Te_keV_psi``.
            model: The :class:`ProfilePredictorTorax` instance (provides
                ``psigrid``, ``dt_steady``, ``n_rho``, ``t_edge_keV``).
            max_iter: Maximum Nelder-Mead iterations per sample.
            show_progress: Whether to log progress.

        Returns:
            Dataset with an additional ``fitted_params`` variable of shape
            ``(sample, fitted_param)``.
        """
        sample_dim = "sample"

        # Stack to a flat sample dimension if needed.
        if sample_dim not in ds.dims:
            ds = ds.stack({sample_dim: [d for d in ds.dims if d != "psi_n"]})

        n_samples = ds.sizes[sample_dim]
        psi_n = np.asarray(ds.coords["psi_n"])
        fitted = np.zeros((n_samples, _NN_OUT_SIZE), dtype=np.float32)

        nn_raw_init = None  # warm-start from previous fit when possible

        for i in range(n_samples):
            s = ds.isel({sample_dim: i})

            inp = Inputs(
                Ip=float(s["Ip_MA"].values),
                B0=float(s["B0"].values),
                betan=float(s["betan"].values),
                ne20=float(s["ne20_edge"].values),
                R0=float(s["R0"].values),
                a_minor=float(s["a_minor"].values),
                kappa=float(s["kappa"].values),
                delta_top=float(s["delta_top"].values),
                delta_bot=float(s["delta_bot"].values),
                psi=jnp.array(model.psigrid),
            )
            ne_targ = np.asarray(s["ne20_psi"].values)
            te_targ = np.asarray(s["Te_keV_psi"].values)

            raw = fit_transport_params_for_sample(
                inputs=inp,
                ne_target=ne_targ,
                te_target=te_targ,
                psi_n_target=psi_n,
                psigrid=model.psigrid,
                t_edge_keV=model.t_edge_keV,
                dt_steady=model.dt_steady,
                n_rho=model.n_rho,
                raw_init=nn_raw_init,
                max_iter=max_iter,
            )
            fitted[i] = raw
            nn_raw_init = raw  # warm-start next sample

            if show_progress and (i % max(1, n_samples // 20) == 0):
                logger.info(f"Fitted transport params: {i + 1}/{n_samples}")

        ds = ds.assign(
            {
                "fitted_params": xr.DataArray(
                    fitted,
                    dims=[sample_dim, "fitted_param"],
                    coords={"fitted_param": np.arange(_NN_OUT_SIZE)},
                )
            }
        )
        return ds
