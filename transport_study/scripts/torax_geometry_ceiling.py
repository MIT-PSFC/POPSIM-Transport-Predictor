"""Direct TORAX knob-fit probe: circular vs miller representation ceiling.

Bypasses the neural networks entirely. For each selected MAST timeslice,
directly optimizes the TORAX free coefficients (the same bounded knobs the
NNs would output: transport model coefficients, source parameters, edge
boundary conditions) with per-sample Adam through the differentiable TORAX
relaxation, once per geometry builder. The resulting best-fit profile error
is the representation ceiling of that (transport model, geometry) pair on
that sample - no training, no generalization, just "can this simulator
setup match the measured profiles at all".

If the miller geometry specifically fixes high-triangularity MAST shots,
the ceiling gap (circular minus miller) should grow with triangularity.
If both geometries hit the same floor, the missing physics is elsewhere.

The knob parameterization reuses the module's own bounding functions
(sigmoid and tanh ranges from transport_coefficients and
nn_coefficients), so the searchable coefficient set is exactly the set
reachable by a trained network.

Usage:
    python transport_study/scripts/torax_geometry_ceiling.py run \
        --dataset_path /path/to/mast/ds.zarr \
        --out_dir /path/to/output \
        --models cgm,gyrobohm,constant

Writes one netcdf per (model, geometry) arm plus a combined markdown table.
"""

import copy
import json
import time
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from loguru import logger
from torax import experimental as torax_experimental

from transport_study.modules.normalization import CoralFeatureNormalizer
from transport_study.modules.profile_predictor.module import N_NN_INPUTS, Inputs
from transport_study.modules.profile_predictor.torax_module import (
    TAU_REF_S,
    TRANSPORT_COEFFICIENT_NAMES,
    ProfilePredictorTorax,
    _run_loop_jit_with_geo,
)
from transport_study.modules.profile_predictor.train_configs import (
    make_profile_predictor_torax_config,
)

INPUT_SIGNALS = (
    "Ip_MA",
    "B0",
    "betan",
    "ne20_line_avg",
    "R0",
    "a_minor",
    "kappa",
    "delta_top",
    "delta_bot",
)

SOURCE_EDGE_COEFF_NAMES = (
    "S_total",
    "P_aux_total",
    "gaussian_location",
    "gaussian_width",
    "electron_heat_fraction",
    "n_e_right_bc",
    "T_e_right_bc",
)

# Same softening as ProfilePredictorTRB REL_ERROR_FLOOR_FRAC / PROFILE_SCALE_FLOOR
REL_ERROR_FLOOR_FRAC = 0.1
PROFILE_SCALE_FLOOR = 1e-2

Z_CLIP = 6.0


def select_samples(
    ds: xr.Dataset,
    n_low: int,
    n_mid: int,
    n_high: int,
) -> xr.Dataset:
    """One flattop timeslice per shot, shots picked across the triangularity range.

    Candidate slices need fresh profiles, finite scalar inputs, mostly-finite
    measured profiles, and non-degenerate betan and density. Per shot the
    candidate with maximum betan is taken as the peak-performance
    representative (max Ip lands at the start of flattop before the shape
    develops, which confounds the triangularity axis with early-time
    transients). Shots are then sorted by mean triangularity and
    n_low / n_mid / n_high shots are drawn from the bottom, middle and top
    of the distribution.
    """
    finite_inputs = xr.ones_like(ds["Ip_MA"], dtype=bool)
    for sig in INPUT_SIGNALS:
        finite_inputs = finite_inputs & np.isfinite(ds[sig])
    valid = finite_inputs & (ds["betan"] > 0.2) & (ds["ne20_line_avg"] > 0.05)
    if "fresh_profiles" in ds:
        valid = valid & ds["fresh_profiles"].astype(bool)
    frac_finite_te = np.isfinite(ds["Te_keV_rho"]).mean(dim="rho")
    frac_finite_ne = np.isfinite(ds["ne20_rho"]).mean(dim="rho")
    valid = valid & (frac_finite_te > 0.8) & (frac_finite_ne > 0.8)
    valid = valid & (ds["Te_keV_rho"].max(dim="rho") > 0.05)

    betan_masked = ds["betan"].where(valid)
    has_candidate = valid.any(dim="time_idx").values
    best_tidx = betan_masked.fillna(-np.inf).argmax(dim="time_idx").values

    shots = ds["shot"].values[has_candidate]
    tidx = best_tidx[has_candidate]
    slice_ds = ds.isel(time_idx=xr.DataArray(tidx, dims="shot", coords={"shot": shots})).sel(shot=shots)

    delta_mean = ((slice_ds["delta_top"] + slice_ds["delta_bot"]) / 2).values
    order = np.argsort(delta_mean)
    n_shots = len(order)
    lo = order[:n_low]
    mid_center = n_shots // 2
    mid = order[mid_center - n_mid // 2 : mid_center - n_mid // 2 + n_mid]
    hi = order[-n_high:]
    picked = np.concatenate([lo, mid, hi])
    picked = np.unique(picked)
    logger.info(
        f"{n_shots} candidate shots, picked {len(picked)}: "
        f"delta_mean range {delta_mean[picked].min():.3f} to {delta_mean[picked].max():.3f}"
    )
    return slice_ds.isel(shot=picked)


def build_module(
    transport_model: str,
    geometry_builder: str,
    rhogrid: np.ndarray,
    t_final: float,
    fixed_dt: float,
) -> ProfilePredictorTorax:
    cfg = make_profile_predictor_torax_config(transport_model, geometry_builder=geometry_builder)
    mic = cfg["model_init_config"]
    torax_config = copy.deepcopy(mic["torax_config"])
    torax_config["numerics"]["t_final"] = t_final
    torax_config["numerics"]["fixed_dt"] = fixed_dt
    return ProfilePredictorTorax.init(
        rhogrid=np.asarray(rhogrid),
        torax_config=torax_config,
        nn_width=mic["nn_width"],
        nn_depth=mic["nn_depth"],
        prng_seed=0,
        normalizer=CoralFeatureNormalizer.identity(1, N_NN_INPUTS),
        transport_model=transport_model,
        geometry_builder=geometry_builder,
        delta_exponent=mic["delta_exponent"],
    )


def make_inputs(vec: jax.Array, rho: jax.Array) -> Inputs:
    return Inputs(
        Ip=vec[0],
        B0=vec[1],
        betan=vec[2],
        ne20_line_avg=vec[3],
        R0=vec[4],
        a_minor=vec[5],
        kappa=vec[6],
        delta_top=vec[7],
        delta_bot=vec[8],
        ds_source_idx=jnp.asarray(0, dtype=jnp.int32),
        rho=rho,
    )


def _wide_transport_coefficients(module: ProfilePredictorTorax, zt: jax.Array) -> dict:
    """Widened transport knob ranges for the bound-limit probe.

    Run 1 showed the NN-parity fit rails the particle knobs (D at its
    floor, V_face at max pinch, gas puff at max), so the density ceiling
    is bound-limited. These ranges deliberately exceed what the trained
    networks can output, to measure how much ceiling the bounds cost.
    """
    if module.transport_model == "cgm":
        return {
            "chi_e_i_ratio": 0.05 + 9.95 * jax.nn.sigmoid(zt[0:1]),
            "chi_D_ratio": jnp.exp(jnp.log(0.5) + (jnp.log(200.0) - jnp.log(0.5)) * jax.nn.sigmoid(zt[1:2])),
            "VR_D_ratio": 20.0 * jnp.tanh(zt[2:3]),
            "alpha": 1.0 + 2.0 * jax.nn.sigmoid(zt[3:4]),
            "chi_stiff": 0.1 + 9.9 * jax.nn.sigmoid(zt[4:5]),
        }
    elif module.transport_model == "gyrobohm":
        return {
            "chi_bohm_multiplier": jnp.exp(5.0 * jnp.tanh(zt[0:1])),
            "chi_gyrobohm_multiplier": jnp.exp(5.0 * jnp.tanh(zt[1:2])),
            "D_face_c1": 0.01 + 9.99 * jax.nn.sigmoid(zt[2:3]),
            "D_face_c2": 0.01 + 9.99 * jax.nn.sigmoid(zt[3:4]),
            "V_face_coeff": 5.0 * jnp.tanh(zt[4:5]),
        }
    elif module.transport_model == "qlknn":
        return {
            "ITG_flux_ratio_correction": jnp.exp(4.0 * jnp.tanh(zt[0:1])),
            "ETG_correction_factor": (1.0 / 3.0) * jnp.exp(4.0 * jnp.tanh(zt[1:2])),
            "collisionality_multiplier": jnp.exp(4.0 * jnp.tanh(zt[2:3])),
        }
    raise ValueError(f"wide bounds not defined for transport model '{module.transport_model}'")


def coeffs_from_z(
    module: ProfilePredictorTorax,
    inputs: Inputs,
    z: jax.Array,
    wide: bool = False,
    s_scale: float = 1.0,
) -> dict:
    """Map an unbounded parameter vector to TORAX coefficients.

    With wide False, uses exactly the bounding functions the NNs use, so
    the reachable coefficient set matches what a trained network could
    produce. Layout: transport coefficients first (model dependent count),
    then the five source coefficients, then the two edge fractions. With
    wide True, transport / source / edge ranges are widened past the NN
    parameterization to measure how bound-limited the ceiling is.
    """
    n_t = len(TRANSPORT_COEFFICIENT_NAMES[module.transport_model])
    zs = z[n_t : n_t + 5]
    ze = z[n_t + 5 : n_t + 7]
    if wide:
        coeffs = _wide_transport_coefficients(module, z[:n_t])
        # Gas puff up to ~120e21 per s instead of the softplus(Z_CLIP) ~ 6
        coeffs["S_total"] = 20.0 * jax.nn.softplus(zs[0:1])
        coeffs["P_aux_total"] = 8.0 * jax.nn.sigmoid(zs[1:2] - 2.0) * inputs.w_approx / TAU_REF_S
        coeffs["n_e_right_bc"] = (0.01 + 1.49 * jax.nn.sigmoid(ze[0:1])) * inputs.ne20_line_avg
        te_scale = jnp.clip(inputs.te_approx, 0.05, 5.0)
        coeffs["T_e_right_bc"] = 0.005 + jax.nn.sigmoid(ze[1:2] - 3.0) * te_scale
    else:
        coeffs = module.transport_coefficients(z[:n_t])
        # NN-parity mappings, kept in sync with nn_coefficients. The 2026-07-20
        # probe rounds informed the module changes now reflected here: S_total
        # scaled by particle inventory / TAU_REF_S and the 0.01 n_e_right_bc
        # floor. s_scale > 1 additionally multiplies the fueling scale to
        # emulate softplus reach past the probe's Z_CLIP
        inventory = 0.1 * inputs.ne20_line_avg * inputs.volume_approx
        coeffs["S_total"] = s_scale * jax.nn.softplus(zs[0:1]) * inventory / TAU_REF_S
        coeffs["P_aux_total"] = 4.0 * jax.nn.sigmoid(zs[1:2] - 2.0) * inputs.w_approx / TAU_REF_S
        coeffs["n_e_right_bc"] = (0.01 + 0.94 * jax.nn.sigmoid(ze[0:1])) * inputs.ne20_line_avg
        te_scale = jnp.clip(inputs.te_approx, 0.05, 5.0)
        coeffs["T_e_right_bc"] = 0.02 + jax.nn.sigmoid(ze[1:2] - 5.0) * te_scale
    coeffs["gaussian_location"] = 0.8 * jax.nn.sigmoid(zs[2:3])
    coeffs["gaussian_width"] = 0.05 + 0.35 * jax.nn.sigmoid(zs[3:4])
    coeffs["electron_heat_fraction"] = 0.2 + 0.6 * jax.nn.sigmoid(zs[4:5])
    return coeffs


def forward_profiles(module: ProfilePredictorTorax, inputs: Inputs, coeffs: dict) -> tuple[jax.Array, jax.Array]:
    """TORAX relaxation with prescribed coefficients, mirrors module.__call__."""
    new_provider, geo_provider = module.build_provider_and_geo(inputs, coeffs)
    initial_state, initial_post = torax_experimental.get_initial_state_and_post_processed_outputs(
        step_fn=module.step_fn,
        runtime_params_overrides=new_provider,
        geometry_overrides=geo_provider,
    )
    state, _post = _run_loop_jit_with_geo(
        step_fn=module.step_fn,
        input_state=initial_state,
        previous_post_processed_outputs=initial_post,
        runtime_params_overrides=new_provider,
        geo_provider=geo_provider,
        max_steps=module.max_steps,
        wrap_body_in_checkpoint=True,
    )
    ne = state.core_profiles.n_e.value / 1e20
    te = state.core_profiles.T_e.value
    rho_cells = jnp.asarray(module.rho_norm_grid)
    rho_full = jnp.concatenate([jnp.zeros(1), rho_cells, jnp.ones(1)])
    ne_full = jnp.concatenate([ne[:1], ne, coeffs["n_e_right_bc"]])
    te_full = jnp.concatenate([te[:1], te, coeffs["T_e_right_bc"]])
    ne_interp = jnp.interp(inputs.rho, rho_full, ne_full)
    te_interp = jnp.interp(inputs.rho, rho_full, te_full)
    return te_interp, ne_interp


def rel_error(pred: jax.Array, targ: jax.Array) -> jax.Array:
    """NaN-masked mean relative error with the study's softened denominator."""
    finite = jnp.isfinite(targ)
    targ_safe = jnp.nan_to_num(targ)
    peak = jnp.maximum(jnp.max(jnp.abs(jnp.where(finite, targ_safe, 0.0))), PROFILE_SCALE_FLOOR)
    err = jnp.abs(pred - targ_safe) / (jnp.abs(targ_safe) + REL_ERROR_FLOOR_FRAC * peak)
    return jnp.sum(jnp.where(finite, err, 0.0)) / jnp.maximum(jnp.sum(finite), 1)


def fit_arm(
    module: ProfilePredictorTorax,
    sample_vecs: np.ndarray,
    te_targ: np.ndarray,
    ne_targ: np.ndarray,
    rho: np.ndarray,
    n_starts: int,
    n_adam_steps: int,
    lr: float,
    seed: int,
    wide: bool = False,
    s_scale: float = 1.0,
) -> dict:
    """Per-sample multistart Adam fit of the knob vector, vmapped on device."""
    n_samples = sample_vecs.shape[0]
    n_params = len(TRANSPORT_COEFFICIENT_NAMES[module.transport_model]) + 7
    rho_j = jnp.asarray(rho)
    # Decay to lr/10 over the run so late steps refine instead of orbiting
    schedule = optax.exponential_decay(lr, transition_steps=max(n_adam_steps // 3, 1), decay_rate=10 ** (-1 / 3))
    opt = optax.adam(schedule)

    def objective(z, vec, te_t, ne_t):
        inputs = make_inputs(vec, rho_j)
        coeffs = coeffs_from_z(module, inputs, z, wide=wide, s_scale=s_scale)
        te_pred, ne_pred = forward_profiles(module, inputs, coeffs)
        te_rel = rel_error(te_pred, te_t)
        ne_rel = rel_error(ne_pred, ne_t)
        return 0.5 * (te_rel + ne_rel)

    def fit_one(z0, vec, te_t, ne_t):
        def step(carry, _):
            z, opt_state, best_loss, best_z = carry
            loss, grad = jax.value_and_grad(objective)(z, vec, te_t, ne_t)
            grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
            updates, opt_state = opt.update(grad, opt_state)
            z_new = jnp.clip(optax.apply_updates(z, updates), -Z_CLIP, Z_CLIP)
            better = jnp.isfinite(loss) & (loss < best_loss)
            best_loss = jnp.where(better, loss, best_loss)
            best_z = jnp.where(better, z, best_z)
            return (z_new, opt_state, best_loss, best_z), loss

        carry0 = (z0, opt.init(z0), jnp.inf, z0)
        (_zf, _, best_loss, best_z), losses = jax.lax.scan(step, carry0, None, length=n_adam_steps)
        return best_loss, best_z, losses

    def eval_one(z, vec, te_t, ne_t):
        inputs = make_inputs(vec, rho_j)
        coeffs = coeffs_from_z(module, inputs, z, wide=wide, s_scale=s_scale)
        te_pred, ne_pred = forward_profiles(module, inputs, coeffs)
        return rel_error(te_pred, te_t), rel_error(ne_pred, ne_t), te_pred, ne_pred

    # Batch layout: sample-major, starts within a sample are contiguous
    vecs_b = jnp.repeat(jnp.asarray(sample_vecs), n_starts, axis=0)
    te_b = jnp.repeat(jnp.asarray(te_targ), n_starts, axis=0)
    ne_b = jnp.repeat(jnp.asarray(ne_targ), n_starts, axis=0)
    key = jax.random.PRNGKey(seed)
    z0_random = jax.random.uniform(key, (n_samples, n_starts, n_params), minval=-2.0, maxval=2.0)
    z0 = z0_random.at[:, 0, :].set(0.0).reshape(n_samples * n_starts, n_params)

    t0 = time.monotonic()
    best_loss, best_z, losses = jax.jit(jax.vmap(fit_one))(z0, vecs_b, te_b, ne_b)
    best_loss = np.asarray(best_loss)
    logger.info(f"fit done in {time.monotonic() - t0:.1f}s, loss histories shape {losses.shape}")

    # Winner start per sample
    best_loss_ss = best_loss.reshape(n_samples, n_starts)
    winner = best_loss_ss.argmin(axis=1)
    flat_idx = np.arange(n_samples) * n_starts + winner
    z_win = np.asarray(best_z)[flat_idx]

    te_rel, ne_rel, te_pred, ne_pred = jax.jit(jax.vmap(eval_one))(
        jnp.asarray(z_win),
        jnp.asarray(sample_vecs),
        jnp.asarray(te_targ),
        jnp.asarray(ne_targ),
    )

    coeff_names = list(TRANSPORT_COEFFICIENT_NAMES[module.transport_model]) + list(SOURCE_EDGE_COEFF_NAMES)
    coeff_rows = []
    for i in range(n_samples):
        inputs = make_inputs(jnp.asarray(sample_vecs[i]), rho_j)
        coeffs = coeffs_from_z(module, inputs, jnp.asarray(z_win[i]), wide=wide, s_scale=s_scale)
        coeff_rows.append([float(np.asarray(coeffs[name]).squeeze()) for name in coeff_names])

    return {
        "best_loss": best_loss_ss.min(axis=1),
        "te_rel": np.asarray(te_rel),
        "ne_rel": np.asarray(ne_rel),
        "te_pred": np.asarray(te_pred),
        "ne_pred": np.asarray(ne_pred),
        "z_win": z_win,
        "coeff_names": coeff_names,
        "coeff_values": np.array(coeff_rows),
        "loss_histories": np.asarray(losses).reshape(n_samples, n_starts, n_adam_steps),
    }


def run(
    dataset_path: str,
    out_dir: str,
    models: str = "cgm,gyrobohm,constant",
    geometries: str = "circular,miller,ellipse",
    n_low: int = 8,
    n_mid: int = 8,
    n_high: int = 8,
    n_starts: int = 4,
    n_adam_steps: int = 250,
    lr: float = 0.03,
    t_final: float = 0.2,
    fixed_dt: float = 0.01,
    seed: int = 0,
    wide_bounds: bool = False,
    s_scale: float = 1.0,
):
    """Fit TORAX knobs per sample for each (model, geometry) arm and compare."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Fire parses comma-separated flags into a tuple already
    def as_list(flag):
        if isinstance(flag, (list, tuple)):
            return [str(m).strip() for m in flag]
        return [m.strip() for m in str(flag).split(",") if m.strip()]

    model_list = as_list(models)
    geometry_list = as_list(geometries)

    ds = xr.open_zarr(dataset_path)
    samples = select_samples(ds, n_low, n_mid, n_high).load()
    rho = samples["rho"].values
    sample_vecs = np.stack([samples[sig].values for sig in INPUT_SIGNALS], axis=1).astype(np.float64)
    te_targ = samples["Te_keV_rho"].values.astype(np.float64)
    ne_targ = samples["ne20_rho"].values.astype(np.float64)
    shots = samples["shot"].values
    times = samples["time"].values
    delta_mean = (samples["delta_top"].values + samples["delta_bot"].values) / 2

    # The ellipse arm is the miller builder with the triangularities zeroed:
    # an exact quadrature ellipse. It isolates metric accuracy from shaping,
    # since the circular builder uses large-aspect-ratio analytic formulas
    # that are approximate at MAST aspect ratio while miller integrates the
    # contour exactly
    delta_cols = [INPUT_SIGNALS.index("delta_top"), INPUT_SIGNALS.index("delta_bot")]
    ellipse_vecs = sample_vecs.copy()
    ellipse_vecs[:, delta_cols] = 0.0

    summary_rows = {}
    for transport_model in model_list:
        for geometry_arm in geometry_list:
            suffix = "_wide" if wide_bounds else ("_frees" if s_scale != 1.0 else "")
            arm = f"{transport_model}_{geometry_arm}{suffix}"
            logger.info(f"=== arm {arm} ===")
            geometry_builder = "miller" if geometry_arm == "ellipse" else geometry_arm
            arm_vecs = ellipse_vecs if geometry_arm == "ellipse" else sample_vecs
            module = build_module(transport_model, geometry_builder, rho, t_final, fixed_dt)
            result = fit_arm(
                module,
                arm_vecs,
                te_targ,
                ne_targ,
                rho,
                n_starts=n_starts,
                n_adam_steps=n_adam_steps,
                lr=lr,
                seed=seed,
                wide=wide_bounds,
                s_scale=s_scale,
            )
            arm_ds = xr.Dataset(
                {
                    "best_loss": ("sample", result["best_loss"]),
                    "te_rel": ("sample", result["te_rel"]),
                    "ne_rel": ("sample", result["ne_rel"]),
                    "te_pred": (("sample", "rho"), result["te_pred"]),
                    "ne_pred": (("sample", "rho"), result["ne_pred"]),
                    "te_targ": (("sample", "rho"), te_targ),
                    "ne_targ": (("sample", "rho"), ne_targ),
                    "coeff_values": (("sample", "coeff"), result["coeff_values"]),
                    "z_win": (("sample", "param"), result["z_win"]),
                    "loss_history": (("sample", "start", "adam_step"), result["loss_histories"]),
                    "delta_mean": ("sample", delta_mean),
                    "kappa": ("sample", samples["kappa"].values),
                    "time_s": ("sample", times),
                },
                coords={
                    "sample": np.arange(len(shots)),
                    "shot": ("sample", shots),
                    "rho": rho,
                    "coeff": result["coeff_names"],
                },
                attrs={
                    "transport_model": transport_model,
                    "geometry_arm": geometry_arm,
                    "geometry_builder": geometry_builder,
                    "wide_bounds": int(wide_bounds),
                    "t_final": t_final,
                    "fixed_dt": fixed_dt,
                    "n_starts": n_starts,
                    "n_adam_steps": n_adam_steps,
                    "lr": lr,
                },
            )
            arm_ds.to_netcdf(out_path / f"arm_{arm}.nc")
            summary_rows[arm] = result["best_loss"]
            logger.info(f"arm {arm}: mean best loss {result['best_loss'].mean():.4f}")

    # Markdown summary: per sample sorted by triangularity, one column per arm
    order = np.argsort(delta_mean)
    arms = list(summary_rows)
    lines = [
        "# TORAX knob-fit ceiling, circular vs miller",
        "",
        f"Samples: {len(shots)} MAST flattop timeslices, {n_starts} starts x {n_adam_steps} Adam steps, "
        f"t_final={t_final}, fixed_dt={fixed_dt}",
        "",
        "| shot | t [s] | delta | kappa | " + " | ".join(arms) + " |",
        "|" + "---|" * (4 + len(arms)),
    ]
    for i in order:
        cells = " | ".join(f"{summary_rows[a][i]:.4f}" for a in arms)
        lines.append(f"| {shots[i]} | {times[i]:.3f} | {delta_mean[i]:.3f} | {samples['kappa'].values[i]:.2f} | {cells} |")
    mean_cells = " | ".join(f"{summary_rows[a].mean():.4f}" for a in arms)
    lines.append(f"| mean | | | | {mean_cells} |")
    (out_path / "summary.md").write_text("\n".join(lines) + "\n")

    meta = {
        "dataset_path": str(dataset_path),
        "shots": [int(s) for s in shots],
        "delta_mean": [float(d) for d in delta_mean],
        "arms": {a: [float(v) for v in summary_rows[a]] for a in arms},
    }
    (out_path / "summary.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Wrote {out_path / 'summary.md'}")


if __name__ == "__main__":
    fire.Fire({"run": run})
