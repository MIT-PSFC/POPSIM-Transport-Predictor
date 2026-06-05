from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loguru import logger
from popsim.cfspopcon_jax.density_peaking import calc_effective_collisionality

matplotlib.use("Agg")

from transport_study.config import config
from transport_study.datasets.d3d.d3d_dataset import INNER_WALL
from transport_study.modules.profile_predictor.module import (
    Inputs as ProfilePredictorInputs,
)
from transport_study.modules.profile_trajectory.data import (
    add_gapin_prog,
    correct_B0_prog,
    get_ds,
)
from transport_study.profile_transfer.restore_predictor import (
    restore_profile_predictor_from_checkpoint,
)
from transport_study.trajectory_optimization.optimize import (
    TRAJ_TIMES,
    TrajectoryOptimization,
    run_trajectory_optimization,
)
from transport_study.trajectory_optimization.predict_first import get_traj_shot_data
from transport_study.trajectory_optimization.setup_data import make_augmented_dataset

BACKGROUND_COLOR = "#2F2F2F"
FACE_COLOR = "#1A1A1A"
TEXT_COLOR = "white"

MAX_NUM_TRAJ_TIMES = 6
Z_EFF = 1.5  # Typical Zeff for DIII-D H-mode plasmas, matching trb.py

PROFILE_MODULES = [
    # Best overall on historic data
    # "unstructured_nn.cmod.physics..True.-1",
    # "shape_init_kmeans.tcv.physics..True.-1",
    # "shape_init_pca.cmod_tcv.physics.transfer.False.32",
    # "shape_init_pca.cmod.physics.transfer.False.32",
    # Best on target shot compared to IDA
    # "shape_init_pca.tcv.physics.transfer.False.-1",
    "shape_init_pca.exnihilo.physics..False.-1",
    # "shape_init_kmeans.cmod.physics.mixing.True.32",
]


def _profile_module_to_checkpoint_dir(profile_module: str, profopt_models_dir: Path | str) -> Path:
    """Convert a dot-separated profile module string into the ProfileStudy checkpoint directory.

    Input format:  model_type.training_data.data_normalization.domain_adaptation.freeze_shapes.num_hp_shots
    e.g.           unstructured_nn.cmod.physics..True.-1
                   shape_init_pca.cmod_tcv.physics.transfer.False.32

    Replicates ProfileStudy.Case.__str__() naming logic.
    """
    model_type, training_data, _data_norm, domain_adaptation, freeze_str, num_hp_str = profile_module.split(".")
    freeze_shapes = freeze_str == "True"
    num_hp_shots = int(num_hp_str)

    if domain_adaptation:
        case_str = f"case.{model_type}.td_{training_data}.da_{domain_adaptation}.freeze_{freeze_shapes}.hp_{num_hp_shots}"
    elif training_data == "exnihilo":
        case_str = f"case.{model_type}.td_{training_data}.freeze_{freeze_shapes}.hp_{num_hp_shots}"
    else:
        case_str = f"case.{model_type}.td_{training_data}.freeze_{freeze_shapes}"

    return Path(profopt_models_dir) / case_str


def _compute_loss_metrics(
    ds_ref: xr.Dataset,
    profile_predictor,
    optimized_trajectory_dir: Path | str | None,
) -> dict[str, float]:
    """Run profile predictor on the augmented reference trajectory and return mean loss components.

    If optimized_trajectory_dir is provided, overwrites the programmed trajectory with the
    optimized one before augmentation.

    Returns a dict with keys: total, peaking, q_loss, nu_loss, gw_loss, q_star, nu_star, fGW.
    Each value is the mean over all augmented shots and time steps.
    """
    ds = ds_ref.copy()

    # Step 0: Overwrite programmed signals with optimized trajectory if provided
    if optimized_trajectory_dir is not None:
        traj_path = Path(optimized_trajectory_dir) / "optimized_trajectory.nc"
        ds_traj = xr.open_dataset(traj_path)
        traj_signal_map = {
            "iptipp": ("Ip_MA_prog", 1e-6),  # A → MA
            "bttbt": ("B0_prog", 1.0),
            "bmtpwrtar": ("betan_prog", 1.0),
            "dstdenp": ("ne20_edge_prog", 0.1),  # 1e19 → 1e20 m^-3
            "idtrp": ("R0_prog", 1.0),
            "idtrxbot": ("rxbot_prog", 1.0),
            "idtzxbot": ("zxbot_prog", 1.0),
            "idtrxtop": ("rxtop_prog", 1.0),
            "idtzxtop": ("zxtop_prog", 1.0),
        }
        updated = {}
        for traj_var, (ds_var, scale) in traj_signal_map.items():
            if traj_var in ds_traj and ds_var in ds:
                orig = ds[ds_var].load()
                new_vals = orig.copy()
                new_vals.loc[{"shot": config.ref_shot}] = ds_traj[traj_var].values * scale
                updated[ds_var] = new_vals
        if updated:
            ds = ds.assign(updated)

    ds = correct_B0_prog(ds)
    ds = add_gapin_prog(ds)
    # gapin_opt must be applied AFTER add_gapin_prog, which overwrites gapin_prog
    if optimized_trajectory_dir is not None and "gapin_opt" in ds_traj:
        orig = ds["gapin_prog"].load()
        new_vals = orig.copy()
        gapin_opt_vals = ds_traj["gapin_opt"].values.copy().astype(float)
        # Back-fill then forward-fill any NaN edge values (float32/float64 mask
        # mismatch in _build_opt_waveform can leave the first index as NaN in
        # older saved trajectories)
        nan_mask = np.isnan(gapin_opt_vals)
        if nan_mask.any():
            valid_idxs = np.where(~nan_mask)[0]
            if len(valid_idxs):
                fill = np.interp(
                    np.arange(len(gapin_opt_vals)),
                    valid_idxs,
                    gapin_opt_vals[valid_idxs],
                )
                gapin_opt_vals[nan_mask] = fill[nan_mask]
        new_vals.loc[{"shot": config.ref_shot}] = gapin_opt_vals
        ds = ds.assign({"gapin_prog": new_vals})
    ds_aug = make_augmented_dataset(ds)

    # Derive shape variables (matches predict_first logic)
    R0 = ds_aug["R0_prog"].values  # (shot_alt, time_idx)
    gapin = ds_aug["gapin_prog"].values
    rxbot = ds_aug["rxbot_prog"].values
    zxbot = ds_aug["zxbot_prog"].values
    rxtop = ds_aug["rxtop_prog"].values
    zxtop = ds_aug["zxtop_prog"].values

    a_minor = R0 - gapin - INNER_WALL
    kappa = np.abs(zxtop - zxbot) / (a_minor * 2)
    delta_bot = (R0 - rxbot) / a_minor
    delta_top = (R0 - rxtop) / a_minor

    n_shot_alt, n_time = R0.shape
    psi_grid = np.array(profile_predictor.psigrid)

    Ip_flat = ds_aug["Ip_MA_prog"].values.reshape(-1)
    valid_mask = ~np.isnan(Ip_flat)
    psi_tiled = jnp.tile(jnp.array(psi_grid), (n_shot_alt * n_time, 1))

    inputs_batched = ProfilePredictorInputs(
        Ip=ds_aug["Ip_MA_prog"].values.reshape(-1),
        B0=ds_aug["B0_prog"].values.reshape(-1),
        betan=ds_aug["betan_prog"].values.reshape(-1),
        ne20=ds_aug["ne20_edge_prog"].values.reshape(-1),
        R0=R0.reshape(-1),
        a_minor=a_minor.reshape(-1),
        kappa=kappa.reshape(-1),
        delta_top=delta_top.reshape(-1),
        delta_bot=delta_bot.reshape(-1),
        psi=psi_tiled,
    )

    def _predict(inp):
        out = profile_predictor(inp)
        return out.ne.data, out.te.data

    ne_flat, te_flat = jax.vmap(_predict)(inputs_batched)

    valid_jnp = jnp.array(valid_mask)[:, None]
    ne_flat = jnp.where(valid_jnp, ne_flat, jnp.nan)
    te_flat = jnp.where(valid_jnp, te_flat, jnp.nan)

    ne = np.array(ne_flat).reshape(n_shot_alt, n_time, -1)  # (shot_alt, time, psi)
    te = np.array(te_flat).reshape(n_shot_alt, n_time, -1)

    ne_nan_frac = np.isnan(ne).mean()
    if ne_nan_frac > 0.5:
        label = f"optimized_trajectory_dir={optimized_trajectory_dir!r}" if optimized_trajectory_dir else "baseline"
        logger.warning(
            f"[_compute_loss_metrics] {ne_nan_frac:.1%} of ne predictions are NaN for {label}. "
            "Loss metrics will be unreliable. This usually means the trajectory contains out-of-range "
            "shape parameters — check gapin_opt values and derived shape bounds."
        )

    # Compute q_star and fGW from ProfilePredictorInputs properties via scalar vmap
    q_star_flat = np.array(jax.vmap(lambda inp: inp.q_star)(inputs_batched))
    fGW_flat = np.array(jax.vmap(lambda inp: inp.fGW)(inputs_batched))
    q_star = q_star_flat.reshape(n_shot_alt, n_time)
    fGW = fGW_flat.reshape(n_shot_alt, n_time)

    # Loss components (matching trb.py, computed per (shot_alt, time) then averaged)
    P = ne * te  # (shot_alt, time, psi)
    avg_P = np.nanmean(P, axis=-1)  # (shot_alt, time)
    safe_avg_P = np.maximum(avg_P, 1e-6)
    peaking = np.nanmax(P, axis=-1) / safe_avg_P

    q_loss = np.maximum(3.5 - q_star, 0) + np.maximum(3.0 - q_star, 0) + np.maximum(2.5 - q_star, 0)

    ne_avg_1e19 = np.nanmean(ne, axis=-1) * 10.0  # (shot_alt, time)
    te_avg_keV = np.nanmean(te, axis=-1)
    R0_2d = R0  # already (shot_alt, time)
    nu_star = np.array(
        jax.vmap(lambda ne_i, te_i, r0_i: calc_effective_collisionality(ne_i, te_i, r0_i, Z_EFF))(
            ne_avg_1e19.reshape(-1), te_avg_keV.reshape(-1), R0_2d.reshape(-1)
        )
    ).reshape(n_shot_alt, n_time)

    nu_loss = np.maximum(nu_star - 0.3, 0) + np.maximum(nu_star - 0.6, 0) + np.maximum(nu_star - 1.0, 0)

    gw_loss = np.log1p(np.exp(10.0 * (fGW - 1.3)))  # softplus

    total = 3.0 * peaking + q_loss + nu_loss + 0.5 * gw_loss

    def _mean(arr):
        return float(np.nanmean(arr))

    return {
        "total": _mean(total),
        "peaking": _mean(peaking),
        "q_loss": _mean(q_loss),
        "nu_loss": _mean(nu_loss),
        "gw_loss": _mean(gw_loss),
        "q_star": _mean(q_star),
        "nu_star": _mean(nu_star),
        "fGW": _mean(fGW),
    }


def run_trajopts(
    working_dir_base: Path | str,
    profopt_models_dir: Path | str,
):
    working_dir_base = Path(working_dir_base) / f"trajopt_{config.ref_shot}"
    for profile_module in PROFILE_MODULES:
        run_trajectory_optimization(
            trajopt_name=profile_module,
            working_dir_base=working_dir_base,
            profile_module_checkpoint_dir=_profile_module_to_checkpoint_dir(profile_module, profopt_models_dir),
            traj_times=TRAJ_TIMES,
            max_num_traj_times=MAX_NUM_TRAJ_TIMES,
            enable_parallelism=True,
        )


def compare_trajopts(
    working_dir_base: Path | str,
    profopt_models_dir: Path | str,
):
    """For each profile module and each optimized trajectory case, compute the mean loss
    and its components, then plot vs num_traj_times.

    Baseline (num_traj_times=0) uses the programmed waveform without any optimization.
    All comparisons are made on the same augmented dataset (100 perturbed trajectories
    around the reference shot) to ensure a fair comparison.
    """
    working_dir_base = Path(working_dir_base) / f"trajopt_{config.ref_shot}"
    shot_data_dir = Path(config.scratch_dir) / "predict_first" / "raw_data"
    ds_ref_path = shot_data_dir / f"{config.ref_shot}.nc"
    # Ensure the data file exists (fetches if needed), then load with the same
    # windowing/resampling used by output_optimized_trajectory so that the
    # saved trajectory waveforms (126 pts) match ds_ref's time dimension.
    get_traj_shot_data(config.ref_shot, ds_ref_path)
    ds_ref, _ = get_ds(
        ds_ref_path,
        selected_shots={config.ref_shot: {"start": 2.6, "end": 5.1}},
        fresh_profiles=False,
        debug=False,
    )

    # metrics[profile_module][(num_traj_times, optimize_density)] = dict of scalars
    all_metrics: dict[str, dict[tuple, dict]] = {}

    for profile_module in PROFILE_MODULES:
        logger.info(f"Evaluating {profile_module}")
        checkpoint_dir = _profile_module_to_checkpoint_dir(profile_module, profopt_models_dir)
        profile_predictor = restore_profile_predictor_from_checkpoint(checkpoint_dir)

        # Mirror the debug-suffix logic from run_trajectory_optimization
        trajopt_name = f"{profile_module}_debug" if config.debug else profile_module

        trajopt = TrajectoryOptimization(
            name=trajopt_name,
            working_dir_base=working_dir_base,
            profile_module_checkpoint_dir=checkpoint_dir,
            traj_times=TRAJ_TIMES,
            max_num_traj_times=MAX_NUM_TRAJ_TIMES,
        )

        module_metrics: dict[tuple, dict] = {}

        # Baseline: programmed trajectory, no optimization
        logger.info("  baseline")
        module_metrics[(0, True)] = _compute_loss_metrics(ds_ref, profile_predictor, None)
        module_metrics[(0, False)] = module_metrics[(0, True)]  # same baseline for both od flags

        for case in trajopt.cases:
            if not trajopt.output_path(case).exists():
                logger.warning(f"  Skipping {case} — output not found")
                continue
            logger.info(f"  {case}")
            output_dir = trajopt.output_path(case).parent
            module_metrics[(case.num_traj_times, case.optimize_density)] = _compute_loss_metrics(ds_ref, profile_predictor, output_dir)

        all_metrics[profile_module] = module_metrics

    _plot_comparison(all_metrics, working_dir_base)


def _plot_comparison(all_metrics: dict, plot_dir: Path | str):
    from matplotlib import gridspec
    from matplotlib.lines import Line2D

    loss_components = [
        ("total", "Total loss", "Loss"),
        ("peaking", "Pressure peaking factor", "P_max / P_avg"),
        ("q_star", "Edge safety factor q*", "q*"),
        ("fGW", "Greenwald fraction", "f_GW"),
        ("nu_star", "Eff. collisionality nu*", "nu*"),
    ]

    colors = plt.colormaps["tab10"].resampled(len(PROFILE_MODULES))
    module_colors = {m: colors(i) for i, m in enumerate(PROFILE_MODULES)}
    linestyles = {True: "-", False: "--"}

    n_metrics = len(loss_components)
    fig = plt.figure(figsize=(5 * n_metrics + 3, 5))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    gs = gridspec.GridSpec(
        1,
        n_metrics + 1,
        width_ratios=[5] * n_metrics + [2.5],
        figure=fig,
        wspace=0.35,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_metrics)]
    legend_ax = fig.add_subplot(gs[0, -1])

    for ax, (key, title, ylabel) in zip(axes, loss_components, strict=True):
        ax.set_facecolor(FACE_COLOR)
        ax.set_title(title, color=TEXT_COLOR, fontsize=11)
        ax.set_xlabel("num_traj_times\n(0 = programmed)", color=TEXT_COLOR)
        ax.set_ylabel(ylabel, color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(TEXT_COLOR)

        for profile_module, module_metrics in all_metrics.items():
            color = module_colors[profile_module]
            for optimize_density in [True, False]:
                xs, ys = [], []
                for (n, od), metrics in sorted(module_metrics.items()):
                    if od == optimize_density or n == 0:
                        if key in metrics:
                            xs.append(n)
                            ys.append(metrics[key])
                if xs:
                    ax.plot(
                        xs,
                        ys,
                        color=color,
                        linestyle=linestyles[optimize_density],
                        marker="o",
                        markersize=4,
                        linewidth=1.5,
                    )

    # Dedicated legend panel — no axes decoration
    legend_ax.set_facecolor(FACE_COLOR)
    for spine in legend_ax.spines.values():
        spine.set_visible(False)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])

    # Color handles: one per profile module (shortened to model_type.training_data)
    module_handles = [
        Line2D(
            [0],
            [0],
            color=module_colors[m],
            linewidth=2,
            label=".".join(m.split(".")[:2]),
        )
        for m in PROFILE_MODULES
        if m in all_metrics
    ]
    # Line style handles: solid = density optimized, dashed = density fixed
    style_handles = [
        Line2D([0], [0], color="white", linewidth=2, linestyle="-", label="density opt."),
        Line2D([0], [0], color="white", linewidth=2, linestyle="--", label="density fixed"),
    ]

    # Blank separator
    spacer = Line2D([], [], color="none", label="")

    legend_ax.legend(
        handles=[*module_handles, spacer, *style_handles],
        loc="center",
        fontsize=8,
        facecolor=BACKGROUND_COLOR,
        edgecolor=TEXT_COLOR,
        labelcolor=TEXT_COLOR,
        framealpha=0.9,
        title="Profile module",
        title_fontsize=9,
    )
    legend_ax.get_legend().get_title().set_color(TEXT_COLOR)

    fig.suptitle(
        f"Trajectory optimization comparison - ref shot {config.ref_shot}",
        color=TEXT_COLOR,
        fontsize=13,
    )
    fig.tight_layout()

    plot_dir = Path(plot_dir)
    out_path = plot_dir / "comparison.png"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    fire.Fire(
        {
            "run_trajopts": run_trajopts,
            "compare_trajopts": compare_trajopts,
        }
    )
