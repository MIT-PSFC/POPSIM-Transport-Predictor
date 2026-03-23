import os

import fire
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import xarray as xr
from loguru import logger

from transport_study.datasets.d3d.d3d_dataset import D3DDataWorkflow

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transport_study.config import config
from transport_study.datasets.d3d.d3d_dataset import INNER_WALL
from transport_study.modules.profile_predictor.module import Inputs
from transport_study.modules.profile_trajectory.data import (
    add_gapin_prog,
    correct_B0_prog,
)
from transport_study.profile_transfer.restore_predictor import (
    restore_profile_predictor_from_checkpoint,
)
from transport_study.trajectory_optimization.setup_data import (
    FEEDBACK_CONTROL_SHOTS,
    make_augmented_dataset,
)

BACKGROUND_COLOR = "#2F2F2F"
FACE_COLOR = "#1A1A1A"
TEXT_COLOR = "white"
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 13

# Bright colors ordered warm→cool (core psi=0 → edge psi=1), all visible on dark backgrounds
BRIGHT_COLORS = [
    "#FF2400",  # red       (core)
    "#FF8C00",  # orange
    "#FFD700",  # gold
    "#00FF80",  # spring green
    "#00FFFF",  # cyan
    "#00BFFF",  # sky blue  (edge)
]


def run_preshot_prediction(  # noqa: PLR0915
    ref_shot: int,
    targ_shot: int | None,
    profile_predictor_checkpoint_dir: str,
    optimized_trajectory_checkpoint_dir: str | None = None,
    scratch_dir: str | None = config.scratch_dir,
):
    """Run selected profile predictor to get distribution of profiles over time

    0. If an optimized trajectory is provided, go get that and overwrite programmed trajectory with its values
    1. Make augmented dataset where input parameters are randomly perturbed within their typical error ranges
    2. Create time-independent dataloader for profile predictor
    3. Restore profile predictor from checkpoint
    4. Run profile predictor on dataloader
    5. Save predicted profiles as a dataset for later use
    6 + nice plots and gifs with error bars over time, etc.

    Args:
        ref_shot: int
            The shot to use as the reference trajectory to perturb around
        targ_shot: int
            The shot that we are predicting for (should be programmed with the anticipated optimized trajectory)
        profile_predictor_checkpoint_dir: str
            The checkpoint directory to restore the profile predictor from
        optimized_trajectory_checkpoint_dir: str | None
            If not None, the checkpoint directory to restore the optimized trajectory from. If None, uses the programmed trajectory from the dataset as-is.
        scratch_dir: str | None
            The directory to use for temporary files during prediction. If None, uses the value from config.scratch_dir.
    """

    def _setup_directories(scratch_dir):
        working_dir = os.path.join(scratch_dir, "predict_first")
        shot_data_dir = os.path.join(working_dir, "shot_data")
        result_dir = os.path.join(working_dir, f"shot_{targ_shot}")
        os.makedirs(shot_data_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        return working_dir, shot_data_dir, result_dir

    _working_dir, shot_data_dir, result_dir = _setup_directories(scratch_dir)

    # Get trajectories and profiles if they are not already saved in the shot_data_dir
    ds_ref_path = os.path.join(shot_data_dir, f"{ref_shot}.nc")
    ds_ref = _get_shot_data(ref_shot, ds_ref_path)
    if targ_shot is not None:
        ds_target_path = os.path.join(shot_data_dir, f"{targ_shot}.nc")
        ds_targ = _get_shot_data(targ_shot, ds_target_path)

    # Step 0: Overwrite programmed trajectory if an optimized trajectory is provided
    if optimized_trajectory_checkpoint_dir is not None:
        traj_path = os.path.join(
            optimized_trajectory_checkpoint_dir, "optimized_trajectory.nc"
        )
        logger.info(f"Loading optimized trajectory from {traj_path}")
        ds_traj = xr.open_dataset(traj_path)
        # Map PCS signal names → dataset variable names and unit scale factors
        traj_signal_map = {
            "iptipp": ("Ip_MA_prog", 1e-6),  # A → MA
            "bttbt": ("B0_prog", 1.0),
            "bmtpwrtar": ("betan_prog", 1.0),
            "dstdenp": ("ne20_edge_prog", 0.1),  # 1e19 m^-3 → 1e20 m^-3
            "idtrp": ("R0_prog", 1.0),
            "idtrxbot": ("rxbot_prog", 1.0),
            "idtzxbot": ("zxbot_prog", 1.0),
            "idtrxtop": ("rxtop_prog", 1.0),
            "idtzxtop": ("zxtop_prog", 1.0),
        }
        updated_vars = {}
        for traj_var, (ds_var, scale) in traj_signal_map.items():
            if traj_var in ds_traj and ds_var in ds_ref:
                orig = ds_ref[ds_var].load()
                new_vals = orig.copy()
                new_vals.loc[{"shot": ref_shot}] = ds_traj[traj_var].values * scale
                updated_vars[ds_var] = new_vals
        if updated_vars:
            ds_ref = ds_ref.assign(updated_vars)

    ds_ref = correct_B0_prog(ds_ref)
    ds_ref = add_gapin_prog(ds_ref)
    if targ_shot is not None:
        ds_targ = correct_B0_prog(ds_targ)
        ds_targ = add_gapin_prog(ds_targ)

    # Step 1: Make augmented dataset with perturbed inputs
    logger.info("Building augmented dataset with perturbed inputs")
    ds_aug = make_augmented_dataset(ds_ref, shots_times=FEEDBACK_CONTROL_SHOTS)

    # Ensure no NaNs
    for var in ds_aug.data_vars:
        if ds_aug[var].isnull().any():
            logger.critical(f"Variable {var} contains NaNs after augmentation!")

    # Step 2: Derive shape variables from programmed signals (matches PCSInputMapper logic)
    R0_prog = ds_aug["R0_prog"].values  # (shot_alt, time_idx)
    gapin_prog = ds_aug["gapin_prog"].values
    rxbot_prog = ds_aug["rxbot_prog"].values
    zxbot_prog = ds_aug["zxbot_prog"].values
    rxtop_prog = ds_aug["rxtop_prog"].values
    zxtop_prog = ds_aug["zxtop_prog"].values

    a_minor = R0_prog - gapin_prog - INNER_WALL
    kappa = np.abs(zxtop_prog - zxbot_prog) / (a_minor * 2)
    delta_bot = (R0_prog - rxbot_prog) / a_minor
    delta_top = (R0_prog - rxtop_prog) / a_minor

    # Step 3: Restore profile predictor from checkpoint
    profile_predictor = restore_profile_predictor_from_checkpoint(
        profile_predictor_checkpoint_dir
    )

    # Build batched Inputs for vmap
    n_shot_alt, n_time = R0_prog.shape
    psi_grid = np.array(profile_predictor.psigrid)
    n_psi = len(psi_grid)

    Ip_flat = ds_aug["Ip_MA_prog"].values.reshape(-1)
    valid_mask = ~np.isnan(Ip_flat)

    psi_tiled = jnp.tile(jnp.array(psi_grid), (n_shot_alt * n_time, 1))

    inputs_batched = Inputs(
        Ip=ds_aug["Ip_MA_prog"].values.reshape(-1),
        B0=ds_aug["B0_prog"].values.reshape(-1),
        betan=ds_aug["betan_prog"].values.reshape(-1),
        ne20=ds_aug["ne20_edge_prog"].values.reshape(-1),
        R0=R0_prog.reshape(-1),
        a_minor=a_minor.reshape(-1),
        kappa=kappa.reshape(-1),
        delta_top=delta_top.reshape(-1),
        delta_bot=delta_bot.reshape(-1),
        psi=psi_tiled,
    )

    # Step 4: Run profile predictor on all samples via vmap
    logger.info(
        f"Running profile predictor on {n_shot_alt * n_time} samples "
        f"({n_shot_alt} shot_alts x {n_time} time steps)"
    )

    def _predict(inputs: Inputs):
        outputs = profile_predictor(inputs)
        return outputs.ne.data, outputs.te.data

    ne_flat, te_flat = jax.vmap(_predict)(inputs_batched)

    # Restore NaN for padded time steps
    valid_jnp = jnp.array(valid_mask)[:, None]
    ne_flat = jnp.where(valid_jnp, ne_flat, jnp.nan)
    te_flat = jnp.where(valid_jnp, te_flat, jnp.nan)

    ne_arr = np.array(ne_flat).reshape(n_shot_alt, n_time, n_psi)
    te_arr = np.array(te_flat).reshape(n_shot_alt, n_time, n_psi)

    # Step 5: Build output dataset
    ds_pred = xr.Dataset(
        {
            "ne": (["shot_alt", "time_idx", "psi_n"], ne_arr),
            "te": (["shot_alt", "time_idx", "psi_n"], te_arr),
        },
        coords={
            "shot_alt": ds_aug["shot_alt"].values,
            "time_idx": ds_aug["time_idx"].values,
            "psi_n": psi_grid,
        },
    )
    if "time" in ds_aug:
        ds_pred["time"] = ds_aug["time"]

    pred_path = os.path.join(result_dir, "predicted_profiles.nc")
    ds_pred.to_netcdf(pred_path)
    logger.info(f"Saved predicted profiles to {pred_path}")

    # Step 6: Plots with error bars over time
    _plot_predictor_inputs(ds_aug, a_minor, kappa, delta_top, delta_bot, result_dir)
    _plot_preshot_predictions(ds_pred, result_dir)


def _plot_predictor_inputs(
    ds_aug: xr.Dataset,
    a_minor: np.ndarray,
    kappa: np.ndarray,
    delta_top: np.ndarray,
    delta_bot: np.ndarray,
    output_dir: str,
):
    """Plot mean ± 1 std of all 9 profile-predictor input signals across augmented perturbations.

    Saves one figure per base shot: a 3x3 grid showing each input signal over time.
    """
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    shot_alt_vals = ds_aug["shot_alt"].values
    base_shots = list(dict.fromkeys(str(sa).split("_")[0] for sa in shot_alt_vals))

    # Bundle all inputs into a single dict of (shot_alt, time_idx) arrays
    input_signals = {
        "Ip_MA_prog [MA]": ds_aug["Ip_MA_prog"].values,
        "B0_prog [T]": ds_aug["B0_prog"].values,
        "betan_prog": ds_aug["betan_prog"].values,
        "ne20_edge_prog [1e20/m³]": ds_aug["ne20_edge_prog"].values,
        "R0_prog [m]": ds_aug["R0_prog"].values,
        "a_minor [m]": a_minor,
        "kappa": kappa,
        "delta_top": delta_top,
        "delta_bot": delta_bot,
    }

    n_signals = len(input_signals)
    n_cols = 3
    n_rows = (n_signals + n_cols - 1) // n_cols

    for base_shot in base_shots:
        mask = np.array([str(sa).startswith(f"{base_shot}_") for sa in shot_alt_vals])

        # Time values (same for all permutations of the same base shot)
        if "time" in ds_aug:
            time_2d = ds_aug["time"].values  # (shot_alt, time_idx)
            time_vals = time_2d[mask][0]
        else:
            time_vals = ds_aug["time_idx"].values.astype(float)
        valid_t = ~np.isnan(time_vals)
        t = time_vals[valid_t]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        fig.suptitle(
            f"Predictor Inputs — Shot {base_shot} ({mask.sum()} perturbations)",
            fontsize=TITLE_FONTSIZE,
            color=TEXT_COLOR,
        )
        axes_flat = axes.reshape(-1)

        for ax, (label, arr) in zip(axes_flat, input_signals.items(), strict=False):
            perturbed = arr[mask]  # (n_perturb, time_idx)
            mean_t = np.nanmean(perturbed, axis=0)[valid_t]
            std_t = np.nanstd(perturbed, axis=0)[valid_t]

            ax.set_facecolor(FACE_COLOR)
            ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
            for spine in ax.spines.values():
                spine.set_color(TEXT_COLOR)

            ax.plot(t, mean_t, color="cyan", linewidth=2, label="mean")
            ax.fill_between(
                t,
                mean_t - std_t,
                mean_t + std_t,
                color="cyan",
                alpha=0.3,
                label="±1 std",
            )

            ax.set_title(label, fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
            ax.set_xlabel("Time [s]", fontsize=TICK_FONTSIZE, color=TEXT_COLOR)
            legend = ax.legend(
                fontsize=LEGEND_FONTSIZE - 2,
                facecolor=BACKGROUND_COLOR,
                edgecolor=BACKGROUND_COLOR,
            )
            for text in legend.get_texts():
                text.set_color(TEXT_COLOR)

        # Hide any unused subplot panels
        for ax in axes_flat[n_signals:]:
            ax.set_visible(False)

        fig.tight_layout()
        fig.savefig(
            os.path.join(plot_dir, f"shot_{base_shot}_predictor_inputs.png"),
            dpi=150,
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)


def _plot_preshot_predictions(ds_pred: xr.Dataset, output_dir: str):  # noqa: PLR0915
    """Plot predicted profiles with mean ± 1 std across perturbations.

    For each base shot, produces:
    - Time traces of ne and te at select psi values (mean ± std shading)
    - Profile snapshots at select times (mean ± std shading)
    """
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    shot_alt_vals = ds_pred["shot_alt"].values
    base_shots = list(dict.fromkeys(str(sa).split("_")[0] for sa in shot_alt_vals))
    psi_plot_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    psi_colors = BRIGHT_COLORS[: len(psi_plot_vals)]

    for base_shot in base_shots:
        mask = np.array([str(sa).startswith(f"{base_shot}_") for sa in shot_alt_vals])
        ds_shot = ds_pred.isel(shot_alt=mask)

        ne_mean = ds_shot["ne"].mean(dim="shot_alt")  # (time_idx, psi_n)
        ne_std = ds_shot["ne"].std(dim="shot_alt")
        te_mean = ds_shot["te"].mean(dim="shot_alt")
        te_std = ds_shot["te"].std(dim="shot_alt")

        # Time array: use first shot_alt's time (all permutations of same base shot have same time)
        if "time" in ds_pred:
            time_2d = ds_pred["time"].values  # (shot_alt, time_idx)
            time_vals = time_2d[mask][0]  # (time_idx,)
        else:
            time_vals = ds_pred["time_idx"].values.astype(float)

        valid_t = ~np.isnan(time_vals)

        # --- Time trace plot ---
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        fig.suptitle(
            f"Preshot Prediction — Shot {base_shot}",
            fontsize=TITLE_FONTSIZE,
            color=TEXT_COLOR,
        )

        for ax, (mean_da, std_da, ylabel) in zip(
            axes,
            [
                (ne_mean, ne_std, r"$n_e$ [$10^{20}$ m$^{-3}$]"),
                (te_mean, te_std, r"$T_e$ [keV]"),
            ],
            strict=True,
        ):
            ax.set_facecolor(FACE_COLOR)
            ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
            for spine in ax.spines.values():
                spine.set_color(TEXT_COLOR)

            for psi_val, color in zip(psi_plot_vals, psi_colors, strict=True):
                mean_ts = mean_da.sel(psi_n=psi_val, method="nearest").values
                std_ts = std_da.sel(psi_n=psi_val, method="nearest").values
                t = time_vals[valid_t]
                m = mean_ts[valid_t]
                s = std_ts[valid_t]
                ax.plot(t, m, color=color, linewidth=2, label=f"psi={psi_val:.1f}")
                ax.fill_between(t, m - s, m + s, color=color, alpha=0.25)

            ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
            legend = ax.legend(
                fontsize=LEGEND_FONTSIZE,
                facecolor=BACKGROUND_COLOR,
                edgecolor=BACKGROUND_COLOR,
            )
            for text in legend.get_texts():
                text.set_color(TEXT_COLOR)

        axes[-1].set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        fig.tight_layout()
        fig.savefig(
            os.path.join(plot_dir, f"shot_{base_shot}_time_traces.png"),
            dpi=150,
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)

        # --- Profile snapshot plot: pick up to 5 evenly-spaced valid times ---
        valid_idx = np.where(valid_t)[0]
        if len(valid_idx) >= 2:
            snap_idx = valid_idx[
                np.round(
                    np.linspace(0, len(valid_idx) - 1, min(5, len(valid_idx)))
                ).astype(int)
            ]
            snap_colors = BRIGHT_COLORS[: len(snap_idx)]
            psi_grid_vals = ds_pred["psi_n"].values

            fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
            fig2.patch.set_facecolor(BACKGROUND_COLOR)
            fig2.suptitle(
                f"Preshot Profile Snapshots — Shot {base_shot}",
                fontsize=TITLE_FONTSIZE,
                color=TEXT_COLOR,
            )

            for ax2, (mean_da, std_da, ylabel) in zip(
                axes2,
                [
                    (ne_mean, ne_std, r"$n_e$ [$10^{20}$ m$^{-3}$]"),
                    (te_mean, te_std, r"$T_e$ [keV]"),
                ],
                strict=True,
            ):
                ax2.set_facecolor(FACE_COLOR)
                ax2.grid(True, color="gray", linestyle="--", linewidth=0.5)
                ax2.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
                for spine in ax2.spines.values():
                    spine.set_color(TEXT_COLOR)

                for tidx, color in zip(snap_idx, snap_colors, strict=True):
                    t_label = f"t={time_vals[tidx]:.2f}s"
                    m = mean_da.isel(time_idx=tidx).values
                    s = std_da.isel(time_idx=tidx).values
                    ax2.plot(psi_grid_vals, m, color=color, linewidth=2, label=t_label)
                    ax2.fill_between(
                        psi_grid_vals, m - s, m + s, color=color, alpha=0.25
                    )

                ax2.set_xlabel(r"$\psi_n$", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
                ax2.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
                legend2 = ax2.legend(
                    fontsize=LEGEND_FONTSIZE,
                    facecolor=BACKGROUND_COLOR,
                    edgecolor=BACKGROUND_COLOR,
                )
                for text in legend2.get_texts():
                    text.set_color(TEXT_COLOR)

            fig2.tight_layout()
            fig2.savefig(
                os.path.join(plot_dir, f"shot_{base_shot}_profile_snapshots.png"),
                dpi=150,
                facecolor=fig2.get_facecolor(),
            )
            plt.close(fig2)

    logger.info(f"Saved plots to {plot_dir}")


def _get_shot_data(shot: int, ds_path: str) -> xr.Dataset:
    if not os.path.exists(ds_path):
        logger.info(
            f"Extracting shot {shot} data from D3D servers (MUST BE RUN ON OMEGA)"
        )
        data_assembly_dir = os.path.dirname(ds_path)
        # Make a temporary text file with just this shot in it
        shotlist_path = os.path.join(data_assembly_dir, f"{shot}_shotlist.txt")
        with open(shotlist_path, "w") as f:
            f.write(f"{shot}\n")

        workflow = D3DDataWorkflow(
            ds_name="predict_first_input",
            shotlist_file=shotlist_path,
            data_assembly_dir=data_assembly_dir,
            max_num_shots=1,
            mode="raw",
            use_ida=False,  # We're comparing against ZIPFITs, since those are what'll be available on the run day
        )
        workflow.make_raw_data_files()

        # Clean up shotlist file
        os.remove(shotlist_path)

    logger.info(f"Loading shot {shot} data from {ds_path}")
    ds = xr.open_dataset(ds_path)
    return ds


if __name__ == "__main__":
    fire.Fire(
        {
            "run_preshot_prediction": run_preshot_prediction,
        }
    )
