from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

BACKGROUND_COLOR = "#2F2F2F"
FACE_COLOR = "#1A1A1A"
TEXT_COLOR = "white"

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18


def profile_comparison(  # noqa: PLR0912
    profile_dir: Path | str,
    ds_targ: xr.Dataset,
    ds_pred_list: list[xr.Dataset] | None = None,
    ds_pred_labels: list[str] | None = None,
):
    """Plot individual profile signals from the dataset with multiple predictions.

    Args:
        profile_dir: Directory to save profile plots
        ds_targ: Target dataset with ne20_psi and Te_keV_psi profiles
        ds_pred_list: List of predicted datasets with ne and te profiles
        ds_pred_labels: Labels for each predicted dataset
    """

    profile_dir = Path(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Set default labels if not provided
    if ds_pred_list is not None and ds_pred_labels is None:
        ds_pred_labels = [f"Prediction {i + 1}" for i in range(len(ds_pred_list))]

    # Compute global y-limits across all shots for consistent axes
    all_ne_values = [ds_targ["ne20_psi"].values]
    all_te_values = [ds_targ["Te_keV_psi"].values]

    if ds_pred_list is not None:
        for ds_pred in ds_pred_list:
            all_ne_values.append(ds_pred["ne"].values)
            all_te_values.append(ds_pred["te"].values)

    ylim_ne = (0, float(np.nanmax([np.nanmax(vals) for vals in all_ne_values])) * 1.1)
    ylim_te = (0, float(np.nanmax([np.nanmax(vals) for vals in all_te_values])) * 1.1)

    for shot in ds_targ["shot"].data:
        shot_ds_targ = ds_targ.where(ds_targ["shot"] == shot, drop=True).squeeze()

        # Get predicted datasets for this shot
        shot_ds_pred_list = []
        if ds_pred_list is not None:
            for ds_pred in ds_pred_list:
                try:
                    shot_ds_pred = ds_pred.where(ds_pred["shot"] == shot, drop=True).squeeze()
                    shot_ds_pred_list.append(shot_ds_pred)
                except (KeyError, ValueError):
                    # Shot not found in prediction dataset or other error
                    shot_ds_pred_list.append(None)

        shot_dir = profile_dir / str(shot)
        shot_dir.mkdir(parents=True, exist_ok=True)

        # Get psi coordinates
        psi = shot_ds_targ["psi"].values

        for _, time in enumerate(shot_ds_targ["time"].values):
            time_ds_targ = shot_ds_targ.where(shot_ds_targ["time"] == time, drop=True).squeeze()

            ne_profile_targ = time_ds_targ["ne20_psi"].values
            te_profile_targ = time_ds_targ["Te_keV_psi"].values

            # Skip if profiles are all NaN
            if np.all(np.isnan(ne_profile_targ)) and np.all(np.isnan(te_profile_targ)):
                continue

            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            fig.patch.set_facecolor(BACKGROUND_COLOR)

            fig.suptitle(
                f"DIII-D Shot {shot} @ t={time:.3f}s",
                fontsize=TITLE_FONTSIZE,
                color=TEXT_COLOR,
            )

            # Define colors for multiple predictions
            pred_colors = ["blue", "green", "purple", "orange", "brown", "pink"]

            # Density profile
            ax_ne = axes[0]
            ax_ne.plot(psi, ne_profile_targ, label="ne20", color="cyan", linewidth=3)

            # Plot all predictions
            if ds_pred_list is not None:
                for i, (shot_ds_pred, label) in enumerate(zip(shot_ds_pred_list, ds_pred_labels, strict=True)):
                    if shot_ds_pred is not None:
                        try:
                            time_ds_pred = shot_ds_pred.where(shot_ds_pred["time"] == time, drop=True).squeeze()
                            ne_profile_pred = time_ds_pred["ne"].values
                            color = pred_colors[i % len(pred_colors)]
                            ax_ne.plot(
                                psi,
                                ne_profile_pred,
                                label=f"{label}",
                                color=color,
                                linewidth=2,
                                linestyle="--",
                            )
                        except (KeyError, ValueError):
                            # Time not found in prediction or other error
                            continue

            ax_ne.set_ylabel(r"$n_e$ [$10^{20}$ m$^{-3}$]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
            ax_ne.set_ylim(ylim_ne)
            ax_ne.legend(
                fontsize=LEGEND_FONTSIZE,
                facecolor=BACKGROUND_COLOR,
                edgecolor=BACKGROUND_COLOR,
                loc="upper right",
            )

            # Temperature profile
            ax_te = axes[1]
            ax_te.plot(psi, te_profile_targ, label="Te", color="red", linewidth=3)

            # Plot all predictions
            if ds_pred_list is not None:
                for i, (shot_ds_pred, label) in enumerate(zip(shot_ds_pred_list, ds_pred_labels, strict=True)):
                    if shot_ds_pred is not None:
                        try:
                            time_ds_pred = shot_ds_pred.where(shot_ds_pred["time"] == time, drop=True).squeeze()
                            te_profile_pred = time_ds_pred["te"].values
                            color = pred_colors[i % len(pred_colors)]
                            ax_te.plot(
                                psi,
                                te_profile_pred,
                                label=f"{label}",
                                color=color,
                                linewidth=2,
                                linestyle="--",
                            )
                        except (KeyError, ValueError):
                            # Time not found in prediction or other error
                            continue

            ax_te.set_ylabel(r"$T_e$ [keV]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
            ax_te.set_xlabel(r"$\psi_n$", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
            ax_te.set_ylim(ylim_te)
            ax_te.legend(
                fontsize=LEGEND_FONTSIZE,
                facecolor=BACKGROUND_COLOR,
                edgecolor=BACKGROUND_COLOR,
                loc="upper right",
            )

            for ax in axes:
                ax.set_facecolor(FACE_COLOR)
                ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
                ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
                ax.set_xlim(0, 1.2)
                for spine in ax.spines.values():
                    spine.set_color(TEXT_COLOR)
                try:
                    for text in ax.get_legend().get_texts():
                        text.set_color(TEXT_COLOR)
                except AttributeError:
                    pass

            fig.tight_layout()
            fig.savefig(shot_dir / f"t{time:.3f}.png")
            plt.close(fig)


def trajectory_performance_comparison(
    ds_perf_list: list[xr.DataArray],
    ds_perf_labels: list[str],
    save_dir: Path | str,
    title: str,
):
    """Compare performance of different trajectories on the same plot, in a both per-shot and per-timeslice manner.
    Expects each DataArray in ds_perf_list to have dimensions (sample, time_idx) and coords (shot, time, shot_alt), where sample is the dimension corresponding to different trajectories for the same shot and time (e.g. from different permutations or from the optimization trajectory). The "time" coordinate should be the actual time value in seconds, which will be used for the x-axis in the timeslice performance plot.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timeslice_performance_list = []
    shot_performance_list = []
    for ds_perf in ds_perf_list:
        timeslice_perf = ds_perf.data[~np.isnan(ds_perf.data)]
        timeslice_performance_list.append(timeslice_perf)

        shot_perf = ds_perf.mean(dim="time_idx", skipna=True).data
        shot_performance_list.append(shot_perf)

    # Boxplots
    for perf_list, perf_label in zip(
        [timeslice_performance_list, shot_performance_list],
        ["Timeslice", "Per-Shot"],
        strict=True,
    ):
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        box_dict = ax.boxplot(perf_list, labels=ds_perf_labels)
        ax.set_title(f"{title} - {perf_label}", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)
        ax.set_ylabel("Peaking Factor", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        ax.set_facecolor(FACE_COLOR)
        ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
        # Set the boxes to be blue and the whiskers to be red
        for box in box_dict["boxes"]:
            box.set(color="cyan", linewidth=2)
        for whisker in box_dict["whiskers"]:
            whisker.set(color="red", linewidth=2)
        for cap in box_dict["caps"]:
            cap.set(color="lime", linewidth=2)
        for median in box_dict["medians"]:
            median.set(color="orange", linewidth=2)
        for flier in box_dict["fliers"]:
            flier.set(marker="o", markeredgecolor="white", markersize=4)

        fig.tight_layout()
        fig.savefig(save_dir / f"performance_comparison_{perf_label.lower()}.png")
        plt.close(fig)


def trajectory_shapes_comparison(  # noqa: PLR0915
    trajectory_shapes: list[dict],
    trajectory_labels: list[str],
    orig_traj: xr.Dataset,
    save_dir: Path | str,
    input_ranges: dict[str, tuple[float, float]] | None = None,
):
    """For each trajectory shape variable, plot in a 3-column layout:
    - Col 0: all original shots (grey) + optimized trajectories
    - Col 1: 201XXX-series shots (faint, colored by shot) + optimized trajectories
    - Col 2: 199XXX-series shots (faint, colored by shot) + optimized trajectories
    Allowed min/max bounds from input_ranges are shown as dashed lines when provided.

    Args:
        trajectory_shapes: list of dicts containing trajectory shapes and their times
        trajectory_labels: list of labels for each trajectory
        orig_traj: dataset containing original shots
        save_dir: directory to save the figure
        input_ranges: optional dict mapping each trajectory var to (min, max) bounds

    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    trajectory_vars = [var for var in trajectory_shapes[0].keys() if var != "shape_times"]
    n_vars = len(trajectory_vars)

    # 3 columns: [all-grey, 201xxx series, 199xxx series]
    col_titles = ["All shots", "201XXX shots", "199XXX shots"]
    fig, axes_grid = plt.subplots(n_vars, 3, figsize=(18, 4 * n_vars), sharex=False)
    # Normalise to 2-D array regardless of n_vars
    if n_vars == 1:
        axes_grid = axes_grid.reshape(1, 3)

    fig.patch.set_facecolor(BACKGROUND_COLOR)

    pred_colors = ["cyan", "lime", "orange", "magenta", "yellow", "deepskyblue"]
    orig_palette = plt.cm.tab20.colors  # 20 distinct colors for original shots

    shot_alts = orig_traj["shot_alt"].values

    def _shot_number(shot_alt: str) -> str:
        return str(shot_alt).split("_")[0]

    unique_shot_numbers = list(dict.fromkeys(_shot_number(sa) for sa in shot_alts))
    shot_color_map = {sn: orig_palette[i % len(orig_palette)] for i, sn in enumerate(unique_shot_numbers)}

    # Classify shot_alts into series
    def _series(shot_alt: str) -> str:
        sn = _shot_number(shot_alt)
        if sn.startswith("201"):
            return "201"
        if sn.startswith("199"):
            return "199"
        return "other"

    def _style_ax(ax: plt.Axes) -> None:
        ax.set_facecolor(FACE_COLOR)
        ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(TEXT_COLOR)

    def _plot_orig_shots(ax: plt.Axes, var: str, series_filter: str | None, colored: bool) -> None:
        """Plot original shot trajectories on ax.  If series_filter is set, only
        shots from that series are drawn.  colored=True uses per-shot colors;
        colored=False uses grey."""
        if var not in orig_traj:
            return
        labeled: set[str] = set()
        for shot_alt in shot_alts:
            sn = _shot_number(shot_alt)
            if series_filter is not None and not sn.startswith(series_filter):
                continue
            ds_shot = orig_traj.sel(shot_alt=shot_alt)
            t = ds_shot["time"].values
            v = ds_shot[var].values
            valid = ~np.isnan(t) & ~np.isnan(v)
            if not valid.any():
                continue
            color = shot_color_map[sn] if colored else "gray"
            alpha = 1.0 if colored else 0.2
            lw = 1.5 if colored else 1
            label = sn if (colored and sn not in labeled) else None
            ax.plot(
                t[valid],
                v[valid],
                color=color,
                alpha=alpha,
                linewidth=lw,
                zorder=1,
                label=label,
            )
            labeled.add(sn)

    def _plot_optimized(ax: plt.Axes, var: str) -> None:
        for i, (traj, label) in enumerate(zip(trajectory_shapes, trajectory_labels, strict=True)):
            shape_times = np.array(traj["shape_times"])
            var_vals = np.array(traj[var])
            color = pred_colors[i % len(pred_colors)]
            ax.plot(
                shape_times,
                var_vals,
                color=color,
                linewidth=2,
                marker="o",
                markersize=8,
                label=label,
                zorder=2,
            )

    def _plot_bounds(ax: plt.Axes, var: str) -> None:
        if input_ranges is None or var not in input_ranges:
            return
        lo, hi = input_ranges[var]
        ax.axhline(
            lo,
            color="red",
            linewidth=2.0,
            linestyle="--",
            alpha=0.9,
            zorder=4,
            label="Bounds",
        )
        ax.axhline(hi, color="red", linewidth=2.0, linestyle="--", alpha=0.9, zorder=4)

    for row, var in enumerate(trajectory_vars):
        ax_all, ax_201, ax_199 = axes_grid[row]

        for ax in (ax_all, ax_201, ax_199):
            _style_ax(ax)

        # Column 0: grey originals + optimized
        _plot_orig_shots(ax_all, var, series_filter=None, colored=False)
        _plot_optimized(ax_all, var)
        _plot_bounds(ax_all, var)

        # Column 1: 201XXX shots (colored) only
        _plot_orig_shots(ax_201, var, series_filter="201", colored=True)
        _plot_bounds(ax_201, var)

        # Column 2: 199XXX shots (colored) only
        _plot_orig_shots(ax_199, var, series_filter="199", colored=True)
        _plot_bounds(ax_199, var)

        ax_all.set_ylabel(var, fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)

        # Only label x-axis on the bottom row
        if row == n_vars - 1:
            for ax in (ax_all, ax_201, ax_199):
                ax.set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)

    # Sync y-axis limits across all three columns for each row
    for row in range(n_vars):
        row_axes = axes_grid[row]
        all_ylims = [ax.get_ylim() for ax in row_axes]
        combined_lo = min(lo for lo, _ in all_ylims)
        combined_hi = max(hi for _, hi in all_ylims)
        for ax in row_axes:
            ax.set_ylim(combined_lo, combined_hi)

    # Column titles on the top row
    for ax, title in zip(axes_grid[0], col_titles, strict=True):
        ax.set_title(title, fontsize=TITLE_FONTSIZE, color=TEXT_COLOR, pad=6)

    # Single figure-level legend aggregated from all axes (deduped)
    seen_labels: set[str] = set()
    legend_handles: list = []
    legend_label_list: list[str] = []
    # Gather optimized + bounds labels first (from col 0), then shot labels (cols 1+2)
    for col in range(3):
        for row in range(n_vars):
            for handle, label in zip(*axes_grid[row, col].get_legend_handles_labels(), strict=True):
                if label not in seen_labels:
                    seen_labels.add(label)
                    legend_handles.append(handle)
                    legend_label_list.append(label)

    legend = fig.legend(
        legend_handles,
        legend_label_list,
        fontsize=LEGEND_FONTSIZE - 4,
        facecolor=BACKGROUND_COLOR,
        edgecolor=BACKGROUND_COLOR,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        bbox_transform=fig.transFigure,
        ncols=min(len(legend_handles), 8),
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(
        save_dir / "trajectory_shapes_comparison.png",
        bbox_inches="tight",
    )
    plt.close(fig)
