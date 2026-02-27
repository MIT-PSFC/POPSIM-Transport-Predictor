import os

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


def profile_comparison(  # noqa: PLR0915, PLR0912
    profile_dir: str,
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

    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)

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
                    shot_ds_pred = ds_pred.where(
                        ds_pred["shot"] == shot, drop=True
                    ).squeeze()
                    shot_ds_pred_list.append(shot_ds_pred)
                except (KeyError, ValueError):
                    # Shot not found in prediction dataset or other error
                    shot_ds_pred_list.append(None)

        shot_dir = os.path.join(profile_dir, str(shot))
        if not os.path.exists(shot_dir):
            os.makedirs(shot_dir)

        # Get psi coordinates
        psi = shot_ds_targ["psi"].values

        for _, time in enumerate(shot_ds_targ["time"].values):
            time_ds_targ = shot_ds_targ.where(
                shot_ds_targ["time"] == time, drop=True
            ).squeeze()

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
                for i, (shot_ds_pred, label) in enumerate(
                    zip(shot_ds_pred_list, ds_pred_labels, strict=True)
                ):
                    if shot_ds_pred is not None:
                        try:
                            time_ds_pred = shot_ds_pred.where(
                                shot_ds_pred["time"] == time, drop=True
                            ).squeeze()
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

            ax_ne.set_ylabel(
                r"$n_e$ [$10^{20}$ m$^{-3}$]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR
            )
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
                for i, (shot_ds_pred, label) in enumerate(
                    zip(shot_ds_pred_list, ds_pred_labels, strict=True)
                ):
                    if shot_ds_pred is not None:
                        try:
                            time_ds_pred = shot_ds_pred.where(
                                shot_ds_pred["time"] == time, drop=True
                            ).squeeze()
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
            fig.savefig(os.path.join(shot_dir, f"t{time:.3f}.png"))
            plt.close(fig)


def trajectory_performance_comparison(
    ds_perf_list: list[xr.DataArray],
    ds_perf_labels: list[str],
    save_dir: str,
    title: str,
):
    """Compare performance of different trajectories on the same plot, in a both per-shot and per-timeslice manner.
    Expects each DataArray in ds_perf_list to have dimensions (sample, time_idx) and coords (shot, time, shot_alt), where sample is the dimension corresponding to different trajectories for the same shot and time (e.g. from different permutations or from the optimization trajectory). The "time" coordinate should be the actual time value in seconds, which will be used for the x-axis in the timeslice performance plot.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
        ax.set_title(
            f"{title} - {perf_label}", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR
        )
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
        fig.savefig(
            os.path.join(save_dir, f"performance_comparison_{perf_label.lower()}.png")
        )
        plt.close(fig)
