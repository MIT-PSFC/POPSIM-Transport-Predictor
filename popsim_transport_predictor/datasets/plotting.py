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


def ds_time_plot(  # noqa: PLR0915 PLR0912
    ds_path: str,
    fig_dir: str,
    num_shots: int | None = 9999,
    title: str = "Dataset Time Traces",
):
    """Plot time traces of signals from the dataset"""
    if ds_path.endswith(".zarr"):
        ds = xr.open_zarr(ds_path)
    else:
        ds = xr.open_dataset(ds_path)

    os.makedirs(fig_dir, exist_ok=True)

    # Compute global y-limits across all shots for consistent axes
    ylim_ip = (0, float(np.nanmax(np.abs(ds["Ip_MA"].values))) * 1.1)
    ylim_wtot = (0, float(np.nanmax(ds["Wtot_MJ"].values)) * 1.1)

    power_signals = [
        "P_oh_MW",
        "P_rad_MW",
        "P_NBI_MW",
        "P_ECRH_MW",
        "P_LH_MW",
        "P_ICRF_MW",
    ]
    power_max = max(float(np.nanmax(ds[sig].values)) for sig in power_signals)
    if "LH_transition_threshold_MW" in ds:
        lh_thresh_max = float(np.nanmax(ds["LH_transition_threshold_MW"].values / 1e6))
    else:
        lh_thresh_max = 0
    ylim_power = (0, max(power_max, lh_thresh_max) * 1.1)

    ylim_ne = (0, float(np.nanmax(ds["ne20_line_avg"].values)) * 1.1)

    # B0 y-limits for density plot right axis
    if "B0" in ds:
        ylim_b0 = (0, float(np.nanmax(ds["B0"].values)) * 1.1)
    else:
        ylim_b0 = (0, 5)  # Default range

    shape_signals = ["a_minor", "kappa", "delta_top", "delta_bottom"]
    shape_min = min(float(np.nanmin(ds[sig].values)) for sig in shape_signals)
    shape_max = max(float(np.nanmax(ds[sig].values)) for sig in shape_signals)
    ylim_shape = (
        shape_min * 0.9 if shape_min > 0 else shape_min * 1.1,
        shape_max * 1.1,
    )

    # R0 y-limits for shaping plot right axis
    if "R0" in ds:
        ylim_r0 = (0, float(np.nanmax(ds["R0"].values)) * 1.1)
    else:
        ylim_r0 = (0, 3)  # Default range

    for shot in ds["shot"].data[:num_shots]:
        shot_ds = ds.sel(shot=shot)

        fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
        fig.patch.set_facecolor(BACKGROUND_COLOR)

        fig.suptitle(f"{title} - {shot}", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)

        # Ip and Wtot
        ax_ip = axes[0]
        # Put Ip on the left y axis and Wtot on the right y axis
        ax_ip.plot(shot_ds["time"], shot_ds["Ip_MA"], label="Ip [MA]", color="cyan")
        ax_ip.set_ylabel("Ip [MA]", fontsize=LABEL_FONTSIZE, color="cyan")
        ax_ip.set_ylim(ylim_ip)
        ax_wtot = ax_ip.twinx()
        if "Wtot_MJ" in shot_ds:
            ax_wtot.plot(
                shot_ds["time"], shot_ds["Wtot_MJ"], label="Wtot [MJ]", color="red"
            )
        if "Wmhd_MJ" in shot_ds:
            ax_wtot.plot(
                shot_ds["time"],
                shot_ds["Wmhd_MJ"],
                label="Wmhd [MJ]",
                color="orange",
                linestyle="--",
            )
        ax_wtot.set_ylabel("Stored Energy [MJ]", fontsize=LABEL_FONTSIZE, color="red")
        ax_wtot.set_ylim(ylim_wtot)
        ax_wtot.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)

        # powers
        ax_power = axes[1]
        ax_power.plot(
            shot_ds["time"], shot_ds["P_oh_MW"], label="P_oh [MW]", color="orange"
        )
        ax_power.plot(
            shot_ds["time"], shot_ds["P_rad_MW"], label="P_rad [MW]", color="red"
        )
        ax_power.plot(
            shot_ds["time"], shot_ds["P_NBI_MW"], label="P_NBI [MW]", color="cyan"
        )
        ax_power.plot(
            shot_ds["time"], shot_ds["P_ECRH_MW"], label="P_ECRH [MW]", color="lime"
        )
        if "LH_transition_threshold_MW" in shot_ds:
            ax_power.plot(
                shot_ds["time"],
                shot_ds["LH_transition_threshold_MW"] / 1e6,
                label="LH_Thresh [MW]",
                color="white",
                linestyle="--",
            )
        ax_power.set_ylabel("Power [MW]", fontsize=LABEL_FONTSIZE, color="white")
        ax_power.set_ylim(ylim_power)
        ax_power.legend(
            fontsize=LEGEND_FONTSIZE,
            facecolor=BACKGROUND_COLOR,
            edgecolor=BACKGROUND_COLOR,
            loc="upper left",
        )

        # density
        ax_ne = axes[2]
        ax_ne.plot(
            shot_ds["time"],
            shot_ds["ne20_line_avg"],
            label="ne20_line_avg [m^-3]",
            color="white",
        )
        ax_ne.set_ylabel("ne20_line_avg [m^-3]", fontsize=LABEL_FONTSIZE, color="white")
        ax_ne.set_ylim(ylim_ne)

        # Add B0 on right axis
        ax_b0 = ax_ne.twinx()
        if "B0" in shot_ds:
            ax_b0.plot(
                shot_ds["time"],
                shot_ds["B0"],
                label="B0 [T]",
                color="magenta",
                linestyle="-",
            )
        ax_b0.set_ylabel("B0 [T]", fontsize=LABEL_FONTSIZE, color="magenta")
        ax_b0.set_ylim(ylim_b0)
        ax_b0.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)

        # Shaping
        ax_shape = axes[3]
        ax_shape.plot(shot_ds["time"], shot_ds["a_minor"], label="a_minor", color="red")
        ax_shape.plot(shot_ds["time"], shot_ds["kappa"], label="kappa", color="yellow")
        ax_shape.plot(
            shot_ds["time"], shot_ds["delta_top"], label="delta_top", color="lime"
        )
        ax_shape.plot(
            shot_ds["time"],
            shot_ds["delta_bottom"],
            label="delta_bottom",
            color="green",
        )
        ax_shape.set_ylabel("Shaping", fontsize=LABEL_FONTSIZE, color="white")
        ax_shape.set_ylim(ylim_shape)
        ax_shape.set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE, color="white")
        ax_shape.legend(
            fontsize=LEGEND_FONTSIZE,
            facecolor=BACKGROUND_COLOR,
            edgecolor=BACKGROUND_COLOR,
            loc="upper left",
        )

        # Add R0 on right axis
        ax_r0 = ax_shape.twinx()
        if "R0" in shot_ds:
            ax_r0.plot(
                shot_ds["time"],
                shot_ds["R0"],
                label="R0 [m]",
                color="cyan",
                linestyle="--",
            )
        ax_r0.set_ylabel("R0 [m]", fontsize=LABEL_FONTSIZE, color="cyan")
        ax_r0.set_ylim(ylim_r0)
        ax_r0.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)

        for ax in axes:
            ax.set_facecolor(FACE_COLOR)
            ax.grid(True, color="gray", linestyle="--", linewidth=0.1)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
            try:
                for text in ax.get_legend().get_texts():
                    text.set_color(TEXT_COLOR)
            except AttributeError:
                pass

        fig.tight_layout()
        fig.savefig(f"{fig_dir}/{shot}.png")
        plt.close(fig)


def compare_powers(  # noqa: PLR0912
    ds_path: str,
    fig_dir: str,
    title: str = "Power Signal Comparison",
    num_shots: int | None = 9999,
):
    """Compare power signals with their _alt versions"""
    if ds_path.endswith(".zarr"):
        ds = xr.open_zarr(ds_path)
    else:
        ds = xr.open_dataset(ds_path)

    os.makedirs(fig_dir, exist_ok=True)

    # Find all power signals and their corresponding _alt versions
    power_signals = [
        "P_oh_MW",
        "P_rad_MW",
        "P_NBI_MW",
        "P_ECRH_MW",
        "P_LH_MW",
        "P_ICRF_MW",
    ]

    # Find which power signals have _alt versions in the dataset
    available_comparisons = []
    for sig in power_signals:
        alt_sig = f"{sig}_alt"
        if sig in ds.data_vars and alt_sig in ds.data_vars:
            available_comparisons.append((sig, alt_sig))

    if not available_comparisons:
        print("No power signals with _alt versions found for comparison")
        return

    # Compute global y-limits for each signal pair
    signal_ylims = {}
    for sig, alt_sig in available_comparisons:
        max_val = max(
            float(np.nanmax(ds[sig].values)), float(np.nanmax(ds[alt_sig].values))
        )
        signal_ylims[sig] = (0, max_val * 1.1)

    for shot in ds["shot"].data[:num_shots]:
        shot_ds = ds.sel(shot=shot)

        # Create subplots - one for each power signal comparison
        num_comparisons = len(available_comparisons)
        fig, axes = plt.subplots(
            num_comparisons, 1, figsize=(16, 4 * num_comparisons), sharex=True
        )
        if num_comparisons == 1:
            axes = [axes]  # Make it iterable for single subplot

        fig.patch.set_facecolor(BACKGROUND_COLOR)
        fig.suptitle(
            f"{title} - Shot {shot}", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR
        )

        for i, (sig, alt_sig) in enumerate(available_comparisons):
            ax = axes[i]

            # Plot original signal
            ax.plot(
                shot_ds["time"],
                shot_ds[sig],
                label=f"{sig.replace('_', ' ')}",
                color="cyan",
                linewidth=2,
            )

            # Plot alt signal
            ax.plot(
                shot_ds["time"],
                shot_ds[alt_sig],
                label=f"{alt_sig.replace('_', ' ')}",
                color="orange",
                linewidth=2,
                linestyle="--",
            )

            # Calculate and plot difference if both signals have data
            if not (
                np.all(np.isnan(shot_ds[sig])) or np.all(np.isnan(shot_ds[alt_sig]))
            ):
                diff = shot_ds[sig] - shot_ds[alt_sig]
                ax_diff = ax.twinx()
                ax_diff.plot(
                    shot_ds["time"],
                    diff,
                    label="Difference",
                    color="red",
                    alpha=0.7,
                    linewidth=1,
                )
                ax_diff.set_ylabel(
                    "Difference [MW]", fontsize=LABEL_FONTSIZE, color="red"
                )
                ax_diff.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors="red")

            ax.set_ylabel("Power [MW]", fontsize=LABEL_FONTSIZE, color="white")
            ax.set_ylim(signal_ylims[sig])
            ax.legend(
                fontsize=LEGEND_FONTSIZE,
                facecolor=BACKGROUND_COLOR,
                edgecolor=BACKGROUND_COLOR,
                loc="upper left",
            )

        # Set x-label on the bottom plot only
        axes[-1].set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE, color="white")

        # Style all axes
        for ax in axes:
            ax.set_facecolor(FACE_COLOR)
            ax.grid(True, color="gray", linestyle="--", linewidth=0.1)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
            try:
                for text in ax.get_legend().get_texts():
                    text.set_color(TEXT_COLOR)
            except AttributeError:
                pass

        fig.tight_layout()
        fig.savefig(
            f"{fig_dir}/power_comparison_{shot}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)


if __name__ == "__main__":
    ds_path = (
        "/fusion/projects/disruption_warning/data/popsim/tpt_d3d_final/d3d_hp.zarr"
    )
    fig_dir = "/fusion/projects/disruption_warning/data/popsim/tpt_d3d_final/power_comparisons"
    compare_powers(ds_path, fig_dir, title="DIII-D Power Signal Comparison")
