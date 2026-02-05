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


def ds_time_plot(  # noqa: PLR0915
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

    shape_signals = ["a_minor", "kappa", "delta_top", "delta_bottom"]
    shape_min = min(float(np.nanmin(ds[sig].values)) for sig in shape_signals)
    shape_max = max(float(np.nanmax(ds[sig].values)) for sig in shape_signals)
    ylim_shape = (
        shape_min * 0.9 if shape_min > 0 else shape_min * 1.1,
        shape_max * 1.1,
    )

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


if __name__ == "__main__":
    ds_path = (
        "/fusion/projects/disruption_warning/data/popsim/tpt_d3d_final/d3d_hp.zarr"
    )
    fig_dir = "/fusion/projects/disruption_warning/data/popsim/tpt_d3d_final/kept_shots"
    ds_time_plot(ds_path, fig_dir, title="DIII-D Dataset Time Traces")
