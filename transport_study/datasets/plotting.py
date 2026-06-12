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


def ds_power_balance_time_plot(  # noqa: PLR0915
    ds: str | xr.Dataset,
    fig_dir: Path | str,
    num_shots: int | None = 9999,
    title: str = "Dataset Time Traces",
):
    """Plot time traces of signals from the dataset"""
    if isinstance(ds, (str, Path)):
        ds_path = str(ds)
        if ds_path.endswith(".zarr"):
            ds = xr.open_zarr(ds_path)
        else:
            ds = xr.open_dataset(ds_path)

    Path(fig_dir).mkdir(parents=True, exist_ok=True)

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
    power_max = min(max(float(np.nanmax(ds[sig].values)) for sig in power_signals), 10)
    if "LH_transition_threshold_MW" in ds:
        lh_thresh_max = float(np.nanmax(ds["LH_transition_threshold_MW"].values / 1e6))
    else:
        lh_thresh_max = 0
    ylim_power = (0, max(power_max, lh_thresh_max) * 1.1)

    density_signals = ["ne20_line_avg", "ne20_edge"]
    density_max = min(max(float(np.nanmax(ds[sig].values)) for sig in density_signals), 5)
    ylim_ne = (0, density_max * 1.1)

    # B0 y-limits for density plot right axis
    if "B0" in ds:
        ylim_b0 = (0, float(np.nanmax(ds["B0"].values)) * 1.1)
    else:
        ylim_b0 = (0, 5)  # Default range

    shape_signals = ["a_minor", "kappa", "delta_top", "delta_bot"]
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
            ax_wtot.plot(shot_ds["time"], shot_ds["Wtot_MJ"], label="Wtot [MJ]", color="red")
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
        ax_power.plot(shot_ds["time"], shot_ds["P_oh_MW"], label="P_oh [MW]", color="orange")
        ax_power.plot(shot_ds["time"], shot_ds["P_rad_MW"], label="P_rad [MW]", color="red")
        ax_power.plot(shot_ds["time"], shot_ds["P_NBI_MW"], label="P_NBI [MW]", color="cyan")
        ax_power.plot(shot_ds["time"], shot_ds["P_ECRH_MW"], label="P_ECRH [MW]", color="lime")
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

        # line avg and edge density
        ax_ne = axes[2]
        ax_ne.plot(
            shot_ds["time"],
            shot_ds["ne20_line_avg"],
            label="ne20_line_avg",
            color="white",
        )
        ax_ne.plot(
            shot_ds["time"],
            shot_ds["ne20_edge"],
            label="ne20_edge",
            color="yellow",
        )
        ax_ne.set_ylabel("ne20 [m^-3]", fontsize=LABEL_FONTSIZE, color="white")
        ax_ne.set_ylim(ylim_ne)

        # Dots at 0 for fresh profiles
        ax_ne.plot(
            shot_ds["time"],
            np.where(shot_ds["fresh_profiles"] > 0, 0, np.nan),
            color="green",
            marker="o",
            linestyle="None",
        )

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
        ax_shape.plot(shot_ds["time"], shot_ds["delta_top"], label="delta_top", color="lime")
        ax_shape.plot(
            shot_ds["time"],
            shot_ds["delta_bot"],
            label="delta_bot",
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
        fig.savefig(f"{fig_dir}/{shot}_trace.png")
        plt.close(fig)


def ds_profile_time_plot(  # noqa: PLR0915, PLR0912
    ds: str | xr.Dataset,
    fig_dir: Path | str,
    num_shots: int | None = 9999,
    title: str = "Profile dataset Time Traces",
):
    """Plot time traces of all signals of interest from the dataset.

    Four axes:
    1. Ip_MA, Wtot_MJ, and betan
    2. Line averaged density and B0
    3. All power sources and sinks
    4. Shaping parameters
    """
    if isinstance(ds, (str, Path)):
        ds_path = str(ds)
        if ds_path.endswith(".zarr"):
            ds = xr.open_zarr(ds_path)
        else:
            ds = xr.open_dataset(ds_path)

    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    # Compute global y-limits across all shots for consistent axes
    ylim_ip = (0, float(np.nanmax(np.abs(ds["Ip_MA"].values))) * 1.1)
    if "Wtot_MJ" in ds:
        ylim_wtot = (0, float(np.nanmax(ds["Wtot_MJ"].values)) * 1.1)
    else:
        ylim_wtot = (0, 1)  # Default range
    if "betan" in ds:
        ylim_betan = (0, float(np.nanmax(ds["betan"].values)) * 1.1)
    else:
        ylim_betan = (0, 1)  # Default range

    density_signals = [sig for sig in ["ne20_line_avg", "ne20_edge"] if sig in ds]
    density_max = min(max(float(np.nanmax(ds[sig].values)) for sig in density_signals), 5)
    ylim_ne = (0, density_max * 1.1)

    # B0 y-limits for density plot right axis
    if "B0" in ds:
        ylim_b0 = (0, float(np.nanmax(ds["B0"].values)) * 1.1)
    else:
        ylim_b0 = (0, 5)  # Default range

    power_signals = [
        sig
        for sig in [
            "P_oh_MW",
            "P_rad_MW",
            "P_NBI_MW",
            "P_ECRH_MW",
            "P_LH_MW",
            "P_ICRF_MW",
        ]
        if sig in ds
    ]
    if power_signals:
        power_max = min(max(float(np.nanmax(ds[sig].values)) for sig in power_signals), 10)
    else:
        power_max = 1
    if "LH_transition_threshold_MW" in ds:
        lh_thresh_max = float(np.nanmax(ds["LH_transition_threshold_MW"].values / 1e6))
    else:
        lh_thresh_max = 0
    ylim_power = (0, max(power_max, lh_thresh_max) * 1.1)

    shape_signals = ["a_minor", "kappa", "delta_top", "delta_bot"]
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

    # If any ylim is NaN or infinite, set it to a default range
    if not np.isfinite(ylim_ip).all():
        ylim_ip = (0, 1)
    if not np.isfinite(ylim_wtot).all():
        ylim_wtot = (0, 1)
    if not np.isfinite(ylim_betan).all():
        ylim_betan = (0, 1)
    if not np.isfinite(ylim_ne).all():
        ylim_ne = (0, 1)
    if not np.isfinite(ylim_b0).all():
        ylim_b0 = (0, 1)
    if not np.isfinite(ylim_power).all():
        ylim_power = (0, 1)
    if not np.isfinite(ylim_shape).all():
        ylim_shape = (0, 1)
    if not np.isfinite(ylim_r0).all():
        ylim_r0 = (0, 1)

    for shot in ds["shot"].data[:num_shots]:
        shot_ds = ds.sel(shot=shot)

        fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
        fig.patch.set_facecolor(BACKGROUND_COLOR)

        fig.suptitle(f"{title} - {shot}", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)

        # Ip, Wtot, and betan
        ax_ip = axes[0]
        # Put Ip and Wtot on the left y axis and betan on the right y axis
        ax_ip.plot(shot_ds["time"], shot_ds["Ip_MA"], label="Ip [MA]", color="cyan")
        if "Wtot_MJ" in shot_ds:
            ax_ip.plot(shot_ds["time"], shot_ds["Wtot_MJ"], label="Wtot [MJ]", color="red")
        ax_ip.set_ylabel("Ip [MA] / Wtot [MJ]", fontsize=LABEL_FONTSIZE, color="white")
        ax_ip.set_ylim((0, max(ylim_ip[1], ylim_wtot[1])))
        ax_ip.legend(
            fontsize=LEGEND_FONTSIZE,
            facecolor=BACKGROUND_COLOR,
            edgecolor=BACKGROUND_COLOR,
            loc="upper left",
        )
        ax_betan = ax_ip.twinx()
        if "betan" in shot_ds:
            ax_betan.plot(shot_ds["time"], shot_ds["betan"], label="betan", color="magenta")
        ax_betan.set_ylabel("Normalized Beta", fontsize=LABEL_FONTSIZE, color="magenta")
        ax_betan.set_ylim(ylim_betan)
        ax_betan.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)

        # line avg and edge density
        ax_ne = axes[1]
        if "ne20_line_avg" in shot_ds:
            ax_ne.plot(
                shot_ds["time"],
                shot_ds["ne20_line_avg"],
                label="ne20_line_avg",
                color="white",
            )
        if "ne20_edge" in shot_ds:
            ax_ne.plot(
                shot_ds["time"],
                shot_ds["ne20_edge"],
                label="ne20_edge",
                color="yellow",
            )
        ax_ne.set_ylabel("ne20 [10^20 m^-3]", fontsize=LABEL_FONTSIZE, color="white")
        ax_ne.set_ylim(ylim_ne)

        # Dots at 0 for fresh profiles
        if "fresh_profiles" in shot_ds:
            ax_ne.plot(
                shot_ds["time"],
                np.where(shot_ds["fresh_profiles"] > 0, 0, np.nan),
                color="green",
                marker="o",
                linestyle="None",
            )
        ax_ne.legend(
            fontsize=LEGEND_FONTSIZE,
            facecolor=BACKGROUND_COLOR,
            edgecolor=BACKGROUND_COLOR,
            loc="upper left",
        )

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

        # All power sources and sinks
        ax_power = axes[2]
        power_colors = {
            "P_oh_MW": "orange",
            "P_rad_MW": "red",
            "P_NBI_MW": "cyan",
            "P_ECRH_MW": "lime",
            "P_LH_MW": "yellow",
            "P_ICRF_MW": "magenta",
        }
        for sig in power_signals:
            ax_power.plot(
                shot_ds["time"],
                shot_ds[sig],
                label=f"{sig.replace('_MW', '')} [MW]",
                color=power_colors[sig],
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

        # Shaping
        ax_shape = axes[3]
        ax_shape.plot(shot_ds["time"], shot_ds["a_minor"], label="a_minor", color="red")
        ax_shape.plot(shot_ds["time"], shot_ds["kappa"], label="kappa", color="yellow")
        ax_shape.plot(shot_ds["time"], shot_ds["delta_top"], label="delta_top", color="lime")
        ax_shape.plot(
            shot_ds["time"],
            shot_ds["delta_bot"],
            label="delta_bot",
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
        fig.savefig(f"{fig_dir}/{shot}_trace.png")
        plt.close(fig)


def ds_profile_plot(
    ds: str | xr.Dataset,
    fig_dir: Path | str,
    num_shots: int | None = 9999,
    title: str = "Dataset Time Traces",
):
    """Plot profile traces of signals from the dataset"""
    if isinstance(ds, (str, Path)):
        ds_path = str(ds)
        if ds_path.endswith(".zarr"):
            ds = xr.open_zarr(ds_path)
        else:
            ds = xr.open_dataset(ds_path)

    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    for shot in ds["shot"].data[:num_shots]:
        shot_ds = ds.sel(shot=shot)

        psi_n = shot_ds["psi_n"].values
        time = shot_ds["time"].values

        # Extract 2D arrays for density and temperature
        ne_data = shot_ds["ne20_psi"].values.T  # shape: (psi_n, time) - transposed
        te_data = shot_ds["Te_keV_psi"].values.T  # shape: (psi_n, time) - transposed

        # Create masks for timesteps with NaN values
        ne_nan_mask = np.isnan(ne_data).any(axis=0)  # True if any NaN in that timestep
        te_nan_mask = np.isnan(te_data).any(axis=0)  # True if any NaN in that timestep

        fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        fig.suptitle(
            f"{title} - {shot}",
            fontsize=TITLE_FONTSIZE,
            color=TEXT_COLOR,
        )

        ax_ne = axes[0]
        ax_te = axes[1]

        # Plot density heatmap
        ne_plot_data = ne_data.copy()
        ne_plot_data[:, ne_nan_mask] = np.nan  # Set NaN timesteps to NaN for proper masking
        im_ne = ax_ne.imshow(
            ne_plot_data,
            cmap="viridis",
            aspect="auto",
            origin="lower",
            extent=[0, np.nanmax(time), psi_n.min(), psi_n.max()],
        )

        # Overlay bright pink for NaN timesteps
        if np.any(ne_nan_mask):
            ne_pink_data = np.full_like(ne_data, np.nan)
            ne_pink_data[:, ne_nan_mask] = 1.0  # Use constant value for bright color
            ax_ne.imshow(
                ne_pink_data,
                cmap="Reds",
                aspect="auto",
                origin="lower",
                alpha=0.8,
                extent=[0, np.nanmax(time), psi_n.min(), psi_n.max()],
            )

        ax_ne.set_ylabel(r"$\psi_n$", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        ax_ne.set_title(r"$n_e$ [$10^{20}$ m$^{-3}$]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        cbar_ne = plt.colorbar(im_ne, ax=ax_ne)
        cbar_ne.ax.tick_params(labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)

        # Plot temperature heatmap
        te_plot_data = te_data.copy()
        te_plot_data[:, te_nan_mask] = np.nan  # Set NaN timesteps to NaN for proper masking
        im_te = ax_te.imshow(
            te_plot_data,
            cmap="plasma",
            aspect="auto",
            origin="lower",
            extent=[0, np.nanmax(time), psi_n.min(), psi_n.max()],
        )

        # Overlay bright pink for NaN timesteps
        if np.any(te_nan_mask):
            te_pink_data = np.full_like(te_data, np.nan)
            te_pink_data[:, te_nan_mask] = 1.0  # Use constant value for bright color
            ax_te.imshow(
                te_pink_data,
                cmap="Reds",
                aspect="auto",
                origin="lower",
                alpha=0.8,
                extent=[0, np.nanmax(time), psi_n.min(), psi_n.max()],
            )

        ax_te.set_ylabel(r"$\psi_n$", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        ax_te.set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        ax_te.set_title(r"$T_e$ [keV]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        cbar_te = plt.colorbar(im_te, ax=ax_te)
        cbar_te.ax.tick_params(labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)

        for ax in axes:
            ax.set_facecolor(FACE_COLOR)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
            ax.set_xlim(np.nanmin(time), np.nanmax(time))
            for spine in ax.spines.values():
                spine.set_color(TEXT_COLOR)

        fig.tight_layout()
        fig.savefig(Path(fig_dir) / f"{shot}_profiles.png", dpi=150)
        plt.close(fig)


def compare_powers(
    ds_path: Path | str,
    fig_dir: Path | str,
    title: str = "Power Signal Comparison",
    num_shots: int | None = 9999,
):
    """Compare power signals with their _alt versions"""
    if ds_path.endswith(".zarr"):
        ds = xr.open_zarr(ds_path)
    else:
        ds = xr.open_dataset(ds_path)

    Path(fig_dir).mkdir(parents=True, exist_ok=True)

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
        max_val = max(float(np.nanmax(ds[sig].values)), float(np.nanmax(ds[alt_sig].values)))
        signal_ylims[sig] = (0, max_val * 1.1)

    for shot in ds["shot"].data[:num_shots]:
        shot_ds = ds.sel(shot=shot)

        # Create subplots - one for each power signal comparison
        num_comparisons = len(available_comparisons)
        fig, axes = plt.subplots(num_comparisons, 1, figsize=(16, 4 * num_comparisons), sharex=True)
        if num_comparisons == 1:
            axes = [axes]  # Make it iterable for single subplot

        fig.patch.set_facecolor(BACKGROUND_COLOR)
        fig.suptitle(f"{title} - Shot {shot}", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)

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
            if not (np.all(np.isnan(shot_ds[sig])) or np.all(np.isnan(shot_ds[alt_sig]))):
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
                ax_diff.set_ylabel("Difference [MW]", fontsize=LABEL_FONTSIZE, color="red")
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
        fig.savefig(f"{fig_dir}/power_comparison_{shot}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
