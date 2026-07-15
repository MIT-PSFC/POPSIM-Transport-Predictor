from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages

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

        rho = shot_ds["rho"].values
        time = shot_ds["time"].values

        # Extract 2D arrays for density and temperature
        ne_data = shot_ds["ne20_rho"].values.T  # shape: (rho, time) - transposed
        te_data = shot_ds["Te_keV_rho"].values.T  # shape: (rho, time) - transposed

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
            extent=[0, np.nanmax(time), rho.min(), rho.max()],
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
                extent=[0, np.nanmax(time), rho.min(), rho.max()],
            )

        ax_ne.set_ylabel(r"$\rho$", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
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
            extent=[0, np.nanmax(time), rho.min(), rho.max()],
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
                extent=[0, np.nanmax(time), rho.min(), rho.max()],
            )

        ax_te.set_ylabel(r"$\rho$", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
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


# Panel layout for the per-shot summary report: one row per dict, left/right axes
_SUMMARY_PANEL_DEFS = [
    dict(
        left_vars=["Ip_MA"],
        left_label="Ip [MA]",
        left_colors={"Ip_MA": "cyan"},
        left_floor_zero=True,
        left_abs=True,
        right_vars=["B0"],
        right_label="B0 [T]",
        right_colors={"B0": "magenta"},
        right_floor_zero=True,
    ),
    dict(
        left_vars=["Wtot_MJ"],
        left_label="Wtot [MJ]",
        left_colors={"Wtot_MJ": "red"},
        left_floor_zero=True,
        right_vars=["betan"],
        right_label="betan",
        right_colors={"betan": "magenta"},
        right_floor_zero=True,
    ),
    dict(
        left_vars=["ne20_line_avg", "ne20_edge"],
        left_label="ne20 [1e20 m^-3]",
        left_colors={"ne20_line_avg": "white", "ne20_edge": "yellow"},
        left_floor_zero=True,
        left_cap=5,
        show_fresh_profiles=True,
    ),
    dict(
        left_vars=["a_minor", "kappa", "delta_top", "delta_bot"],
        left_label="Shaping",
        left_colors={"a_minor": "red", "kappa": "yellow", "delta_top": "lime", "delta_bot": "green"},
    ),
    dict(
        left_vars=["P_oh_MW", "P_rad_MW", "P_NBI_MW", "P_ECRH_MW", "P_LH_MW", "P_ICRF_MW"],
        left_label="Power [MW]",
        left_colors={
            "P_oh_MW": "orange",
            "P_rad_MW": "red",
            "P_NBI_MW": "cyan",
            "P_ECRH_MW": "lime",
            "P_LH_MW": "yellow",
            "P_ICRF_MW": "magenta",
        },
        left_floor_zero=True,
        left_cap=10,
    ),
]


def _axis_ylim(ds, variables, floor_zero=False, use_abs=False, cap=None):
    """Global (lo, hi) y-limits for a set of variables across the whole dataset"""
    present = [v for v in variables if v in ds]
    if not present:
        return None
    hi = max(float(np.nanmax(np.abs(ds[v].values) if use_abs else ds[v].values)) for v in present)
    if cap is not None:
        hi = min(hi, cap)
    if floor_zero:
        lo = 0.0
    else:
        lo = min(float(np.nanmin(ds[v].values)) for v in present)
        lo = lo * 0.9 if lo > 0 else lo * 1.1
    hi = hi * 1.1
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return (0, 1)
    return (lo, hi)


def _summary_ylims(ds):
    """Precompute per-panel left/right y-limits so axes are consistent across shots"""
    all_ylims = []
    for panel in _SUMMARY_PANEL_DEFS:
        ylims = {
            "left": _axis_ylim(
                ds,
                panel["left_vars"],
                floor_zero=panel.get("left_floor_zero", False),
                use_abs=panel.get("left_abs", False),
                cap=panel.get("left_cap"),
            )
        }
        if panel.get("right_vars"):
            ylims["right"] = _axis_ylim(
                ds,
                panel["right_vars"],
                floor_zero=panel.get("right_floor_zero", False),
                use_abs=panel.get("right_abs", False),
                cap=panel.get("right_cap"),
            )
        all_ylims.append(ylims)
    return all_ylims


def _plot_summary_panel(ax, shot_ds, panel, ylims):
    """Plot one panel's signals for a single shot onto ax (and a twin axis if right_vars given)"""
    for var in panel["left_vars"]:
        if var in shot_ds:
            ax.plot(shot_ds["time"], shot_ds[var], label=var, color=panel["left_colors"].get(var))
    ax.set_ylabel(panel["left_label"], fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    if ylims.get("left") is not None:
        ax.set_ylim(ylims["left"])

    # Green dots at 0 marking timesteps that have fresh profiles
    if panel.get("show_fresh_profiles") and "fresh_profiles" in shot_ds:
        ax.plot(
            shot_ds["time"],
            np.where(shot_ds["fresh_profiles"] > 0, 0, np.nan),
            color="green",
            marker="o",
            linestyle="None",
        )

    axes = [ax]
    if panel.get("right_vars"):
        ax_right = ax.twinx()
        for var in panel["right_vars"]:
            if var in shot_ds:
                ax_right.plot(shot_ds["time"], shot_ds[var], label=var, color=panel["right_colors"].get(var), linestyle="--")
        ax_right.set_ylabel(panel["right_label"], fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        ax_right.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
        if ylims.get("right") is not None:
            ax_right.set_ylim(ylims["right"])
        axes.append(ax_right)
    return axes


def _shot_summary_page(shot_ds, shot, title, all_ylims):
    """Build one page of input-signal time traces for a single shot"""
    fig, axes = plt.subplots(len(_SUMMARY_PANEL_DEFS), 1, figsize=(11, 16), sharex=True)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    fig.suptitle(f"{title} - shot {shot}", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)

    all_axes = []
    for ax, panel, ylims in zip(axes, _SUMMARY_PANEL_DEFS, all_ylims, strict=True):
        all_axes += _plot_summary_panel(ax, shot_ds, panel, ylims)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(
                fontsize=LEGEND_FONTSIZE,
                facecolor=BACKGROUND_COLOR,
                edgecolor=BACKGROUND_COLOR,
                loc="upper left",
            )

    axes[-1].set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)

    for ax in all_axes:
        ax.set_facecolor(FACE_COLOR)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
    for ax in axes:
        ax.grid(True, color="gray", linestyle="--", linewidth=0.1)
        try:
            for text in ax.get_legend().get_texts():
                text.set_color(TEXT_COLOR)
        except AttributeError:
            pass

    fig.tight_layout()
    return fig


def _summary_stats_page(ds, title):
    """Build a table figure of min / max / mean / std for every data variable"""
    rows = []
    for var in ds.data_vars:
        values = ds[var]
        try:
            var_min = float(values.min(skipna=True).compute())
            var_max = float(values.max(skipna=True).compute())
            var_mean = float(values.mean(skipna=True).compute())
            var_std = float(values.std(skipna=True).compute())
        except (ValueError, TypeError):
            continue  # No valid values, or a non-numeric variable
        rows.append([var, f"{var_min:.4g}", f"{var_max:.4g}", f"{var_mean:.4g}", f"{var_std:.4g}"])

    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.axis("off")
    fig.suptitle(f"{title} - Summary Statistics", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)

    table = ax.table(
        cellText=rows,
        colLabels=["Variable", "Min", "Max", "Mean", "Std"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("gray")
        cell.get_text().set_color(TEXT_COLOR)
        cell.set_facecolor(FACE_COLOR if row > 0 else BACKGROUND_COLOR)

    fig.tight_layout()
    return fig


def ds_summary_report(
    ds: str | xr.Dataset,
    pdf_path: Path | str,
    title: str = "Dataset Summary Report",
    num_shots: int | None = None,
):
    """Make a multi-page PDF report of time traces for all input signals, plus a summary stats page.

    One page per shot, each with 5 panels:
    1. Ip_MA and B0
    2. Wtot_MJ and betan
    3. ne20 (line-average and edge), with green dots marking fresh-profile timesteps
    4. Shaping parameters (a_minor, kappa, delta_top, delta_bot)
    5. Power sources and sinks

    Y-limits are computed once across the whole dataset so axes are consistent between shots.
    A final page has summary statistics (min / max / mean / std) for every variable.
    """
    if isinstance(ds, (str, Path)):
        ds_path = str(ds)
        ds = xr.open_zarr(ds_path) if ds_path.endswith(".zarr") else xr.open_dataset(ds_path)

    Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
    shots = ds["shot"].data if num_shots is None else ds["shot"].data[:num_shots]
    all_ylims = _summary_ylims(ds)

    with PdfPages(pdf_path) as pdf:
        for shot in shots:
            fig = _shot_summary_page(ds.sel(shot=shot), shot, title, all_ylims)
            pdf.savefig(fig)
            plt.close(fig)

        fig = _summary_stats_page(ds, title)
        pdf.savefig(fig)
        plt.close(fig)


def fit_mean_ylim(fit_mean: np.ndarray, fallback: float, pad: float = 1.05) -> float:
    """Top y-limit for a TS fit panel: a little over the largest GP fit mean.

    Computed across the whole shot so every page shares the same axis. Returns
    fallback when the fit is all-NaN or non-positive.
    """
    hi = float(np.nanmax(fit_mean)) if np.isfinite(fit_mean).any() else np.nan
    if not np.isfinite(hi) or hi <= 0:
        return fallback
    return hi * pad
