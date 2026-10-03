"""Diagnostic plots of a device store, drawn in the study's working units."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from transport_validation_datasets.machine.generic import UNIFORM_TIMEBASE_DT

from transport_study import RADIAL_DIM, TIME_DIM
from transport_study.plot_style import BACKGROUND_COLOR, FACE_COLOR, TEXT_COLOR
from transport_study.signals import convert_to_working_units

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18


POWER_COLORS = {
    "power_ohm_MW": "orange",
    "power_radiated_MW": "red",
    "power_nbi_MW": "cyan",
    "power_ec_MW": "lime",
    "power_lh_MW": "yellow",
    "power_ic_MW": "magenta",
}
SHAPE_COLORS = {"minor_radius": "red", "elongation": "yellow", "triangularity_upper": "lime", "triangularity_lower": "green"}


def _working_unit_dataset(ds: str | Path | xr.Dataset) -> xr.Dataset:
    """A device store (path or dataset) in the study's working units."""
    if isinstance(ds, (str, Path)):
        ds_path = str(ds)
        ds = xr.open_zarr(ds_path) if ds_path.endswith(".zarr") else xr.open_dataset(ds_path)
    return convert_to_working_units(ds)


def ds_profile_time_plot(  # noqa: PLR0915
    ds: str | xr.Dataset,
    fig_dir: Path | str,
    num_shots: int | None = 9999,
    title: str = "Profile dataset Time Traces",
):
    """Plot time traces of all signals of interest from the dataset.

    Four axes:
    1. ip, energy_mhd, and beta_tor_norm
    2. Line averaged density and b0
    3. All power sources and sinks
    4. Shaping parameters
    """
    ds = _working_unit_dataset(ds)

    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    # Compute global y-limits across all shots for consistent axes
    ylim_ip = (0, float(np.nanmax(np.abs(ds["ip_MA"].values))) * 1.1)
    ylim_wtot = (0, float(np.nanmax(ds["energy_mhd_MJ"].values)) * 1.1)
    ylim_betan = (0, float(np.nanmax(ds["beta_tor_norm"].values)) * 1.1)

    density_max = min(float(np.nanmax(ds["n_e_line_average_1e20"].values)), 5)
    ylim_ne = (0, density_max * 1.1)

    # b0 y-limits for density plot right axis
    ylim_b0 = (0, float(np.nanmax(ds["b0"].values)) * 1.1)

    power_max = min(max(float(np.nanmax(ds[sig].values)) for sig in POWER_COLORS), 10)
    ylim_power = (0, power_max * 1.1)

    shape_min = min(float(np.nanmin(ds[sig].values)) for sig in SHAPE_COLORS)
    shape_max = max(float(np.nanmax(ds[sig].values)) for sig in SHAPE_COLORS)
    ylim_shape = (
        shape_min * 0.9 if shape_min > 0 else shape_min * 1.1,
        shape_max * 1.1,
    )

    # geometric_axis_r y-limits for shaping plot right axis
    ylim_r0 = (0, float(np.nanmax(ds["geometric_axis_r"].values)) * 1.1)

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

        # ip, energy_mhd, and beta_tor_norm
        ax_ip = axes[0]
        # Put ip and energy_mhd on the left y axis and beta_tor_norm on the right y axis
        ax_ip.plot(shot_ds["time"], shot_ds["ip_MA"], label="ip [MA]", color="cyan")
        ax_ip.plot(shot_ds["time"], shot_ds["energy_mhd_MJ"], label="energy_mhd [MJ]", color="red")
        ax_ip.set_ylabel("ip [MA] / energy_mhd [MJ]", fontsize=LABEL_FONTSIZE, color="white")
        ax_ip.set_ylim((0, max(ylim_ip[1], ylim_wtot[1])))
        ax_ip.legend(
            fontsize=LEGEND_FONTSIZE,
            facecolor=BACKGROUND_COLOR,
            edgecolor=BACKGROUND_COLOR,
            loc="upper left",
        )
        ax_betan = ax_ip.twinx()
        ax_betan.plot(shot_ds["time"], shot_ds["beta_tor_norm"], label="beta_tor_norm", color="magenta")
        ax_betan.set_ylabel("Normalized Beta", fontsize=LABEL_FONTSIZE, color="magenta")
        ax_betan.set_ylim(ylim_betan)
        ax_betan.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)

        # line averaged density
        ax_ne = axes[1]
        ax_ne.plot(
            shot_ds["time"],
            shot_ds["n_e_line_average_1e20"],
            label="n_e_line_average",
            color="white",
        )
        ax_ne.set_ylabel("n_e [10^20 m^-3]", fontsize=LABEL_FONTSIZE, color="white")
        ax_ne.set_ylim(ylim_ne)

        # Dots at 0 for fresh profiles
        ax_ne.plot(
            shot_ds["time"],
            np.where(shot_ds["fresh_profile"] > 0, 0, np.nan),
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

        # Add b0 on right axis
        ax_b0 = ax_ne.twinx()
        ax_b0.plot(
            shot_ds["time"],
            shot_ds["b0"],
            label="b0 [T]",
            color="magenta",
            linestyle="-",
        )
        ax_b0.set_ylabel("b0 [T]", fontsize=LABEL_FONTSIZE, color="magenta")
        ax_b0.set_ylim(ylim_b0)
        ax_b0.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)

        # All power sources and sinks
        ax_power = axes[2]
        for sig, color in POWER_COLORS.items():
            ax_power.plot(
                shot_ds["time"],
                shot_ds[sig],
                label=f"{sig.removesuffix('_MW')} [MW]",
                color=color,
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
        for sig, color in SHAPE_COLORS.items():
            ax_shape.plot(shot_ds["time"], shot_ds[sig], label=sig, color=color)
        ax_shape.set_ylabel("Shaping", fontsize=LABEL_FONTSIZE, color="white")
        ax_shape.set_ylim(ylim_shape)
        ax_shape.set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE, color="white")
        ax_shape.legend(
            fontsize=LEGEND_FONTSIZE,
            facecolor=BACKGROUND_COLOR,
            edgecolor=BACKGROUND_COLOR,
            loc="upper left",
        )

        # Add geometric_axis_r on right axis
        ax_r0 = ax_shape.twinx()
        ax_r0.plot(
            shot_ds["time"],
            shot_ds["geometric_axis_r"],
            label="geometric_axis_r [m]",
            color="cyan",
            linestyle="--",
        )
        ax_r0.set_ylabel("geometric_axis_r [m]", fontsize=LABEL_FONTSIZE, color="cyan")
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
    ds = _working_unit_dataset(ds)

    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    panels = [
        ("n_e_1e20", "viridis", r"$n_e$ [$10^{20}$ m$^{-3}$]"),
        ("t_e_keV", "plasma", r"$T_e$ [keV]"),
    ]
    for shot in ds["shot"].data[:num_shots]:
        shot_ds = ds.sel(shot=shot)
        # Store padding has NaN time.
        # Valid columns go onto the uniform grid, so timeslices dropped by filtering show as NaN columns.
        mask_time_valid = np.isfinite(shot_ds["time"].values)
        shot_ds_valid = shot_ds.isel({TIME_DIM: mask_time_valid})
        rho = shot_ds_valid[RADIAL_DIM].values
        time_valid = shot_ds_valid["time"].values
        grid_offset = (time_valid - time_valid.min()) / UNIFORM_TIMEBASE_DT
        grid_idx = np.rint(grid_offset).astype(int)
        time = time_valid.min() + np.arange(grid_idx.max() + 1) * UNIFORM_TIMEBASE_DT

        fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        fig.suptitle(
            f"{title} - {shot}",
            fontsize=TITLE_FONTSIZE,
            color=TEXT_COLOR,
        )

        for ax, (name, cmap, panel_title) in zip(axes, panels, strict=True):
            profile_data = np.full((rho.size, time.size), np.nan)
            profile_data[:, grid_idx] = shot_ds_valid[name].transpose(RADIAL_DIM, ...).values
            # Timesteps without a profile are overlaid in red.
            # Profiles may end inside the grid (DIII-D at the IDA domain), so only all-NaN columns count.
            mask_nan_timestep = np.isnan(profile_data).all(axis=0)
            mesh = ax.pcolormesh(time, rho, profile_data, cmap=cmap, shading="nearest")
            if np.any(mask_nan_timestep):
                nan_overlay = np.full_like(profile_data, np.nan)
                nan_overlay[:, mask_nan_timestep] = 1.0
                ax.pcolormesh(time, rho, nan_overlay, cmap="Reds", alpha=0.8, shading="nearest")

            ax.set_ylabel(r"$\rho_{tor,N}$", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
            ax.set_title(panel_title, fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
            cbar = plt.colorbar(mesh, ax=ax)
            cbar.ax.tick_params(labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
            ax.set_facecolor(FACE_COLOR)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
            ax.set_xlim(time.min(), time.max())
            for spine in ax.spines.values():
                spine.set_color(TEXT_COLOR)
        axes[-1].set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)

        fig.tight_layout()
        fig.savefig(Path(fig_dir) / f"{shot}_profiles.png", dpi=150)
        plt.close(fig)


# Panel layout for the per-shot summary report: one row per dict, left/right axes
_SUMMARY_PANEL_DEFS = [
    dict(
        left_vars=["ip_MA"],
        left_label="ip [MA]",
        left_colors={"ip_MA": "cyan"},
        left_floor_zero=True,
        left_abs=True,
        right_vars=["b0"],
        right_label="b0 [T]",
        right_colors={"b0": "magenta"},
        right_floor_zero=True,
    ),
    dict(
        left_vars=["energy_mhd_MJ"],
        left_label="energy_mhd [MJ]",
        left_colors={"energy_mhd_MJ": "red"},
        left_floor_zero=True,
        right_vars=["beta_tor_norm"],
        right_label="beta_tor_norm",
        right_colors={"beta_tor_norm": "magenta"},
        right_floor_zero=True,
    ),
    dict(
        left_vars=["n_e_line_average_1e20"],
        left_label="n_e [1e20 m^-3]",
        left_colors={"n_e_line_average_1e20": "white"},
        left_floor_zero=True,
        left_cap=5,
        show_fresh_profiles=True,
    ),
    dict(
        left_vars=list(SHAPE_COLORS),
        left_label="Shaping",
        left_colors=SHAPE_COLORS,
    ),
    dict(
        left_vars=list(POWER_COLORS),
        left_label="Power [MW]",
        left_colors=POWER_COLORS,
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
    if panel.get("show_fresh_profiles"):
        ax.plot(
            shot_ds["time"],
            np.where(shot_ds["fresh_profile"] > 0, 0, np.nan),
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
    1. ip and b0
    2. energy_mhd and beta_tor_norm
    3. Line averaged density, with green dots marking fresh-profile timesteps
    4. Shaping parameters (minor_radius, elongation, triangularity_upper, triangularity_lower)
    5. Power sources and sinks

    Y-limits are computed once across the whole dataset so axes are consistent between shots.
    A final page has summary statistics (min / max / mean / std) for every variable.
    """
    ds = _working_unit_dataset(ds)

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
