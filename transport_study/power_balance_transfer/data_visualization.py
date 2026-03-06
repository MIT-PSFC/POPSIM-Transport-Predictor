from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import patches
from scipy.spatial import ConvexHull

BACKGROUND_COLOR = "#2F2F2F"
FACE_COLOR = "#1A1A1A"
TEXT_COLOR = "white"

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14

SOURCE_COLORS = {
    "cmod": "#ff4d4d",
    "tcv": "#8dff36",
    "d3d_lp": "#0095ff",
    "d3d_hp": "#ff60ec",
}

DATASET_SHAPES = {
    "train": ("s", 120),
    "val": ("o", 160),
    "test": ("*", 240),
}


def performance_extrapolation_plot(  # noqa: PLR0915
    save_path: str,
    ds_list: list[xr.Dataset],
    ds_type_list: list[str],
    x_var: str = "Ip_MA",
    y_var: str = "Wtot_MJ",
):
    """
    Generate performance extrapolation plots

    Args:
        save_path: Path to save the generated figure
        ds_list: List of xarray Datasets to plot. Each dataset has a coordinate "ds_source" indicating the data source (e.g. "cmod", "tcv", "d3d_lp", "d3d_hp")
        ds_type_list: List of dataset types corresponding to each dataset in ds_list (e.g. "train", "val", "test"). Used for marker shapes.
        x_var: Name of the variable to plot on the x-axis (default: "Ip_MA")
        y_var: Name of the variable to plot on the y-axis (default: "Wtot_MJ")
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(FACE_COLOR)

    all_x = []
    all_y = []
    all_sources = []
    all_ds_types = []

    for dataset_idx, ds in enumerate(ds_list):
        x_var_p95 = f"{x_var}_p95"
        y_var_p95 = f"{y_var}_p95"

        x_vals = ds[x_var_p95].values
        y_vals = ds[y_var_p95].values
        source_vals = ds.coords["ds_source"].values

        x_vals = np.atleast_1d(x_vals).flatten()
        y_vals = np.atleast_1d(y_vals).flatten()

        if np.isscalar(source_vals) or source_vals.size == 1:
            source_vals = np.full_like(x_vals, source_vals, dtype=object)
        else:
            source_vals = np.atleast_1d(source_vals).flatten()

        ds_type = (
            ds_type_list[dataset_idx] if dataset_idx < len(ds_type_list) else "unknown"
        )

        all_x.extend(x_vals)
        all_y.extend(y_vals)
        all_sources.extend(source_vals)
        all_ds_types.extend([ds_type] * len(x_vals))

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_sources = np.array(all_sources)
    all_ds_types = np.array(all_ds_types)

    valid_mask = ~(np.isnan(all_x) | np.isnan(all_y))
    all_x = all_x[valid_mask]
    all_y = all_y[valid_mask]
    all_sources = all_sources[valid_mask]
    all_ds_types = all_ds_types[valid_mask]

    x_max = all_x.max() * 1.1 if all_x.size > 0 else 1
    y_max = all_y.max() * 1.1 if all_y.size > 0 else 1

    handles = []
    for ds_type in np.unique(all_ds_types):
        mask = all_ds_types == ds_type
        marker, size = DATASET_SHAPES.get(ds_type.lower(), ("o", 60))

        for source in np.unique(all_sources[mask]):
            sub_mask = mask & (all_sources == source)
            scatter = ax.scatter(
                all_x[sub_mask],
                all_y[sub_mask],
                c=SOURCE_COLORS.get(source, "black"),
                marker=marker,
                label=f"{ds_type} - {source}",
                s=size,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
            )
            handles.append(scatter)

    ax.set_xlabel(x_var, fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    ax.set_ylabel(y_var, fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    ax.set_title(
        f"Performance Extrapolation in {x_var}-{y_var} Space",
        fontsize=TITLE_FONTSIZE,
        color=TEXT_COLOR,
    )

    if handles:
        legend = ax.legend(
            title="Dataset / Data Source", loc="upper left", framealpha=0.9
        )
        legend.get_title().set_color(TEXT_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)
        legend.get_frame().set_facecolor(FACE_COLOR)
        legend.get_frame().set_edgecolor(TEXT_COLOR)

    ax.tick_params(axis="both", colors=TEXT_COLOR, labelsize=TICK_FONTSIZE)
    ax.grid(True, color="gray", linestyle="--", linewidth=0.3)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def domain_plot(  # noqa: PLR0915, PLR0912
    ds: xr.Dataset,
    var_groups: list[list[str]],
    title: str,
    save_path: str,
):
    """Make a 2x2 grid of scatter plots showing the domain of each dataset in different variable spaces, colored by data source."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    # Get the median value of the x and y variables for each shot (across the time_idx dimension)
    for var in {var for group in var_groups for var in group}:
        var_median = ds[var].median(dim="time_idx", skipna=True)
        ds[f"{var}_median"] = var_median

    for ax in axes.flatten():
        ax.set_facecolor(FACE_COLOR)

    for i, var_pair in enumerate(var_groups):
        x_var, y_var = var_pair

        x_vals = ds[f"{x_var}_median"].values
        y_vals = ds[f"{y_var}_median"].values
        source_vals = ds.coords["ds_source"].values

        x_vals = np.atleast_1d(x_vals).flatten()
        y_vals = np.atleast_1d(y_vals).flatten()

        if np.isscalar(source_vals) or source_vals.size == 1:
            source_vals = np.full_like(x_vals, source_vals, dtype=object)
        else:
            source_vals = np.atleast_1d(source_vals).flatten()

        valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        x_vals = x_vals[valid_mask]
        y_vals = y_vals[valid_mask]
        source_vals = source_vals[valid_mask]

        ax = axes[i // 2, i % 2]
        for source in np.unique(source_vals):
            device_mask = source_vals == source
            x_source = x_vals[device_mask]
            y_source = y_vals[device_mask]

            ax.scatter(
                x_source,
                y_source,
                c=SOURCE_COLORS.get(source, "black"),
                label=source,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
            )

            # Calculate ellipse from full dataset (all time points)
            device_mask_xr = ds.coords["ds_source"] == source
            x_all = ds[x_var].where(device_mask_xr, drop=True).values.flatten()
            y_all = ds[y_var].where(device_mask_xr, drop=True).values.flatten()

            # Remove NaNs
            valid_full = ~(np.isnan(x_all) | np.isnan(y_all))
            x_all = x_all[valid_full]
            y_all = y_all[valid_full]

            if len(x_all) > 0 and len(y_all) > 0:
                x_median_full = np.median(x_all)
                y_median_full = np.median(y_all)

                x_q75, x_q25 = np.percentile(x_all, [75, 25])
                y_q75, y_q25 = np.percentile(y_all, [75, 25])

                x_iqr = x_q75 - x_q25
                y_iqr = y_q75 - y_q25

                # Use ellipse with width=x_iqr and height=y_iqr from full dataset
                if x_iqr > 0 and y_iqr > 0:
                    ellipse = patches.Ellipse(
                        (x_median_full, y_median_full),
                        width=x_iqr,  # IQR in x-direction from full data
                        height=y_iqr,  # IQR in y-direction from full data
                        fill=False,
                        edgecolor=SOURCE_COLORS.get(source, "black"),
                        alpha=0.5,
                        linewidth=2,
                        linestyle="--",
                    )
                    ax.add_patch(ellipse)

                # Add convex hull showing maximum extent
                if len(x_all) >= 3:  # Need at least 3 points for a meaningful hull
                    try:
                        points = np.column_stack((x_all, y_all))
                        hull = ConvexHull(points)

                        # Get hull vertices
                        hull_points = points[hull.vertices]
                        # Close the polygon by adding first point at the end
                        hull_points = np.vstack([hull_points, hull_points[0]])

                        # Plot convex hull
                        ax.plot(
                            hull_points[:, 0],
                            hull_points[:, 1],
                            color=SOURCE_COLORS.get(source, "black"),
                            alpha=0.3,
                            linewidth=1,
                            linestyle="-",
                        )

                        # Optional: fill the hull area
                        hull_polygon = patches.Polygon(
                            hull_points[:-1],  # Exclude the duplicate closing point
                            closed=True,
                            facecolor=SOURCE_COLORS.get(source, "black"),
                            alpha=0.05,
                            edgecolor="none",
                        )
                        ax.add_patch(hull_polygon)

                    except Exception:
                        # Skip convex hull if computation fails (e.g., collinear points)
                        pass

        ax.set_xlabel(x_var, fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        ax.set_ylabel(y_var, fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        ax.set_title(f"{x_var} vs {y_var}", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)
        ax.tick_params(axis="both", colors=TEXT_COLOR, labelsize=TICK_FONTSIZE)
        ax.grid(True, color="gray", linestyle="--", linewidth=0.3)
        for spine in ax.spines.values():
            spine.set_color(TEXT_COLOR)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        legend = fig.legend(
            handles,
            labels,
            title="Data Source",
            loc="center",
            framealpha=0.9,
        )
        legend.get_title().set_color(TEXT_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)
        legend.get_frame().set_facecolor(FACE_COLOR)
        legend.get_frame().set_edgecolor(TEXT_COLOR)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def transfer_learning_losses(  # noqa: PLR0915
    loss_ds_list: list[xr.Dataset],
    label_list: list[str],
    hp_shots_included: list[int],
    save_path: str,
):
    """
    Plot the loss vs number of high-performance shots included in training for different transfer learning cases.

    One plot for average + std deviation of integrated loss across the entire trajectory,
    One plot for average + std deviation of loss at each timestep.

    loss_ds_list: List of xarray Datasets containing the losses for each transfer learning case.
        Each dataset should have:
        - coordinate "num_hp_shots" (int, with -1 representing "all" shots)
        - variables "integrated_loss_mean", "integrated_loss_std" with dim (num_hp_shots,)
        - variables "timestep_loss_mean", "timestep_loss_std" with dims (num_hp_shots, time_idx)
    label_list: Labels for each dataset in loss_ds_list (e.g. model architecture names).
    hp_shots_included: Full list of HP shot counts used in the study (may contain None for "all").
    save_path: Path to save the output figure.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax_left.set_facecolor(FACE_COLOR)
    ax_right.set_facecolor(FACE_COLOR)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(loss_ds_list), 1)))

    # Build x-axis tick positions and labels.
    # hp_shots_included may contain None (meaning "all").
    numeric_hp = [h for h in hp_shots_included if h is not None]
    all_x_pos = max(numeric_hp) * 1.5 if numeric_hp else 100

    x_tick_values = []
    x_tick_labels = []
    hp_to_x: dict[int, float] = {}  # coord value (-1 for None) -> x position
    for hp in hp_shots_included:
        if hp is None:
            x_tick_values.append(all_x_pos)
            x_tick_labels.append("all")
            hp_to_x[-1] = all_x_pos
        else:
            x_tick_values.append(hp)
            x_tick_labels.append(str(hp))
            hp_to_x[hp] = float(hp)

    # ---- Left panel: Integrated loss vs number of HP shots ----
    for i, (loss_ds, label) in enumerate(zip(loss_ds_list, label_list, strict=True)):
        hp_coords = loss_ds["num_hp_shots"].values
        mean_vals = loss_ds["integrated_loss_mean"].values.copy()
        std_vals = loss_ds["integrated_loss_std"].values.copy()

        x_pos = np.array([hp_to_x.get(int(h), float(h)) for h in hp_coords])
        sort_idx = np.argsort(x_pos)
        x_pos = x_pos[sort_idx]
        mean_vals = mean_vals[sort_idx]
        std_vals = std_vals[sort_idx]

        ax_left.plot(
            x_pos,
            mean_vals,
            "o-",
            color=colors[i],
            label=label,
            linewidth=2,
            markersize=6,
        )
        ax_left.fill_between(
            x_pos,
            mean_vals - std_vals,
            mean_vals + std_vals,
            color=colors[i],
            alpha=0.2,
        )

    ax_left.set_xlabel(
        "Number of HP Shots in Training", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR
    )
    ax_left.set_ylabel(
        "Integrated Loss (MSE)", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR
    )
    ax_left.set_title(
        "Transfer Learning: Integrated Loss", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR
    )
    ax_left.set_xticks(x_tick_values)
    ax_left.set_xticklabels(x_tick_labels)
    ax_left.tick_params(axis="both", colors=TEXT_COLOR, labelsize=TICK_FONTSIZE)
    ax_left.grid(True, color="gray", linestyle="--", linewidth=0.3)
    for spine in ax_left.spines.values():
        spine.set_color(TEXT_COLOR)
    if loss_ds_list:
        legend = ax_left.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.9)
        legend.get_frame().set_facecolor(FACE_COLOR)
        legend.get_frame().set_edgecolor(TEXT_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

    # ---- Right panel: Per-timestep loss profile (at the largest num_hp_shots) ----
    for i, (loss_ds, label) in enumerate(zip(loss_ds_list, label_list, strict=True)):
        if "timestep_loss_mean" not in loss_ds:
            continue

        ts_mean = loss_ds["timestep_loss_mean"].isel(num_hp_shots=-1).values
        ts_std = loss_ds["timestep_loss_std"].isel(num_hp_shots=-1).values

        # Mask out NaN padding from datasets with different time lengths
        valid = ~np.isnan(ts_mean)
        time_steps = np.arange(len(ts_mean))[valid]
        ts_mean = ts_mean[valid]
        ts_std = ts_std[valid]

        hp_val = int(loss_ds["num_hp_shots"].values[-1])
        hp_str = "all" if hp_val == -1 else str(hp_val)

        ax_right.plot(
            time_steps,
            ts_mean,
            "-",
            color=colors[i],
            label=f"{label} ({hp_str} HP shots)",
            linewidth=2,
        )
        ax_right.fill_between(
            time_steps,
            ts_mean - ts_std,
            ts_mean + ts_std,
            color=colors[i],
            alpha=0.2,
        )

    ax_right.set_xlabel("Time Step", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    ax_right.set_ylabel("Loss (MSE)", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    ax_right.set_title(
        "Per-Timestep Loss Profile", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR
    )
    ax_right.tick_params(axis="both", colors=TEXT_COLOR, labelsize=TICK_FONTSIZE)
    ax_right.grid(True, color="gray", linestyle="--", linewidth=0.3)
    for spine in ax_right.spines.values():
        spine.set_color(TEXT_COLOR)
    if loss_ds_list:
        legend = ax_right.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.9)
        legend.get_frame().set_facecolor(FACE_COLOR)
        legend.get_frame().set_edgecolor(TEXT_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
