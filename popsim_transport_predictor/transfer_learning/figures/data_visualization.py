from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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
        x_var_p75 = f"{x_var}_p75"
        y_var_p75 = f"{y_var}_p75"

        x_vals = ds[x_var_p75].values
        y_vals = ds[y_var_p75].values
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


def domain_plot(
    ds: xr.Dataset,
    var_groups: list[list[str]],
    title: str,
    save_path: str,
):
    """Make a 2x2 grid of scatter plots showing the domain of each dataset in different variable spaces, colored by data source."""
