from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

SOURCE_COLORS = {
    "cmod": "#d62728",
    "tcv": "#ff7f0e",
    "d3d_lp": "#1f77b4",
    "d3d_hp": "#ff00e1",
}

# Train is a square, val is a circle, test is a star
DATASET_SHAPES = {
    "train": "s",
    "val": "o",
    "test": "*",
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
    _fig, ax = plt.subplots(figsize=(10, 8))

    # Collect all data points for contour plotting
    all_x = []
    all_y = []
    all_performance = []
    all_sources = []
    all_dataset_labels = []  # Track which dataset each point belongs to

    for dataset_idx, ds in enumerate(ds_list):
        # Extract coordinates - use _p75 suffix for scalar per-shot values
        x_var_p75 = f"{x_var}_p75"
        y_var_p75 = f"{y_var}_p75"

        x_vals = ds[x_var_p75].values
        y_vals = ds[y_var_p75].values
        source_vals = ds.coords["ds_source"].values

        # Flatten if needed
        x_vals = np.atleast_1d(x_vals).flatten()
        y_vals = np.atleast_1d(y_vals).flatten()

        # Handle ds_source - could be array or single value
        if np.isscalar(source_vals) or source_vals.size == 1:
            source_vals = np.full_like(x_vals, source_vals, dtype=object)
        else:
            source_vals = np.atleast_1d(source_vals).flatten()

        all_x.extend(x_vals)
        all_y.extend(y_vals)
        all_sources.extend(source_vals)
        all_dataset_labels.extend([dataset_idx] * len(x_vals))

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_performance = np.array(all_performance)
    all_sources = np.array(all_sources)
    all_dataset_labels = np.array(all_dataset_labels)

    # Filter out NaN values
    valid_mask = ~(np.isnan(all_x) | np.isnan(all_y) | np.isnan(all_performance))
    all_x = all_x[valid_mask]
    all_y = all_y[valid_mask]
    all_sources = all_sources[valid_mask]
    all_dataset_labels = all_dataset_labels[valid_mask]

    # Find limits for plotting
    x_min, x_max = 0, all_x.max() * 1.1
    y_min, y_max = 0, all_y.max() * 1.1

    # Plot scatter points colored by data source
    # and shaped by dataset (if multiple datasets)
    source_handles = []
    dataset_handles = []
    for source in np.unique(all_sources):
        mask = all_sources == source
        scatter = ax.scatter(
            all_x[mask],
            all_y[mask],
            c=SOURCE_COLORS.get(source, "black"),
            label=source,
            s=50,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )
        source_handles.append(scatter)

    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_title(f"Performance Extrapolation in {x_var}-{y_var} Space")

    # Create combined legend with both datasets and data sources
    if dataset_handles:
        # Add a separator between datasets and sources

        all_handles = [*dataset_handles, *source_handles]
        all_labels = [
            *[h.get_label() for h in dataset_handles],
            "",
            *[h.get_label() for h in source_handles],
        ]

        # Create legend with two sections
        ax.legend(
            all_handles,
            all_labels,
            title="Dataset / Data Source",
            loc="upper left",
            framealpha=0.9,
        )
    else:
        ax.legend(title="Data Source")

    ax.grid(True, alpha=0.3)
    # Set the x and y limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
