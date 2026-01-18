from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import griddata


def performance_extrapolation_plot(  # noqa: PLR0915
    save_path: str,
    ds_list: list[xr.Dataset],
    labels: list[str],
    performance_metric: str = "performance",
    x_var: str = "Ip_MA",
    y_var: str = "Wtot_MJ",
):
    """
    Generate performance extrapolation plots with contour lines and color-coded data sources
    """
    _fig, ax = plt.subplots(figsize=(10, 8))

    # Define colors for each data source
    source_colors = {
        "cmod": "#1f77b4",
        "tcv": "#ff7f0e",
        "d3d_lp": "#2ca02c",
        "d3d_hp": "#d62728",
    }

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
        perf_vals = ds[performance_metric].values
        source_vals = ds.coords["ds_source"].values

        # Flatten if needed
        x_vals = np.atleast_1d(x_vals).flatten()
        y_vals = np.atleast_1d(y_vals).flatten()
        perf_vals = np.atleast_1d(perf_vals).flatten()

        # Handle ds_source - could be array or single value
        if np.isscalar(source_vals) or source_vals.size == 1:
            source_vals = np.full_like(x_vals, source_vals, dtype=object)
        else:
            source_vals = np.atleast_1d(source_vals).flatten()

        all_x.extend(x_vals)
        all_y.extend(y_vals)
        all_performance.extend(perf_vals)
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
    all_performance = all_performance[valid_mask]
    all_sources = all_sources[valid_mask]
    all_dataset_labels = all_dataset_labels[valid_mask]

    # Create grid for contour plot
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    )

    # Interpolate dataset labels onto grid to show separation boundaries
    grid_dataset = griddata(
        (all_x, all_y),
        all_dataset_labels,
        (grid_x, grid_y),
        method="nearest",  # Use nearest neighbor to preserve discrete labels
    )

    # Shade regions by dataset with light colors
    n_datasets = len(ds_list)
    if n_datasets > 1:
        # Define light colors for shading each dataset region
        dataset_cmap = plt.cm.Pastel1

        # Create filled contours for shading
        _contourf = ax.contourf(
            grid_x,
            grid_y,
            grid_dataset,
            levels=np.arange(-0.5, n_datasets, 1),
            cmap=dataset_cmap,
            alpha=0.3,
        )

        # Draw boundary lines
        boundary_levels = [i + 0.5 for i in range(n_datasets - 1)]
        ax.contour(
            grid_x,
            grid_y,
            grid_dataset,
            levels=boundary_levels,
            colors="gray",
            alpha=0.6,
            linewidths=2,
        )

        # Add dataset labels to legend using proxy artists
        from matplotlib.patches import Patch

        dataset_handles = [
            Patch(
                facecolor=dataset_cmap(i), alpha=0.3, edgecolor="gray", label=labels[i]
            )
            for i in range(n_datasets)
        ]
    else:
        dataset_handles = []

    # Plot scatter points colored by data source
    source_handles = []
    for source in np.unique(all_sources):
        mask = all_sources == source
        scatter = ax.scatter(
            all_x[mask],
            all_y[mask],
            c=source_colors.get(source, "black"),
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
        from matplotlib.lines import Line2D

        separator = Line2D([0], [0], color="none", label="")

        all_handles = [*dataset_handles, separator, *source_handles]
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
            loc="upper right",
            framealpha=0.9,
        )
    else:
        ax.legend(title="Data Source")

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
