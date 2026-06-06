from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from loguru import logger
from matplotlib import patches
from scipy.spatial import ConvexHull

from transport_study.orchestration.organize_data import (
    TrainingData,
    dataset_config,
    get_train_test_datasets,
    get_train_val_datasets,
)


def _td(s: str) -> TrainingData:
    """Parse string like 'cmod_tcv' into TrainingData."""
    return TrainingData(sources_unsorted=s.split("_"))


BACKGROUND_COLOR = "#2F2F2F"
FACE_COLOR = "#1A1A1A"
TEXT_COLOR = "white"

TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 16

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


def performance_extrapolation_plot(
    save_path: Path | str,
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
    fig, ax = plt.subplots(figsize=(8, 6))
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

        ds_type = ds_type_list[dataset_idx] if dataset_idx < len(ds_type_list) else "unknown"

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
        legend = ax.legend(title="Dataset / Data Source", loc="upper left", framealpha=0.9)
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
    save_path: Path | str,
):
    """Make a 2x2 grid of scatter plots showing the domain of each dataset in different variable spaces, colored by data source."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
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
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            framealpha=0.9,
            fontsize=LEGEND_FONTSIZE * 1.2,
            title_fontsize=LEGEND_FONTSIZE * 1.2,
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


class DataVisualization:
    """
    Visualizations of the datasets used in training and testing.
    """

    def _get_largest_dataset():
        # Determine the biggest dataset we can use so that we only need to make one plot
        # for the domain overlap visualization. Want to only do this once since it's expensive.
        if (
            dataset_config.dataset_paths.get("cmod")
            and dataset_config.dataset_paths.get("tcv")
            and dataset_config.dataset_paths.get("d3d_lp")
        ):
            training_data = _td("cmod_tcv_d3d_lp")
        elif dataset_config.dataset_paths.get("cmod") and dataset_config.dataset_paths.get("tcv"):
            training_data = _td("cmod_tcv")
        elif dataset_config.dataset_paths.get("cmod"):
            training_data = _td("cmod")
        elif dataset_config.dataset_paths.get("tcv"):
            training_data = _td("tcv")
        elif dataset_config.dataset_paths.get("d3d_lp"):
            training_data = _td("d3d_lp")
        else:
            raise ValueError("No dataset paths provided in config, cannot determine largest dataset case for domain overlap plot.")

        return training_data

    @staticmethod
    def performance_extrapolation(
        figure_dir: Path | str,
    ):
        """
        Performance is ip**2 + Wtot_MJ**2

        With all data present this creates the following figures:
        1. Performance extrapolation for C-Mod
        2. Performance extrapolation for TCV
        3. Performance extrapolation for DIII-D low-performance shots
        4. Performance extrapolation for C-Mod + TCV
        5. Performance extrapolation for C-Mod + TCV + DIII-D low-performance shots
        6. Showing there is no overlap in parameter space between the DIII-D low-performance shots, high-performance shots used in training, and high-performance shots used in testing
        7. Put DIII-D high-performance shots in context of all training data
        """

        save_dir = Path(figure_dir) / "data_visualization" / "performance_extrapolation"

        # C-Mod
        if dataset_config.dataset_paths.get("cmod"):
            fig_path = save_dir / "cmod_performance_extrapolation.png"
            if not fig_path.exists():
                train_ds, val_ds = get_train_val_datasets(
                    training_data=_td("cmod"),
                    data_normalization="raw",
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, val_ds],
                    ds_type_list=["train", "val"],
                    x_var="Ip_MA",
                    y_var="Wtot_MJ",
                )
        else:
            logger.warning("C-Mod dataset path not provided, skipping C-Mod figures.")

        # TCV
        if dataset_config.dataset_paths.get("tcv"):
            fig_path = save_dir / "tcv_performance_extrapolation.png"
            if not fig_path.exists():
                train_ds, val_ds = get_train_val_datasets(
                    training_data=_td("tcv"),
                    data_normalization="raw",
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, val_ds],
                    ds_type_list=["train", "val"],
                    x_var="Ip_MA",
                    y_var="Wtot_MJ",
                )
        else:
            logger.warning("TCV dataset path not provided, skipping TCV figures.")

        # C-Mod + TCV
        if dataset_config.dataset_paths.get("tcv") and dataset_config.dataset_paths.get("cmod"):
            fig_path = save_dir / "cmod_tcv_performance_extrapolation.png"
            if not fig_path.exists():
                train_ds, val_ds = get_train_val_datasets(
                    training_data=_td("cmod_tcv"),
                    data_normalization="raw",
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, val_ds],
                    ds_type_list=["train", "val"],
                    x_var="Ip_MA",
                    y_var="Wtot_MJ",
                )
        else:
            logger.warning("TCV or C-Mod dataset path not provided, skipping combined C-Mod + TCV figures.")

        # DIII-D low-performance
        if dataset_config.dataset_paths.get("d3d_lp"):
            fig_path = save_dir / "d3d_lp_performance_extrapolation.png"
            if not fig_path.exists():
                train_ds, val_ds = get_train_val_datasets(
                    training_data=_td("d3d_lp"),
                    data_normalization="raw",
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, val_ds],
                    ds_type_list=["train", "val"],
                    x_var="Ip_MA",
                    y_var="Wtot_MJ",
                )
        else:
            logger.warning("DIII-D low-performance dataset path not provided, skipping DIII-D low-performance figures.")

        # C-Mod + TCV + DIII-D low-performance
        if (
            dataset_config.dataset_paths.get("cmod")
            and dataset_config.dataset_paths.get("tcv")
            and dataset_config.dataset_paths.get("d3d_lp")
        ):
            fig_path = save_dir / "cmod_tcv_d3d_lp_performance_extrapolation.png"
            if not fig_path.exists():
                train_ds, val_ds = get_train_val_datasets(
                    training_data=_td("cmod_tcv_d3d_lp"),
                    data_normalization="raw",
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, val_ds],
                    ds_type_list=["train", "val"],
                    x_var="Ip_MA",
                    y_var="Wtot_MJ",
                )
        else:
            logger.warning(
                "C-Mod, TCV, or DIII-D low-performance dataset path not provided, skipping combined C-Mod + TCV + DIII-D low-performance figures."
            )

        # DIII-D performance overlap
        if dataset_config.dataset_paths.get(dataset_config.target_device) and dataset_config.dataset_paths.get("d3d_lp"):
            fig_path = save_dir / "d3d_performance_overlap.png"
            if not fig_path.exists():
                train_ds, test_ds = get_train_test_datasets(
                    training_data=_td("d3d_lp"),
                    num_hp_shots=33,
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, test_ds],
                    ds_type_list=["train", "test"],
                    x_var="Ip_MA",
                    y_var="Wtot_MJ",
                )

        # DIII-D high-performance in context of available training data
        context_dict = {
            "cmod": {
                "training_data": _td("cmod"),
                "condition": dataset_config.dataset_paths.get("cmod"),
                "fig_name": "d3d_hp_in_context_cmod_performance_extrapolation.png",
            },
            "tcv": {
                "training_data": _td("tcv"),
                "condition": dataset_config.dataset_paths.get("tcv"),
                "fig_name": "d3d_hp_in_context_tcv_performance_extrapolation.png",
            },
            "d3d_lp": {
                "training_data": _td("d3d_lp"),
                "condition": dataset_config.dataset_paths.get("d3d_lp"),
                "fig_name": "d3d_hp_in_context_d3d_lp_performance_extrapolation.png",
            },
            "cmod_tcv": {
                "training_data": _td("cmod_tcv"),
                "condition": dataset_config.dataset_paths.get("cmod") and dataset_config.dataset_paths.get("tcv"),
                "fig_name": "d3d_hp_in_context_cmod_tcv_performance_extrapolation.png",
            },
            "cmod_tcv_d3d_lp": {
                "training_data": _td("cmod_tcv_d3d_lp"),
                "condition": dataset_config.dataset_paths.get("cmod")
                and dataset_config.dataset_paths.get("tcv")
                and dataset_config.dataset_paths.get("d3d_lp"),
                "fig_name": "d3d_hp_in_context_cmod_tcv_d3d_lp_performance_extrapolation.png",
            },
        }

        for context_info in context_dict.values():
            if context_info["condition"] and dataset_config.dataset_paths.get(dataset_config.target_device):
                fig_path = save_dir / context_info["fig_name"]
                if not fig_path.exists():
                    train_ds, test_ds = get_train_test_datasets(
                        training_data=context_info["training_data"],
                        data_normalization="raw",
                        domain_adaptation="mixing",
                        num_hp_shots=33,
                        hp_test_set_size=60,
                    )
                    performance_extrapolation_plot(
                        save_path=fig_path,
                        ds_list=[train_ds, test_ds],
                        ds_type_list=["train", "test"],
                        x_var="Ip_MA",
                        y_var="Wtot_MJ",
                    )

    @staticmethod
    def domain_overlap(
        figure_dir: Path | str,
    ):
        """
        Compare different data preparation cases to how the parameter space overlaps.
        This is different from the performance extrapolation plots because here it is desirable to have a lot of overlap.
        While we are ALWAYS extrapolating in real units (Ip and Wtot, things that WILL break the device)
        first normalizing the data should help with transfer learning.

        Basically, this normalization doesn't impact the transfer learning, because the dataset is being split into train and val/test beforehand.
        """

        training_data = DataVisualization._get_largest_dataset()

        for method in ["raw", "physics", "z_score", "coral"]:
            if method == "raw":
                var_groups = [
                    ["Ip_MA", "Wtot_MJ"],
                    ["R0", "a_minor"],
                    ["ne20_line_avg", "B0"],
                    ["P_aux_MW", "kappa"],
                ]
            elif method == "physics":
                var_groups = [
                    ["Ip_MA", "beta"],
                    ["q_star", "epsilon"],
                    ["f_G", "aB0"],
                    ["surface_power_density", "kappa"],
                ]
            elif method == "z_score":
                var_groups = [
                    ["Ip_MA_z", "Wtot_MJ_z"],
                    ["R0_z", "a_minor_z"],
                    ["ne20_line_avg_z", "B0_z"],
                    ["P_aux_MW_z", "kappa_z"],
                ]
            elif method == "coral":
                var_groups = [
                    ["Ip_MA_coral", "Wtot_MJ_coral"],
                    ["R0_coral", "a_minor_coral"],
                    ["ne20_line_avg_coral", "B0_coral"],
                    ["P_aux_MW_coral", "kappa_coral"],
                ]
            else:
                raise ValueError(f"Unknown normalization method '{method}' specified.")

            if dataset_config.dataset_paths.get(dataset_config.target_device):
                ds, _ = get_train_test_datasets(
                    training_data=training_data,
                    data_normalization=method,
                    domain_adaptation="mixing",
                    num_hp_shots=-1,
                    hp_test_set_size=60,
                )
            else:
                ds, _ = get_train_val_datasets(training_data=training_data, data_normalization=method)

            fig_path = Path(figure_dir) / "domain_overlap" / f"{training_data}_domain_overlap_{method}.png"
            domain_plot(
                ds=ds,
                var_groups=var_groups,
                title=f"{training_data} domain overlap {method}",
                save_path=fig_path,
            )
