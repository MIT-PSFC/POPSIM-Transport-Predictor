from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from loguru import logger
from matplotlib import patches
from scipy.spatial import ConvexHull

from transport_study.config import config
from transport_study.orchestration.organize_data import (
    TrainingData,
    add_performance,
    concat_with_nan_padding,
    get_ds,
    get_train_test_datasets,
    get_train_val_datasets,
    normalize_domain,
)
from transport_study.plot_style import BACKGROUND_COLOR, FACE_COLOR, TEXT_COLOR

TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 16

# Marker shape/size per dataset split. Shared across all devices so device is encoded by
# color and split (train/val/test) is encoded by marker.
DATASET_SHAPES = {
    "train": ("s", 120),
    "val": ("o", 160),
    "test": ("*", 240),
}


def _source_devices() -> list[str]:
    """All non-target devices that have a dataset path, in deterministic order."""
    target = config.target_device
    return [d for d in sorted(config.dataset_paths) if d != target]


def _all_devices() -> list[str]:
    """Source devices followed by the target device (if it has a dataset path)."""
    devices = list(_source_devices())
    target = config.target_device
    if target and config.dataset_paths.get(target):
        devices.append(target)
    return devices


def _device_colors() -> dict[str, str]:
    """Assign each device a distinct color, deterministic in sorted device order.

    Device-agnostic: works for any number of source datasets plus the target.
    """
    devices = sorted(_all_devices())
    cmap = plt.get_cmap("tab10" if len(devices) <= 10 else "hsv")
    if len(devices) <= 10:
        return {d: cmap(i) for i, d in enumerate(devices)}
    return {d: cmap(frac) for d, frac in zip(devices, np.linspace(0.0, 0.9, len(devices)), strict=True)}


def _td(devices: list[str]) -> TrainingData:
    """Build a TrainingData from an explicit device list (no underscore splitting)."""
    return TrainingData(sources_unsorted=list(devices))


def performance_extrapolation_plot(
    save_path: Path | str,
    ds_list: list[xr.Dataset],
    ds_type_list: list[str],
    source_colors: dict[str, str],
    x_var: str = "Ip_MA",
    y_var: str = "Wtot_MJ",
):
    """
    Scatter each shot in x_var-y_var (performance) space.

    Color encodes the data source device, marker encodes the dataset split
    (train/val/test). Uses the per-shot p95 values added by `add_performance`.

    Args:
        save_path: Path to save the generated figure.
        ds_list: xarray Datasets to plot. Each has a coordinate "ds_source".
        ds_type_list: Dataset split for each dataset in ds_list ("train"/"val"/"test").
        source_colors: Mapping from device name to plot color.
        x_var: Variable on the x-axis (default "Ip_MA").
        y_var: Variable on the y-axis (default "Wtot_MJ").
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(FACE_COLOR)

    all_x = []
    all_y = []
    all_sources = []
    all_ds_types = []

    for dataset_idx, ds in enumerate(ds_list):
        x_vals = np.atleast_1d(ds[f"{x_var}_p95"].values).flatten()
        y_vals = np.atleast_1d(ds[f"{y_var}_p95"].values).flatten()
        source_vals = ds.coords["ds_source"].values

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
                c=[source_colors.get(source, "black")],
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
        legend = ax.legend(title="Split / Data Source", loc="upper left", framealpha=0.9)
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
    source_colors: dict[str, str],
):
    """Make a 2x2 grid of scatter plots showing where each device lives in different
    input-variable spaces, colored by data source, with IQR ellipse and convex hull."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    # Median value per shot (across time_idx) for the scatter points.
    for var in {var for group in var_groups for var in group}:
        ds[f"{var}_median"] = ds[var].median(dim="time_idx", skipna=True)

    for ax in axes.flatten():
        ax.set_facecolor(FACE_COLOR)

    for i, (x_var, y_var) in enumerate(var_groups):
        x_vals = np.atleast_1d(ds[f"{x_var}_median"].values).flatten()
        y_vals = np.atleast_1d(ds[f"{y_var}_median"].values).flatten()
        source_vals = ds.coords["ds_source"].values

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
            color = source_colors.get(source, "black")
            ax.scatter(
                x_vals[device_mask],
                y_vals[device_mask],
                c=[color],
                label=source,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
            )

            # Ellipse (IQR) and convex hull computed from the full timeseries, not just medians.
            device_mask_xr = ds.coords["ds_source"] == source
            x_all = ds[x_var].where(device_mask_xr, drop=True).values.flatten()
            y_all = ds[y_var].where(device_mask_xr, drop=True).values.flatten()

            valid_full = ~(np.isnan(x_all) | np.isnan(y_all))
            x_all = x_all[valid_full]
            y_all = y_all[valid_full]

            if len(x_all) == 0 or len(y_all) == 0:
                continue

            x_median_full = np.median(x_all)
            y_median_full = np.median(y_all)
            x_q75, x_q25 = np.percentile(x_all, [75, 25])
            y_q75, y_q25 = np.percentile(y_all, [75, 25])
            x_iqr = x_q75 - x_q25
            y_iqr = y_q75 - y_q25

            if x_iqr > 0 and y_iqr > 0:
                ellipse = patches.Ellipse(
                    (x_median_full, y_median_full),
                    width=x_iqr,
                    height=y_iqr,
                    fill=False,
                    edgecolor=color,
                    alpha=0.5,
                    linewidth=2,
                    linestyle="--",
                )
                ax.add_patch(ellipse)

            if len(x_all) >= 3:
                try:
                    points = np.column_stack((x_all, y_all))
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])
                    ax.plot(
                        hull_points[:, 0],
                        hull_points[:, 1],
                        color=color,
                        alpha=0.3,
                        linewidth=1,
                        linestyle="-",
                    )
                    hull_polygon = patches.Polygon(
                        hull_points[:-1],
                        closed=True,
                        facecolor=color,
                        alpha=0.05,
                        edgecolor="none",
                    )
                    ax.add_patch(hull_polygon)
                except Exception:
                    # Skip hull if computation fails (e.g. collinear points)
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

    fig.suptitle(title, fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _combined_raw_dataset(study_type: str) -> tuple[xr.Dataset, str]:
    """Concatenate every device (sources + target) into one dataset with a per-shot
    ds_source coordinate and performance/p95 metrics added."""
    datasets = []
    episode_coord = None
    for device in _all_devices():
        ds, episode_coord = get_ds(device, study_type=study_type)
        ds = add_performance(ds, episode_coord)
        # Profile-transfer datasets have no aux-power signal, but z_score/coral
        # normalization iterate over it unconditionally. Zero-fill so they run
        # (a no-op for power-balance datasets, which always carry P_aux_MW).
        if "P_aux_MW" not in ds:
            ds["P_aux_MW"] = xr.zeros_like(ds["Ip_MA"])
        ds = ds.assign_coords(ds_source=device)
        datasets.append(ds)

    if not datasets:
        raise ValueError("No dataset paths provided in config, cannot build domain-overlap dataset.")

    if len(datasets) == 1:
        return datasets[0], episode_coord
    return concat_with_nan_padding(datasets, concat_dim=episode_coord), episode_coord


class DataVisualizationBase:
    """Device-agnostic visualizations of the datasets used in training and testing.

    Adapts to an arbitrary number of source datasets and a single distinct target
    dataset, both read from `config.dataset_paths` / `config.target_device`.
    Subclasses set STUDY_TYPE and VAR_GROUPS (input-variable pairs to plot per
    normalization method).
    """

    STUDY_TYPE: ClassVar[str]
    VAR_GROUPS: ClassVar[dict[str, list[list[str]]]]

    @classmethod
    def performance_extrapolation(cls, figure_dir: Path | str):
        """Plot where each dataset lands in performance (Ip_MA-Wtot_MJ) space.

        Generates:
        1. One plot per source device (train/val).
        2. One plot with all source devices combined (train/val), if >1 source.
        3. One plot putting the target device (test split) in context of all source
           devices (train split), if a target dataset is available.
        """
        save_dir = Path(figure_dir) / "data_visualization" / "performance_extrapolation"
        colors = _device_colors()
        sources = _source_devices()
        target = config.target_device

        # 1. Each source device on its own.
        for source in sources:
            fig_path = save_dir / f"{source}_performance_extrapolation.png"
            if fig_path.exists():
                continue
            train_ds, val_ds = get_train_val_datasets(training_data=_td([source]), study_type=cls.STUDY_TYPE)
            performance_extrapolation_plot(
                save_path=fig_path,
                ds_list=[train_ds, val_ds],
                ds_type_list=["train", "val"],
                source_colors=colors,
            )

        # 2. All source devices combined.
        if len(sources) > 1:
            fig_path = save_dir / f"{'_'.join(sources)}_performance_extrapolation.png"
            if not fig_path.exists():
                train_ds, val_ds = get_train_val_datasets(training_data=_td(sources), study_type=cls.STUDY_TYPE)
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, val_ds],
                    ds_type_list=["train", "val"],
                    source_colors=colors,
                )

        # 3. Target device (test set) in context of all source training data.
        if target and config.dataset_paths.get(target):
            if not sources:
                logger.warning("No source devices available, skipping target-in-context performance plot.")
                return
            fig_path = save_dir / f"target_{target}_in_context_performance_extrapolation.png"
            if not fig_path.exists():
                # num_target_shots=0 keeps the target purely in the test split so it reads
                # as one distinct target dataset against the source training data.
                train_ds, test_ds = get_train_test_datasets(
                    training_data=_td(sources),
                    domain_adaptation="addition",
                    num_target_shots=0,
                    target_test_set_size=config.target_test_set_size,
                    study_type=cls.STUDY_TYPE,
                )
                performance_extrapolation_plot(
                    save_path=fig_path,
                    ds_list=[train_ds, test_ds],
                    ds_type_list=["train", "test"],
                    source_colors=colors,
                )

    @classmethod
    def domain_overlap(cls, figure_dir: Path | str):
        """Show how the input parameter space of every device overlaps, per normalization.

        Unlike performance extrapolation (where extrapolation is unavoidable in real
        units), here more overlap is better - normalization should align the devices
        for transfer learning.
        """
        save_dir = Path(figure_dir) / "data_visualization" / "domain_overlap"
        colors = _device_colors()

        methods = ["raw", "physics", "z_score", "coral", "physics-coral"]
        if all((save_dir / f"domain_overlap_{method}.png").exists() for method in methods):
            return

        combined, _ = _combined_raw_dataset(cls.STUDY_TYPE)

        for method in methods:
            fig_path = save_dir / f"domain_overlap_{method}.png"
            if fig_path.exists():
                continue
            try:
                ds_norm, _ = normalize_domain(combined.copy(deep=True), method=method)
                domain_plot(
                    ds=ds_norm,
                    var_groups=cls.VAR_GROUPS[method],
                    title=f"Domain overlap ({method})",
                    save_path=fig_path,
                    source_colors=colors,
                )
            except Exception as exc:
                logger.warning(f"Skipping domain overlap plot for '{method}' normalization: {exc}")
