"""Case-comparison figures for transport transfer study results.

Same layout as the power balance study figures: a 2x2 grid with one row per
error kind (absolute / relative) and one column per error domain (per-shot
time-integrated / per-timeslice), the number of target-device shots on a
semilog x axis, one line per member of the comparison, one figure per
combination of the remaining case dimensions. The grid, series, and finalize
machinery is imported from the power balance plotting module; only the axis
vocabulary (no data_normalization, plus geometry_builder and torax_state) is
transport-specific.

Consumes the per-case scalar summary dataset written by
TransportStudy.collect_results (collected_results.nc): dims case_idx, coords
model_type / training_data / domain_adaptation / freeze_submodules /
geometry_builder / torax_state / num_target_shots, data vars err_E_D_S.
"""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loguru import logger

from transport_study.power_balance_transfer.plotting import (
    DA_COLORS,
    DA_LABELS,
    DOMAIN_NAMES,
    METRIC_NAMES,
    check_results_ds,
    coord_values,
    finalize_grid,
    grid_figure,
    mask_select,
    plot_series,
)

MODEL_COLORS = {
    "transformer": "#ffb347",
    "sciml": "#0095ff",
    "torax-constant": "#c0c0c0",
    "torax-cgm": "#8dff36",
    "torax-gyrobohm": "#ff60ec",
    "torax-qlknn": "#ff4d4d",
}

MODEL_LABELS = {
    "transformer": "Transformer",
    "sciml": "SciML",
    "torax-constant": "TORAX constant",
    "torax-cgm": "TORAX CGM",
    "torax-gyrobohm": "TORAX Bohm-GyroBohm",
    "torax-qlknn": "TORAX QLKNN",
}

# Submodule prereq cases predict Wtot / P_oh / P_rad or timeslice profiles,
# not profile evolution, so their errors are not comparable to the main
# models and are excluded from every comparison
SUBMODULE_MODEL_TYPES = ("power_balance", "profile", "p_oh", "p_rad")

# Every case-grid field a figure either compares along or holds fixed
_GRID_FIELDS = ("model_type", "training_data", "domain_adaptation", "freeze_submodules", "geometry_builder", "torax_state")

# Filename token per fixed field, mirroring the case-string vocabulary
_FIELD_TOKENS = {
    "model_type": "{}",
    "training_data": "td_{}",
    "domain_adaptation": "da_{}",
    "freeze_submodules": "freeze_{}",
    "geometry_builder": "geom_{}",
    "torax_state": "tstate_{}",
}


def _model_types(ds: xr.Dataset) -> list:
    return [mt for mt in coord_values(ds, "model_type") if mt not in SUBMODULE_MODEL_TYPES]


def _field_mask(ds: xr.Dataset, field: str, value) -> np.ndarray:
    """Boolean case_idx mask for field == value.

    Case coords shared by every case (e.g. freeze_submodules with a single
    configured option) are scalar in the collected file, so the comparison is
    broadcast back to the case_idx length before indexing."""
    matches = np.atleast_1d(ds[field].values) == value
    return np.broadcast_to(matches, (ds.sizes["case_idx"],))


def _comparison_figures(
    results_ds: xr.Dataset,
    figure_dir: Path,
    family: str,
    series_field: str,
    series_values,
    series_colors: dict,
    series_labels: dict,
    title_prefix: str,
):
    """One line per series_field value, one figure per combination of the
    remaining grid fields (submodule model types excluded throughout)."""
    if not check_results_ds(results_ds, family):
        return
    out_dir = Path(figure_dir) / "comparison" / family
    fixed_fields = [f for f in _GRID_FIELDS if f != series_field]
    fixed_values = []
    for field in fixed_fields:
        values = _model_types(results_ds) if field == "model_type" else coord_values(results_ds, field)
        fixed_values.append(values)

    for combo in product(*fixed_values):
        mask = np.ones(results_ds.sizes["case_idx"], dtype=bool)
        for field, value in zip(fixed_fields, combo, strict=True):
            mask = mask & _field_mask(results_ds, field, value)
        sub = mask_select(results_ds, mask)
        if sub.sizes.get("case_idx", 0) == 0:
            continue
        fig, axes = grid_figure()
        drew = False
        for series_value in series_values:
            if series_value not in coord_values(sub, series_field):
                continue
            series_sub = mask_select(sub, _field_mask(sub, series_field, series_value))
            color = series_colors.get(series_value, "white")
            label = series_labels.get(series_value, str(series_value))
            for row, metric in enumerate(METRIC_NAMES):
                for col, domain in enumerate(DOMAIN_NAMES):
                    drew |= plot_series(axes[row, col], series_sub, metric, domain, color, label)
        if not drew:
            plt.close(fig)
            continue
        fixed_desc = " / ".join(f"{field}: {value}" for field, value in zip(fixed_fields, combo, strict=True))
        filename = ".".join(_FIELD_TOKENS[field].format(value) for field, value in zip(fixed_fields, combo, strict=True))
        finalize_grid(fig, axes, sub, f"{title_prefix} - {fixed_desc}", out_dir / f"{filename}.png")
    logger.info(f"Saved {family} figures to {out_dir}")


def model_comparison(results_ds: xr.Dataset, figure_dir: Path):
    """One line per model type, one figure per combination of the remaining axes."""
    _comparison_figures(
        results_ds,
        figure_dir,
        family="model_comparison",
        series_field="model_type",
        series_values=_model_types(results_ds) if results_ds.data_vars else [],
        series_colors=MODEL_COLORS,
        series_labels=MODEL_LABELS,
        title_prefix="Model comparison",
    )


def training_dataset_comparison(results_ds: xr.Dataset, figure_dir: Path):
    """One line per training dataset, one figure per combination of the remaining axes."""
    if not results_ds.data_vars:
        logger.warning("No collected results available, skipping training dataset comparison figures")
        return
    training_datasets = coord_values(results_ds, "training_data")
    cmap = plt.colormaps["tab10"].resampled(max(len(training_datasets), 1))
    td_colors = {td: cmap(i) for i, td in enumerate(training_datasets)}
    _comparison_figures(
        results_ds,
        figure_dir,
        family="training_dataset_comparison",
        series_field="training_data",
        series_values=training_datasets,
        series_colors=td_colors,
        series_labels={td: str(td) for td in training_datasets},
        title_prefix="Training dataset comparison",
    )


def domain_adaptation_comparison(results_ds: xr.Dataset, figure_dir: Path):
    """One line per domain adaptation method, one figure per combination of the
    remaining axes.

    The no-adaptation baseline only exists at num_target_shots = 0 for
    non-exnihilo training data, so it typically shows up as a single point
    rather than a trend.
    """
    _comparison_figures(
        results_ds,
        figure_dir,
        family="domain_adaptation_comparison",
        series_field="domain_adaptation",
        series_values=list(DA_COLORS) if results_ds.data_vars else [],
        series_colors=DA_COLORS,
        series_labels=DA_LABELS,
        title_prefix="Domain adaptation comparison",
    )
