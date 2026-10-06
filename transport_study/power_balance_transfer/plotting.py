"""Case-comparison figures of power balance transfer study results (see orchestration.comparison_figures).

A 2x2 grid with one row per error kind (absolute / relative)
and one column per error domain (per-shot time-integrated / per-timeslice),
each line the top-K mean of the domain mean with the top-K std as its error bar.
Consumes the per-case scalar summary of PowerBalanceStudy.collect_results (collected_results.nc):
dims case_idx, the case-grid coords and the err_E_D_S data vars.
"""

import numpy as np
import xarray as xr

from transport_study.modules.power_balance.module import SUBMODULE_MODEL_TYPES
from transport_study.orchestration.comparison_figures import (
    DA_COLORS,
    DA_LABELS,
    FREEZE_COLORS,
    FREEZE_LABELS,
    MULTIOBJECTIVE_COLORS,
    MULTIOBJECTIVE_LABELS,
    NORM_COLORS,
    NORM_LABELS,
    ORDER_COLORS,
    ORDER_LABELS,
    TD_COLORS,
    TD_LABELS,
    ComparisonFamily,
    ComparisonLayout,
)

# Rows of the comparison grid: error kind
METRIC_LABELS = {
    "err_abs": "Absolute error",
    "err_rel": "Relative error",
}

# Columns of the comparison grid: error domain
DOMAIN_LABELS = {
    "shot": "Per shot (time-integrated)",
    "ts": "Per timeslice",
}

MODEL_COLORS = {
    "sciml-taue-scalinglaw": "#8dff36",
    "sciml-taue-nn": "#0095ff",
    "mlp": "#ff4d4d",
    "transformer": "#ffb347",
}

MODEL_LABELS = {
    "sciml-taue-scalinglaw": "SciML (tau_e scaling law)",
    "sciml-taue-nn": "SciML (tau_e NN)",
    "mlp": "MLP",
    "transformer": "Transformer",
}


def summary_series_stats(ds: xr.Dataset, metric: str, domain: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-case top-K mean and top-K std of one error kind's mean over one domain of a per-case scalar summary."""
    return ds[f"{metric}_{domain}_mean"].values, ds[f"{metric}_{domain}_mean_ckpt_std"].values


LAYOUT = ComparisonLayout(
    row_labels=METRIC_LABELS,
    col_labels=DOMAIN_LABELS,
    cell_size=(4.6, 3.4),
    series_stats=summary_series_stats,
    grid_fields=(
        "model_type",
        "training_data",
        "data_normalization",
        "domain_adaptation",
        "freeze_submodules",
        "multiobjective",
        "target_shot_order",
    ),
    field_tokens={
        "model_type": "{}",
        "training_data": "td_{}",
        "data_normalization": "norm_{}",
        "domain_adaptation": "da_{}",
        "freeze_submodules": "freeze_{}",
        "multiobjective": "mo_{}",
        "target_shot_order": "order_{}",
    },
    value_labels={
        "model_type": MODEL_LABELS,
        "training_data": TD_LABELS,
        "data_normalization": NORM_LABELS,
        "domain_adaptation": DA_LABELS,
        "freeze_submodules": FREEZE_LABELS,
        "multiobjective": MULTIOBJECTIVE_LABELS,
        "target_shot_order": ORDER_LABELS,
    },
    excluded_model_types=SUBMODULE_MODEL_TYPES,
)

COMPARISON_FAMILIES = (
    ComparisonFamily("training_dataset_comparison", "training_data", "Training dataset comparison", TD_COLORS, include_exnihilo=True),
    ComparisonFamily("model_comparison", "model_type", "Model comparison", MODEL_COLORS),
    ComparisonFamily("data_normalization_comparison", "data_normalization", "Data normalization comparison", NORM_COLORS),
    # The no-adaptation baseline only exists at num_target_shots = 0 for non-exnihilo training data,
    # so it typically shows up as a single point rather than a trend
    ComparisonFamily("domain_adaptation_comparison", "domain_adaptation", "Domain adaptation comparison", DA_COLORS),
    # Without target shots every case takes the base order, so the other orders start at 1 target shot
    ComparisonFamily("target_shot_order_comparison", "target_shot_order", "Target shot order comparison", ORDER_COLORS, min_series=2),
    # Both only render in the child studies that vary them (hs1_pb_freeze, hs1_pb_multiobjective)
    ComparisonFamily("freeze_submodules_comparison", "freeze_submodules", "Frozen vs unfrozen submodules", FREEZE_COLORS, min_series=2),
    ComparisonFamily("multiobjective_comparison", "multiobjective", "Multiobjective comparison", MULTIOBJECTIVE_COLORS, min_series=2),
)
