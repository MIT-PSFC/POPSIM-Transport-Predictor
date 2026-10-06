"""Case-comparison figures of transport transfer study results (see orchestration.comparison_figures).

The power balance study's 2x2 error grid over the transport case-grid fields
(no data_normalization, plus geometry_builder and torax_state).
Consumes the per-case scalar summary of TransportStudy.collect_results (collected_results.nc).
"""

from transport_study.modules.transport_predictor.module import SUBMODULE_MODEL_TYPES
from transport_study.orchestration.comparison_figures import (
    DA_COLORS,
    DA_LABELS,
    ORDER_COLORS,
    ORDER_LABELS,
    ComparisonFamily,
    ComparisonLayout,
)
from transport_study.power_balance_transfer.plotting import (
    DOMAIN_LABELS,
    METRIC_LABELS,
    summary_series_stats,
)

MODEL_COLORS = {
    "transformer": "#ffb347",
    "sciml": "#0095ff",
    "torax-constant": "#c0c0c0",
    "torax-gyrobohm": "#ff60ec",
    "torax-qlknn": "#ff4d4d",
}

MODEL_LABELS = {
    "transformer": "Transformer",
    "sciml": "SciML",
    "torax-constant": "TORAX constant",
    "torax-gyrobohm": "TORAX Bohm-GyroBohm",
    "torax-qlknn": "TORAX QLKNN",
}

LAYOUT = ComparisonLayout(
    row_labels=METRIC_LABELS,
    col_labels=DOMAIN_LABELS,
    cell_size=(4.6, 3.4),
    series_stats=summary_series_stats,
    grid_fields=(
        "model_type",
        "training_data",
        "domain_adaptation",
        "freeze_submodules",
        "geometry_builder",
        "torax_state",
        "target_shot_order",
    ),
    field_tokens={
        "model_type": "{}",
        "training_data": "td_{}",
        "domain_adaptation": "da_{}",
        "freeze_submodules": "freeze_{}",
        "geometry_builder": "geom_{}",
        "torax_state": "tstate_{}",
        "target_shot_order": "order_{}",
    },
    value_labels={"model_type": MODEL_LABELS, "domain_adaptation": DA_LABELS, "target_shot_order": ORDER_LABELS},
    excluded_model_types=SUBMODULE_MODEL_TYPES,
)

COMPARISON_FAMILIES = (
    ComparisonFamily("training_dataset_comparison", "training_data", "Training dataset comparison"),
    ComparisonFamily("model_comparison", "model_type", "Model comparison", MODEL_COLORS),
    ComparisonFamily("domain_adaptation_comparison", "domain_adaptation", "Domain adaptation comparison", DA_COLORS),
    # Without target shots every case takes the base order, so the other orders start at 1 target shot
    ComparisonFamily("target_shot_order_comparison", "target_shot_order", "Target shot order comparison", ORDER_COLORS, min_series=2),
)
