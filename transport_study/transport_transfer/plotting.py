"""Case-comparison figures of transport transfer study results (see orchestration.comparison_figures).

A grid with one row per metric (combined chi, relative error, diverged fraction, see study_metrics)
and one column per shot stage (all, rampup, flattop, flattop ohmic, flattop aux, rampdown),
each line the top-K mean with the top-K std as its error bar.
Consumes the stage-resolved collected metrics (collected_metrics.nc, dims case_idx x stage),
over the transport case-grid fields (no data_normalization, plus geometry_builder, torax_state and multiobjective).
"""

from transport_study.modules.transport_predictor.module import SUBMODULE_MODEL_TYPES
from transport_study.orchestration.comparison_figures import (
    DA_COLORS,
    DA_LABELS,
    MULTIOBJECTIVE_COLORS,
    MULTIOBJECTIVE_LABELS,
    ORDER_COLORS,
    ORDER_LABELS,
    STAGE_LABELS,
    TD_COLORS,
    TD_LABELS,
    ComparisonFamily,
    ComparisonLayout,
    stage_series_stats,
)
from transport_study.orchestration.stages import STAGE_AGG_NAMES

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

# Rows of the comparison grid
METRIC_LABELS = {
    "combined": "Chi (value + gradient)",
    "rel": "Relative error",
    "diverged": "Diverged fraction",
}

LAYOUT = ComparisonLayout(
    row_labels=METRIC_LABELS,
    col_labels={stage: STAGE_LABELS[stage] for stage in STAGE_AGG_NAMES},
    cell_size=(3.4, 2.9),
    series_stats=stage_series_stats,
    grid_fields=(
        "model_type",
        "training_data",
        "domain_adaptation",
        "freeze_submodules",
        "geometry_builder",
        "torax_state",
        "multiobjective",
        "target_shot_order",
    ),
    field_tokens={
        "model_type": "{}",
        "training_data": "td_{}",
        "domain_adaptation": "da_{}",
        "freeze_submodules": "freeze_{}",
        "geometry_builder": "geom_{}",
        "torax_state": "tstate_{}",
        "multiobjective": "mo_{}",
        "target_shot_order": "order_{}",
    },
    value_labels={
        "model_type": MODEL_LABELS,
        "training_data": TD_LABELS,
        "domain_adaptation": DA_LABELS,
        "multiobjective": MULTIOBJECTIVE_LABELS,
        "target_shot_order": ORDER_LABELS,
    },
    excluded_model_types=SUBMODULE_MODEL_TYPES,
)

COMPARISON_FAMILIES = (
    ComparisonFamily("training_dataset_comparison", "training_data", "Training dataset comparison", TD_COLORS, include_exnihilo=True),
    ComparisonFamily("model_comparison", "model_type", "Model comparison", MODEL_COLORS),
    ComparisonFamily("domain_adaptation_comparison", "domain_adaptation", "Domain adaptation comparison", DA_COLORS),
    # Without target shots every case takes the base order, so the other orders start at 1 target shot
    ComparisonFamily("target_shot_order_comparison", "target_shot_order", "Target shot order comparison", ORDER_COLORS, min_series=2),
    # Only renders in a study that varies it (hs1_transport_multiobjective)
    ComparisonFamily("multiobjective_comparison", "multiobjective", "Multiobjective comparison", MULTIOBJECTIVE_COLORS, min_series=2),
)
