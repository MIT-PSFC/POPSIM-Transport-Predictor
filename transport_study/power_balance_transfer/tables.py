"""Per-axis comparison tables for power balance transfer study results.

For every case axis (model type, training dataset, data normalization, domain
adaptation, frozen submodules, multiobjective, number of target shots, target shot order)
and every combination of the remaining axes,
one markdown table comparing the cases that differ only along that axis.
Columns combine the stage-resolved TIME-AVERAGED errors from
collected_metrics.nc (per-timeslice means, free of the shot-duration confound
in the raw time-integrated per-shot errors) with the time-integrated
medians from collected_results.nc.
The headline metric also gets its best checkpoint's value and its std over the top-K checkpoints (orchestration.topk_results).

A flat case_stats.csv with one row per case and every column is written next to the tables for ad hoc analysis.
The frame, table and csv writing is the shared orchestration.tables machinery, this module only declares the spec.
"""

from transport_study.modules.power_balance.module import SUBMODULE_MODEL_TYPES
from transport_study.orchestration.tables import ComparisonTableSpec
from transport_study.power_balance_transfer.study_metrics import METRIC_NAMES

CASE_FIELD_ORDER = (
    "model_type",
    "training_data",
    "data_normalization",
    "domain_adaptation",
    "freeze_submodules",
    "multiobjective",
    "num_target_shots",
    "target_shot_order",
)

SPEC = ComparisonTableSpec(
    axis_names=(
        "model_type",
        "training_data",
        "data_normalization",
        "domain_adaptation",
        "freeze_submodules",
        "multiobjective",
        "num_target_shots",
        "target_shot_order",
    ),
    case_field_order=CASE_FIELD_ORDER,
    # Filename tokens per grouping field, mirroring the case-string vocabulary
    field_tokens={
        "model_type": "{}",
        "training_data": "td_{}",
        "data_normalization": "norm_{}",
        "domain_adaptation": "da_{}",
        "freeze_submodules": "freeze_{}",
        "multiobjective": "mo_{}",
        "num_target_shots": "targ_{}",
        "target_shot_order": "order_{}",
    },
    columns=(
        ("rel err (time avg)", "rel_mean_all"),
        ("best", "rel_mean_best_all"),
        ("ckpt std", "rel_mean_ckpt_std_all"),
        ("rampup", "rel_mean_rampup"),
        ("flattop", "rel_mean_flattop"),
        ("flattop ohmic", "rel_mean_flattop_ohmic"),
        ("flattop aux", "rel_mean_flattop_aux"),
        ("rampdown", "rel_mean_rampdown"),
        ("abs err (time avg) [MJ]", "abs_mean_all"),
        ("diverged", "diverged_mean_all"),
        ("rel err (integral, med)", "err_rel_shot_med"),
        ("abs err (integral, med)", "err_abs_shot_med"),
    ),
    stage_metrics=METRIC_NAMES,
    # Submodule prereq cases predict other signals, their errors are not comparable to the main models
    excluded_model_types=SUBMODULE_MODEL_TYPES,
)
