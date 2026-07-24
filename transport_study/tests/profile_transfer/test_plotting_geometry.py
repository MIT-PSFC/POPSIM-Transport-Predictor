"""Test stubs for geometry_builder handling in profile transfer comparison figures.

Blocked out for review, not implemented. Covers the new
geometry_builder_comparison family and the geometry_builder conditioning
added to the four existing families plus freeze_shapes_comparison.
"""


def test_geometry_builder_comparison_filters_to_torax_models():
    """Non-torax model types produce no figures even when present in metrics_ds."""


def test_geometry_builder_comparison_skips_single_geometry_combos():
    """Combinations where only circular exists produce no figure."""


def test_geometry_builder_comparison_filename_tokens():
    """Output files follow model_type.td_x.norm_x.da_x.freeze_x.png with the
    compared geom axis absent from the filename."""


def test_model_comparison_splits_by_geometry_builder():
    """With circular and miller cases present for one torax model type, each
    geometry lands in its own figure (geom_ token in filename) and no series
    mixes shots from both geometries into one zigzag line."""


def test_domain_adaptation_comparison_splits_by_geometry_builder():
    """Same split-by-geometry guarantee for the domain adaptation family."""


def test_data_normalization_comparison_splits_by_geometry_builder():
    """Same split-by-geometry guarantee for the data normalization family."""


def test_training_dataset_comparison_splits_by_geometry_builder():
    """Same split-by-geometry guarantee for the training dataset family."""


def test_freeze_shapes_comparison_no_duplicate_shot_keys_across_geometries():
    """With both geometries present at the same num_target_shots values the
    frozen and unfrozen by-shot lookups stay per-geometry, so the frozen minus
    unfrozen difference never pairs circular with miller."""
