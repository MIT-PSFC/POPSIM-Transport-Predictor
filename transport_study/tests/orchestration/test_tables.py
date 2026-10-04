"""Tests for the generic per-axis comparison table writer."""

import pandas as pd

from transport_study.orchestration.tables import (
    ComparisonTableSpec,
    write_case_comparison_tables,
)

SPEC = ComparisonTableSpec(
    axis_names=("a", "num_target_shots"),
    case_field_order=("a", "b", "num_target_shots"),
    field_tokens={"a": "a_{}", "b": "b_{}", "num_target_shots": "targ_{}"},
    columns=(("metric", "metric"),),
    stage_metrics=("metric",),
)


def _example_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": ["x", "y", "x", "x"],
            "b": ["b1", "b1", "b1", "b1"],
            "num_target_shots": [0, 0, 5, -1],
            "case_idx": [0, 1, 2, 3],
            "metric": [1.0, 2.5, 0.5, float("nan")],
        }
    )


def test_writes_case_stats_csv(tmp_path):
    write_case_comparison_tables(_example_frame(), SPEC, tmp_path)
    csv_path = tmp_path / "tables" / "case_stats.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 4
    assert list(df.columns) == ["a", "b", "num_target_shots", "case_idx", "metric"]


def test_group_table_content_and_sentinel_ordering(tmp_path):
    write_case_comparison_tables(_example_frame(), SPEC, tmp_path)

    # Three rows share (a=x, b=b1) so the num_target_shots axis gets a table,
    # with the -1 all-shots sentinel listed after the real counts and the NaN
    # metric rendered as a dash
    table_path = tmp_path / "tables" / "num_target_shots" / "a_x.b_b1.md"
    expected = (
        "# num_target_shots comparison\n"
        "\n"
        "Fixed: a=x, b=b1\n"
        "\n"
        "| num_target_shots | metric |\n"
        "|---|---|\n"
        "| 0 | 1 |\n"
        "| 5 | 0.5 |\n"
        "| -1 | - |\n"
    )
    assert table_path.read_text() == expected


def test_singleton_groups_skipped(tmp_path):
    write_case_comparison_tables(_example_frame(), SPEC, tmp_path)

    # Along the a axis only (b1, targ 0) has two members
    a_tables = sorted(p.name for p in (tmp_path / "tables" / "a").glob("*.md"))
    assert a_tables == ["b_b1.targ_0.md"]


def test_empty_frame_writes_nothing(tmp_path):
    write_case_comparison_tables(_example_frame().iloc[:0], SPEC, tmp_path)
    assert not (tmp_path / "tables").exists()
