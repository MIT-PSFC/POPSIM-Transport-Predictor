"""Stub Study / Case for the orchestration tests.

The loop pickup, watchdog, and parallel-analysis machinery under test is
entirely generic over the case grid, so a real ProfileStudy / PowerBalanceStudy
grid would only add irrelevant failure modes (dataset loading, prereq chains,
model-type validation). These stubs give the driver exactly the interface it
reads and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from transport_study.orchestration.study import CaseGridConfig, Study


class StubConfig(CaseGridConfig):
    """Minimal case-grid config: the base Study only reads grid-wide fields.

    The case grid is injected, so the axes carry inert defaults.
    """

    target_test_set_size: int = 0
    training_datasets: tuple[str, ...] = ()


@dataclass
class StubCase:
    """A case identified purely by name, carrying the attributes Study reads.

    Deliberately not a Study.Case subclass: the base dataclass validates
    model_type against VALID_MODEL_TYPES and builds a prereq chain, neither of
    which these tests want. check_data_requirements reads training_data and
    domain_adaptation, so both carry inert defaults.
    """

    name: str
    prereqs: list[StubCase] | None = None
    training_data: SimpleNamespace = field(default_factory=lambda: SimpleNamespace(sources=[], exnihilo=False))
    domain_adaptation: str | None = None

    def is_hyperparam_case(self) -> bool:
        return False

    def is_impossible(self) -> bool:
        return False

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)


class StubStudy(Study):
    """Study whose case list is injected rather than built from a case grid."""

    Config = StubConfig
    STUDY_TYPE = "stub"
    CASE_AXIS_FIELDS = ()
    # This module doubles as the per-case analysis reports module (see below),
    # which is all orchestration.case_analysis resolves off the study class
    ANALYSIS_REPORTS_MODULE = "transport_study.tests.stubs"
    ANALYSIS_METRICS_MODULE = "transport_study.tests.stubs"

    def __init__(self, cfg: CaseGridConfig, cases: list[StubCase]):
        self.injected_cases = list(cases)
        super().__init__(cfg)

    def make_cases(self) -> list[StubCase]:
        return self.injected_cases


# Marker file standing in for a real study's per-case report artifacts
ANALYSIS_MARKER_NAME = "analysis_done"


def analysis_case_done(study: Study, case, figure_dir) -> bool:
    """Reports-module hook the parallel analysis driver polls for completion."""
    return (figure_dir / str(case) / ANALYSIS_MARKER_NAME).exists()


def mark_analysis_done(study: Study, case) -> None:
    """Stand in for a finished analysis job having written its artifacts."""
    marker = study.figure_dir / str(case) / ANALYSIS_MARKER_NAME
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
