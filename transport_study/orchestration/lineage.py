"""Read-only views of the studies a study borrows cases from.

For lack of a better term I'm calling this 'hyperscrabble'.
A regular study runs all combinations of the axes in its cases,
meaning additional options in any of the axes get run across all the other dimensions.
This is thorough but also very slow.
So, instead, enable studies to build off of others which only add additional options along one of the axes.
Kinda like hyperdimensional scrabble.

A study may name a parent study of the same type (the parent_study config field).
Every case the parent has trained, or that the parent's current grid will train, is borrowed instead of retrained:
its checkpoints, tuned config and result file are read from the parent's working dir, never written.
The parent may itself have a parent, so a chain of studies can grow a grid one restricted direction at a time.

A borrowed case is only the same model when every locked config field matches,
so a child refuses to start unless its locked fields equal each ancestor's.
Each config lock records its parent's stamp at creation (see config_lock.py),
so resetting any ancestor stops every descendant until that descendant is cleaned too.

Ownership is root-first: a case belongs to the root-most study of the chain that claims it,
so every study of the chain agrees on where each case lives,
and a borrowed case is never rebuilt around prereqs from a different study than the one that trained it.
"""

from dataclasses import dataclass
from pathlib import Path

RESULT_FILENAME = "result_data.nc"


@dataclass(frozen=True, eq=False)
class StudyArchive:
    """One study's working dir as its descendants see it, plus the chain of studies above it.

    case_names is the study's current grid, as its config lock records it.
    """

    name: str
    working_dir: Path
    stamp: str
    parent: "StudyArchive | None"
    case_names: frozenset[str]

    @property
    def model_dir(self) -> Path:
        return self.working_dir / "models"

    @property
    def result_dir(self) -> Path:
        return self.working_dir / "results"

    def result_path(self, case_name: str) -> Path:
        return self.result_dir / case_name / RESULT_FILENAME

    def has_result(self, case_name: str) -> bool:
        return self.result_path(case_name).exists()

    def claims(self, case_name: str) -> bool:
        """Whether the study has trained the case or its current grid will.

        A leftover checkpoint dir of a case no longer in the grid is not a claim, nobody will finish it.
        """
        return case_name in self.case_names or self.has_result(case_name)

    def chain(self) -> list["StudyArchive"]:
        """This study, then its parent, and so on up to the root."""
        studies = [self]
        while studies[-1].parent is not None:
            studies.append(studies[-1].parent)
        return studies
