"""DIII-D dataset, built from a transport-validation-datasets internal store.

DIII-D data has no release permission, so transport-validation-datasets stops at its internal store.
"""

from transport_study.datasets.workflow import StoreWorkflow

INNER_WALL = 1.05  # Location of the inner wall, used to calculate minor radius from gapin and R0


class D3DDataWorkflow(StoreWorkflow):
    """DIII-D stores from an internal store, nothing DIII-D specific beyond the base."""
