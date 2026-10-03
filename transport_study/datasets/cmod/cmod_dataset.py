"""C-Mod dataset, built from a transport-validation-datasets published store."""

from transport_study.datasets.workflow import StoreWorkflow


class CModDataWorkflow(StoreWorkflow):
    """C-Mod stores from a published store, nothing C-Mod specific beyond the base."""
