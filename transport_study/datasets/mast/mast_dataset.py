"""MAST dataset, built from a transport-validation-datasets published store."""

from transport_study.datasets.workflow import StoreWorkflow


class MASTDataWorkflow(StoreWorkflow):
    """MAST stores from a published store, nothing MAST specific beyond the base."""
