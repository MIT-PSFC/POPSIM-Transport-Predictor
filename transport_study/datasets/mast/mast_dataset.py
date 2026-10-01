"""MAST dataset, built from a transport-validation-datasets published store."""

from transport_study.datasets.workflow import PublishedStoreWorkflow


class MASTDataWorkflow(PublishedStoreWorkflow):
    """MAST stores from a published store, nothing MAST specific beyond the base."""
