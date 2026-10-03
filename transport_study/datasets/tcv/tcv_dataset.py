"""TCV dataset, built from a transport-validation-datasets internal store.

TCV data has no release permission, so transport-validation-datasets stops at its internal store.
"""

from transport_study.datasets.workflow import StoreWorkflow


class TCVDataWorkflow(StoreWorkflow):
    """TCV stores from an internal store, nothing TCV specific beyond the base."""
