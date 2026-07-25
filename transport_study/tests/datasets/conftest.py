"""Fixtures shared by the dataset workflow tests."""

import pytest
from loguru import logger


@pytest.fixture
def logged():
    """Every loguru message emitted during the test, in order.

    The workflow culls report which rule fired only through the log, so tests
    that need to tell two culls apart read this instead of the return value.
    """
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    yield messages
    logger.remove(sink_id)
