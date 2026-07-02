import pytest

from transport_study.config import reset_config


@pytest.fixture(autouse=True)
def _fresh_global_config():
    """Clear the one-shot global config around every test.

    The global config can normally only be loaded once per process, which made
    test outcomes depend on execution order (whichever test loaded it first
    won, later load_config calls raised). With this fixture every test starts
    uninitialized and must load the config it wants, as if it were the first
    test in the process.
    """
    reset_config()
    yield
    reset_config()
