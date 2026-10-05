"""Fixtures shared across the whole test suite."""

from pathlib import Path

import pytest

from transport_study.config import reset_config
from transport_study.tests.datasets.synthetic_store import write_synthetic_store
from transport_study.tests.stubs import StubCase, StubConfig, StubStudy


@pytest.fixture(autouse=True)
def fresh_global_config():
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


@pytest.fixture
def make_stub_study(tmp_path):
    """Factory building a StubStudy rooted at tmp_path from a list of StubCases."""

    def _make(cases: list[StubCase], study_name: str = "stub_study", **config_overrides) -> StubStudy:
        cfg = StubConfig(
            study_name=study_name,
            working_dir_base=Path(tmp_path),
            dataset_paths={},
            target_device="cmod",
            **config_overrides,
        )
        return StubStudy(cfg, cases)

    return _make


@pytest.fixture(scope="session")
def synthetic_device_stores(tmp_path_factory) -> dict[str, Path]:
    """Two synthetic stores in the TVD schema, cmod and mast, written once per session.

    Both hold the same three shots, so a study config takes either as the target.
    """
    store_dir = tmp_path_factory.mktemp("synthetic_device_stores")
    store_paths = {}
    for device in ("cmod", "mast"):
        store_paths[device] = store_dir / f"{device}.zarr"
        write_synthetic_store(store_paths[device], internal=False)
    return store_paths
