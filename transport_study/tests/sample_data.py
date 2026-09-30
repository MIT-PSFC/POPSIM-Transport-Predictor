"""Sample device datasets for the tests that need real data.

The sample .nc files (cmod-low1, cmod-low2, cmod-high, mast-low1, mast-low2, mast-high) live outside the repo.
PTPS_TEST_SAMPLE_DIR points at the directory holding them.
Tests that open them carry requires_sample_data and skip when it is unset.
"""

import os
from pathlib import Path

import pytest

SAMPLE_DIR_ENV_VAR = "PTPS_TEST_SAMPLE_DIR"

# The fallback is never opened, skipped and config-only tests just carry the path
SAMPLE_DIR = Path(os.environ.get(SAMPLE_DIR_ENV_VAR, f"{SAMPLE_DIR_ENV_VAR}_unset"))

requires_sample_data = pytest.mark.skipif(SAMPLE_DIR_ENV_VAR not in os.environ, reason=f"{SAMPLE_DIR_ENV_VAR} is not set")
