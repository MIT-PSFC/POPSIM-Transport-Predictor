import os
import sys
from datetime import datetime
from pathlib import Path
from loguru import logger

os.environ.setdefault("JAX_DEBUG_NANS", "0")
os.environ.setdefault("MPLBACKEND", "Agg")
# The jax CUDA plugin version check hangs forever on a node without a GPU.
if not any(Path("/dev").glob("nvidia[0-9]*")):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent
DEBUG_LOG_DIR = PROJECT_ROOT / "scratch" / "debug_logs"
DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_LOG_PATH = DEBUG_LOG_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

logger.remove()
logger.add(sys.stderr, colorize=True)
logger.add(DEBUG_LOG_PATH)
    
class _StdoutTee:
    def __init__(self, orig, path):
        self._orig = orig
        self._file = open(path, "a")

    def write(self, data):
        self._orig.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._orig.flush()
        self._file.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)


sys.stdout = _StdoutTee(sys.stdout, DEBUG_LOG_PATH)
