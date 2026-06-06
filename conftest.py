import os
import sys
from loguru import logger

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

logger.remove()
logger.add(sys.stderr, colorize=True)
