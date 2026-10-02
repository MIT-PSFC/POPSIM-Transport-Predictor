from pathlib import Path

from transport_validation_datasets import EPISODE_DIM, RADIAL_DIM, TIME_COORD, TIME_DIM

PACKAGE_ROOT = Path(__file__).parent

# The store dimension names come from transport-validation-datasets, which builds the C-Mod and MAST stores
__all__ = ["EPISODE_DIM", "PACKAGE_ROOT", "RADIAL_DIM", "TIME_COORD", "TIME_DIM"]
