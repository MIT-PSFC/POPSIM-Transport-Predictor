from pathlib import Path

from dynaconf import Dynaconf

from transport_study import PACKAGE_ROOT

# DIII-D dataset settings, shared by every module of this package
config = Dynaconf(settings_files=[Path(PACKAGE_ROOT) / "datasets/d3d/config.toml"])
