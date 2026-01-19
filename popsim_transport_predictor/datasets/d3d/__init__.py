import os
from popsim_transport_predictor import PACKAGE_ROOT

from loguru import logger

SUMMARY_TABLE = "summaries"
IPMAX = 400e3  # [A]
PULSE_LENGTH = 0.1  # [s]
MIN_SHOT = 156199
MAX_SHOT = 204996 # Original was 177061 from 1 kHz EFIT bounds, though might not be relevant anymore

HP_SHOTLIST_FILES = [
    "HBP_shotlist_2013_2025",
    "HBP_shotlist_2019_2022",  # Completely encompassed by 2013_2025
    "HBP_shotlist_2024",       # Completely encompassed by 2013_2025
]
sd = {}
for i, shotlist_file in enumerate(HP_SHOTLIST_FILES):
    with open(os.path.join(PACKAGE_ROOT, "datasets", "d3d", shotlist_file), "r") as f:
        lines = f.readlines()
        shot_numbers = [int(line.strip()) for line in lines if line.strip().isdigit()]
    sd[i] = shot_numbers

# Our target shot is 201927
HP_SHOTLIST = sorted(set().union(*sd.values()))

D3D_DATASET_SIGNALS = [
    # Profiles being predicted
    "te_rho",  # Electron temperature profile [eV]
    "ne_rho",  # Electron density profile [m^-3]
    # Global quantities
    "ip",  # Plasma current
    "bt",  # On-axis magnetic field
    "wmhdf",  # Total stored energy # From pedestal
    "betapf",  # Plasma beta
    "n_e",  # Line average electron density [m^-3]
    "aminor",  # Plasma minor radius
    "kappa",  # Plasma elongation
    "tritop",  # Top triangularity
    "tribot",  # Bottom triangularity
    "R0",  # Major radius [m]
    # # Power sources and sinks
    "p_ohm",  # Ohmic heating power
    "p_rad",  # Bulk radiated heating power
    "p_nbi",  # Absorbed NBI heating power
    "p_ech",  # Absorbed ECH heating power
    # Other
    # TODO(ZanderKeith): Add gas valves when we get to that point
]

D3D_SIGNAL_BOUNDS = {
    "Wtot_MJ": {"bounds": (1e-3, None)},
    "beta_p": {"bounds": (0, 1)},
    "P_rad_MW": {"bounds": (0, 3.5)},
    "P_oh_MW": {"bounds": (0, 3.5)},
    "P_ICRF_MW": {"bounds": (0, 4.0)},
}

