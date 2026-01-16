# Email from J. Hughes 2025-12-12
BLESSED_THOMSON_DAY_RANGES = [
    1160503,
    1160527,
    range(1160607, 1160610),
    1160621,
    1160628,
    1160630,
    1160708,
    range(1160712, 1160719),
    range(1160803, 1160820),
    range(1160823, 1160903),
    range(1160908, 1160916),
    range(1160919, 1160924),
    range(1160927, 1160931),
]

BLESSED_THOMSON_DAYS = []
for item in BLESSED_THOMSON_DAY_RANGES:
    if isinstance(item, range):
        BLESSED_THOMSON_DAYS.extend(list(item))
    else:
        BLESSED_THOMSON_DAYS.append(item)

SUMMARY_TABLE = "summary"
IPMAX = 100e3  # [A]
PULSE_LENGTH = 0.1  # [s]
MIN_SHOT = 1050204013
MAX_SHOT = 1160930043

CMOD_DATASET_SIGNALS = [
    # Global quantities
    "ip",  # Plasma current
    "btor",  # On-axis magnetic field
    "wmhd",  # Total stored energy (TODO(ZanderKeith): I don't think C-Mod has a consistent fast particle measurement, so this is all we've got)
    "beta_p",  # Plasma beta
    "n_e",  # Line average electron density [m^-3]
    "a_minor",  # Plasma minor radius
    "kappa",  # Plasma elongation
    "tritop",  # Top triangularity
    "tribot",  # Bottom triangularity
    "rmagx",  # Major radius [m]
    # Power sources and sinks
    "p_oh",  # Ohmic heating power
    "p_rad",  # Bulk radiated heating power
    "p_icrf",  # ICRF heating power
    "p_lh",  # Lower hybrid heating power (yes this is actually lower hybrid on C-Mod, NOT the LH transition threshold like on TCV)
    # Other
]
