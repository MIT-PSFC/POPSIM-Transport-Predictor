CMOD_DATASET_SIGNALS = [
    # Global quantities
    "ip",  # Plasma current
    "btor",  # On-axis magnetic field
    "wmhd",  # Total stored energy (C-Mod has no consistent fast particle measurement, so this is all we've got)
    "n_e",  # Line average electron density [m^-3]
    "beta_n",  # Normalized beta
    "a_minor",  # Plasma minor radius
    "kappa",  # Plasma elongation
    "tritop",  # Top triangularity
    "tribot",  # Bottom triangularity
    "rout",  # Major radius [m]
    # Power sources and sinks
    "p_oh",  # Ohmic heating power
    "p_rad",  # Bulk radiated heating power
    "p_icrf",  # ICRF heating power
    "p_lh",  # Lower hybrid heating power (yes this is actually lower hybrid on C-Mod, NOT the LH transition threshold like on TCV)
    # Other
    "beta_p",  # Poloidal beta
]
