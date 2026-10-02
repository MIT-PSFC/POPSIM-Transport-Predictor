"""Signal names and units, on disk and in the study.

Every tensorized device store (and every sample dataset) uses IMAS names in SI units.
The study works in engineering units instead, so organize_data converts each store once on load.
In the study a bare IMAS name always means SI, and any other unit is a suffix on the name
(ip_MA, energy_mhd_MJ, t_e_keV, n_e_1e20, ...).
"""

import xarray as xr

PREDICTION_STORE_NAME = "ds"

STORE_PROFILES = ("t_e", "n_e")
PROFILE_COMPANION_SUFFIXES = ("_error", "_gradient", "_gradient_error")
STORE_PROFILE_COMPANIONS = tuple(f"{profile}{suffix}" for profile in STORE_PROFILES for suffix in PROFILE_COMPANION_SUFFIXES)

# IMAS summary/heating_current_drive power_additional is their sum
STORE_HEATING_POWERS = ("power_nbi", "power_ic", "power_lh", "power_ec")
STORE_POWERS = ("power_ohm", "power_radiated", *STORE_HEATING_POWERS)

# The 0D signals of a device store, finite at every stored time (the finite filter of every device)
STORE_0D_SIGNALS = (
    "ip",
    "b0",
    "energy_mhd",
    "beta_tor_norm",
    "n_e_line_average",
    "minor_radius",
    "geometric_axis_r",
    "elongation",
    "triangularity_upper",
    "triangularity_lower",
    *STORE_POWERS,
)

# Every signal of a device store, in store order, with its SI unit.
# Unit strings match the transport-validation-datasets published stores.
STORE_SIGNAL_UNITS = {
    "ip": "A",
    "b0": "T",
    "energy_mhd": "J",
    "beta_tor_norm": "dimensionless",
    "n_e_line_average": "m^-3",
    "minor_radius": "m",
    "geometric_axis_r": "m",
    "elongation": "dimensionless",
    "triangularity_upper": "dimensionless",
    "triangularity_lower": "dimensionless",
    **dict.fromkeys(STORE_POWERS, "W"),
    "t_e": "eV",
    "t_e_error": "eV",
    "t_e_gradient": "eV per unit rho_tor_norm",
    "t_e_gradient_error": "eV per unit rho_tor_norm",
    "n_e": "m^-3",
    "n_e_error": "m^-3",
    "n_e_gradient": "m^-3 per unit rho_tor_norm",
    "n_e_gradient_error": "m^-3 per unit rho_tor_norm",
    "fresh_profile": "dimensionless",
}
STORE_SIGNALS = tuple(STORE_SIGNAL_UNITS)

# Study (working-unit) names for the summed heating power and the converted profiles
POWER_ADDITIONAL_MW = "power_additional_MW"
T_E_KEV = "t_e_keV"
N_E_1E20 = "n_e_1e20"

# On-disk name -> (study name, factor from SI, study unit), for every signal not already in study units
WORKING_UNIT_CONVERSIONS = {
    "ip": ("ip_MA", 1e-6, "MA"),
    "energy_mhd": ("energy_mhd_MJ", 1e-6, "MJ"),
    "n_e_line_average": ("n_e_line_average_1e20", 1e-20, "1e20 m^-3"),
    **{power: (f"{power}_MW", 1e-6, "MW") for power in STORE_POWERS},
    "t_e": (T_E_KEV, 1e-3, "keV"),
    "t_e_error": (f"{T_E_KEV}_error", 1e-3, "keV"),
    "t_e_gradient": (f"{T_E_KEV}_gradient", 1e-3, "keV per unit rho_tor_norm"),
    "t_e_gradient_error": (f"{T_E_KEV}_gradient_error", 1e-3, "keV per unit rho_tor_norm"),
    "n_e": (N_E_1E20, 1e-20, "1e20 m^-3"),
    "n_e_error": (f"{N_E_1E20}_error", 1e-20, "1e20 m^-3"),
    "n_e_gradient": (f"{N_E_1E20}_gradient", 1e-20, "1e20 m^-3 per unit rho_tor_norm"),
    "n_e_gradient_error": (f"{N_E_1E20}_gradient_error", 1e-20, "1e20 m^-3 per unit rho_tor_norm"),
}
HEATING_POWERS_MW = tuple(WORKING_UNIT_CONVERSIONS[power][0] for power in STORE_HEATING_POWERS)


def convert_to_working_units(ds: xr.Dataset) -> xr.Dataset:
    """A device store (or a selection of one) renamed and rescaled to the study's working units.

    Converts whichever WORKING_UNIT_CONVERSIONS signals are present,
    so a store whose radial dim was dropped converts too.
    Adds power_additional_MW when the heating powers are there, which needs all four of them.

    Raises:
        ValueError: If only some of the heating powers are present.
    """
    conversions_present = {name: conversion for name, conversion in WORKING_UNIT_CONVERSIONS.items() if name in ds}
    renames = {name: study_name for name, (study_name, _, _) in conversions_present.items()}
    ds_working = ds.rename(renames)
    for study_name, factor, unit in conversions_present.values():
        da_scaled = ds_working[study_name] * factor
        ds_working[study_name] = da_scaled.assign_attrs(ds_working[study_name].attrs | {"units": unit})

    heating_present = [power for power in HEATING_POWERS_MW if power in ds_working]
    if not heating_present:
        return ds_working
    if len(heating_present) != len(HEATING_POWERS_MW):
        raise ValueError(f"Summing {POWER_ADDITIONAL_MW} needs all of {HEATING_POWERS_MW}, the dataset has only {heating_present}")
    power_additional = ds_working[HEATING_POWERS_MW[0]]
    for power in HEATING_POWERS_MW[1:]:
        power_additional = power_additional + ds_working[power]
    ds_working[POWER_ADDITIONAL_MW] = power_additional.assign_attrs(units="MW")
    return ds_working
