"""Signal names and units, on disk and in the study.

Every tensorized device store (and every sample dataset) uses IMAS names in SI units,
the shared schema of transport-validation-datasets (store_schema.STORE_SIGNAL_ATTRS).
The study works in engineering units, so organize_data converts each store once on load.
In the study a bare IMAS name always means SI, and any other unit is a suffix on the name
(ip_MA, energy_mhd_MJ, t_e_keV, n_e_1e20, ...).
The stores keep the source sign of ip and b0 (their cocos variable records the convention),
the study works with magnitudes, taken in convert_to_working_units.
"""

import xarray as xr
from transport_validation_datasets.store_schema import (
    HEATING_POWERS,
    POWER_SIGNALS,
    STORE_SIGNALS,
)

# The shared schema's signals the studies read, every one but fresh_equilibrium
STUDY_STORE_SIGNALS = tuple(name for name in STORE_SIGNALS if name != "fresh_equilibrium")

# Signals the stores keep signed, taken as magnitudes on conversion
SIGNED_SIGNALS = ("ip", "b0")

# Study (working-unit) names for the summed heating power, the field at the geometric axis and the converted profiles
POWER_ADDITIONAL_MW = "power_additional_MW"
B_GEO = "b_geo"
T_E_KEV = "t_e_keV"
N_E_1E20 = "n_e_1e20"

# Store signals each derived study signal needs
B_GEO_SOURCE_SIGNALS = ("b0", "r0", "geometric_axis_r")

# On-disk name -> (study name, factor from SI, study unit), for every signal not already in study units
WORKING_UNIT_CONVERSIONS = {
    "ip": ("ip_MA", 1e-6, "MA"),
    "energy_mhd": ("energy_mhd_MJ", 1e-6, "MJ"),
    "n_e_line_average": ("n_e_line_average_1e20", 1e-20, "1e20 m^-3"),
    **{power: (f"{power}_MW", 1e-6, "MW") for power in POWER_SIGNALS},
    "t_e": (T_E_KEV, 1e-3, "keV"),
    "t_e_error": (f"{T_E_KEV}_error", 1e-3, "keV"),
    "t_e_gradient": (f"{T_E_KEV}_gradient", 1e-3, "keV per unit rho_tor_norm"),
    "t_e_gradient_error": (f"{T_E_KEV}_gradient_error", 1e-3, "keV per unit rho_tor_norm"),
    "n_e": (N_E_1E20, 1e-20, "1e20 m^-3"),
    "n_e_error": (f"{N_E_1E20}_error", 1e-20, "1e20 m^-3"),
    "n_e_gradient": (f"{N_E_1E20}_gradient", 1e-20, "1e20 m^-3 per unit rho_tor_norm"),
    "n_e_gradient_error": (f"{N_E_1E20}_gradient_error", 1e-20, "1e20 m^-3 per unit rho_tor_norm"),
}
HEATING_POWERS_MW = tuple(WORKING_UNIT_CONVERSIONS[power][0] for power in HEATING_POWERS)
# Study name -> on-disk name, for the converted signals
STORE_NAMES = {study_name: name for name, (study_name, _, _) in WORKING_UNIT_CONVERSIONS.items()}


def store_signals_for(study_signals: list[str] | tuple[str, ...]) -> list[str]:
    """The store signals convert_to_working_units needs to yield the given study signals, in store order.

    Derived signals expand to their sources (b_geo to b0, r0 and geometric_axis_r,
    power_additional_MW to the four heating powers), converted ones map back to their IMAS name,
    anything else (time, fresh_profile, ...) is already a store name.
    """
    store_names: set[str] = set()
    for study_name in study_signals:
        if study_name == B_GEO:
            store_names.update(B_GEO_SOURCE_SIGNALS)
        elif study_name == POWER_ADDITIONAL_MW:
            store_names.update(HEATING_POWERS)
        else:
            store_names.add(STORE_NAMES.get(study_name, study_name))
    ordered = [name for name in STUDY_STORE_SIGNALS if name in store_names]
    return ordered + sorted(store_names - set(ordered))


def convert_to_working_units(ds: xr.Dataset) -> xr.Dataset:
    """A device store (or a selection of one) renamed and rescaled to the study's working units.

    Converts whichever WORKING_UNIT_CONVERSIONS signals are present,
    so a store whose radial dim was dropped converts too.
    Takes ip and b0 as magnitudes, the stores keep their source sign.
    Adds b_geo when b0 is there, the vacuum toroidal field at the geometric axis,
    where TORAX, q_star, aB0 and the confinement scalings quote it.
    The vacuum field falls off as 1/R, so b_geo = |b0| r0 / geometric_axis_r.
    beta_tor_norm keeps the IMAS normalization with |b0| at r0, so the betan inversions use b0.
    Adds power_additional_MW when the heating powers are there, which needs all four of them.

    Raises:
        ValueError: If b0 is present without r0 or geometric_axis_r, or only some of the heating powers are present.
    """
    conversions_present = {name: conversion for name, conversion in WORKING_UNIT_CONVERSIONS.items() if name in ds}
    renames = {name: study_name for name, (study_name, _, _) in conversions_present.items()}
    ds_working = ds.rename(renames)
    for name in SIGNED_SIGNALS:
        study_name = renames.get(name, name)
        if study_name in ds_working:
            ds_working[study_name] = abs(ds_working[study_name]).assign_attrs(ds_working[study_name].attrs)
    for study_name, factor, unit in conversions_present.values():
        da_scaled = ds_working[study_name] * factor
        ds_working[study_name] = da_scaled.assign_attrs(ds_working[study_name].attrs | {"units": unit})

    if "b0" in ds_working:
        missing_geometry = {name for name in B_GEO_SOURCE_SIGNALS if name not in ds_working.variables}
        if missing_geometry:
            raise ValueError(f"Deriving {B_GEO} needs r0 and geometric_axis_r, the dataset lacks {sorted(missing_geometry)}")
        b_geo = ds_working["b0"] * ds_working["r0"] / ds_working["geometric_axis_r"]
        ds_working[B_GEO] = b_geo.assign_attrs(
            units="T", description="Vacuum toroidal field at geometric_axis_r, |b0| r0 / geometric_axis_r"
        )

    heating_present = [power for power in HEATING_POWERS_MW if power in ds_working]
    if not heating_present:
        return ds_working
    if len(heating_present) != len(HEATING_POWERS_MW):
        raise ValueError(f"Summing {POWER_ADDITIONAL_MW} needs all of {HEATING_POWERS_MW}, the dataset has only {heating_present}")
    power_additional = sum((ds_working[power] for power in HEATING_POWERS_MW[1:]), ds_working[HEATING_POWERS_MW[0]])
    ds_working[POWER_ADDITIONAL_MW] = power_additional.assign_attrs(units="MW")
    return ds_working
