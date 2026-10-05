"""Visualize how a trained TORAX profile predictor relaxes ne/Te towards the target shape.

Rebuilds one torax case of a profile study through its make_train_config,
runs it on a single (shot, timestep) slice of a device store,
records the core profiles after every internal TORAX step,
and plots the density and temperature evolution against the measured target profiles.

Example:
    python -m transport_study.profile_transfer.plot_torax_evolution \
        --study_config studies/icddps2/profile_predictor.toml \
        --case case.torax-gyrobohm.td_cmod.norm_physics.freeze_True.geom_circular.targ_10.da_transfer \
        --shot 1160920013 --timestep 264 \
        --prescribed '{"chi_bohm_multiplier": 2.0, "S_total": 1.0}'
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loguru import logger
from matplotlib import cm

from transport_study import EPISODE_DIM, RADIAL_DIM, TIME_DIM
from transport_study.config import config
from transport_study.modules.profile_predictor.module import NN_INPUT_SOURCE_VARS
from transport_study.modules.profile_predictor.torax_module import (
    SOURCE_COEFFICIENT_NAMES,
    TRANSPORT_COEFFICIENT_NAMES,
)
from transport_study.plot_style import BACKGROUND_COLOR, TEXT_COLOR, style_axis
from transport_study.signals import convert_to_working_units

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 13


def valid_timesteps(shot_ds: xr.Dataset) -> np.ndarray:
    """Timesteps where all inputs are finite and the target profiles are freshly measured."""
    valid = np.ones(shot_ds.sizes[TIME_DIM], dtype=bool)
    for var in NN_INPUT_SOURCE_VARS:
        valid &= ~np.isnan(shot_ds[var].values)
    valid &= shot_ds["fresh_profile"].values.astype(bool)
    valid &= ~np.all(np.isnan(shot_ds["n_e_1e20"].values), axis=-1)
    valid &= ~np.all(np.isnan(shot_ds["t_e_keV"].values), axis=-1)
    return np.flatnonzero(valid)


def load_timeslice(device: str, shot: int, timestep: int) -> xr.Dataset:
    """One timeslice of a device store in working units, the inputs a profile predictor takes.

    Raises when the timeslice has NaN inputs or stale or all-NaN target profiles.
    """
    ds_store = xr.open_dataset(config.dataset_paths[device])
    shot_ds = convert_to_working_units(ds_store.sel({EPISODE_DIM: shot}))
    valid = valid_timesteps(shot_ds)
    if timestep not in valid:
        if len(valid) == 0:
            raise ValueError(f"Shot {shot} has no timestep with finite inputs and fresh target profiles")
        nearest = valid[np.argmin(np.abs(valid - timestep))]
        raise ValueError(
            f"Timestep {timestep} of shot {shot} has NaN inputs, stale profiles, or all-NaN target profiles. "
            f"Nearest valid timestep: {nearest} (valid range {valid.min()}-{valid.max()}, {len(valid)} total)"
        )
    timeslice = shot_ds.isel({TIME_DIM: timestep})
    # Raw device stores lack the device index organize_data assigns,
    # the module's normalizer needs it to pick the per-device statistics
    timeslice["ds_source_idx"] = float(config.ds_source_to_idx[device])
    return timeslice


def plot_relaxation(
    steps: list[dict],
    coeffs: dict,
    timeslice: xr.Dataset,
    transport_model: str,
    title_context: str,
    plot_path: Path,
    prescribed_names: set[str] | None = None,
) -> Path:
    """Render the recorded TORAX relaxation steps against the measured target profiles.

    Args:
        steps: Recorded TORAX states from ProfilePredictorTorax.evolve.
        coeffs: Transport/source coefficients actually used, from evolve.
        timeslice: The single-timeslice dataset the module was evaluated on.
        transport_model: TORAX transport model name, for the title.
        title_context: Shot / time description appended to the title.
        plot_path: Where to save the figure.
        prescribed_names: Coefficient names that were prescribed instead of NN-predicted.
    """
    prescribed_names = prescribed_names or set()

    rho = timeslice[RADIAL_DIM].values
    ne_targ = timeslice["n_e_1e20"].values
    te_targ = timeslice["t_e_keV"].values

    # Constrained layout keeps the multi-line suptitle clear of the top panel
    fig, (ax_ne, ax_te) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, layout="constrained")
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    def _coeff_label(name: str) -> str:
        source = "prescribed" if name in prescribed_names else "NN"
        return f"{name}={coeffs[name]:.2f} ({source})"

    coeff_line = ", ".join(_coeff_label(name) for name in TRANSPORT_COEFFICIENT_NAMES[transport_model])
    source_line = ", ".join(_coeff_label(name) for name in SOURCE_COEFFICIENT_NAMES)
    bc_line = ", ".join(_coeff_label(name) for name in ["n_e_right_bc", "T_e_right_bc"])
    fig.suptitle(
        f"TORAX profile relaxation ({transport_model}) - {title_context}\n{coeff_line}\n{source_line}\n{bc_line}",
        fontsize=TITLE_FONTSIZE - 4,
        color=TEXT_COLOR,
    )

    # Start colormap above 0 so the earliest steps stay visible on the dark background
    colors = cm.viridis(np.linspace(0.25, 1.0, len(steps)))
    for i, step in enumerate(steps):
        label = f"t={step['t'] * 1e3:.0f} ms"
        ax_ne.plot(step[RADIAL_DIM], step["n_e_1e20"], color=colors[i], linewidth=2, label=label)
        ax_te.plot(step[RADIAL_DIM], step["t_e_keV"], color=colors[i], linewidth=2, label=label)

    ax_ne.plot(rho, ne_targ, color="white", linewidth=3, linestyle="--", label="Measured target")
    ax_te.plot(rho, te_targ, color="white", linewidth=3, linestyle="--", label="Measured target")

    ax_ne.set_ylabel(r"$n_e$ [$10^{20}$ m$^{-3}$]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    ax_te.set_ylabel(r"$T_e$ [keV]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    # TORAX evolves profiles on rho_norm, its normalized toroidal flux coordinate,
    # which is the rho_tor_norm the measured targets are on.
    ax_te.set_xlabel(r"$\rho_{tor,N}$", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)

    for ax in (ax_ne, ax_te):
        style_axis(ax, TICK_FONTSIZE)
        ax.legend(
            fontsize=LEGEND_FONTSIZE,
            labelcolor=TEXT_COLOR,
            facecolor=BACKGROUND_COLOR,
            edgecolor=TEXT_COLOR,
            ncols=2,
            loc="upper right",
        )

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot to {plot_path}")
    return plot_path


def plot_torax_evolution(
    study_config: str,
    case: str,
    shot: int,
    timestep: int,
    device: str | None = None,
    prescribed: dict | None = None,
    output_dir: str | None = None,
) -> Path:
    """Plot ne/Te profile evolution across the internal TORAX relaxation steps of one trained torax case.

    The module is built through the study's make_train_config,
    so its tuned network size, geometry builder and solver settings match the checkpoint.

    Args:
        study_config: TOML of the profile study the case belongs to.
        case: Case string, the name of the case's checkpoint directory.
        shot: Shot number to select.
        timestep: time_idx index of the timeslice to predict.
        device: Device key of the shot, the study's target device by default.
        prescribed: Optional dict of coefficients bypassing the NN outputs.
            Valid keys are the transport coefficients of the case's TORAX model
            (constant: chi_i, chi_e, D_e [m^2/s], V_e [m/s],
            gyrobohm: chi_bohm_multiplier, chi_gyrobohm_multiplier, D_face_c1, D_face_c2, V_face_coeff,
            qlknn: ITG_flux_ratio_correction, ETG_correction_factor, collisionality_multiplier),
            the source coefficients (S_total [1e21 /s], P_aux_total [MW],
            gaussian_location, gaussian_width, electron_heat_fraction)
            and n_e_right_bc [1e20 m^-3], T_e_right_bc [keV].
            From the CLI pass a dict literal, e.g. --prescribed '{"S_total": 1.0}'.
        output_dir: Directory to save the figure in (default: current directory).
    """
    # Function-level imports: profile_study imports this module through case_reports, a top-level import would cycle
    from transport_study.profile_transfer.profile_study import ProfileStudy
    from transport_study.profile_transfer.restore_predictor import (
        restore_profile_predictor,
    )

    study = ProfileStudy(study_config)
    profile_case = study.case_by_name(case)
    transport_model = profile_case.model_type.removeprefix("torax-")
    if transport_model not in TRANSPORT_COEFFICIENT_NAMES:
        raise ValueError(f"Case {case} is not a torax case")
    timeslice = load_timeslice(device or config.target_device, shot, timestep)
    time_s = float(timeslice["time"].values)
    train_config = study.make_train_config(profile_case)
    module = restore_profile_predictor(train_config)

    prescribed = prescribed or {}
    prescribed_names = {name for name, value in prescribed.items() if value is not None}
    steps, coeffs = module.evolve(timeslice, prescribed=prescribed)
    logger.info(f"TORAX relaxation recorded {len(steps)} states (initial + {len(steps) - 1} steps)")

    out_dir = Path(output_dir) if output_dir is not None else Path.cwd()
    plot_path = out_dir / f"torax_evolution_{case}_shot{shot}_ts{timestep}.png"
    return plot_relaxation(
        steps,
        coeffs,
        timeslice,
        transport_model,
        title_context=f"{case}\nshot {shot} @ t={time_s:.3f}s (time_idx {timestep})",
        plot_path=plot_path,
        prescribed_names=prescribed_names,
    )


if __name__ == "__main__":
    fire.Fire(plot_torax_evolution)
