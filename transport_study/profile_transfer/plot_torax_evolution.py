"""Visualize how the TORAX profile predictor relaxes ne/Te towards the target shape.

Runs the ProfilePredictorTorax module on a single (shot, timestep) slice of a dataset,
recording the core profiles after every internal TORAX step, and plots the density and
temperature evolution against the measured target profiles.

Example:
    python -m transport_study.profile_transfer.plot_torax_evolution \
        --dataset transport_study/datasets/sample/cmod-low1.nc \
        --shot 1160824011 --timestep 800 \
        --transport_model cgm \
        --checkpoint /path/to/trained/torax/checkpoint \
        --prescribed '{"chi_e_i_ratio": 2.0, "S_total": 1.0}'
"""

from pathlib import Path

import fire
import jax
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loguru import logger
from matplotlib import cm
from popsim.ml.checkpointing import create_default_checkpoint_manager, restore_model

from transport_study.modules.normalization import CoralFeatureNormalizer
from transport_study.modules.profile_predictor.module import N_NN_INPUTS
from transport_study.modules.profile_predictor.torax_module import (
    SOURCE_COEFFICIENT_NAMES,
    TRANSPORT_COEFFICIENT_NAMES,
    ProfilePredictorTorax,
)
from transport_study.modules.profile_predictor.train_configs import (
    PROFILE_PREDICTOR_TORAX_CONFIGS,
)
from transport_study.plot_style import BACKGROUND_COLOR, TEXT_COLOR, style_axis

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 13

INPUT_VARS = [
    "Ip_MA",
    "B0",
    "betan",
    "ne20_line_avg",
    "R0",
    "a_minor",
    "kappa",
    "delta_top",
    "delta_bot",
]


def valid_timesteps(shot_ds: xr.Dataset) -> np.ndarray:
    """Timesteps where all inputs are finite and the target profiles are freshly measured."""
    valid = np.ones(shot_ds.sizes["time_idx"], dtype=bool)
    for var in INPUT_VARS:
        valid &= ~np.isnan(shot_ds[var].values)
    valid &= shot_ds["fresh_profiles"].values.astype(bool)
    valid &= ~np.all(np.isnan(shot_ds["ne20_rho"].values), axis=-1)
    valid &= ~np.all(np.isnan(shot_ds["Te_keV_rho"].values), axis=-1)
    return np.flatnonzero(valid)


def _load_timeslice(dataset: str | Path, shot: int, timestep: int, ds_source_idx: int = 0) -> xr.Dataset:
    ds_path = Path(dataset)
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found: {ds_path}")
    ds = xr.open_dataset(ds_path)
    shot_ds = ds.sel(shot=shot)

    valid = valid_timesteps(shot_ds)
    if timestep not in valid:
        if len(valid) == 0:
            raise ValueError(f"Shot {shot} has no timestep with finite inputs and fresh target profiles")
        nearest = valid[np.argmin(np.abs(valid - timestep))]
        raise ValueError(
            f"Timestep {timestep} of shot {shot} has NaN inputs, stale profiles, or all-NaN target profiles. "
            f"Nearest valid timestep: {nearest} (valid range {valid.min()}-{valid.max()}, {len(valid)} total)"
        )
    timeslice = shot_ds.isel(time_idx=timestep)
    # Raw device files lack the device index organize_data assigns, the
    # module's normalizer needs it to pick the right per-device statistics
    timeslice["ds_source_idx"] = float(ds_source_idx)
    return timeslice


def _build_module(timeslice: xr.Dataset, checkpoint: str | Path | None, transport_model: str, n_devices: int = 1) -> ProfilePredictorTorax:
    model_cfg = PROFILE_PREDICTOR_TORAX_CONFIGS[transport_model]["model_init_config"]
    module = ProfilePredictorTorax(
        nn_width=model_cfg["nn_width"],
        nn_depth=model_cfg["nn_depth"],
        rhogrid=tuple(timeslice["rho"].values.tolist()),
        torax_config=model_cfg["torax_config"],
        key=jax.random.PRNGKey(model_cfg["prng_seed"]),
        # Identity buffers, restore_model overwrites them with the trained
        # statistics when a checkpoint is given (n_devices must match it)
        normalizer=CoralFeatureNormalizer.identity(n_devices, N_NN_INPUTS),
        transport_model=transport_model,
        geometry_builder=model_cfg.get("geometry_builder", "circular"),
        delta_exponent=model_cfg.get("delta_exponent", 2.0),
    )
    if checkpoint is not None:
        manager = create_default_checkpoint_manager(checkpoint)
        module = restore_model(manager, module)
        logger.info(f"Restored profile predictor from checkpoint {checkpoint}")
    else:
        logger.warning("No checkpoint given, using randomly initialized network weights")
    return module


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

    rho = timeslice["rho"].values
    ne_targ = timeslice["ne20_rho"].values
    te_targ = timeslice["Te_keV_rho"].values

    fig, (ax_ne, ax_te) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
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
        ax_ne.plot(step["rho"], step["ne20"], color=colors[i], linewidth=2, label=label)
        ax_te.plot(step["rho"], step["te_keV"], color=colors[i], linewidth=2, label=label)

    ax_ne.plot(rho, ne_targ, color="white", linewidth=3, linestyle="--", label="Measured target")
    ax_te.plot(rho, te_targ, color="white", linewidth=3, linestyle="--", label="Measured target")

    ax_ne.set_ylabel(r"$n_e$ [$10^{20}$ m$^{-3}$]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    ax_te.set_ylabel(r"$T_e$ [keV]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    # TORAX evolves profiles on rho_norm, which for circular geometry equals the
    # normalized minor radius, so both the TORAX steps and the measured targets are in rho.
    ax_te.set_xlabel(r"$\rho$", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)

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
    dataset: str,
    shot: int,
    timestep: int,
    transport_model: str = "cgm",
    checkpoint: str | None = None,
    n_devices: int = 1,
    ds_source_idx: int = 0,
    prescribed: dict | None = None,
    output_dir: str | None = None,
):
    """Plot ne/Te profile evolution across the internal TORAX relaxation steps.

    Args:
        dataset: Path to a NetCDF dataset with dims (shot, time_idx, rho).
        shot: Shot number to select.
        timestep: time_idx index of the timeslice to predict.
        transport_model: TORAX transport model: "constant", "cgm", or "gyrobohm".
        checkpoint: Optional checkpoint directory of a trained torax profile predictor
            (must have been trained with the same transport_model).
        n_devices: Number of devices the checkpoint was trained with (sizes the
            normalizer buffers so the checkpoint restores).
        ds_source_idx: Device index of the plotted dataset in the training
            source ordering (selects the normalizer's per-device statistics).
        prescribed: Optional dict of coefficients bypassing the NN outputs.
            Valid keys are the transport coefficients of the chosen model
            (constant: chi_i, chi_e, D_e [m^2/s], V_e [m/s];
            cgm: chi_e_i_ratio, chi_D_ratio, VR_D_ratio, alpha, chi_stiff;
            gyrobohm: chi_bohm_multiplier, chi_gyrobohm_multiplier, D_face_c1,
            D_face_c2, V_face_coeff)
            plus the source coefficients (S_total [1e21 /s], P_aux_total [MW],
            gaussian_location, gaussian_width, electron_heat_fraction) and
            n_e_right_bc [1e20 m^-3], T_e_right_bc [keV].
            From the CLI pass as a dict literal, e.g. --prescribed '{"S_total": 1.0}'.
        output_dir: Directory to save the figure in (default: current directory).
    """
    if transport_model not in TRANSPORT_COEFFICIENT_NAMES:
        raise ValueError(f"Unknown transport model '{transport_model}', valid: {sorted(TRANSPORT_COEFFICIENT_NAMES)}")
    timeslice = _load_timeslice(dataset, shot, timestep, ds_source_idx=ds_source_idx)
    time_s = float(timeslice["time"].values)
    module = _build_module(timeslice, checkpoint, transport_model, n_devices=n_devices)

    prescribed = prescribed or {}
    prescribed_names = {name for name, value in prescribed.items() if value is not None}

    steps, coeffs = module.evolve(timeslice, prescribed=prescribed)
    logger.info(f"TORAX relaxation recorded {len(steps)} states (initial + {len(steps) - 1} steps)")

    out_dir = Path(output_dir) if output_dir is not None else Path.cwd()
    plot_path = out_dir / f"torax_evolution_{transport_model}_{Path(dataset).stem}_{shot}_ts{timestep}.png"
    return plot_relaxation(
        steps,
        coeffs,
        timeslice,
        transport_model,
        title_context=f"shot {shot} @ t={time_s:.3f}s (time_idx {timestep})",
        plot_path=plot_path,
        prescribed_names=prescribed_names,
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "plot_torax_evolution": plot_torax_evolution,
        }
    )
