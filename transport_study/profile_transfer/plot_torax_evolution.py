"""Visualize how the TORAX profile predictor relaxes ne/Te towards the target shape.

Runs the ProfilePredictorTorax module on a single (shot, timestep) slice of a dataset,
recording the core profiles after every internal TORAX step, and plots the density and
temperature evolution against the measured target profiles.

Example:
    python -m transport_study.profile_transfer.plot_torax_evolution \
        --dataset transport_study/datasets/sample/cmod-low1.nc \
        --shot 1160824011 --timestep 800 \
        --checkpoint /path/to/trained/torax/checkpoint \
        --chi_e 2.5 --S_total 1.0
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

from transport_study.modules.profile_predictor.torax_module import ProfilePredictorTorax
from transport_study.modules.profile_predictor.train_configs import (
    PROFILE_PREDICTOR_TORAX_CONFIG,
)

BACKGROUND_COLOR = "#2F2F2F"
FACE_COLOR = "#1A1A1A"
TEXT_COLOR = "white"

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


def _valid_timesteps(shot_ds: xr.Dataset) -> np.ndarray:
    """Timesteps where all inputs are finite and the target profiles are freshly measured."""
    valid = np.ones(shot_ds.sizes["time_idx"], dtype=bool)
    for var in INPUT_VARS:
        valid &= ~np.isnan(shot_ds[var].values)
    valid &= shot_ds["fresh_profiles"].values.astype(bool)
    valid &= ~np.all(np.isnan(shot_ds["ne20_psi"].values), axis=-1)
    valid &= ~np.all(np.isnan(shot_ds["Te_keV_psi"].values), axis=-1)
    return np.flatnonzero(valid)


def _load_timeslice(dataset: str | Path, shot: int, timestep: int) -> xr.Dataset:
    ds_path = Path(dataset)
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found: {ds_path}")
    ds = xr.open_dataset(ds_path)
    shot_ds = ds.sel(shot=shot)

    valid = _valid_timesteps(shot_ds)
    if timestep not in valid:
        if len(valid) == 0:
            raise ValueError(f"Shot {shot} has no timestep with finite inputs and fresh target profiles")
        nearest = valid[np.argmin(np.abs(valid - timestep))]
        raise ValueError(
            f"Timestep {timestep} of shot {shot} has NaN inputs, stale profiles, or all-NaN target profiles. "
            f"Nearest valid timestep: {nearest} (valid range {valid.min()}-{valid.max()}, {len(valid)} total)"
        )
    return shot_ds.isel(time_idx=timestep)


def _build_module(timeslice: xr.Dataset, checkpoint: str | Path | None) -> ProfilePredictorTorax:
    model_cfg = PROFILE_PREDICTOR_TORAX_CONFIG["model_init_config"]
    module = ProfilePredictorTorax(
        nn_width=model_cfg["nn_width"],
        nn_depth=model_cfg["nn_depth"],
        psigrid=tuple(timeslice["psi_n"].values.tolist()),
        torax_config=model_cfg["torax_config"],
        key=jax.random.PRNGKey(model_cfg["prng_seed"]),
    )
    if checkpoint is not None:
        manager = create_default_checkpoint_manager(checkpoint)
        module = restore_model(manager, module)
        logger.info(f"Restored profile predictor from checkpoint {checkpoint}")
    else:
        logger.warning("No checkpoint given, using randomly initialized network weights")
    return module


def _style_axis(ax):
    ax.set_facecolor(FACE_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=TICK_FONTSIZE)
    for spine in ax.spines.values():
        spine.set_edgecolor(TEXT_COLOR)
    ax.grid(True, alpha=0.2, color=TEXT_COLOR)


def plot_torax_evolution(
    dataset: str,
    shot: int,
    timestep: int,
    checkpoint: str | None = None,
    chi_i: float | None = None,
    chi_e: float | None = None,
    D_e: float | None = None,
    V_e: float | None = None,
    S_total: float | None = None,
    n_e_right_bc: float | None = None,
    T_e_right_bc: float | None = None,
    output_dir: str | None = None,
):
    """Plot ne/Te profile evolution across the internal TORAX relaxation steps.

    Args:
        dataset: Path to a NetCDF dataset with dims (shot, time_idx, psi_n).
        shot: Shot number to select.
        timestep: time_idx index of the timeslice to predict.
        checkpoint: Optional checkpoint directory of a trained torax profile predictor.
        chi_i: Optional prescribed ion heat diffusivity [m^2/s], bypasses NN output.
        chi_e: Optional prescribed electron heat diffusivity [m^2/s], bypasses NN output.
        D_e: Optional prescribed particle diffusivity [m^2/s], bypasses NN output.
        V_e: Optional prescribed particle pinch velocity [m/s], bypasses NN output.
        S_total: Optional prescribed gas puff particle source [1e21 /s], bypasses NN output.
        n_e_right_bc: Optional prescribed edge density BC [1e20 m^-3], bypasses NN output.
        T_e_right_bc: Optional prescribed edge temperature BC [keV], bypasses NN output.
        output_dir: Directory to save the figure in (default: current directory).
    """
    timeslice = _load_timeslice(dataset, shot, timestep)
    time_s = float(timeslice["time"].values)
    module = _build_module(timeslice, checkpoint)

    prescribed = {
        "chi_i": chi_i,
        "chi_e": chi_e,
        "D_e": D_e,
        "V_e": V_e,
        "S_total": S_total,
        "n_e_right_bc": n_e_right_bc,
        "T_e_right_bc": T_e_right_bc,
    }
    prescribed_names = {name for name, value in prescribed.items() if value is not None}

    steps, coeffs = module.evolve(timeslice, prescribed=prescribed)
    logger.info(f"TORAX relaxation recorded {len(steps)} states (initial + {len(steps) - 1} steps)")

    psi_n = timeslice["psi_n"].values
    ne_targ = timeslice["ne20_psi"].values
    te_targ = timeslice["Te_keV_psi"].values

    fig, (ax_ne, ax_te) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    def _coeff_label(name: str) -> str:
        source = "prescribed" if name in prescribed_names else "NN"
        return f"{name}={coeffs[name]:.2f} ({source})"

    coeff_line = ", ".join(_coeff_label(name) for name in ["chi_i", "chi_e", "D_e", "V_e", "S_total"])
    bc_line = ", ".join(_coeff_label(name) for name in ["n_e_right_bc", "T_e_right_bc"])
    fig.suptitle(
        f"TORAX profile relaxation - shot {shot} @ t={time_s:.3f}s (time_idx {timestep})\n{coeff_line}\n{bc_line}",
        fontsize=TITLE_FONTSIZE - 4,
        color=TEXT_COLOR,
    )

    # Start colormap above 0 so the earliest steps stay visible on the dark background
    colors = cm.viridis(np.linspace(0.25, 1.0, len(steps)))
    for i, step in enumerate(steps):
        label = f"t={step['t'] * 1e3:.0f} ms"
        ax_ne.plot(step["psi_n"], step["ne20"], color=colors[i], linewidth=2, label=label)
        ax_te.plot(step["psi_n"], step["te_keV"], color=colors[i], linewidth=2, label=label)

    ax_ne.plot(psi_n, ne_targ, color="white", linewidth=3, linestyle="--", label="Measured target")
    ax_te.plot(psi_n, te_targ, color="white", linewidth=3, linestyle="--", label="Measured target")

    ax_ne.set_ylabel(r"$n_e$ [$10^{20}$ m$^{-3}$]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    ax_te.set_ylabel(r"$T_e$ [keV]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    # TORAX states are mapped from rho_norm to psi_n via each state's evolved psi profile,
    # so both the TORAX steps and the measured targets are in psi_n.
    ax_te.set_xlabel(r"$\psi_n$", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)

    for ax in (ax_ne, ax_te):
        _style_axis(ax)
        ax.legend(
            fontsize=LEGEND_FONTSIZE,
            labelcolor=TEXT_COLOR,
            facecolor=BACKGROUND_COLOR,
            edgecolor=TEXT_COLOR,
            ncols=2,
            loc="upper right",
        )

    out_dir = Path(output_dir) if output_dir is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"torax_evolution_{Path(dataset).stem}_{shot}_ts{timestep}.png"
    fig.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot to {plot_path}")
    return plot_path


if __name__ == "__main__":
    fire.Fire(
        {
            "plot_torax_evolution": plot_torax_evolution,
        }
    )
