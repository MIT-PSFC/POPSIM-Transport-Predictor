import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

BACKGROUND_COLOR = "#2F2F2F"
FACE_COLOR = "#1A1A1A"
TEXT_COLOR = "white"

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18


def profile_comparison(  # noqa: PLR0915
    profile_dir: str,
    ds_targ: xr.Dataset,
    ds_pred: xr.Dataset | None = None,
):
    """Plot individual profile signals from the dataset"""

    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)

    # Compute global y-limits across all shots for consistent axes
    if ds_pred is None:
        ylim_ne = (0, float(np.nanmax(ds_targ["ne20_psi"].values)) * 1.1)
        ylim_te = (0, float(np.nanmax(ds_targ["Te_keV_psi"].values)) * 1.1)
    else:
        ylim_ne = (
            0,
            float(np.nanmax([ds_targ["ne20_psi"].values, ds_pred["ne20_psi"].values]))
            * 1.1,
        )
        ylim_te = (
            0,
            float(
                np.nanmax([ds_targ["Te_keV_psi"].values, ds_pred["Te_keV_psi"].values])
            )
            * 1.1,
        )

    for shot in ds_targ["shot"].data:
        shot_ds_targ = ds_targ.where(ds_targ["shot"] == shot, drop=True).squeeze()
        shot_ds_pred = ds_pred.sel(shot=shot) if ds_pred is not None else None
        shot_dir = os.path.join(profile_dir, str(shot))
        if not os.path.exists(shot_dir):
            os.makedirs(shot_dir)

        # Get psi coordinates
        psi = shot_ds_targ["psi"].values

        for _, time in enumerate(shot_ds_targ["time"].values):
            time_ds_targ = shot_ds_targ.where(
                shot_ds_targ["time"] == time, drop=True
            ).squeeze()
            time_ds_pred = shot_ds_pred.sel(time=time) if ds_pred is not None else None

            ne_profile_targ = time_ds_targ["ne20_psi"].values
            te_profile_targ = time_ds_targ["Te_keV_psi"].values
            ne_profile_pred = (
                time_ds_pred["ne20_psi"].values
                if ds_pred is not None
                else np.full_like(ne_profile_targ, np.nan)
            )
            te_profile_pred = (
                time_ds_pred["Te_keV_psi"].values
                if ds_pred is not None
                else np.full_like(te_profile_targ, np.nan)
            )

            # Skip if profiles are all NaN
            if np.all(np.isnan(ne_profile_targ)) and np.all(np.isnan(te_profile_targ)):
                continue

            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            fig.patch.set_facecolor(BACKGROUND_COLOR)

            fig.suptitle(
                f"TCV Shot {shot} @ t={time:.3f}s",
                fontsize=TITLE_FONTSIZE,
                color=TEXT_COLOR,
            )

            # Density profile
            ax_ne = axes[0]
            ax_ne.plot(psi, ne_profile_targ, label="ne20", color="cyan", linewidth=2)
            ax_ne.plot(
                psi,
                ne_profile_pred,
                label="ne20_pred",
                color="blue",
                linewidth=2,
                linestyle="--",
            )
            ax_ne.set_ylabel(
                r"$n_e$ [$10^{20}$ m$^{-3}$]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR
            )
            ax_ne.set_ylim(ylim_ne)
            ax_ne.legend(
                fontsize=LEGEND_FONTSIZE,
                facecolor=BACKGROUND_COLOR,
                edgecolor=BACKGROUND_COLOR,
                loc="upper right",
            )

            # Temperature profile
            ax_te = axes[1]
            ax_te.plot(psi, te_profile_targ, label="Te", color="red", linewidth=2)
            ax_te.plot(
                psi,
                te_profile_pred,
                label="Te_pred",
                color="orange",
                linewidth=2,
                linestyle="--",
            )
            ax_te.set_ylabel(r"$T_e$ [keV]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
            ax_te.set_xlabel(r"$\psi_n$", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
            ax_te.set_ylim(ylim_te)
            ax_te.legend(
                fontsize=LEGEND_FONTSIZE,
                facecolor=BACKGROUND_COLOR,
                edgecolor=BACKGROUND_COLOR,
                loc="upper right",
            )

            for ax in axes:
                ax.set_facecolor(FACE_COLOR)
                ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
                ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
                ax.set_xlim(0, 1.2)
                for spine in ax.spines.values():
                    spine.set_color(TEXT_COLOR)
                try:
                    for text in ax.get_legend().get_texts():
                        text.set_color(TEXT_COLOR)
                except AttributeError:
                    pass

            fig.tight_layout()
            fig.savefig(os.path.join(shot_dir, f"t{time:.3f}.png"))
            plt.close(fig)
