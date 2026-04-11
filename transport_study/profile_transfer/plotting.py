import os

import matplotlib.pyplot as plt
import netCDF4  # noqa: F401
import numpy as np
import xarray as xr
from matplotlib import cm

result_data = "/home/zkeith/orcd/scratch/popsim_studies/profopt/working_dir/profopt_sweep/results/case.unstructured_nn.td_exnihilo.freeze_True.hp_-1/result_data.nc"
plot_dir = "/orcd/home/002/zkeith/proj/popsim_dirs/POPSIM-Transport-Predictor/scratch/fastplots"
os.makedirs(plot_dir, exist_ok=True)

BACKGROUND_COLOR = "#2F2F2F"
FACE_COLOR = "#1A1A1A"
TEXT_COLOR = "white"

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 13

VAR_LABELS = {
    "Te_keV_psi": r"$T_e$ [keV]",
    "ne20_psi": r"$n_e$ [$10^{20}$ m$^{-3}$]",
}

ds = xr.open_dataset(result_data)
ds = ds.sortby("shot")

psi_vals = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
colors = cm.hsv(np.linspace(0.0, 0.85, len(psi_vals)))

for shot in ds.shot.values:
    ds_shot = ds.sel(shot=shot)
    time = ds_shot.time.values
    valid = ~np.isnan(time)
    sort_order = np.argsort(time[valid])

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    for i, var in enumerate(["Te_keV_psi", "ne20_psi"]):
        ax = axes[i]
        ax.set_facecolor(FACE_COLOR)

        pred = ds_shot[f"{var}_pred"]
        targ = ds_shot[f"{var}_targ"]

        for j, psi in enumerate(psi_vals):
            color = colors[j]
            pred_psi = pred.sel(psi_n=psi, method="nearest").values[valid][sort_order]
            targ_psi = targ.sel(psi_n=psi, method="nearest").values[valid][sort_order]
            t_sorted = time[valid][sort_order]
            ax.plot(
                t_sorted, pred_psi, color=color, linewidth=1.5, label=f"psi={psi:.1f}"
            )
            ax.plot(
                t_sorted,
                targ_psi,
                color=color,
                linewidth=1.5,
                linestyle="dashed",
                alpha=0.6,
            )

        ax.set_title(
            f"Shot {shot} — {VAR_LABELS[var]}",
            color=TEXT_COLOR,
            fontsize=TITLE_FONTSIZE,
        )
        ax.set_xlabel("Time [s]", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(VAR_LABELS[var], color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
        ax.tick_params(colors=TEXT_COLOR, labelsize=TICK_FONTSIZE)
        for spine in ax.spines.values():
            spine.set_edgecolor(TEXT_COLOR)

        legend = ax.legend(
            loc="upper right",
            fontsize=LEGEND_FONTSIZE,
            labelcolor=TEXT_COLOR,
            facecolor=BACKGROUND_COLOR,
            edgecolor=TEXT_COLOR,
            title="solid=pred  dashed=targ",
        )
        legend.get_title().set_color(TEXT_COLOR)
        legend.get_title().set_fontsize(LEGEND_FONTSIZE - 1)

    fig.tight_layout()
    fig.savefig(
        f"{plot_dir}/shot_{shot}_pred_vs_true.png",
        dpi=150,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

print(f"Saved {len(ds.shot.values)} plots to {plot_dir}")
