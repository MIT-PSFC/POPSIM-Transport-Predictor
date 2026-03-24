"""Quick script to find the best 10 cases in collected_results.nc by per-timeslice and per-shot error."""

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

DS_PATH = "/home/zkeith/orcd/scratch/popsim_studies/profopt/working_dir/profopt_sweep/results/collected_results.nc"

ds = xr.open_dataset(DS_PATH)

case_coords = [
    "model_type",
    "training_data",
    "data_normalization",
    "domain_adaptation",
    "freeze_shapes",
    "num_hp_shots",
]

df = pd.DataFrame({var: ds[var].values for var in ds.data_vars})
for c in case_coords:
    if "case_idx" in ds[c].dims:
        df[c] = ds[c].values
    else:
        df[c] = ds[c].item()  # scalar coord, same for all cases

df["case"] = df[case_coords].astype(str).agg(".".join, axis=1)

print("=" * 80)
print("TOP 10 BY PER-TIMESLICE ERROR (err_rel_ts_mean)")
print("=" * 80)
print(
    df.nsmallest(10, "err_rel_ts_mean")[
        ["case", "err_rel_ts_mean", "err_rel_ts_std", "err_rel_ts_med"]
    ].to_string(index=False)
)

print()
print("=" * 80)
print("TOP 10 BY PER-SHOT ERROR (err_rel_shot_mean)")
print("=" * 80)
print(
    df.nsmallest(10, "err_rel_shot_mean")[
        ["case", "err_rel_shot_mean", "err_rel_shot_std", "err_rel_shot_med"]
    ].to_string(index=False)
)

df_hp = df[df["num_hp_shots"] != -1]
print()
print("=" * 80)
print("TOP 10 BY PER-TIMESLICE ERROR (err_rel_ts_mean) — num_hp_shots != -1")
print("=" * 80)
print(
    df_hp.nsmallest(10, "err_rel_ts_mean")[
        ["case", "err_rel_ts_mean", "err_rel_ts_std", "err_rel_ts_med"]
    ].to_string(index=False)
)

print()
print("=" * 80)
print("TOP 10 BY PER-SHOT ERROR (err_rel_shot_mean) — num_hp_shots != -1")
print("=" * 80)
print(
    df_hp.nsmallest(10, "err_rel_shot_mean")[
        ["case", "err_rel_shot_mean", "err_rel_shot_std", "err_rel_shot_med"]
    ].to_string(index=False)
)

# --- Bar charts ---
BACKGROUND_COLOR = "#2F2F2F"
FACE_COLOR = "#1A1A1A"
TEXT_COLOR = "white"

hp_values = sorted(df["num_hp_shots"].unique())
cmap = plt.colormaps["tab10"].resampled(len(hp_values))
hp_color = {v: cmap(i) for i, v in enumerate(hp_values)}


def bar_chart(ax, df_plot, metric, title):
    df_sorted = df_plot.sort_values(metric).reset_index(drop=True)
    colors = [hp_color[v] for v in df_sorted["num_hp_shots"]]
    ax.bar(
        range(len(df_sorted)), df_sorted[metric], color=colors, width=1.0, linewidth=0
    )
    ax.set_title(title, color=TEXT_COLOR)
    ax.set_ylabel(metric, color=TEXT_COLOR)
    ax.set_xticks([])
    ax.tick_params(colors=TEXT_COLOR)
    ax.set_facecolor(FACE_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(TEXT_COLOR)
    handles = [plt.Rectangle((0, 0), 1, 1, color=hp_color[v]) for v in hp_values]
    ax.legend(
        handles,
        [f"num_hp_shots={v}" for v in hp_values],
        fontsize=7,
        loc="upper left",
        facecolor=BACKGROUND_COLOR,
        labelcolor=TEXT_COLOR,
        edgecolor=TEXT_COLOR,
    )


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BACKGROUND_COLOR)
bar_chart(axes[0], df, "err_rel_ts_mean", "Per-timeslice error (all cases, sorted)")
bar_chart(axes[1], df, "err_rel_shot_mean", "Per-shot error (all cases, sorted)")
fig.tight_layout()
plt.savefig("best_cases.png", dpi=150)
print("\nSaved bar charts to best_cases.png")
