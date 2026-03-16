import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

DS_PATH = "/home/zkeith/orcd/scratch/popsim_studies/orchestration_test/working_dir/send_400/results/collected_results.nc"

BACKGROUND_COLOR = "#2F2F2F"
FACE_COLOR = "#1A1A1A"
TEXT_COLOR = "white"

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 16

MODEL_COLORS = {
    "sciml": "#0095ff",
    "unstructured_nn": "#ff4d4d",
}

NORM_COLORS = {
    "raw": "#ff4d4d",
    "z_score": "#0095ff",
    "physics": "#8dff36",
    "coral": "#ff60ec",
}

NORM_LABELS = {
    "raw": "Raw",
    "z_score": "Z-score",
    "physics": "Physics",
    "coral": "CORAL",
}

MODEL_LABELS = {
    "sciml": "SciML",
    "unstructured_nn": "Unstructured NN",
    "p_oh": r"$P_\mathrm{OH}$ baseline",
    "p_rad": r"$P_\mathrm{rad}$ baseline",
}

DA_COLORS = {
    "mixing": "#ad2cfe",
    "transfer": "#00ff5e",
}

DA_LABELS = {
    "mixing": "Mixing",
    "transfer": "Transfer",
}

# Absolute error threshold above which a result is treated as diverged
DIVERGED_THRESHOLD = 1000.0


def important_plots(ds: xr.Dataset):
    """Plot mean absolute shot error vs. number of hyperparameter shots for
    comparing sciml to unstructured neural network

    Args:
        ds: Full results dataset loaded from collected_results.nc.

    Returns:
        Tuple of (fig, ax, sciml_results) where sciml_results is a list of
        (num_hp_shots, err_abs_shot_mean) pairs for the SciML model.
    """

    fig_dir = "fastplots/important_plots"
    os.makedirs(fig_dir, exist_ok=True)

    for data_normalization in ["raw", "z_score", "physics", "coral"]:
        for freeze_submodules in [False, True]:
            for domain_adaptation in ["mixing", "transfer"]:
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor(BACKGROUND_COLOR)
                ax.set_facecolor(FACE_COLOR)
                case_mask = (
                    (ds.domain_adaptation == domain_adaptation)
                    & (ds.freeze_submodules == freeze_submodules)
                    & (ds.data_normalization == data_normalization)
                )
                case_data = ds.isel(case_idx=case_mask)

                hp_shots = sorted(set(case_data.num_hp_shots.values.tolist()))
                hp_shot_to_idx = {v: i for i, v in enumerate(hp_shots)}

                for model_type, color in MODEL_COLORS.items():
                    model_data = case_data.isel(
                        case_idx=(case_data.model_type == model_type)
                    )

                    # Sort by num_hp_shots so lines are drawn left-to-right
                    order = np.argsort(model_data.num_hp_shots.values)
                    shots = model_data.num_hp_shots.values[order].tolist()
                    x = np.array([hp_shot_to_idx[s] for s in shots], dtype=float)
                    y = model_data.err_abs_shot_mean.values[order].copy()
                    yerr = model_data.err_abs_shot_std.values[order].copy()

                    # Mask diverged runs so they don't distort the axes
                    diverged = y > DIVERGED_THRESHOLD
                    y[diverged] = np.nan
                    yerr[diverged] = np.nan

                    ax.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        label=MODEL_LABELS[model_type],
                        color=color,
                        marker="o",
                        markersize=7,
                        linewidth=2,
                        capsize=5,
                        capthick=1.5,
                    )

                ax.set_xticks(range(len(hp_shots)))
                ax.set_xticklabels(
                    [str(s) for s in hp_shots], color=TEXT_COLOR, fontsize=TICK_FONTSIZE
                )
                ax.tick_params(colors=TEXT_COLOR, labelsize=TICK_FONTSIZE)

                ax.set_xlabel(
                    "HP Shots Included", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE
                )
                ax.set_ylabel(
                    "Mean integrated absolute error [MJ s]",
                    color=TEXT_COLOR,
                    fontsize=LABEL_FONTSIZE,
                )
                ax.set_title(
                    f"{data_normalization.upper()} / {domain_adaptation.upper()} - Model Type",
                    color=TEXT_COLOR,
                    fontsize=TITLE_FONTSIZE,
                )

                for spine in ax.spines.values():
                    spine.set_edgecolor(TEXT_COLOR)

                ax.legend(
                    loc="upper right",
                    fontsize=LEGEND_FONTSIZE,
                    labelcolor=TEXT_COLOR,
                    facecolor=BACKGROUND_COLOR,
                    edgecolor=TEXT_COLOR,
                )

                fig.tight_layout()
                fig.savefig(
                    f"{fig_dir}/{data_normalization}_{domain_adaptation}_{freeze_submodules}_error_vs_hp_shots.png",
                    dpi=300,
                    facecolor=fig.get_facecolor(),
                )
                plt.close(fig)


def data_normalization_comparison(ds: xr.Dataset):
    """Compare the performance of different data normalization types.

    For each combination of model type, freeze_submodules, and domain adaptation,
    plots mean absolute shot error vs. num_hp_shots with one line per normalization.

    Args:
        ds: Full results dataset loaded from collected_results.nc.
    """
    fig_dir = "fastplots/data_normalization"
    os.makedirs(fig_dir, exist_ok=True)

    for model_type in ["sciml", "unstructured_nn"]:
        for freeze_submodules in [False, True]:
            for domain_adaptation in ["mixing", "transfer"]:
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor(BACKGROUND_COLOR)
                ax.set_facecolor(FACE_COLOR)

                case_mask = (
                    (ds.domain_adaptation == domain_adaptation)
                    & (ds.freeze_submodules == freeze_submodules)
                    & (ds.model_type == model_type)
                )
                case_data = ds.isel(case_idx=case_mask)

                hp_shots = sorted(set(case_data.num_hp_shots.values.tolist()))
                hp_shot_to_idx = {v: i for i, v in enumerate(hp_shots)}

                for data_normalization, color in NORM_COLORS.items():
                    norm_data = case_data.isel(
                        case_idx=(case_data.data_normalization == data_normalization)
                    )
                    if norm_data.sizes["case_idx"] == 0:
                        continue

                    order = np.argsort(norm_data.num_hp_shots.values)
                    shots = norm_data.num_hp_shots.values[order].tolist()
                    x = np.array([hp_shot_to_idx[s] for s in shots], dtype=float)
                    y = norm_data.err_abs_shot_mean.values[order].copy()
                    yerr = norm_data.err_abs_shot_std.values[order].copy()

                    diverged = y > DIVERGED_THRESHOLD
                    y[diverged] = np.nan
                    yerr[diverged] = np.nan

                    ax.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        label=NORM_LABELS[data_normalization],
                        color=color,
                        marker="o",
                        markersize=7,
                        linewidth=2,
                        capsize=5,
                        capthick=1.5,
                    )

                ax.set_xticks(range(len(hp_shots)))
                ax.set_xticklabels(
                    [str(s) for s in hp_shots], color=TEXT_COLOR, fontsize=TICK_FONTSIZE
                )
                ax.tick_params(colors=TEXT_COLOR, labelsize=TICK_FONTSIZE)

                ax.set_xlabel(
                    "HP Shots Included", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE
                )
                ax.set_ylabel(
                    "Mean integrated absolute error [MJ s]",
                    color=TEXT_COLOR,
                    fontsize=LABEL_FONTSIZE,
                )

                ax.set_title(
                    f"{MODEL_LABELS[model_type]} / {domain_adaptation.upper()} - Data Normalization",
                    color=TEXT_COLOR,
                    fontsize=TITLE_FONTSIZE,
                )

                for spine in ax.spines.values():
                    spine.set_edgecolor(TEXT_COLOR)

                ax.legend(
                    loc="upper right",
                    fontsize=LEGEND_FONTSIZE,
                    labelcolor=TEXT_COLOR,
                    facecolor=BACKGROUND_COLOR,
                    edgecolor=TEXT_COLOR,
                )

                fig.tight_layout()
                fig.savefig(
                    f"{fig_dir}/{model_type}_{domain_adaptation}_{freeze_submodules}_norm_comparison.png",
                    dpi=300,
                    facecolor=fig.get_facecolor(),
                )
                plt.close(fig)


def domain_adaptation_comparison(ds: xr.Dataset):
    """Compare mixing vs. transfer learning domain adaptation strategies.

    For each combination of model type, data normalization, and freeze_submodules,
    plots mean absolute shot error vs. num_hp_shots with one line per domain
    adaptation strategy.

    Args:
        ds: Full results dataset loaded from collected_results.nc.
    """
    fig_dir = "fastplots/domain_adaptation"
    os.makedirs(fig_dir, exist_ok=True)

    for model_type in ["sciml", "unstructured_nn"]:
        for freeze_submodules in [False, True]:
            for data_normalization in ["raw", "z_score", "physics", "coral"]:
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor(BACKGROUND_COLOR)
                ax.set_facecolor(FACE_COLOR)

                case_mask = (
                    (ds.data_normalization == data_normalization)
                    & (ds.freeze_submodules == freeze_submodules)
                    & (ds.model_type == model_type)
                    & (ds.num_hp_shots >= 0)
                )
                case_data = ds.isel(case_idx=case_mask)

                hp_shots = sorted(set(case_data.num_hp_shots.values.tolist()))
                hp_shot_to_idx = {v: i for i, v in enumerate(hp_shots)}

                for domain_adaptation, color in DA_COLORS.items():
                    da_data = case_data.isel(
                        case_idx=(case_data.domain_adaptation == domain_adaptation)
                    )
                    if da_data.sizes["case_idx"] == 0:
                        continue

                    order = np.argsort(da_data.num_hp_shots.values)
                    shots = da_data.num_hp_shots.values[order].tolist()
                    x = np.array([hp_shot_to_idx[s] for s in shots], dtype=float)
                    y = da_data.err_abs_shot_mean.values[order].copy()
                    yerr = da_data.err_abs_shot_std.values[order].copy()

                    diverged = y > DIVERGED_THRESHOLD
                    y[diverged] = np.nan
                    yerr[diverged] = np.nan

                    ax.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        label=DA_LABELS[domain_adaptation],
                        color=color,
                        marker="o",
                        markersize=7,
                        linewidth=2,
                        capsize=5,
                        capthick=1.5,
                    )

                ax.set_xticks(range(len(hp_shots)))
                ax.set_xticklabels(
                    [str(s) for s in hp_shots], color=TEXT_COLOR, fontsize=TICK_FONTSIZE
                )
                ax.tick_params(colors=TEXT_COLOR, labelsize=TICK_FONTSIZE)

                ax.set_xlabel(
                    "HP Shots Included", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE
                )
                ax.set_ylabel(
                    "Mean integrated absolute error [MJ s]",
                    color=TEXT_COLOR,
                    fontsize=LABEL_FONTSIZE,
                )

                ax.set_title(
                    f"{MODEL_LABELS[model_type]} / {NORM_LABELS[data_normalization]} - Domain Adaptation",
                    color=TEXT_COLOR,
                    fontsize=TITLE_FONTSIZE,
                )

                for spine in ax.spines.values():
                    spine.set_edgecolor(TEXT_COLOR)

                ax.legend(
                    loc="upper right",
                    fontsize=LEGEND_FONTSIZE,
                    labelcolor=TEXT_COLOR,
                    facecolor=BACKGROUND_COLOR,
                    edgecolor=TEXT_COLOR,
                )

                fig.tight_layout()
                fig.savefig(
                    f"{fig_dir}/{model_type}_{data_normalization}_{freeze_submodules}_da_comparison.png",
                    dpi=300,
                    facecolor=fig.get_facecolor(),
                )
                plt.close(fig)


def submodule_freezing_comparison(ds: xr.Dataset):
    """
    For the sciml model, we can either freeze or not freeze the submodules.

    For each unique combination of model type, data normalization, and domain adaptation,
    where there exists data for both freeze_submodules=True and freeze_submodules=False,
    plot the difference in mean absolute shot error between the two freezing strategies as a function of num_hp_shots.

    This should put all lines on the same plot, with one line per combination of model type, data normalization, and domain adaptation, and the x-axis being num_hp_shots and the y-axis being (err_abs_shot_mean with freeze) - (err_abs_shot_mean without freeze).

    """
    fig_dir = "fastplots/submodule_freezing"
    os.makedirs(fig_dir, exist_ok=True)

    # Filter out sentinel hp_shots=-1
    ds_filtered = ds.isel(case_idx=(ds.num_hp_shots >= 0))

    for model_type in ["sciml", "unstructured_nn"]:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        ax.set_facecolor(FACE_COLOR)

        model_ds = ds_filtered.isel(case_idx=(ds_filtered.model_type == model_type))

        # Build a combined colormap: 4 norms x 2 DA strategies = 8 lines
        norm_list = ["raw", "z_score", "physics", "coral"]
        da_list = ["mixing", "transfer"]
        # Use norm color with solid/dashed linestyle for DA
        da_linestyles = {"mixing": "-", "transfer": "--"}

        hp_shots_all = sorted(set(model_ds.num_hp_shots.values.tolist()))
        hp_shot_to_idx = {v: i for i, v in enumerate(hp_shots_all)}

        any_line_plotted = False
        for data_normalization in norm_list:
            for domain_adaptation in da_list:
                frozen = model_ds.isel(
                    case_idx=(
                        (model_ds.data_normalization == data_normalization)
                        & (model_ds.domain_adaptation == domain_adaptation)
                        & (model_ds.freeze_submodules)
                    )
                )
                unfrozen = model_ds.isel(
                    case_idx=(
                        (model_ds.data_normalization == data_normalization)
                        & (model_ds.domain_adaptation == domain_adaptation)
                        & (not model_ds.freeze_submodules)
                    )
                )

                if frozen.sizes["case_idx"] == 0 or unfrozen.sizes["case_idx"] == 0:
                    continue

                # Align on shared hp_shots
                frozen_shots = frozen.num_hp_shots.values.tolist()
                unfrozen_shots = unfrozen.num_hp_shots.values.tolist()
                shared_shots = sorted(set(frozen_shots) & set(unfrozen_shots))
                if not shared_shots:
                    continue

                frozen_order = np.argsort(frozen.num_hp_shots.values)
                unfrozen_order = np.argsort(unfrozen.num_hp_shots.values)
                frozen_sorted = dict(
                    zip(
                        frozen.num_hp_shots.values[frozen_order].tolist(),
                        frozen.err_abs_shot_mean.values[frozen_order].tolist(),
                        strict=True,
                    )
                )
                unfrozen_sorted = dict(
                    zip(
                        unfrozen.num_hp_shots.values[unfrozen_order].tolist(),
                        unfrozen.err_abs_shot_mean.values[unfrozen_order].tolist(),
                        strict=True,
                    )
                )

                x = np.array([hp_shot_to_idx[s] for s in shared_shots], dtype=float)
                diff = np.array(
                    [
                        frozen_sorted[s] - unfrozen_sorted[s]
                        if frozen_sorted[s] < DIVERGED_THRESHOLD
                        and unfrozen_sorted[s] < DIVERGED_THRESHOLD
                        else np.nan
                        for s in shared_shots
                    ]
                )

                color = NORM_COLORS[data_normalization]
                linestyle = da_linestyles[domain_adaptation]
                label = f"{NORM_LABELS[data_normalization]} / {DA_LABELS[domain_adaptation]}"

                ax.plot(
                    x,
                    diff,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    marker="o",
                    markersize=7,
                    label=label,
                )
                any_line_plotted = True

        ax.axhline(0, color=TEXT_COLOR, linewidth=0.8, linestyle=":")

        ax.set_xticks(range(len(hp_shots_all)))
        ax.set_xticklabels(
            [str(s) for s in hp_shots_all], color=TEXT_COLOR, fontsize=TICK_FONTSIZE
        )
        ax.tick_params(colors=TEXT_COLOR, labelsize=TICK_FONTSIZE)
        ax.set_xlim(left=0)

        ax.set_xlabel("HP Shots Included", color=TEXT_COLOR, fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(
            r"$\Delta$ Error frozen - unfrozen (+ means unfrozen is better) [MJ s]",
            color=TEXT_COLOR,
            fontsize=LABEL_FONTSIZE,
        )
        ax.set_title(
            f"{MODEL_LABELS[model_type]} - Submodule Freezing Effect",
            color=TEXT_COLOR,
            fontsize=TITLE_FONTSIZE,
        )

        for spine in ax.spines.values():
            spine.set_edgecolor(TEXT_COLOR)

        if any_line_plotted:
            ax.legend(
                loc="upper left",
                fontsize=LEGEND_FONTSIZE - 2,
                labelcolor=TEXT_COLOR,
                facecolor=BACKGROUND_COLOR,
                edgecolor=TEXT_COLOR,
            )

        fig.tight_layout()
        fig.savefig(
            f"{fig_dir}/{model_type}_freezing_comparison.png",
            dpi=300,
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)


def vibe_check():
    ds_path = "/home/zkeith/orcd/scratch/popsim_studies/orchestration_test/working_dir/send_400/results/case.sciml.td_cmod_tcv.dn_physics.da_mixing.freezesub_False.hp_33/result_data.nc"
    ds = xr.open_dataset(ds_path)
    for shot in ds.shot.values:
        ds_shot = ds.where(ds.shot == shot, drop=True)
        fig, ax = plt.subplots()
        time = ds_shot.time.values
        targ = ds_shot["Wtot_MJ_targ"]
        pred = ds_shot["Wtot_MJ_pred"]
        ax.plot(time, targ, label="Target")
        ax.plot(time, pred, label="Prediction")
        ax.set_title(f"Shot {shot}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"vibe_check/shot_{shot}.png")
        plt.close(fig)


if __name__ == "__main__":
    ds = xr.open_dataset(DS_PATH)
    important_plots(ds)
    data_normalization_comparison(ds)
    domain_adaptation_comparison(ds)
    submodule_freezing_comparison(ds)
    vibe_check()
