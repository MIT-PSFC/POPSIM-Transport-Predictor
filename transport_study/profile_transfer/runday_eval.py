"""Evaluate all profile predictor checkpoints against IDA profiles in the reference shot.

Runs each predictor on measured (non-prog) inputs from the reference shot and computes
the psi-integrated relative error against IDA ne/Te profiles. Output formatted similar
to best_cases.py.
"""

import os
import traceback

import fire
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from transport_study.config import config
from transport_study.modules.profile_predictor.module import Inputs
from transport_study.profile_transfer.restore_predictor import (
    checkpoint_to_profile_case,
    restore_profile_predictor_from_checkpoint,
)

BACKGROUND_COLOR = "#2F2F2F"
FACE_COLOR = "#1A1A1A"
TEXT_COLOR = "white"

CASE_COORDS = [
    "model_type",
    "training_data",
    "domain_adaptation",
    "freeze_shapes",
    "num_hp_shots",
]

INPUT_VARS = [
    "Ip_MA",
    "B0",
    "betan",
    "ne20_edge",
    "R0",
    "a_minor",
    "kappa",
    "delta_top",
    "delta_bot",
]


def load_valid_timeslices(ref_shot_path: str) -> xr.Dataset:
    """Return timeslices where all profile-predictor inputs and IDA profiles are non-NaN."""
    ds = xr.open_dataset(ref_shot_path).isel(shot=0)

    valid = np.ones(ds.sizes["time_idx"], dtype=bool)
    for v in INPUT_VARS:
        valid &= ds[v].notnull().values
    for v in ["ne20_psi", "Te_keV_psi"]:
        valid &= ds[v].notnull().all(dim="psi_n").values

    ds_valid = ds.isel(time_idx=valid)
    logger.info(f"Valid timeslices: {int(valid.sum())} / {ds.sizes['time_idx']}")
    return ds_valid


def evaluate_checkpoint(checkpoint_dir: str, ds_valid: xr.Dataset) -> dict | None:
    """Evaluate one checkpoint against IDA profiles. Returns metrics dict or None on failure."""
    try:
        predictor = restore_profile_predictor_from_checkpoint(checkpoint_dir)
    except Exception:
        logger.warning(
            f"Failed to restore {os.path.basename(checkpoint_dir)}:\n{traceback.format_exc()}"
        )
        return None

    # Use dataset psi_n grid so predictions land on the same grid as targets.
    psi_n = ds_valid["psi_n"].values  # (n_psi,)
    n_ts = ds_valid.sizes["time_idx"]
    psi_tiled = jnp.tile(jnp.array(psi_n), (n_ts, 1))

    inputs_batched = Inputs(
        Ip=jnp.array(ds_valid["Ip_MA"].values),
        B0=jnp.array(ds_valid["B0"].values),
        betan=jnp.array(ds_valid["betan"].values),
        ne20=jnp.array(ds_valid["ne20_edge"].values),
        R0=jnp.array(ds_valid["R0"].values),
        a_minor=jnp.array(ds_valid["a_minor"].values),
        kappa=jnp.array(ds_valid["kappa"].values),
        delta_top=jnp.array(ds_valid["delta_top"].values),
        delta_bot=jnp.array(ds_valid["delta_bot"].values),
        psi=psi_tiled,
    )

    def _predict(inp):
        out = predictor(inp)
        return out.ne.data, out.te.data

    try:
        ne_pred, te_pred = jax.vmap(_predict)(inputs_batched)
    except Exception:
        logger.warning(
            f"Failed to run {os.path.basename(checkpoint_dir)}:\n{traceback.format_exc()}"
        )
        return None

    ne_pred = np.array(ne_pred)  # (n_ts, n_psi)
    te_pred = np.array(te_pred)

    ne_targ = ds_valid["ne20_psi"].values  # (n_ts, n_psi)
    te_targ = ds_valid["Te_keV_psi"].values

    # Psi-integrated relative error per timeslice
    ne_err_rel = np.trapezoid(
        np.abs(ne_pred - ne_targ) / (np.abs(ne_targ) + 0.1), psi_n, axis=-1
    )
    te_err_rel = np.trapezoid(
        np.abs(te_pred - te_targ) / (np.abs(te_targ) + 0.1), psi_n, axis=-1
    )
    err_rel_ts = 0.5 * (ne_err_rel + te_err_rel)

    return {
        "err_rel_ts_mean": float(np.mean(err_rel_ts)),
        "err_rel_ts_std": float(np.std(err_rel_ts)),
        "err_rel_ts_med": float(np.median(err_rel_ts)),
        "ne_err_rel_mean": float(np.mean(ne_err_rel)),
        "te_err_rel_mean": float(np.mean(te_err_rel)),
    }


def runday_eval(  # noqa: PLR0915
    profopt_models_dir: str,
):
    ref_shot_path = os.path.join(
        config.scratch_dir, "predict_first/raw_data", f"{config.ref_shot}.nc"
    )
    ds_valid = load_valid_timeslices(ref_shot_path)

    ref_shot = os.path.basename(ref_shot_path).split(".")[0]
    checkpoint_dirs = sorted(
        d for d in os.listdir(profopt_models_dir) if d.startswith("case.")
    )

    rows = []
    for name in checkpoint_dirs:
        checkpoint_dir = os.path.join(profopt_models_dir, name)
        try:
            case = checkpoint_to_profile_case(checkpoint_dir)
        except Exception:
            logger.warning(f"Could not parse case from {name}")
            continue

        logger.info(f"Evaluating {name}")
        metrics = evaluate_checkpoint(checkpoint_dir, ds_valid)
        if metrics is None:
            continue

        row = {
            "model_type": case.model_type,
            "training_data": case.training_data,
            "domain_adaptation": str(case.domain_adaptation),
            "freeze_shapes": case.freeze_shapes,
            "num_hp_shots": case.num_hp_shots,
            **metrics,
        }
        row["case"] = ".".join(str(row[c]) for c in CASE_COORDS)
        rows.append(row)
        logger.info(f"  err_rel_ts_mean={metrics['err_rel_ts_mean']:.4f}")

    if not rows:
        logger.error("No results collected.")
        return

    df = pd.DataFrame(rows)

    # --- Print tables ---
    cols = [
        "case",
        "err_rel_ts_mean",
        "err_rel_ts_std",
        "err_rel_ts_med",
        "ne_err_rel_mean",
        "te_err_rel_mean",
    ]

    print("=" * 80)
    print(f"TOP 20 BY MEAN REL ERROR — ref shot {ref_shot} IDA profiles (all cases)")
    print("=" * 80)
    print(df.nsmallest(20, "err_rel_ts_mean")[cols].to_string(index=False))

    df_hp = df[df["num_hp_shots"] != -1]
    if not df_hp.empty:
        print()
        print("=" * 80)
        print("TOP 20 BY MEAN REL ERROR — num_hp_shots != -1 (domain adaptation cases)")
        print("=" * 80)
        print(df_hp.nsmallest(20, "err_rel_ts_mean")[cols].to_string(index=False))

    # --- Bar chart ---
    hp_values = sorted(df["num_hp_shots"].unique())
    cmap = plt.colormaps["tab10"].resampled(len(hp_values))
    hp_color = {v: cmap(i) for i, v in enumerate(hp_values)}

    def bar_chart(ax, df_plot, title):
        df_sorted = df_plot.sort_values("err_rel_ts_mean").reset_index(drop=True)
        colors = [hp_color[v] for v in df_sorted["num_hp_shots"]]
        ax.bar(
            range(len(df_sorted)),
            df_sorted["err_rel_ts_mean"],
            color=colors,
            width=1.0,
            linewidth=0,
        )
        ax.set_title(title, color=TEXT_COLOR)
        ax.set_ylabel("err_rel_ts_mean", color=TEXT_COLOR)
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
    bar_chart(axes[0], df, "All cases — sorted by err_rel_ts_mean")
    bar_chart(
        axes[1],
        df_hp if not df_hp.empty else df,
        "Domain adaptation (num_hp_shots != -1)",
    )
    fig.suptitle(
        f"Profile predictor performance vs. ref shot {ref_shot} IDA profiles",
        color=TEXT_COLOR,
    )
    fig.tight_layout()
    plt.savefig("runday_eval.png", dpi=150)
    print("\nSaved bar chart to runday_eval.png")


if __name__ == "__main__":
    fire.Fire(
        {
            "runday_eval": runday_eval,
        }
    )
