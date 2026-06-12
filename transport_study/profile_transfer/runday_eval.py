"""Evaluate all profile predictor checkpoints against IDA profiles in the reference shot.

Runs each predictor on measured (non-prog) inputs from the reference shot and computes
the rho-integrated relative error against IDA ne/Te profiles. Output formatted similar
to best_cases.py.

Each evaluation is submitted as an independent SLURM job; the main script waits for all
results before printing the ranked tables and bar chart.

Usage:
    # Submit jobs and wait for results:
    python runday_eval.py runday_eval --profopt_models_dir <path>

    # (Internal) called by each SLURM worker:
    python runday_eval.py _eval_worker --checkpoint_dir <path> --result_path <path>
"""

import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

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
from transport_study.orchestration.slurm_utils import (
    count_idle_gpus,
    resources_available,
)
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
    "data_normalization",
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

POLL_INTERVAL_S = 15


def load_valid_timeslices(ref_shot_path: Path | str) -> xr.Dataset:
    """Return timeslices where all profile-predictor inputs and IDA profiles are non-NaN."""
    ds = xr.open_dataset(ref_shot_path).isel(shot=0)

    valid = np.ones(ds.sizes["time_idx"], dtype=bool)
    for v in INPUT_VARS:
        valid &= ds[v].notnull().values
    for v in ["ne20_rho", "Te_keV_rho"]:
        valid &= ds[v].notnull().all(dim="rho").values

    ds_valid = ds.isel(time_idx=valid)
    logger.info(f"Valid timeslices: {int(valid.sum())} / {ds.sizes['time_idx']}")
    return ds_valid


def evaluate_checkpoint(checkpoint_dir: Path | str, ds_valid: xr.Dataset) -> dict | None:
    """Evaluate one checkpoint against IDA profiles. Returns metrics dict or None on failure."""
    try:
        predictor = restore_profile_predictor_from_checkpoint(checkpoint_dir)
    except Exception:
        logger.warning(f"Failed to restore {Path(checkpoint_dir).name}:\n{traceback.format_exc()}")
        return None

    # Predictor evaluates on its own rhogrid; targets must be interpolated onto it.
    rho_pred = np.array(predictor.rhogrid)  # (n_pred_rho,)
    rho_ds = ds_valid["rho"].values  # (n_ds_rho,)
    n_ts = ds_valid.sizes["time_idx"]
    rho_tiled = jnp.tile(jnp.array(rho_pred), (n_ts, 1))

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
        rho=rho_tiled,
    )

    def _predict(inp):
        out = predictor(inp)
        return out.ne.data, out.te.data

    try:
        ne_pred, te_pred = jax.vmap(_predict)(inputs_batched)
    except Exception:
        logger.warning(f"Failed to run {Path(checkpoint_dir).name}:\n{traceback.format_exc()}")
        return None

    ne_pred = np.array(ne_pred)  # (n_ts, n_pred_rho)
    te_pred = np.array(te_pred)

    # Interpolate profile targets from dataset rho grid onto predictor rhogrid
    ne_targ_raw = ds_valid["ne20_rho"].values  # (n_ts, n_ds_rho)
    te_targ_raw = ds_valid["Te_keV_rho"].values
    ne_targ = np.stack([np.interp(rho_pred, rho_ds, ne_targ_raw[i]) for i in range(n_ts)])
    te_targ = np.stack([np.interp(rho_pred, rho_ds, te_targ_raw[i]) for i in range(n_ts)])

    # Rho-integrated relative error per timeslice
    ne_err_rel = np.trapezoid(np.abs(ne_pred - ne_targ) / (np.abs(ne_targ) + 0.1), rho_pred, axis=-1)
    te_err_rel = np.trapezoid(np.abs(te_pred - te_targ) / (np.abs(te_targ) + 0.1), rho_pred, axis=-1)
    err_rel_ts = 0.5 * (ne_err_rel + te_err_rel)

    return {
        "err_rel_ts_mean": float(np.mean(err_rel_ts)),
        "err_rel_ts_std": float(np.std(err_rel_ts)),
        "err_rel_ts_med": float(np.median(err_rel_ts)),
        "ne_err_rel_mean": float(np.mean(ne_err_rel)),
        "te_err_rel_mean": float(np.mean(te_err_rel)),
    }


def _eval_worker(checkpoint_dir: Path | str, result_path: Path | str, ref_shot_path: Path | str):
    """Entry point for each SLURM worker. Evaluates one checkpoint and writes a JSON result."""
    ds_valid = load_valid_timeslices(ref_shot_path)
    metrics = evaluate_checkpoint(checkpoint_dir, ds_valid)
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    payload = metrics if metrics is not None else {"error": "evaluation failed"}
    with open(result_path, "w") as f:
        json.dump(payload, f)
    logger.info(f"Saved result to {result_path}")


def _launch_eval_job(
    checkpoint_dir: Path | str,
    result_path: Path | str,
    ref_shot_path: Path | str,
    log_dir: Path | str,
    partition: str,
) -> None:
    """Submit a SLURM job that runs _eval_worker for one checkpoint."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    name = Path(checkpoint_dir).name
    log_path = Path(log_dir) / f"{name}.log"

    # Write a small launcher script so the sbatch command stays clean
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix=f"rde_{name[:20]}_", dir=log_dir) as f:
        f.write(
            f"from transport_study.profile_transfer.runday_eval import _eval_worker\n"
            f"_eval_worker(\n"
            f"    checkpoint_dir={checkpoint_dir!r},\n"
            f"    result_path={result_path!r},\n"
            f"    ref_shot_path={ref_shot_path!r},\n"
            f")\n"
        )
        script_path = f.name

    sbatch_script = f"""\
#!/bin/bash
#SBATCH --job-name=rde.{name[:40]}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --exclude=node2301,node2101
#SBATCH --output={log_path}
#SBATCH --error={log_path}

export WANDB_MODE=offline
{sys.executable} {script_path}
rm -f {script_path}
"""

    result = subprocess.run(["sbatch"], input=sbatch_script, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed for {name}: {result.stderr}")
    else:
        logger.info(f"Submitted {name}: {result.stdout.strip()}")


def runday_eval(  # noqa: PLR0915
    profopt_models_dir: Path | str,
    partition: str = config.partition,
):
    ref_shot_path = Path(config.scratch_dir) / "predict_first/raw_data" / f"{config.ref_shot}.nc"
    ref_shot = str(config.ref_shot)

    results_dir = Path(config.scratch_dir) / "runday_eval" / "results"
    log_dir = Path(config.scratch_dir) / "runday_eval" / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_names = sorted(p.name for p in Path(profopt_models_dir).iterdir() if p.name.startswith("case."))

    # Map each valid case name -> expected result JSON path
    case_result_paths: dict[str, Path] = {}
    for name in checkpoint_names:
        checkpoint_dir = Path(profopt_models_dir) / name
        try:
            checkpoint_to_profile_case(checkpoint_dir)
        except Exception:
            logger.warning(f"Could not parse case from {name}, skipping")
            continue
        case_result_paths[name] = results_dir / f"{name}.json"

    total = len(case_result_paths)
    logger.info(f"Total cases to evaluate: {total}")

    # Submit jobs for any case whose result doesn't exist yet, respecting cluster availability
    pending = [name for name, result_path in case_result_paths.items() if not result_path.exists()]
    submitted = 0
    for name in pending:
        while not resources_available(partition=partition):
            idle = count_idle_gpus(partition=partition)
            logger.info(
                f"No resources available (idle GPUs after buffer: {idle}) — "
                f"waiting {POLL_INTERVAL_S}s before submitting next job "
                f"({submitted}/{len(pending)} submitted so far)"
            )
            time.sleep(POLL_INTERVAL_S)
        _launch_eval_job(
            checkpoint_dir=Path(profopt_models_dir) / name,
            result_path=case_result_paths[name],
            ref_shot_path=ref_shot_path,
            log_dir=log_dir,
            partition=partition,
        )
        submitted += 1

    if submitted == 0:
        logger.info("All results already present, skipping submission.")
    else:
        logger.info(f"Submitted {submitted} SLURM jobs.")

    # Poll until all results exist
    while True:
        done = sum(1 for p in case_result_paths.values() if p.exists())
        remaining = total - done
        if remaining == 0:
            logger.info(f"All {total} results collected.")
            break
        logger.info(f"Progress: {done}/{total} done, {remaining} remaining — checking again in {POLL_INTERVAL_S}s")
        time.sleep(POLL_INTERVAL_S)

    # Collect results into a DataFrame
    rows = []
    for name, result_path in case_result_paths.items():
        with open(result_path) as f:
            payload = json.load(f)
        if "error" in payload:
            logger.warning(f"Case {name} failed: {payload['error']}")
            continue

        checkpoint_dir = Path(profopt_models_dir) / name
        case = checkpoint_to_profile_case(checkpoint_dir)
        row = {
            "model_type": case.model_type,
            "training_data": case.training_data,
            "data_normalization": case.data_normalization,
            "domain_adaptation": case.domain_adaptation or "",
            "freeze_shapes": case.freeze_shapes,
            "num_hp_shots": case.num_hp_shots,
            **payload,
        }
        row["case"] = ".".join(str(row[c]) for c in CASE_COORDS)
        rows.append(row)

    if not rows:
        logger.error("No successful results to display.")
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
    bar_chart(axes[0], df, "All cases - sorted by err_rel_ts_mean")
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
            "_eval_worker": _eval_worker,
        }
    )
