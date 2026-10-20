"""Kappa-binned per-shot error analysis for profile transfer studies.

Bins the target-device test shots by their time-averaged elongation and
reports the per-shot relative error (error_rel_shot, the study headline
metric) per bin, one row per case. Built for the circular-vs-miller
geometry comparison: if the miller geometry specifically closes the
high-elongation gap, the delta between matched geom_circular and
geom_miller cases should grow with kappa.

Works on any study layout that writes results/case.*/result_data.nc with
per-shot error variables and a shot coordinate matching the source zarr.

Usage:
    python transport_study/scripts/kappa_binned_error.py run \
        --results_dir /path/to/working_dir/results \
        --dataset_path /path/to/mast/ds.zarr \
        --case_glob "case.*targ_0*" \
        --out_dir /path/to/output

Writes a markdown table and a png figure per invocation.
"""

import fnmatch
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loguru import logger

from transport_study.plot_style import BACKGROUND_COLOR, TEXT_COLOR, style_axis

DEFAULT_METRIC = "error_rel_shot"


def per_shot_kappa(dataset_path: str) -> dict[int, float]:
    """Time-averaged kappa per shot from the device zarr."""
    ds = xr.open_zarr(dataset_path)
    kappa = ds["kappa"].mean(dim="time_idx", skipna=True).compute()
    return {int(s): float(k) for s, k in zip(kappa["shot"].values, kappa.values, strict=True)}


def load_case_errors(case_dir: Path, metric: str, normalize_by_duration: bool) -> dict[int, float] | None:
    """Per-shot metric values for one case, None when no result file.

    normalize_by_duration divides each shot's value by its scored time span,
    turning the study's time-integral metrics into time averages. MAST shot
    duration correlates with kappa (low-kappa shots run longer), so raw
    integrals bias kappa-binned comparisons.
    """
    result_path = case_dir / "result_data.nc"
    if not result_path.exists():
        return None
    ds = xr.load_dataset(result_path)
    if metric not in ds:
        logger.warning(f"{case_dir.name}: no {metric} variable, skipping")
        return None
    vals = ds[metric]
    extra_dims = [d for d in vals.dims if d != "shot"]
    if extra_dims:
        vals = vals.mean(dim=extra_dims, skipna=True)
    values = vals.values.astype(float)
    if normalize_by_duration:
        times = ds["time"].values
        durations = np.array([np.nanmax(t) - np.nanmin(t) for t in times])
        durations[durations <= 0] = np.nan
        values = values / durations
    return {int(s): float(v) for s, v in zip(ds["shot"].values, values, strict=True)}


def bin_edges_from_quantiles(kappas: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(kappas, qs)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def run(
    results_dir: str,
    dataset_path: str,
    out_dir: str,
    case_glob: str = "case.*",
    metric: str = DEFAULT_METRIC,
    n_bins: int = 4,
    device: str | None = None,
    normalize_by_duration: bool = False,
):
    """Bin per-shot errors by time-averaged kappa for every matching case.

    device restricts shots to one ds_source value when result files hold a
    mixed test set (profile transfer target test sets are single device, so
    the default None is usually fine).
    """
    results_dir = Path(results_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir() and fnmatch.fnmatch(d.name, case_glob))
    if not case_dirs:
        logger.error(f"No case dirs match {case_glob} under {results_dir}")
        return

    kappa_by_shot = per_shot_kappa(dataset_path)

    per_case: dict[str, dict[int, float]] = {}
    for case_dir in case_dirs:
        errors = load_case_errors(case_dir, metric, normalize_by_duration)
        if errors is None:
            continue
        if device is not None:
            ds = xr.load_dataset(case_dir / "result_data.nc")
            keep = {int(s) for s, src in zip(ds["shot"].values, ds["ds_source"].values, strict=True) if str(src) == device}
            errors = {s: v for s, v in errors.items() if s in keep}
        errors = {s: v for s, v in errors.items() if s in kappa_by_shot and np.isfinite(v)}
        if errors:
            per_case[case_dir.name] = errors

    if not per_case:
        logger.error("No cases with usable per-shot errors")
        return

    shared_shots = sorted(set.intersection(*(set(e) for e in per_case.values())))
    logger.info(f"{len(per_case)} cases, {len(shared_shots)} shared scored shots")
    kappas = np.array([kappa_by_shot[s] for s in shared_shots])
    edges = bin_edges_from_quantiles(kappas, n_bins)
    bin_idx = np.digitize(kappas, edges) - 1
    bin_labels = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(n_bins)]
    bin_counts = [int((bin_idx == i).sum()) for i in range(n_bins)]

    metric_label = f"{metric} per second" if normalize_by_duration else metric
    lines = [
        f"# Kappa-binned {metric_label} (mean per bin, quantile edges over {len(shared_shots)} shared shots)",
        "",
        "| case | " + " | ".join(f"kappa {lab} (n={n})" for lab, n in zip(bin_labels, bin_counts, strict=True)) + " | all |",
        "|" + "---|" * (n_bins + 2),
    ]
    binned_by_case = {}
    for name, errors in sorted(per_case.items()):
        vals = np.array([errors[s] for s in shared_shots])
        row = [float(np.mean(vals[bin_idx == i])) if (bin_idx == i).any() else np.nan for i in range(n_bins)]
        binned_by_case[name] = row
        cells = " | ".join(f"{v:.4f}" for v in row)
        lines.append(f"| {name} | {cells} | {float(np.mean(vals)):.4f} |")

    table_path = out_path / f"kappa_binned_{metric}.md"
    table_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {table_path}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    style_axis(ax, 9)
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]
    for name, row in binned_by_case.items():
        ax.plot(centers, row, marker="o", markersize=4, linewidth=1.4, label=name)
    ax.set_xlabel("Time-averaged kappa (bin center)", color=TEXT_COLOR)
    ax.set_ylabel(f"Mean {metric}", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.legend(fontsize=6, labelcolor=TEXT_COLOR, facecolor=BACKGROUND_COLOR, edgecolor=TEXT_COLOR, ncols=1, loc="best")
    fig.suptitle(f"Kappa-binned {metric}", color=TEXT_COLOR)
    fig.tight_layout()
    fig_path = out_path / f"kappa_binned_{metric}.png"
    fig.savefig(fig_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Wrote {fig_path}")


if __name__ == "__main__":
    fire.Fire({"run": run})
