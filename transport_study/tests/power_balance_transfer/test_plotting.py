from itertools import product

import numpy as np
import pytest
import xarray as xr

from transport_study.power_balance_transfer.plotting import (
    data_normalization_comparison,
    domain_adaptation_comparison,
    model_comparison,
    training_dataset_comparison,
)


@pytest.fixture()
def results_ds() -> xr.Dataset:
    """Synthetic collected_results.nc matching PowerBalanceStudy.collect_results:
    one row per case along case_idx with the case-grid coords and err_E_D_S vars."""
    model_types = ["scaling_law", "sciml", "unstructured_nn", "transformer", "p_oh"]
    training_datasets = ["cmod", "exnihilo"]
    normalizations = ["raw", "coral"]
    adaptations = ["none", "weighted", "addition", "transfer"]
    shots_options = [0, 1, 10, -1]

    rows = []
    for mt, td, dn, da, shots in product(model_types, training_datasets, normalizations, adaptations, shots_options):
        # Mirror the case grid: no-adaptation baselines only exist at shots=0,
        # transfer and exnihilo need at least one target shot
        if da == "none" and td != "exnihilo" and shots != 0:
            continue
        if (da == "transfer" or td == "exnihilo") and shots == 0:
            continue
        rows.append((mt, td, dn, da, True, shots))

    rng = np.random.default_rng(0)
    n = len(rows)
    ds = xr.Dataset(coords={"case_idx": np.arange(n)})
    for coord_name, idx in zip(
        ("model_type", "training_data", "data_normalization", "domain_adaptation", "freeze_submodules", "num_target_shots"),
        range(6),
        strict=True,
    ):
        ds = ds.assign_coords({coord_name: ("case_idx", [row[idx] for row in rows])})

    for err in ("err_abs", "err_rel"):
        for domain in ("shot", "ts"):
            for stat in ("mean", "std", "med", "p25", "p75", "min", "max"):
                ds[f"{err}_{domain}_{stat}"] = ("case_idx", rng.uniform(0.01, 1.0, n))

    # One diverged case to exercise the divergence masking
    ds["err_abs_shot_mean"][0] = 1e6
    return ds


@pytest.mark.parametrize(
    ("comparison_fn", "family"),
    [
        (model_comparison, "model_comparison"),
        (training_dataset_comparison, "training_dataset_comparison"),
        (data_normalization_comparison, "data_normalization_comparison"),
        (domain_adaptation_comparison, "domain_adaptation_comparison"),
    ],
)
def test_comparison_figures(results_ds, tmp_path, comparison_fn, family):
    comparison_fn(results_ds, tmp_path)
    family_dir = tmp_path / "comparison" / family
    figures = list(family_dir.glob("*.png"))
    assert figures, f"No {family} figures were generated"

    # Submodule prereq cases predict P_oh/P_rad, not Wtot, and must not
    # appear as figures of their own
    assert not [fig for fig in figures if fig.name.startswith("p_oh")]


def test_empty_results_skip(tmp_path):
    empty = xr.Dataset()
    model_comparison(empty, tmp_path)
    training_dataset_comparison(empty, tmp_path)
    data_normalization_comparison(empty, tmp_path)
    domain_adaptation_comparison(empty, tmp_path)
    assert not (tmp_path / "comparison").exists()
