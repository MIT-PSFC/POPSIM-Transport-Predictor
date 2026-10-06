from itertools import product

import numpy as np
import pytest
import xarray as xr

from transport_study.orchestration.comparison_figures import comparison_figures
from transport_study.power_balance_transfer.plotting import COMPARISON_FAMILIES, LAYOUT


@pytest.fixture()
def results_ds() -> xr.Dataset:
    """Synthetic collected_results.nc matching PowerBalanceStudy.collect_results:
    one row per case along case_idx with the case-grid coords and err_E_D_S vars."""
    model_types = ["sciml-taue-scalinglaw", "sciml-taue-nn", "mlp", "transformer", "p_oh"]
    training_datasets = ["cmod", "exnihilo"]
    normalizations = ["raw", "coral"]
    adaptations = ["none", "weighted", "addition", "transfer"]
    shots_options = [0, 1, 10]

    rows = []
    for mt, td, dn, da, shots in product(model_types, training_datasets, normalizations, adaptations, shots_options):
        # Mirror the case grid: no-adaptation baselines only exist at shots=0,
        # transfer and exnihilo need at least one target shot
        if da == "none" and td != "exnihilo" and shots != 0:
            continue
        if (da == "transfer" or td == "exnihilo") and shots == 0:
            continue
        rows.append((mt, td, dn, da, False, shots, "ascending", False))
        # Without target shots every case takes the base order
        if shots > 0:
            rows.append((mt, td, dn, da, False, shots, "spanning", False))
        # The child-study axes: frozen sciml submodules and multiobjective transformers
        if mt.startswith("sciml"):
            rows.append((mt, td, dn, da, True, shots, "ascending", False))
        if mt == "transformer":
            rows.append((mt, td, dn, da, False, shots, "ascending", True))

    rng = np.random.default_rng(0)
    n = len(rows)
    ds = xr.Dataset(coords={"case_idx": np.arange(n)})
    for coord_name, idx in zip(
        (
            "model_type",
            "training_data",
            "data_normalization",
            "domain_adaptation",
            "freeze_submodules",
            "num_target_shots",
            "target_shot_order",
            "multiobjective",
        ),
        range(8),
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


@pytest.mark.slow
@pytest.mark.parametrize("family", COMPARISON_FAMILIES, ids=lambda family: family.name)
def test_comparison_figures(results_ds, tmp_path, family):
    comparison_figures(results_ds, LAYOUT, family, tmp_path)
    family_dir = tmp_path / "comparison" / family.name
    figures = list(family_dir.glob("*.png"))
    assert figures, f"No {family.name} figures were generated"

    # Submodule prereq cases predict P_oh/P_rad, not Wtot, and must not
    # appear as figures of their own
    assert not [fig for fig in figures if fig.name.startswith("p_oh")]


def test_empty_results_skip(tmp_path):
    for family in COMPARISON_FAMILIES:
        comparison_figures(xr.Dataset(), LAYOUT, family, tmp_path)
    assert not (tmp_path / "comparison").exists()
