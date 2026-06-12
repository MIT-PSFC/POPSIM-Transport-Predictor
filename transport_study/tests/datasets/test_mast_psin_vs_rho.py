"""Compare GP fits of MAST Thomson data with chords mapped to psi_n vs rho.

rho is the normalised minor radius computed from the midplane equilibrium (see
_map_thomson_midplane), not sqrt(psi_n). Near the magnetic axis the core
occupies a narrow psi_n range, so a GP fit in psi_n can make the core look
more peaked than it should; fitting against rho spreads the core back out.
The production workflow now fits in rho; this test keeps the side-by-side
comparison against fits done in psi_n.

For MAST shot 30284 the requested reference timesteps are fit both ways using
the same fixed-hyperparameter method as test_mast_dataset_integration.py, and
a per-timestep comparison PDF is written with four axes per page: Te and ne vs
psi_n (top row) and Te and ne vs rho (bottom row), each showing the raw
measurements and the GP fit done in that coordinate.

Requires network access to s3.echo.stfc.ac.uk; skips otherwise. The PDF is
saved under scratch/mast_rho_comparison/ so it survives the test run.
"""

import numpy as np
import pytest

from transport_study import PACKAGE_ROOT
from transport_study.datasets.gp_fitting.fit_worker import ShotFitInput, fit_batch

SHOT = 30284
REQUESTED_TIMES = [0.138, 0.168, 0.183, 0.198, 0.213]  # [s]

# Plausible free hyperparameters for GibbsKernel1dTanh (sigma_f, l1, l2, lw, x0),
# same fixed-hyperparameter method as test_mast_dataset_integration.py
FIXED_HYPERPARAMS = np.array([2.0, 0.8, 0.3, 0.1, 0.95])

OUTPUT_DIR = PACKAGE_ROOT.parent / "scratch" / "mast_rho_comparison"

pytestmark = pytest.mark.slow


def _store_reachable() -> bool:
    try:
        from transport_study.datasets.mast.mast_dataset import (
            _check_required_signals,
            config,
        )

        return _check_required_signals(SHOT, config["data_sources"])
    except Exception:
        return False


@pytest.fixture(scope="module")
def workflow(tmp_path_factory):
    from transport_study.datasets.mast.mast_dataset import MASTDataWorkflow

    if not _store_reachable():
        pytest.skip(f"MAST S3 store unreachable or shot {SHOT} missing")

    tmp_path = tmp_path_factory.mktemp("mast_rho_comparison")
    shotlist_file = tmp_path / "shotlist"
    shotlist_file.write_text(f"{SHOT}\n")
    return MASTDataWorkflow(
        ds_name="mast_test",
        shotlist_file=shotlist_file,
        data_assembly_dir=tmp_path,
        max_num_shots=1,
    )


def test_psin_vs_rho_fit_comparison(workflow):
    import matplotlib
    import xarray as xr

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    fit_input = workflow._prepare_shot(SHOT)
    assert fit_input is not None, f"Shot {SHOT} failed validation/retrieval"

    ds_staging = xr.load_dataset(workflow._staging_path(SHOT))
    ts_time = ds_staging["ts_time"].values
    psi_n_all = ds_staging["ts_psi_n"].values

    # Select the requested measurement times
    indices = np.unique([int(np.argmin(np.abs(ts_time - t))) for t in REQUESTED_TIMES])
    matched = ts_time[indices]
    assert np.all(np.abs(matched - np.asarray(REQUESTED_TIMES)) < 0.02), (
        f"Requested times {REQUESTED_TIMES} not found near TS times {matched}"
    )

    # fit_input.x is rho (the production coordinate); psi_n comes from staging
    sub = {attr: np.ascontiguousarray(getattr(fit_input, attr)[indices, :]) for attr in ("x", "te_y", "te_err", "ne_y", "ne_err")}
    psin_sub = np.ascontiguousarray(psi_n_all[indices, :])
    rho_sub = sub["x"]
    assert np.isfinite(rho_sub).any(), "rho mapping produced no valid points"

    rho_input = ShotFitInput(**sub)
    psin_input = ShotFitInput(**{**sub, "x": psin_sub})

    # Same output grid in both coordinates; for the psi_n fit the grid values are psi_n locations
    x_star = workflow.gp_fit_rho

    fit_kwargs = dict(
        x_star=x_star,
        min_points=workflow.fit_min_points,
        scale_per_slice=workflow.fit_scale_per_slice,
        num_workers=1,
        optimize_hyperparams=False,
        hyperparams=FIXED_HYPERPARAMS,
    )
    out_psin = fit_batch({SHOT: psin_input}, **fit_kwargs)[SHOT]
    out_rho = fit_batch({SHOT: rho_input}, **fit_kwargs)[SHOT]

    for out, coord in [(out_psin, "psi_n"), (out_rho, "rho")]:
        assert np.isfinite(out.te_fit).any(), f"GP fit in {coord} produced no Te profiles"
        assert np.isfinite(out.ne_fit).any(), f"GP fit in {coord} produced no ne profiles"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / f"{SHOT}_psin_vs_rho.pdf"

    with PdfPages(pdf_path) as pdf:
        for i_sub, i_time in enumerate(indices):
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            for i_var, (var, label) in enumerate([("te", "Te [keV]"), ("ne", "ne [1e20 m^-3]")]):
                y = sub[f"{var}_y"][i_sub, :]
                err = sub[f"{var}_err"][i_sub, :]
                for i_row, (coord, x_ch, fit) in enumerate(
                    [
                        ("psi_n", psin_sub[i_sub, :], getattr(out_psin, f"{var}_fit")[i_sub, :]),
                        ("rho", rho_sub[i_sub, :], getattr(out_rho, f"{var}_fit")[i_sub, :]),
                    ]
                ):
                    ax = axes[i_row, i_var]
                    valid = np.isfinite(x_ch) & np.isfinite(y)
                    if valid.any():
                        ax.errorbar(x_ch[valid], y[valid], yerr=err[valid], fmt="o", ms=4, color="tab:blue", label="TS", zorder=3)
                    ok = np.isfinite(fit)
                    if ok.any():
                        ax.plot(x_star[ok], fit[ok], color="black", label=f"GP fit in {coord}")
                    ax.set_xlabel(coord)
                    ax.set_ylabel(label)
                    ax.set_ylim(bottom=0)
                    ax.set_title(f"shot {SHOT}  t={ts_time[i_time]:.3f} s  ({coord})")
                    ax.grid(alpha=0.3)
                    ax.legend(fontsize=8)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved psi_n vs rho comparison to {pdf_path}")
    assert pdf_path.exists()
