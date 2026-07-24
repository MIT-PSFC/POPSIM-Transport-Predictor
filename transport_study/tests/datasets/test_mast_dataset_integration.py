"""Integration test for the MAST raw-data workflow against the open-access S3 store.

Shot 30284 is the reference shot: the full prepare -> fit -> assemble pipeline
must produce valid Te/ne profiles for it. The bulk fit uses fixed GP
hyperparameters so the test stays fast; one slice is also fit with real
hyperparameter optimization to validate the production fitting path on real
measurements.

Requires network access to s3.echo.stfc.ac.uk; skips otherwise.
"""

import numpy as np
import pytest
import xarray as xr

from transport_study.datasets.gp_fitting.fit_worker import fit_batch, gp_profile

SHOT = 30284

# Plausible free hyperparameters for GibbsKernel1dTanh (sigma_f, l1, l2, lw, x0),
# used to make the bulk fit fast by skipping per-slice hyperparameter optimization
FIXED_HYPERPARAMS = np.array([2.0, 0.8, 0.3, 0.1, 0.95])


def _store_reachable() -> bool:
    try:
        from transport_study.datasets.mast.mast_dataset import (
            check_required_signals,
            config,
        )

        return check_required_signals(SHOT, config["data_sources"])
    except Exception:
        return False


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def workflow(tmp_path_factory):
    from transport_study.datasets.mast.mast_dataset import MASTDataWorkflow

    if not _store_reachable():
        pytest.skip(f"MAST S3 store unreachable or shot {SHOT} missing")

    tmp_path = tmp_path_factory.mktemp("mast_integration")
    shotlist_file = tmp_path / "shotlist"
    shotlist_file.write_text(f"{SHOT}\n")
    return MASTDataWorkflow(
        ds_name="mast_test",
        shotlist_file=shotlist_file,
        data_assembly_dir=tmp_path,
        max_num_shots=1,
    )


@pytest.fixture(scope="module")
def fit_input(workflow):
    workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
    fit_input = workflow.prepare_shot(SHOT)
    assert fit_input is not None, f"Shot {SHOT} failed validation/retrieval"
    return fit_input


def test_shot_30284_has_fittable_thomson_slices(fit_input, workflow):
    valid_per_slice = np.sum(np.isfinite(fit_input.x) & np.isfinite(fit_input.te_y), axis=1)
    n_fittable = int(np.sum(valid_per_slice >= workflow.fit_min_points))
    assert n_fittable > 0, f"Shot {SHOT}: no Thomson slices with >= {workflow.fit_min_points} valid points"


def test_shot_30284_production_fit_on_one_slice(fit_input, workflow):
    """Real hyperparameter optimization on one real measured profile.

    Uses the slice with the most valid channels (mid-shot, hot plasma) so the
    core Te plausibility check below is meaningful; the first fittable slice
    can be very early in the shot where core Te is only tens of eV.
    """
    valid_per_slice = np.sum(np.isfinite(fit_input.x) & np.isfinite(fit_input.te_y), axis=1)
    i_time = int(np.argmax(valid_per_slice))
    assert valid_per_slice[i_time] >= workflow.fit_min_points

    scale = float(np.nanmax(fit_input.te_y[i_time, :]))
    y_star, _, _, _, _ = gp_profile(
        data_X=fit_input.x[i_time, :].astype(float),
        data_y=fit_input.te_y[i_time, :].astype(float) / scale,
        err_y=fit_input.te_err[i_time, :].astype(float) / scale,
        X_star=workflow.gp_fit_rho,
        optimize_hyperparams=True,
    )
    assert y_star is not None
    te_fit = np.asarray(y_star).ravel() * scale
    assert np.isfinite(te_fit).all()
    # Core Te of a MAST plasma should be in a physically plausible range
    assert 0.1 < te_fit[0] < 5.0


def test_shot_30284_workflow_produces_profiles(fit_input, workflow):
    """Full pipeline: staged data + batch fit -> raw netCDF with valid profiles."""
    outputs = fit_batch(
        {SHOT: fit_input},
        x_star=workflow.gp_fit_rho,
        min_points=workflow.fit_min_points,
        scale_per_slice=workflow.fit_scale_per_slice,
        num_workers=1,
        optimize_hyperparams=False,
        hyperparams=FIXED_HYPERPARAMS,
    )
    out = outputs[SHOT]
    n_fitted = int(np.sum(np.isfinite(out.te_fit).all(axis=1)))
    assert n_fitted > 0, f"Shot {SHOT}: GP fit produced no profiles"

    assert workflow.assemble_shot(SHOT, out), f"Shot {SHOT}: assembly failed"

    raw_path = workflow.raw_data_dir / f"{SHOT}.nc"
    assert raw_path.exists()
    ds = xr.load_dataset(raw_path)

    for var in ("Te_keV_rho", "ne20_rho"):
        assert var in ds, f"{var} missing from raw dataset"
        values = ds[var].values
        finite = np.isfinite(values)
        assert finite.any(), f"{var} is all NaN for shot {SHOT}"
        assert (values[finite] >= 0).all(), f"{var} has negative values"
        assert values[finite].max() > 0, f"{var} is all zeros"

    # Profiles must be on the configured rho grid
    np.testing.assert_allclose(ds["rho"].values, workflow.gp_fit_rho, rtol=1e-6)
