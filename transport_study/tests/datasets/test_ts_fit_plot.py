"""Tests for the shared TS-fit diagnostic PDF (datasets/plotting.ts_fit_pdf).

The C-Mod and MAST workflows both write this PDF at the end of every assembled
shot, wrapped in a try/except so a plotting failure never loses the raw file -
which also means a broken plotter is silent in production. These cover it
directly, without the device data sources the workflow tests need.
"""

import numpy as np
import pytest

from transport_study.datasets.gp_fitting.fit_worker import ShotFitOutput, fit_batch
from transport_study.datasets.plotting import fit_mean_ylim, ts_fit_pdf

N_T = 4
N_CH = 12
RHO_FIT = np.linspace(0.0, 1.1, 30)
FIXED_HYPERPARAMS = np.array([2.0, 0.8, 0.3, 0.1, 0.95])


@pytest.fixture
def slice_data():
    """Channel data and its GP fit for a few pedestal-like slices."""
    from transport_study.datasets.gp_fitting.fit_worker import ShotFitInput

    rho_ch = np.tile(np.linspace(0.02, 1.05, N_CH), (N_T, 1))
    profile = 2.0 * (1 - np.tanh((rho_ch - 0.9) / 0.08)) / 2 + 0.05
    err = np.maximum(0.1 * profile, 0.05)
    fit_input = ShotFitInput(x=rho_ch, te_y=profile, te_err=err, ne_y=profile / 2, ne_err=err)
    fit_output = fit_batch(
        {1: fit_input},
        RHO_FIT,
        min_points=1,
        num_workers=1,
        optimize_hyperparams=False,
        hyperparams=FIXED_HYPERPARAMS,
    )[1]
    channel_data = {"te": (fit_input.te_y, fit_input.te_err), "ne": (fit_input.ne_y, fit_input.ne_err)}
    return rho_ch, channel_data, fit_output


def _pdf_page_count(path) -> int:
    """Pages in a matplotlib-written PDF (no pdf reader dependency)."""
    raw = path.read_bytes()
    return raw.count(b"/Type /Page") - raw.count(b"/Type /Pages")


def test_ts_fit_pdf_writes_one_page_per_fitted_slice(tmp_path, slice_data):
    rho_ch, channel_data, fit_output = slice_data
    pdf_path = tmp_path / "plots" / "1_ts_gp_fit.pdf"

    n_pages = ts_fit_pdf(pdf_path, 1, np.arange(N_T) * 0.1, rho_ch, channel_data, fit_output, RHO_FIT)

    assert pdf_path.exists()  # parent directory is created
    assert n_pages == N_T
    assert _pdf_page_count(pdf_path) == N_T


def test_ts_fit_pdf_skips_unfitted_slices(tmp_path, slice_data):
    """Slices outside the plasma come back all-NaN from the fit and would plot
    as blank pages, so they are not paged over at all."""
    rho_ch, channel_data, fit_output = slice_data
    for field in ("te_fit", "ne_fit"):
        getattr(fit_output, field)[1:, :] = np.nan
    pdf_path = tmp_path / "1_ts_gp_fit.pdf"

    assert ts_fit_pdf(pdf_path, 1, np.arange(N_T) * 0.1, rho_ch, channel_data, fit_output, RHO_FIT) == 1
    assert _pdf_page_count(pdf_path) == 1


def test_ts_fit_pdf_channel_groups(tmp_path, slice_data):
    """The C-Mod core/edge channel split reaches the plot without error, and a
    group covering no channel is simply not drawn."""
    rho_ch, channel_data, fit_output = slice_data
    is_core = np.arange(N_CH) < N_CH // 2
    pdf_path = tmp_path / "1_ts_gp_fit.pdf"

    assert (
        ts_fit_pdf(
            pdf_path,
            1,
            np.arange(N_T) * 0.1,
            rho_ch,
            channel_data,
            fit_output,
            RHO_FIT,
            channel_groups=[(is_core, "tab:blue", "TS core"), (np.zeros(N_CH, dtype=bool), "tab:orange", "TS edge")],
        )
        == N_T
    )
    assert pdf_path.exists()


def test_ts_fit_pdf_all_nan_fit_writes_nothing(tmp_path, slice_data):
    """A shot whose every slice failed to fit writes no file at all, instead of
    raising (the caller only wraps this in a try/except)."""
    rho_ch, channel_data, _ = slice_data
    nan_arrays = {
        field: np.full((N_T, RHO_FIT.size), np.nan)
        for field in ("te_fit", "te_std", "ne_fit", "ne_std", "te_grad", "te_grad_std", "ne_grad", "ne_grad_std")
    }
    empty = ShotFitOutput(**nan_arrays, te_hyps=np.full((N_T, 5), np.nan), ne_hyps=np.full((N_T, 5), np.nan))
    pdf_path = tmp_path / "1_ts_gp_fit.pdf"

    assert ts_fit_pdf(pdf_path, 1, np.arange(N_T) * 0.1, rho_ch, channel_data, empty, RHO_FIT) == 0
    assert not pdf_path.exists()


def test_fit_mean_ylim_falls_back_on_degenerate_fits():
    assert fit_mean_ylim(np.array([[1.0, 2.0]]), fallback=5.0) == pytest.approx(2.1)
    assert fit_mean_ylim(np.full((2, 3), np.nan), fallback=5.0) == 5.0
    assert fit_mean_ylim(np.zeros((2, 3)), fallback=1.8) == 1.8
