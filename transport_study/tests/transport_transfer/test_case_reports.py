"""Transport case reports on the result files of every case kind the study writes."""

from types import SimpleNamespace

import numpy as np
import xarray as xr

from transport_study.power_balance_transfer import study_metrics
from transport_study.signals import HEATING_POWERS_MW
from transport_study.transport_transfer.case_reports import generate_case_report

N_TS = 40
SHOTS = [101, 102]


def _device_ds() -> xr.Dataset:
    ip = np.concatenate([np.linspace(0.0, 1.0, 10), np.full(20, 1.0), np.linspace(1.0, 0.0, 10)])
    time = np.arange(N_TS) * 1e-3
    data_vars = {sig: (("shot", "time_idx"), np.zeros((len(SHOTS), N_TS))) for sig in HEATING_POWERS_MW}
    data_vars["ip_MA"] = (("shot", "time_idx"), np.stack([ip, ip]))
    return xr.Dataset(data_vars, coords={"shot": SHOTS, "time": (("shot", "time_idx"), np.stack([time, time]))})


def _scalar_result_ds() -> xr.Dataset:
    """The result file of a power_balance / p_oh / p_rad prereq case, no profile variables."""
    time = np.arange(N_TS) * 1e-3
    targ = np.ones((N_TS, len(SHOTS)))
    err = np.stack([np.full(N_TS, 0.1), np.full(N_TS, 0.3)], axis=1)
    dims = ("time_idx", "shot")
    return xr.Dataset(
        data_vars={
            "power_ohm_MW_targ": (dims, targ),
            "power_ohm_MW_pred": (dims, targ + err),
            "error_abs_ts": (dims, err),
            "error_rel_ts": (dims, err / 1.1),
        },
        coords={"shot": SHOTS, "time": (dims, np.stack([time, time], axis=1)), "ds_source": ("shot", ["testdev", "testdev"])},
    )


def test_scalar_prereq_case_gets_a_report(tmp_path, monkeypatch):
    """The scalar prereq cases render the power balance page instead of failing on missing profile targets."""
    monkeypatch.setattr(study_metrics, "load_stage_dataset", lambda device: _device_ds())
    result_path = tmp_path / "result_data.nc"
    _scalar_result_ds().to_netcdf(result_path)
    study = SimpleNamespace(result_path=lambda case: result_path)

    generate_case_report(study, "case.p_oh.td_cmod.freeze_True", tmp_path / "figures")

    pdf_path = tmp_path / "figures" / "case_reports" / "case.p_oh.td_cmod.freeze_True" / "best_worst_shots.pdf"
    assert pdf_path.stat().st_size > 0
