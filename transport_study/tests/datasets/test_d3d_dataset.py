"""Tests for the DIII-D dataset workflow.

The fast tests cover the psi_n -> rho_tor_norm map, the IDA regridding and hold, the strict DISPY EFIT
selection, signal standardization, and the IMAS attributes of both stores, all on synthetic data.
The slow test pulls one real shot, so it needs the IDA database and the DIII-D data servers.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from disruption_py.core.physics_method.params import PhysicsMethodParams
from disruption_py.inout.mds import mdsExceptions
from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import TimeSettingParams
from disruption_py.settings.nickname_setting import NicknameSettingParams
from loguru import logger
from transport_validation_datasets.dispy_utils import register_verbose_level
from transport_validation_datasets.store_schema import STORE_SIGNAL_ATTRS, STORE_SIGNALS

from transport_study import RADIAL_DIM
from transport_study.datasets.d3d import config
from transport_study.datasets.d3d.d3d_dataset import (
    D3D_SIGNAL_ATTRS,
    D3D_TRAJOPT_STORE_SIGNALS,
    IDA_SOURCE_ATTR,
    PREDICTION_SOURCES,
    RAW_IDA_PATH_ATTR,
    TRAJOPT_SOURCES,
    TRAJOPT_STORE_NAME,
    D3DDataWorkflow,
)
from transport_study.datasets.d3d.physics_methods import (
    D3DDatasetMethods,
    DispyEfitNicknameSetting,
    Uniform1kHzTimeSetting,
)
from transport_study.datasets.d3d.profiles import (
    IDA_PSI_COLUMNS,
    IDA_RHO_COLUMNS,
    PSI_NORM_DIM,
    PSI_NORM_GRID,
    RHO_TOR_NORM_GRID,
    IdaDatabase,
    find_ida_path,
    find_ida_shots,
    gradient_and_error,
    ida_profiles_on_grids,
)
from transport_study.signals import PREDICTION_STORE_NAME

IDA_DIR = Path("/fusion/projects/results/ida-results/HBP_database")


# Synthetic IDA slice times [ms]: a doubled interior gap (180 -> 260) inside the hold limit
IDA_TIMES_MS = np.array([100.0, 140.0, 180.0, 260.0, 300.0])
IDA_UNMAPPED_TIME_MS = 180.0


def _synthetic_ida(psi_n: np.ndarray) -> xr.Dataset:
    """IDA file layout with T_e linear in psi_n, so the psi_norm regrid is exact."""
    n_slices = IDA_TIMES_MS.size
    t_e = np.tile(1000.0 * (1.2 - psi_n), (n_slices, 1))
    n_e = np.tile(4e19 * (1.3 - psi_n), (n_slices, 1))
    data_vars = {
        "T_e": (("time", "psi_n"), t_e),
        "T_e_err": (("time", "psi_n"), np.full_like(t_e, 50.0)),
        "n_e": (("time", "psi_n"), n_e),
        "n_e_err": (("time", "psi_n"), np.full_like(n_e, 1e18)),
    }
    return xr.Dataset(data_vars, coords={"time": IDA_TIMES_MS, "psi_n": psi_n})


def _profiles_on_grids(psi_n: np.ndarray) -> tuple[xr.Dataset, np.ndarray]:
    """IDA profiles of the synthetic file on the store grids, with no valid EFIT near IDA_UNMAPPED_TIME_MS."""
    efit_time = np.arange(50, 501) * 1e-3
    qpsi = np.full((efit_time.size, 65), 2.0)
    efit_unmapped_offset = np.abs(efit_time - IDA_UNMAPPED_TIME_MS / 1e3)
    mask_efit_valid = efit_unmapped_offset > 0.005
    times = np.arange(0, 600) * 1e-3
    ida = _synthetic_ida(psi_n)
    return ida_profiles_on_grids(ida, efit_time, qpsi, mask_efit_valid, times), times


def test_ida_profiles_land_on_store_grids():
    """Constant q maps psi_n to rho_tor_norm = sqrt(psi_n), so T_e = 1000 (1.2 - rho^2) on the rho grid.

    The native grid starts off axis, so the axis value is the innermost IDA value (left clamp),
    and grid points past the IDA domain (sqrt(1.2) ~ 1.095) are NaN.
    """
    psi_n_native = 1e-3 + (1.2 - 1e-3) * np.linspace(0.0, 1.0, 150) ** 1.2
    profiles, times = _profiles_on_grids(psi_n_native)

    assert set(profiles.data_vars) == {*IDA_RHO_COLUMNS, *IDA_PSI_COLUMNS, "fresh_profile"}
    np.testing.assert_array_equal(profiles[RADIAL_DIM].values, RHO_TOR_NORM_GRID.astype(np.float32))
    te_rho = profiles["te_rho"].values[np.argmin(np.abs(times - 0.12))]
    mask_rho_inside = RHO_TOR_NORM_GRID**2 >= psi_n_native[0]
    mask_rho_covered = RHO_TOR_NORM_GRID <= np.sqrt(1.2)
    mask_rho_interpolated = mask_rho_inside & mask_rho_covered
    te_rho_expected = 1000.0 * (1.2 - RHO_TOR_NORM_GRID**2)
    np.testing.assert_allclose(te_rho[mask_rho_interpolated], te_rho_expected[mask_rho_interpolated], atol=1.0)
    assert te_rho[0] == pytest.approx(1000.0 * (1.2 - psi_n_native[0]), abs=1e-3)
    assert np.isnan(te_rho[~mask_rho_covered]).all()

    te_gradient = profiles["te_rho_grad"].values[np.argmin(np.abs(times - 0.12))]
    rho_mid = (RHO_TOR_NORM_GRID > 0.3) & (RHO_TOR_NORM_GRID < 0.9)
    np.testing.assert_allclose(te_gradient[rho_mid], -2000.0 * RHO_TOR_NORM_GRID[rho_mid], rtol=1e-2)


def test_differing_native_psi_grids_land_on_one_psi_norm_grid():
    """Each IDA file has its own psi_n grid, but every shot's trajopt profiles share PSI_NORM_GRID."""
    psi_n_native_a = 1.2 * np.linspace(0.0, 1.0, 150) ** 1.2
    psi_n_native_b = 1.2 * np.linspace(0.0, 1.0, 140) ** 1.05
    te_psi_expected = (1000.0 * (1.2 - PSI_NORM_GRID)).astype(np.float32)

    for psi_n_native in [psi_n_native_a, psi_n_native_b]:
        profiles, times = _profiles_on_grids(psi_n_native)
        np.testing.assert_array_equal(profiles[PSI_NORM_DIM].values, PSI_NORM_GRID.astype(np.float32))
        te_psi = profiles["te_psi"].values[np.argmin(np.abs(times - 0.12))]
        np.testing.assert_allclose(te_psi, te_psi_expected, rtol=1e-5)


def test_ida_hold_and_unmapped_slice():
    """Slices hold across interior gaps, and the hold ends max_hold_ida_steps median steps after the last slice.

    A slice with no valid EFIT within match_max_ms keeps its psi_norm profiles.
    It is dropped from the rho_tor_norm hold, so the slice before holds over it and it is not fresh.
    """
    psi_n_native = 1.2 * np.linspace(0.0, 1.0, 150) ** 1.2
    profiles, times = _profiles_on_grids(psi_n_native)
    te_rho_axis = profiles["te_rho"].values[:, 0]
    te_psi_axis = profiles["te_psi"].values[:, 0]
    times_ms = np.round(times * 1e3)

    assert np.isnan(te_rho_axis[times_ms < 100]).all()
    mask_slices_held = (times_ms >= 100) & (times_ms <= 300)
    assert np.isfinite(te_rho_axis[mask_slices_held]).all()
    assert np.isfinite(te_psi_axis[mask_slices_held]).all()
    mask_fresh = profiles["fresh_profile"].values == 1
    mapped_times_ms = IDA_TIMES_MS[IDA_TIMES_MS != IDA_UNMAPPED_TIME_MS]
    np.testing.assert_array_equal(times_ms[mask_fresh], mapped_times_ms)

    ida_step_median_ms = np.median(np.diff(IDA_TIMES_MS))
    hold_end_ms = IDA_TIMES_MS[-1] + config["profile_grid"]["max_hold_ida_steps"] * ida_step_median_ms
    # One slice of margin either side of the limit, where float round-off decides
    assert np.isfinite(te_psi_axis[(times_ms > 300) & (times_ms < hold_end_ms)]).all()
    assert np.isnan(te_psi_axis[times_ms > hold_end_ms]).all()


def test_gradient_and_error():
    """Linear profiles have a constant gradient, the error is the central-difference propagation."""
    x = np.linspace(0.0, 1.0, 11)
    values = np.tile(2.0 * x + 1.0, (3, 1))
    errors = np.full_like(values, 0.5)

    gradient, gradient_error = gradient_and_error(values, errors, x)

    np.testing.assert_allclose(gradient, 2.0)
    x_step = x[1] - x[0]
    np.testing.assert_allclose(gradient_error[:, 1:-1], np.sqrt(2 * 0.5**2) / (2 * x_step))
    np.testing.assert_allclose(gradient_error[:, 0], gradient_error[:, 1])
    np.testing.assert_allclose(gradient_error[:, -1], gradient_error[:, -2])


def test_ida_databases_priority_wildcards_shotlists_and_union(tmp_path):
    """The first database with a file wins, wildcards match VVUQ-style names, a database with a shotlist
    serves only its shots, and the default shotlist is the union of what every database serves."""
    primary, fallback, wild, general = tmp_path / "primary", tmp_path / "fallback", tmp_path / "wild", tmp_path / "general"
    for directory in [primary, fallback, wild, general]:
        directory.mkdir()
    databases = [
        IdaDatabase(str(primary / "IDA_{shot}_.cdf")),
        IdaDatabase(str(fallback / "ida{shot}.nc")),
        IdaDatabase(str(wild / "IDA_{shot}_*_.cdf")),
        IdaDatabase(str(general / "IDA_{shot}_.cdf"), shots=frozenset({2, 5})),
    ]
    files = [
        primary / "IDA_2_.cdf",
        fallback / "ida1.nc",
        fallback / "ida2.nc",
        wild / "IDA_4_0.2_4.0_.cdf",
        general / "IDA_2_.cdf",
        general / "IDA_5_.cdf",
        general / "IDA_6_.cdf",
    ]
    for path in files:
        path.touch()

    assert find_ida_path(2, databases) == primary / "IDA_2_.cdf"
    assert find_ida_path(1, databases) == fallback / "ida1.nc"
    assert find_ida_path(3, databases) is None
    assert find_ida_path(4, databases) == wild / "IDA_4_0.2_4.0_.cdf"
    assert find_ida_path(5, databases) == general / "IDA_5_.cdf"
    assert find_ida_path(6, databases) is None
    assert find_ida_shots(databases) == [1, 2, 4, 5]


class _StubDatabase:
    """Answers the code_rundb query with fixed rows and records it."""

    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def query(self, query, use_pandas=True):
        self.queries.append(query)
        return self.rows


def _nickname_params(database) -> NicknameSettingParams:
    return NicknameSettingParams(shot_id=199051, mds_conn=None, database=database, disruption_time=None, tokamak=Tokamak.D3D)


def test_dispy_nickname_takes_latest_run_and_raises_without_one():
    """The latest DISPY run is the EFIT tree, and a shot without one fails instead of falling back to efit01."""
    database = _StubDatabase([("EFIT04",), ("EFIT07",)])

    tree = DispyEfitNicknameSetting().get_tree_name(_nickname_params(database))

    assert tree == "EFIT07"
    assert "runtag = 'DISPY'" in database.queries[0]
    with pytest.raises(ValueError, match="no EFIT run under runtag DISPY"):
        DispyEfitNicknameSetting().get_tree_name(_nickname_params(_StubDatabase([])))


class _StubEfitConnection:
    """Serves one atime array [ms] for every get_data call."""

    def __init__(self, atime_ms):
        self.atime_ms = atime_ms

    def get_data(self, path, tree_name=None):
        return self.atime_ms


def _time_params(atime_ms) -> TimeSettingParams:
    connection = _StubEfitConnection(atime_ms)
    return TimeSettingParams(shot_id=199051, mds_conn=connection, database=None, disruption_time=None, tokamak=Tokamak.D3D)


def test_time_setting_is_1khz_and_rejects_slow_efit():
    """A 1 kHz EFIT gives the uniform timebase out to 8 s, a 50 Hz one (not DISPY) fails the shot."""
    atime_1khz_ms = np.arange(100.0, 5001.0)
    times = Uniform1kHzTimeSetting().get_times(_time_params(atime_1khz_ms))

    assert times[0] == 0.0
    assert times[-1] == pytest.approx(8.0)
    np.testing.assert_allclose(np.diff(times), 1e-3, atol=1e-6)
    atime_50hz_ms = np.arange(100.0, 5001.0, 20.0)
    with pytest.raises(ValueError, match="not a 1 kHz reconstruction"):
        Uniform1kHzTimeSetting().get_times(_time_params(atime_50hz_ms))


class _StubDensityConnection:
    """Serves \\density [cm^-3] (or raises TreeNODATA when it is None) and PTDATA dssdenest [1e19 m^-3], both on 0-1000 ms."""

    def __init__(self, density_cm3):
        self.density_cm3 = density_cm3
        self.time_ms = np.linspace(0.0, 1000.0, 11)

    def get_data_with_dims(self, path, tree_name=None):
        if path == r"\density":
            if self.density_cm3 is None:
                raise mdsExceptions.TreeNODATA()
            return np.full(self.time_ms.size, self.density_cm3), self.time_ms
        assert "dssdenest" in path
        return np.full(self.time_ms.size, 5.0), self.time_ms


def _line_average_density(density_cm3) -> np.ndarray:
    # disruption-py's physics_method decorator logs at the VERBOSE level
    register_verbose_level()
    times = np.linspace(0.1, 0.9, 5)
    connection = _StubDensityConnection(density_cm3)
    params = PhysicsMethodParams(shot_id=199264, tokamak=Tokamak.D3D, disruption_time=None, mds_conn=connection, times=times)
    return D3DDatasetMethods.get_line_average_density(params=params)["n_e_line_average"]


def test_line_average_density_falls_back_to_pcs_estimate():
    """\\density [cm^-3] where the DISPY tree has it, else dssdenest [1e19 m^-3] (199264 has no \\density)."""
    np.testing.assert_allclose(_line_average_density(4e13), 4e19)
    np.testing.assert_allclose(_line_average_density(None), 5e19)
    np.testing.assert_allclose(_line_average_density(np.nan), 5e19)


N_TIME = 700  # 0.7 s, over min_pulse_length_s once the end margin is cut
RAW_SCALAR_VALUES = {
    "ip": -1.2e6,
    "bcoil": -1.2e5,
    "wmhd": 8e5,
    "beta_n": 2.0,
    # The flat ne_rho profile over it, so the density ratio cull passes
    "n_e_line_average": 6e19,
    "aminor": 0.6,
    "rsurf": 1.7,
    "kappa": 1.8,
    "tritop": 0.4,
    "tribot": 0.5,
    "p_ohm": 1e6,
    "p_rad": 2e6,
    "p_nbi": 5e6,
    "p_ech": 1e6,
    "ip_prog": -1.0e6,
    "bmtpwrtar": 2.5,
    "idtrp": 1.7,
    "idtrxbot": 1.3,
    "idtzxbot": -1.1,
    "idtrxtop": 1.4,
    "idtzxtop": 1.05,
    "gapin": 0.09,
    "rxpt1": 1.31,
    "zxpt1": -1.12,
    "rxpt2": 1.41,
    "zxpt2": 1.06,
    "dssneped": 3.5,
    "bttbt": 2.0,
    "dstdenp": 5.2,
    "ieeseg07": 0.004,
}
RAW_PROFILE_VALUES = {
    "te_rho": 2000.0,
    "te_rho_error": 100.0,
    "te_rho_grad": -500.0,
    "te_rho_grad_error": 50.0,
    "ne_rho": 6e19,
    "ne_rho_error": 1e18,
    "ne_rho_grad": -1e19,
    "ne_rho_grad_error": 5e17,
    "te_psi": 3000.0,
    "ne_psi": 5e19,
}


def _raw_dataset(shot: int = 199051) -> xr.Dataset:
    """A shot of raw disruption-py columns, as _get_shot_dataset returns it, that passes every filter and cull."""
    data_vars = {name: (("shot", "time"), np.full((1, N_TIME), value)) for name, value in RAW_SCALAR_VALUES.items()}
    for name, value in RAW_PROFILE_VALUES.items():
        radial_dim, radial_grid = (PSI_NORM_DIM, PSI_NORM_GRID) if name in IDA_PSI_COLUMNS else (RADIAL_DIM, RHO_TOR_NORM_GRID)
        data_vars[name] = (("shot", "time", radial_dim), np.full((1, N_TIME, radial_grid.size), value))
    # One IDA slice every 20 ms
    fresh_profile = (np.arange(N_TIME) % 20 == 0).astype(np.float32)
    data_vars["fresh_profile"] = (("shot", "time"), fresh_profile[np.newaxis, :])
    # float32 grids, as get_ida_profiles writes them
    coords = {
        "shot": [shot],
        "time": np.arange(N_TIME) * 1e-3,
        RADIAL_DIM: RHO_TOR_NORM_GRID.astype(np.float32),
        PSI_NORM_DIM: PSI_NORM_GRID.astype(np.float32),
    }
    return xr.Dataset(data_vars, coords=coords)


@pytest.fixture
def workflow(tmp_path) -> D3DDataWorkflow:
    shotlist = tmp_path / "shotlist"
    shotlist.write_text("199051\n199052\n")
    d3d_workflow = D3DDataWorkflow(ds_name="d3d_test", shotlist_file=shotlist, data_assembly_dir=tmp_path)
    d3d_workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
    return d3d_workflow


def test_standardize_builds_both_stores_in_si(workflow):
    """Every store signal is built from its raw column, signed currents and fields become magnitudes,
    dssneped [1e19 m^-3] becomes m^-3, and a missing column or an all-NaN profile skips the shot."""
    ds = workflow.standardize_signal_names(_raw_dataset())

    assert set(STORE_SIGNALS) <= set(ds.data_vars)
    assert set(D3D_TRAJOPT_STORE_SIGNALS) <= set(ds.data_vars)
    np.testing.assert_allclose(ds["ip"], 1.2e6)
    # mu0 / (2 pi) = 2e-7 of the 144 turn TF coil, at r0 = 1.6955 m
    np.testing.assert_allclose(ds["b0"], 1.2e5 * 2e-7 * 144 / 1.6955)
    np.testing.assert_allclose(ds["ip_reference"], 1.0e6)
    np.testing.assert_allclose(ds["n_e_pedestal"], 3.5e19)
    np.testing.assert_allclose(ds["energy_mhd"], 8e5)
    np.testing.assert_allclose(ds["t_e_gradient"], -500.0)
    np.testing.assert_allclose(ds["power_ic"], 0.0)
    assert {"time_idx", "shot"} <= set(ds.dims)
    assert "time" in ds.coords

    assert workflow.standardize_signal_names(_raw_dataset().drop_vars("wmhd")) is None
    raw_no_profile = _raw_dataset()
    raw_no_profile["te_rho"] = xr.full_like(raw_no_profile["te_rho"], np.nan)
    assert workflow.standardize_signal_names(raw_no_profile) is None


def test_missing_efit_boundary_leaves_nothing_to_keep(workflow):
    """A shot with profiles but no EFIT geometry fails the finite filter at every time."""
    ds = workflow.standardize_signal_names(_raw_dataset())
    assert workflow.filter_ds(ds.copy()) is not None

    ds["geometric_axis_r"] = xr.full_like(ds["geometric_axis_r"], np.nan)
    assert workflow.filter_ds(ds) is None


SYNTHETIC_IDA_PATHS = {
    199051: "/ida/HBP_database/IDA_199051_.cdf",
    199052: "/ida/TMDB_V1c/Output/IDA_199052_.cdf",
}


@pytest.fixture
def built_stores(workflow):
    """Both stores built from two synthetic raw files, with profiles from different IDA databases."""
    for shot, ida_path in SYNTHETIC_IDA_PATHS.items():
        raw = _raw_dataset(shot)
        raw_standardized = workflow.standardize_signal_names(raw)
        raw_standardized.attrs[RAW_IDA_PATH_ATTR] = ida_path
        raw_standardized.to_netcdf(workflow.raw_data_dir / f"{shot}.nc")
    workflow.run_processed_data_workflow()
    ds_prediction = xr.open_zarr(workflow.store_path(PREDICTION_STORE_NAME))
    ds_trajopt = xr.open_zarr(workflow.store_path(TRAJOPT_STORE_NAME))
    return ds_prediction, ds_trajopt


def test_stores_carry_imas_attributes_and_grids(built_stores):
    """Every stored variable and profile coordinate has a description and units,
    the ref (IMAS path) of D3D_SIGNAL_ATTRS wherever IMAS has a leaf, and the store grids."""
    ds_prediction, ds_trajopt = built_stores

    assert set(ds_prediction.data_vars) == {*STORE_SIGNALS, "time"}
    assert set(ds_trajopt.data_vars) == {*D3D_TRAJOPT_STORE_SIGNALS, "time"}
    for ds_store in [ds_prediction, ds_trajopt]:
        assert ds_store.attrs["efit_runtag"] == "DISPY"
        assert ds_store.attrs["sol_extension"] == config["profile_grid"]["sol_extension"]
        names = [name for name in ds_store.variables if name not in ("shot", "time_idx")]
        for name in names:
            attrs = ds_store[name].attrs
            assert {"description", "units"} <= set(attrs), name
            # The shared schema's ref wins over the device's own
            expected_attrs = STORE_SIGNAL_ATTRS.get(name, D3D_SIGNAL_ATTRS[name])
            if "ref" in expected_attrs:
                assert attrs["ref"] == expected_attrs["ref"], name
    np.testing.assert_array_equal(ds_prediction[RADIAL_DIM].values, RHO_TOR_NORM_GRID.astype(np.float32))
    np.testing.assert_array_equal(ds_trajopt[PSI_NORM_DIM].values, PSI_NORM_GRID.astype(np.float32))


def test_stores_record_each_shots_ida_folder(built_stores):
    """Both stores map every stored shot to the folder of its IDA file, and no raw IDA path leaks into them."""
    expected = {str(shot): str(Path(ida_path).parent) for shot, ida_path in SYNTHETIC_IDA_PATHS.items()}
    for ds_store in built_stores:
        assert json.loads(ds_store.attrs[IDA_SOURCE_ATTR]) == expected
        assert RAW_IDA_PATH_ATTR not in ds_store.attrs


def test_every_source_column_has_attrs():
    """A store signal added to the sources without attributes would be written undocumented."""
    store_names = {*PREDICTION_SOURCES, *TRAJOPT_SOURCES, *STORE_SIGNALS}
    assert store_names <= set(D3D_SIGNAL_ATTRS)


@pytest.mark.slow
@pytest.mark.skipif(not IDA_DIR.exists(), reason="needs the /fusion IDA database and DIII-D data server access")
def test_live_single_shot(tmp_path):
    """One real shot end to end: thin-client MDSplus, the DISPY EFIT, physical ranges, and no netCDF in tmp."""
    dispy_tmp_dir = Path(os.getenv("LOCALSCRATCH", "/tmp")) / os.environ["USER"] / "disruption-py"
    netcdf_before = set(dispy_tmp_dir.rglob("*.nc"))
    d3d_workflow = D3DDataWorkflow(ds_name="d3d_live", shotlist_file=None, data_assembly_dir=tmp_path, max_num_shots=1)
    d3d_workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
    # A file sink, since disruption-py logs from a forked worker process
    log_path = tmp_path / "live.log"
    sink_id = logger.add(log_path)

    try:
        d3d_workflow.make_raw_data_files()
    finally:
        logger.remove(sink_id)

    shot = d3d_workflow.shotlist[0]
    raw_path = d3d_workflow.raw_data_dir / f"{shot}.nc"
    assert raw_path.exists(), f"raw file for shot {shot} was not written"
    assert "mdsthin" in sys.modules
    assert "(runtag DISPY)" in log_path.read_text()
    netcdf_after = set(dispy_tmp_dir.rglob("*.nc"))
    assert netcdf_after == netcdf_before, "disruption-py wrote netCDF files to its temporary folder"

    ds = xr.open_dataset(raw_path)
    ip_max = float(ds["ip"].max())
    assert 3e5 < ip_max < 2.5e6
    assert 1.0 < float(ds["b0"].median()) < 2.5
    assert 0.5 < float(ds["minor_radius"].median()) < 0.7
    assert 1.6 < float(ds["geometric_axis_r"].median()) < 1.8
    assert 1e19 < float(ds["n_e_line_average"].median()) < 1.5e20
    t_e_axis = ds["t_e"].sel({RADIAL_DIM: 0}).values
    assert 500 < np.nanmax(t_e_axis) < 1e4
    # The EFIT P_oh sits well under the 2 MW transient threshold, and prad_tot covers the whole plasma
    mask_flattop = ds["ip"].values.squeeze() > 0.5 * ip_max
    p_ohm_flattop = ds["power_ohm"].values.squeeze()[mask_flattop]
    p_rad_flattop = ds["power_radiated"].values.squeeze()[mask_flattop]
    assert 0 < np.nanmedian(p_ohm_flattop) < 2e6
    assert np.nanpercentile(p_ohm_flattop, 95) < 2e6
    assert np.isfinite(p_rad_flattop).all()
    assert 1e5 < np.median(p_rad_flattop) < 1e7
    # Every held slice has a value at the axis (left clamp)
    t_e = ds["t_e"].values.squeeze()
    mask_held = np.isfinite(t_e).any(axis=-1)
    assert np.isfinite(t_e[mask_held][:, 0]).all()
