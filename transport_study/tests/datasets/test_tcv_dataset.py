"""Tests for the TCV dataset workflow.

The fast tests cover the LIUQE rho_pol -> rho_tor_norm map, the DEFUSE regridding and hold, the fringe-jump correction,
reading the MATLAB v7.3 layout of a DEFUSE export, signal standardization, and the IMAS attributes of the store,
all on synthetic data.
The slow test builds one real shot, so it needs the DEFUSE exports and the MEQ databases.
"""

import h5py
import numpy as np
import pytest
import xarray as xr
from transport_validation_datasets.machine.generic import phi_n_map
from transport_validation_datasets.store_schema import STORE_SIGNAL_ATTRS, STORE_SIGNALS

from transport_study import RADIAL_DIM, TIME_COORD
from transport_study.datasets.tcv import config
from transport_study.datasets.tcv.profiles import (
    RHO_TOR_NORM_GRID,
    DefuseProfile,
    LiuqeEquilibria,
    defuse_profile_on_grid,
    liuqe_q_profiles,
    liuqe_usable,
)
from transport_study.datasets.tcv.sources import meqdb_path, read_defuse, read_liuqe
from transport_study.datasets.tcv.tcv_dataset import (
    DEFUSE_SIGNALS,
    PREDICTION_SOURCES,
    TCV_SIGNAL_ATTRS,
    TCVDataWorkflow,
    _remove_fringe_jumps,
)
from transport_study.signals import PREDICTION_STORE_NAME

RHO_POL_SURFACES = np.linspace(0.0, 1.0, 41)
LIVE_SHOT = 60001


# DEFUSE slice times [s], with no usable reconstruction near DEFUSE_UNMAPPED_TIME
# and a fit that fails toward the edge at DEFUSE_INCOMPLETE_TIME
DEFUSE_TIMES = np.array([0.100, 0.117, 0.134, 0.151, 0.168, 0.185])
DEFUSE_UNMAPPED_TIME = 0.151
DEFUSE_INCOMPLETE_TIME = 0.117


def _profile_on_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """T_e = 1000 (1 - rho_pol^2) on a DEFUSE-like grid, refined toward the edge, through constant-q equilibria."""
    rho_pol = 1.0 - (1.0 - np.linspace(0.0, 1.0, 200)) ** 1.5
    values = np.tile(1000.0 * (1.0 - rho_pol**2), (DEFUSE_TIMES.size, 1))
    idx_incomplete = int(np.flatnonzero(DEFUSE_TIMES == DEFUSE_INCOMPLETE_TIME)[0])
    values[idx_incomplete, -5:] = np.nan
    profile = DefuseProfile(time=DEFUSE_TIMES, rho_pol=rho_pol, values=values)

    eq_time = np.arange(50, 301) * 1e-3
    inverse_q = np.full((eq_time.size, RHO_POL_SURFACES.size), 0.5)
    mask_eq_broken = np.abs(eq_time - DEFUSE_UNMAPPED_TIME) < 0.005
    inverse_q[mask_eq_broken] = np.nan
    equilibria = LiuqeEquilibria(time=eq_time, rho_pol=RHO_POL_SURFACES, inverse_q=inverse_q)

    times = np.arange(0, 400) * 1e-3
    mask_eq_usable = liuqe_usable(equilibria)
    values_on_grid, gradient_on_grid, fresh = defuse_profile_on_grid(profile, equilibria, mask_eq_usable, times)
    return values_on_grid, gradient_on_grid, fresh, times


def test_defuse_profiles_land_on_store_grid():
    """Constant q maps rho_pol to rho_tor_norm unchanged, so T_e = 1000 (1 - rho^2) with gradient -2000 rho.

    The gradient is taken on the DEFUSE points, so it is finite at the LCFS,
    and the grid points past the LCFS are NaN.
    """
    values_on_grid, gradient_on_grid, _, times = _profile_on_grid()
    i_time = int(np.argmin(np.abs(times - 0.105)))
    mask_inside = RHO_TOR_NORM_GRID <= 1.0

    te_expected = 1000.0 * (1.0 - RHO_TOR_NORM_GRID**2)
    np.testing.assert_allclose(values_on_grid[i_time, mask_inside], te_expected[mask_inside], atol=1.0)
    assert np.isnan(values_on_grid[i_time, ~mask_inside]).all()

    gradient_expected = -2000.0 * RHO_TOR_NORM_GRID
    mask_mid = (RHO_TOR_NORM_GRID > 0.2) & mask_inside
    np.testing.assert_allclose(gradient_on_grid[i_time, mask_mid], gradient_expected[mask_mid], rtol=1e-2)
    idx_lcfs = int(np.flatnonzero(RHO_TOR_NORM_GRID == 1.0)[0])
    assert gradient_on_grid[i_time, idx_lcfs] == pytest.approx(-2000.0, rel=1e-2)
    assert np.isnan(gradient_on_grid[i_time, ~mask_inside]).all()


def test_defuse_hold_and_unmapped_slice():
    """Slices hold until the next one, and the hold ends max_hold_defuse_steps median steps after the last slice.

    A slice with no usable reconstruction within match_max_ms, or with a NaN fit point, is dropped before the hold,
    so the slice before it holds over it and it is not fresh.
    """
    values_on_grid, _, fresh, times = _profile_on_grid()
    te_axis = values_on_grid[:, 0]
    times_ms = np.round(times * 1e3)
    defuse_times_ms = np.round(DEFUSE_TIMES * 1e3)
    mask_slice_usable = ~np.isin(DEFUSE_TIMES, [DEFUSE_UNMAPPED_TIME, DEFUSE_INCOMPLETE_TIME])
    usable_times_ms = defuse_times_ms[mask_slice_usable]

    assert np.isnan(te_axis[times_ms < defuse_times_ms[0]]).all()
    mask_slices_held = (times_ms >= defuse_times_ms[0]) & (times_ms <= defuse_times_ms[-1])
    assert np.isfinite(te_axis[mask_slices_held]).all()
    np.testing.assert_array_equal(times_ms[fresh], usable_times_ms)

    usable_step_median_ms = np.median(np.diff(usable_times_ms))
    hold_end_ms = usable_times_ms[-1] + config["profile_grid"]["max_hold_defuse_steps"] * usable_step_median_ms
    # One slice of margin either side of the limit, where float round-off decides
    assert np.isfinite(te_axis[(times_ms > defuse_times_ms[-1]) & (times_ms < hold_end_ms - 1)]).all()
    assert np.isnan(te_axis[times_ms > hold_end_ms + 1]).all()


FIR_SAMPLE_STEP = 40e-6
FIR_DENSITY = 3e19


def _fir_trace(steps: list[tuple[float, float]], seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1 s of NEavg at 25 kHz with 2e17 noise: (times, true density, trace with a fringe jump of each size at each time)."""
    sample_time = np.arange(0.0, 1.0, FIR_SAMPLE_STEP)
    rng = np.random.default_rng(seed)
    density_true = FIR_DENSITY + 2e17 * rng.standard_normal(sample_time.size)
    density_trace = density_true.copy()
    for jump_time, jump_size in steps:
        density_trace[sample_time >= jump_time] += jump_size
    return sample_time, density_true, density_trace


def test_fringe_jumps_removed_and_real_changes_kept():
    """A fringe slip is removed, a dropout that recovers and a spike that decays leave the level where it was,
    and a real 1.5e19 drop over 3 ms is kept."""
    sample_time, density_true, density_trace = _fir_trace([(0.2, -2e19)])
    mask_dropout = (sample_time >= 0.400) & (sample_time < 0.404)
    density_trace[mask_dropout] = -1e19
    time_since_spike = sample_time - 0.6
    mask_spike = time_since_spike >= 0
    spike_decay = np.exp(-time_since_spike[mask_spike] / 1e-3)
    density_trace[mask_spike] += 1.5e19 * spike_decay
    drop_fraction = np.clip((sample_time - 0.8) / 3e-3, 0.0, 1.0)
    real_drop = 1.5e19 * drop_fraction
    density_true = density_true - real_drop
    density_trace = density_trace - real_drop

    density_corrected, cut_time = _remove_fringe_jumps(sample_time, density_trace)

    assert cut_time is None
    mask_off_spike = ~((sample_time >= 0.5995) & (sample_time < 0.605))
    np.testing.assert_allclose(density_corrected[mask_off_spike], density_true[mask_off_spike], atol=1.5e18)


def test_fringe_burst_cuts_the_rest_of_the_record():
    """Three slips within FRINGE_BURST_WINDOW_S mean the interferometer lost count, NaN from the first on."""
    sample_time, density_true, density_trace = _fir_trace([(0.2, -2e19), (0.22, -2e19), (0.24, 2e19)])

    density_corrected, cut_time = _remove_fringe_jumps(sample_time, density_trace)

    assert cut_time == pytest.approx(0.2, abs=1e-3)
    mask_before = sample_time < 0.199
    np.testing.assert_allclose(density_corrected[mask_before], density_true[mask_before])
    assert np.isnan(density_corrected[sample_time >= 0.2]).all()


def _write_matlab(group: h5py.Group, name: str, data: np.ndarray, matlab_class: str = "single", empty: bool = False):
    """One dataset with the attributes MATLAB v7.3 writes."""
    group[name] = data
    group[name].attrs["MATLAB_class"] = np.bytes_(matlab_class)
    if empty:
        group[name].attrs["MATLAB_empty"] = np.uint8(1)


def test_read_defuse_handles_the_matlab_layout(tmp_path):
    """MATLAB v7.3 rows come back flat, per-gyrotron ECRH rows are summed, repeated times are dropped,
    and both DEFUSE placeholders (a flagged empty array, a uint64 [0 0]) read as absent."""
    path = tmp_path / "TCVno70000.h5"
    time = np.array([[0.0, 0.1, 0.1, 0.2, 0.3]])
    with h5py.File(path, "w") as defuse_file:
        signal_root = defuse_file.create_group("SIG")
        ip_group = signal_root.create_group("I_P")
        _write_matlab(ip_group, "signal", -np.array([[1e5, 2e5, 2e5, 3e5, 4e5]]))
        _write_matlab(ip_group, "time", time)
        ecrh_group = signal_root.create_group("ECRH")
        ecrh_rows = np.array([[0.5, 0.5, 0.5, 0.5, 0.5], [0.0, 1.0, 1.0, 1.0, 1.0], [np.nan, 0.25, 0.25, 0.25, np.nan]])
        _write_matlab(ecrh_group, "signal", ecrh_rows)
        _write_matlab(ecrh_group, "time", time)
        nbi_group = signal_root.create_group("NBI")
        for key in ["signal", "time"]:
            _write_matlab(nbi_group, key, np.array([1, 0], dtype=np.uint64), empty=True)
        te_fit_group = signal_root.create_group("Te_rho").create_group("signal")
        _write_matlab(te_fit_group, "t", np.array([[0.05, 0.15]]))
        _write_matlab(te_fit_group, "x", np.array([[0.0, 0.5, 1.0]]))
        _write_matlab(te_fit_group, "z", np.array([[1000.0, 600.0, 50.0], [1100.0, 650.0, 60.0]]))
        ne_fit_group = signal_root.create_group("Ne_rho").create_group("signal")
        for key in ["t", "x", "z"]:
            _write_matlab(ne_fit_group, key, np.array([[0, 0]], dtype=np.uint64), matlab_class="uint64")

    signals, profiles = read_defuse(path, ("I_P", "ECRH", "NBI", "NBI2"), ("Te_rho", "Ne_rho"))

    assert set(signals) == {"I_P", "ECRH"}
    np.testing.assert_array_equal(signals["I_P"].time, [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_array_equal(signals["I_P"].values, [-1e5, -2e5, -3e5, -4e5])
    np.testing.assert_allclose(signals["ECRH"].values, [0.5, 1.75, 1.75, 1.5])
    assert set(profiles) == {"Te_rho"}
    np.testing.assert_array_equal(profiles["Te_rho"].rho_pol, [0.0, 0.5, 1.0])
    assert profiles["Te_rho"].values.shape == (2, 3)


N_TIME = 700
NUM_GRID = RHO_TOR_NORM_GRID.size
# DEFUSE values of the synthetic shot, constant in time, in DEFUSE units (SI, heating in MW)
RAW_SCALAR_VALUES = {
    "I_P": -3e5,
    "BZERO": -1.4,
    "Wtot": 3e4,
    "BETAN": 1.5,
    "NEavg": 3e19,
    "a_minor": 0.24,
    "R_geom": 0.88,
    "KAPPA": 1.5,
    "DELTA_TOP": 0.3,
    "DELTA_BOTTOM": 0.2,
    # Opposite sign to I_P, as DEFUSE stores it, so Ip * Vloop is 0.3 MW of ohmic power
    "Vloop": 1.0,
    "LI": 1.0,
    "PradTot": 1e5,
    "NBI": 0.5,
    "ECRH": 1.2,
}


def _raw_dataset(shot: int = 70000) -> xr.Dataset:
    """What _get_shot_dataset hands to standardize_signal_names: DEFUSE names on the uniform timebase,
    profiles on the store grid and NaN past the LCFS. NBI2 is absent, as in a single-beam shot."""
    time = np.arange(N_TIME) * 1e-3
    mask_inside = RHO_TOR_NORM_GRID <= 1.0
    te = np.where(mask_inside, 1000.0 * (1.0 - RHO_TOR_NORM_GRID**2) + 50.0, np.nan)
    ne = np.where(mask_inside, 3.5e19 * (1.0 - 0.5 * RHO_TOR_NORM_GRID**2), np.nan)
    # One profile per 17 ms DEFUSE slice, held in between
    slice_scale = 1.0 + 0.001 * (np.arange(N_TIME) // 17)
    profiles = {
        "Te_rho": te[np.newaxis, :] * slice_scale[:, np.newaxis],
        "Ne_rho": ne[np.newaxis, :] * slice_scale[:, np.newaxis],
        "Te_rho_grad": np.tile(np.gradient(te, RHO_TOR_NORM_GRID), (N_TIME, 1)),
        "Ne_rho_grad": np.tile(np.gradient(ne, RHO_TOR_NORM_GRID), (N_TIME, 1)),
    }
    data_vars = {name: (("time",), np.full(N_TIME, value)) for name, value in RAW_SCALAR_VALUES.items()}
    data_vars |= {name: (("time", RADIAL_DIM), values) for name, values in profiles.items()}
    fresh_profile = (np.arange(N_TIME) % 17 == 0).astype(np.float32)
    data_vars["fresh_profile"] = (("time",), fresh_profile)
    data_vars["fresh_equilibrium"] = (("time",), np.ones(N_TIME, dtype=np.float32))
    ds = xr.Dataset(data_vars, coords={"time": time, RADIAL_DIM: RHO_TOR_NORM_GRID.astype(np.float32)})
    return ds.expand_dims(shot=[shot])


@pytest.fixture
def workflow(tmp_path) -> TCVDataWorkflow:
    shotlist = tmp_path / "shotlist.txt"
    shotlist.write_text("70000\n70001\n")
    workflow = TCVDataWorkflow(ds_name="tcv_test", shotlist_file=shotlist, data_assembly_dir=tmp_path)
    workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
    return workflow


def test_standardize_builds_store_signals_in_si(workflow):
    """Magnitudes, MW heating to W with absent beams as zero, zero errors only where the profile exists,
    and None when a DEFUSE signal is missing or a profile is all NaN."""
    ds = workflow.standardize_signal_names(_raw_dataset())

    assert ds is not None
    assert set(ds.data_vars) == set(STORE_SIGNALS)
    assert ds.sizes["time_idx"] == N_TIME
    assert ds["ip"].isel(time_idx=0).item() == pytest.approx(3e5)
    assert ds["b0"].isel(time_idx=0).item() == pytest.approx(1.4)
    # Constant Ip, li and R leave Ip V_loop, from the second sample on (backward difference)
    assert ds["power_ohm"].isel(time_idx=10).item() == pytest.approx(3e5)
    assert ds["power_nbi"].isel(time_idx=0).item() == pytest.approx(0.5e6)
    assert ds["power_ec"].isel(time_idx=0).item() == pytest.approx(1.2e6)
    assert float(np.abs(ds["power_ic"]).max()) == 0.0
    mask_profile = ds["t_e"].notnull()
    for error in ["t_e_error", "t_e_gradient_error", "n_e_error", "n_e_gradient_error"]:
        assert (ds[error].where(mask_profile) == 0).sum() == mask_profile.sum()
        assert ds[error].where(~mask_profile).isnull().all()

    assert workflow.standardize_signal_names(_raw_dataset().drop_vars("Vloop")) is None
    ds_no_te = _raw_dataset()
    ds_no_te["Te_rho"] = ds_no_te["Te_rho"] * np.nan
    assert workflow.standardize_signal_names(ds_no_te) is None


@pytest.fixture
def built_store(workflow):
    """The store built from two synthetic shots, through the real processing chain."""
    for shot in [70000, 70001]:
        ds_standardized = workflow.standardize_signal_names(_raw_dataset(shot))
        ds_standardized.to_netcdf(workflow.raw_data_dir / f"{shot}.nc")
    workflow.run_processed_data_workflow()
    return xr.open_zarr(workflow.store_path(PREDICTION_STORE_NAME))


def test_store_carries_imas_attributes_and_grid(built_store):
    """The store holds the shared schema, companions included, every variable and coordinate has a description
    and units, the ref (IMAS path) of the shared schema or TCV_SIGNAL_ATTRS wherever IMAS has a leaf, and the store grid."""
    assert set(built_store.data_vars) == {*STORE_SIGNALS, TIME_COORD}
    assert list(built_store["shot"].values) == [70000, 70001]
    assert built_store.attrs["profile_source"] == "DEFUSE"
    names = [name for name in built_store.variables if name not in ("shot", "time_idx")]
    for name in names:
        attrs = built_store[name].attrs
        assert {"description", "units"} <= set(attrs), name
        # The shared schema's ref wins over the device's own
        expected_attrs = STORE_SIGNAL_ATTRS.get(name, TCV_SIGNAL_ATTRS[name])
        if "ref" in expected_attrs:
            assert attrs["ref"] == expected_attrs["ref"], name
    np.testing.assert_array_equal(built_store[RADIAL_DIM].values, RHO_TOR_NORM_GRID.astype(np.float32))


def test_every_store_signal_and_source_has_attrs():
    """A store signal or source added without attributes would be written undocumented."""
    assert set(STORE_SIGNALS) | {TIME_COORD, RADIAL_DIM} <= set(TCV_SIGNAL_ATTRS)
    assert set(PREDICTION_SOURCES) <= set(TCV_SIGNAL_ATTRS)
    assert set(RAW_SCALAR_VALUES) <= set(DEFUSE_SIGNALS)


@pytest.mark.slow
@pytest.mark.skipif(not meqdb_path(LIVE_SHOT).exists(), reason="needs the DEFUSE exports and the TCV MEQ databases")
def test_live_single_shot(tmp_path):
    """One real shot: physical ranges, profiles ending at the LCFS, and rho_tor_norm inside rho_pol in the core."""
    shotlist = tmp_path / "shotlist.txt"
    shotlist.write_text(f"{LIVE_SHOT}\n")
    workflow = TCVDataWorkflow(ds_name="tcv_live", shotlist_file=shotlist, data_assembly_dir=tmp_path, max_num_shots=1)
    workflow.raw_data_dir.mkdir(parents=True, exist_ok=True)
    workflow.make_raw_data_files()

    raw_path = workflow.raw_data_dir / f"{LIVE_SHOT}.nc"
    assert raw_path.exists()
    with xr.open_dataset(raw_path) as raw_file:
        raw = raw_file.load()
    assert 5e4 < float(raw["ip"].max()) < 1e6
    assert 1.0 < float(raw["b0"].median()) < 1.6
    assert 0.15 < float(raw["minor_radius"].median()) < 0.3
    assert 1e18 < float(raw["n_e_line_average"].median()) < 2e20
    te_axis = raw["t_e"].isel({RADIAL_DIM: 0})
    assert te_axis.notnull().sum() > 100
    assert 100 < float(te_axis.median()) < 2e4
    mask_outside = raw[RADIAL_DIM] > 1.0
    assert raw["t_e"].where(mask_outside).isnull().all()
    assert raw["t_e"].sel({RADIAL_DIM: 1.0}).notnull().sum() == te_axis.notnull().sum()

    equilibria = read_liuqe(meqdb_path(LIVE_SHOT))
    mask_usable = liuqe_usable(equilibria)
    assert mask_usable.mean() > 0.9
    psi_n_surfaces, q_surfaces = liuqe_q_profiles(equilibria)
    mask_core = (equilibria.rho_pol > 0.05) & (equilibria.rho_pol < 0.8)
    for q_slice in q_surfaces[mask_usable][::50]:
        phi_n = phi_n_map(psi_n_surfaces, q_slice, "secant").phi_n(psi_n_surfaces)
        rho_tor_norm = np.sqrt(phi_n)
        assert (rho_tor_norm[mask_core] < equilibria.rho_pol[mask_core]).all()
